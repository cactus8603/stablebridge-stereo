"""Rebuild the final training targets and reduced evidence from licensed Spring data."""
import argparse,json,time,sys
from pathlib import Path
import cv2,numpy as np,torch
from stablebridge.training import load_data
from stablebridge.paths import writable_output
from stablebridge.data import _full_views_and_center_gt,_full_left_gt,CellJob,materialize_condition,compute_metrics
from stablebridge.treatments import core3_candidates_uint8,normalize_uint8_rgb
from stablebridge.backbone import CroCoStereoAdapter,DEFAULT_CHECKPOINT,sha256_file
from stablebridge.features import FeatureExtractor
ROOT=Path(__file__).resolve().parent


def reference_pairs(left,right):
    # Two additional reference outcomes define the paper's five-choice upper bound.
    joint=np.concatenate((left.reshape(-1,3),right.reshape(-1,3)),axis=0).astype(np.float32)
    low,high=np.quantile(joint,(.01,.99),axis=0);scale=255/np.maximum(high-low,12)
    normalized=tuple(np.clip(np.rint((v.astype(np.float32)-low)*scale),0,255).astype(np.uint8) for v in (left,right))
    saturation=np.concatenate((cv2.cvtColor(left,cv2.COLOR_RGB2HSV)[...,1].reshape(-1),cv2.cvtColor(right,cv2.COLOR_RGB2HSV)[...,1].reshape(-1))).astype(np.float32)
    scale=float(np.clip(160/max(float(np.quantile(saturation,.95)),1),.35,1))
    compressed=[]
    for value in (left,right):
        hsv=cv2.cvtColor(value,cv2.COLOR_RGB2HSV)
        hsv[...,1]=np.clip(np.rint(hsv[...,1].astype(np.float32)*scale),0,255).astype(np.uint8)
        compressed.append(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB))
    return normalized,tuple(compressed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'runs/training_features.npz')
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=0,help='Partial GPU smoke only; output will not train')
    args=p.parse_args()
    args.output=writable_output(args.output)
    if args.limit<0:raise ValueError("--limit must be nonnegative")
    if not torch.cuda.is_available():raise SystemExit("CUDA unavailable")
    if args.device!='cuda' and not args.device.startswith('cuda:'):raise SystemExit('The fixed evidence recipe requires CUDA BF16')
    torch.cuda.reset_peak_memory_stats(args.device)
    started=time.monotonic()
    template,_,_=load_data(ROOT/'data/training_features.npz')
    records={r['id']:r for r in map(json.loads,(ROOT/'configs/samples.jsonl').read_text().splitlines())}
    scratch=args.output.parent/(args.output.stem+'-cache');scratch.mkdir(parents=True,exist_ok=True)
    contract={'checkpoint_sha256':sha256_file(DEFAULT_CHECKPOINT),
        'samples_sha256':sha256_file(ROOT/'configs/samples.jsonl'),
        'template_sha256':sha256_file(ROOT/'data/training_features.npz'),
        'source_sha256':{p.name:sha256_file(p) for p in (ROOT/'stablebridge').glob('*.py')},
        'prepare_sha256':sha256_file(Path(__file__)),
        'device':args.device,'gpu':torch.cuda.get_device_name(args.device),
        'torch':torch.__version__,'cuda_build':torch.version.cuda,
        'dataset_root':str(args.dataset_root.resolve())}
    c=scratch/'contract.json'
    if c.exists() and json.loads(c.read_text())!=contract:raise SystemExit('Rebuild input contract differs')
    c.write_text(json.dumps(contract,indent=2)+'\n')
    count=len(template['cell_id']);indices=list(range(count))[:args.limit or count]
    def array(name,shape,dtype=np.float32,fill=np.nan):
        path=scratch/(name+'.npy')
        if not path.exists():
            value=np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=dtype);value[:]=fill;value.flush();del value
        return np.load(path,mmap_mode='r+')
    outcomes=array('outcomes',(count,5,3),np.float64);target_done=array('targets-complete',(count,),np.bool_,False)
    feature_done=array('features-complete',(count,),np.bool_,False)
    features={d:array(f'features{d}',(count,3,227)) for d in (1,8)}
    adapter=CroCoStereoAdapter(device=args.device,amp_dtype='none')
    for i in indices:
        if target_done[i]:continue
        record=records[str(template['sample_id'][i])]
        left,right,_,y,x=_full_views_and_center_gt(record,args.dataset_root,(352,704))
        gt=_full_left_gt(record,args.dataset_root,left.shape[:2])
        job=CellJob(record,str(template['condition'][i]),int(template['severity'][i]))
        left,right,target,valid,_=materialize_condition(left,right,gt,job,int(template['sample_seed'][i]),'official')
        candidates=(*core3_candidates_uint8(left,right),*reference_pairs(left,right))
        for k,(a,b) in enumerate(candidates):
            a=normalize_uint8_rgb(a)[:,:,y:y+352,x:x+704].to(adapter.device)
            b=normalize_uint8_rgb(b)[:,:,y:y+352,x:x+704].to(adapter.device)
            with torch.inference_mode():prediction=adapter.model(a,b)[0,0].float().cpu().numpy()
            metric=compute_metrics(prediction,target[y:y+352,x:x+704],valid[y:y+352,x:x+704])
            outcomes[i,k]=[metric['abs_px'],metric['over_1px_percent'],metric['d1_percent']]
        outcomes.flush();target_done[i]=True;target_done.flush()
        if i%100==0:print(json.dumps({'targets_completed':int(target_done.sum()),'rows':count}),flush=True)
    del adapter
    torch.cuda.empty_cache()
    extractor=FeatureExtractor(args.device,DEFAULT_CHECKPOINT)
    for start in range(0,len(indices),4):
        batch=indices[start:start+4]
        # Recompute the original group to preserve padding and batch arithmetic on resume.
        if all(feature_done[i] for i in batch):continue
        values=extractor.rows(template,batch,[records[str(template['sample_id'][i])] for i in batch],args.dataset_root)
        for i,row in zip(batch,values,strict=True):
            for depth in (1,8):features[depth][i]=row[depth]
        for v in features.values():v.flush()
        feature_done[batch]=True;feature_done.flush()
    payload={k:v.copy() for k,v in template.items()}
    full_abs=np.asarray(outcomes[:,:,0]);full_gain=full_abs[:,:1]-full_abs
    payload['action_abs_px']=full_abs.astype(template['action_abs_px'].dtype)
    payload['action_over_1px_percent']=np.asarray(outcomes[:,:,1]).astype(template['action_over_1px_percent'].dtype)
    payload['action_d1_percent']=np.asarray(outcomes[:,:,2]).astype(template['action_d1_percent'].dtype)
    payload['gain_abs_px']=full_gain.astype(template['gain_abs_px'].dtype)
    payload['oracle_gain_abs_px']=np.maximum(full_gain.max(axis=1),0).astype(template['oracle_gain_abs_px'].dtype)
    payload.update({f'features{d}':np.asarray(v) for d,v in features.items()})
    temporary=args.output.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**payload)
    temporary.replace(args.output)
    torch.cuda.synchronize(args.device)
    (scratch/'resources.json').write_text(json.dumps({
        'scope':'This invocation, including resumed work only',
        'wall_seconds':time.monotonic()-started,'gpu_count':1,
        'gpu_type':torch.cuda.get_device_name(args.device),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device),
        'python':sys.version,'torch':torch.__version__,'cuda_build':torch.version.cuda},indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'complete':bool(target_done.all() and feature_done.all()),'rows':count}))
if __name__=='__main__':main()
