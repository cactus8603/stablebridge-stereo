"""Evaluate the final scene-held-out decisions and paired scene uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np
from stablebridge.training import load_data,load_evaluation
from stablebridge.paths import writable_output
from stablebridge.metrics import policy_metrics,scene_bootstrap
ROOT=Path(__file__).resolve().parent

def evaluate(data,predictions):
    teacher,_,folds=load_data(data)
    _,action=load_evaluation(predictions,teacher,folds,data)
    row=np.arange(len(action));clean=teacher['condition'].astype(str)=='clean'
    result={'rows':len(row),'scenes':len(set(teacher['scene']))}
    for name,mask in [('clean',clean),('corrupt',~clean)]:
        base=teacher['action_abs_px'][mask,0].astype(np.float64)
        routed=teacher['action_abs_px'][row,action][mask].astype(np.float64)
        result[name]={'unchanged_abs_px':float(base.mean()),'routed_abs_px':float(routed.mean()),
            'relative_reduction_percent':float(100*(base.sum()-routed.sum())/base.sum())}
    result['metrics']=policy_metrics(teacher,action)
    result['scene_intervals']=scene_bootstrap(teacher,action,5000)
    scenes=sorted(set(teacher['scene'].astype(str)));base=[];gain=[]
    for scene in scenes:
        mask=(teacher['scene'].astype(str)==scene)&(~clean)
        base.append(teacher['action_abs_px'][mask,0].astype(np.float64).sum())
        gain.append(teacher['gain_abs_px'][row,action][mask].astype(np.float64).sum())
    base=np.asarray(base);gain=np.asarray(gain);rng=np.random.default_rng(20260906)
    estimates=[]
    for _ in range(5000):
        weight=np.bincount(rng.integers(0,len(scenes),size=len(scenes)),minlength=len(scenes))
        estimates.append(100*(weight@gain)/(weight@base))
    result['pooled_relative_reduction_95ci_percent']=np.quantile(estimates,[.025,.975]).tolist()
    result['uncertainty_scope']='Scene resampling with fitted models, decisions and thresholds fixed.'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/training_features.npz')
    p.add_argument('--predictions',type=Path,default=ROOT/'data/heldout_predictions.npz')
    p.add_argument('--output',type=Path,default=ROOT/'runs/evaluation.json')
    args=p.parse_args();args.output=writable_output(args.output)
    result=evaluate(args.data,args.predictions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
