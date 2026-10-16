"""Final three-treatment selector fitting with scene-disjoint calibration."""
from __future__ import annotations
import argparse,json,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .optimization import target_arrays,train_fold_seed
from .metrics import calibrate_threshold,apply_threshold,policy_metrics,scene_bootstrap
from .backbone import sha256_file
from .paths import writable_output
ROOT=Path(__file__).resolve().parents[1]


def load_data(path):
    with np.load(path,allow_pickle=False) as archive:
        features={d:archive[f"features{d}"].astype(np.float32) for d in (1,8)}
        teacher={k:archive[k] for k in archive.files if not k.startswith("features")}
    folds=teacher["outer_fold"].astype(np.int64)
    if len(folds)!=8784 or sorted(set(folds.tolist()))!=list(range(6)):
        raise ValueError("Expected 8,784 rows in six scene folds")
    if len(set(teacher["scene"]))!=36 or len(set(teacher["cell_id"]))!=len(folds):
        raise ValueError("Scene or cell identity contract failed")
    for scene in np.unique(teacher["scene"]):
        if len(np.unique(folds[teacher["scene"]==scene]))!=1:
            raise ValueError("Scene leakage across folds")
    for value in features.values():
        if value.shape!=(8784,3,227) or not np.isfinite(value).all():
            raise ValueError("Incomplete reduced evidence")
    for key in ["gain_abs_px","action_abs_px","oracle_gain_abs_px"]:
        if not np.isfinite(teacher[key]).all():raise ValueError(f"Nonfinite {key}")
    return teacher,features,folds


def load_evaluation(path, teacher, folds, data_path):
    """Bind held-out scores to the exact fitting cache and row identities."""
    with np.load(path, allow_pickle=False) as archive:
        for key in ("folds", "cell_id", "training_data_sha256", "output", "action"):
            if key not in archive:
                raise ValueError(f"Evaluation is missing {key}")
        if not np.array_equal(archive["folds"], folds):
            raise ValueError("Evaluation fold order mismatch")
        if not np.array_equal(archive["cell_id"], teacher["cell_id"]):
            raise ValueError("Evaluation row identity mismatch")
        if str(archive["training_data_sha256"].item()) != sha256_file(data_path):
            raise ValueError("Evaluation belongs to different training data; rerun cross-validation")
        output, action = archive["output"], archive["action"]
        if output.shape != (len(folds), 2, 6) or not np.isfinite(output).all():
            raise ValueError("Invalid held-out outputs")
        if action.shape != (len(folds),) or action.dtype.kind not in "iu" or np.any((action < 0) | (action > 2)):
            raise ValueError("Invalid held-out actions")
        return output, action.astype(np.int64)


def fuse(early,deep):
    value=np.zeros_like(early)
    value[:,:,:3]=early[:,:,:3]
    value[:,:,3:5]=deep[:,:,3:5]
    return value


def save_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2)+"\n");tmp.replace(path)


def ensemble(features,arrays,train,test,outer,inner,args,root):
    values={}
    for depth in (1,8):
        predictions=[]
        for seed in (17,43,89):
            actual_seed=seed+1000003*outer+10007*(0 if inner is None else inner+1)+100003*depth
            directory=root/f"fold-{outer}"/("fit" if inner is None else f"calibration-{inner}")
            directory=directory/f"block-{depth}-seed-{seed}"
            directory.mkdir(parents=True,exist_ok=True)
            prediction=directory/"prediction.npz"
            if prediction.exists():
                with np.load(prediction,allow_pickle=False) as saved:
                    if not np.array_equal(saved["indices"],test):raise ValueError("Resume row mismatch")
                    value=saved["output"]
            else:
                value,_=train_fold_seed(feature=features[depth],arrays=arrays,train=train,
                    test=test,seed=actual_seed,args=args,checkpoint_path=directory/"head.pt")
                temporary=directory/"prediction.tmp"
                with temporary.open("wb") as f:np.savez_compressed(f,indices=test,output=value)
                temporary.replace(prediction)
            predictions.append(value)
        values[depth]=np.mean(predictions,axis=0)
    return fuse(values[1],values[8])


def cross_validate(teacher,features,folds,args,root,data_path):
    arrays=target_arrays(teacher)
    output=np.full((len(folds),2,6),np.nan,np.float32)
    action=np.zeros(len(folds),np.int8)
    thresholds=[]
    for outer in range(6):
        train=np.flatnonzero(folds!=outer);test=np.flatnonzero(folds==outer)
        inner_output=np.full((len(train),2,6),np.nan,np.float32)
        position={row:i for i,row in enumerate(train.tolist())}
        for inner in sorted(set(folds[train].tolist())):
            fit=np.flatnonzero((folds!=outer)&(folds!=inner))
            held=np.flatnonzero(folds==inner)
            value=ensemble(features,arrays,fit,held,outer,int(inner),args,root)
            inner_output[np.asarray([position[i] for i in held])]=value
        if not np.isfinite(inner_output).all():raise ValueError("Incomplete inner predictions")
        local={k:v[train] for k,v in teacher.items()}
        calibration=calibrate_threshold(local,inner_output,401)
        threshold=float("inf") if calibration["best"] is None else float(calibration["best"]["score_threshold"])
        value=ensemble(features,arrays,train,test,outer,None,args,root)
        output[test]=value;action[test]=apply_threshold(value,threshold)
        thresholds.append(threshold)
        save_json(root/f"fold-{outer}/calibration.json",calibration)
        print(json.dumps({"completed_fold":outer,"threshold":threshold}),flush=True)
    np.savez_compressed(root/"heldout_predictions.npz",output=output,action=action,folds=folds,
        cell_id=teacher["cell_id"],training_data_sha256=np.asarray(sha256_file(data_path)))
    result={"rows":len(folds),"trained_heads":216,"fold_thresholds":thresholds,
        "metrics":policy_metrics(teacher,action),"scene_intervals":scene_bootstrap(teacher,action,5000)}
    save_json(root/"metrics.json",result)


def fit_final(teacher,features,folds,args,root,evaluation,fixed,data_path):
    output,_=load_evaluation(evaluation,teacher,folds,data_path)
    calibration=calibrate_threshold(teacher,output,401)
    if calibration["best"] is None:raise ValueError("No safe development operating point")
    arrays=target_arrays(teacher);rows=np.arange(len(folds),dtype=np.int64);heads=[]
    for depth in (1,8):
        for seed in (17,43,89):
            checkpoint=root/f"heads/depth-{depth}/seed-{seed}.pt"
            actual_seed=seed+100003*depth
            _,count=train_fold_seed(feature=features[depth],arrays=arrays,train=rows,test=rows,
                seed=actual_seed,args=args,checkpoint_path=checkpoint)
            heads.append({"depth":depth,"seed":seed,"actual_seed":actual_seed,
                "path":str(checkpoint.relative_to(root)),"sha256":sha256_file(checkpoint),
                "trainable_parameters":count})
    shutil.copy2(fixed,root/"fixed-evidence.pt")
    bundle={"schema_version":1,"method_id":"croco-stablebridge-final",
        "status":"frozen_development_bundle","training_rows":len(rows),"training_scenes":36,
        "seeds":[17,43,89],"depths":[1,8],"epochs":80,"width":128,"feature_dim":227,
        "evidence_seed":20260824,"evidence":{"path":"fixed-evidence.pt",
        "sha256":sha256_file(root/"fixed-evidence.pt")},"heads":heads,
        "policy":{"selection":"q10","score":"benefit_minus_harm",
        "threshold":float(calibration["best"]["score_threshold"]),
        "threshold_source":"scene-held-out development predictions","identity_fallback":True,
        "actions":["P0","P1","P2"]}}
    save_json(root/"model.json",bundle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["cross-validation","final"],default="final")
    p.add_argument("--data",type=Path,default=ROOT/"data/training_features.npz")
    p.add_argument("--evaluation",type=Path,default=ROOT/"data/heldout_predictions.npz")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--validate-only",action="store_true")
    args=p.parse_args()
    teacher,features,folds=load_data(args.data)
    if args.stage=="final":load_evaluation(args.evaluation,teacher,folds,args.data)
    if args.validate_only:
        print(json.dumps({"status":"PASS","rows":len(folds),"scenes":36,"features":[3,227]}));return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise SystemExit("CUDA unavailable")
    root=writable_output(args.output)
    root.mkdir(parents=True,exist_ok=True)
    fixed=ROOT/"weights/fixed-evidence.pt"
    contract={"stage":args.stage,"data_sha256":sha256_file(args.data),
        "evaluation_sha256":sha256_file(args.evaluation) if args.stage=="final" else None,
        "device":args.device,"torch":torch.__version__,"cuda_build":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "optimizer":{"width":128,"learning_rate":.001,"weight_decay":.001,"epochs":80,"batch_size":512},"fixed_evidence_sha256":sha256_file(fixed),
        "source_sha256":{f.name:sha256_file(f) for f in (ROOT/"stablebridge").glob("*.py")}}
    contract_path=root/"training-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise SystemExit("Training contract changed; use a new output directory")
    save_json(contract_path,contract)
    train_args=SimpleNamespace(device=args.device,width=128,learning_rate=0.001,
        weight_decay=0.001,epochs=80,train_batch_size=512)
    if args.device.startswith("cuda"):torch.cuda.reset_peak_memory_stats(args.device)
    started=time.monotonic()
    if args.stage=="cross-validation":cross_validate(teacher,features,folds,train_args,root,args.data)
    else:fit_final(teacher,features,folds,train_args,root,args.evaluation,fixed,args.data)
    gpu=args.device.startswith("cuda")
    if gpu:torch.cuda.synchronize(args.device)
    save_json(root/"resources.json",{"scope":"This invocation, including resumed work only",
        "wall_seconds":time.monotonic()-started,
        "gpu_count":int(gpu),"gpu_type":torch.cuda.get_device_name(args.device) if gpu else None,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(args.device) if gpu else None,
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(args.device) if gpu else None,
        "python":sys.version,"torch":torch.__version__,"cuda_build":torch.version.cuda})
    return 0
