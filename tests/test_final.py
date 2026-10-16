from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stablebridge.training import load_data
from stablebridge.model import EvidenceProbe
from stablebridge.router import StableBridgeRouter,aggregate_tile_actions
from stablebridge.optimization import target_arrays,train_fold_seed
from stablebridge.metrics import calibrate_threshold,policy_metrics
from stablebridge.treatments import core3_candidates_uint8
ROOT=Path(__file__).resolve().parents[1]

def test_training_data_are_scene_disjoint_and_complete():
    teacher,features,folds=load_data(ROOT/'data/training_features.npz')
    assert len(folds)==8784 and len(set(teacher['scene']))==36
    assert features[1].shape==features[8].shape==(8784,3,227)

def test_final_weights_strict_load_and_predict_heads():
    router=StableBridgeRouter(torch.nn.Identity(),bundle_path=ROOT/'weights/model.json',device='cpu')
    assert router.bundle['policy']['threshold']==0.6998358964920044
    _,features,_=load_data(ROOT/'data/training_features.npz')
    for depth in [1,8]:
        with torch.inference_mode():value=router._ensemble(depth,torch.from_numpy(features[depth][:8]))
        assert value.shape==(8,2,6) and torch.isfinite(value).all()

def test_threshold_matches_frozen_definition():
    teacher,_,folds=load_data(ROOT/'data/training_features.npz')
    with np.load(ROOT/'data/heldout_predictions.npz',allow_pickle=False) as a:
        assert np.array_equal(folds,a['folds'])
        calibration=calibrate_threshold(teacher,a['output'],401)
    assert calibration['best']['score_threshold']==0.6998358964920044

def test_one_epoch_real_feature_training_and_serialization(tmp_path):
    teacher,features,folds=load_data(ROOT/'data/training_features.npz')
    fit=np.flatnonzero(folds!=0)[:32];held=np.flatnonzero(folds==0)[:8]
    args=SimpleNamespace(device='cpu',width=128,learning_rate=.001,weight_decay=.001,epochs=1,train_batch_size=16)
    value,count=train_fold_seed(feature=features[1],arrays=target_arrays(teacher),train=fit,test=held,
        seed=100020,args=args,checkpoint_path=tmp_path/'head.pt')
    assert count==134942 and np.isfinite(value).all()
    saved=torch.load(tmp_path/'head.pt',map_location='cpu',weights_only=True)
    model=EvidenceProbe(227,128).eval();model.load_state_dict(saved['model'],strict=True)
    with torch.inference_mode():replayed=model(torch.from_numpy(features[1][held])).numpy()
    np.testing.assert_allclose(value,replayed,atol=1e-6,rtol=1e-6)

def test_saved_final_evaluation_matches_paper():
    teacher,_,_=load_data(ROOT/'data/training_features.npz')
    with np.load(ROOT/'data/heldout_predictions.npz',allow_pickle=False) as a:metric=policy_metrics(teacher,a['action'])
    assert abs(metric['corrupt_improvement_percent']-5.078738417)<1e-7
    assert metric['clean_improvement_percent']==0

def test_identity_treatment_is_bit_exact():
    rgb=np.random.default_rng(17).integers(0,256,(40,60,3),dtype=np.uint8)
    values=core3_candidates_uint8(rgb,rgb)
    assert np.array_equal(values[0][0],rgb)
    assert all(v.shape==rgb.shape and v.dtype==np.uint8 for pair in values for v in pair)

def test_pair_vote_requires_majority():
    assert aggregate_tile_actions([0,1,1,2,2])==0
    assert aggregate_tile_actions([1,1,1,0,2])==1
    assert aggregate_tile_actions([2,2,2,1,0])==2


def test_pinned_vendor_marker_works_inside_a_parent_git_repository(tmp_path):
    from unittest.mock import patch
    from stablebridge.backbone import repository_commit
    (tmp_path/'REVISION').write_text('5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb\n')
    with patch('stablebridge.backbone.subprocess.check_output') as git:
        assert repository_commit(tmp_path)=='5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb'
        git.assert_not_called()
