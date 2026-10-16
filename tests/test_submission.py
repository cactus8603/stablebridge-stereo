from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from stablebridge.submission.contract import write_dsp5,inspect_dsp5,EXPECTED_FILES_PER_VARIANT
from stablebridge.submission.tiling import iter_pair_jobs

def test_official_prediction_roundtrip(tmp_path):
    path=tmp_path/'test.dsp5'
    write_dsp5(path,np.ones((1080,1920),np.float32))
    result=inspect_dsp5(path,check_values=True)
    assert tuple(result['shape'])[:2]==(1080,1920) and result['finite']

def test_pair_jobs_share_expected_reference_paths(tmp_path):
    jobs=list(iter_pair_jobs(tmp_path/'pred',tmp_path/'clean',tmp_path/'robust',['clean']))
    assert len(jobs)==1000
    assert jobs[0].left_prediction.suffix=='.dsp5'
    assert 'disp1_right' in str(jobs[0].right_prediction)
    assert EXPECTED_FILES_PER_VARIANT==2000


def test_bundled_official_packers_match_verified_hashes():
    from stablebridge.submission.packaging import validate_tools
    errors, report = validate_tools(ROOT / 'tools', ('clean', 'robust'))
    assert not errors, errors
    assert len(report) == 2 and all(row['verified'] for row in report.values())
