"""Regression checks for row alignment, resume provenance and portable paths."""
import json
import numpy as np
import pytest
from stablebridge.training import load_data, load_evaluation
from stablebridge.paths import ROOT, writable_output
from stablebridge.router import StableBridgeRouter
from stablebridge.submission.provenance import bind_output


@pytest.mark.parametrize("corruption", ["row", "data", "nonfinite", "action"])
def test_heldout_scores_reject_misaligned_or_invalid_data(tmp_path, corruption):
    data = ROOT / "data/training_features.npz"
    teacher, _, folds = load_data(data)
    with np.load(ROOT / "data/heldout_predictions.npz", allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    if corruption == "row":
        # Swap identities within one fold: a fold-only check would miss this.
        pair = np.flatnonzero(folds == folds[0])[:2]
        payload["cell_id"][pair] = payload["cell_id"][pair[::-1]]
    elif corruption == "data":
        payload["training_data_sha256"] = np.asarray("0" * 64)
    elif corruption == "nonfinite":
        payload["output"][0, 0, 0] = np.nan
    else:
        payload["action"][0] = 3
    path = tmp_path / "scores.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        load_evaluation(path, teacher, folds, data)


def test_prediction_resume_refuses_mixed_model_settings(tmp_path):
    original = {"model": "final", "precision": "float32"}
    bind_output(tmp_path, original)
    bind_output(tmp_path, original)
    (tmp_path / "prediction.dsp5").write_bytes(b"existing")
    with pytest.raises(ValueError, match="contract changed"):
        bind_output(tmp_path, {**original, "model": "another"})
    assert json.loads((tmp_path / "prediction-contract.json").read_text()) == original
    assert (tmp_path / "prediction.dsp5").read_bytes() == b"existing"


def test_prediction_resume_refuses_unknown_existing_maps(tmp_path):
    (tmp_path / "prediction.dsp5").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="no contract"):
        bind_output(tmp_path, {"model": "final"})
    assert not (tmp_path / "prediction-contract.json").exists()


def test_model_artifact_resolution_ignores_current_directory(tmp_path, monkeypatch):
    (tmp_path / "fixed-evidence.pt").write_bytes(b"unrelated")
    monkeypatch.chdir(tmp_path)
    router = object.__new__(StableBridgeRouter)
    router.bundle_path = ROOT / "weights/model.json"
    assert router._resolve_artifact("fixed-evidence.pt") == ROOT / "weights/fixed-evidence.pt"


@pytest.mark.parametrize("path", [ROOT, ROOT / "weights/model.json", ROOT / "data"])
def test_generated_output_cannot_overwrite_release_inputs(path):
    with pytest.raises(ValueError, match="preserve the release"):
        writable_output(path)
