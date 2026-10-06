from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tracks.stereo.baselines.croco.adapter import load_rgb_tensor
from tracks.stereo.croco_stablebridge.candidates import normalize_uint8_rgb
from tracks.stereo.run_croco_submission import PairJob
from tracks.stereo.run_stablebridge_submission import (
    _select_jobs,
    parse_args,
    read_rgb,
    route_record_path,
)


def _job(root: Path, scene: str, frame: int) -> PairJob:
    return PairJob(
        variant="clean",
        scene=scene,
        frame=frame,
        left_image=root / "left.png",
        right_image=root / "right.png",
        left_prediction=root / "left.dsp5",
        right_prediction=root / "right.dsp5",
    )


def test_p0_uint8_loading_and_normalization_match_released_runner(tmp_path: Path) -> None:
    rng = np.random.default_rng(29)
    value = rng.integers(0, 256, size=(27, 41, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(value).save(path)
    loaded = read_rgb(path)
    assert np.array_equal(loaded, value)
    assert torch.equal(normalize_uint8_rgb(loaded), load_rgb_tensor(path))


def test_one_per_scene_and_route_record_contract(tmp_path: Path) -> None:
    jobs = [_job(tmp_path, "0003", 1), _job(tmp_path, "0003", 2), _job(tmp_path, "0019", 1)]
    selected = _select_jobs(jobs, parse_args(["--one-per-scene"]))
    assert [(job.scene, job.frame) for job in selected] == [("0003", 1), ("0019", 1)]
    assert route_record_path(tmp_path, jobs[0]) == (
        tmp_path / "routes/clean/test/0003/frame_0001.json"
    )


def test_clean_treatment_is_disallowed_by_default() -> None:
    args = parse_args([])
    assert args.variants == ["clean"]
    assert args.allow_clean_treatment is False
    assert args.batch_pairs == 4

