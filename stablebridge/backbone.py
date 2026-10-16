#!/usr/bin/env python3
"""Load the frozen CroCo-Stereo backbone and validate disparity outputs."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stablebridge.paths import checkpoint_path


EXPECTED_REPOSITORY_COMMIT = "5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb"
EXPECTED_CHECKPOINT_SHA256 = (
    "123273a722a58134c6816efe5122c427815c92f82896be4d484e6d21b52da575"
)
DEFAULT_CHECKPOINT = checkpoint_path("croco", "crocostereo_finetune_spring.pth")
DEFAULT_CROCO_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "croco"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class PredictionStats:
    shape: tuple[int, int]
    dtype: str
    finite_fraction: float
    nonnegative_fraction: float
    minimum: float
    maximum: float


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit(croco_root: Path) -> str:
    revision_file = croco_root / "REVISION"
    if not (croco_root / ".git").exists() and revision_file.is_file():
        return revision_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(croco_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Source archives record the upstream commit in REVISION.
        if revision_file.is_file():
            return revision_file.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            f"cannot determine CroCo revision from Git or {revision_file}"
        )


def verify_repository(croco_root: Path) -> str:
    if not (croco_root / "stereoflow" / "engine.py").is_file():
        raise FileNotFoundError(f"official CroCo checkout not found: {croco_root}")
    commit = repository_commit(croco_root)
    if commit != EXPECTED_REPOSITORY_COMMIT:
        raise RuntimeError(
            "CroCo checkout is not at the validated commit: "
            f"expected {EXPECTED_REPOSITORY_COMMIT}, got {commit}"
        )
    return commit


def _ensure_upstream_imports(croco_root: Path) -> None:
    root = str(croco_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)



def right_reference_inputs(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a left-looking equivalent of the original right-reference pair.

    CroCo-Stereo's official inference path predicts disparity on its first
    image. Swapping the views alone reverses the epipolar search direction;
    swapping and horizontally flipping both images restores the official
    left-looking geometry. The resulting prediction must be flipped back to
    the original right-image grid after inference.
    """

    return torch.flip(right, dims=(-1,)), torch.flip(left, dims=(-1,))


def canonicalize_reference_prediction(
    prediction: torch.Tensor, *, reference: str
) -> torch.Tensor:
    """Map an official first-image prediction to the requested image grid."""

    if reference not in {"left", "right"}:
        raise ValueError("reference must be 'left' or 'right'")
    return torch.flip(prediction, dims=(-1,)) if reference == "right" else prediction


def validate_prediction(prediction: np.ndarray) -> PredictionStats:
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be HxW, got shape {prediction.shape}")
    if prediction.dtype != np.float32:
        raise TypeError(f"prediction must be float32, got {prediction.dtype}")
    if prediction.size == 0:
        raise ValueError("prediction is empty")
    finite = np.isfinite(prediction)
    finite_fraction = float(finite.mean())
    if finite_fraction != 1.0:
        raise ValueError(f"prediction contains non-finite pixels: {finite_fraction=}")
    # CroCo's regression head is unconstrained.  Preserve finite negative
    # estimates so the common evaluator counts them as errors; clipping would
    # silently improve the baseline and rejecting them would make results
    # depend on scene content.
    nonnegative_fraction = float((prediction >= 0).mean())
    return PredictionStats(
        shape=(int(prediction.shape[0]), int(prediction.shape[1])),
        dtype=str(prediction.dtype),
        finite_fraction=finite_fraction,
        nonnegative_fraction=nonnegative_fraction,
        minimum=float(prediction.min()),
        maximum=float(prediction.max()),
    )


class CroCoStereoAdapter:
    """Load the frozen Spring-fine-tuned CroCo-Stereo backbone."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        croco_root: Path = DEFAULT_CROCO_ROOT,
        device: str = "cuda",
        tile_overlap: float = 0.9,
        amp_dtype: str = "none",
    ) -> None:
        if not 0.0 <= tile_overlap < 1.0:
            raise ValueError("tile_overlap must be in [0, 1)")
        if amp_dtype not in {"none", "bf16"}:
            raise ValueError("amp_dtype must be 'none' or 'bf16'")
        self.checkpoint = checkpoint.resolve()
        self.croco_root = croco_root.resolve()
        self.device = torch.device(device)
        self.tile_overlap = tile_overlap
        self.amp_dtype = amp_dtype
        self.commit = verify_repository(self.croco_root)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        if self.checkpoint.stat().st_size != 1_749_871_601:
            raise RuntimeError(
                f"unexpected checkpoint size: {self.checkpoint.stat().st_size} bytes"
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        if sha256_file(self.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("CroCo checkpoint SHA256 mismatch")

        _ensure_upstream_imports(self.croco_root)
        downstream = importlib.import_module("models.croco_downstream")
        heads = importlib.import_module("models.head_downstream")
        # This checkpoint is downloaded from the official Naver URL and contains
        # an argparse.Namespace. PyTorch >=2.6 therefore requires an explicit,
        # trusted weights_only=False load.
        checkpoint_data = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_args = checkpoint_data["args"]
        if checkpoint_args.task != "stereo":
            raise ValueError(f"expected stereo checkpoint, got {checkpoint_args.task!r}")
        criterion_spec = str(checkpoint_args.criterion)
        self.with_confidence = criterion_spec.startswith("LaplacianLoss")
        head = heads.PixelwiseTaskWithDPT()
        head.num_channels = 1 + int(self.with_confidence)
        self.model = downstream.CroCoDownstreamBinocular(
            head, **checkpoint_args.croco_args
        )
        load_result = self.model.load_state_dict(checkpoint_data["model"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"strict checkpoint load failed: {load_result}")
        self.model.eval().to(self.device)
        self.crop = tuple(int(v) for v in checkpoint_args.crop)
        self.tile_conf_mode = str(checkpoint_args.tile_conf_mode)
        self.checkpoint_training = {
            "task": str(checkpoint_args.task),
            "criterion": criterion_spec,
            "crop": list(self.crop),
            "tile_conf_mode": self.tile_conf_mode,
        }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
