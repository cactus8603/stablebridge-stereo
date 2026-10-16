from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import h5py,numpy as np
from PIL import Image
from .corruptions import corrupt_stereo,transform_left_disparity

def _full_views_and_center_gt(
    record: dict[str, Any], dataset_root: Path, crop: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    left = np.array(
        Image.open(dataset_root / record["left"]).convert("RGB"), copy=True
    )
    right = np.array(
        Image.open(dataset_root / record["right"]).convert("RGB"), copy=True
    )
    if left.shape != right.shape:
        raise ValueError(f"view shape mismatch for {record['id']}")
    height, width = left.shape[:2]
    crop_height, crop_width = crop
    if crop_height > height or crop_width > width:
        raise ValueError(f"crop {crop} exceeds image {(height, width)}")
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    with h5py.File(dataset_root / record["gt"], "r") as handle:
        raw_gt = handle["disparity"][()]
    if raw_gt.shape == (height * 2, width * 2):
        gt = raw_gt[
            2 * y0 : 2 * (y0 + crop_height) : 2,
            2 * x0 : 2 * (x0 + crop_width) : 2,
        ]
    elif raw_gt.shape == (height, width):
        gt = raw_gt[y0 : y0 + crop_height, x0 : x0 + crop_width]
    else:
        raise ValueError(
            f"unexpected GT shape {raw_gt.shape} for RGB {(height, width)}"
        )
    return left, right, gt.astype(np.float32, copy=False), y0, x0


def _full_left_gt(record: dict[str, Any], dataset_root: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load Spring left disparity at the RGB resolution."""

    height, width = image_shape
    with h5py.File(dataset_root / record["gt"], "r") as handle:
        raw = handle["disparity"][()]
    if raw.shape == (height * 2, width * 2):
        return raw[::2, ::2].astype(np.float32, copy=False)
    if raw.shape == (height, width):
        return raw.astype(np.float32, copy=False)
    raise ValueError(f"unexpected GT shape {raw.shape} for RGB {(height, width)}")
@dataclass(frozen=True)
class CellJob:
    record: dict[str, Any]
    condition: str
    severity: int

def materialize_condition(
    left: np.ndarray,
    right: np.ndarray,
    full_gt: np.ndarray,
    job: CellJob,
    sample_seed: int,
    coupling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    clean_valid = np.isfinite(full_gt) & (np.abs(full_gt) < 10_000.0)
    if job.condition == "clean":
        return (
            left.copy(),
            right.copy(),
            full_gt,
            clean_valid,
            {
                "backend": "robust_proxy20",
                "name": "clean",
                "family": "clean",
                "severity": 0,
                "seed": int(sample_seed),
                "changes_geometry": False,
            },
        )
    result = corrupt_stereo(
        left,
        right,
        job.condition,
        severity=job.severity,  # type: ignore[arg-type]
        coupling=coupling,  # type: ignore[arg-type]
        seed=sample_seed,
        left_disparity=full_gt,
    )
    if bool(result.metadata["changes_geometry"]):
        target, target_valid = transform_left_disparity(full_gt, result)
    else:
        target, target_valid = full_gt, clean_valid
    return result.left, result.right, target, target_valid, result.metadata


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray | None = None,
) -> dict[str, int | float]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            f"prediction/target must be same HxW shape, got {prediction.shape}/{target.shape}"
        )
    valid = np.isfinite(target) & (np.abs(target) < 10_000.0)
    if target_valid is not None:
        if np.asarray(target_valid).shape != target.shape:
            raise ValueError("target_valid shape mismatch")
        valid &= np.asarray(target_valid, dtype=bool)
    valid &= np.isfinite(prediction)
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise ValueError("cell has zero valid evaluation pixels")
    error = np.abs(prediction[valid].astype(np.float64) - target[valid].astype(np.float64))
    magnitude = np.abs(target[valid].astype(np.float64))
    over_1 = error > 1.0
    d1 = (error > 3.0) & (error > 0.05 * magnitude)
    return {
        "abs_px": float(error.mean(dtype=np.float64)),
        "over_1px_percent": float(100.0 * over_1.mean(dtype=np.float64)),
        "d1_percent": float(100.0 * d1.mean(dtype=np.float64)),
        "valid_pixels": valid_pixels,
        "abs_error_sum": float(error.sum(dtype=np.float64)),
        "over_1px_count": int(over_1.sum()),
        "d1_count": int(d1.sum()),
    }
