"""Identity, denoising and sharpening candidates for stereo pairs."""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageFilter
import torch

from stablebridge.backbone import IMAGENET_MEAN, IMAGENET_STD


CORE_ACTION_IDS = ("P0", "P1", "P2")
CORE_ACTION_NAMES = ("identity", "binomial3_denoise", "unsharp_r2_p150")


def _apply_core_expert(value: np.ndarray, action: str) -> np.ndarray:
    """Apply a denoising or sharpening treatment to one RGB image."""

    if action == "binomial3_denoise":
        kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
        filtered = cv2.sepFilter2D(
            value.astype(np.float32),
            ddepth=cv2.CV_32F,
            kernelX=kernel,
            kernelY=kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        output = np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
    elif action == "unsharp_r2_p150":
        output = np.array(
            Image.fromarray(value).filter(
                ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=2)
            ),
            dtype=np.uint8,
            copy=True,
        )
    else:
        raise ValueError(f"unsupported Core-3 action: {action}")
    if output.shape != value.shape or output.dtype != np.uint8:
        raise RuntimeError(f"Core-3 action {action} changed the RGB contract")
    return output


def _validate_view(value: np.ndarray, name: str) -> None:
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be HxWx3 uint8 RGB")


def core3_candidates_uint8(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return exact P0/P1/P2 pairs in frozen action-grammar order."""

    _validate_view(left, "left")
    _validate_view(right, "right")
    if left.shape != right.shape:
        raise ValueError("left/right shape mismatch")
    values: list[tuple[np.ndarray, np.ndarray]] = [(left.copy(), right.copy())]
    for action in CORE_ACTION_NAMES[1:]:
        treated = (_apply_core_expert(left, action), _apply_core_expert(right, action))
        if treated[0].shape != left.shape or treated[1].shape != right.shape:
            raise RuntimeError(f"{action} changed stereo image geometry")
        values.append(treated)
    return tuple(values)


def normalize_uint8_rgb(value: np.ndarray) -> torch.Tensor:
    """Match the existing CroCo adapter's uint8-to-ImageNet path exactly."""

    _validate_view(value, "value")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)
    tensor = tensor.float().div_(255.0).unsqueeze(0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD
