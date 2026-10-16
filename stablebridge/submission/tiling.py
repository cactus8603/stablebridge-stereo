from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable,Iterator
import types
import torch
from .contract import CORRUPTIONS,TEST_SEQUENCE_LENGTHS

@dataclass(frozen=True)
class PairJob:
    variant: str
    scene: str
    frame: int
    left_image: Path
    right_image: Path
    left_prediction: Path
    right_prediction: Path


def enable_fixed_geometry_rope_cache(model: torch.nn.Module) -> int:
    """Remove a redundant GPU-to-CPU sync from fallback RoPE inference.

    The upstream PyTorch fallback obtains the maximum position with
    ``int(positions.max())`` on every transformer layer and every tile.  The
    submission runner always feeds the checkpoint's fixed crop geometry, so
    the required sequence length is constant.  Caching that scalar preserves
    the exact RoPE operations while avoiding thousands of synchronizations per
    image.  CUDA-compiled RoPE modules are left untouched.
    """

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "RoPE2D" or not hasattr(
            module, "get_cos_sin"
        ):
            continue
        module._submission_seq_len_cache = {}

        def cached_forward(
            self: torch.nn.Module,
            tokens: torch.Tensor,
            positions: torch.Tensor,
        ) -> torch.Tensor:
            if tokens.size(3) % 2 != 0:
                raise ValueError("RoPE feature dimension must be even")
            if positions.ndim != 3 or positions.shape[-1] != 2:
                raise ValueError("RoPE positions must have shape BxNx2")
            dimension = tokens.size(3) // 2
            key = (
                dimension,
                int(positions.shape[1]),
                positions.device,
                positions.dtype,
            )
            sequence_length = self._submission_seq_len_cache.get(key)
            if sequence_length is None:
                sequence_length = int(positions.max()) + 1
                self._submission_seq_len_cache[key] = sequence_length
            cos, sin = self.get_cos_sin(
                dimension,
                sequence_length,
                tokens.device,
                tokens.dtype,
            )
            y_tokens, x_tokens = tokens.chunk(2, dim=-1)
            y_tokens = self.apply_rope1d(
                y_tokens, positions[:, :, 0], cos, sin
            )
            x_tokens = self.apply_rope1d(
                x_tokens, positions[:, :, 1], cos, sin
            )
            return torch.cat((y_tokens, x_tokens), dim=-1)

        module.forward = types.MethodType(cached_forward, module)
        patched += 1
    return patched


def iter_pair_jobs(
    output_root: Path,
    clean_test_root: Path,
    robust_test_root: Path,
    variants: Iterable[str],
) -> Iterator[PairJob]:
    """Yield one stereo inference job for every official test frame."""

    for variant in variants:
        if variant not in CORRUPTIONS:
            raise ValueError(f"unknown RobustSpring variant: {variant}")
        image_root = (
            clean_test_root if variant == "clean" else robust_test_root / variant / "test"
        )
        prediction_root = output_root / variant / "test"
        for scene, length in TEST_SEQUENCE_LENGTHS.items():
            for frame in range(1, length + 1):
                yield PairJob(
                    variant=variant,
                    scene=scene,
                    frame=frame,
                    left_image=(
                        image_root
                        / scene
                        / "frame_left"
                        / f"frame_left_{frame:04d}.png"
                    ),
                    right_image=(
                        image_root
                        / scene
                        / "frame_right"
                        / f"frame_right_{frame:04d}.png"
                    ),
                    left_prediction=(
                        prediction_root
                        / scene
                        / "disp1_left"
                        / f"disp1_left_{frame:04d}.dsp5"
                    ),
                    right_prediction=(
                        prediction_root
                        / scene
                        / "disp1_right"
                        / f"disp1_right_{frame:04d}.dsp5"
                    ),
                )
