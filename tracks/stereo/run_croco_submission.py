#!/usr/bin/env python3
"""Generate official Spring/RobustSpring ``.dsp5`` files with CroCo-Stereo.

This runner deliberately implements only the released CroCo identity path.
Experimental routing must not be added until its full-image policy is frozen
and evaluated end to end.  The runner is resumable, writes both reference
views, and never reads test ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import types
from typing import Any, Iterable, Iterator

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tracks.stereo.baselines.croco.adapter import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CROCO_ROOT,
    EXPECTED_CHECKPOINT_SHA256,
    CroCoStereoAdapter,
    canonicalize_reference_prediction,
    load_rgb_tensor,
    right_reference_inputs,
    sha256_file,
    validate_prediction,
    write_json,
)
from tracks.stereo.package_submission import DEFAULT_CLEAN_TEST, DEFAULT_ROBUST_TEST
from tracks.stereo.submission_contract import (
    CORRUPTIONS,
    TEST_SEQUENCE_LENGTHS,
    inspect_dsp5,
    write_dsp5,
)


DEFAULT_OUTPUT = Path(
    "/local/neurips/experiments/stereo/test/croco_stereo_released/spring-robust"
)


@dataclass(frozen=True)
class PairJob:
    variant: str
    scene: str
    frame: int
    left_image: Path
    right_image: Path
    left_prediction: Path
    right_prediction: Path


@dataclass(frozen=True)
class DirectionRequest:
    """One reference-view output to include in a batched CroCo call."""

    job: PairJob
    reference: str
    output: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-test-root", type=Path, default=DEFAULT_CLEAN_TEST)
    parser.add_argument("--robust-test-root", type=Path, default=DEFAULT_ROBUST_TEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--croco-root", type=Path, default=DEFAULT_CROCO_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-overlap", type=float, default=0.9)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="none")
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=4,
        help="Stereo pairs per call (both references use a batch of 2x this value).",
    )
    parser.add_argument(
        "--disable-rope-position-cache",
        action="store_true",
        help="Disable the fixed-tile fallback-RoPE synchronization optimization.",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS)
    )
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument(
        "--one-per-scene",
        action="store_true",
        help="Canary: retain only the first requested pair in each sequence.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _valid_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    inspect_dsp5(path, check_values=True)
    return True


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


@torch.inference_mode()
def predict_direction_batch(
    adapter: CroCoStereoAdapter,
    requests: list[DirectionRequest],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Predict several pairs/references in one official tiled pass."""

    if not requests:
        raise ValueError("direction batch must not be empty")
    image1_batch: list[torch.Tensor] = []
    image2_batch: list[torch.Tensor] = []
    shape: tuple[int, ...] | None = None
    loaded: dict[tuple[Path, Path], tuple[torch.Tensor, torch.Tensor]] = {}
    for request in requests:
        if request.reference not in {"left", "right"}:
            raise ValueError("reference must be 'left' or 'right'")
        key = (request.job.left_image, request.job.right_image)
        if key not in loaded:
            left = load_rgb_tensor(request.job.left_image)
            right = load_rgb_tensor(request.job.right_image)
            if left.shape != right.shape:
                raise ValueError(
                    "left/right image shapes differ: "
                    f"{tuple(left.shape)} vs {tuple(right.shape)}"
                )
            loaded[key] = (left, right)
        left, right = loaded[key]
        first, second = (
            right_reference_inputs(left, right)
            if request.reference == "right"
            else (left, right)
        )
        if shape is None:
            shape = tuple(first.shape)
        elif tuple(first.shape) != shape:
            raise ValueError("all images in a direction batch must share one shape")
        image1_batch.append(first)
        image2_batch.append(second)

    image1 = torch.cat(image1_batch).to(adapter.device, non_blocking=True)
    image2 = torch.cat(image2_batch).to(adapter.device, non_blocking=True)
    engine = importlib.import_module("stereoflow.engine")
    if adapter.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(adapter.device)
        torch.cuda.synchronize(adapter.device)
    started = time.perf_counter()
    amp_enabled = adapter.amp_dtype == "bf16"
    with torch.autocast(
        device_type=adapter.device.type,
        dtype=torch.bfloat16 if amp_enabled else None,
        enabled=amp_enabled,
    ):
        prediction, _unused_loss, _confidence, upstream_seconds = engine.tiled_pred(
            adapter.model,
            None,
            image1,
            image2,
            None,
            conf_mode=adapter.tile_conf_mode,
            overlap=adapter.tile_overlap,
            crop=adapter.crop,
            with_conf=adapter.with_confidence,
            return_time=True,
        )
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(adapter.device))
    else:
        peak_vram_bytes = 0
    wall_seconds = time.perf_counter() - started

    height, width = shape[-2:] if shape is not None else (0, 0)
    disparities: list[np.ndarray] = []
    for index, request in enumerate(requests):
        canonical = canonicalize_reference_prediction(
            prediction[index : index + 1], reference=request.reference
        )
        disparity = (
            canonical[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
        )
        if disparity.shape != (height, width):
            raise RuntimeError(
                "CroCo did not return input-resolution disparity: "
                f"input={(height, width)}, output={disparity.shape}"
            )
        validate_prediction(disparity)
        disparities.append(disparity)
    return disparities, {
        "directions": len(requests),
        "pairs_loaded": len(loaded),
        "runtime_seconds": float(wall_seconds),
        "upstream_cuda_event_seconds": float(upstream_seconds),
        "peak_vram_bytes": peak_vram_bytes,
    }


def _run_record(args: argparse.Namespace, requested_pairs: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "croco_stereo_official_test_inference",
        "status": "initializing",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "policy": "released_croco_identity",
        "claim_boundary": (
            "format/reference implementation only; does not include the experimental "
            "safe-denoise router or ShiftGuard"
        ),
        "requested_pairs": requested_pairs,
        "completed_pairs": 0,
        "new_outputs": 0,
        "skipped_outputs": 0,
        "output_root": str(args.output_root.resolve()),
        "clean_test_root": str(args.clean_test_root.resolve()),
        "robust_test_root": str(args.robust_test_root.resolve()),
        "variants": list(args.variants),
        "tile_overlap": float(args.tile_overlap),
        "amp_dtype": args.amp_dtype,
        "batch_pairs": int(args.batch_pairs),
        "rope_position_cache_requested": not args.disable_rope_position_cache,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "bytes": args.checkpoint.stat().st_size if args.checkpoint.is_file() else None,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "gpu": None,
        },
        "output_contract": {
            "dataset": "/disparity",
            "shape": [1080, 1920, 1],
            "dtype": "float32",
            "references": ["left", "right"],
            "atomic_writes": True,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit_pairs is not None and args.limit_pairs <= 0:
        raise ValueError("--limit-pairs must be positive")
    if args.batch_pairs <= 0:
        raise ValueError("--batch-pairs must be positive")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must be unique")

    jobs = list(
        iter_pair_jobs(
            args.output_root,
            args.clean_test_root,
            args.robust_test_root,
            args.variants,
        )
    )
    if args.one_per_scene:
        selected: list[PairJob] = []
        seen: set[tuple[str, str]] = set()
        for job in jobs:
            key = (job.variant, job.scene)
            if key not in seen:
                selected.append(job)
                seen.add(key)
        jobs = selected
    if args.limit_pairs is not None:
        jobs = jobs[: args.limit_pairs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = args.output_root / "run.json"
    run = _run_record(args, len(jobs))
    write_json(run_path, run)

    try:
        checkpoint_sha256 = sha256_file(args.checkpoint)
        run["checkpoint"]["sha256"] = checkpoint_sha256
        if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "checkpoint SHA256 mismatch: expected "
                f"{EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
            )
        adapter = CroCoStereoAdapter(
            checkpoint=args.checkpoint,
            croco_root=args.croco_root,
            device=args.device,
            tile_overlap=args.tile_overlap,
            amp_dtype=args.amp_dtype,
        )
        adapter.model.requires_grad_(False).eval()
        rope_modules_patched = (
            0
            if args.disable_rope_position_cache
            else enable_fixed_geometry_rope_cache(adapter.model)
        )
        run["repository_commit"] = adapter.commit
        run["rope_position_cache"] = {
            "enabled": rope_modules_patched > 0,
            "patched_modules": rope_modules_patched,
            "scope": "fixed checkpoint tile geometry",
        }
        run["environment"]["gpu"] = (
            torch.cuda.get_device_name(adapter.device)
            if adapter.device.type == "cuda"
            else None
        )
        run["status"] = "running"
        run["updated_at_utc"] = utc_now()
        write_json(run_path, run)

        for start in range(0, len(jobs), args.batch_pairs):
            batch_jobs = jobs[start : start + args.batch_pairs]
            requests: list[DirectionRequest] = []
            for job in batch_jobs:
                if not job.left_image.is_file() or not job.right_image.is_file():
                    raise FileNotFoundError(
                        f"missing input pair: {job.left_image}, {job.right_image}"
                    )
                for reference, output in (
                    ("left", job.left_prediction),
                    ("right", job.right_prediction),
                ):
                    if output.exists() and not args.overwrite:
                        _valid_existing(output)
                        run["skipped_outputs"] += 1
                    else:
                        requests.append(DirectionRequest(job, reference, output))

            batch_metadata: dict[str, Any] | None = None
            if requests:
                disparities, batch_metadata = predict_direction_batch(adapter, requests)
                for request, disparity in zip(requests, disparities, strict=True):
                    write_dsp5(request.output, disparity)
                    inspect_dsp5(request.output, check_values=True)
                    run["new_outputs"] += 1

            run["completed_pairs"] += len(batch_jobs)
            run["last_batch"] = {
                "first": {
                    "variant": batch_jobs[0].variant,
                    "scene": batch_jobs[0].scene,
                    "frame": batch_jobs[0].frame,
                },
                "last": {
                    "variant": batch_jobs[-1].variant,
                    "scene": batch_jobs[-1].scene,
                    "frame": batch_jobs[-1].frame,
                },
                "pairs": len(batch_jobs),
                "inference": batch_metadata,
            }
            run["updated_at_utc"] = utc_now()
            write_json(run_path, run)

        run["status"] = "completed"
        run["finished_at_utc"] = utc_now()
        run["updated_at_utc"] = run["finished_at_utc"]
        write_json(run_path, run)
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        run["traceback"] = traceback.format_exc()
        run["updated_at_utc"] = utc_now()
        write_json(run_path, run)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
