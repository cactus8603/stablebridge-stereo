#!/usr/bin/env python3
"""Generate official Spring ``.dsp5`` files with frozen StableBridge S0.

Every stereo pair is routed once from its raw RGB views.  The chosen Core-3
action is shared by the left/right images and by both official disparity
reference requests.  Test ground truth and corruption labels are never used.
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
from typing import Any, Iterable

import numpy as np
from PIL import Image
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
    right_reference_inputs,
    sha256_file,
    validate_prediction,
    write_json,
)
from tracks.stereo.croco_stablebridge.candidates import normalize_uint8_rgb
from tracks.stereo.croco_stablebridge.deploy import (
    DEFAULT_BUNDLE,
    StableBridgeS0Router,
)
from tracks.stereo.package_submission import DEFAULT_CLEAN_TEST, DEFAULT_ROBUST_TEST
from tracks.stereo.run_croco_submission import (
    PairJob,
    enable_fixed_geometry_rope_cache,
    iter_pair_jobs,
)
from tracks.stereo.submission_contract import CORRUPTIONS, inspect_dsp5, write_dsp5


DEFAULT_OUTPUT = Path(
    "/local/neurips/experiments/stereo/test/"
    "croco-stablebridge-s0/spring-robust"
)


@dataclass(frozen=True)
class RoutedPair:
    job: PairJob
    left: torch.Tensor
    right: torch.Tensor
    action: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RoutedDirection:
    pair: RoutedPair
    reference: str
    output: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-test-root", type=Path, default=DEFAULT_CLEAN_TEST)
    parser.add_argument("--robust-test-root", type=Path, default=DEFAULT_ROBUST_TEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--croco-root", type=Path, default=DEFAULT_CROCO_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-overlap", type=float, default=0.9)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="none")
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--route-tile-batch-size", type=int, default=1)
    parser.add_argument("--variants", nargs="+", choices=CORRUPTIONS, default=["clean"])
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument(
        "--one-per-scene", action="store_true",
        help="Canary: retain only the first requested pair in each sequence.",
    )
    parser.add_argument(
        "--allow-clean-treatment", action="store_true",
        help="Permit non-P0 routing on clean inputs (disabled for safety by default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-rope-position-cache", action="store_true")
    return parser.parse_args(argv)


def _valid_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    inspect_dsp5(path, check_values=True)
    return True


def route_record_path(output_root: Path, job: PairJob) -> Path:
    return (
        output_root / "routes" / job.variant / "test" / job.scene
        / f"frame_{job.frame:04d}.json"
    )


@torch.inference_mode()
def predict_routed_batch(
    adapter: CroCoStereoAdapter,
    requests: list[RoutedDirection],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not requests:
        raise ValueError("routed direction batch must not be empty")
    first_views: list[torch.Tensor] = []
    second_views: list[torch.Tensor] = []
    shape: tuple[int, ...] | None = None
    for request in requests:
        first, second = (
            right_reference_inputs(request.pair.left, request.pair.right)
            if request.reference == "right"
            else (request.pair.left, request.pair.right)
        )
        if shape is None:
            shape = tuple(first.shape)
        elif tuple(first.shape) != shape:
            raise ValueError("all images in a routed batch must share one shape")
        first_views.append(first)
        second_views.append(second)
    image1 = torch.cat(first_views).to(adapter.device, non_blocking=True)
    image2 = torch.cat(second_views).to(adapter.device, non_blocking=True)

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
        value = canonicalize_reference_prediction(
            prediction[index : index + 1], reference=request.reference
        )
        disparity = value[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
        if disparity.shape != (height, width):
            raise RuntimeError(
                f"CroCo output shape mismatch: {(height, width)} vs {disparity.shape}"
            )
        validate_prediction(disparity)
        disparities.append(disparity)
    return disparities, {
        "directions": len(requests),
        "pairs_loaded": len({id(request.pair) for request in requests}),
        "runtime_seconds": float(wall_seconds),
        "upstream_cuda_event_seconds": float(upstream_seconds),
        "peak_vram_bytes": peak_vram_bytes,
    }


def _select_jobs(jobs: list[PairJob], args: argparse.Namespace) -> list[PairJob]:
    if args.one_per_scene:
        kept: list[PairJob] = []
        seen: set[tuple[str, str]] = set()
        for job in jobs:
            key = (job.variant, job.scene)
            if key not in seen:
                kept.append(job)
                seen.add(key)
        jobs = kept
    if args.limit_pairs is not None:
        jobs = jobs[: args.limit_pairs]
    return jobs


def _run_record(args: argparse.Namespace, jobs: list[PairJob]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "croco-stablebridge-s0-official-test-inference",
        "status": "initializing",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "policy": "frozen_s0_global_pair_five_tile_majority",
        "claim_boundary": (
            "Submission inference and target-free routing only; official test ground "
            "truth is never read and the frozen threshold is not tuned on test inputs."
        ),
        "requested_pairs": len(jobs),
        "completed_pairs": 0,
        "new_outputs": 0,
        "skipped_outputs": 0,
        "route_action_counts": {"P0": 0, "P1": 0, "P2": 0},
        "output_root": str(args.output_root.resolve()),
        "clean_test_root": str(args.clean_test_root.resolve()),
        "robust_test_root": str(args.robust_test_root.resolve()),
        "variants": list(args.variants),
        "clean_identity_required": not args.allow_clean_treatment,
        "tile_overlap": float(args.tile_overlap),
        "amp_dtype": args.amp_dtype,
        "batch_pairs": int(args.batch_pairs),
        "route_tile_batch_size": int(args.route_tile_batch_size),
        "bundle": {"path": str(args.bundle.resolve())},
        "checkpoint": {"path": str(args.checkpoint.resolve())},
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
            "same_route_action_both_references": True,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_pairs <= 0 or args.route_tile_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.limit_pairs is not None and args.limit_pairs <= 0:
        raise ValueError("--limit-pairs must be positive")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must be unique")
    jobs = _select_jobs(list(iter_pair_jobs(
        args.output_root, args.clean_test_root, args.robust_test_root, args.variants
    )), args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = args.output_root / "run.json"
    run = _run_record(args, jobs)
    write_json(run_path, run)

    try:
        checkpoint_hash = sha256_file(args.checkpoint)
        if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("CroCo checkpoint SHA256 mismatch")
        run["checkpoint"]["sha256"] = checkpoint_hash
        run["bundle"]["sha256"] = sha256_file(args.bundle)
        adapter = CroCoStereoAdapter(
            checkpoint=args.checkpoint,
            croco_root=args.croco_root,
            device=args.device,
            tile_overlap=args.tile_overlap,
            amp_dtype=args.amp_dtype,
        )
        adapter.model.requires_grad_(False).eval()
        patched = 0 if args.disable_rope_position_cache else enable_fixed_geometry_rope_cache(
            adapter.model
        )
        router = StableBridgeS0Router(
            adapter.model,
            bundle_path=args.bundle,
            device=args.device,
            crop=adapter.crop,
            overlap=adapter.tile_overlap,
            tile_batch_size=args.route_tile_batch_size,
        )
        run["repository_commit"] = adapter.commit
        run["rope_position_cache"] = {"patched_modules": patched, "enabled": patched > 0}
        run["environment"]["gpu"] = (
            torch.cuda.get_device_name(adapter.device)
            if adapter.device.type == "cuda" else None
        )
        run["status"] = "running"
        write_json(run_path, run)

        for start in range(0, len(jobs), args.batch_pairs):
            batch_jobs = jobs[start : start + args.batch_pairs]
            requests: list[RoutedDirection] = []
            routed_pairs: list[RoutedPair] = []
            for job in batch_jobs:
                if not job.left_image.is_file() or not job.right_image.is_file():
                    raise FileNotFoundError(f"missing input pair: {job.left_image}, {job.right_image}")
                outputs_needed: list[tuple[str, Path]] = []
                for reference, output in (
                    ("left", job.left_prediction), ("right", job.right_prediction)
                ):
                    if output.exists() and not args.overwrite:
                        _valid_existing(output)
                        run["skipped_outputs"] += 1
                    else:
                        outputs_needed.append((reference, output))
                if not outputs_needed:
                    stored_route = route_record_path(args.output_root, job)
                    if stored_route.is_file():
                        stored = json.loads(stored_route.read_text(encoding="utf-8"))
                        action_id = stored["diagnostics"]["action_id"]
                        if action_id not in run["route_action_counts"]:
                            raise RuntimeError(f"invalid stored route action: {stored_route}")
                        run["route_action_counts"][action_id] += 1
                    continue
                route_started = time.monotonic()
                route = router.route(read_rgb(job.left_image), read_rgb(job.right_image))
                if job.variant == "clean" and route.action != 0 and not args.allow_clean_treatment:
                    raise RuntimeError(
                        f"clean safety stop: {job.scene}/{job.frame:04d} selected "
                        f"{route.diagnostics['action_id']}"
                    )
                treated_left, treated_right = route.selected_pair
                pair = RoutedPair(
                    job=job,
                    left=normalize_uint8_rgb(treated_left),
                    right=normalize_uint8_rgb(treated_right),
                    action=route.action,
                    diagnostics=route.diagnostics,
                )
                routed_pairs.append(pair)
                run["route_action_counts"][route.diagnostics["action_id"]] += 1
                write_json(route_record_path(args.output_root, job), {
                    "schema_version": 1,
                    "variant": job.variant,
                    "scene": job.scene,
                    "frame": job.frame,
                    "elapsed_seconds": time.monotonic() - route_started,
                    "diagnostics": route.diagnostics,
                })
                requests.extend(
                    RoutedDirection(pair, reference, output)
                    for reference, output in outputs_needed
                )

            inference = None
            if requests:
                disparities, inference = predict_routed_batch(adapter, requests)
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
                "routed_pairs": len(routed_pairs),
                "inference": inference,
            }
            run["updated_at_utc"] = utc_now()
            write_json(run_path, run)
            print(json.dumps({
                "completed_pairs": run["completed_pairs"],
                "requested_pairs": run["requested_pairs"],
                "route_action_counts": run["route_action_counts"],
                "new_outputs": run["new_outputs"],
                "last_batch": run["last_batch"],
            }), flush=True)

        run["status"] = "completed"
        run["finished_at_utc"] = utc_now()
        run["updated_at_utc"] = run["finished_at_utc"]
        write_json(run_path, run)
        print(json.dumps(run, indent=2, sort_keys=True), flush=True)
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
