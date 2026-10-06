#!/usr/bin/env python3
"""Validate and package Stereo predictions with the official Spring tools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracks.stereo.submission_contract import (
    CORRUPTIONS,
    EXPECTED_FILES_PER_VARIANT,
    inspect_dsp5,
    inspect_submission_hdf5,
    inventory_digest,
    iter_prediction_items,
    sha256_file,
)


DEFAULT_PREDICTIONS = Path(
    "/local/neurips/experiments/stereo/test/raft_stereo_sceneflow/spring-robust"
)
DEFAULT_TOOLS = Path(
    "/local/neurips/tools/subsampling_executables_Spring_RobustSpring"
)
DEFAULT_RUNTIME = Path(
    "/local/neurips/tools/hdf5-1.14.6-runtime/usr/lib/x86_64-linux-gnu"
)
DEFAULT_OUTPUT = Path("/local/neurips/experiments/stereo/submission")
DEFAULT_CLEAN_TEST = Path("/local/neurips/datasets/spring/extracted/spring/test")
DEFAULT_ROBUST_TEST = Path(
    "/local/neurips/datasets/robustspring/extracted/robust_spring"
)
OFFICIAL_TOOL_SHA256 = {
    "disp1_subsampling": "ca9dcf3f7730837afcde193c09176ae8d4c8bc6c2a5dd91c98304255fa154e95",
    "disp1_robust_subsampling": "8a1da4efe0f3f871af1a223b219564e222022680139b376b414cefc3755d28be",
}
OUTPUT_NAMES = {
    "clean": "disp1_submission.hdf5",
    "robust": "disp1_robustness.hdf5",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selected_modes(mode: str) -> tuple[str, ...]:
    return ("clean", "robust") if mode == "all" else (mode,)


def selected_variants(modes: tuple[str, ...]) -> tuple[str, ...]:
    # The robust package includes clean, so avoid validating the clean tree twice.
    return CORRUPTIONS if "robust" in modes else ("clean",)


def disparity_count(root: Path) -> int:
    return sum(1 for path in root.rglob("*.dsp5") if path.is_file())


def validate_predictions(
    root: Path,
    modes: tuple[str, ...],
    *,
    clean_test_root: Path,
    robust_test_root: Path,
    validation_level: str,
    max_examples: int = 20,
) -> tuple[list[str], dict[str, Any]]:
    """Validate paths and HDF5 contracts against the official hidden-test inputs."""

    errors: list[str] = []
    variants = selected_variants(modes)
    report: dict[str, Any] = {
        "predictions_root": str(root.resolve()),
        "clean_test_root": str(clean_test_root.resolve()),
        "robust_test_root": str(robust_test_root.resolve()),
        "validation_level": validation_level,
        "expected_files_per_variant": EXPECTED_FILES_PER_VARIANT,
        "variants": {},
    }
    inventory_rows: list[tuple[str, int, str, tuple[int, ...]]] = []

    if "robust" in modes and root.is_dir():
        found_variants = {path.name for path in root.iterdir() if path.is_dir()}
        missing_variants = sorted(set(CORRUPTIONS) - found_variants)
        extra_variants = sorted(found_variants - set(CORRUPTIONS))
        report["variant_directories"] = {
            "missing": missing_variants,
            "extra": extra_variants,
        }
        if missing_variants:
            errors.append(f"missing prediction variant directories: {missing_variants}")
        if extra_variants:
            errors.append(f"unexpected prediction variant directories: {extra_variants}")

    for variant in variants:
        items = list(
            iter_prediction_items(
                root, clean_test_root, robust_test_root, variants=(variant,)
            )
        )
        missing_images: list[str] = []
        missing_predictions: list[str] = []
        invalid_predictions: list[str] = []
        checked = 0
        negative_values = 0
        minimum: float | None = None
        maximum: float | None = None
        expected_paths = {item.prediction_path.resolve() for item in items}

        for item in items:
            if not item.image_path.is_file():
                if len(missing_images) < max_examples:
                    missing_images.append(str(item.image_path))
            if not item.prediction_path.is_file():
                if len(missing_predictions) < max_examples:
                    missing_predictions.append(str(item.prediction_path))
                continue
            try:
                info = inspect_dsp5(
                    item.prediction_path,
                    check_values=validation_level == "full",
                ) if validation_level != "paths" else {
                    "shape": [],
                    "dtype": "unchecked",
                    "negative_values": None,
                    "minimum": None,
                    "maximum": None,
                }
            except (OSError, ValueError) as exc:
                if len(invalid_predictions) < max_examples:
                    invalid_predictions.append(f"{item.prediction_path}: {exc}")
                continue
            checked += 1
            relative = item.prediction_path.relative_to(root).as_posix()
            inventory_rows.append(
                (
                    relative,
                    item.prediction_path.stat().st_size,
                    info["dtype"],
                    tuple(info["shape"]),
                )
            )
            if info["negative_values"] is not None:
                negative_values += int(info["negative_values"])
                if info["minimum"] is not None:
                    minimum = (
                        float(info["minimum"])
                        if minimum is None
                        else min(minimum, float(info["minimum"]))
                    )
                    maximum = (
                        float(info["maximum"])
                        if maximum is None
                        else max(maximum, float(info["maximum"]))
                    )

        variant_root = root / variant / "test"
        actual_paths = (
            {path.resolve() for path in variant_root.rglob("*.dsp5") if path.is_file()}
            if variant_root.is_dir()
            else set()
        )
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        missing_image_count = sum(1 for item in items if not item.image_path.is_file())
        missing_prediction_count = sum(
            1 for item in items if not item.prediction_path.is_file()
        )
        invalid_count = len(items) - missing_prediction_count - checked

        if missing_image_count:
            errors.append(
                f"{variant}: missing {missing_image_count} official input images; "
                f"examples={missing_images}"
            )
        if missing_prediction_count:
            errors.append(
                f"{variant}: missing {missing_prediction_count}/{len(items)} predictions; "
                f"examples={missing_predictions}"
            )
        if invalid_count:
            errors.append(
                f"{variant}: {invalid_count} invalid predictions; "
                f"examples={invalid_predictions}"
            )
        if unexpected:
            errors.append(
                f"{variant}: {len(unexpected)} unexpected .dsp5 files; "
                f"examples={unexpected[:max_examples]}"
            )

        report["variants"][variant] = {
            "expected": len(items),
            "checked": checked,
            "missing_input_images": missing_image_count,
            "missing_predictions": missing_prediction_count,
            "invalid_predictions": invalid_count,
            "unexpected_predictions": len(unexpected),
            "negative_values": negative_values if validation_level == "full" else None,
            "minimum": minimum,
            "maximum": maximum,
        }

    inventory_rows.sort()
    report["inventory_sha256"] = inventory_digest(inventory_rows)
    report["inventory_files"] = len(inventory_rows)
    report["status"] = "PASS" if not errors else "FAIL"
    return errors, report


def check_predictions(root: Path, modes: tuple[str, ...]) -> list[str]:
    """Backward-compatible count check used by older callers."""

    errors: list[str] = []
    for variant in selected_variants(modes):
        variant_root = root / variant / "test"
        count = disparity_count(variant_root) if variant_root.is_dir() else 0
        if count != EXPECTED_FILES_PER_VARIANT:
            errors.append(
                f"{variant} predictions: {count}/{EXPECTED_FILES_PER_VARIANT} disparity files"
            )
    return errors


def validate_tools(tools: Path, modes: tuple[str, ...]) -> tuple[list[str], dict]:
    errors: list[str] = []
    details = {}
    tool_names = {
        "clean": "disp1_subsampling",
        "robust": "disp1_robust_subsampling",
    }
    for mode in modes:
        name = tool_names[mode]
        path = tools / name
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"missing executable: {path}")
            continue
        digest = sha256_file(path)
        expected = OFFICIAL_TOOL_SHA256[name]
        details[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "expected_sha256": expected,
            "verified": digest == expected,
        }
        if digest != expected:
            errors.append(
                f"official tool hash mismatch for {path}: expected {expected}, got {digest}"
            )
    return errors, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--tools-root", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-test-root", type=Path, default=DEFAULT_CLEAN_TEST)
    parser.add_argument("--robust-test-root", type=Path, default=DEFAULT_ROBUST_TEST)
    parser.add_argument("--mode", choices=("clean", "robust", "all"), default="all")
    parser.add_argument(
        "--validation-level",
        choices=("paths", "metadata", "full"),
        default="metadata",
        help="full additionally reads every disparity value and rejects NaN/Inf",
    )
    parser.add_argument("--method-id", help="Stable benchmark method/run identifier")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Checkpoint(s) to hash into the submission record",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        action="append",
        default=[],
        help="Additional JSON provenance file(s) to hash and embed",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create HDF5 files. Without this option only validate and print commands.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing package files in output-dir",
    )
    parser.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="Allow execute without a checkpoint or run-metadata (not for final submissions)",
    )
    return parser.parse_args()


def _provenance_files(args: argparse.Namespace) -> tuple[list[str], list[dict]]:
    errors: list[str] = []
    rows: list[dict] = []
    for kind, paths in (("checkpoint", args.checkpoint), ("run_metadata", args.run_metadata)):
        for raw_path in paths:
            path = raw_path.resolve()
            if not path.is_file():
                errors.append(f"missing {kind}: {path}")
                continue
            row = {
                "kind": kind,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if kind == "run_metadata":
                try:
                    row["content"] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid run metadata {path}: {exc}")
            rows.append(row)
    return errors, rows


def main() -> int:
    args = parse_args()
    predictions = args.predictions_root.resolve()
    tools = args.tools_root.resolve()
    output = args.output_dir.resolve()
    modes = selected_modes(args.mode)

    commands = {
        "clean": [
            str(tools / "disp1_subsampling"),
            str(predictions / "clean" / "test"),
        ],
        "robust": [str(tools / "disp1_robust_subsampling"), str(predictions)],
    }
    errors, prediction_report = validate_predictions(
        predictions,
        modes,
        clean_test_root=args.clean_test_root.resolve(),
        robust_test_root=args.robust_test_root.resolve(),
        validation_level=args.validation_level,
    )
    tool_errors, tool_report = validate_tools(tools, modes)
    errors.extend(tool_errors)
    provenance_errors, provenance = _provenance_files(args)
    errors.extend(provenance_errors)

    if args.execute and not args.method_id:
        errors.append("--method-id is required with --execute")
    if (
        args.execute
        and not args.allow_missing_provenance
        and not args.checkpoint
        and not args.run_metadata
    ):
        errors.append(
            "final packaging requires --checkpoint or --run-metadata "
            "(or explicitly use --allow-missing-provenance for a smoke test)"
        )
    for mode in modes:
        target = output / OUTPUT_NAMES[mode]
        if args.execute and target.exists() and not args.overwrite:
            errors.append(f"output exists (use --overwrite to replace): {target}")
        print(f"working directory: {output}")
        print(f"command: {shlex.join(commands[mode])}")

    if errors:
        print("Submission preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(
        "submission preflight: PASS "
        f"({prediction_report['inventory_files']} exact prediction files, "
        f"validation={args.validation_level})"
    )
    if not args.execute:
        print("dry run only; add --execute and provenance arguments to build HDF5 files")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "submission-record.json"
    log_path = output / "submission-packaging.log"
    if record_path.exists() and not args.overwrite:
        raise SystemExit(f"submission record exists (use --overwrite): {record_path}")

    env = os.environ.copy()
    runtime = str(args.runtime_root.resolve())
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = runtime if not current else f"{runtime}:{current}"

    started_at = utc_now()
    package_report: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=".packaging-", dir=output) as temporary:
        temporary_path = Path(temporary)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"started_at_utc: {started_at}\n")
            for mode in modes:
                command = commands[mode]
                log.write(f"command: {shlex.join(command)}\n")
                log.flush()
                subprocess.run(
                    command,
                    cwd=temporary_path,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

        for mode in modes:
            name = OUTPUT_NAMES[mode]
            generated = temporary_path / name
            if not generated.is_file() or generated.stat().st_size == 0:
                raise SystemExit(f"official tool did not create a usable output: {generated}")
            package_report[mode] = inspect_submission_hdf5(generated, mode=mode)
            destination = output / name
            if destination.exists() and args.overwrite:
                destination.unlink()
            os.replace(generated, destination)
            package_report[mode]["path"] = str(destination)
            print(
                f"created: {destination} ({destination.stat().st_size} bytes, "
                f"sha256={package_report[mode]['sha256']})"
            )

    record = {
        "schema_version": 1,
        "status": "completed",
        "method_id": args.method_id,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "modes": list(modes),
        "prediction_validation": prediction_report,
        "official_tools": tool_report,
        "provenance_files": provenance,
        "packages": package_report,
        "packaging_log": str(log_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "package_script": str(Path(__file__).resolve()),
            "package_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "official_sources": {
            "submission_faq": "https://spring-benchmark.org/faq",
            "stereo_devkit": (
                "https://github.com/hmorimitsu/roco-spring-devkit/"
                "blob/main/roco_spring_devkit/stereo/README.md"
            ),
        },
    }
    atomic_json(record_path, record)
    print(f"recorded: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
