#!/usr/bin/env python3
"""Run a resumable, Codex-independent Stereo inference/package pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracks.stereo.package_submission import (  # noqa: E402
    DEFAULT_CLEAN_TEST,
    DEFAULT_ROBUST_TEST,
    DEFAULT_RUNTIME,
    DEFAULT_TOOLS,
    OUTPUT_NAMES,
    selected_modes,
    validate_predictions,
    validate_tools,
)
from tracks.stereo.submission_contract import (  # noqa: E402
    CORRUPTIONS,
    iter_prediction_items,
    sha256_file,
)


DEFAULT_CONFIG = PROJECT_ROOT / "tracks/stereo/configs/submission-r0-raft-stereo.json"
DEFAULT_STATE = PROJECT_ROOT / "tracks/stereo/results/submission-pipeline-state.json"
DEFAULT_LOG = PROJECT_ROOT / "tracks/stereo/results/submission-pipeline.log"
DEFAULT_LOCK = PROJECT_ROOT / "tracks/stereo/results/baseline-daemon.lock"


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


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pipeline config {path}: {exc}") from exc
    required = ("method_id", "predictions_root", "output_dir", "inference")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"pipeline config is missing fields: {missing}")
    inference = config["inference"]
    if not isinstance(inference, dict) or not isinstance(inference.get("command"), list):
        raise ValueError("inference.command must be a JSON list of arguments")
    if not inference["command"] or not all(
        isinstance(value, str) and value for value in inference["command"]
    ):
        raise ValueError("inference.command must contain non-empty strings")
    return config


def base_state(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pending",
        "pid": os.getpid(),
        "method_id": config["method_id"],
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "stages": {},
    }


def update_state(path: Path, state: dict[str, Any], **values: Any) -> None:
    state.update(values)
    state["updated_at_utc"] = utc_now()
    atomic_json(path, state)


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "predictions": Path(config["predictions_root"]).resolve(),
        "output": Path(config["output_dir"]).resolve(),
        "tools": Path(config.get("tools_root", DEFAULT_TOOLS)).resolve(),
        "runtime": Path(config.get("runtime_root", DEFAULT_RUNTIME)).resolve(),
        "clean_test": Path(config.get("clean_test_root", DEFAULT_CLEAN_TEST)).resolve(),
        "robust_test": Path(config.get("robust_test_root", DEFAULT_ROBUST_TEST)).resolve(),
    }


def preflight(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    paths = resolve_paths(config)
    errors: list[str] = []
    modes = selected_modes(config.get("mode", "all"))

    if stage == "all":
        inference_cwd = Path(config["inference"].get("cwd", PROJECT_ROOT)).resolve()
        command_executable = Path(config["inference"]["command"][0])
        if not inference_cwd.is_dir():
            errors.append(f"missing inference cwd: {inference_cwd}")
        if command_executable.is_absolute() and not command_executable.is_file():
            errors.append(f"missing inference executable: {command_executable}")

    tool_errors, tool_report = validate_tools(paths["tools"], modes)
    errors.extend(tool_errors)
    if not paths["runtime"].is_dir():
        errors.append(f"missing HDF5 runtime directory: {paths['runtime']}")

    variants = CORRUPTIONS if "robust" in modes else ("clean",)
    dataset_counts: dict[str, dict[str, int]] = {}
    for variant in variants:
        items = iter_prediction_items(
            paths["predictions"],
            paths["clean_test"],
            paths["robust_test"],
            (variant,),
        )
        expected = 0
        present = 0
        for item in items:
            expected += 1
            present += int(item.image_path.is_file())
        dataset_counts[variant] = {"expected_images": expected, "present_images": present}
        if present != expected:
            errors.append(
                f"{variant}: official input images are incomplete ({present}/{expected})"
            )

    provenance = []
    for raw in [*config.get("checkpoints", []), *config.get("run_metadata", [])]:
        path = Path(raw).resolve()
        if not path.is_file():
            errors.append(f"missing provenance file: {path}")
        else:
            provenance.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not provenance:
        errors.append("pipeline config must provide checkpoints or run_metadata")

    storage_root = paths["predictions"].parent
    existing = storage_root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    free_gib = usage.free / 2**30
    minimum_free_gib = float(config.get("minimum_free_gib", 300.0))
    if stage == "all" and free_gib < minimum_free_gib:
        errors.append(
            f"only {free_gib:.2f} GiB free at {existing}; "
            f"pipeline requires {minimum_free_gib:.2f} GiB"
        )

    gpu: dict[str, Any] | None = None
    if stage == "all":
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if query.returncode != 0 or not query.stdout.strip():
            errors.append(f"GPU preflight failed: {query.stderr.strip()}")
        else:
            gpu = {"nvidia_smi": query.stdout.strip().splitlines()}

    report = {
        "status": "PASS" if not errors else "FAIL",
        "checked_at_utc": utc_now(),
        "stage": stage,
        "modes": list(modes),
        "paths": {key: str(value) for key, value in paths.items()},
        "official_tools": tool_report,
        "dataset_counts": dataset_counts,
        "gpu": gpu,
        "provenance": provenance,
        "storage": {
            "path": str(existing),
            "free_gib": free_gib,
            "minimum_free_gib": minimum_free_gib,
        },
        "errors": errors,
    }
    return report


def completed_package(config: dict[str, Any]) -> bool:
    output = Path(config["output_dir"]).resolve()
    record_path = output / "submission-record.json"
    if not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") != "completed" or record.get("method_id") != config["method_id"]:
        return False
    for mode in selected_modes(config.get("mode", "all")):
        path = output / OUTPUT_NAMES[mode]
        stored = record.get("packages", {}).get(mode, {})
        if not path.is_file() or stored.get("sha256") != sha256_file(path):
            return False
    return True


def predictions_complete(
    config: dict[str, Any], *, validation_level: str | None = None
) -> tuple[bool, dict[str, Any]]:
    paths = resolve_paths(config)
    errors, report = validate_predictions(
        paths["predictions"],
        selected_modes(config.get("mode", "all")),
        clean_test_root=paths["clean_test"],
        robust_test_root=paths["robust_test"],
        validation_level=validation_level or config.get("validation_level", "full"),
    )
    report["errors"] = errors
    return not errors, report


def package_command(config: dict[str, Any]) -> list[str]:
    paths = resolve_paths(config)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tracks/stereo/package_submission.py"),
        "--predictions-root",
        str(paths["predictions"]),
        "--tools-root",
        str(paths["tools"]),
        "--runtime-root",
        str(paths["runtime"]),
        "--output-dir",
        str(paths["output"]),
        "--clean-test-root",
        str(paths["clean_test"]),
        "--robust-test-root",
        str(paths["robust_test"]),
        "--mode",
        config.get("mode", "all"),
        "--validation-level",
        config.get("validation_level", "full"),
        "--method-id",
        config["method_id"],
        "--execute",
    ]
    for path in config.get("checkpoints", []):
        command.extend(("--checkpoint", str(Path(path).resolve())))
    for path in config.get("run_metadata", []):
        command.extend(("--run-metadata", str(Path(path).resolve())))
    return command


def run_pipeline(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    state = base_state(config_path, config)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)

    with args.lock.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another submission pipeline owns lock: {args.lock}")

        try:
            update_state(args.state, state, status="preflight", current="preflight")
            report = preflight(config, stage=args.stage)
            state["stages"]["preflight"] = report
            if report["errors"]:
                raise RuntimeError("; ".join(report["errors"]))
            update_state(args.state, state, status="running", current=None)

            if args.stage == "all":
                complete, validation = predictions_complete(config, validation_level="paths")
                if complete:
                    complete, validation = predictions_complete(config)
                if complete:
                    state["stages"]["inference"] = {
                        "status": "skipped_valid_existing_predictions",
                        "finished_at_utc": utc_now(),
                        "validation": validation,
                    }
                else:
                    command = config["inference"]["command"]
                    cwd = Path(config["inference"].get("cwd", PROJECT_ROOT)).resolve()
                    state["stages"]["inference"] = {
                        "status": "running",
                        "started_at_utc": utc_now(),
                        "cwd": str(cwd),
                        "command": command,
                        "preexisting_validation": validation,
                    }
                    update_state(args.state, state, status="inference", current="inference")
                    print(f"inference cwd: {cwd}", flush=True)
                    print(f"inference command: {shlex.join(command)}", flush=True)
                    subprocess.run(command, cwd=cwd, check=True)
                    complete, validation = predictions_complete(config)
                    if not complete:
                        raise RuntimeError(
                            "inference returned success but strict prediction validation failed: "
                            + "; ".join(validation["errors"])
                        )
                    state["stages"]["inference"].update(
                        status="completed",
                        finished_at_utc=utc_now(),
                        validation=validation,
                    )
                update_state(args.state, state, status="running", current=None)

            if completed_package(config):
                state["stages"]["package"] = {
                    "status": "skipped_valid_existing_package",
                    "finished_at_utc": utc_now(),
                }
            else:
                command = package_command(config)
                state["stages"]["package"] = {
                    "status": "running",
                    "started_at_utc": utc_now(),
                    "command": command,
                }
                update_state(args.state, state, status="packaging", current="package")
                print(f"package command: {shlex.join(command)}", flush=True)
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                if not completed_package(config):
                    raise RuntimeError("packager returned success but package audit failed")
                state["stages"]["package"].update(
                    status="completed", finished_at_utc=utc_now()
                )

            update_state(
                args.state,
                state,
                status="completed",
                current=None,
                finished_at_utc=utc_now(),
            )
            print(f"submission pipeline completed: {args.state}")
            return 0
        except BaseException as exc:
            state["error_type"] = type(exc).__name__
            state["error"] = str(exc)
            state["traceback"] = traceback.format_exc()
            update_state(
                args.state,
                state,
                status="failed",
                current=None,
                finished_at_utc=utc_now(),
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "start", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--stage", choices=("all", "package"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.state = args.state.resolve()
    args.log = args.log.resolve()
    args.lock = args.lock.resolve()

    if args.command == "status":
        if not args.state.is_file():
            print(f"no state file: {args.state}")
            return 1
        print(args.state.read_text(encoding="utf-8"), end="")
        return 0

    config = load_config(args.config.resolve())
    if args.command == "preflight":
        report = preflight(config, stage=args.stage)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if report["status"] == "PASS" else 2

    if args.command == "start":
        args.log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--config",
            str(args.config.resolve()),
            "--state",
            str(args.state),
            "--log",
            str(args.log),
            "--lock",
            str(args.lock),
            "--stage",
            args.stage,
        ]
        with args.log.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launching = base_state(args.config.resolve(), config)
        launching.update(
            status="launching",
            pid=process.pid,
            log=str(args.log),
            command=command,
        )
        atomic_json(args.state, launching)
        print(f"started submission pipeline pid={process.pid}")
        print(f"state: {args.state}")
        print(f"log: {args.log}")
        return 0

    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
