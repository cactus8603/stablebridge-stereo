"""Canonical project-local paths for Stereo assets.

Checkpoint paths are derived from the source checkout, so relocating the
whole ``roco_spring`` directory does not require editing Python files.  Set
``ROCO_SPRING_CHECKPOINT_ROOT`` only when the weights live somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ASSET_ROOT = WORKSPACE_ROOT / "stereo"
CHECKPOINT_ROOT = Path(
    os.environ.get("ROCO_SPRING_CHECKPOINT_ROOT", ASSET_ROOT / "checkpoints")
).expanduser()
PYTHON = Path(
    os.environ.get("ROCO_SPRING_PYTHON", WORKSPACE_ROOT / ".conda-stereo/bin/python")
).expanduser()
EXPERIMENT_ROOT = Path(
    os.environ.get("ROCO_SPRING_EXPERIMENT_ROOT", ASSET_ROOT / "experiments")
).expanduser()
SPRING_DATASET_ROOT = Path(
    os.environ.get(
        "ROCO_SPRING_DATASET_ROOT",
        "/ssd4/cactus8603/rocospring/datasets/spring/extracted/spring",
    )
).expanduser()
ROBUSTSPRING_DATASET_ROOT = Path(
    os.environ.get(
        "ROCO_ROBUSTSPRING_DATASET_ROOT",
        "/ssd4/cactus8603/rocospring/datasets/robustspring/extracted/robust_spring",
    )
).expanduser()
SUBSAMPLING_TOOLS_ROOT = ASSET_ROOT / "tools" / "subsampling_tools"


def checkpoint_path(*parts: str) -> Path:
    """Return a path below the canonical checkpoint root."""

    return CHECKPOINT_ROOT.joinpath(*parts)
