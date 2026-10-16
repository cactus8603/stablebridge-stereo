"""Strict Spring/RobustSpring Stereo submission validation helpers.

The official bundlers use a fixed set of ten Spring test sequences and expect
one left- and one right-reference ``.dsp5`` file per frame.  This module keeps
that path contract separate from model inference and from the opaque official
subsampling executables.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterable, Iterator
import uuid

import h5py
import numpy as np


CORRUPTIONS = (
    "clean",
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "rain",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
)

# This is the sequence contract compiled into the official subsampling tools.
TEST_SEQUENCE_LENGTHS = {
    "0003": 131,
    "0019": 111,
    "0028": 39,
    "0029": 135,
    "0031": 73,
    "0034": 47,
    "0035": 120,
    "0040": 111,
    "0042": 116,
    "0046": 117,
}
SIDES = ("left", "right")
EXPECTED_FILES_PER_VARIANT = 2 * sum(TEST_SEQUENCE_LENGTHS.values())
EXPECTED_HEIGHT = 1080
EXPECTED_WIDTH = 1920
# Exact lengths emitted by the currently pinned official binaries.  Clean adds
# ten full-resolution hero frames after subsampling all 2,000 direction maps;
# robust stores a second 5% subsample for every one of the 21 variants.
REDUCED_VALUES_PER_MAP = 51_839
CLEAN_HERO_FRAMES = 10
EXPECTED_CLEAN_PACKAGE_ELEMENTS = (
    EXPECTED_FILES_PER_VARIANT * REDUCED_VALUES_PER_MAP
    + CLEAN_HERO_FRAMES * EXPECTED_HEIGHT * EXPECTED_WIDTH
)
EXPECTED_ROBUST_ELEMENTS_PER_VARIANT = int(
    0.05 * EXPECTED_FILES_PER_VARIANT * REDUCED_VALUES_PER_MAP
)


@dataclass(frozen=True)
class PredictionItem:
    variant: str
    scene: str
    side: str
    frame: int
    image_path: Path
    prediction_path: Path

def iter_prediction_items(
    predictions_root: Path,
    clean_test_root: Path,
    robust_test_root: Path,
    variants: Iterable[str],
) -> Iterator[PredictionItem]:
    """Yield the exact official image/prediction pairs for ``variants``."""

    for variant in variants:
        if variant not in CORRUPTIONS:
            raise ValueError(f"unknown RobustSpring variant: {variant}")
        image_test_root = (
            clean_test_root if variant == "clean" else robust_test_root / variant / "test"
        )
        prediction_test_root = predictions_root / variant / "test"
        for scene, length in TEST_SEQUENCE_LENGTHS.items():
            for side in SIDES:
                for frame in range(1, length + 1):
                    stem = f"{side}_{frame:04d}"
                    yield PredictionItem(
                        variant=variant,
                        scene=scene,
                        side=side,
                        frame=frame,
                        image_path=(
                            image_test_root
                            / scene
                            / f"frame_{side}"
                            / f"frame_{stem}.png"
                        ),
                        prediction_path=(
                            prediction_test_root
                            / scene
                            / f"disp1_{side}"
                            / f"disp1_{stem}.dsp5"
                        ),
                    )


def _canonical_shape(shape: tuple[int, ...]) -> tuple[int, int] | None:
    if shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        return EXPECTED_HEIGHT, EXPECTED_WIDTH
    if shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH, 1):
        return EXPECTED_HEIGHT, EXPECTED_WIDTH
    return None


def inspect_dsp5(path: Path, *, check_values: bool) -> dict:
    """Validate one official disparity file and return compact diagnostics."""

    with h5py.File(path, "r") as handle:
        if "disparity" not in handle:
            raise ValueError("missing /disparity dataset")
        dataset = handle["disparity"]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("/disparity is not a dataset")
        shape = tuple(int(value) for value in dataset.shape)
        if _canonical_shape(shape) is None:
            raise ValueError(
                f"unexpected disparity shape {shape}; expected "
                f"({EXPECTED_HEIGHT}, {EXPECTED_WIDTH}) with optional trailing singleton"
            )
        if dataset.dtype != np.dtype("float32"):
            raise ValueError(f"unexpected disparity dtype {dataset.dtype}; expected float32")

        result = {
            "shape": list(shape),
            "dtype": str(dataset.dtype),
            "finite": None,
            "negative_values": None,
            "minimum": None,
            "maximum": None,
        }
        if check_values:
            values = np.asarray(dataset[()])
            finite = np.isfinite(values)
            result["finite"] = bool(finite.all())
            result["negative_values"] = int(np.count_nonzero(values < 0))
            if finite.any():
                finite_values = values[finite]
                result["minimum"] = float(finite_values.min())
                result["maximum"] = float(finite_values.max())
            if not result["finite"]:
                raise ValueError(
                    f"contains {int(values.size - np.count_nonzero(finite))} NaN/Inf values"
                )
        return result


def write_dsp5(path: Path, disparity: np.ndarray) -> None:
    """Atomically write one full-resolution Spring disparity prediction.

    The official examples use a trailing singleton channel, so the writer
    emits ``[1080, 1920, 1]`` even though the validator also accepts ``HxW``.
    Compression keeps the 42,000-file RobustSpring prediction tree within the
    documented storage budget.  A temporary file in the destination directory
    prevents an interrupted inference process from leaving a valid-looking
    partial prediction behind.
    """

    values = np.asarray(disparity)
    if values.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(
            f"unexpected disparity shape {values.shape}; expected "
            f"({EXPECTED_HEIGHT}, {EXPECTED_WIDTH})"
        )
    if values.dtype != np.dtype("float32"):
        raise ValueError(f"unexpected disparity dtype {values.dtype}; expected float32")
    if not np.isfinite(values).all():
        raise ValueError("disparity contains NaN/Inf")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset(
                "disparity",
                data=values[..., None],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inventory_digest(rows: Iterable[tuple[str, int, str, tuple[int, ...]]]) -> str:
    """Hash a deterministic metadata inventory without rereading all file bytes."""

    digest = hashlib.sha256()
    for relative, size, dtype, shape in rows:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, shape)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_submission_hdf5(
    path: Path, *, mode: str, enforce_expected_size: bool = True
) -> dict:
    """Validate the HDF5 generated by an official Stereo subsampling tool."""

    with h5py.File(path, "r") as handle:
        if mode == "clean":
            if set(handle.keys()) != {"disparity"}:
                raise ValueError(
                    f"clean package keys are {sorted(handle.keys())}, expected ['disparity']"
                )
            datasets = {"disparity": handle["disparity"]}
        elif mode == "robust":
            if set(handle.keys()) != set(CORRUPTIONS):
                missing = sorted(set(CORRUPTIONS) - set(handle.keys()))
                extra = sorted(set(handle.keys()) - set(CORRUPTIONS))
                raise ValueError(
                    f"robust package group mismatch; missing={missing}, extra={extra}"
                )
            datasets = {
                variant: handle[f"{variant}/disparity"] for variant in CORRUPTIONS
            }
        else:
            raise ValueError(f"unknown submission mode: {mode}")

        details = {}
        for name, dataset in datasets.items():
            if dataset.ndim != 1 or dataset.size <= 0:
                raise ValueError(f"{name}: expected a non-empty 1-D dataset")
            if dataset.dtype != np.dtype("float16"):
                raise ValueError(f"{name}: expected float16, got {dataset.dtype}")
            expected_size = (
                EXPECTED_CLEAN_PACKAGE_ELEMENTS
                if mode == "clean"
                else EXPECTED_ROBUST_ELEMENTS_PER_VARIANT
            )
            if enforce_expected_size and dataset.size != expected_size:
                raise ValueError(
                    f"{name}: package has {dataset.size} elements, expected {expected_size}"
                )
            # Packaged files are small enough for a final finite-value audit and this
            # catches float16 overflow introduced by the official conversion.
            values = np.asarray(dataset[()])
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: packaged disparity contains NaN/Inf")
            details[name] = {
                "elements": int(dataset.size),
                "expected_elements": expected_size,
                "dtype": str(dataset.dtype),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }

    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "datasets": details,
    }
