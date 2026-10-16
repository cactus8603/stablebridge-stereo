"""Render deterministic local corruptions for Spring training pairs.

Motion and weather are image-space approximations of RobustSpring. Elastic
transforms return sampling maps for disparity remapping.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image


Severity = Literal[1, 2, 3]
Coupling = Literal["official", "shared", "independent"]
RENDERER_VERSION = "0.2.0"


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    family: str
    official_coupling: Literal["shared", "independent"]
    official_mean_ssim: float
    time_consistent: bool
    depth_consistent: bool = False
    changes_geometry: bool = False
    exact_official_renderer: bool = True


@dataclass(frozen=True)
class StereoCorruption:
    left: np.ndarray
    right: np.ndarray
    metadata: dict[str, Any]
    left_map_x: np.ndarray | None = None
    left_map_y: np.ndarray | None = None
    right_map_x: np.ndarray | None = None
    right_map_y: np.ndarray | None = None


_SPECS = (
    CorruptionSpec("brightness", "color", "shared", 0.70, True),
    CorruptionSpec("contrast", "color", "shared", 0.70, True),
    CorruptionSpec("saturate", "color", "shared", 0.72, True),
    CorruptionSpec("defocus_blur", "blur", "shared", 0.70, True),
    CorruptionSpec("gaussian_blur", "blur", "shared", 0.70, True),
    CorruptionSpec("glass_blur", "blur", "independent", 0.73, True),
    CorruptionSpec("motion_blur", "blur", "independent", 0.75, True, True, exact_official_renderer=False),
    CorruptionSpec("zoom_blur", "blur", "shared", 0.70, True),
    CorruptionSpec("gaussian_noise", "noise", "independent", 0.20, False),
    CorruptionSpec("impulse_noise", "noise", "independent", 0.20, False),
    CorruptionSpec("speckle_noise", "noise", "independent", 0.20, False),
    CorruptionSpec("shot_noise", "noise", "independent", 0.22, False),
    CorruptionSpec("pixelate", "quality", "shared", 0.70, True),
    CorruptionSpec("jpeg_compression", "quality", "shared", 0.70, True),
    CorruptionSpec("elastic_transform", "quality", "independent", 0.70, True, changes_geometry=True),
    CorruptionSpec("spatter", "weather", "independent", 0.72, True, exact_official_renderer=False),
    CorruptionSpec("frost", "weather", "independent", 0.73, True, exact_official_renderer=False),
    CorruptionSpec("snow", "weather", "shared", 0.70, True, True, exact_official_renderer=False),
    CorruptionSpec("rain", "weather", "shared", 0.70, True, True, exact_official_renderer=False),
    CorruptionSpec("fog", "weather", "shared", 0.71, True, True, exact_official_renderer=False),
)

CORRUPTION_SPECS = {spec.name: spec for spec in _SPECS}
CORRUPTION_NAMES = tuple(spec.name for spec in _SPECS)


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 uint8 RGB array")
    return np.ascontiguousarray(value)


def _stable_seed(seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{token}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _float_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)


def _severity_parameters(name: str, severity: Severity) -> dict[str, float | int]:
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    table: dict[str, tuple[dict[str, float | int], ...]] = {
        "brightness": ({"offset": 0.20}, {"offset": 0.39}, {"offset": 0.55}),
        "contrast": ({"factor": 0.58}, {"factor": 0.16}, {"factor": 0.08}),
        "saturate": (
            {"alpha": 1.6, "beta": 0.005},
            {"alpha": 2.3, "beta": 0.01},
            {"alpha": 3.0, "beta": 0.015},
        ),
        "defocus_blur": ({"radius": 3}, {"radius": 6}, {"radius": 9}),
        "gaussian_blur": ({"sigma": 2.0}, {"sigma": 4.0}, {"sigma": 6.0}),
        "glass_blur": (
            {"sigma": 0.8, "iterations": 1, "radius": 2},
            {"sigma": 1.2, "iterations": 1, "radius": 3},
            {"sigma": 1.8, "iterations": 2, "radius": 4},
        ),
        "motion_blur": ({"length": 9}, {"length": 17}, {"length": 25}),
        "zoom_blur": ({"zoom_max": 1.12}, {"zoom_max": 1.24}, {"zoom_max": 1.36}),
        "gaussian_noise": ({"alpha": 0.06}, {"alpha": 0.115}, {"alpha": 0.17}),
        "impulse_noise": ({"fraction": 0.035}, {"fraction": 0.075}, {"fraction": 0.12}),
        "speckle_noise": ({"alpha": 0.25}, {"alpha": 0.45}, {"alpha": 0.65}),
        "shot_noise": ({"count": 45.0}, {"count": 23.0}, {"count": 12.0}),
        "pixelate": ({"fraction": 0.32}, {"fraction": 0.16}, {"fraction": 0.10}),
        "jpeg_compression": ({"quality": 25}, {"quality": 6}, {"quality": 3}),
        "elastic_transform": (
            {"alpha": 55.0, "sigma": 5.0},
            {"alpha": 110.0, "sigma": 5.0},
            {"alpha": 165.0, "sigma": 5.0},
        ),
        "spatter": ({"coverage": 0.14}, {"coverage": 0.26}, {"coverage": 0.38}),
        "frost": ({"strength": 0.38}, {"strength": 0.65}, {"strength": 0.82}),
        "snow": ({"density": 0.002}, {"density": 0.004}, {"density": 0.008}),
        "rain": ({"density": 0.0015}, {"density": 0.003}, {"density": 0.006}),
        "fog": ({"strength": 0.40}, {"strength": 0.62}, {"strength": 0.78}),
    }
    if name not in table:
        raise ValueError(f"unsupported corruption: {name}")
    return dict(table[name][severity - 1])


def _disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    return kernel / kernel.sum()


def _zoom(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    y0 = (resized.shape[0] - height) // 2
    x0 = (resized.shape[1] - width) // 2
    return resized[y0 : y0 + height, x0 : x0 + width]


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    theta = math.radians(angle)
    dx = math.cos(theta) * center
    dy = math.sin(theta) * center
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        color=1.0,
        thickness=1,
    )
    total = float(kernel.sum())
    return kernel / max(total, 1.0)


def _elastic_maps(shape: tuple[int, int], rng: np.random.Generator, alpha: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    dx = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
    # torchvision normalizes displacement by W/H and grid_sample converts
    # normalized coordinates back with a W/2 or H/2 factor.  The corresponding
    # pixel displacement is therefore approximately alpha/2.
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma) * (alpha / 2.0)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma) * (alpha / 2.0)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return xx + dx, yy + dy


def _procedural_field(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    field = np.zeros((height, width), dtype=np.float32)
    for divisor, weight in ((32, 0.55), (16, 0.30), (8, 0.15)):
        small = rng.random((max(2, height // divisor), max(2, width // divisor)), dtype=np.float32)
        field += weight * cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, 0.0, 1.0)


def _depth_fog_mask(disparity: np.ndarray | None, shape: tuple[int, int], strength: float) -> np.ndarray:
    if disparity is None:
        yy = np.linspace(0.7, 1.0, shape[0], dtype=np.float32)[:, None]
        return np.broadcast_to(strength * yy, shape).copy()
    value = np.asarray(disparity, dtype=np.float32)
    if value.shape != shape:
        raise ValueError("disparity shape must match its RGB view")
    valid = np.isfinite(value) & (value > 1e-3) & (np.abs(value) < 10_000)
    if not np.any(valid):
        return np.full(shape, strength, dtype=np.float32)
    inverse = np.zeros_like(value)
    inverse[valid] = 1.0 / value[valid]
    lo, hi = np.quantile(inverse[valid], (0.05, 0.95))
    normalized = np.clip((inverse - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    normalized[~valid] = 1.0
    return strength * (0.25 + 0.75 * normalized)


def _apply_one(
    image: np.ndarray,
    name: str,
    parameters: dict[str, float | int],
    seed: int,
    *,
    disparity: np.ndarray | None,
    elastic_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    rng = np.random.default_rng(seed)
    source = _float_image(image)
    if name == "brightness":
        return _uint8(source + float(parameters["offset"])), None
    if name == "contrast":
        mean = source.mean(axis=(0, 1), keepdims=True)
        return _uint8((source - mean) * float(parameters["factor"]) + mean), None
    if name == "saturate":
        hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(
            hsv[..., 1] * float(parameters["alpha"]) + float(parameters["beta"]),
            0.0,
            1.0,
        )
        return _uint8(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)), None
    if name == "defocus_blur":
        kernel = _disk_kernel(int(parameters["radius"]))
        return _uint8(cv2.filter2D(source, -1, kernel, borderType=cv2.BORDER_REFLECT_101)), None
    if name == "gaussian_blur":
        sigma = float(parameters["sigma"])
        return _uint8(cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)), None
    if name == "glass_blur":
        sigma = float(parameters["sigma"])
        radius = int(parameters["radius"])
        result = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
        height, width = result.shape[:2]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        for _ in range(int(parameters["iterations"])):
            dx = rng.integers(-radius, radius + 1, size=(height, width)).astype(np.float32)
            dy = rng.integers(-radius, radius + 1, size=(height, width)).astype(np.float32)
            result = cv2.remap(
                result,
                xx + dx,
                yy + dy,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        return _uint8(cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)), None
    if name == "motion_blur":
        kernel = _motion_kernel(int(parameters["length"]), float(rng.uniform(-45.0, 45.0)))
        return _uint8(cv2.filter2D(source, -1, kernel, borderType=cv2.BORDER_REFLECT_101)), None
    if name == "zoom_blur":
        factors = np.arange(1.02, float(parameters["zoom_max"]) + 1e-6, 0.02)
        total = source.astype(np.float32)
        for factor in factors:
            total += _zoom(source, float(factor))
        return _uint8(total / float(len(factors) + 1)), None
    if name == "gaussian_noise":
        return _uint8(source + rng.normal(0.0, float(parameters["alpha"]), source.shape)), None
    if name == "impulse_noise":
        output = source.copy()
        mask = rng.random(source.shape[:2]) < float(parameters["fraction"])
        salt = rng.random(source.shape[:2]) < 0.5
        output[mask & salt] = 1.0
        output[mask & ~salt] = 0.0
        return _uint8(output), None
    if name == "speckle_noise":
        noise = rng.normal(0.0, 1.0, source.shape).astype(np.float32)
        return _uint8(source + source * float(parameters["alpha"]) * noise), None
    if name == "shot_noise":
        count = float(parameters["count"])
        return _uint8(rng.poisson(source * count).astype(np.float32) / count), None
    if name == "pixelate":
        fraction = float(parameters["fraction"])
        height, width = source.shape[:2]
        small = cv2.resize(source, (max(1, round(width * fraction)), max(1, round(height * fraction))), interpolation=cv2.INTER_AREA)
        return _uint8(cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)), None
    if name == "jpeg_compression":
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=int(parameters["quality"]), subsampling=2)
        stream.seek(0)
        return np.array(Image.open(stream).convert("RGB"), dtype=np.uint8, copy=True), None
    if name == "elastic_transform":
        maps = elastic_maps or _elastic_maps(source.shape[:2], rng, float(parameters["alpha"]), float(parameters["sigma"]))
        result = cv2.remap(
            source,
            maps[0],
            maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return _uint8(result), maps
    if name == "spatter":
        coverage = float(parameters["coverage"])
        field = cv2.GaussianBlur(rng.random(source.shape[:2], dtype=np.float32), (0, 0), 3.0)
        threshold = float(np.quantile(field, 1.0 - coverage))
        mask = np.clip((field - threshold) * 30.0, 0.0, 0.9)[..., None]
        color = np.array([0.65, 0.78, 0.72], dtype=np.float32)
        return _uint8(source * (1.0 - mask) + color * mask), None
    if name == "frost":
        strength = float(parameters["strength"])
        field = _procedural_field(source.shape[:2], rng)
        mask = np.clip((field - 0.42) * 2.4, 0.0, 1.0)[..., None] * strength
        color = np.array([0.78, 0.90, 1.0], dtype=np.float32)
        return _uint8(source * (1.0 - mask) + color * mask), None
    if name in {"snow", "rain"}:
        height, width = source.shape[:2]
        layer = np.zeros((height, width), dtype=np.float32)
        count = max(1, round(height * width * float(parameters["density"])))
        xs = rng.integers(0, width, size=count)
        ys = rng.integers(0, height, size=count)
        if name == "snow":
            radii = rng.integers(1, 4, size=count)
            for x, y, radius in zip(xs, ys, radii, strict=True):
                cv2.circle(layer, (int(x), int(y)), int(radius), 1.0, -1)
            layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
            opacity = np.clip(layer * 0.75, 0.0, 0.85)[..., None]
        else:
            for x, y in zip(xs, ys, strict=True):
                cv2.line(layer, (int(x), int(y)), (int(x + 3), min(height - 1, int(y + 16))), 1.0, 1)
            layer = cv2.GaussianBlur(layer, (3, 3), 0.7)
            opacity = np.clip(layer * 0.55, 0.0, 0.7)[..., None]
        return _uint8(source * (1.0 - opacity) + opacity), None
    if name == "fog":
        mask = _depth_fog_mask(disparity, source.shape[:2], float(parameters["strength"]))[..., None]
        sky = np.full_like(source, 0.8)
        return _uint8(source * (1.0 - mask) + sky * mask), None
    raise ValueError(f"unsupported corruption: {name}")


def corrupt_stereo(
    left: np.ndarray,
    right: np.ndarray,
    name: str,
    *,
    severity: Severity = 2,
    coupling: Coupling = "official",
    seed: int = 0,
    left_disparity: np.ndarray | None = None,
    right_disparity: np.ndarray | None = None,
) -> StereoCorruption:
    """Apply one deterministic corruption to a stereo pair.

    ``shared`` means shared random realization/parameters on the two image
    grids; it does not claim depth-aware correspondence consistency.  The
    metadata records this limitation explicitly.
    """

    left = _validate_image(left, "left")
    right = _validate_image(right, "right")
    if left.shape != right.shape:
        raise ValueError("left and right images must have identical shape")
    if name not in CORRUPTION_SPECS:
        raise ValueError(f"unsupported corruption: {name}")
    if coupling not in {"official", "shared", "independent"}:
        raise ValueError("coupling must be official, shared, or independent")
    spec = CORRUPTION_SPECS[name]
    resolved = spec.official_coupling if coupling == "official" else coupling
    parameters = _severity_parameters(name, severity)
    left_seed = _stable_seed(seed, f"{name}:left")
    right_seed = left_seed if resolved == "shared" else _stable_seed(seed, f"{name}:right")
    shared_maps = None
    if name == "elastic_transform" and resolved == "shared":
        rng = np.random.default_rng(left_seed)
        shared_maps = _elastic_maps(
            left.shape[:2], rng, float(parameters["alpha"]), float(parameters["sigma"])
        )
    left_out, left_maps = _apply_one(
        left,
        name,
        parameters,
        left_seed,
        disparity=left_disparity,
        elastic_maps=shared_maps,
    )
    right_out, right_maps = _apply_one(
        right,
        name,
        parameters,
        right_seed,
        disparity=right_disparity,
        elastic_maps=shared_maps,
    )
    metadata = {
        "schema_version": 1,
        "renderer_version": RENDERER_VERSION,
        "name": name,
        "family": spec.family,
        "severity": severity,
        "parameters": parameters,
        "requested_coupling": coupling,
        "resolved_coupling": resolved,
        "shared_grid_realization_is_depth_consistent": False,
        "official_mean_ssim": spec.official_mean_ssim,
        "time_consistent_in_official_dataset": spec.time_consistent,
        "depth_consistent_in_official_dataset": spec.depth_consistent,
        "changes_geometry": spec.changes_geometry,
        "exact_official_renderer": spec.exact_official_renderer,
        "seed": int(seed),
    }
    return StereoCorruption(
        left=left_out,
        right=right_out,
        metadata=metadata,
        left_map_x=None if left_maps is None else left_maps[0],
        left_map_y=None if left_maps is None else left_maps[1],
        right_map_x=None if right_maps is None else right_maps[0],
        right_map_y=None if right_maps is None else right_maps[1],
    )


def _sample(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(array, dtype=np.float32),
        x.astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        # Bounds are checked explicitly by the caller.  Replication avoids
        # OpenCV blending NaN border values into otherwise exact edge samples.
        borderMode=cv2.BORDER_REPLICATE,
    )


def transform_left_disparity(
    disparity: np.ndarray,
    corruption: StereoCorruption,
    *,
    vertical_tolerance: float = 1.5,
    inverse_iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a clean left disparity through an elastic stereo pair.

    Maps follow OpenCV's output-to-source convention.  Right-map inversion is
    solved by fixed-point iterations.  Pixels whose transformed correspondence
    is out of bounds or no longer approximately rectified are marked invalid.
    """

    maps = (
        corruption.left_map_x,
        corruption.left_map_y,
        corruption.right_map_x,
        corruption.right_map_y,
    )
    if any(value is None for value in maps):
        value = np.asarray(disparity, dtype=np.float32)
        valid = np.isfinite(value) & (np.abs(value) < 10_000)
        return value.copy(), valid
    left_x, left_y, right_x, right_y = (np.asarray(value, dtype=np.float32) for value in maps)  # type: ignore[arg-type]
    height, width = left_x.shape
    source_disparity = _sample(disparity, left_x, left_y)
    source_right_x = left_x - source_disparity
    source_right_y = left_y

    output_right_x = source_right_x.copy()
    output_right_y = source_right_y.copy()
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
    displacement_x = right_x - grid_x
    displacement_y = right_y - grid_y
    for _ in range(max(1, inverse_iterations)):
        output_right_x = source_right_x - _sample(displacement_x, output_right_x, output_right_y)
        output_right_y = source_right_y - _sample(displacement_y, output_right_x, output_right_y)

    output_disparity = grid_x - output_right_x
    valid = (
        np.isfinite(source_disparity)
        & (np.abs(source_disparity) < 10_000)
        & np.isfinite(output_disparity)
        & (output_right_x >= 0)
        & (output_right_x <= width - 1)
        & (output_right_y >= 0)
        & (output_right_y <= height - 1)
        & (np.abs(grid_y - output_right_y) <= vertical_tolerance)
    )
    output_disparity = output_disparity.astype(np.float32)
    output_disparity[~valid] = np.nan
    return output_disparity, valid
