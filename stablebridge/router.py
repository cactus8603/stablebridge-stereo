"""Deployment helpers for the frozen final CroCo-StableBridge controller.

The controller observes five fixed, official-grid-aligned crops and chooses
one geometry-preserving action for the complete stereo pair.  It never reads
ground truth or corruption labels.  The same action is applied to both views
and therefore also to both official disparity reference requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from stablebridge.backbone import sha256_file
from stablebridge.treatments import (
    CORE_ACTION_IDS,
    CORE_ACTION_NAMES,
    core3_candidates_uint8,
    normalize_uint8_rgb,
)
from stablebridge.model import (
    EvidenceProbe,
    StereoEvidence,
    encode_frozen_prefix,
)
from stablebridge.paths import checkpoint_path


DEFAULT_BUNDLE = checkpoint_path("model.json")


@dataclass(frozen=True)
class RouteResult:
    action: int
    candidates: tuple[tuple[np.ndarray, np.ndarray], ...]
    diagnostics: dict[str, Any]

    @property
    def selected_pair(self) -> tuple[np.ndarray, np.ndarray]:
        return self.candidates[self.action]


def overlapping_offsets(total: int, window: int, overlap: float) -> tuple[int, ...]:
    """Reproduce the official CroCo tiler's integer window offsets."""

    if total < window:
        raise ValueError(f"image dimension {total} is below crop dimension {window}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if total == window:
        return (0,)
    count = 1 + int(np.ceil((total - window) / ((1.0 - overlap) * window)))
    values = np.linspace(0, total - window, count).round().astype(np.int64)
    return tuple(int(value) for value in values)


def representative_tiles(
    height: int,
    width: int,
    *,
    crop: tuple[int, int] = (352, 704),
    overlap: float = 0.9,
) -> tuple[tuple[int, int], ...]:
    """Return center plus four quadrant crops snapped to the official grid."""

    crop_h, crop_w = crop
    ys = np.asarray(overlapping_offsets(height, crop_h, overlap), dtype=np.int64)
    xs = np.asarray(overlapping_offsets(width, crop_w, overlap), dtype=np.int64)
    target_centers = (
        (0.50 * height, 0.50 * width),
        (0.25 * height, 0.25 * width),
        (0.25 * height, 0.75 * width),
        (0.75 * height, 0.25 * width),
        (0.75 * height, 0.75 * width),
    )
    selected: list[tuple[int, int]] = []
    for center_y, center_x in target_centers:
        desired_y = center_y - crop_h / 2.0
        desired_x = center_x - crop_w / 2.0
        y = int(ys[np.argmin(np.abs(ys - desired_y))])
        x = int(xs[np.argmin(np.abs(xs - desired_x))])
        if (y, x) not in selected:
            selected.append((y, x))
    return tuple(selected)


def aggregate_tile_actions(tile_actions: Iterable[int]) -> int:
    """Conservative global-pair vote with identity on disagreement or a tie."""

    values = np.asarray(list(tile_actions), dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("tile_actions must be a non-empty vector")
    if np.any((values < 0) | (values >= len(CORE_ACTION_IDS))):
        raise ValueError("tile action outside Core-3 grammar")
    counts = np.bincount(values, minlength=len(CORE_ACTION_IDS))
    non_identity = int(np.argmax(counts[1:]) + 1)
    # A treatment needs a strict majority of all representative tiles.  P1/P2
    # disagreement, a tie, and an identity majority all safely abstain to P0.
    return non_identity if counts[non_identity] > len(values) // 2 else 0


class StableBridgeRouter:
    """Load and execute the frozen three-seed, dual-depth final controller."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        bundle_path: Path = DEFAULT_BUNDLE,
        device: str | torch.device = "cuda",
        crop: tuple[int, int] = (352, 704),
        overlap: float = 0.9,
        tile_batch_size: int = 1,
    ) -> None:
        if tile_batch_size <= 0:
            raise ValueError("tile_batch_size must be positive")
        self.device = torch.device(device)
        self.backbone = backbone.requires_grad_(False).eval()
        self.bundle_path = bundle_path.resolve()
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        self.crop = tuple(int(v) for v in crop)
        self.overlap = float(overlap)
        self.tile_batch_size = int(tile_batch_size)
        self._validate_bundle()

        evidence_row = self.bundle["evidence"]
        evidence_path = self._resolve_artifact(evidence_row["path"])
        self._verify_file(evidence_path, evidence_row["sha256"])
        saved_evidence = torch.load(
            evidence_path, map_location="cpu", weights_only=True
        )
        self.evidence = StereoEvidence(
            projection_dim=int(saved_evidence["projection_dim"])
        )
        self.evidence.load_state_dict(saved_evidence["model"], strict=True)
        self.evidence.to(self.device).requires_grad_(False).eval()

        self.heads: dict[int, nn.ModuleList] = {1: nn.ModuleList(), 8: nn.ModuleList()}
        for row in self.bundle["heads"]:
            depth = int(row["depth"])
            path = self._resolve_artifact(row["path"])
            self._verify_file(path, row["sha256"])
            saved = torch.load(path, map_location="cpu", weights_only=True)
            head = EvidenceProbe(
                feature_dim=int(saved["feature_dim"]), width=int(saved["width"])
            )
            head.load_state_dict(saved["model"], strict=True)
            self.heads[depth].append(head.to(self.device).requires_grad_(False).eval())
        if any(len(value) != len(self.bundle["seeds"]) for value in self.heads.values()):
            raise RuntimeError("deployment bundle is missing ensemble heads")

    def _resolve_artifact(self, value: str) -> Path:
        """Resolve artifacts relative to their model manifest."""
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.bundle_path.parent / path).resolve()

    @staticmethod
    def _verify_file(path: Path, expected_sha256: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"deployment artifact hash mismatch for {path}: {actual}"
            )

    def _validate_bundle(self) -> None:
        if self.bundle.get("status") != "frozen_development_bundle":
            raise RuntimeError("StableBridge deployment bundle is not frozen")
        if self.bundle.get("depths") != [1, 8]:
            raise RuntimeError("StableBridge requires the frozen D1/D8 contract")
        policy = self.bundle.get("policy", {})
        if policy.get("actions") != list(CORE_ACTION_IDS):
            raise RuntimeError("StableBridge Core-3 action order mismatch")
        if policy.get("selection") != "q10" or policy.get("score") != "benefit_minus_harm":
            raise RuntimeError("StableBridge policy definition mismatch")
        if not policy.get("identity_fallback"):
            raise RuntimeError("StableBridge deployment requires identity fallback")

    def _ensemble(self, depth: int, feature: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(feature) for head in self.heads[depth]]).mean(dim=0)

    @torch.inference_mode()
    def _route_tile_batch(
        self,
        candidates: tuple[tuple[np.ndarray, np.ndarray], ...],
        locations: tuple[tuple[int, int], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        crop_h, crop_w = self.crop
        normalized: list[torch.Tensor] = []
        for y, x in locations:
            for candidate_left, candidate_right in candidates:
                normalized.extend(
                    (
                        normalize_uint8_rgb(
                            candidate_left[y : y + crop_h, x : x + crop_w]
                        ),
                        normalize_uint8_rgb(
                            candidate_right[y : y + crop_h, x : x + crop_w]
                        ),
                    )
                )
        images = torch.cat(normalized).to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.device.type == "cuda" else None,
            enabled=self.device.type == "cuda",
        ):
            current, positions = encode_frozen_prefix(self.backbone, images, depth=1)
        batch = len(locations)
        feature1 = self.evidence(current.reshape(batch, 3, 2, 968, 1024))
        output1 = self._ensemble(1, feature1)
        for depth in range(2, 9):
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.device.type == "cuda" else None,
                enabled=self.device.type == "cuda",
            ):
                current = self.backbone.enc_blocks[depth - 1](current, positions)
        feature8 = self.evidence(current.reshape(batch, 3, 2, 968, 1024))
        output8 = self._ensemble(8, feature8)
        return output1.float().cpu().numpy(), output8.float().cpu().numpy()

    def route(self, left: np.ndarray, right: np.ndarray) -> RouteResult:
        candidates = core3_candidates_uint8(left, right)
        height, width = left.shape[:2]
        locations = representative_tiles(
            height, width, crop=self.crop, overlap=self.overlap
        )
        depth1_rows: list[np.ndarray] = []
        depth8_rows: list[np.ndarray] = []
        for start in range(0, len(locations), self.tile_batch_size):
            batch_locations = locations[start : start + self.tile_batch_size]
            output1, output8 = self._route_tile_batch(candidates, batch_locations)
            depth1_rows.append(output1)
            depth8_rows.append(output8)
        depth1 = np.concatenate(depth1_rows)
        depth8 = np.concatenate(depth8_rows)
        fused = np.zeros_like(depth1)
        fused[:, :, :3] = depth1[:, :, :3]
        fused[:, :, 3:5] = depth8[:, :, 3:5]

        selected = fused[:, :, 1].argmax(axis=1)
        row = np.arange(len(fused))
        benefit = 1.0 / (1.0 + np.exp(-fused[row, selected, 2]))
        harm = 1.0 / (1.0 + np.exp(-fused[row, selected, 3]))
        score = benefit - harm
        threshold = float(self.bundle["policy"]["threshold"])
        tile_actions = np.where(score >= threshold, selected + 1, 0).astype(np.int64)
        action = aggregate_tile_actions(tile_actions)

        diagnostics = {
            "action": int(action),
            "action_id": CORE_ACTION_IDS[action],
            "action_name": CORE_ACTION_NAMES[action],
            "threshold": threshold,
            "tile_locations_yx": [list(value) for value in locations],
            "tile_actions": tile_actions.tolist(),
            "tile_action_ids": [CORE_ACTION_IDS[value] for value in tile_actions],
            "selected_treatment_indices": selected.tolist(),
            "selected_q10": fused[row, selected, 1].astype(float).tolist(),
            "selected_benefit_probability": benefit.astype(float).tolist(),
            "selected_harm_probability": harm.astype(float).tolist(),
            "selected_score": score.astype(float).tolist(),
            "depth1_output": depth1.astype(float).tolist(),
            "depth8_output": depth8.astype(float).tolist(),
        }
        return RouteResult(action=action, candidates=candidates, diagnostics=diagnostics)
