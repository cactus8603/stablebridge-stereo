from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .backbone import CroCoStereoAdapter
from .treatments import normalize_uint8_rgb as _normalize,core3_candidates_uint8
from .model import StereoEvidence,encode_frozen_prefix
from .optimization import seed_everything
from .data import _full_views_and_center_gt,_full_left_gt,CellJob,materialize_condition
EVIDENCE_SEED=20260824
FEATURE_DIM=227
CROP=(352,704)

class FeatureExtractor:
    def __init__(self, device: str, checkpoint: Path) -> None:
        seed_everything(EVIDENCE_SEED)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                "the frozen feature precision contract requires CUDA bf16 autocast"
            )
        self.adapter = CroCoStereoAdapter(
            checkpoint=checkpoint, device=device, amp_dtype="bf16"
        )
        self.backbone = self.adapter.model.requires_grad_(False).eval()
        self.evidence = (
            StereoEvidence(projection_dim=96)
            .to(self.device)
            .requires_grad_(False)
            .eval()
        )
        if self.evidence.output_dim != FEATURE_DIM:
            raise RuntimeError("fixed evidence dimension changed")
        dummy = torch.zeros(1, 3, *CROP, device=self.device)
        with torch.inference_mode():
            _tokens, positions = self.backbone.patch_embed(dummy)
        self.base_positions = positions

    def _prefix(
        self,
        teacher: dict[str, np.ndarray],
        index: int,
        record: dict[str, Any],
        dataset_root: Path,
    ) -> torch.Tensor:
        left, right, _target, y0, x0 = _full_views_and_center_gt(
            record, dataset_root, CROP
        )
        full_gt = _full_left_gt(record, dataset_root, left.shape[:2])
        job = CellJob(
            record=record,
            condition=str(teacher["condition"][index]),
            severity=int(teacher["severity"][index]),
        )
        corrupted_left, corrupted_right, _target, _valid, _metadata = (
            materialize_condition(
                left,
                right,
                full_gt,
                job,
                int(teacher["sample_seed"][index]),
                "official",
            )
        )
        candidates = core3_candidates_uint8(corrupted_left, corrupted_right)
        normalized = []
        for candidate_left, candidate_right in candidates:
            normalized.extend(
                (
                    _normalize(candidate_left)[
                        :, :, y0 : y0 + CROP[0], x0 : x0 + CROP[1]
                    ],
                    _normalize(candidate_right)[
                        :, :, y0 : y0 + CROP[0], x0 : x0 + CROP[1]
                    ],
                )
            )
        images = torch.cat(normalized, dim=0).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.device.type == "cuda" else None,
            enabled=self.device.type == "cuda",
        ):
            prefix, _positions = encode_frozen_prefix(
                self.backbone, images, depth=1
            )
        # Quantize prefix tokens to FP16 before evidence reduction.
        return prefix.to(torch.float16)

    def rows(
        self,
        teacher: dict[str, np.ndarray],
        indices: list[int],
        records: list[dict[str, Any]],
        dataset_root: Path,
    ) -> list[dict[int, np.ndarray]]:
        if not 1 <= len(indices) <= 4 or len(records) != len(indices):
            raise ValueError("fixed evidence batches must contain one to four rows")
        prefixes = [
            self._prefix(teacher, index, record, dataset_root)
            for index, record in zip(indices, records, strict=True)
        ]
        real_rows = len(prefixes)
        # Reduce evidence in fixed batches of four rows. CUDA
        # kernels can differ by one float32 ulp when the batch shape changes;
        # pad only the final partial batch and discard its repeated outputs.
        prefixes.extend(prefixes[-1:] * (4 - real_rows))
        evidence_batch = 4
        current = torch.cat(prefixes, dim=0)
        positions = self.base_positions.expand(current.shape[0], -1, -1)
        result: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            result[1] = (
                self.evidence(
                    current.reshape(evidence_batch, 3, 2, 968, 1024)
                )
                .float()
                .cpu()
                .numpy()[:real_rows]
            )
        for depth in range(2, 9):
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.device.type == "cuda" else None,
                enabled=self.device.type == "cuda",
            ):
                current = self.backbone.enc_blocks[depth - 1](current, positions)
            if depth == 8:
                with torch.inference_mode():
                    result[8] = (
                        self.evidence(
                            current.reshape(evidence_batch, 3, 2, 968, 1024)
                        )
                        .float()
                        .cpu()
                        .numpy()[:real_rows]
                    )
        for depth, value in result.items():
            if value.shape != (real_rows, 3, FEATURE_DIM) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid depth-{depth} evidence")
        return [
            {depth: value[row] for depth, value in result.items()}
            for row in range(real_rows)
        ]

    def row(
        self,
        teacher: dict[str, np.ndarray],
        index: int,
        record: dict[str, Any],
        dataset_root: Path,
    ) -> dict[int, np.ndarray]:
        return self.rows(teacher, [index], [record], dataset_root)[0]
