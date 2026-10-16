from __future__ import annotations
import argparse,random,os
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from .model import EvidenceProbe,trainable_parameter_count

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def positive_weight(target: np.ndarray) -> torch.Tensor:
    positive = target.sum(axis=0).astype(np.float64)
    negative = len(target) - positive
    return torch.as_tensor(
        np.clip(negative / np.maximum(positive, 1.0), 0.25, 12.0),
        dtype=torch.float32,
    )


def atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
def fit_weights(arrays: dict[str, np.ndarray], train_indices: np.ndarray) -> dict[str, Any]:
    gain = arrays["gain"][train_indices]
    positive = np.maximum(gain.max(axis=1), 0.0)
    nonzero = positive[positive > 0]
    median_positive = float(np.median(nonzero)) if len(nonzero) else 1.0
    contribution = np.ones(len(arrays["gain"]), dtype=np.float32)
    contribution[train_indices] = np.clip(
        1.0 + positive / max(median_positive, 1e-6), 1.0, 12.0
    )
    return {
        "contribution": contribution,
        "benefit": positive_weight(arrays["benefit"][train_indices]),
        "harm": positive_weight(arrays["harm"][train_indices]),
        "catastrophe": positive_weight(arrays["catastrophe"][train_indices]),
    }


def loss_batch(
    output: torch.Tensor,
    batch_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    weights: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    utility = torch.as_tensor(arrays["utility"][batch_indices], device=device)
    benefit = torch.as_tensor(arrays["benefit"][batch_indices].astype(np.float32), device=device)
    harm = torch.as_tensor(arrays["harm"][batch_indices].astype(np.float32), device=device)
    catastrophe = torch.as_tensor(
        arrays["catastrophe"][batch_indices].astype(np.float32), device=device
    )
    sample_weight = torch.as_tensor(weights["contribution"][batch_indices], device=device)[:, None]
    mean, q10 = output[:, :, 0], output[:, :, 1]
    magnitude_weight = sample_weight * (1.0 + 4.0 * utility.abs())
    utility_loss = (
        F.smooth_l1_loss(mean, utility, reduction="none", beta=0.02) * magnitude_weight
    ).mean()
    residual = utility - q10
    q10_loss = (
        torch.maximum(0.1 * residual, -0.9 * residual) * magnitude_weight
    ).mean()
    benefit_loss = (
        F.binary_cross_entropy_with_logits(
            output[:, :, 2], benefit, pos_weight=weights["benefit"].to(device), reduction="none"
        )
        * sample_weight
    ).mean()
    harm_loss = (
        F.binary_cross_entropy_with_logits(
            output[:, :, 3], harm, pos_weight=weights["harm"].to(device), reduction="none"
        )
        * sample_weight
    ).mean()
    catastrophe_loss = (
        F.binary_cross_entropy_with_logits(
            output[:, :, 4], catastrophe,
            pos_weight=weights["catastrophe"].to(device), reduction="none"
        )
        * sample_weight
    ).mean()
    difference = utility[:, 0] - utility[:, 1]
    rank_weight = difference.abs().clamp_max(1.0) * sample_weight[:, 0]
    ranking_loss = (
        F.softplus(-difference.sign() * (mean[:, 0] - mean[:, 1])) * rank_weight
    ).sum() / rank_weight.sum().clamp_min(1e-6)
    quantile_order = F.relu(q10 - mean).mean()
    pieces = {
        "utility": utility_loss,
        "q10": q10_loss,
        "benefit": benefit_loss,
        "harm": harm_loss,
        "catastrophe": catastrophe_loss,
        "ranking": ranking_loss,
        "quantile_order": quantile_order,
    }
    total = (
        2.0 * utility_loss
        + 0.75 * q10_loss
        + 0.75 * benefit_loss
        + 0.75 * harm_loss
        + catastrophe_loss
        + ranking_loss
        + 0.25 * quantile_order
    )
    return total, pieces
def target_arrays(teacher: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    gain = teacher["gain_abs_px"][:, 1:3].astype(np.float32)
    baseline = np.maximum(teacher["action_abs_px"][:, 0], 1e-5).astype(np.float32)
    utility = np.clip(gain / baseline[:, None], -1.0, 1.0).astype(np.float32)
    return {
        "gain": gain,
        "utility": utility,
        "benefit": gain > 0.001,
        "harm": gain < -0.001,
        "catastrophe": gain < -0.05,
    }


def train_fold_seed(
    *,
    feature: np.ndarray,
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, int]:
    seed_everything(seed)
    device = torch.device(args.device)
    model = EvidenceProbe(feature.shape[-1], args.width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    weights = fit_weights(arrays, train)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(args.epochs):
        model.train()
        order = train[torch.randperm(len(train), generator=generator).numpy()]
        for start in range(0, len(order), args.train_batch_size):
            batch = order[start : start + args.train_batch_size]
            x = torch.as_tensor(np.asarray(feature[batch]).copy(), device=device)
            output = model(x)
            loss, _pieces = loss_batch(output, batch, arrays, weights, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(test), 2048):
            batch = test[start : start + 2048]
            x = torch.as_tensor(np.asarray(feature[batch]).copy(), device=device)
            values.append(model(x).cpu().numpy())
    if checkpoint_path is not None:
        atomic_torch(
            checkpoint_path,
            {
                "schema_version": 1,
                "model": model.state_dict(),
                "feature_dim": feature.shape[-1],
                "width": args.width,
                "seed": seed,
                "epochs": args.epochs,
                "train_rows": len(train),
                "trainable_parameters": trainable_parameter_count(model),
            },
        )
    return np.concatenate(values).astype(np.float32), trainable_parameter_count(model)
