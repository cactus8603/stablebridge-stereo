"""CroCo-aware evidence and counterfactual utility verifier.

The expensive CroCo backbone is frozen.  All candidates share its patch
embedding and encoder block 0.  The verifier then predicts relative utility
and risk for the two non-identity Core-3 actions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceProbe(nn.Module):
    """Small frozen routing head used by both training and deployment.

    Keeping the architecture beside the deployment evidence model prevents the
    submission runtime from importing the SB3E0 training pipeline merely to
    reconstruct checkpointed heads.
    """

    def __init__(self, feature_dim: int, width: int) -> None:
        super().__init__()
        self.action = nn.Embedding(3, 8)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim * 4 + 8, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 6),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        identity = evidence[:, :1]
        context_mean = evidence.mean(dim=1, keepdim=True).expand_as(evidence)
        context_max = evidence.max(dim=1, keepdim=True).values.expand_as(evidence)
        action = self.action(torch.arange(3, device=evidence.device))[None].expand(
            len(evidence), -1, -1
        )
        value = torch.cat(
            (
                evidence,
                identity.expand_as(evidence),
                evidence - identity,
                context_mean + context_max,
                action,
            ),
            dim=-1,
        )[:, 1:]
        return self.scorer(value)


def encode_frozen_prefix(
    model: nn.Module,
    images: torch.Tensor,
    *,
    depth: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact CroCo encoder prefix without masking."""

    if depth < 0 or depth > len(model.enc_blocks):
        raise ValueError(f"invalid encoder prefix depth: {depth}")
    tokens, positions = model.patch_embed(images)
    if model.enc_pos_embed is not None:
        tokens = tokens + model.enc_pos_embed[None, ...]
    for block in model.enc_blocks[:depth]:
        tokens = block(tokens, positions)
    return tokens, positions


def continue_frozen_encoder(
    model: nn.Module,
    prefix_tokens: torch.Tensor,
    positions: torch.Tensor,
    *,
    prefix_depth: int = 1,
    return_all_blocks: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """Continue an exact CroCo encoder from a cached prefix."""

    if prefix_depth < 0 or prefix_depth > len(model.enc_blocks):
        raise ValueError(f"invalid encoder prefix depth: {prefix_depth}")
    tokens = prefix_tokens
    if return_all_blocks:
        outputs = [tokens]
        for block in model.enc_blocks[prefix_depth:]:
            tokens = block(tokens, positions)
            outputs.append(tokens)
        outputs[-1] = model.enc_norm(outputs[-1])
        return outputs
    for block in model.enc_blocks[prefix_depth:]:
        tokens = block(tokens, positions)
    return model.enc_norm(tokens)


def binocular_forward_from_prefix(
    model: nn.Module,
    prefix_tokens: torch.Tensor,
    positions: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    prefix_depth: int = 1,
) -> torch.Tensor:
    """Reproduce the official binocular/DPT path from a shared prefix."""

    if prefix_tokens.shape[0] % 2:
        raise ValueError("prefix batch must contain concatenated left/right views")
    encoded = continue_frozen_encoder(
        model,
        prefix_tokens,
        positions,
        prefix_depth=prefix_depth,
        return_all_blocks=True,
    )
    assert isinstance(encoded, list)
    chunks = [value.chunk(2, dim=0) for value in encoded]
    left_features = [value[0] for value in chunks]
    right_feature = chunks[-1][1]
    left_positions, right_positions = positions.chunk(2, dim=0)
    decoded = model._decoder(
        left_features[-1],
        left_positions,
        None,
        right_feature,
        right_positions,
        return_all_blocks=True,
    )
    return model.head(
        left_features + decoded,
        {"height": image_height, "width": image_width},
    )


def _summary(value: torch.Tensor) -> torch.Tensor:
    """Robustly pool the last dimension into differentiable statistics."""

    value = value.float()
    return torch.stack(
        (
            value.mean(dim=-1),
            value.std(dim=-1, unbiased=False),
            torch.quantile(value, 0.10, dim=-1),
            torch.quantile(value, 0.50, dim=-1),
            torch.quantile(value, 0.90, dim=-1),
        ),
        dim=-1,
    )


class StereoEvidence(nn.Module):
    """Convert paired CroCo prefix tokens into candidate matchability cues."""

    def __init__(
        self,
        *,
        token_dim: int = 1024,
        projection_dim: int = 96,
        grid_height: int = 22,
        grid_width: int = 44,
        max_disparity_tokens: int = 32,
    ) -> None:
        super().__init__()
        if max_disparity_tokens >= grid_width:
            raise ValueError("max_disparity_tokens must be below grid width")
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_disparity_tokens = max_disparity_tokens
        self.norm = nn.LayerNorm(token_dim)
        self.projector = nn.Linear(token_dim, projection_dim, bias=False)
        self.patch_mlp = nn.Sequential(
            nn.Linear(projection_dim * 6, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.patch_attention = nn.Linear(64, 1)
        self.output_dim = 35 + 64 * 3

    def _correlation_features(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        batch, actions, height, width, channels = left.shape
        correlations: list[torch.Tensor] = []
        for disparity in range(self.max_disparity_tokens + 1):
            value = left.new_full((batch, actions, height, width), -1e4)
            if disparity == 0:
                value = (left * right).sum(dim=-1)
            else:
                value[:, :, :, disparity:] = (
                    left[:, :, :, disparity:] * right[:, :, :, : width - disparity]
                ).sum(dim=-1)
            correlations.append(value)
        volume = torch.stack(correlations, dim=-1)
        valid_count = torch.arange(width, device=left.device).clamp_max(
            self.max_disparity_tokens
        ) + 1
        valid = (
            torch.arange(self.max_disparity_tokens + 1, device=left.device)[None, :]
            < valid_count[:, None]
        )
        probability = torch.softmax(volume.float(), dim=-1).to(volume.dtype)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / valid_count.to(entropy.dtype).log().clamp_min(1.0)[None, None, None, :]
        top = volume.topk(2, dim=-1).values
        best = top[..., 0]
        margin = top[..., 0] - top[..., 1]
        support = valid_count >= 5
        best = best[:, :, :, support].flatten(2)
        margin = margin[:, :, :, support].flatten(2)
        entropy = entropy[:, :, :, support].flatten(2)
        disparity_axis = torch.arange(
            self.max_disparity_tokens + 1, device=left.device, dtype=probability.dtype
        )
        expected_disparity = (probability * disparity_axis).sum(dim=-1)
        expected_disparity = expected_disparity[:, :, :, support].flatten(2)
        return torch.cat(
            (
                _summary(best),
                _summary(margin),
                _summary(entropy),
                _summary(expected_disparity / max(1, self.max_disparity_tokens)),
            ),
            dim=-1,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5 or tokens.shape[2] != 2:
            raise ValueError("tokens must have shape BxKx2xNxC")
        batch, actions, _views, patches, _channels = tokens.shape
        if patches != self.grid_height * self.grid_width:
            raise ValueError(f"unexpected token count: {patches}")
        projected = F.normalize(self.projector(self.norm(tokens.float())), dim=-1)
        grid = projected.view(
            batch,
            actions,
            2,
            self.grid_height,
            self.grid_width,
            projected.shape[-1],
        )
        left, right = grid[:, :, 0], grid[:, :, 1]
        correlation = self._correlation_features(left, right)

        identity_left = left[:, :1]
        identity_right = right[:, :1]
        patch_input = torch.cat(
            (
                left,
                right,
                (left - right).abs(),
                left - identity_left,
                right - identity_right,
                left * right,
            ),
            dim=-1,
        )
        patch_latent = self.patch_mlp(patch_input).flatten(2, 3)
        patch_mean = patch_latent.mean(dim=2)
        patch_max = patch_latent.max(dim=2).values
        patch_weight = torch.softmax(self.patch_attention(patch_latent), dim=2)
        patch_attention = (patch_latent * patch_weight).sum(dim=2)
        learned_patch_evidence = torch.cat(
            (patch_mean, patch_max, patch_attention), dim=-1
        )

        identity = projected[:, :1]
        cosine_drift = 1.0 - (projected * identity).sum(dim=-1)
        drift = _summary(cosine_drift.flatten(2))

        horizontal_tv = (grid[:, :, :, :, 1:] - grid[:, :, :, :, :-1]).square().mean(dim=-1)
        vertical_tv = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).square().mean(dim=-1)
        tv = torch.stack(
            (
                horizontal_tv.mean(dim=(2, 3, 4)),
                vertical_tv.mean(dim=(2, 3, 4)),
            ),
            dim=-1,
        )
        identity_tv = tv[:, :1]
        tv_features = torch.cat((tv, tv - identity_tv), dim=-1)

        left_norm = tokens[:, :, 0].float().norm(dim=-1)
        right_norm = tokens[:, :, 1].float().norm(dim=-1)
        symmetry = _summary(
            ((left_norm - right_norm).abs() / (left_norm + right_norm + 1e-6))
        )

        left_mean = projected[:, :, 0].mean(dim=2)
        right_mean = projected[:, :, 1].mean(dim=2)
        global_cosine = (left_mean * right_mean).sum(dim=-1, keepdim=True)
        evidence = torch.cat(
            (
                correlation,
                drift,
                tv_features,
                symmetry,
                global_cosine,
                learned_patch_evidence,
            ),
            dim=-1,
        )
        if evidence.shape != (batch, actions, self.output_dim):
            raise RuntimeError(f"evidence shape contract failed: {evidence.shape}")
        return evidence


class StableBridgeVerifier(nn.Module):
    """Shared action scorer for relative utility and intervention risk."""

    OUTPUT_NAMES = (
        "mean_utility",
        "q10_utility",
        "benefit_logit",
        "harm_logit",
        "catastrophe_logit",
        "support_logit",
    )

    def __init__(
        self,
        *,
        token_dim: int = 1024,
        projection_dim: int = 96,
        width: int = 256,
        action_embedding_dim: int = 16,
        weak_features: int = 0,
    ) -> None:
        super().__init__()
        self.evidence = StereoEvidence(
            token_dim=token_dim, projection_dim=projection_dim
        )
        self.action_embedding = nn.Embedding(3, action_embedding_dim)
        self.weak_features = weak_features
        weak_dim = 0
        if weak_features:
            weak_dim = 64
            self.weak_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(weak_features),
                nn.Linear(weak_features, 128),
                nn.SiLU(),
                nn.Linear(128, weak_dim),
                nn.SiLU(),
            )
        else:
            self.weak_encoder = None
        evidence_dim = self.evidence.output_dim
        scorer_dim = evidence_dim * 4 + action_embedding_dim + weak_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, len(self.OUTPUT_NAMES)),
        )
        with torch.no_grad():
            self.scorer[-1].bias[0:2].fill_(-0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        weak: torch.Tensor | None = None,
    ) -> torch.Tensor:
        evidence = self.evidence(tokens)
        identity = evidence[:, :1]
        context_mean = evidence.mean(dim=1, keepdim=True).expand_as(evidence)
        context_max = evidence.max(dim=1, keepdim=True).values.expand_as(evidence)
        action_ids = torch.arange(3, device=tokens.device)
        action = self.action_embedding(action_ids)[None].expand(tokens.shape[0], -1, -1)
        parts = (evidence, identity.expand_as(evidence), evidence - identity,
                 context_mean + context_max, action)
        feature = torch.cat(parts, dim=-1)[:, 1:]
        if self.weak_encoder is not None:
            if weak is None or weak.shape[-1] != self.weak_features:
                raise ValueError("weak feature contract mismatch")
            encoded = self.weak_encoder(weak.float())[:, None].expand(-1, 2, -1)
            feature = torch.cat((feature, encoded), dim=-1)
        output = self.scorer(feature)
        if output.shape != (tokens.shape[0], 2, len(self.OUTPUT_NAMES)):
            raise RuntimeError(f"verifier output contract failed: {output.shape}")
        return output


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def verifier_contract(module: StableBridgeVerifier) -> dict[str, Any]:
    return {
        "actions": ["P1", "P2"],
        "outputs": list(module.OUTPUT_NAMES),
        "trainable_parameters": trainable_parameter_count(module),
        "projection_dim": module.evidence.projector.out_features,
        "weak_features": module.weak_features,
        "prefix_depth": 1,
    }
