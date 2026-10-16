from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidenceProbe(nn.Module):
    """Score non-identity treatments from per-candidate evidence."""

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


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
