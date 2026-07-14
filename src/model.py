from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torchvision import models


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BeMambaConfig:
    num_classes: int = 64
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    patch_grid: int = 6
    dropout: float = 0.2
    gps_hidden_dim: int = 96
    gps_input_dim: int = 2
    pretrained_backbones: bool = True
    backbone_stage: int = 2
    temporal_layers: int = 1
    fusion_layers: int = 1
    freeze_image_stem: bool = False
    temporal_order: str = "forward"
    spatial_scan: str = "vertical"
    missing_enabled: bool = False

    # ── DMAF ablation switches ──
    # All default True — use --no-* flags to disable individual components.
    use_mask_weighted_pool: bool = True
    use_mask_embed: bool = True
    use_cross_attn: bool = True
    use_reliability: bool = True
    model_variant: str = "bemamba"
    clean_cross_attn: bool = False
    spatial_mixer_layers: int = 0
    use_order_gate: bool = False
    modal_order_count: int = 3
    use_attn_head: bool = False
    use_branch_ensemble: bool = False
    use_beam_query_head: bool = False
    use_multiscale_backbone: bool = False
    use_ordinal_head: bool = False
    use_temporal_attn_pool: bool = False
    use_beam_neighbor_head: bool = False
    use_candidate_reranker: bool = False
    use_bounded_candidate_reranker: bool = False
    use_modality_feature_dropout: bool = False
    modality_feature_dropout: float = 0.0
    candidate_topk: int = 7
    candidate_delta_bound: float = 0.20
    candidate_embed_dropout: float = 0.0
    return_aux_logits: bool = False
    aux_loss_weight: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Shared building blocks (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

class ChannelEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(sequence)
        sequence = self.conv1d(sequence.transpose(1, 2)).transpose(1, 2)
        return sequence


class TFMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.ssm_in = nn.Linear(d_model, d_model)
        self.ssm_conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.ssm_act = nn.SiLU()
        self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.gate_in = nn.Linear(d_model, d_model)
        self.gate_act = nn.SiLU()
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        ssm_branch = self.ssm_in(sequence)
        ssm_branch = self.ssm_conv(ssm_branch.transpose(1, 2)).transpose(1, 2)
        ssm_branch = self.ssm_act(ssm_branch)
        ssm_branch = self.ssm(ssm_branch)

        gate_branch = self.gate_act(self.gate_in(sequence))
        fused = self.out_proj(ssm_branch * gate_branch)
        return self.dropout(fused)


class MBMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.channel_encoding = ChannelEncoding(d_model)
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.weight_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.channel_encoding(sequence)
        forward = self.forward_mamba(normalized)
        backward = torch.flip(
            self.backward_mamba(torch.flip(normalized, dims=[1])),
            dims=[1],
        )
        weight = self.weight_proj(normalized)
        return self.dropout(weight * forward + weight * backward)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Spatial encoders
# ═══════════════════════════════════════════════════════════════════════════

class ModalityBackbone(nn.Module):
    """Per-frame spatial feature extraction via ResNet.

    Input:  [B, C, H, W]
    Output: [B, d_model, patch_grid, patch_grid]
    """

    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        d_model: int,
        patch_grid: int,
        pretrained: bool,
        backbone_stage: int = 2,
    ):
        super().__init__()
        if backbone_stage not in {2, 3, 4}:
            raise ValueError(f"Unsupported backbone_stage: {backbone_stage}")
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        out_channels = {2: 128, 3: 256, 4: 512}[backbone_stage]

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if pretrained:
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = new_conv

        feature_layers = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        ]
        if backbone_stage >= 3:
            feature_layers.append(backbone.layer3)
        if backbone_stage >= 4:
            feature_layers.append(backbone.layer4)
        self.features = nn.Sequential(*feature_layers)
        self.pool = nn.AdaptiveAvgPool2d((patch_grid, patch_grid))
        self.project = nn.Identity() if out_channels == d_model else nn.Conv2d(out_channels, d_model, kernel_size=1, bias=False)

    def freeze_stem(self) -> None:
        for module in (self.features[0], self.features[1]):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.features[1].eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = self.project(feat)
        return feat  # [B, d_model, patch_grid, patch_grid]


class MultiScaleModalityBackbone(nn.Module):
    """ResNet layer2+layer3 feature fusion for clean-data variants.

    Layer2 keeps finer spatial cues, while layer3 contributes stronger semantic
    context. Both are pooled to the same token grid before a 1x1 projection.
    """

    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        d_model: int,
        patch_grid: int,
        pretrained: bool,
    ):
        super().__init__()
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if pretrained:
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = new_conv

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.pool = nn.AdaptiveAvgPool2d((patch_grid, patch_grid))
        self.project = nn.Sequential(
            nn.Conv2d(128 + 256, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

    def freeze_stem(self) -> None:
        for module in (self.stem[0], self.stem[1]):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.stem[1].eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        feat = torch.cat([self.pool(layer2), self.pool(layer3)], dim=1)
        return self.project(feat)


class GPSProjection(nn.Module):
    """Pure GPS feature projection (no mask handling — see MaskEncoder).

    Input:  [B, 2, gps_input_dim]
    Output: [B, 2, d_model]
    """

    def __init__(self, input_dim: int, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, gps: torch.Tensor) -> torch.Tensor:
        return self.net(gps)  # [B, 2, d_model]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Mask Encoder (standalone — NEW)
# ═══════════════════════════════════════════════════════════════════════════

class MaskEncoder(nn.Module):
    """Standalone mask embedding injection.

    Learns two d_model-dimensional vectors:
      - index 0 → "missing" signature
      - index 1 → "present" signature

    Injects these signatures into per-frame features before temporal processing,
    so the SSM can distinguish "real zero" from "missing frame".

    Works for both:
      - spatial features  [B, seq_len, d_model, H, W]
      - vector features   [B, seq_len, d_model]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mask_embed = nn.Embedding(2, d_model)
        nn.init.normal_(self.mask_embed.weight, std=0.02)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inject mask embedding into features.

        Args:
            features: [B, seq_len, ..., d_model]
            mask:     [B, seq_len]  (0 = missing, 1 = present)

        Returns:
            mask-aware features, same shape as input.
        """
        mask_emb = self.mask_embed(mask.long())  # [B, seq_len, d_model]

        # Insert singleton dims between seq_len and d_model to match features
        n_extra = features.dim() - mask_emb.dim()
        for _ in range(n_extra):
            mask_emb = mask_emb.unsqueeze(-1)  # insert before last dim

        mask_emb = mask_emb.expand_as(features)
        return features + mask_emb


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Temporal Processor (extracted from TimeSequenceBranch — NEW)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalProcessor(nn.Module):
    """TFMamba temporal processing + mask-weighted aggregation.

    Takes per-frame spatial features, processes them through TFMamba
    to model temporal dependencies, then aggregates via mask-weighted mean.

    Input:  [B, seq_len, d_model, pooled_h, pooled_w]
    Output: [B, spatial_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        temporal_layers: int,
        temporal_order: str,
        spatial_scan: str,
        use_attn_pool: bool = False,
    ):
        super().__init__()
        if temporal_order not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported temporal_order: {temporal_order}")
        if spatial_scan not in {"row", "vertical"}:
            raise ValueError(f"Unsupported spatial_scan: {spatial_scan}")

        self.temporal_order = temporal_order
        self.spatial_scan = spatial_scan
        self.use_attn_pool = use_attn_pool

        self.temporal_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "channel_encoding": ChannelEncoding(d_model),
                        "tf_mamba": TFMamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            dropout=dropout,
                        ),
                    }
                )
                for _ in range(temporal_layers)
            ]
        )
        if self.use_attn_pool:
            self.pool_norm = nn.LayerNorm(d_model)
            self.pool_score = nn.Linear(d_model, 1)
        else:
            self.pool_norm = None
            self.pool_score = None

    def _flatten_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dims according to scan mode."""
        if self.spatial_scan == "vertical":
            x = x.transpose(-1, -2)
        return x.flatten(start_dim=-2)  # [B, seq, d_model, spatial]

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Process temporal dimension and aggregate.

        Args:
            x:    [B, seq_len, d_model, pooled_h, pooled_w]
            mask: [B, seq_len] or None — used for weighted aggregation

        Returns:
            tokens: [B, spatial_len, d_model]
        """
        B, seq_len, d_model, pooled_h, pooled_w = x.shape
        spatial_len = pooled_h * pooled_w

        # Flatten spatial dimensions
        x = self._flatten_spatial(x)  # [B, seq_len, d_model, spatial_len]

        # Reverse temporal order if configured
        if self.temporal_order == "reverse":
            x = torch.flip(x, dims=[1])

        # Reshape for TFMamba: [B, seq_len * spatial_len, d_model]
        time_sequence = x.permute(0, 1, 3, 2).reshape(B, seq_len * spatial_len, d_model)

        # TFMamba processing
        fused = time_sequence
        for block in self.temporal_blocks:
            channel_encoded = block["channel_encoding"](fused)
            fused = fused + block["tf_mamba"](channel_encoded)

        # Reshape back to per-frame: [B, seq_len, spatial_len, d_model]
        per_time = fused.reshape(B, seq_len, spatial_len, d_model)

        # Temporal aggregation
        if self.use_attn_pool:
            assert self.pool_norm is not None and self.pool_score is not None
            pool_scores = self.pool_score(self.pool_norm(per_time)).squeeze(-1)  # [B, seq_len, spatial_len]
            if mask is not None:
                agg_mask = torch.flip(mask, dims=[1]) if self.temporal_order == "reverse" else mask
                valid = agg_mask.unsqueeze(-1).bool()
                pool_scores = pool_scores.masked_fill(~valid, -1e4)
            pool_weights = torch.softmax(pool_scores, dim=1).unsqueeze(-1)
            aggregated = (per_time * pool_weights).sum(dim=1) * seq_len
        elif mask is not None:
            agg_mask = torch.flip(mask, dims=[1]) if self.temporal_order == "reverse" else mask
            weights = agg_mask.unsqueeze(-1).unsqueeze(-1)  # [B, seq_len, 1, 1]
            valid_count = weights.sum(dim=1).clamp(min=1)
            # Preserve the original BeMamba sum scale when all frames are present.
            aggregated = (per_time * weights).sum(dim=1) * (seq_len / valid_count)
        else:
            aggregated = per_time.sum(dim=1)

        return aggregated  # [B, spatial_len, d_model]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Reliability Estimator (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════

class ReliabilityEstimator(nn.Module):
    """Estimates per-modality reliability from mask ratio and feature statistics.

    For each modality, combines:
      - Feature statistics (mean + std over tokens)
      - Mask availability ratio

    Outputs a d_model-dimensional reliability vector in (0, 1]
    that scales modality tokens before fusion.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        inner_dim = max(d_model // 4, 8)

        self.feat_proj = nn.Sequential(
            nn.Linear(d_model * 2, inner_dim),
            nn.GELU(),
        )
        self.mask_proj = nn.Linear(1, inner_dim)

        self.mlp = nn.Sequential(
            nn.Linear(inner_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Estimate reliability for a single modality.

        Args:
            tokens: [B, N, d_model] or [B, seq_len, d_model]
            mask:   [B, seq_len]  (binary, 1=present, 0=missing)

        Returns:
            reliability: [B, d_model] in (0, 1]
        """
        feat_mean = tokens.mean(dim=1)  # [B, d_model]
        feat_std = tokens.std(dim=1, unbiased=False)  # [B, d_model]
        feat_stats = torch.cat([feat_mean, feat_std], dim=-1)  # [B, 2*d_model]

        mask_ratio = mask.mean(dim=1, keepdim=True)  # [B, 1]

        feat_encoded = self.feat_proj(feat_stats)  # [B, inner_dim]
        mask_encoded = self.mask_proj(mask_ratio)  # [B, inner_dim]

        combined = torch.cat([feat_encoded, mask_encoded], dim=-1)
        return self.mlp(combined)  # [B, d_model] in (0, 1]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Cross-Modal Fusion (direct modality-pair attention — NEW)
# ═══════════════════════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    """Single-head cross-attention block with FFN.

    Q attends to KV; residual connection + FFN.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.attn(q, kv, kv)
        out = query + self.dropout(attn_out)
        out = out + self.ffn(self.norm_out(out))
        return out


class CrossModalFusion(nn.Module):
    """Direct cross-modal attention between image, radar, and lidar tokens.

    Each modality serves as query, attending to the other two modalities'
    tokens as key-value. This enables direct cross-modal compensation:
      - Missing image  → borrow from radar + lidar
      - Missing lidar  → borrow from image + radar
      - Missing radar  → borrow from image + lidar

    GPS is handled separately through the MBMamba sequence bookends.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.img_cross = CrossModalAttention(d_model, nhead, dropout)
        self.radar_cross = CrossModalAttention(d_model, nhead, dropout)
        self.lidar_cross = CrossModalAttention(d_model, nhead, dropout)

    def forward(
        self,
        img: torch.Tensor,
        radar: torch.Tensor,
        lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-modal fusion across spatial modalities.

        Args:
            img:   [B, N, d]
            radar: [B, N, d]
            lidar: [B, N, d]

        Returns:
            Enhanced tokens, same shapes.
        """
        # Image attends to radar + lidar
        kv_for_img = torch.cat([radar, lidar], dim=1)  # [B, 2N, d]
        img_enh = self.img_cross(img, kv_for_img)

        # Radar attends to image + lidar
        kv_for_radar = torch.cat([img, lidar], dim=1)  # [B, 2N, d]
        radar_enh = self.radar_cross(radar, kv_for_radar)

        # Lidar attends to image + radar
        kv_for_lidar = torch.cat([img, radar], dim=1)  # [B, 2N, d]
        lidar_enh = self.lidar_cross(lidar, kv_for_lidar)

        return img_enh, radar_enh, lidar_enh


class SpatialTokenMixerBlock(nn.Module):
    """Lightweight self-attention over spatial tokens inside one modality."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm_attn(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        tokens = tokens + self.dropout(self.ffn(self.norm_ffn(tokens)))
        return tokens


class SpatialTokenMixer(nn.Module):
    """Stacked spatial token mixing used by the clean_plus variant."""

    def __init__(self, d_model: int, num_layers: int, dropout: float = 0.1, nhead: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatialTokenMixerBlock(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class OrderFusionGate(nn.Module):
    """Sample-adaptive fusion for the three BeMamba modal orderings."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, encoded_stack: torch.Tensor) -> torch.Tensor:
        summary = torch.cat(
            [
                encoded_stack.mean(dim=2),
                encoded_stack.std(dim=2, unbiased=False),
            ],
            dim=-1,
        )  # [B, 3, 2d]
        weights = torch.softmax(self.gate(summary).squeeze(-1), dim=1)  # [B, 3]
        fused = (encoded_stack * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return fused * encoded_stack.size(1)


class AttentivePredictionHead(nn.Module):
    """Flattened-token head augmented with attention/mean/max pooling."""

    def __init__(self, d_model: int, spatial_len: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        input_dim = d_model * spatial_len + d_model * 3
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.pool_score(self.pool_norm(tokens)), dim=1)
        attn_pool = (tokens * weights).sum(dim=1)
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.amax(dim=1)
        features = torch.cat(
            [
                tokens.reshape(tokens.size(0), -1),
                attn_pool,
                mean_pool,
                max_pool,
            ],
            dim=1,
        )
        return self.net(features)


class TokenClassifierHead(nn.Module):
    """Compact classifier for one modality/token stream."""

    def __init__(self, d_model: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def summarize(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.pool_score(self.pool_norm(tokens)), dim=1)
        attn_pool = (tokens * weights).sum(dim=1)
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.amax(dim=1)
        return torch.cat([attn_pool, mean_pool, max_pool], dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(self.summarize(tokens))


class CleanBranchEnsembleHead(nn.Module):
    """Blend fused logits with auxiliary modality logits for clean-data training."""

    def __init__(self, d_model: int, num_classes: int, dropout: float):
        super().__init__()
        self.img_head = TokenClassifierHead(d_model, num_classes, dropout)
        self.radar_head = TokenClassifierHead(d_model, num_classes, dropout)
        self.lidar_head = TokenClassifierHead(d_model, num_classes, dropout)
        self.gps_head = TokenClassifierHead(d_model, num_classes, dropout)
        self.gate_norm = nn.LayerNorm(d_model * 5)
        self.gate_hidden = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_out = nn.Linear(d_model, 5)
        nn.init.zeros_(self.gate_out.weight)
        with torch.no_grad():
            self.gate_out.bias.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0, -0.5]))

    def forward(
        self,
        fused_logits: torch.Tensor,
        fused_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        radar_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        gps_tokens: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        branch_logits = torch.stack(
            [
                fused_logits,
                self.img_head(img_tokens),
                self.radar_head(radar_tokens),
                self.lidar_head(lidar_tokens),
                self.gps_head(gps_tokens),
            ],
            dim=1,
        )  # [B, 5, num_classes]
        gate_context = torch.cat(
            [
                fused_tokens.mean(dim=1),
                img_tokens.mean(dim=1),
                radar_tokens.mean(dim=1),
                lidar_tokens.mean(dim=1),
                gps_tokens.mean(dim=1),
            ],
            dim=1,
        )
        gate_logits = self.gate_out(self.gate_hidden(self.gate_norm(gate_context)))
        weights = torch.softmax(gate_logits, dim=1)
        ensembled = (branch_logits * weights.unsqueeze(-1)).sum(dim=1)
        if return_aux:
            return ensembled, branch_logits
        return ensembled


# ═══════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════

class BeamQueryRefinementHead(nn.Module):
    """Class-query refinement for beam-specific evidence gathering."""

    def __init__(self, d_model: int, num_classes: int, dropout: float, nhead: int = 4):
        super().__init__()
        self.beam_queries = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.normal_(self.beam_queries, std=0.02)

        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, base_logits: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = fused_tokens.size(0)
        queries = self.query_norm(self.beam_queries).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.token_norm(fused_tokens)
        attn_out, _ = self.attn(queries, tokens, tokens, need_weights=False)
        queries = queries + self.dropout(attn_out)
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        refine_logits = self.score(queries).squeeze(-1)
        return base_logits + self.refine_scale * refine_logits


class BeamOrdinalPriorHead(nn.Module):
    """Continuous beam-index prior added to classification logits."""

    def __init__(self, d_model: int, num_classes: int, dropout: float):
        super().__init__()
        self.num_classes = num_classes
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 2),
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.15))
        self.register_buffer(
            "beam_positions",
            torch.arange(num_classes, dtype=torch.float32).view(1, num_classes),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.pool_score(self.pool_norm(fused_tokens)), dim=1)
        attn_pool = (fused_tokens * weights).sum(dim=1)
        mean_pool = fused_tokens.mean(dim=1)
        max_pool = fused_tokens.amax(dim=1)
        raw_center, raw_width = self.regressor(
            torch.cat([attn_pool, mean_pool, max_pool], dim=1)
        ).chunk(2, dim=1)
        center = torch.sigmoid(raw_center) * (self.num_classes - 1)
        width = nn.functional.softplus(raw_width) + 1.5
        prior = -((self.beam_positions - center) ** 2) / (2.0 * width.square())
        prior = prior - prior.mean(dim=1, keepdim=True)
        return logits + self.prior_scale * prior


class BeamNeighborhoodRefinementHead(nn.Module):
    """Local logit refinement over adjacent beam indices."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float,
        kernel_size: int = 5,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2

        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.kernel_proj = nn.Linear(d_model, kernel_size)
        self.gate = nn.Linear(d_model, 1)
        self.logit_norm = nn.LayerNorm(num_classes)
        self.refine_scale = nn.Parameter(torch.tensor(0.10))

        nn.init.zeros_(self.kernel_proj.weight)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, logits: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.pool_score(self.pool_norm(fused_tokens)), dim=1)
        attn_pool = (fused_tokens * weights).sum(dim=1)
        mean_pool = fused_tokens.mean(dim=1)
        max_pool = fused_tokens.amax(dim=1)
        context = self.context_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=1))

        kernel = torch.softmax(self.kernel_proj(context), dim=-1)
        gate = torch.sigmoid(self.gate(context))

        normalized_logits = self.logit_norm(logits).unsqueeze(1)
        padded = F.pad(normalized_logits, (self.radius, self.radius), mode="replicate")
        windows = padded.unfold(dimension=2, size=self.kernel_size, step=1)
        local_logits = (windows * kernel.view(logits.size(0), 1, 1, -1)).sum(dim=-1).squeeze(1)
        return logits + self.refine_scale * gate * local_logits


class CandidateRerankerHead(nn.Module):
    """Rerank the current Top-K beam candidates with token context and beam position."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float,
        topk: int = 7,
        bounded_delta: bool = False,
        delta_bound: float = 0.20,
        embed_dropout: float = 0.0,
    ):
        super().__init__()
        if topk < 3:
            raise ValueError("candidate topk must be >= 3")
        if delta_bound <= 0:
            raise ValueError("candidate delta bound must be > 0")
        if embed_dropout < 0 or embed_dropout >= 1:
            raise ValueError("candidate embed dropout must be in [0, 1)")
        self.num_classes = num_classes
        self.topk = min(topk, num_classes)
        self.bounded_delta = bool(bounded_delta)
        self.delta_bound = float(delta_bound)
        self.embed_dropout = float(embed_dropout)

        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.beam_embed = nn.Embedding(num_classes, d_model)
        self.logit_norm = nn.LayerNorm(self.topk)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 4),
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.rerank_scale = nn.Parameter(torch.tensor(0.10))
        self.register_buffer(
            "beam_positions",
            torch.linspace(-1.0, 1.0, steps=num_classes).view(1, num_classes),
            persistent=False,
        )

        final_linear = self.score[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, logits: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.pool_score(self.pool_norm(fused_tokens)), dim=1)
        attn_pool = (fused_tokens * weights).sum(dim=1)
        mean_pool = fused_tokens.mean(dim=1)
        max_pool = fused_tokens.amax(dim=1)
        context = self.context_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=1))

        candidate_logits, candidate_idx = logits.topk(self.topk, dim=1)
        normalized_candidate_logits = self.logit_norm(candidate_logits).unsqueeze(-1)
        candidate_probs = torch.softmax(candidate_logits, dim=1).unsqueeze(-1)
        rank_pos = torch.linspace(
            0.0,
            1.0,
            steps=self.topk,
            device=logits.device,
            dtype=logits.dtype,
        ).view(1, self.topk, 1).expand(logits.size(0), -1, -1)
        beam_pos = self.beam_positions.to(dtype=logits.dtype).expand(logits.size(0), -1)
        candidate_pos = beam_pos.gather(1, candidate_idx).unsqueeze(-1)
        top1_pos = candidate_pos[:, :1, :]
        rel_pos = candidate_pos - top1_pos

        beam_features = self.beam_embed(candidate_idx)
        if self.embed_dropout > 0:
            beam_features = F.dropout(beam_features, p=self.embed_dropout, training=self.training)

        candidate_features = torch.cat(
            [
                beam_features,
                context.unsqueeze(1).expand(-1, self.topk, -1),
                normalized_candidate_logits,
                candidate_probs,
                rank_pos,
                rel_pos,
            ],
            dim=-1,
        )
        rerank_delta = self.score(candidate_features).squeeze(-1)
        rerank_delta = rerank_delta - rerank_delta.mean(dim=1, keepdim=True)
        if self.bounded_delta:
            return logits.scatter_add(1, candidate_idx, self.delta_bound * torch.tanh(rerank_delta))
        return logits.scatter_add(1, candidate_idx, self.rerank_scale * rerank_delta)


class BeMambaModel(nn.Module):
    """BeMamba with DMAF (Dynamic Mask Adaptive Fusion).

    Architecture pipeline (when missing_enabled=True):
      Phase 1: Spatial Encoding  → per-frame features
      Phase 2: Mask Encoding     → mask-aware features
      Phase 3: Temporal Proc.    → aggregated tokens
      Phase 4: Reliability Est.  → scaled tokens
      Phase 5: Cross-Modal Fusion → modality compensation
      Phase 6: MBMamba Fusion    → sequence fusion
      Phase 7: Classification    → beam prediction

    When missing_enabled=False, Phases 2/4/5 are skipped → baseline BeMamba.
    """

    def __init__(self, config: BeMambaConfig | None = None):
        super().__init__()
        self.config = config or BeMambaConfig()
        cfg = self.config
        self.spatial_len = cfg.patch_grid * cfg.patch_grid
        self.missing_enabled = cfg.missing_enabled
        if cfg.model_variant not in {"bemamba", "clean_plus", "clean_plus_v2", "clean_plus_v3", "clean_plus_v4", "clean_plus_v5", "clean_plus_v6", "clean_plus_v7", "clean_plus_v8", "clean_plus_v9", "clean_plus_v10", "clean_plus_v11", "clean_plus_v12", "clean_plus_v13", "clean_plus_v14", "clean_plus_v15"}:
            raise ValueError(f"Unsupported model_variant: {cfg.model_variant}")
        clean_plus = cfg.model_variant == "clean_plus"
        clean_plus_v2 = cfg.model_variant == "clean_plus_v2"
        clean_plus_v3 = cfg.model_variant == "clean_plus_v3"
        clean_plus_v4 = cfg.model_variant == "clean_plus_v4"
        clean_plus_v5 = cfg.model_variant == "clean_plus_v5"
        clean_plus_v6 = cfg.model_variant == "clean_plus_v6"
        clean_plus_v7 = cfg.model_variant == "clean_plus_v7"
        clean_plus_v8 = cfg.model_variant == "clean_plus_v8"
        clean_plus_v9 = cfg.model_variant == "clean_plus_v9"
        clean_plus_v10 = cfg.model_variant == "clean_plus_v10"
        clean_plus_v11 = cfg.model_variant == "clean_plus_v11"
        clean_plus_v12 = cfg.model_variant == "clean_plus_v12"
        clean_plus_v13 = cfg.model_variant == "clean_plus_v13"
        clean_plus_v14 = cfg.model_variant == "clean_plus_v14"
        clean_plus_v15 = cfg.model_variant == "clean_plus_v15"
        spatial_mixer_layers = cfg.spatial_mixer_layers
        if clean_plus and spatial_mixer_layers <= 0:
            spatial_mixer_layers = 1
        if spatial_mixer_layers < 0:
            raise ValueError("spatial_mixer_layers must be >= 0")
        if cfg.modal_order_count not in {1, 3}:
            raise ValueError("modal_order_count must be 1 or 3")

        # Resolve optional modules. Mask/reliability stay DMAF-only; clean_plus can use cross-attn.
        self.use_mask_weighted_pool = cfg.missing_enabled and cfg.use_mask_weighted_pool
        self.use_mask_embed = cfg.missing_enabled and cfg.use_mask_embed
        self.use_cross_attn = cfg.use_cross_attn and (cfg.missing_enabled or cfg.clean_cross_attn or clean_plus)
        self.use_reliability = cfg.missing_enabled and cfg.use_reliability
        self.use_spatial_mixer = spatial_mixer_layers > 0
        self.use_order_gate = cfg.use_order_gate or clean_plus
        self.modal_order_count = cfg.modal_order_count
        if self.use_order_gate and self.modal_order_count != 3:
            raise ValueError("OrderFusionGate requires modal_order_count=3")
        self.use_attn_head = cfg.use_attn_head or clean_plus
        self.use_branch_ensemble = cfg.use_branch_ensemble or clean_plus_v2 or clean_plus_v3 or clean_plus_v4 or clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15
        self.use_beam_query_head = cfg.use_beam_query_head or clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15
        self.use_multiscale_backbone = cfg.use_multiscale_backbone or clean_plus_v6
        self.use_ordinal_head = cfg.use_ordinal_head or clean_plus_v7
        self.use_temporal_attn_pool = cfg.use_temporal_attn_pool or clean_plus_v8
        self.use_beam_neighbor_head = cfg.use_beam_neighbor_head or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15
        self.use_candidate_reranker = cfg.use_candidate_reranker or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15
        self.use_bounded_candidate_reranker = cfg.use_bounded_candidate_reranker or clean_plus_v12
        self.use_modality_feature_dropout = cfg.use_modality_feature_dropout or clean_plus_v14 or clean_plus_v15
        self.modality_feature_dropout = float(cfg.modality_feature_dropout)
        if self.modality_feature_dropout < 0 or self.modality_feature_dropout >= 1:
            raise ValueError("modality_feature_dropout must be in [0, 1)")
        self.return_aux_logits = cfg.return_aux_logits or clean_plus_v3

        # ── Phase 1: Spatial encoders ──
        backbone_cls = MultiScaleModalityBackbone if self.use_multiscale_backbone else ModalityBackbone
        if self.use_multiscale_backbone:
            self.img_backbone = backbone_cls("resnet34", 3, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones)
            self.radar_backbone = backbone_cls("resnet18", 2, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones)
            self.lidar_backbone = backbone_cls("resnet18", 1, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones)
        else:
            self.img_backbone = backbone_cls("resnet34", 3, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones, cfg.backbone_stage)
            self.radar_backbone = backbone_cls("resnet18", 2, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones, cfg.backbone_stage)
            self.lidar_backbone = backbone_cls("resnet18", 1, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones, cfg.backbone_stage)
        if cfg.freeze_image_stem:
            self.img_backbone.freeze_stem()

        self.gps_projection = GPSProjection(cfg.gps_input_dim, cfg.d_model, cfg.gps_hidden_dim, cfg.dropout)

        # ── Phase 2: Mask encoders (standalone) ──
        self.img_mask_encoder = MaskEncoder(cfg.d_model)
        self.radar_mask_encoder = MaskEncoder(cfg.d_model)
        self.lidar_mask_encoder = MaskEncoder(cfg.d_model)
        self.gps_mask_encoder = MaskEncoder(cfg.d_model)

        # ── Phase 3: Temporal processors ──
        temporal_kwargs = dict(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
            temporal_layers=cfg.temporal_layers,
            temporal_order=cfg.temporal_order,
            spatial_scan=cfg.spatial_scan,
            use_attn_pool=self.use_temporal_attn_pool,
        )
        self.img_temporal = TemporalProcessor(**temporal_kwargs)
        self.radar_temporal = TemporalProcessor(**temporal_kwargs)
        self.lidar_temporal = TemporalProcessor(**temporal_kwargs)
        if self.use_spatial_mixer:
            self.img_token_mixer = SpatialTokenMixer(cfg.d_model, spatial_mixer_layers, cfg.dropout)
            self.radar_token_mixer = SpatialTokenMixer(cfg.d_model, spatial_mixer_layers, cfg.dropout)
            self.lidar_token_mixer = SpatialTokenMixer(cfg.d_model, spatial_mixer_layers, cfg.dropout)
        else:
            self.img_token_mixer = nn.Identity()
            self.radar_token_mixer = nn.Identity()
            self.lidar_token_mixer = nn.Identity()

        # ── Phase 4: Reliability estimators ──
        self.img_reliability = ReliabilityEstimator(cfg.d_model, cfg.dropout)
        self.radar_reliability = ReliabilityEstimator(cfg.d_model, cfg.dropout)
        self.lidar_reliability = ReliabilityEstimator(cfg.d_model, cfg.dropout)
        self.gps_reliability = ReliabilityEstimator(cfg.d_model, cfg.dropout)

        # ── Phase 5: Cross-modal fusion ──
        self.cross_modal_fusion = CrossModalFusion(cfg.d_model, dropout=cfg.dropout)

        # ── Phase 6: Alignment + MBMamba fusion ──
        self.image_align = nn.Linear(cfg.d_model, cfg.d_model)
        self.lidar_align = nn.Linear(cfg.d_model, cfg.d_model)
        self.radar_align = nn.Linear(cfg.d_model, cfg.d_model)

        self.modal_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MBMamba(
                            d_model=cfg.d_model,
                            d_state=cfg.d_state,
                            d_conv=cfg.d_conv,
                            expand=cfg.expand,
                            dropout=cfg.dropout,
                        )
                        for _ in range(self.modal_order_count)
                    ]
                )
                for _ in range(cfg.fusion_layers)
            ]
        )
        self.order_gate = OrderFusionGate(cfg.d_model, cfg.dropout) if self.use_order_gate else None

        # ── Phase 7: Classification head ──
        flattened_dim = cfg.d_model * self.spatial_len
        if self.use_attn_head:
            self.head = AttentivePredictionHead(cfg.d_model, self.spatial_len, cfg.num_classes, cfg.dropout)
        else:
            self.head = nn.Sequential(
                nn.Linear(flattened_dim, 2048),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(2048, cfg.num_classes),
            )
        self.branch_ensemble_head = (
            CleanBranchEnsembleHead(cfg.d_model, cfg.num_classes, cfg.dropout)
            if self.use_branch_ensemble
            else None
        )
        self.beam_query_head = (
            BeamQueryRefinementHead(cfg.d_model, cfg.num_classes, cfg.dropout)
            if self.use_beam_query_head
            else None
        )
        self.ordinal_head = (
            BeamOrdinalPriorHead(cfg.d_model, cfg.num_classes, cfg.dropout)
            if self.use_ordinal_head
            else None
        )
        self.beam_neighbor_head = (
            BeamNeighborhoodRefinementHead(cfg.d_model, cfg.num_classes, cfg.dropout)
            if self.use_beam_neighbor_head
            else None
        )
        self.candidate_reranker = (
            CandidateRerankerHead(
                cfg.d_model,
                cfg.num_classes,
                cfg.dropout,
                topk=cfg.candidate_topk,
                bounded_delta=self.use_bounded_candidate_reranker,
                delta_bound=cfg.candidate_delta_bound,
                embed_dropout=cfg.candidate_embed_dropout,
            )
            if self.use_candidate_reranker
            else None
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_image_stem:
            if hasattr(self.img_backbone, "features"):
                self.img_backbone.features[1].eval()
            elif hasattr(self.img_backbone, "stem"):
                self.img_backbone.stem[1].eval()
        return self

    # ── Helper: apply 2D backbone to 5-frame sequence ────────────────

    @staticmethod
    def _encode_frames(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a ModalityBackbone to a batch of frame sequences.

        Args:
            backbone: ModalityBackbone instance
            x:        [B, seq_len, C, H, W]

        Returns:
            [B, seq_len, d_model, patch_grid, patch_grid]
        """
        B, S, C, H, W = x.shape
        flat = x.reshape(B * S, C, H, W)
        encoded = backbone(flat)  # [B*S, d_model, ph, pw]
        _, d, ph, pw = encoded.shape
        return encoded.reshape(B, S, d, ph, pw)

    # ── Phase 6 helpers ──────────────────────────────────────────────

    def _build_modal_sequences(
        self,
        image_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        radar_tokens: torch.Tensor,
        gps_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Build one or three mixed-modality sequences (BeMamba paper Eq. 11-13).

        The one-order ablation keeps only Eq. 11. The default three-order path
        keeps all permutations used to handle Mamba's order sensitivity.
        """
        N = self.spatial_len
        gps_start = gps_tokens[:, 0, :].unsqueeze(1).expand(-1, N, -1)  # [B, N, d]
        gps_end = gps_tokens[:, 1, :].unsqueeze(1).expand(-1, N, -1)

        image_tokens = self.image_align(image_tokens)
        lidar_tokens = self.lidar_align(lidar_tokens)
        radar_tokens = self.radar_align(radar_tokens)

        seq_c1 = torch.stack([gps_start, image_tokens, lidar_tokens, radar_tokens, gps_end], dim=2)
        seq_c2 = torch.stack([gps_start, lidar_tokens, radar_tokens, image_tokens, gps_end], dim=2)
        seq_c3 = torch.stack([gps_start, radar_tokens, image_tokens, lidar_tokens, gps_end], dim=2)

        B, N, _, d = seq_c1.shape
        seq_c1 = seq_c1.reshape(B, N * 5, d)
        seq_c2 = seq_c2.reshape(B, N * 5, d)
        seq_c3 = seq_c3.reshape(B, N * 5, d)
        sequences = (seq_c1, seq_c2, seq_c3)
        return sequences[: self.modal_order_count]

    def _modal_fusion(
        self,
        sequences: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """MBMamba fusion → mean-reduce → sum.

        No cross-attention here — that's handled by CrossModalFusion (Phase 5).
        """
        running_sequences = list(sequences)
        for layer_blocks in self.modal_blocks:
            next_sequences = []
            for block, sequence in zip(layer_blocks, running_sequences):
                next_sequences.append(sequence + block(sequence))
            running_sequences = next_sequences

        # Mean-reduce the 5-modality dimension, then fuse the active orderings.
        encoded_sequences = []
        for sequence in running_sequences:
            seq = sequence.reshape(sequence.size(0), self.spatial_len, 5, self.config.d_model)
            encoded_sequences.append(seq.mean(dim=2))  # [B, N, d]

        if self.order_gate is not None:
            return self.order_gate(torch.stack(encoded_sequences, dim=1))
        if len(encoded_sequences) == 1:
            # Match the nominal scale of the original three-order sum so this
            # ablation changes ordering diversity/capacity, not feature scale.
            return encoded_sequences[0] * 3.0
        return encoded_sequences[0] + encoded_sequences[1] + encoded_sequences[2]

    def _apply_modality_feature_dropout(
        self,
        image_tokens: torch.Tensor,
        radar_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        gps_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-only modality-stream dropout for clean-plus regularization."""
        p = self.modality_feature_dropout
        if (not self.training) or (not self.use_modality_feature_dropout) or p <= 0:
            return image_tokens, radar_tokens, lidar_tokens, gps_tokens

        batch_size = image_tokens.size(0)
        keep = image_tokens.new_empty(batch_size, 4).bernoulli_(1.0 - p)

        # Avoid the degenerate case where a sample loses every modality stream.
        empty_rows = keep.sum(dim=1) == 0
        if torch.any(empty_rows):
            keep[empty_rows] = 0.0
            row_idx = empty_rows.nonzero(as_tuple=True)[0]
            restore_idx = torch.randint(
                0,
                4,
                (int(empty_rows.sum().item()),),
                device=keep.device,
            )
            keep[row_idx, restore_idx] = 1.0

        keep = keep / (1.0 - p)
        image_tokens = image_tokens * keep[:, 0].view(batch_size, 1, 1)
        radar_tokens = radar_tokens * keep[:, 1].view(batch_size, 1, 1)
        lidar_tokens = lidar_tokens * keep[:, 2].view(batch_size, 1, 1)
        gps_tokens = gps_tokens * keep[:, 3].view(batch_size, 1, 1)
        return image_tokens, radar_tokens, lidar_tokens, gps_tokens

    # ── Main forward ─────────────────────────────────────────────────

    def forward(
        self,
        imgs: torch.Tensor,
        radars: torch.Tensor,
        lidars: torch.Tensor,
        gps: torch.Tensor,
        img_mask: torch.Tensor | None = None,
        radar_mask: torch.Tensor | None = None,
        lidar_mask: torch.Tensor | None = None,
        gps_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Forward pass with modular 7-phase pipeline.

        Args:
            imgs:   [B, 5, 3, H, W]
            radars: [B, 5, 2, H, W]
            lidars: [B, 5, 1, H, W]
            gps:    [B, 2, 2]
            *_mask: [B, seq_len] or None  (0=missing, 1=present)
            return_features: when True, also return intermediate tokens for reranking

        Returns:
            logits [B, num_classes], or (logits, features)
        """
        B = imgs.shape[0]
        device = imgs.device

        # ── Auto-generate all-ones masks for clean input (DMAF mode) ──
        if self.missing_enabled:
            img_mask = img_mask if img_mask is not None else torch.ones(B, 5, device=device)
            radar_mask = radar_mask if radar_mask is not None else torch.ones(B, 5, device=device)
            lidar_mask = lidar_mask if lidar_mask is not None else torch.ones(B, 5, device=device)
            gps_mask = gps_mask if gps_mask is not None else torch.ones(B, 2, device=device)

        # ═══════════════════════════════════════════════════════════
        # Phase 1: Spatial Encoding
        # ═══════════════════════════════════════════════════════════
        img_spatial = self._encode_frames(self.img_backbone, imgs)      # [B, 5, d, ph, pw]
        radar_spatial = self._encode_frames(self.radar_backbone, radars)
        lidar_spatial = self._encode_frames(self.lidar_backbone, lidars)
        gps_feat = self.gps_projection(gps)                              # [B, 2, d]

        # ═══════════════════════════════════════════════════════════
        # Phase 2: Mask Encoding (standalone)
        # ═══════════════════════════════════════════════════════════
        if self.use_mask_embed:
            img_spatial = self.img_mask_encoder(img_spatial, img_mask)
            radar_spatial = self.radar_mask_encoder(radar_spatial, radar_mask)
            lidar_spatial = self.lidar_mask_encoder(lidar_spatial, lidar_mask)
            gps_feat = self.gps_mask_encoder(gps_feat, gps_mask)

        # ═══════════════════════════════════════════════════════════
        # Phase 3: Temporal Processing
        # ═══════════════════════════════════════════════════════════
        # Mask-weighted pooling is independently switchable from MaskEncoder.
        temporal_img_mask = img_mask if self.use_mask_weighted_pool else None
        temporal_radar_mask = radar_mask if self.use_mask_weighted_pool else None
        temporal_lidar_mask = lidar_mask if self.use_mask_weighted_pool else None
        img_tokens = self.img_temporal(img_spatial, mask=temporal_img_mask)
        radar_tokens = self.radar_temporal(radar_spatial, mask=temporal_radar_mask)
        lidar_tokens = self.lidar_temporal(lidar_spatial, mask=temporal_lidar_mask)
        img_tokens = self.img_token_mixer(img_tokens)
        radar_tokens = self.radar_token_mixer(radar_tokens)
        lidar_tokens = self.lidar_token_mixer(lidar_tokens)
        gps_tokens = gps_feat  # [B, 2, d] — no temporal processing for 2 GPS points

        # ═══════════════════════════════════════════════════════════
        # Phase 4: Reliability Estimation
        # ═══════════════════════════════════════════════════════════
        if self.use_reliability:
            img_tokens = img_tokens * self.img_reliability(img_tokens, img_mask).unsqueeze(1)
            radar_tokens = radar_tokens * self.radar_reliability(radar_tokens, radar_mask).unsqueeze(1)
            lidar_tokens = lidar_tokens * self.lidar_reliability(lidar_tokens, lidar_mask).unsqueeze(1)
            gps_rel = self.gps_reliability(gps_tokens, gps_mask).unsqueeze(1)
            gps_tokens = gps_tokens * gps_rel

        # ═══════════════════════════════════════════════════════════
        # Phase 5: Cross-Modal Fusion (direct modality-pair)
        # ═══════════════════════════════════════════════════════════
        if self.use_cross_attn:
            img_tokens, radar_tokens, lidar_tokens = self.cross_modal_fusion(
                img_tokens, radar_tokens, lidar_tokens
            )

        # ═══════════════════════════════════════════════════════════
        # Phase 6: Build sequences + MBMamba fusion
        # ═══════════════════════════════════════════════════════════
        img_tokens, radar_tokens, lidar_tokens, gps_tokens = self._apply_modality_feature_dropout(
            img_tokens,
            radar_tokens,
            lidar_tokens,
            gps_tokens,
        )

        sequences = self._build_modal_sequences(img_tokens, lidar_tokens, radar_tokens, gps_tokens)
        fused = self._modal_fusion(sequences)  # [B, N, d]

        # ═══════════════════════════════════════════════════════════
        # Phase 7: Classification head
        # ═══════════════════════════════════════════════════════════
        fused_logits = self.head(fused) if self.use_attn_head else self.head(fused.reshape(B, -1))
        features = {
            "fused_tokens": fused,
            "img_tokens": img_tokens,
            "radar_tokens": radar_tokens,
            "lidar_tokens": lidar_tokens,
            "gps_tokens": gps_tokens,
            "fused_logits": fused_logits,
        }
        logits = fused_logits
        if self.branch_ensemble_head is not None:
            logits = self.branch_ensemble_head(
                fused_logits,
                fused,
                img_tokens,
                radar_tokens,
                lidar_tokens,
                gps_tokens,
                return_aux=(self.return_aux_logits and self.training),
            )
            if isinstance(logits, tuple):
                main_logits, aux_logits = logits
                if self.beam_query_head is not None:
                    main_logits = self.beam_query_head(main_logits, fused)
                if self.ordinal_head is not None:
                    main_logits = self.ordinal_head(main_logits, fused)
                if self.beam_neighbor_head is not None:
                    main_logits = self.beam_neighbor_head(main_logits, fused)
                if self.candidate_reranker is not None:
                    main_logits = self.candidate_reranker(main_logits, fused)
                if return_features:
                    return (main_logits, aux_logits), features
                return main_logits, aux_logits
        if self.beam_query_head is not None:
            logits = self.beam_query_head(logits, fused)
        if self.ordinal_head is not None:
            logits = self.ordinal_head(logits, fused)
        if self.beam_neighbor_head is not None:
            logits = self.beam_neighbor_head(logits, fused)
        if self.candidate_reranker is not None:
            logits = self.candidate_reranker(logits, fused)
        if return_features:
            return logits, features
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint migration: old TimeSequenceBranch → new modular architecture
# ═══════════════════════════════════════════════════════════════════════════

def migrate_legacy_state_dict(state_dict: dict) -> dict:
    """Map old architecture keys to new modular architecture keys.

    Old (TimeSequenceBranch):          New (modular):
      image_branch.encoder.*      →    img_backbone.*
      image_branch.temporal_*     →    img_temporal.temporal_*
      image_branch.mask_embed.*   →    img_mask_encoder.mask_embed.*
      cross_attn_0/1/2.*         →    cross_modal_fusion.img/radar/lidar_cross.*
      gps_projection.mask_embed.* →    gps_mask_encoder.mask_embed.*
    """
    KEY_MAP = [
        # ── Backbones (encoder → backbone) ──
        ("image_branch.encoder.", "img_backbone."),
        ("lidar_branch.encoder.", "lidar_backbone."),
        ("radar_branch.encoder.", "radar_backbone."),
        # ── Temporal processors ──
        ("image_branch.temporal_blocks.", "img_temporal.temporal_blocks."),
        ("lidar_branch.temporal_blocks.", "lidar_temporal.temporal_blocks."),
        ("radar_branch.temporal_blocks.", "radar_temporal.temporal_blocks."),
        # ── Mask embeddings (branch-internal → standalone MaskEncoder) ──
        ("image_branch.mask_embed.", "img_mask_encoder.mask_embed."),
        ("lidar_branch.mask_embed.", "lidar_mask_encoder.mask_embed."),
        ("radar_branch.mask_embed.", "radar_mask_encoder.mask_embed."),
        ("gps_projection.mask_embed.", "gps_mask_encoder.mask_embed."),
        # ── Cross-attention (top-level → nested in CrossModalFusion) ──
        ("cross_attn_0.", "cross_modal_fusion.img_cross."),
        ("cross_attn_1.", "cross_modal_fusion.radar_cross."),
        ("cross_attn_2.", "cross_modal_fusion.lidar_cross."),
        # ── GPS projection net (unchanged path) ──
        # "gps_projection.net." stays "gps_projection.net."
        # ── Unchanged keys (pass through) ──
        # image_align, lidar_align, radar_align, modal_blocks, head,
        # img_reliability, radar_reliability, lidar_reliability, gps_reliability
    ]

    new_state = {}
    for old_key, value in state_dict.items():
        new_key = old_key
        for old_prefix, new_prefix in KEY_MAP:
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                break
        new_state[new_key] = value

    return new_state


def load_checkpoint(model: "BeMambaModel", ckpt_path: str, device: torch.device) -> set:
    """Load a checkpoint, auto-migrating from old architecture if needed.

    Returns:
        missing_keys: set of model keys not found in checkpoint
                      (empty if migration was clean)
    """
    state = torch.load(ckpt_path, map_location=device)

    # ── Try direct load (new checkpoint) ──
    try:
        model.load_state_dict(state, strict=True)
        return set()  # clean load
    except RuntimeError:
        pass

    # ── Migrate from old architecture ──
    migrated = migrate_legacy_state_dict(state)
    result = model.load_state_dict(migrated, strict=False)

    # Only report keys that aren't expected new DMAF modules
    expected_new_modules = (
        "img_mask_encoder", "radar_mask_encoder", "lidar_mask_encoder", "gps_mask_encoder",
        "img_temporal", "radar_temporal", "lidar_temporal",
        "img_reliability", "radar_reliability", "lidar_reliability", "gps_reliability",
        "cross_modal_fusion",
    )
    real_missing = [k for k in result.missing_keys
                    if not any(m in k for m in expected_new_modules)]

    if real_missing:
        print(f"  [WARN] Missing keys (non-DMAF): {real_missing[:5]}...")
    if result.unexpected_keys:
        print(f"  [WARN] Unexpected keys (ignored): {len(result.unexpected_keys)} keys")

    print(f"  Loaded legacy checkpoint from {ckpt_path}")
    return set(real_missing)
