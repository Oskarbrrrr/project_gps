from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
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
    pretrained_backbones: bool = True
    temporal_layers: int = 1
    fusion_layers: int = 1
    freeze_image_stem: bool = False
    temporal_order: str = "forward"
    spatial_scan: str = "vertical"
    missing_enabled: bool = False

    # ── DMAF ablation switches ──
    # All default True — use --no-* flags to disable individual components.
    use_mask_embed: bool = True
    use_cross_attn: bool = True
    use_reliability: bool = True
    model_variant: str = "bemamba"
    clean_cross_attn: bool = False
    spatial_mixer_layers: int = 0
    use_order_gate: bool = False
    use_attn_head: bool = False


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

    def __init__(self, backbone_name: str, in_channels: int, d_model: int, patch_grid: int, pretrained: bool):
        super().__init__()
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
            out_channels = 128
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            out_channels = 128
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

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
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


class GPSProjection(nn.Module):
    """Pure GPS feature projection (no mask handling — see MaskEncoder).

    Input:  [B, 2, 2]  (2 time points × {dist, angle})
    Output: [B, 2, d_model]
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
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
    ):
        super().__init__()
        if temporal_order not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported temporal_order: {temporal_order}")
        if spatial_scan not in {"row", "vertical"}:
            raise ValueError(f"Unsupported spatial_scan: {spatial_scan}")

        self.temporal_order = temporal_order
        self.spatial_scan = spatial_scan

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

        # Mask-weighted aggregation
        if mask is not None:
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


# ═══════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════

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
        if cfg.model_variant not in {"bemamba", "clean_plus"}:
            raise ValueError(f"Unsupported model_variant: {cfg.model_variant}")
        clean_plus = cfg.model_variant == "clean_plus"
        spatial_mixer_layers = cfg.spatial_mixer_layers
        if clean_plus and spatial_mixer_layers <= 0:
            spatial_mixer_layers = 1
        if spatial_mixer_layers < 0:
            raise ValueError("spatial_mixer_layers must be >= 0")

        # Resolve optional modules. Mask/reliability stay DMAF-only; clean_plus can use cross-attn.
        self.use_mask_embed = cfg.missing_enabled and cfg.use_mask_embed
        self.use_cross_attn = cfg.use_cross_attn and (cfg.missing_enabled or cfg.clean_cross_attn or clean_plus)
        self.use_reliability = cfg.missing_enabled and cfg.use_reliability
        self.use_spatial_mixer = spatial_mixer_layers > 0
        self.use_order_gate = cfg.use_order_gate or clean_plus
        self.use_attn_head = cfg.use_attn_head or clean_plus

        # ── Phase 1: Spatial encoders ──
        self.img_backbone = ModalityBackbone("resnet34", 3, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones)
        self.radar_backbone = ModalityBackbone("resnet18", 2, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones)
        self.lidar_backbone = ModalityBackbone("resnet18", 1, cfg.d_model, cfg.patch_grid, cfg.pretrained_backbones)
        if cfg.freeze_image_stem:
            self.img_backbone.freeze_stem()

        self.gps_projection = GPSProjection(cfg.d_model, cfg.gps_hidden_dim, cfg.dropout)

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
                        for _ in range(3)
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_image_stem:
            self.img_backbone.features[1].eval()
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build three mixed-modality sequences (BeMamba paper Eq. 11-13).

        Each sequence is a different permutation of [gps_start, img, lidar, radar, gps_end],
        to handle Mamba's order sensitivity.
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
        return seq_c1, seq_c2, seq_c3

    def _modal_fusion(
        self,
        sequences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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

        # Mean-reduce the 5-modality dimension, then sum the 3 orderings
        encoded_sequences = []
        for sequence in running_sequences:
            seq = sequence.reshape(sequence.size(0), self.spatial_len, 5, self.config.d_model)
            encoded_sequences.append(seq.mean(dim=2))  # [B, N, d]

        if self.order_gate is not None:
            return self.order_gate(torch.stack(encoded_sequences, dim=1))
        return encoded_sequences[0] + encoded_sequences[1] + encoded_sequences[2]

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
    ) -> torch.Tensor:
        """Forward pass with modular 7-phase pipeline.

        Args:
            imgs:   [B, 5, 3, H, W]
            radars: [B, 5, 2, H, W]
            lidars: [B, 5, 1, H, W]
            gps:    [B, 2, 2]
            *_mask: [B, seq_len] or None  (0=missing, 1=present)

        Returns:
            logits: [B, num_classes]
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
        # Pass masks for weighted aggregation (active whenever masks exist)
        img_tokens = self.img_temporal(img_spatial, mask=img_mask if self.missing_enabled else None)
        radar_tokens = self.radar_temporal(radar_spatial, mask=radar_mask if self.missing_enabled else None)
        lidar_tokens = self.lidar_temporal(lidar_spatial, mask=lidar_mask if self.missing_enabled else None)
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
        sequences = self._build_modal_sequences(img_tokens, lidar_tokens, radar_tokens, gps_tokens)
        fused = self._modal_fusion(sequences)  # [B, N, d]

        # ═══════════════════════════════════════════════════════════
        # Phase 7: Classification head
        # ═══════════════════════════════════════════════════════════
        if self.use_attn_head:
            return self.head(fused)
        return self.head(fused.reshape(B, -1))


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
