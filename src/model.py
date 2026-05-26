from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torchvision import models


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


class GPSProjection(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, gps: torch.Tensor) -> torch.Tensor:
        return self.net(gps)


class MaskEncoder(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(mask)


class ReliabilityGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
            nn.Sigmoid(),
        )

    def forward(self, img_mask: torch.Tensor, radar_mask: torch.Tensor, lidar_mask: torch.Tensor) -> torch.Tensor:
        ratios = torch.stack(
            [img_mask.mean(dim=1), radar_mask.mean(dim=1), lidar_mask.mean(dim=1)],
            dim=1,
        )
        return self.net(ratios)


class ModalityBackbone(nn.Module):
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
        return feat


class TimeSequenceBranch(nn.Module):
    def __init__(self, backbone_name: str, in_channels: int, config: BeMambaConfig):
        super().__init__()
        self.encoder = ModalityBackbone(
            backbone_name=backbone_name,
            in_channels=in_channels,
            d_model=config.d_model,
            patch_grid=config.patch_grid,
            pretrained=config.pretrained_backbones,
        )
        if config.temporal_order not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported temporal_order: {config.temporal_order}")
        if config.spatial_scan not in {"row", "vertical"}:
            raise ValueError(f"Unsupported spatial_scan: {config.spatial_scan}")
        self.temporal_order = config.temporal_order
        self.spatial_scan = config.spatial_scan
        self.temporal_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "channel_encoding": ChannelEncoding(config.d_model),
                        "tf_mamba": TFMamba(
                            d_model=config.d_model,
                            d_state=config.d_state,
                            d_conv=config.d_conv,
                            expand=config.expand,
                            dropout=config.dropout,
                        ),
                    }
                )
                for _ in range(config.temporal_layers)
            ]
        )

    def freeze_stem(self) -> None:
        self.encoder.freeze_stem()

    def _flatten_spatial(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.spatial_scan == "vertical":
            encoded = encoded.transpose(-1, -2)
        return encoded.flatten(start_dim=-2)

    def _tokens_to_map(self, tokens: torch.Tensor, pooled_h: int, pooled_w: int) -> torch.Tensor:
        if self.spatial_scan == "vertical":
            return tokens.reshape(tokens.size(0), pooled_w, pooled_h, tokens.size(-1)).transpose(1, 2)
        return tokens.reshape(tokens.size(0), pooled_h, pooled_w, tokens.size(-1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels, height, width = x.shape
        encoded = self.encoder(x.reshape(batch_size * seq_len, channels, height, width))
        _, d_model, pooled_h, pooled_w = encoded.shape
        spatial_len = pooled_h * pooled_w

        encoded = encoded.reshape(batch_size, seq_len, d_model, pooled_h, pooled_w)
        encoded = self._flatten_spatial(encoded)
        if self.temporal_order == "reverse":
            encoded = torch.flip(encoded, dims=[1])
        time_sequence = encoded.permute(0, 1, 3, 2).reshape(batch_size, seq_len * spatial_len, d_model)
        fused_sequence = time_sequence
        for block in self.temporal_blocks:
            channel_encoded = block["channel_encoding"](fused_sequence)
            fused_sequence = fused_sequence + block["tf_mamba"](channel_encoded)

        per_time = fused_sequence.reshape(batch_size, seq_len, spatial_len, d_model)
        aggregated = per_time.sum(dim=1)
        feature_map = self._tokens_to_map(aggregated, pooled_h, pooled_w).permute(0, 3, 1, 2)
        return aggregated, feature_map


class BeMambaModel(nn.Module):
    def __init__(self, config: BeMambaConfig | None = None):
        super().__init__()
        self.config = config or BeMambaConfig()
        cfg = self.config
        self.spatial_len = cfg.patch_grid * cfg.patch_grid
        self.freeze_image_stem = cfg.freeze_image_stem
        self.missing_enabled = cfg.missing_enabled

        self.image_branch = TimeSequenceBranch("resnet34", 3, cfg)
        self.lidar_branch = TimeSequenceBranch("resnet18", 1, cfg)
        self.radar_branch = TimeSequenceBranch("resnet18", 2, cfg)
        if cfg.freeze_image_stem:
            self.image_branch.freeze_stem()
        self.gps_projection = GPSProjection(cfg.d_model, cfg.gps_hidden_dim, cfg.dropout)

        self.image_align = nn.Linear(cfg.d_model, cfg.d_model)
        self.lidar_align = nn.Linear(cfg.d_model, cfg.d_model)
        self.radar_align = nn.Linear(cfg.d_model, cfg.d_model)

        if self.missing_enabled:
            self.img_mask_encoder = MaskEncoder(5, cfg.d_model)
            self.radar_mask_encoder = MaskEncoder(5, cfg.d_model)
            self.lidar_mask_encoder = MaskEncoder(5, cfg.d_model)
            self.gps_mask_encoder = MaskEncoder(2, cfg.d_model)
            self.reliability_gate = ReliabilityGate(cfg.d_model)

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

        flattened_dim = cfg.d_model * self.spatial_len
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
        if self.freeze_image_stem:
            self.image_branch.encoder.features[1].eval()
        return self

    def _build_modal_sequences(
        self,
        image_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
        radar_tokens: torch.Tensor,
        gps_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gps_start = gps_tokens[:, 0, :].unsqueeze(1).expand(-1, self.spatial_len, -1)
        gps_end = gps_tokens[:, 1, :].unsqueeze(1).expand(-1, self.spatial_len, -1)

        image_tokens = self.image_align(image_tokens)
        lidar_tokens = self.lidar_align(lidar_tokens)
        radar_tokens = self.radar_align(radar_tokens)

        seq_c1 = torch.stack([gps_start, image_tokens, lidar_tokens, radar_tokens, gps_end], dim=2)
        seq_c2 = torch.stack([gps_start, lidar_tokens, radar_tokens, image_tokens, gps_end], dim=2)
        seq_c3 = torch.stack([gps_start, radar_tokens, image_tokens, lidar_tokens, gps_end], dim=2)

        batch_size, spatial_len, _, d_model = seq_c1.shape
        seq_c1 = seq_c1.reshape(batch_size, spatial_len * 5, d_model)
        seq_c2 = seq_c2.reshape(batch_size, spatial_len * 5, d_model)
        seq_c3 = seq_c3.reshape(batch_size, spatial_len * 5, d_model)
        return seq_c1, seq_c2, seq_c3

    def _modal_fusion(
        self,
        sequences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        reliability_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        running_sequences = list(sequences)
        for layer_blocks in self.modal_blocks:
            next_sequences = []
            for block, sequence in zip(layer_blocks, running_sequences):
                encoded = block(sequence)
                next_sequences.append(sequence + encoded)
            running_sequences = next_sequences

        encoded_sequences = []
        for sequence in running_sequences:
            encoded = sequence.reshape(sequence.size(0), self.spatial_len, 5, self.config.d_model)
            encoded = encoded.mean(dim=2)
            encoded_sequences.append(encoded)

        if reliability_weights is None:
            fused = encoded_sequences[0] + encoded_sequences[1] + encoded_sequences[2]
        else:
            w = reliability_weights  # [B, 3]
            w = w.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 3]
            fused = (
                w[:, :, :, 0] * encoded_sequences[0]
                + w[:, :, :, 1] * encoded_sequences[1]
                + w[:, :, :, 2] * encoded_sequences[2]
            )
        return fused

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
        image_tokens, _ = self.image_branch(imgs)
        lidar_tokens, _ = self.lidar_branch(lidars)
        radar_tokens, _ = self.radar_branch(radars)
        gps_tokens = self.gps_projection(gps)

        if self.missing_enabled and img_mask is not None:
            image_tokens = image_tokens + self.img_mask_encoder(img_mask).unsqueeze(1)
        if self.missing_enabled and radar_mask is not None:
            radar_tokens = radar_tokens + self.radar_mask_encoder(radar_mask).unsqueeze(1)
        if self.missing_enabled and lidar_mask is not None:
            lidar_tokens = lidar_tokens + self.lidar_mask_encoder(lidar_mask).unsqueeze(1)
        if self.missing_enabled and gps_mask is not None:
            gps_tokens = gps_tokens + self.gps_mask_encoder(gps_mask)

        modal_sequences = self._build_modal_sequences(image_tokens, lidar_tokens, radar_tokens, gps_tokens)

        reliability_weights = None
        if self.missing_enabled and img_mask is not None:
            reliability_weights = self.reliability_gate(img_mask, radar_mask, lidar_mask)

        fused_tokens = self._modal_fusion(modal_sequences, reliability_weights)
        flattened = fused_tokens.reshape(fused_tokens.size(0), -1)
        return self.head(flattened)
