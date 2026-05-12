from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torchvision import models


@dataclass
class BeMambaConfig:
    num_classes: int = 64
    d_model: int = 192
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    patch_grid: int = 6
    temporal_layers: int = 2
    fusion_layers: int = 2
    dropout: float = 0.2
    gps_hidden_dim: int = 96
    pretrained_backbones: bool = True
    freeze_image_stem: bool = False


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        out_forward = self.forward_mamba(x_norm)
        out_backward = torch.flip(
            self.backward_mamba(torch.flip(x_norm, dims=[1])),
            dims=[1],
        )
        gated = self.gate(x_norm) * (out_forward + out_backward)
        return residual + self.dropout(self.out_proj(gated))


class MambaStack(nn.Module):
    def __init__(self, num_layers: int, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


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


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        d_model: int,
        patch_grid: int,
        pretrained: bool,
        freeze_stem: bool = False,
    ):
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
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

        if freeze_stem:
            for param in self.features[0].parameters():
                param.requires_grad = False
            for param in self.features[1].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = self.proj(feat)
        return feat


class TemporalModalityBranch(nn.Module):
    def __init__(
        self,
        name: str,
        backbone_name: str,
        in_channels: int,
        config: BeMambaConfig,
        modality_index: int,
    ):
        super().__init__()
        self.name = name
        self.patch_grid = config.patch_grid
        self.num_patches = config.patch_grid * config.patch_grid
        self.encoder = ModalityEncoder(
            backbone_name=backbone_name,
            in_channels=in_channels,
            d_model=config.d_model,
            patch_grid=config.patch_grid,
            pretrained=config.pretrained_backbones,
            freeze_stem=(config.freeze_image_stem and name == "image"),
        )
        self.temporal_stack = MambaStack(
            num_layers=config.temporal_layers,
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=config.dropout,
        )
        self.frame_pos = nn.Parameter(torch.zeros(1, 5, config.d_model))
        self.patch_pos = nn.Parameter(torch.zeros(1, self.num_patches, config.d_model))
        self.modality_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.modality_token, std=0.02)
        self.modality_index = modality_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape
        encoded = self.encoder(x.view(batch_size * seq_len, channels, height, width))
        encoded = encoded.view(batch_size, seq_len, -1, self.num_patches).permute(0, 1, 3, 2)
        encoded = encoded + self.frame_pos[:, :seq_len].unsqueeze(2) + self.patch_pos.unsqueeze(1)

        patch_first = encoded.permute(0, 2, 1, 3).reshape(batch_size * self.num_patches, seq_len, -1)
        temporal_out = self.temporal_stack(patch_first)
        temporal_out = temporal_out.mean(dim=1).view(batch_size, self.num_patches, -1)
        return temporal_out + self.modality_token


class BeMambaModel(nn.Module):
    def __init__(self, config: BeMambaConfig | None = None):
        super().__init__()
        self.config = config or BeMambaConfig()
        cfg = self.config

        self.branches = nn.ModuleDict(
            {
                "image": TemporalModalityBranch("image", "resnet34", 3, cfg, modality_index=0),
                "radar": TemporalModalityBranch("radar", "resnet18", 2, cfg, modality_index=1),
                "lidar": TemporalModalityBranch("lidar", "resnet18", 1, cfg, modality_index=2),
            }
        )
        self.gps_projection = GPSProjection(cfg.d_model, cfg.gps_hidden_dim, cfg.dropout)
        self.fusion_stacks = nn.ModuleList(
            [
                MambaStack(
                    num_layers=cfg.fusion_layers,
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                )
                for _ in range(3)
            ]
        )
        self.permutations: List[Tuple[str, str, str]] = [
            ("image", "lidar", "radar"),
            ("lidar", "radar", "image"),
            ("radar", "image", "lidar"),
        ]
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def encode_modalities(
        self,
        imgs: torch.Tensor,
        radars: torch.Tensor,
        lidars: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "image": self.branches["image"](imgs),
            "radar": self.branches["radar"](radars),
            "lidar": self.branches["lidar"](lidars),
        }

    def forward(self, imgs: torch.Tensor, radars: torch.Tensor, lidars: torch.Tensor, gps: torch.Tensor) -> torch.Tensor:
        modality_tokens = self.encode_modalities(imgs, radars, lidars)
        gps_start = self.gps_projection(gps[:, 0, :]).unsqueeze(1)
        gps_end = self.gps_projection(gps[:, 1, :]).unsqueeze(1)

        fused_sequences = []
        for fusion_stack, ordering in zip(self.fusion_stacks, self.permutations):
            sequence = [gps_start]
            for modality_name in ordering:
                sequence.append(modality_tokens[modality_name])
            sequence.append(gps_end)
            stacked = torch.cat(sequence, dim=1)
            fused_sequences.append(fusion_stack(stacked))

        fused = torch.stack(fused_sequences, dim=0).mean(dim=0)
        pooled = fused.mean(dim=1)
        return self.head(pooled)
