import argparse
import csv
import datetime
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.dataset import MultimodalDataset
from src.model import BeMambaConfig, BeMambaModel, load_checkpoint
from src.utils import calculate_apl, calculate_dba_score, calculate_topk_accuracy


@dataclass
class TrainConfig:
    data_root: str = "./Data/Multi_Modal"
    split_root: str = "./Data/splits"
    output_root: str = "./outputs"
    merge_train_val: bool = True
    image_subdir: str = "camera_data"
    lidar_representation: str = "count"
    scenarios: Tuple[str, ...] = ("scenario32", "scenario33", "scenario34")
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 8
    lr: float = 1e-4
    backbone_lr_scale: float = 1.0
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    weight_decay: float = 0.0
    min_lr: float = 1e-6
    warmup_epochs: int = 0
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.0
    ema_start_epoch: int = 8
    patience: int = 0
    early_stop_metric: str = "acc3"
    early_stop_mode: str = "max"
    min_delta: float = 0.0
    gamma: float = 2.0
    loss_name: str = "ce"
    label_smoothing: float = 0.0
    soft_power_temperature: float = 0.2
    hard_loss_weight: float = 0.5
    amp: bool = True
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every: int = 1
    save_every_epoch: bool = False
    device: str = "cuda"
    missing_enabled: bool = False
    missing_aug_enabled: bool = False
    dmaf_enabled: bool = False
    missing_frame_prob: float = 0.0
    missing_burst_prob: float = 0.0
    missing_burst_min: int = 2
    missing_burst_max: int = 3
    missing_modality_prob: float = 0.0
    missing_modality_min: int = 1
    missing_modality_max: int = 2
    missing_modalities: str = "img,radar,lidar,gps"
    missing_seed: int = 42
    use_mask_embed: bool = True
    use_cross_attn: bool = True
    use_reliability: bool = True
    model_variant: str = "bemamba"
    clean_cross_attn: bool = False
    spatial_mixer_layers: int = 0
    use_order_gate: bool = False
    use_attn_head: bool = False
    use_branch_ensemble: bool = False
    aux_loss_weight: float = 0.0
    backbone_stage: int = 2
    rank_margin_weight: float = 0.0
    rank_margin: float = 0.2
    rank_topk: int = 3


class AlphaFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_weight = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        return (alpha_weight * ((1.0 - pt) ** self.gamma) * ce_loss).mean()


class PowerSoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature: float = 0.2, hard_weight: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, power_vec: torch.Tensor) -> torch.Tensor:
        power_vec = torch.nan_to_num(power_vec.to(inputs.device), nan=0.0, posinf=0.0, neginf=0.0)
        power_vec = torch.clamp(power_vec, min=0.0)
        target_power = power_vec / (power_vec.sum(dim=1, keepdim=True) + 1e-8)

        soft_logits = torch.log(target_power + 1e-8) / max(self.temperature, 1e-8)
        soft_targets = torch.softmax(soft_logits, dim=1)
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        soft_loss = -(soft_targets * log_probs).sum(dim=1).mean()

        if self.hard_weight <= 0:
            return soft_loss
        hard_loss = self.hard_loss(inputs, targets)
        return self.hard_weight * hard_loss + (1.0 - self.hard_weight) * soft_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_alpha_weights(csv_path: str, num_classes: int = 64) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    beams = df["unit1_beam"].values - 1
    class_counts = np.bincount(beams, minlength=num_classes)
    total_samples = len(beams)
    alpha = total_samples / (num_classes * (class_counts + 1))
    alpha = alpha / np.mean(alpha)
    return torch.tensor(alpha, dtype=torch.float32)


def build_combined_train_csv(split_root: str, scenario_name: str, output_dir: str, merge_train_val: bool = True) -> str:
    train_csv = os.path.join(split_root, f"{scenario_name}_train.csv")
    val_csv = os.path.join(split_root, f"{scenario_name}_val.csv")
    test_csv = os.path.join(split_root, f"{scenario_name}_test.csv")

    if merge_train_val and os.path.exists(train_csv) and os.path.exists(val_csv):
        merged = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
        merged_path = os.path.join(output_dir, f"{scenario_name}_trainval.csv")
        merged.to_csv(merged_path, index=False)
        return merged_path

    if os.path.exists(train_csv):
        return train_csv

    raise FileNotFoundError(f"Could not find train split for {scenario_name}: {train_csv}")


def build_dataloader(dataset: MultimodalDataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
        drop_last=False,
    )


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.backbone_lr_scale <= 0:
        raise ValueError("backbone_lr_scale must be > 0")

    if config.backbone_lr_scale == 1.0:
        param_groups = model.parameters()
    else:
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("img_backbone.", "radar_backbone.", "lidar_backbone.")):
                backbone_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {"params": other_params, "lr": config.lr},
            {"params": backbone_params, "lr": config.lr * config.backbone_lr_scale},
        ]

    if config.optimizer_name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    if config.scheduler_name == "none":
        return None
    if config.scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {config.scheduler_name}")

    cosine_epochs = max(config.epochs - config.warmup_epochs, 1)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=config.min_lr)
    if config.warmup_epochs <= 0:
        return cosine

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        if not (0.0 < decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current_state = model.state_dict()
        for key, value in current_state.items():
            shadow_value = self.shadow[key]
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value.detach())

    def store(self, model: nn.Module) -> None:
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None


def is_better_metric(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported early stop mode: {mode}")


def move_batch_to_device(batch, device: torch.device):
    if len(batch) == 6:
        imgs, radars, lidars, gps, targets, power_vec = batch
        return (
            imgs.to(device, non_blocking=True),
            radars.to(device, non_blocking=True),
            lidars.to(device, non_blocking=True),
            gps.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
            power_vec,
            None,
            None,
            None,
            None,
        )
    imgs, radars, lidars, gps, targets, power_vec, img_m, rad_m, lid_m, gps_m = batch
    return (
        imgs.to(device, non_blocking=True),
        radars.to(device, non_blocking=True),
        lidars.to(device, non_blocking=True),
        gps.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        power_vec,
        img_m.to(device, non_blocking=True),
        rad_m.to(device, non_blocking=True),
        lid_m.to(device, non_blocking=True),
        gps_m.to(device, non_blocking=True),
    )


def run_epoch(
    model: BeMambaModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float,
    ema: ModelEMA | None = None,
    rank_margin_weight: float = 0.0,
    rank_margin: float = 0.2,
    rank_topk: int = 3,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_total = 0.0
    sample_total = 0
    acc1_total = 0.0
    acc2_total = 0.0
    acc3_total = 0.0
    dba_total = 0.0
    apl_total = 0.0

    for batch in loader:
            imgs, radars, lidars, gps, targets, power_vec, img_mask, radar_mask, lidar_mask, gps_mask = move_batch_to_device(batch, device)
            batch_size = targets.size(0)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with autocast(enabled=amp_enabled):
                    model_outputs = model(imgs, radars, lidars, gps, img_mask, radar_mask, lidar_mask, gps_mask)
                    aux_logits = None
                    if isinstance(model_outputs, tuple):
                        outputs, aux_logits = model_outputs
                    else:
                        outputs = model_outputs
                    loss = compute_loss(criterion, outputs, targets, power_vec)
                    if training and rank_margin_weight > 0:
                        loss = loss + rank_margin_weight * topk_margin_loss(
                            outputs,
                            targets,
                            topk=rank_topk,
                            margin=rank_margin,
                        )
                    aux_weight = getattr(model.config, "aux_loss_weight", 0.0)
                    if aux_logits is not None and aux_weight > 0:
                        aux_loss = 0.0
                        for branch_idx in range(aux_logits.size(1)):
                            aux_loss = aux_loss + compute_loss(
                                criterion,
                                aux_logits[:, branch_idx, :],
                                targets,
                                power_vec,
                            )
                        loss = loss + aux_weight * aux_loss / aux_logits.size(1)

            if training:
                assert optimizer is not None
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                if ema is not None:
                    ema.update(model)

            loss_total += loss.item() * batch_size
            sample_total += batch_size

            acc1, acc2, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 2, 3))
            acc1_total += acc1 * batch_size
            acc2_total += acc2 * batch_size
            acc3_total += acc3 * batch_size
            dba_total += calculate_dba_score(outputs, targets) * batch_size
            apl_total += calculate_apl(outputs, power_vec) * batch_size

    return {
        "loss": loss_total / max(sample_total, 1),
        "acc1": acc1_total / max(sample_total, 1),
        "acc2": acc2_total / max(sample_total, 1),
        "acc3": acc3_total / max(sample_total, 1),
        "dba": dba_total / max(sample_total, 1),
        "apl": apl_total / max(sample_total, 1),
    }


def save_config(run_dir: str, train_config: TrainConfig, model_config: BeMambaConfig) -> None:
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write("[train]\n")
        for key, value in asdict(train_config).items():
            handle.write(f"{key}={value}\n")
        handle.write("\n[model]\n")
        for key, value in asdict(model_config).items():
            handle.write(f"{key}={value}\n")


def build_criterion(train_config: TrainConfig, alpha_weights: torch.Tensor | None) -> nn.Module:
    if train_config.loss_name == "focal":
        return AlphaFocalLoss(alpha=alpha_weights, gamma=train_config.gamma)
    if train_config.loss_name == "power_soft_ce":
        return PowerSoftCrossEntropyLoss(
            temperature=train_config.soft_power_temperature,
            hard_weight=train_config.hard_loss_weight,
            label_smoothing=train_config.label_smoothing,
        )
    if train_config.loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    raise ValueError(f"Unsupported loss: {train_config.loss_name}")


def compute_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    power_vec: torch.Tensor,
) -> torch.Tensor:
    if isinstance(criterion, PowerSoftCrossEntropyLoss):
        return criterion(outputs, targets, power_vec)
    return criterion(outputs, targets)


def topk_margin_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: int = 3,
    margin: float = 0.2,
) -> torch.Tensor:
    target_logits = outputs.gather(1, targets.unsqueeze(1))
    negative_logits = outputs.masked_fill(
        torch.zeros_like(outputs, dtype=torch.bool).scatter_(1, targets.unsqueeze(1), True),
        float("-inf"),
    )
    kth_negative = negative_logits.topk(topk, dim=1).values[:, -1:].detach()
    return nn.functional.relu(kth_negative + margin - target_logits).mean()


def run_missing_robustness_tests(
    model: BeMambaModel,
    train_config: TrainConfig,
    scenario_name: str,
    gps_stats: dict,
    test_csv_path: str,
    device: torch.device,
    criterion: nn.Module,
    run_dir: str,
) -> list:
    # Protocol format: (name, frame_prob, burst_prob, modality_prob,
    #                    burst_min, burst_max, mod_min, mod_max)
    # None means "use dataset defaults" — only set when the protocol overrides.
    test_protocols = [
        # ── Clean baseline ──
        ("clean",       0.0, 0.0, 0.0, None, None, None, None),
        # ── Random frame missing (no burst, no modality) ──
        ("frame_p01",   0.1, 0.0, 0.0, None, None, None, None),
        ("frame_p03",   0.3, 0.0, 0.0, None, None, None, None),
        ("frame_p05",   0.5, 0.0, 0.0, None, None, None, None),
        # ── Burst missing: consecutive frames, no frame/modality ──
        ("burst_p02",   0.0, 0.2, 0.0, 2,    3,    None, None),
        ("burst_p04",   0.0, 0.4, 0.0, 2,    3,    None, None),
        ("burst_p06",   0.0, 0.6, 0.0, 2,    3,    None, None),
        # ── Modality missing: exactly 1 modality at a time ──
        ("modal_p02",   0.0, 0.0, 0.2, None, None, 1,    1),
        ("modal_p04",   0.0, 0.0, 0.4, None, None, 1,    1),
        ("modal_p06",   0.0, 0.0, 0.6, None, None, 1,    1),
        # ── Training-config hybrid (for reference) ──
        (
            "hybrid",
            train_config.missing_frame_prob,
            train_config.missing_burst_prob,
            train_config.missing_modality_prob,
            train_config.missing_burst_min,
            train_config.missing_burst_max,
            train_config.missing_modality_min,
            train_config.missing_modality_max,
        ),
    ]

    test_seed = train_config.missing_seed + 10000
    print(f"\n{'='*60}")
    print(f"  Missing Robustness Tests — {scenario_name}")
    print(f"{'='*60}")
    print(f"{'Protocol':<14} {'Top-1':>7} {'Top-2':>7} {'Top-3':>7} {'DBA':>8} {'APL':>9} {'Ret':>7}")
    print("-" * 62)

    clean_acc3 = None
    results = []

    for name, fp, bp, mp, b_min, b_max, m_min, m_max in test_protocols:
        has_missing = name != "clean"
        ds = MultimodalDataset(
            mode="test",
            data_root=train_config.data_root,
            split_root=train_config.split_root,
            scenario_name=scenario_name,
            csv_path=test_csv_path,
            image_subdir=train_config.image_subdir,
            gps_stats=gps_stats,
            lidar_representation=train_config.lidar_representation,
            missing_enabled=has_missing,
            return_missing_masks=(has_missing and train_config.dmaf_enabled),
            missing_frame_prob=fp,
            missing_burst_prob=bp,
            missing_burst_min=b_min if b_min is not None else 2,
            missing_burst_max=b_max if b_max is not None else 3,
            missing_modality_prob=mp,
            missing_modality_min=m_min if m_min is not None else 1,
            missing_modality_max=m_max if m_max is not None else 2,
            missing_modalities=train_config.missing_modalities,
            missing_seed=test_seed,
        )
        loader = build_dataloader(ds, train_config, shuffle=False)
        metrics = run_epoch(model, loader, criterion, None, None, device, False, 0.0)

        if name == "clean":
            clean_acc3 = metrics["acc3"]

        retention = (metrics["acc3"] / clean_acc3 * 100) if clean_acc3 and clean_acc3 > 0 else 0.0
        results.append((name, metrics, retention))

        del loader, ds
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"{name:<14} {metrics['acc1']:>6.2f}% {metrics['acc2']:>6.2f}% "
            f"{metrics['acc3']:>6.2f}% {metrics['dba']:>7.4f} {metrics['apl']:>8.4f} dB "
            f"{retention:>6.1f}%"
        )

    result_path = os.path.join(run_dir, "missing_test_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Missing Robustness Test Results ({scenario_name})\n")
        f.write(f"Reference clean Top-3: {clean_acc3:.2f}%\n\n")
        header = f"{'Protocol':<14} {'Top-1':>7} {'Top-2':>7} {'Top-3':>7} {'DBA':>8} {'APL':>9} {'Ret':>7}\n"
        f.write(header)
        f.write("-" * 62 + "\n")
        for name, metrics, retention in results:
            f.write(
                f"{name:<14} {metrics['acc1']:>6.2f}% {metrics['acc2']:>6.2f}% "
                f"{metrics['acc3']:>6.2f}% {metrics['dba']:>7.4f} {metrics['apl']:>8.4f} dB "
                f"{retention:>6.1f}%\n"
            )

    csv_path = os.path.join(run_dir, "missing_test_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "acc1", "acc2", "acc3", "dba", "apl", "retention_pct"])
        for name, metrics, retention in results:
            writer.writerow(
                [
                    name,
                    f"{metrics['acc1']:.4f}",
                    f"{metrics['acc2']:.4f}",
                    f"{metrics['acc3']:.4f}",
                    f"{metrics['dba']:.6f}",
                    f"{metrics['apl']:.6f}",
                    f"{retention:.2f}",
                ]
            )

    return results


def run_scenario(scenario_name: str, train_config: TrainConfig, model_config: BeMambaConfig, device: torch.device) -> None:
    print(f"\n========== Running {scenario_name} ==========")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_config.output_root, scenario_name, timestamp)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_csv_path = build_combined_train_csv(
        train_config.split_root,
        scenario_name,
        run_dir,
        merge_train_val=train_config.merge_train_val,
    )
    test_csv_path = os.path.join(train_config.split_root, f"{scenario_name}_test.csv")
    alpha_weights = None
    if train_config.loss_name == "focal":
        alpha_weights = calculate_alpha_weights(train_csv_path, num_classes=model_config.num_classes).to(device)

    train_ds = MultimodalDataset(
        mode="train",
        data_root=train_config.data_root,
        split_root=train_config.split_root,
        scenario_name=scenario_name,
        csv_path=train_csv_path,
        image_subdir=train_config.image_subdir,
        lidar_representation=train_config.lidar_representation,
        missing_enabled=train_config.missing_aug_enabled,
        return_missing_masks=(train_config.missing_aug_enabled and train_config.dmaf_enabled),
        missing_frame_prob=train_config.missing_frame_prob,
        missing_burst_prob=train_config.missing_burst_prob,
        missing_burst_min=train_config.missing_burst_min,
        missing_burst_max=train_config.missing_burst_max,
        missing_modality_prob=train_config.missing_modality_prob,
        missing_modality_min=train_config.missing_modality_min,
        missing_modality_max=train_config.missing_modality_max,
        missing_modalities=train_config.missing_modalities,
        missing_seed=train_config.missing_seed,
    )
    gps_stats = train_ds.get_gps_stats()
    test_ds = MultimodalDataset(
        mode="test",
        data_root=train_config.data_root,
        split_root=train_config.split_root,
        scenario_name=scenario_name,
        csv_path=test_csv_path,
        image_subdir=train_config.image_subdir,
        gps_stats=gps_stats,
        lidar_representation=train_config.lidar_representation,
    )

    train_loader = build_dataloader(train_ds, train_config, shuffle=True)
    test_loader = build_dataloader(test_ds, train_config, shuffle=False)

    model = BeMambaModel(model_config).to(device)
    ema = None
    criterion = build_criterion(train_config, alpha_weights)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    scaler = GradScaler(enabled=(train_config.amp and device.type == "cuda"))
    save_config(run_dir, train_config, model_config)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc1",
                "train_acc2",
                "train_acc3",
                "test_loss",
                "test_acc1",
                "test_acc2",
                "test_acc3",
                "test_dba",
                "test_apl",
                "is_best",
            ]
        )

    start_time = time.time()
    final_test_metrics = None
    best_epoch = 0
    best_metric_value = None
    best_test_metrics = None
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")

    for epoch in range(1, train_config.epochs + 1):
        if train_config.ema_decay > 0 and ema is None and epoch >= train_config.ema_start_epoch:
            ema = ModelEMA(model, train_config.ema_decay)
        if train_config.missing_aug_enabled:
            train_ds.set_missing_epoch(epoch)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=(train_config.amp and device.type == "cuda"),
            grad_clip_norm=train_config.grad_clip_norm,
            ema=ema,
            rank_margin_weight=train_config.rank_margin_weight,
            rank_margin=train_config.rank_margin,
            rank_topk=train_config.rank_topk,
        )
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=(train_config.amp and device.type == "cuda"),
            grad_clip_norm=train_config.grad_clip_norm,
        )
        if ema is not None:
            ema.restore(model)
        final_test_metrics = test_metrics
        monitored_value = test_metrics[train_config.early_stop_metric]

        if is_better_metric(
            monitored_value,
            best_metric_value,
            train_config.early_stop_mode,
            train_config.min_delta,
        ):
            best_metric_value = monitored_value
            best_epoch = epoch
            best_test_metrics = dict(test_metrics)
            epochs_without_improvement = 0
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
            torch.save(model.state_dict(), best_ckpt_path)
            if ema is not None:
                ema.restore(model)
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / epoch
        eta_seconds = int(avg_epoch_time * (train_config.epochs - epoch))
        eta_string = str(datetime.timedelta(seconds=eta_seconds))
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % train_config.log_every == 0:
            print(
                f"Epoch {epoch:03d}/{train_config.epochs} "
                f"| lr={current_lr:.2e} "
                f"| train_loss={train_metrics['loss']:.4f} "
                f"| train_acc3={train_metrics['acc3']:.2f}% "
                f"| test_loss={test_metrics['loss']:.4f} "
                f"| test_acc3={test_metrics['acc3']:.2f}% "
                f"| test_dba={test_metrics['dba']:.4f} "
                f"| test_apl={test_metrics['apl']:.4f} dB "
                f"| ETA {eta_string}"
            )

        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    f"{current_lr:.8f}",
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['acc1']:.4f}",
                    f"{train_metrics['acc2']:.4f}",
                    f"{train_metrics['acc3']:.4f}",
                    f"{test_metrics['loss']:.6f}",
                    f"{test_metrics['acc1']:.4f}",
                    f"{test_metrics['acc2']:.4f}",
                    f"{test_metrics['acc3']:.4f}",
                    f"{test_metrics['dba']:.6f}",
                    f"{test_metrics['apl']:.6f}",
                    epoch == best_epoch,
                ]
            )

        if scheduler is not None:
            scheduler.step()

        if train_config.save_every_epoch:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"epoch_{epoch:03d}.pth"))

        if train_config.patience > 0 and epochs_without_improvement >= train_config.patience:
            print(
                f"Early stopping triggered at epoch {epoch:03d} "
                f"(best_epoch={best_epoch:03d}, best_{train_config.early_stop_metric}={best_metric_value:.4f})"
            )
            break

    final_ckpt = os.path.join(checkpoints_dir, "final_model.pth")
    torch.save(model.state_dict(), final_ckpt)
    if best_test_metrics is None:
        best_test_metrics = final_test_metrics if final_test_metrics is not None else run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=(train_config.amp and device.type == "cuda"),
            grad_clip_norm=train_config.grad_clip_norm,
        )
        best_epoch = epoch
        best_metric_value = best_test_metrics[train_config.early_stop_metric]
        torch.save(model.state_dict(), best_ckpt_path)

    load_checkpoint(model, best_ckpt_path, device)
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=device,
        amp_enabled=(train_config.amp and device.type == "cuda"),
        grad_clip_norm=train_config.grad_clip_norm,
    )

    result_path = os.path.join(run_dir, "final_test_result.txt")
    result_text = (
        f"[Best Results for {scenario_name}]\n"
        f"best_epoch: {best_epoch}\n"
        f"monitor_metric: {train_config.early_stop_metric}\n"
        f"monitor_value: {best_metric_value:.4f}\n"
        f"top1_acc: {test_metrics['acc1']:.2f}%\n"
        f"top2_acc: {test_metrics['acc2']:.2f}%\n"
        f"top3_acc: {test_metrics['acc3']:.2f}%\n"
        f"dba: {test_metrics['dba']:.4f}\n"
        f"apl: {test_metrics['apl']:.4f} dB\n"
    )
    print("\n" + result_text)
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write(result_text)

    if train_config.missing_aug_enabled or train_config.dmaf_enabled:
        run_missing_robustness_tests(
            model=model,
            train_config=train_config,
            scenario_name=scenario_name,
            gps_stats=gps_stats,
            test_csv_path=test_csv_path,
            device=device,
            criterion=criterion,
            run_dir=run_dir,
        )

    if final_test_metrics is not None:
        last_result_path = os.path.join(run_dir, "last_epoch_result.txt")
        last_result_text = (
            f"[Last Epoch Results for {scenario_name}]\n"
            f"epoch: {epoch}\n"
            f"top1_acc: {final_test_metrics['acc1']:.2f}%\n"
            f"top2_acc: {final_test_metrics['acc2']:.2f}%\n"
            f"top3_acc: {final_test_metrics['acc3']:.2f}%\n"
            f"dba: {final_test_metrics['dba']:.4f}\n"
            f"apl: {final_test_metrics['apl']:.4f} dB\n"
        )
        with open(last_result_path, "w", encoding="utf-8") as handle:
            handle.write(last_result_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BeMamba reproduction experiments.")
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--split-root", default="./Data/splits")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--no-merge-trainval", action="store_true")
    parser.add_argument("--image-subdir", default="camera_data")
    parser.add_argument("--lidar-representation", choices=["binary", "count"], default="count")
    parser.add_argument("--scenarios", nargs="+", default=["scenario32", "scenario33", "scenario34"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=1.0,
                        help="Scale the learning rate for img/radar/lidar ResNet backbones")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Use exponential moving average weights for validation/checkpointing")
    parser.add_argument("--ema-start-epoch", type=int, default=8,
                        help="Start EMA after this epoch; avoids averaging random early weights")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", choices=["loss", "acc1", "acc2", "acc3", "dba", "apl"], default="acc3")
    parser.add_argument("--early-stop-mode", choices=["max", "min"], default="max")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--loss", choices=["ce", "focal", "power_soft_ce"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--soft-power-temperature", type=float, default=0.2)
    parser.add_argument("--hard-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--patch-grid", type=int, default=6)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--temporal-order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--spatial-scan", choices=["vertical", "row"], default="vertical")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gps-hidden-dim", type=int, default=96)
    parser.add_argument("--no-pretrained-backbones", action="store_true")
    parser.add_argument("--freeze-image-stem", action="store_true")
    parser.add_argument("--model-variant", choices=["bemamba", "clean_plus", "clean_plus_v2", "clean_plus_v3", "clean_plus_v4", "clean_plus_v5", "clean_plus_v6"], default="bemamba",
                        help="Use the original BeMamba path or the enhanced clean-data variant")
    parser.add_argument("--backbone-stage", type=int, choices=[2, 3, 4], default=None,
                        help="Last ResNet stage used by modality backbones; clean_plus_v4 defaults to 3")
    parser.add_argument("--clean-cross-attn", action="store_true",
                        help="Enable CrossModalFusion even when DMAF/missing masks are disabled")
    parser.add_argument("--spatial-mixer-layers", type=int, default=None,
                        help="Number of per-modality spatial self-attention layers; clean_plus defaults to 1")
    parser.add_argument("--order-gate", action="store_true",
                        help="Use sample-adaptive weights for the three modal ordering sequences")
    parser.add_argument("--attn-head", action="store_true",
                        help="Use attention/mean/max pooling features in the prediction head")
    parser.add_argument("--branch-ensemble", action="store_true",
                        help="Blend fused logits with auxiliary image/radar/lidar/GPS logits")
    parser.add_argument("--aux-loss-weight", type=float, default=None,
                        help="Auxiliary branch loss weight; clean_plus_v3 defaults to 0.25")
    parser.add_argument("--rank-margin-weight", type=float, default=None,
                        help="Top-k ranking margin loss weight; clean_plus_v5/v6 defaults to 0.05")
    parser.add_argument("--rank-margin", type=float, default=0.2,
                        help="Margin used by the Top-k ranking loss")
    parser.add_argument("--rank-topk", type=int, default=3,
                        help="Top-k target for the ranking margin loss")
    parser.add_argument("--missing-enabled", action="store_true",
                        help="Legacy convenience flag: enable both missing augmentation and DMAF")
    parser.add_argument("--missing-aug-enabled", action="store_true",
                        help="Enable dataset-side missing augmentation without requiring DMAF")
    parser.add_argument("--dmaf-enabled", action="store_true",
                        help="Enable DMAF mask-aware model modules")
    parser.add_argument("--no-dmaf", action="store_true",
                        help="Disable DMAF even when --missing-enabled is used; useful for Missing-Aug BeMamba")
    parser.add_argument("--missing-frame-prob", type=float, default=0.0)
    parser.add_argument("--missing-burst-prob", type=float, default=0.0)
    parser.add_argument("--missing-burst-min", type=int, default=2)
    parser.add_argument("--missing-burst-max", type=int, default=3)
    parser.add_argument("--missing-modality-prob", type=float, default=0.0)
    parser.add_argument("--missing-modality-min", type=int, default=1)
    parser.add_argument("--missing-modality-max", type=int, default=2)
    parser.add_argument("--missing-modalities", default="img,radar,lidar,gps")
    parser.add_argument("--missing-seed", type=int, default=42)
    parser.add_argument("--no-mask-embed", action="store_true",
                        help="Disable MaskEncoder (mask embedding injection + weighted aggregation)")
    parser.add_argument("--no-cross-attn", action="store_true",
                        help="Disable CrossModalFusion (direct cross-modal attention)")
    parser.add_argument("--no-reliability", action="store_true",
                        help="Disable ReliabilityEstimator (per-modality reliability weighting)")
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> Tuple[TrainConfig, BeMambaConfig]:
    if args.dmaf_enabled and args.no_dmaf:
        raise ValueError("--dmaf-enabled and --no-dmaf cannot be used together")
    if args.ema_decay != 0.0 and not (0.0 < args.ema_decay < 1.0):
        raise ValueError("--ema-decay must be 0 or in (0, 1)")
    if args.ema_start_epoch < 1:
        raise ValueError("--ema-start-epoch must be >= 1")

    missing_aug_enabled = args.missing_enabled or args.missing_aug_enabled
    dmaf_enabled = (args.missing_enabled or args.dmaf_enabled) and (not args.no_dmaf)
    clean_plus = args.model_variant == "clean_plus"
    clean_plus_v2 = args.model_variant == "clean_plus_v2"
    clean_plus_v3 = args.model_variant == "clean_plus_v3"
    clean_plus_v4 = args.model_variant == "clean_plus_v4"
    clean_plus_v5 = args.model_variant == "clean_plus_v5"
    clean_plus_v6 = args.model_variant == "clean_plus_v6"
    backbone_stage = args.backbone_stage
    if backbone_stage is None:
        backbone_stage = 3 if (clean_plus_v4 or clean_plus_v5 or clean_plus_v6) else 2
    spatial_mixer_layers = args.spatial_mixer_layers
    if spatial_mixer_layers is None:
        spatial_mixer_layers = 1 if clean_plus else 0
    if spatial_mixer_layers < 0:
        raise ValueError("--spatial-mixer-layers must be >= 0")
    clean_cross_attn = args.clean_cross_attn or clean_plus
    use_order_gate = args.order_gate or clean_plus
    use_attn_head = args.attn_head or clean_plus
    use_branch_ensemble = args.branch_ensemble or clean_plus_v2 or clean_plus_v3 or clean_plus_v4 or clean_plus_v5 or clean_plus_v6
    aux_loss_weight = args.aux_loss_weight
    if aux_loss_weight is None:
        aux_loss_weight = 0.25 if clean_plus_v3 else 0.0
    if aux_loss_weight < 0:
        raise ValueError("--aux-loss-weight must be >= 0")
    rank_margin_weight = args.rank_margin_weight
    if rank_margin_weight is None:
        rank_margin_weight = 0.05 if (clean_plus_v5 or clean_plus_v6) else 0.0
    if rank_margin_weight < 0:
        raise ValueError("--rank-margin-weight must be >= 0")
    if args.rank_margin < 0:
        raise ValueError("--rank-margin must be >= 0")
    if args.rank_topk < 1:
        raise ValueError("--rank-topk must be >= 1")

    train_config = TrainConfig(
        data_root=args.data_root,
        split_root=args.split_root,
        output_root=args.output_root,
        merge_train_val=(not args.no_merge_trainval),
        image_subdir=args.image_subdir,
        lidar_representation=args.lidar_representation,
        scenarios=tuple(args.scenarios),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        grad_clip_norm=args.grad_clip_norm,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start_epoch,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_mode=args.early_stop_mode,
        min_delta=args.min_delta,
        gamma=args.gamma,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        soft_power_temperature=args.soft_power_temperature,
        hard_loss_weight=args.hard_loss_weight,
        amp=(not args.no_amp),
        seed=args.seed,
        pin_memory=(not args.no_pin_memory),
        persistent_workers=(not args.no_persistent_workers),
        save_every_epoch=args.save_every_epoch,
        device=args.device,
        missing_enabled=args.missing_enabled,
        missing_aug_enabled=missing_aug_enabled,
        dmaf_enabled=dmaf_enabled,
        missing_frame_prob=args.missing_frame_prob,
        missing_burst_prob=args.missing_burst_prob,
        missing_burst_min=args.missing_burst_min,
        missing_burst_max=args.missing_burst_max,
        missing_modality_prob=args.missing_modality_prob,
        missing_modality_min=args.missing_modality_min,
        missing_modality_max=args.missing_modality_max,
        missing_modalities=args.missing_modalities,
        missing_seed=args.missing_seed,
        use_mask_embed=not args.no_mask_embed,
        use_cross_attn=not args.no_cross_attn,
        use_reliability=not args.no_reliability,
        model_variant=args.model_variant,
        clean_cross_attn=clean_cross_attn,
        spatial_mixer_layers=spatial_mixer_layers,
        use_order_gate=use_order_gate,
        use_attn_head=use_attn_head,
        use_branch_ensemble=use_branch_ensemble,
        aux_loss_weight=aux_loss_weight,
        backbone_stage=backbone_stage,
        rank_margin_weight=rank_margin_weight,
        rank_margin=args.rank_margin,
        rank_topk=args.rank_topk,
    )
    model_config = BeMambaConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        patch_grid=args.patch_grid,
        temporal_layers=args.temporal_layers,
        fusion_layers=args.fusion_layers,
        temporal_order=args.temporal_order,
        spatial_scan=args.spatial_scan,
        dropout=args.dropout,
        gps_hidden_dim=args.gps_hidden_dim,
        backbone_stage=backbone_stage,
        pretrained_backbones=(not args.no_pretrained_backbones),
        freeze_image_stem=args.freeze_image_stem,
        missing_enabled=dmaf_enabled,
        use_mask_embed=not args.no_mask_embed,
        use_cross_attn=not args.no_cross_attn,
        use_reliability=not args.no_reliability,
        model_variant=args.model_variant,
        clean_cross_attn=clean_cross_attn,
        spatial_mixer_layers=spatial_mixer_layers,
        use_order_gate=use_order_gate,
        use_attn_head=use_attn_head,
        use_branch_ensemble=use_branch_ensemble,
        use_beam_query_head=clean_plus_v5 or clean_plus_v6,
        use_multiscale_backbone=clean_plus_v6,
        return_aux_logits=(aux_loss_weight > 0),
        aux_loss_weight=aux_loss_weight,
    )
    return train_config, model_config


def main() -> None:
    args = parse_args()
    train_config, model_config = build_configs(args)
    set_seed(train_config.seed)
    device = prepare_device(train_config.device)
    os.makedirs(train_config.output_root, exist_ok=True)

    for scenario_name in train_config.scenarios:
        run_scenario(
            scenario_name=scenario_name,
            train_config=train_config,
            model_config=model_config,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
