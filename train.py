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
from src.model import BeMambaConfig, BeMambaModel
from src.utils import calculate_apl, calculate_dba_score, calculate_topk_accuracy


@dataclass
class TrainConfig:
    data_root: str = "./Data/Multi_Modal"
    split_root: str = "./Data/splits"
    output_root: str = "./outputs"
    image_subdir: str = "camera_data"
    scenarios: Tuple[str, ...] = ("scenario32", "scenario33", "scenario34")
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    min_lr: float = 1e-6
    warmup_epochs: int = 0
    grad_clip_norm: float = 1.0
    patience: int = 0
    early_stop_metric: str = "acc3"
    early_stop_mode: str = "max"
    min_delta: float = 0.0
    gamma: float = 2.0
    loss_name: str = "ce"
    amp: bool = True
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every: int = 1
    save_every_epoch: bool = False
    device: str = "cuda"


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


def build_combined_train_csv(split_root: str, scenario_name: str, output_dir: str) -> str:
    train_csv = os.path.join(split_root, f"{scenario_name}_train.csv")
    val_csv = os.path.join(split_root, f"{scenario_name}_val.csv")
    test_csv = os.path.join(split_root, f"{scenario_name}_test.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv):
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


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    return None


def is_better_metric(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported early stop mode: {mode}")


def move_batch_to_device(batch, device: torch.device):
    imgs, radars, lidars, gps, targets, power_vec = batch
    return (
        imgs.to(device, non_blocking=True),
        radars.to(device, non_blocking=True),
        lidars.to(device, non_blocking=True),
        gps.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        power_vec,
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
        imgs, radars, lidars, gps, targets, power_vec = move_batch_to_device(batch, device)
        batch_size = targets.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            outputs = model(imgs, radars, lidars, gps)
            loss = criterion(outputs, targets)

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
    if train_config.loss_name == "ce":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {train_config.loss_name}")


def run_scenario(scenario_name: str, train_config: TrainConfig, model_config: BeMambaConfig, device: torch.device) -> None:
    print(f"\n========== Running {scenario_name} ==========")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_config.output_root, scenario_name, timestamp)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_csv_path = build_combined_train_csv(train_config.split_root, scenario_name, run_dir)
    test_csv_path = os.path.join(train_config.split_root, f"{scenario_name}_test.csv")
    alpha_weights = None
    if train_config.loss_name == "focal":
        alpha_weights = calculate_alpha_weights(train_csv_path, num_classes=model_config.num_classes).to(device)

    train_ds = MultimodalDataset(
        data_root=train_config.data_root,
        split_root=train_config.split_root,
        scenario_name=scenario_name,
        csv_path=train_csv_path,
        image_subdir=train_config.image_subdir,
    )
    gps_stats = train_ds.get_gps_stats()
    test_ds = MultimodalDataset(
        data_root=train_config.data_root,
        split_root=train_config.split_root,
        scenario_name=scenario_name,
        csv_path=test_csv_path,
        image_subdir=train_config.image_subdir,
        gps_stats=gps_stats,
    )

    train_loader = build_dataloader(train_ds, train_config, shuffle=True)
    test_loader = build_dataloader(test_ds, train_config, shuffle=False)

    model = BeMambaModel(model_config).to(device)
    criterion = build_criterion(train_config, alpha_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
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
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=(train_config.amp and device.type == "cuda"),
            grad_clip_norm=train_config.grad_clip_norm,
        )
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
            torch.save(model.state_dict(), best_ckpt_path)
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

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
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
    parser.add_argument("--image-subdir", default="camera_data")
    parser.add_argument("--scenarios", nargs="+", default=["scenario32", "scenario33", "scenario34"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", choices=["loss", "acc1", "acc2", "acc3", "dba", "apl"], default="acc3")
    parser.add_argument("--early-stop-mode", choices=["max", "min"], default="max")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
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
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gps-hidden-dim", type=int, default=96)
    parser.add_argument("--no-pretrained-backbones", action="store_true")
    parser.add_argument("--freeze-image-stem", action="store_true")
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> Tuple[TrainConfig, BeMambaConfig]:
    train_config = TrainConfig(
        data_root=args.data_root,
        split_root=args.split_root,
        output_root=args.output_root,
        image_subdir=args.image_subdir,
        scenarios=tuple(args.scenarios),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_mode=args.early_stop_mode,
        min_delta=args.min_delta,
        gamma=args.gamma,
        loss_name=args.loss,
        amp=(not args.no_amp),
        seed=args.seed,
        pin_memory=(not args.no_pin_memory),
        persistent_workers=(not args.no_persistent_workers),
        save_every_epoch=args.save_every_epoch,
        device=args.device,
    )
    model_config = BeMambaConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        patch_grid=args.patch_grid,
        temporal_layers=args.temporal_layers,
        fusion_layers=args.fusion_layers,
        dropout=args.dropout,
        gps_hidden_dim=args.gps_hidden_dim,
        pretrained_backbones=(not args.no_pretrained_backbones),
        freeze_image_stem=args.freeze_image_stem,
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
