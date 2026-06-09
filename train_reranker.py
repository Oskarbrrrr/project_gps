import argparse
import csv
import datetime
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import MultimodalDataset
from src.model import BeMambaConfig, BeMambaModel, load_checkpoint
from src.reranker import TwoStageCandidateReranker
from src.utils import calculate_apl, calculate_dba_score, calculate_topk_accuracy
from train import TrainConfig, build_combined_train_csv, build_dataloader, move_batch_to_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_stage1_config(args: argparse.Namespace) -> BeMambaConfig:
    variant = args.stage1_model_variant
    clean_plus = variant == "clean_plus"
    clean_plus_v2 = variant == "clean_plus_v2"
    clean_plus_v3 = variant == "clean_plus_v3"
    clean_plus_v4 = variant == "clean_plus_v4"
    clean_plus_v5 = variant == "clean_plus_v5"
    clean_plus_v6 = variant == "clean_plus_v6"
    clean_plus_v7 = variant == "clean_plus_v7"
    clean_plus_v8 = variant == "clean_plus_v8"
    clean_plus_v9 = variant == "clean_plus_v9"
    clean_plus_v10 = variant == "clean_plus_v10"
    clean_plus_v11 = variant == "clean_plus_v11"
    clean_plus_v12 = variant == "clean_plus_v12"
    clean_plus_v13 = variant == "clean_plus_v13"
    clean_plus_v14 = variant == "clean_plus_v14"

    stage3_family = (
        clean_plus_v4
        or clean_plus_v5
        or clean_plus_v6
        or clean_plus_v7
        or clean_plus_v8
        or clean_plus_v9
        or clean_plus_v10
        or clean_plus_v11
        or clean_plus_v12
        or clean_plus_v13
        or clean_plus_v14
    )
    backbone_stage = args.backbone_stage if args.backbone_stage is not None else (3 if stage3_family else 2)
    spatial_mixer_layers = args.spatial_mixer_layers
    if spatial_mixer_layers is None:
        spatial_mixer_layers = 1 if clean_plus else 0

    return BeMambaConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        patch_grid=args.patch_grid,
        temporal_layers=args.temporal_layers,
        fusion_layers=args.fusion_layers,
        temporal_order=args.temporal_order,
        spatial_scan=args.spatial_scan,
        dropout=args.stage1_dropout,
        gps_hidden_dim=args.gps_hidden_dim,
        backbone_stage=backbone_stage,
        pretrained_backbones=False,
        missing_enabled=False,
        model_variant=variant,
        clean_cross_attn=args.clean_cross_attn or clean_plus,
        spatial_mixer_layers=spatial_mixer_layers,
        use_order_gate=args.order_gate or clean_plus,
        use_attn_head=args.attn_head or clean_plus,
        use_branch_ensemble=(
            args.branch_ensemble
            or clean_plus_v2
            or clean_plus_v3
            or clean_plus_v4
            or clean_plus_v5
            or clean_plus_v6
            or clean_plus_v7
            or clean_plus_v8
            or clean_plus_v9
            or clean_plus_v10
            or clean_plus_v11
            or clean_plus_v12
            or clean_plus_v13
            or clean_plus_v14
        ),
        use_beam_query_head=(
            clean_plus_v5
            or clean_plus_v6
            or clean_plus_v7
            or clean_plus_v8
            or clean_plus_v9
            or clean_plus_v10
            or clean_plus_v11
            or clean_plus_v12
            or clean_plus_v13
            or clean_plus_v14
        ),
        use_multiscale_backbone=clean_plus_v6,
        use_ordinal_head=clean_plus_v7,
        use_temporal_attn_pool=clean_plus_v8,
        use_beam_neighbor_head=clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14,
        use_candidate_reranker=clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14,
        use_bounded_candidate_reranker=clean_plus_v12,
        candidate_topk=args.stage1_candidate_topk,
        candidate_delta_bound=args.stage1_candidate_delta_bound,
        candidate_embed_dropout=args.stage1_candidate_embed_dropout,
        return_aux_logits=False,
    )


def extract_stage1_outputs(
    stage1_model: BeMambaModel,
    imgs: torch.Tensor,
    radars: torch.Tensor,
    lidars: torch.Tensor,
    gps: torch.Tensor,
    img_mask: torch.Tensor | None,
    radar_mask: torch.Tensor | None,
    lidar_mask: torch.Tensor | None,
    gps_mask: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model_outputs, features = stage1_model(
        imgs,
        radars,
        lidars,
        gps,
        img_mask,
        radar_mask,
        lidar_mask,
        gps_mask,
        return_features=True,
    )
    if isinstance(model_outputs, tuple):
        model_outputs = model_outputs[0]
    return model_outputs.detach(), features["fused_tokens"].detach()


def two_stage_rerank_loss(
    candidate_logits: torch.Tensor,
    candidate_idx: torch.Tensor,
    stage1_candidate_logits: torch.Tensor,
    targets: torch.Tensor,
    power_vec: torch.Tensor,
    hard_weight: float,
    power_weight: float,
    power_temperature: float,
    top3_margin_weight: float,
    top3_margin: float,
    recoverable_only: bool,
    preserve_weight: float,
    delta_reg_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = {}
    total = candidate_logits.new_zeros(())

    target_match = candidate_idx == targets.unsqueeze(1)
    target_in_candidates = target_match.any(dim=1)
    target_pos = target_match.float().argmax(dim=1)
    stage1_target_in_top3 = target_in_candidates & (target_pos < 3)
    stage1_recoverable = target_in_candidates & (~stage1_target_in_top3)
    supervised_mask = stage1_recoverable if recoverable_only else target_in_candidates

    hard_loss = candidate_logits.new_zeros(())
    if hard_weight > 0 and torch.any(supervised_mask):
        hard_loss = F.cross_entropy(
            candidate_logits[supervised_mask],
            target_pos[supervised_mask],
        )
        total = total + hard_weight * hard_loss
    losses["hard_loss"] = float(hard_loss.detach().item())

    power_loss = candidate_logits.new_zeros(())
    if power_weight > 0:
        power_vec = torch.nan_to_num(power_vec.to(candidate_logits.device), nan=0.0, posinf=0.0, neginf=0.0)
        power_vec = torch.clamp(power_vec, min=0.0)
        candidate_power = power_vec.gather(1, candidate_idx)
        valid_power = candidate_power.sum(dim=1) > 0
        if torch.any(valid_power):
            soft_logits = torch.log(candidate_power + 1e-8) / max(power_temperature, 1e-8)
            soft_targets = torch.softmax(soft_logits, dim=1)
            log_probs = F.log_softmax(candidate_logits, dim=1)
            power_loss = -(soft_targets * log_probs).sum(dim=1)
            power_loss = power_loss[valid_power].mean()
            total = total + power_weight * power_loss
    losses["power_loss"] = float(power_loss.detach().item())

    top3_margin_loss = candidate_logits.new_zeros(())
    if top3_margin_weight > 0 and torch.any(stage1_recoverable):
        current_top3 = candidate_logits.detach().topk(min(3, candidate_logits.size(1)), dim=1).indices
        target_in_top3 = (current_top3 == target_pos.unsqueeze(1)).any(dim=1)
        recoverable = stage1_recoverable & (~target_in_top3)
        if torch.any(recoverable):
            third_scores = candidate_logits.gather(1, current_top3[:, -1:].to(candidate_logits.device)).squeeze(1)
            target_scores = candidate_logits.gather(1, target_pos.unsqueeze(1)).squeeze(1)
            top3_margin_loss = F.relu(third_scores + top3_margin - target_scores)
            top3_margin_loss = top3_margin_loss[recoverable].mean()
            total = total + top3_margin_weight * top3_margin_loss
    losses["top3_margin_loss"] = float(top3_margin_loss.detach().item())

    preserve_loss = candidate_logits.new_zeros(())
    if preserve_weight > 0:
        stage1_probs = F.softmax(stage1_candidate_logits.detach(), dim=1)
        log_probs = F.log_softmax(candidate_logits, dim=1)
        preserve_loss = F.kl_div(log_probs, stage1_probs, reduction="batchmean")
        total = total + preserve_weight * preserve_loss
    losses["preserve_loss"] = float(preserve_loss.detach().item())

    delta_reg_loss = candidate_logits.new_zeros(())
    if delta_reg_weight > 0:
        delta_reg_loss = (candidate_logits - stage1_candidate_logits.detach()).square().mean()
        total = total + delta_reg_weight * delta_reg_loss
    losses["delta_reg_loss"] = float(delta_reg_loss.detach().item())

    losses["target_in_candidates"] = float(target_in_candidates.float().mean().detach().item() * 100.0)
    losses["stage1_recoverable"] = float(stage1_recoverable.float().mean().detach().item() * 100.0)
    losses["loss"] = float(total.detach().item())
    return total, losses


def run_epoch(
    stage1_model: BeMambaModel,
    reranker: TwoStageCandidateReranker,
    loader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
) -> Dict[str, float]:
    training = optimizer is not None
    stage1_model.eval()
    reranker.train(training)

    totals = {
        "loss": 0.0,
        "hard_loss": 0.0,
        "power_loss": 0.0,
        "top3_margin_loss": 0.0,
        "preserve_loss": 0.0,
        "delta_reg_loss": 0.0,
        "stage1_acc1": 0.0,
        "stage1_acc2": 0.0,
        "stage1_acc3": 0.0,
        "rerank_acc1": 0.0,
        "rerank_acc2": 0.0,
        "rerank_acc3": 0.0,
        "rerank_dba": 0.0,
        "rerank_apl": 0.0,
        "candidate_coverage": 0.0,
        "stage1_recoverable": 0.0,
    }
    sample_total = 0

    for batch in loader:
        imgs, radars, lidars, gps, targets, power_vec, img_mask, radar_mask, lidar_mask, gps_mask = move_batch_to_device(batch, device)
        batch_size = targets.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(enabled=amp_enabled):
                with torch.no_grad():
                    stage1_logits, fused_tokens = extract_stage1_outputs(
                        stage1_model,
                        imgs,
                        radars,
                        lidars,
                        gps,
                        img_mask,
                        radar_mask,
                        lidar_mask,
                        gps_mask,
                    )
                rerank_output = reranker(stage1_logits, fused_tokens)
                loss, loss_items = two_stage_rerank_loss(
                    rerank_output.candidate_logits,
                    rerank_output.candidate_idx,
                    rerank_output.stage1_candidate_logits,
                    targets,
                    power_vec,
                    hard_weight=args.hard_weight,
                    power_weight=args.power_weight,
                    power_temperature=args.power_temperature,
                    top3_margin_weight=args.top3_margin_weight,
                    top3_margin=args.top3_margin,
                    recoverable_only=args.recoverable_only,
                    preserve_weight=args.preserve_weight,
                    delta_reg_weight=args.delta_reg_weight,
                )

        if training:
            assert optimizer is not None
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), args.grad_clip_norm)
                optimizer.step()

        sample_total += batch_size
        for key in ("loss", "hard_loss", "power_loss", "top3_margin_loss", "preserve_loss", "delta_reg_loss"):
            totals[key] += loss_items[key] * batch_size
        totals["candidate_coverage"] += loss_items["target_in_candidates"] * batch_size
        totals["stage1_recoverable"] += loss_items["stage1_recoverable"] * batch_size

        stage1_acc1, stage1_acc2, stage1_acc3 = calculate_topk_accuracy(stage1_logits, targets, topk=(1, 2, 3))
        rerank_acc1, rerank_acc2, rerank_acc3 = calculate_topk_accuracy(rerank_output.full_logits, targets, topk=(1, 2, 3))
        totals["stage1_acc1"] += stage1_acc1 * batch_size
        totals["stage1_acc2"] += stage1_acc2 * batch_size
        totals["stage1_acc3"] += stage1_acc3 * batch_size
        totals["rerank_acc1"] += rerank_acc1 * batch_size
        totals["rerank_acc2"] += rerank_acc2 * batch_size
        totals["rerank_acc3"] += rerank_acc3 * batch_size
        totals["rerank_dba"] += calculate_dba_score(rerank_output.full_logits, targets) * batch_size
        totals["rerank_apl"] += calculate_apl(rerank_output.full_logits, power_vec) * batch_size

    return {key: value / max(sample_total, 1) for key, value in totals.items()}


def save_args(path: str, args: argparse.Namespace, stage1_config: BeMambaConfig) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("[reranker]\n")
        for key, value in vars(args).items():
            handle.write(f"{key}={value}\n")
        handle.write("\n[stage1_model]\n")
        for key, value in stage1_config.__dict__.items():
            handle.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    variants = [
        "bemamba",
        "clean_plus",
        "clean_plus_v2",
        "clean_plus_v3",
        "clean_plus_v4",
        "clean_plus_v5",
        "clean_plus_v6",
        "clean_plus_v7",
        "clean_plus_v8",
        "clean_plus_v9",
        "clean_plus_v10",
        "clean_plus_v11",
        "clean_plus_v12",
        "clean_plus_v13",
        "clean_plus_v14",
    ]
    parser = argparse.ArgumentParser(description="Train a standalone two-stage beam candidate reranker.")
    parser.add_argument("--stage1-ckpt", required=True, help="Frozen Stage-1 best_model.pth path")
    parser.add_argument("--stage1-model-variant", choices=variants, default="clean_plus_v10")
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--split-root", default="./Data/splits_paper80")
    parser.add_argument("--output-root", default="./outputs_two_stage_reranker")
    parser.add_argument("--scenario", default="scenario32")
    parser.add_argument("--no-merge-trainval", action="store_true")
    parser.add_argument("--image-subdir", default="camera_data_mask")
    parser.add_argument("--image-aug", action="store_true")
    parser.add_argument("--lidar-representation", choices=["binary", "count"], default="count")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--patch-grid", type=int, default=6)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--temporal-order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--spatial-scan", choices=["vertical", "row"], default="vertical")
    parser.add_argument("--stage1-dropout", type=float, default=0.25)
    parser.add_argument("--gps-hidden-dim", type=int, default=96)
    parser.add_argument("--backbone-stage", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--clean-cross-attn", action="store_true")
    parser.add_argument("--spatial-mixer-layers", type=int, default=None)
    parser.add_argument("--order-gate", action="store_true")
    parser.add_argument("--attn-head", action="store_true")
    parser.add_argument("--branch-ensemble", action="store_true")
    parser.add_argument("--stage1-candidate-topk", type=int, default=7)
    parser.add_argument("--stage1-candidate-delta-bound", type=float, default=0.20)
    parser.add_argument("--stage1-candidate-embed-dropout", type=float, default=0.0)

    parser.add_argument("--candidate-topk", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--rerank-layers", type=int, default=2)
    parser.add_argument("--rerank-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--embed-dropout", type=float, default=0.10)
    parser.add_argument("--keep-full-logits", action="store_true",
                        help="Keep non-candidate Stage-1 logits instead of restricting prediction to Top-K candidates")
    parser.add_argument("--hard-weight", type=float, default=0.7)
    parser.add_argument("--power-weight", type=float, default=0.3)
    parser.add_argument("--power-temperature", type=float, default=0.15)
    parser.add_argument("--top3-margin-weight", type=float, default=0.03)
    parser.add_argument("--top3-margin", type=float, default=0.05)
    parser.add_argument("--recoverable-only", action="store_true",
                        help="Apply hard supervision only to Stage-1 Top-K misses that contain the target outside Top-3")
    parser.add_argument("--preserve-weight", type=float, default=0.0,
                        help="KL penalty that preserves Stage-1 candidate distribution")
    parser.add_argument("--delta-reg-weight", type=float, default=0.0,
                        help="L2 penalty on reranker candidate-logit deltas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_topk < 3:
        raise ValueError("--candidate-topk must be >= 3")
    if (
        args.hard_weight < 0
        or args.power_weight < 0
        or args.top3_margin_weight < 0
        or args.preserve_weight < 0
        or args.delta_reg_weight < 0
    ):
        raise ValueError("loss weights must be >= 0")
    if args.power_temperature <= 0:
        raise ValueError("--power-temperature must be > 0")

    set_seed(args.seed)
    device = prepare_device(args.device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, args.scenario, timestamp)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_csv_path = build_combined_train_csv(
        args.split_root,
        args.scenario,
        run_dir,
        merge_train_val=(not args.no_merge_trainval),
    )
    train_ds = MultimodalDataset(
        mode="train",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=train_csv_path,
        image_subdir=args.image_subdir,
        image_aug=args.image_aug,
        lidar_representation=args.lidar_representation,
    )
    gps_stats = train_ds.get_gps_stats()
    test_ds = MultimodalDataset(
        mode="test",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=os.path.join(args.split_root, f"{args.scenario}_test.csv"),
        image_subdir=args.image_subdir,
        gps_stats=gps_stats,
        lidar_representation=args.lidar_representation,
    )
    loader_config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        persistent_workers=(not args.no_persistent_workers),
    )
    train_loader = build_dataloader(train_ds, loader_config, shuffle=True)
    test_loader = build_dataloader(test_ds, loader_config, shuffle=False)

    stage1_config = build_stage1_config(args)
    stage1_model = BeMambaModel(stage1_config).to(device)
    load_checkpoint(stage1_model, args.stage1_ckpt, device)
    stage1_model.eval()
    for param in stage1_model.parameters():
        param.requires_grad_(False)

    reranker = TwoStageCandidateReranker(
        d_model=args.d_model,
        num_classes=stage1_config.num_classes,
        topk=args.candidate_topk,
        hidden_dim=args.hidden_dim,
        num_layers=args.rerank_layers,
        num_heads=args.rerank_heads,
        dropout=args.dropout,
        embed_dropout=args.embed_dropout,
        restrict_to_candidates=(not args.keep_full_logits),
    ).to(device)

    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = GradScaler(enabled=((not args.no_amp) and device.type == "cuda"))
    save_args(os.path.join(run_dir, "config.txt"), args, stage1_config)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc3",
                "test_loss",
                "preserve_loss",
                "delta_reg_loss",
                "stage1_acc3",
                "rerank_acc1",
                "rerank_acc2",
                "rerank_acc3",
                "candidate_coverage",
                "stage1_recoverable",
                "rerank_dba",
                "rerank_apl",
                "is_best",
            ]
        )

    best_acc3 = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_ckpt = os.path.join(checkpoints_dir, "best_reranker.pth")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            stage1_model,
            reranker,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled=((not args.no_amp) and device.type == "cuda"),
            args=args,
        )
        test_metrics = run_epoch(
            stage1_model,
            reranker,
            test_loader,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=((not args.no_amp) and device.type == "cuda"),
            args=args,
        )
        scheduler.step()

        is_best = best_acc3 is None or test_metrics["rerank_acc3"] > best_acc3
        if is_best:
            best_acc3 = test_metrics["rerank_acc3"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": reranker.state_dict(),
                    "args": vars(args),
                    "stage1_config": stage1_config.__dict__,
                    "best_epoch": best_epoch,
                    "best_acc3": best_acc3,
                },
                best_ckpt,
            )
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        eta_seconds = int((elapsed / epoch) * (args.epochs - epoch))
        eta_string = str(datetime.timedelta(seconds=eta_seconds))
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} | train_acc3={train_metrics['rerank_acc3']:.2f}% | "
            f"stage1_acc3={test_metrics['stage1_acc3']:.2f}% | "
            f"test_acc3={test_metrics['rerank_acc3']:.2f}% | "
            f"coverage={test_metrics['candidate_coverage']:.2f}% | "
            f"recoverable={test_metrics['stage1_recoverable']:.2f}% | "
            f"test_dba={test_metrics['rerank_dba']:.4f} | "
            f"test_apl={test_metrics['rerank_apl']:.4f} dB | ETA {eta_string}"
        )
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    f"{current_lr:.8e}",
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['rerank_acc3']:.4f}",
                    f"{test_metrics['loss']:.6f}",
                    f"{test_metrics['preserve_loss']:.6f}",
                    f"{test_metrics['delta_reg_loss']:.6f}",
                    f"{test_metrics['stage1_acc3']:.4f}",
                    f"{test_metrics['rerank_acc1']:.4f}",
                    f"{test_metrics['rerank_acc2']:.4f}",
                    f"{test_metrics['rerank_acc3']:.4f}",
                    f"{test_metrics['candidate_coverage']:.4f}",
                    f"{test_metrics['stage1_recoverable']:.4f}",
                    f"{test_metrics['rerank_dba']:.6f}",
                    f"{test_metrics['rerank_apl']:.6f}",
                    int(is_best),
                ]
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch:03d} "
                f"(best_epoch={best_epoch:03d}, best_acc3={best_acc3:.4f})"
            )
            break

    checkpoint = torch.load(best_ckpt, map_location=device)
    reranker.load_state_dict(checkpoint["model"], strict=True)
    final_metrics = run_epoch(
        stage1_model,
        reranker,
        test_loader,
        optimizer=None,
        scaler=None,
        device=device,
        amp_enabled=((not args.no_amp) and device.type == "cuda"),
        args=args,
    )
    result_text = (
        f"[Best Two-Stage Reranker Results for {args.scenario}]\n"
        f"best_epoch: {best_epoch}\n"
        f"stage1_top3_acc: {final_metrics['stage1_acc3']:.2f}%\n"
        f"rerank_top1_acc: {final_metrics['rerank_acc1']:.2f}%\n"
        f"rerank_top2_acc: {final_metrics['rerank_acc2']:.2f}%\n"
        f"rerank_top3_acc: {final_metrics['rerank_acc3']:.2f}%\n"
        f"candidate_coverage: {final_metrics['candidate_coverage']:.2f}%\n"
        f"stage1_recoverable: {final_metrics['stage1_recoverable']:.2f}%\n"
        f"rerank_dba: {final_metrics['rerank_dba']:.4f}\n"
        f"rerank_apl: {final_metrics['rerank_apl']:.4f} dB\n"
    )
    print("\n" + result_text)
    with open(os.path.join(run_dir, "final_test_result.txt"), "w", encoding="utf-8") as handle:
        handle.write(result_text)


if __name__ == "__main__":
    main()
