"""Evaluate a saved checkpoint under missing-data robustness protocols."""
import argparse
import os
import torch
import torch.nn as nn

from src.dataset import MultimodalDataset
from src.model import BeMambaConfig, BeMambaModel, load_checkpoint
from src.utils import calculate_apl, calculate_dba_score, calculate_topk_accuracy
from train import (
    TrainConfig,
    build_criterion,
    build_dataloader,
    move_batch_to_device,
    run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Missing robustness evaluation for a saved checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pth or other checkpoint")
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--split-root", default="./Data/splits_paper80")
    parser.add_argument("--scenario", default="scenario32")
    parser.add_argument("--image-subdir", default="camera_data_mask")
    parser.add_argument("--lidar-representation", choices=["binary", "count"], default="count")
    parser.add_argument("--gps-feature-mode", choices=["bemamba", "physical_kinematic"], default="bemamba")
    parser.add_argument("--gps-future-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--loss", choices=["ce", "focal", "power_soft_ce"], default="power_soft_ce")
    parser.add_argument("--soft-power-temperature", type=float, default=0.15)
    parser.add_argument("--hard-loss-weight", type=float, default=0.6)
    parser.add_argument("--missing-frame-prob", type=float, default=0.2)
    parser.add_argument("--missing-burst-prob", type=float, default=0.1)
    parser.add_argument("--missing-modality-prob", type=float, default=0.1)
    parser.add_argument("--missing-modalities", default="img,radar,lidar,gps")
    parser.add_argument("--missing-seed", type=int, default=42)
    parser.add_argument("--no-dmaf", action="store_true",
                        help="Evaluate as baseline model (no mask injection, no cross-attention)")
    # ── Test protocol configuration ──
    parser.add_argument("--test-frame-probs", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5],
                        help="Frame missing probabilities for testing (default: 0.1 0.3 0.5)")
    parser.add_argument("--test-burst-probs", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6],
                        help="Burst missing probabilities for testing (default: 0.2 0.4 0.6)")
    parser.add_argument("--test-modality-probs", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6],
                        help="Modality missing probabilities for testing (default: 0.2 0.4 0.6)")
    parser.add_argument("--test-burst-min", type=int, default=2)
    parser.add_argument("--test-burst-max", type=int, default=3)
    parser.add_argument("--test-modality-min", type=int, default=1)
    parser.add_argument("--test-modality-max", type=int, default=1)
    parser.add_argument("--skip-hybrid", action="store_true",
                        help="Skip the hybrid (training config) protocol")
    parser.add_argument("--no-mask-embed", action="store_true",
                        help="Disable MaskEncoder (mask embedding injection)")
    parser.add_argument("--no-cross-attn", action="store_true",
                        help="Disable CrossModalFusion (direct cross-modal attention)")
    parser.add_argument("--no-reliability", action="store_true",
                        help="Disable ReliabilityEstimator (per-modality reliability weighting)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--temporal-order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--spatial-scan", choices=["vertical", "row"], default="vertical")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--model-variant", choices=["bemamba", "clean_plus", "clean_plus_v2", "clean_plus_v3", "clean_plus_v4", "clean_plus_v5", "clean_plus_v6", "clean_plus_v7", "clean_plus_v8", "clean_plus_v9", "clean_plus_v10", "clean_plus_v11", "clean_plus_v12", "clean_plus_v13", "clean_plus_v14", "clean_plus_v15"], default="bemamba")
    parser.add_argument("--backbone-stage", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--clean-cross-attn", action="store_true")
    parser.add_argument("--spatial-mixer-layers", type=int, default=None)
    parser.add_argument("--order-gate", action="store_true")
    parser.add_argument("--attn-head", action="store_true")
    parser.add_argument("--branch-ensemble", action="store_true")
    parser.add_argument("--candidate-rerank-delta-bound", type=float, default=0.20)
    parser.add_argument("--candidate-embed-dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clean_plus = args.model_variant == "clean_plus"
    clean_plus_v2 = args.model_variant == "clean_plus_v2"
    clean_plus_v3 = args.model_variant == "clean_plus_v3"
    clean_plus_v4 = args.model_variant == "clean_plus_v4"
    clean_plus_v5 = args.model_variant == "clean_plus_v5"
    clean_plus_v6 = args.model_variant == "clean_plus_v6"
    clean_plus_v7 = args.model_variant == "clean_plus_v7"
    clean_plus_v8 = args.model_variant == "clean_plus_v8"
    clean_plus_v9 = args.model_variant == "clean_plus_v9"
    clean_plus_v10 = args.model_variant == "clean_plus_v10"
    clean_plus_v11 = args.model_variant == "clean_plus_v11"
    clean_plus_v12 = args.model_variant == "clean_plus_v12"
    clean_plus_v13 = args.model_variant == "clean_plus_v13"
    clean_plus_v14 = args.model_variant == "clean_plus_v14"
    clean_plus_v15 = args.model_variant == "clean_plus_v15"
    backbone_stage = args.backbone_stage
    if backbone_stage is None:
        backbone_stage = 3 if (clean_plus_v4 or clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15) else 2
    spatial_mixer_layers = args.spatial_mixer_layers
    if spatial_mixer_layers is None:
        spatial_mixer_layers = 1 if clean_plus else 0
    if spatial_mixer_layers < 0:
        raise ValueError("--spatial-mixer-layers must be >= 0")
    clean_cross_attn = args.clean_cross_attn or clean_plus
    use_order_gate = args.order_gate or clean_plus
    use_attn_head = args.attn_head or clean_plus
    use_branch_ensemble = args.branch_ensemble or clean_plus_v2 or clean_plus_v3 or clean_plus_v4 or clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15

    model_config = BeMambaConfig(
        d_model=args.d_model,
        temporal_layers=args.temporal_layers,
        fusion_layers=args.fusion_layers,
        temporal_order=args.temporal_order,
        spatial_scan=args.spatial_scan,
        dropout=args.dropout,
        gps_input_dim=15 if args.gps_feature_mode == "physical_kinematic" else 2,
        backbone_stage=backbone_stage,
        missing_enabled=not args.no_dmaf,
        use_mask_embed=not args.no_mask_embed,
        use_cross_attn=not args.no_cross_attn,
        use_reliability=not args.no_reliability,
        model_variant=args.model_variant,
        clean_cross_attn=clean_cross_attn,
        spatial_mixer_layers=spatial_mixer_layers,
        use_order_gate=use_order_gate,
        use_attn_head=use_attn_head,
        use_branch_ensemble=use_branch_ensemble,
        use_beam_query_head=clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15,
        use_multiscale_backbone=clean_plus_v6,
        use_ordinal_head=clean_plus_v7,
        use_temporal_attn_pool=clean_plus_v8,
        use_beam_neighbor_head=clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15,
        use_candidate_reranker=clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13 or clean_plus_v14 or clean_plus_v15,
        use_bounded_candidate_reranker=clean_plus_v12,
        candidate_topk=7,
        candidate_delta_bound=args.candidate_rerank_delta_bound,
        candidate_embed_dropout=args.candidate_embed_dropout,
        return_aux_logits=False,
    )

    train_config = TrainConfig(
        data_root=args.data_root,
        split_root=args.split_root,
        image_subdir=args.image_subdir,
        lidar_representation=args.lidar_representation,
        gps_feature_mode=args.gps_feature_mode,
        gps_future_steps=args.gps_future_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss_name=args.loss,
        soft_power_temperature=args.soft_power_temperature,
        hard_loss_weight=args.hard_loss_weight,
        missing_enabled=False,
        missing_aug_enabled=True,
        dmaf_enabled=not args.no_dmaf,
        missing_frame_prob=args.missing_frame_prob,
        missing_burst_prob=args.missing_burst_prob,
        missing_modality_prob=args.missing_modality_prob,
        missing_modalities=args.missing_modalities,
        missing_seed=args.missing_seed,
        seed=args.seed,
        model_variant=args.model_variant,
        clean_cross_attn=clean_cross_attn,
        spatial_mixer_layers=spatial_mixer_layers,
        use_order_gate=use_order_gate,
        use_attn_head=use_attn_head,
        use_branch_ensemble=use_branch_ensemble,
        backbone_stage=backbone_stage,
    )

    print(f"Loading checkpoint: {args.ckpt}")
    model = BeMambaModel(model_config).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    train_csv_path = os.path.join(args.split_root, f"{args.scenario}_train.csv")
    temp_ds = MultimodalDataset(
        mode="train",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=train_csv_path,
        image_subdir=args.image_subdir,
        lidar_representation=args.lidar_representation,
        gps_feature_mode=args.gps_feature_mode,
        gps_future_steps=args.gps_future_steps,
    )
    gps_stats = temp_ds.get_gps_stats()

    test_csv_path = os.path.join(args.split_root, f"{args.scenario}_test.csv")
    criterion = build_criterion(train_config, None)

    # Build test protocols dynamically from CLI args
    # Protocol format: (name, frame_prob, burst_prob, modality_prob,
    #                    burst_min, burst_max, mod_min, mod_max)
    test_protocols = [
        ("clean", 0.0, 0.0, 0.0, None, None, None, None),
    ]
    for p in args.test_frame_probs:
        test_protocols.append((f"frame_p{int(p*100):02d}", p, 0.0, 0.0, None, None, None, None))
    for p in args.test_burst_probs:
        test_protocols.append((f"burst_p{int(p*100):02d}", 0.0, p, 0.0,
                                args.test_burst_min, args.test_burst_max, None, None))
    for p in args.test_modality_probs:
        test_protocols.append((f"modal_p{int(p*100):02d}", 0.0, 0.0, p,
                                None, None, args.test_modality_min, args.test_modality_max))
    if not args.skip_hybrid:
        test_protocols.append(
            ("hybrid",
             args.missing_frame_prob,
             args.missing_burst_prob,
             args.missing_modality_prob,
             None, None, None, None)
        )

    test_seed = args.missing_seed + 10000
    ckpt_dir = os.path.dirname(os.path.dirname(args.ckpt))
    if not ckpt_dir:
        ckpt_dir = "."

    print(f"\n{'='*60}")
    print(f"  Missing Robustness Tests — {args.scenario}")
    print(f"{'='*60}")
    print(f"{'Protocol':<14} {'Top-1':>7} {'Top-2':>7} {'Top-3':>7} {'DBA':>8} {'APL':>9} {'Ret':>7}")
    print("-" * 62)

    clean_acc3 = None
    results = []

    for name, fp, bp, mp, b_min, b_max, m_min, m_max in test_protocols:
        has_missing = name != "clean"
        ds = MultimodalDataset(
            mode="test",
            data_root=args.data_root,
            split_root=args.split_root,
            scenario_name=args.scenario,
            csv_path=test_csv_path,
            image_subdir=args.image_subdir,
            gps_stats=gps_stats,
            lidar_representation=args.lidar_representation,
            gps_feature_mode=args.gps_feature_mode,
            gps_future_steps=args.gps_future_steps,
            missing_enabled=has_missing,
            return_missing_masks=(has_missing and not args.no_dmaf),
            missing_frame_prob=fp,
            missing_burst_prob=bp,
            missing_burst_min=b_min if b_min is not None else 2,
            missing_burst_max=b_max if b_max is not None else 3,
            missing_modality_prob=mp,
            missing_modality_min=m_min if m_min is not None else 1,
            missing_modality_max=m_max if m_max is not None else 2,
            missing_modalities=args.missing_modalities,
            missing_seed=test_seed,
        )
        loader = build_dataloader(ds, train_config, shuffle=False)
        metrics = run_epoch(model, loader, criterion, None, None, device, False, 0.0)

        if name == "clean":
            clean_acc3 = metrics["acc3"]

        retention = (metrics["acc3"] / clean_acc3 * 100) if clean_acc3 and clean_acc3 > 0 else 0.0
        results.append((name, metrics, retention))

        print(
            f"{name:<14} {metrics['acc1']:>6.2f}% {metrics['acc2']:>6.2f}% "
            f"{metrics['acc3']:>6.2f}% {metrics['dba']:>7.4f} {metrics['apl']:>8.4f} dB "
            f"{retention:>6.1f}%"
        )

        del loader, ds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_path = os.path.join(ckpt_dir, "missing_test_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Missing Robustness Test Results ({args.scenario})\n")
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
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    main()
