"""Evaluate a clean-data logits ensemble from multiple checkpoints."""
import argparse
import os

import torch

from src.dataset import MultimodalDataset
from src.model import BeMambaConfig, BeMambaModel, load_checkpoint
from src.utils import calculate_apl, calculate_dba_score, calculate_topk_accuracy
from train import TrainConfig, build_dataloader, move_batch_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Clean test evaluation for a checkpoint ensemble.")
    parser.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths to ensemble")
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--split-root", default="./Data/splits_paper80")
    parser.add_argument("--scenario", default="scenario32")
    parser.add_argument("--image-subdir", default="camera_data_mask")
    parser.add_argument("--lidar-representation", choices=["binary", "count"], default="count")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--model-variant", choices=["bemamba", "clean_plus", "clean_plus_v2", "clean_plus_v3", "clean_plus_v4", "clean_plus_v5", "clean_plus_v6", "clean_plus_v7", "clean_plus_v8", "clean_plus_v9", "clean_plus_v10", "clean_plus_v11", "clean_plus_v12", "clean_plus_v13"], default="bemamba")
    parser.add_argument("--backbone-stage", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--patch-grid", type=int, default=6)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--temporal-order", default="reverse")
    parser.add_argument("--spatial-scan", default="row")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gps-hidden-dim", type=int, default=96)
    parser.add_argument("--clean-cross-attn", action="store_true")
    parser.add_argument("--spatial-mixer-layers", type=int, default=None)
    parser.add_argument("--order-gate", action="store_true")
    parser.add_argument("--attn-head", action="store_true")
    parser.add_argument("--branch-ensemble", action="store_true")
    parser.add_argument("--candidate-rerank-delta-bound", type=float, default=0.20)
    parser.add_argument("--candidate-embed-dropout", type=float, default=0.0)
    parser.add_argument("--method", choices=["logits", "prob", "rank_vote", "topk_vote"], default="logits")
    parser.add_argument("--vote-topk", type=int, default=10)
    return parser.parse_args()


def build_model_config(args):
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

    backbone_stage = args.backbone_stage
    if backbone_stage is None:
        backbone_stage = 3 if (clean_plus_v4 or clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13) else 2

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
        dropout=args.dropout,
        gps_hidden_dim=args.gps_hidden_dim,
        backbone_stage=backbone_stage,
        pretrained_backbones=False,
        missing_enabled=False,
        model_variant=args.model_variant,
        clean_cross_attn=args.clean_cross_attn or clean_plus,
        spatial_mixer_layers=spatial_mixer_layers,
        use_order_gate=args.order_gate or clean_plus,
        use_attn_head=args.attn_head or clean_plus,
        use_branch_ensemble=args.branch_ensemble or clean_plus_v2 or clean_plus_v3 or clean_plus_v4 or clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13,
        use_beam_query_head=clean_plus_v5 or clean_plus_v6 or clean_plus_v7 or clean_plus_v8 or clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13,
        use_multiscale_backbone=clean_plus_v6,
        use_ordinal_head=clean_plus_v7,
        use_temporal_attn_pool=clean_plus_v8,
        use_beam_neighbor_head=clean_plus_v9 or clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13,
        use_candidate_reranker=clean_plus_v10 or clean_plus_v11 or clean_plus_v12 or clean_plus_v13,
        use_bounded_candidate_reranker=clean_plus_v12,
        candidate_topk=7,
        candidate_delta_bound=args.candidate_rerank_delta_bound,
        candidate_embed_dropout=args.candidate_embed_dropout,
        return_aux_logits=False,
    )


def combine_ensemble_outputs(logits_list, method: str, vote_topk: int):
    stacked = torch.stack(logits_list, dim=0)
    if method == "logits":
        return stacked.mean(dim=0)
    if method == "prob":
        return torch.softmax(stacked, dim=-1).mean(dim=0)

    batch_size, num_classes = logits_list[0].shape
    scores = logits_list[0].new_zeros(batch_size, num_classes)
    if method == "rank_vote":
        weights = torch.linspace(1.0, 0.0, steps=num_classes, device=scores.device)
        for logits in logits_list:
            order = logits.argsort(dim=1, descending=True)
            scores.scatter_add_(1, order, weights.unsqueeze(0).expand(batch_size, -1))
        return scores / len(logits_list)
    if method == "topk_vote":
        k = min(max(vote_topk, 1), num_classes)
        weights = torch.linspace(1.0, 0.1, steps=k, device=scores.device)
        for logits in logits_list:
            indices = logits.topk(k, dim=1).indices
            scores.scatter_add_(1, indices, weights.unsqueeze(0).expand(batch_size, -1))
        return scores / len(logits_list)
    raise ValueError(f"Unsupported ensemble method: {method}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_config = build_model_config(args)

    models = []
    for ckpt_path in args.ckpts:
        model = BeMambaModel(model_config).to(device)
        load_checkpoint(model, ckpt_path, device)
        model.eval()
        models.append(model)

    train_csv_path = os.path.join(args.split_root, f"{args.scenario}_train.csv")
    temp_ds = MultimodalDataset(
        mode="train",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=train_csv_path,
        image_subdir=args.image_subdir,
        lidar_representation=args.lidar_representation,
    )
    gps_stats = temp_ds.get_gps_stats()
    test_csv_path = os.path.join(args.split_root, f"{args.scenario}_test.csv")
    test_ds = MultimodalDataset(
        mode="test",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=test_csv_path,
        image_subdir=args.image_subdir,
        gps_stats=gps_stats,
        lidar_representation=args.lidar_representation,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    loader = build_dataloader(test_ds, train_config, shuffle=False)

    sample_total = 0
    acc1_total = 0.0
    acc2_total = 0.0
    acc3_total = 0.0
    dba_total = 0.0
    apl_total = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs, radars, lidars, gps, targets, power_vec, *_ = move_batch_to_device(batch, device)
            logits_list = []
            for model in models:
                logits = model(imgs, radars, lidars, gps)
                logits_list.append(logits)
            outputs = combine_ensemble_outputs(logits_list, args.method, args.vote_topk)
            batch_size = targets.size(0)
            acc1, acc2, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 2, 3))
            acc1_total += acc1 * batch_size
            acc2_total += acc2 * batch_size
            acc3_total += acc3 * batch_size
            dba_total += calculate_dba_score(outputs, targets) * batch_size
            apl_total += calculate_apl(outputs, power_vec) * batch_size
            sample_total += batch_size

    metrics = {
        "acc1": acc1_total / max(sample_total, 1),
        "acc2": acc2_total / max(sample_total, 1),
        "acc3": acc3_total / max(sample_total, 1),
        "dba": dba_total / max(sample_total, 1),
        "apl": apl_total / max(sample_total, 1),
    }
    print(f"Ensemble checkpoints: {len(models)}")
    print(f"method: {args.method}")
    print(f"top1_acc: {metrics['acc1']:.2f}%")
    print(f"top2_acc: {metrics['acc2']:.2f}%")
    print(f"top3_acc: {metrics['acc3']:.2f}%")
    print(f"dba: {metrics['dba']:.4f}")
    print(f"apl: {metrics['apl']:.4f} dB")


if __name__ == "__main__":
    main()
