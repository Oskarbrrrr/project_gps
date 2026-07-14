"""Train-cache retrieval diagnostic and guarded fixed-test evaluation.

The cache contains only train-split fused features and normalized 64-beam power
profiles. Query samples retrieve nearby train samples, then blend the retrieved
profile with the model probabilities. Shared sensor/GPS paths are purged by
default to avoid overlap leakage between nearby windows. Test evaluation must
use one pre-selected value for every cache hyperparameter.
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from eval_ensemble import build_model_config
from src.dataset import MultimodalDataset
from src.model import BeMambaModel, load_checkpoint
from train import TrainConfig, build_dataloader, move_batch_to_device


MODEL_VARIANTS = [
    "bemamba", "clean_plus", "clean_plus_v2", "clean_plus_v3", "clean_plus_v4",
    "clean_plus_v5", "clean_plus_v6", "clean_plus_v7", "clean_plus_v8",
    "clean_plus_v9", "clean_plus_v10", "clean_plus_v11", "clean_plus_v12",
    "clean_plus_v13", "clean_plus_v14", "clean_plus_v15",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune a train-only power cache on val or evaluate one fixed setting on test."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--split-root", default="./Data/splits_paper80_val")
    parser.add_argument("--scenario", default="scenario32")
    parser.add_argument("--image-subdir", default="camera_data_mask")
    parser.add_argument("--lidar-representation", choices=["binary", "count"], default="count")
    parser.add_argument("--gps-feature-mode", choices=["bemamba", "physical_kinematic"], default="bemamba")
    parser.add_argument("--gps-future-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--query-split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--confirm-fixed-test",
        action="store_true",
        help="Required for test; confirms that K/temperature/alpha were fixed on validation",
    )

    parser.add_argument("--topks", type=int, nargs="+", default=[3, 5, 7, 11])
    parser.add_argument("--retrieval-temperatures", type=float, nargs="+", default=[0.03, 0.05, 0.07, 0.10])
    parser.add_argument("--blend-alphas", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--no-purge-shared-paths", action="store_true")

    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="clean_plus_v14")
    parser.add_argument("--backbone-stage", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--patch-grid", type=int, default=6)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--temporal-order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--spatial-scan", choices=["vertical", "row"], default="vertical")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gps-hidden-dim", type=int, default=96)
    parser.add_argument("--clean-cross-attn", action="store_true")
    parser.add_argument("--spatial-mixer-layers", type=int, default=None)
    parser.add_argument("--order-gate", action="store_true")
    parser.add_argument("--attn-head", action="store_true")
    parser.add_argument("--branch-ensemble", action="store_true")
    parser.add_argument("--candidate-rerank-delta-bound", type=float, default=0.20)
    parser.add_argument("--candidate-embed-dropout", type=float, default=0.0)
    return parser.parse_args()


def build_dataset(args, csv_path, gps_stats=None):
    return MultimodalDataset(
        mode="test",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=csv_path,
        image_subdir=args.image_subdir,
        gps_stats=gps_stats,
        lidar_representation=args.lidar_representation,
        gps_feature_mode=args.gps_feature_mode,
        gps_future_steps=args.gps_future_steps,
    )


def normalize_power(power_vec):
    power = torch.nan_to_num(power_vec.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    sums = power.sum(dim=1, keepdim=True)
    uniform = torch.full_like(power, 1.0 / power.size(1))
    return torch.where(sums > 1e-8, power / sums.clamp_min(1e-8), uniform)


@torch.no_grad()
def extract_split(model, loader, device):
    all_features, all_probs, all_power, all_targets = [], [], [], []
    amp_enabled = device.type == "cuda"
    for batch in loader:
        imgs, radars, lidars, gps, targets, power_vec, *_ = move_batch_to_device(batch, device)
        with autocast(enabled=amp_enabled):
            logits, feature_dict = model(imgs, radars, lidars, gps, return_features=True)
        if isinstance(logits, tuple):
            logits = logits[0]
        tokens = feature_dict["fused_tokens"].float()
        pooled = torch.cat(
            [tokens.mean(dim=1), tokens.std(dim=1, unbiased=False), tokens.amax(dim=1)],
            dim=1,
        )
        all_features.append(F.normalize(pooled, dim=1).cpu())
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
        all_power.append(normalize_power(power_vec).cpu())
        all_targets.append(targets.cpu())
    return {
        "features": torch.cat(all_features),
        "probs": torch.cat(all_probs),
        "power": torch.cat(all_power),
        "targets": torch.cat(all_targets),
    }


def shared_path_columns(df):
    prefixes = ("unit1_rgb_", "unit1_radar_", "unit1_lidar_", "unit2_loc_")
    return [column for column in df.columns if column.startswith(prefixes)]


def row_paths(row, columns):
    paths = set()
    for column in columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        path = str(value).strip().replace("\\", "/").lower()
        if path:
            paths.add(path)
    return paths


def build_blocked_indices(train_df, val_df):
    columns = sorted(set(shared_path_columns(train_df)) & set(shared_path_columns(val_df)))
    inverted = defaultdict(set)
    for index, row in train_df.iterrows():
        for path in row_paths(row, columns):
            inverted[path].add(int(index))

    blocked = []
    for _, row in val_df.iterrows():
        indices = set()
        for path in row_paths(row, columns):
            indices.update(inverted.get(path, ()))
        blocked.append(indices)
    return blocked, columns


def topk_metrics(scores, targets):
    result = {}
    for k in (1, 2, 3):
        hits = (scores.topk(k, dim=1).indices == targets.unsqueeze(1)).any(dim=1)
        result[f"acc{k}"] = float(hits.float().mean().item() * 100.0)
    return result


def main():
    args = parse_args()
    if args.gps_future_steps < 0:
        raise ValueError("--gps-future-steps must be >= 0")
    if any(k < 1 for k in args.topks):
        raise ValueError("--topks values must be >= 1")
    if any(value <= 0 for value in args.retrieval_temperatures):
        raise ValueError("--retrieval-temperatures values must be > 0")
    if any(value < 0 or value > 1 for value in args.blend_alphas):
        raise ValueError("--blend-alphas values must be in [0, 1]")
    if args.query_split == "test":
        if not args.confirm_fixed_test:
            raise ValueError("--query-split test requires --confirm-fixed-test")
        if not (
            len(args.topks) == 1
            and len(args.retrieval_temperatures) == 1
            and len(args.blend_alphas) == 1
        ):
            raise ValueError(
                "Test evaluation forbids parameter sweeps: provide exactly one --topks, "
                "--retrieval-temperatures and --blend-alphas value"
            )
        if args.no_purge_shared_paths:
            raise ValueError("Fixed test evaluation requires shared-path purging")

    train_csv = os.path.join(args.split_root, f"{args.scenario}_train.csv")
    query_csv = os.path.join(args.split_root, f"{args.scenario}_{args.query_split}.csv")
    if not os.path.exists(query_csv):
        raise FileNotFoundError(f"Query split not found: {query_csv}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model_config = build_model_config(args)
    model = BeMambaModel(model_config).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    stats_ds = MultimodalDataset(
        mode="train",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=train_csv,
        image_subdir=args.image_subdir,
        lidar_representation=args.lidar_representation,
        gps_feature_mode=args.gps_feature_mode,
        gps_future_steps=args.gps_future_steps,
    )
    gps_stats = stats_ds.get_gps_stats()
    train_ds = build_dataset(args, train_csv, gps_stats=gps_stats)
    query_ds = build_dataset(args, query_csv, gps_stats=gps_stats)
    loader_config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Extracting train cache: {len(train_ds)} samples")
    train_data = extract_split(model, build_dataloader(train_ds, loader_config, False), device)
    print(f"Extracting {args.query_split} queries: {len(query_ds)} samples")
    query_data = extract_split(model, build_dataloader(query_ds, loader_config, False), device)
    similarities = query_data["features"] @ train_data["features"].T

    blocked, path_columns = build_blocked_indices(train_ds.df, query_ds.df)
    purge_shared = not args.no_purge_shared_paths
    if purge_shared:
        for row_index, train_indices in enumerate(blocked):
            if train_indices:
                similarities[row_index, list(train_indices)] = float("-inf")
    finite_per_query = torch.isfinite(similarities).sum(dim=1)
    min_available = int(finite_per_query.min().item())
    max_blocked = max((len(indices) for indices in blocked), default=0) if purge_shared else 0
    print(
        f"Path purge: {'enabled' if purge_shared else 'disabled'}; "
        f"columns={len(path_columns)}, max_blocked_per_query={max_blocked}, "
        f"min_available_neighbors={min_available}"
    )
    if min_available < 1:
        raise RuntimeError(f"At least one {args.query_split} query has no non-overlapping train neighbor")

    baseline = topk_metrics(query_data["probs"], query_data["targets"])
    print(
        f"Baseline {args.query_split}: Top-1={baseline['acc1']:.2f}% "
        f"Top-2={baseline['acc2']:.2f}% Top-3={baseline['acc3']:.2f}%"
    )
    results = []
    for requested_k in args.topks:
        k = min(int(requested_k), min_available)
        neighbor_scores, neighbor_indices = similarities.topk(k, dim=1)
        neighbor_power = train_data["power"][neighbor_indices]
        for temperature in args.retrieval_temperatures:
            weights = torch.softmax(neighbor_scores / float(temperature), dim=1)
            retrieved_power = (neighbor_power * weights.unsqueeze(2)).sum(dim=1)
            for alpha in args.blend_alphas:
                scores = (1.0 - alpha) * query_data["probs"] + alpha * retrieved_power
                metrics = topk_metrics(scores, query_data["targets"])
                results.append(
                    {
                        "topk": k,
                        "temperature": float(temperature),
                        "alpha": float(alpha),
                        **metrics,
                    }
                )

    results_df = pd.DataFrame(results).drop_duplicates()
    selected_row = results_df.sort_values(
        ["acc3", "acc2", "acc1", "alpha"], ascending=[False, False, False, True]
    ).iloc[0]
    setting_label = "Fixed test cache setting" if args.query_split == "test" else "Best val cache setting"
    print(
        f"{setting_label}: "
        f"K={int(selected_row['topk'])}, temperature={selected_row['temperature']:.3f}, "
        f"alpha={selected_row['alpha']:.2f}, Top-1={selected_row['acc1']:.2f}%, "
        f"Top-2={selected_row['acc2']:.2f}%, Top-3={selected_row['acc3']:.2f}% "
        f"(delta={selected_row['acc3'] - baseline['acc3']:+.2f}pp)"
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_name = "power_cache_fixed_test" if args.query_split == "test" else "power_cache_eval"
        output_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), output_name)
    os.makedirs(output_dir, exist_ok=True)
    result_filename = "test_result.csv" if args.query_split == "test" else "val_grid.csv"
    summary_filename = "test_summary.json" if args.query_split == "test" else "val_summary.json"
    results_df.to_csv(os.path.join(output_dir, result_filename), index=False)
    summary = {
        "checkpoint": args.ckpt,
        "scenario": args.scenario,
        "query_split": args.query_split,
        "purge_shared_paths": purge_shared,
        "path_columns": path_columns,
        "max_blocked_per_query": max_blocked,
        "baseline": baseline,
        "result": {key: float(selected_row[key]) for key in results_df.columns},
    }
    with open(os.path.join(output_dir, summary_filename), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    action = "fixed test evaluation" if args.query_split == "test" else "validation retrieval sweep"
    print(f"Saved {action} to: {output_dir}")


if __name__ == "__main__":
    main()
