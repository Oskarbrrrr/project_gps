"""Clean-test error analysis for one BeMamba/Clean-Plus checkpoint."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_ensemble import build_model_config
from src.dataset import MultimodalDataset
from src.model import BeMambaModel, load_checkpoint
from train import TrainConfig, build_dataloader, move_batch_to_device


MODEL_VARIANTS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze clean-test beam prediction errors for a saved checkpoint."
    )
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pth or another checkpoint")
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--split-root", default="./Data/splits_paper80")
    parser.add_argument("--scenario", default="scenario32")
    parser.add_argument("--image-subdir", default="camera_data_mask")
    parser.add_argument("--lidar-representation", choices=["binary", "count"], default="count")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-source-columns", choices=["none", "key", "all"], default="key")

    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="bemamba")
    parser.add_argument("--backbone-stage", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--patch-grid", type=int, default=6)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--temporal-order", choices=["forward", "reverse"], default="reverse")
    parser.add_argument("--spatial-scan", choices=["vertical", "row"], default="row")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gps-hidden-dim", type=int, default=96)
    parser.add_argument("--clean-cross-attn", action="store_true")
    parser.add_argument("--spatial-mixer-layers", type=int, default=None)
    parser.add_argument("--order-gate", action="store_true")
    parser.add_argument("--attn-head", action="store_true")
    parser.add_argument("--branch-ensemble", action="store_true")
    parser.add_argument("--topk", type=int, default=7, help="Largest Top-K list to store/analyze")
    parser.add_argument("--dba-delta", type=float, default=5.0)
    return parser.parse_args()


def infer_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    ckpt_path = Path(args.ckpt)
    if ckpt_path.parent.name == "checkpoints":
        return str(ckpt_path.parent.parent / "error_analysis")
    return str(Path("outputs") / "analysis" / f"{args.scenario}_{args.model_variant}")


def build_gps_stats(args: argparse.Namespace) -> dict:
    train_csv_path = os.path.join(args.split_root, f"{args.scenario}_train.csv")
    train_ds = MultimodalDataset(
        mode="train",
        data_root=args.data_root,
        split_root=args.split_root,
        scenario_name=args.scenario,
        csv_path=train_csv_path,
        image_subdir=args.image_subdir,
        lidar_representation=args.lidar_representation,
    )
    return train_ds.get_gps_stats()


def selected_source_columns(columns: list[str], mode: str) -> list[str]:
    if mode == "none":
        return []
    if mode == "all":
        return list(columns)

    preferred = [
        "unit1_loc",
        "unit2_loc_1",
        "unit2_loc_2",
        "unit1_pwr_60ghz",
    ]
    for prefix in ("unit1_rgb", "unit1_radar", "unit1_lidar"):
        preferred.extend(f"{prefix}_{idx}" for idx in range(1, 6))
    return [column for column in preferred if column in columns]


def sample_dba(top_indices: np.ndarray, target: int, delta: float = 5.0, k: int = 3) -> float:
    top = top_indices[:k]
    scores = []
    for current_k in range(1, k + 1):
        min_diff = np.abs(top[:current_k] - target).min() / max(delta, 1e-8)
        scores.append(1.0 - min(float(min_diff), 1.0))
    return float(np.mean(scores))


def power_loss_db(power_vec: np.ndarray, pred_idx: int) -> tuple[float, float, float, int]:
    power = np.nan_to_num(power_vec.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    power = np.clip(power, 0.0, None)
    opt_power = float(power.max())
    opt_idx = int(power.argmax())
    pred_power = float(power[pred_idx])
    if opt_power <= 1e-12:
        return 0.0, opt_power, pred_power, opt_idx
    pred_power_safe = max(pred_power, 1e-12)
    return float(10.0 * math.log10(opt_power / pred_power_safe)), opt_power, pred_power, opt_idx


def oracle_power_loss_db(power_vec: np.ndarray, candidate_indices: np.ndarray) -> float:
    losses = [power_loss_db(power_vec, int(idx))[0] for idx in candidate_indices]
    return float(min(losses)) if losses else 0.0


def format_list(values, one_based: bool = False, precision: int | None = None) -> str:
    formatted = []
    for value in values:
        if one_based:
            formatted.append(str(int(value) + 1))
        elif precision is not None:
            formatted.append(f"{float(value):.{precision}f}")
        else:
            formatted.append(str(value))
    return " ".join(formatted)


def append_sample_rows(
    rows: list[dict],
    logits: torch.Tensor,
    targets: torch.Tensor,
    power_vec: torch.Tensor,
    test_df: pd.DataFrame,
    source_columns: list[str],
    start_index: int,
    max_topk: int,
    dba_delta: float,
) -> None:
    probs = torch.softmax(logits, dim=1)
    top_values, top_indices = logits.topk(max_topk, dim=1)
    top_probs = probs.gather(1, top_indices)
    sorted_indices = logits.argsort(dim=1, descending=True)

    logits_np = logits.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    power_np = power_vec.detach().cpu().numpy()
    top_indices_np = top_indices.detach().cpu().numpy()
    top_values_np = top_values.detach().cpu().numpy()
    top_probs_np = top_probs.detach().cpu().numpy()
    sorted_indices_np = sorted_indices.detach().cpu().numpy()

    for batch_idx, target in enumerate(targets_np):
        global_idx = start_index + batch_idx
        target = int(target)
        top_idx = top_indices_np[batch_idx]
        top_logits = top_values_np[batch_idx]
        top_prob = top_probs_np[batch_idx]
        target_rank = int(np.where(sorted_indices_np[batch_idx] == target)[0][0]) + 1
        pred1 = int(top_idx[0])
        pred1_diff = abs(pred1 - target)
        apl_db, opt_power, pred_power, opt_idx = power_loss_db(power_np[batch_idx], pred1)

        row = {
            "sample_index": global_idx,
            "target_beam": target + 1,
            "pred1_beam": pred1 + 1,
            "pred1_abs_diff": pred1_diff,
            "target_rank": target_rank,
            "target_logit": f"{float(logits_np[batch_idx, target]):.6f}",
            "pred1_logit": f"{float(top_logits[0]):.6f}",
            "pred1_prob": f"{float(top_prob[0]):.6f}",
            "top3_beams": format_list(top_idx[:3], one_based=True),
            "top5_beams": format_list(top_idx[:5], one_based=True),
            "top7_beams": format_list(top_idx[:7], one_based=True),
            "top7_logits": format_list(top_logits[:7], precision=6),
            "top7_probs": format_list(top_prob[:7], precision=6),
            "hit_top1": int(target in top_idx[:1]),
            "hit_top2": int(target in top_idx[:2]),
            "hit_top3": int(target in top_idx[:3]),
            "hit_top5": int(target in top_idx[:5]),
            "hit_top7": int(target in top_idx[:7]),
            "min_abs_diff_top3": int(np.abs(top_idx[:3] - target).min()),
            "min_abs_diff_top5": int(np.abs(top_idx[:5] - target).min()),
            "min_abs_diff_top7": int(np.abs(top_idx[:7] - target).min()),
            "near_top3_delta1": int(np.abs(top_idx[:3] - target).min() <= 1),
            "near_top3_delta2": int(np.abs(top_idx[:3] - target).min() <= 2),
            "near_top3_delta3": int(np.abs(top_idx[:3] - target).min() <= 3),
            "near_top3_delta5": int(np.abs(top_idx[:3] - target).min() <= 5),
            "sample_dba": f"{sample_dba(top_idx, target, delta=dba_delta):.6f}",
            "apl_db": f"{apl_db:.6f}",
            "oracle_top3_apl_db": f"{oracle_power_loss_db(power_np[batch_idx], top_idx[:3]):.6f}",
            "oracle_top5_apl_db": f"{oracle_power_loss_db(power_np[batch_idx], top_idx[:5]):.6f}",
            "oracle_top7_apl_db": f"{oracle_power_loss_db(power_np[batch_idx], top_idx[:7]):.6f}",
            "opt_power_beam": opt_idx + 1,
            "target_is_power_opt": int(target == opt_idx),
            "pred1_power": f"{pred_power:.8f}",
            "opt_power": f"{opt_power:.8f}",
        }

        source_row = test_df.iloc[global_idx]
        for column in source_columns:
            row[f"src_{column}"] = source_row[column]
        rows.append(row)


def summarize(rows: list[dict], max_topk: int) -> dict:
    total = len(rows)
    if total == 0:
        return {"total": 0}

    def mean_float(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    def pct_count(count: int) -> float:
        return 100.0 * count / max(total, 1)

    hit_counts = {
        f"top{k}_acc": pct_count(sum(int(row[f"hit_top{k}"]) for row in rows))
        for k in (1, 2, 3, 5, 7)
        if k <= max_topk
    }
    rank_thresholds = [1, 2, 3, 5, 7, 10, 20]
    rank_counts = {
        f"target_rank_le_{threshold}": pct_count(
            sum(int(row["target_rank"]) <= threshold for row in rows)
        )
        for threshold in rank_thresholds
    }

    top3_misses = [row for row in rows if int(row["hit_top3"]) == 0]
    miss_total = len(top3_misses)
    miss_summary = {}
    if miss_total > 0:
        miss_summary = {
            "top3_miss_count": miss_total,
            "top3_miss_target_in_top5": 100.0
            * sum(int(row["hit_top5"]) for row in top3_misses)
            / miss_total,
            "top3_miss_target_in_top7": 100.0
            * sum(int(row["hit_top7"]) for row in top3_misses)
            / miss_total,
            "top3_miss_near_delta1": 100.0
            * sum(int(row["near_top3_delta1"]) for row in top3_misses)
            / miss_total,
            "top3_miss_near_delta2": 100.0
            * sum(int(row["near_top3_delta2"]) for row in top3_misses)
            / miss_total,
            "top3_miss_near_delta3": 100.0
            * sum(int(row["near_top3_delta3"]) for row in top3_misses)
            / miss_total,
            "top3_miss_near_delta5": 100.0
            * sum(int(row["near_top3_delta5"]) for row in top3_misses)
            / miss_total,
        }

    pred1_diffs = np.array([int(row["pred1_abs_diff"]) for row in rows])
    diff_buckets = {
        "pred1_diff_eq_0": pct_count(int((pred1_diffs == 0).sum())),
        "pred1_diff_eq_1": pct_count(int((pred1_diffs == 1).sum())),
        "pred1_diff_eq_2": pct_count(int((pred1_diffs == 2).sum())),
        "pred1_diff_3_to_5": pct_count(int(((pred1_diffs >= 3) & (pred1_diffs <= 5)).sum())),
        "pred1_diff_6_to_10": pct_count(int(((pred1_diffs >= 6) & (pred1_diffs <= 10)).sum())),
        "pred1_diff_gt_10": pct_count(int((pred1_diffs > 10).sum())),
    }

    summary = {
        "total": total,
        **hit_counts,
        **rank_counts,
        **miss_summary,
        **diff_buckets,
        "mean_target_rank": mean_float("target_rank"),
        "median_target_rank": float(np.median([int(row["target_rank"]) for row in rows])),
        "mean_sample_dba": mean_float("sample_dba"),
        "mean_apl_db": mean_float("apl_db"),
        "mean_oracle_top3_apl_db": mean_float("oracle_top3_apl_db"),
        "mean_oracle_top5_apl_db": mean_float("oracle_top5_apl_db"),
        "mean_oracle_top7_apl_db": mean_float("oracle_top7_apl_db"),
        "target_is_power_opt_pct": pct_count(sum(int(row["target_is_power_opt"]) for row in rows)),
    }
    summary["oracle_top5_apl_gain_db"] = summary["mean_apl_db"] - summary["mean_oracle_top5_apl_db"]
    summary["oracle_top7_apl_gain_db"] = summary["mean_apl_db"] - summary["mean_oracle_top7_apl_db"]
    return summary


def write_outputs(rows: list[dict], summary: dict, output_dir: str, args: argparse.Namespace) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{args.scenario}_clean_error_samples.csv")
    summary_txt_path = os.path.join(output_dir, f"{args.scenario}_clean_error_summary.txt")
    summary_json_path = os.path.join(output_dir, f"{args.scenario}_clean_error_summary.json")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    text_lines = [
        f"checkpoint: {args.ckpt}",
        f"scenario: {args.scenario}",
        f"image_subdir: {args.image_subdir}",
        f"model_variant: {args.model_variant}",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            text_lines.append(f"{key}: {value:.4f}")
        else:
            text_lines.append(f"{key}: {value}")

    with open(summary_txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(text_lines) + "\n")

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n".join(text_lines))
    print("")
    print(f"Saved per-sample CSV: {csv_path}")
    print(f"Saved summary TXT : {summary_txt_path}")
    print(f"Saved summary JSON: {summary_json_path}")


def main() -> None:
    args = parse_args()
    if args.topk < 7:
        raise ValueError("--topk should be at least 7 for the default SC32 analysis")

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    model_config = build_model_config(args)

    print(f"Loading checkpoint: {args.ckpt}")
    model = BeMambaModel(model_config).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    gps_stats = build_gps_stats(args)
    test_csv_path = os.path.join(args.split_root, f"{args.scenario}_test.csv")
    test_df = pd.read_csv(test_csv_path)
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
    source_columns = selected_source_columns(list(test_df.columns), args.include_source_columns)

    rows = []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            imgs, radars, lidars, gps, targets, power_vec, *_ = move_batch_to_device(batch, device)
            logits = model(imgs, radars, lidars, gps)
            append_sample_rows(
                rows=rows,
                logits=logits,
                targets=targets,
                power_vec=power_vec,
                test_df=test_df,
                source_columns=source_columns,
                start_index=sample_offset,
                max_topk=args.topk,
                dba_delta=args.dba_delta,
            )
            sample_offset += targets.size(0)

    summary = summarize(rows, max_topk=args.topk)
    output_dir = infer_output_dir(args)
    write_outputs(rows, summary, output_dir, args)


if __name__ == "__main__":
    main()
