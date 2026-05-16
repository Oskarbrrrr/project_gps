import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Generate scenario splits for BeMamba experiments.")
    parser.add_argument("--data-root", default=None, help="Defaults to <repo>/Data/Multi_Modal")
    parser.add_argument("--output-dir", default=None, help="Defaults to <repo>/Data/splits")
    parser.add_argument("--scenarios", nargs="+", default=["scenario32", "scenario33", "scenario34"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["paper80_20", "train_val_test"],
        default="paper80_20",
        help="paper80_20 writes only train/test. train_val_test writes 60/20/20 for legacy compatibility.",
    )
    parser.add_argument("--no-stratify-beam", action="store_true")
    return parser.parse_args()


def resolve_paths(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root or os.path.join(base_dir, "Data", "Multi_Modal")
    output_dir = args.output_dir or os.path.join(base_dir, "Data", "splits")
    return data_root, output_dir


def build_stratify_labels(df: pd.DataFrame, enable: bool):
    if not enable or "unit1_beam" not in df.columns:
        return None
    label_counts = df["unit1_beam"].value_counts()
    if (label_counts < 2).any():
        rare_labels = label_counts[label_counts < 2].index
        if len(rare_labels) > 0:
            print(f"[warn] disable stratify because some beams appear <2 times: {list(rare_labels[:10])}")
        return None
    return df["unit1_beam"]


def split_paper_80_20(df: pd.DataFrame, test_size: float, seed: int, stratify_labels):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_train_val_test(df: pd.DataFrame, seed: int, stratify_labels):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=seed,
        stratify=stratify_labels,
    )
    temp_stratify = temp_df["unit1_beam"] if stratify_labels is not None else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_stratify,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_csv(scenario_name: str, csv_path: str, output_dir: str, args) -> None:
    print(f"[info] processing {scenario_name}")
    if not os.path.exists(csv_path):
        print(f"[skip] missing csv: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    stratify_labels = build_stratify_labels(df, enable=(not args.no_stratify_beam))

    os.makedirs(output_dir, exist_ok=True)
    if args.split_mode == "paper80_20":
        train_df, test_df = split_paper_80_20(df, test_size=args.test_size, seed=args.seed, stratify_labels=stratify_labels)
        train_path = os.path.join(output_dir, f"{scenario_name}_train.csv")
        test_path = os.path.join(output_dir, f"{scenario_name}_test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"[ok] {scenario_name}: train={len(train_df)} test={len(test_df)} -> {train_path}, {test_path}")
        return

    train_df, val_df, test_df = split_train_val_test(df, seed=args.seed, stratify_labels=stratify_labels)
    train_df.to_csv(os.path.join(output_dir, f"{scenario_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{scenario_name}_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{scenario_name}_test.csv"), index=False)
    print(f"[ok] {scenario_name}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")


def main():
    args = parse_args()
    data_root, output_dir = resolve_paths(args)
    for scenario_name in args.scenarios:
        csv_path = os.path.join(data_root, f"{scenario_name}.csv")
        split_csv(scenario_name, csv_path, output_dir, args)


if __name__ == "__main__":
    main()
