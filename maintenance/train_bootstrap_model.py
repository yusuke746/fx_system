"""
ブートストラップCSVから LightGBM 初期モデルを学習する。

入力CSVの想定:
  - maintenance/build_bootstrap_dataset.py の出力
        - 43特徴量（ml.lgbm_model.FEATURE_NAMES と同名）
  - label 列（0=up, 1=flat, 2=down）

使い方:
  python maintenance/train_bootstrap_model.py \
    --input data/USDJPY_bootstrap_train.csv \
    --pair USDJPY \
    --model-dir models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.lgbm_model import FEATURE_NAMES
from ml.trainer import save_model_metrics, train_model, walk_forward_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bootstrap LightGBM model from CSV")
    parser.add_argument("--input", required=True, help="Path to bootstrap training CSV")
    parser.add_argument("--pair", required=True, choices=["USDJPY", "EURUSD", "GBPJPY"], help="Currency pair")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    parser.add_argument("--min-samples", type=int, default=300, help="Minimum required samples")
    parser.add_argument("--skip-wfv", action="store_true", help="Skip walk-forward validation")
    return parser.parse_args()


def _load_xy(input_path: Path, pair: str) -> tuple[np.ndarray, np.ndarray, int, list | None]:
    df = pd.read_csv(input_path)

    if "pair" in df.columns:
        df = df[df["pair"] == pair].copy()

    missing = [c for c in FEATURE_NAMES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before_drop = len(df)
    df = df.dropna(subset=FEATURE_NAMES + ["label"]).copy()
    dropped = before_drop - len(df)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy(dtype=np.int32)

    signal_times = None
    if "signal_time" in df.columns:
        parsed = pd.to_datetime(df["signal_time"], errors="coerce", utc=True)
        times_list = [t.to_pydatetime() for t in parsed if pd.notna(t)]
        if len(times_list) == len(X):
            signal_times = times_list

    return X, y, dropped, signal_times


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    X, y, dropped, signal_times = _load_xy(input_path=input_path, pair=args.pair)

    if len(X) < args.min_samples:
        raise ValueError(f"insufficient_samples({len(X)}<{args.min_samples})")

    val = None
    if not args.skip_wfv:
        val = walk_forward_validate(X, y, signal_times=signal_times, pair=args.pair)
        print(f"walk_forward_accuracy={val.get('accuracy', 0.0):.4f}")
        print(f"walk_forward_balanced_accuracy={val.get('balanced_accuracy', 0.0):.4f}")
        print(f"walk_forward_majority_baseline={val.get('majority_baseline', 0.0):.4f}")
        print(f"walk_forward_folds={val.get('n_folds', len(val.get('fold_results', [])))}")

    train_model(X, y, pair=args.pair, model_dir=args.model_dir)
    save_model_metrics(
        pair=args.pair,
        model_dir=args.model_dir,
        metrics={
            "accuracy": float(val.get("accuracy", 0.0)) if val else 0.0,
            "balanced_accuracy": float(val.get("balanced_accuracy", 0.0)) if val else 0.0,
            "majority_baseline": float(val.get("majority_baseline", 0.0)) if val else 0.0,
            "samples": int(len(X)),
            "source": "bootstrap_csv",
        },
    )

    cls_counts = {
        "up": int(np.sum(y == 0)),
        "flat": int(np.sum(y == 1)),
        "down": int(np.sum(y == 2)),
    }
    print(f"pair={args.pair}")
    print(f"rows={len(X)}")
    print(f"dropped_nan_rows={dropped}")
    print(f"class_counts={cls_counts}")
    print(f"model_saved={Path(args.model_dir) / f'lgbm_{args.pair}.pkl'}")


if __name__ == "__main__":
    main()
