"""
ブートストラップ学習CSVから特徴量重要度を集計する。

使い方:
  python -m maintenance.analyze_feature_importance --pair USDJPY --input data/USDJPY_bootstrap_train.csv
  python -m maintenance.analyze_feature_importance --pair EURUSD --input data/EURUSD_bootstrap_train.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.lgbm_model import FEATURE_NAMES, LGBM_PARAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature importance from bootstrap CSV")
    parser.add_argument("--pair", required=True, choices=["USDJPY", "EURUSD", "GBPJPY"])
    parser.add_argument("--input", required=True, help="Path to bootstrap train CSV")
    parser.add_argument("--output-dir", default="models", help="Directory to save report json")
    parser.add_argument("--top-k", type=int, default=12, help="Top K features to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    df = pd.read_csv(input_path)
    if "pair" in df.columns:
        df = df[df["pair"] == args.pair].copy()

    missing = [c for c in FEATURE_NAMES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.dropna(subset=FEATURE_NAMES + ["label"]).copy()
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy(dtype=np.int32)

    model = lgb.LGBMClassifier(**LGBM_PARAMS, class_weight="balanced")
    model.fit(X, y, feature_name=FEATURE_NAMES)

    gains = model.booster_.feature_importance(importance_type="gain")
    splits = model.booster_.feature_importance(importance_type="split")

    rows = []
    total_gain = float(gains.sum()) if float(gains.sum()) > 0 else 1.0
    for name, gain, split in zip(FEATURE_NAMES, gains, splits):
        rows.append({
            "feature": name,
            "gain": float(gain),
            "gain_ratio": float(gain / total_gain),
            "split": int(split),
        })

    rows_sorted = sorted(rows, key=lambda r: r["gain"], reverse=True)
    zero_gain = [r["feature"] for r in rows_sorted if r["gain"] <= 0.0]
    low_gain = [r["feature"] for r in rows_sorted if r["gain_ratio"] < 0.005]

    print(f"pair={args.pair} samples={len(X)}")
    print("top_features=")
    for r in rows_sorted[: max(1, args.top_k)]:
        print(f"  {r['feature']}: gain_ratio={r['gain_ratio']:.4f}, split={r['split']}")

    print(f"zero_gain_features={zero_gain}")
    print(f"low_gain_features(<0.5%)={low_gain}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lgbm_{args.pair}_feature_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pair": args.pair,
                "samples": int(len(X)),
                "top_features": rows_sorted[: max(1, args.top_k)],
                "all_features": rows_sorted,
                "zero_gain_features": zero_gain,
                "low_gain_features": low_gain,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
