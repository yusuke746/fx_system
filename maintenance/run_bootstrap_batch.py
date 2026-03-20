"""
TradingView CSV から 3ペア一括で初期モデルを作成するバッチ。

処理フロー:
  1) build_bootstrap_dataset.py 相当の学習CSV生成
  2) train_bootstrap_model.py 相当の LightGBM 学習

想定入力ファイル名（既定）:
  data/{PAIR}_chart.csv
  例: data/USDJPY_chart.csv

使い方:
  python maintenance/run_bootstrap_batch.py
  python maintenance/run_bootstrap_batch.py --input-dir data --model-dir models --skip-wfv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# スクリプト直接実行時にも fx_system/ をパスに追加
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from maintenance.build_bootstrap_dataset import build_dataset
from ml.lgbm_model import FEATURE_NAMES
from ml.trainer import train_model, walk_forward_validate

DEFAULT_PAIRS = ["USDJPY", "EURUSD", "GBPJPY"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bootstrap dataset+training for multiple pairs")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS), help="Comma-separated pairs")
    parser.add_argument("--input-dir", default="data", help="Directory containing TradingView chart CSVs")
    parser.add_argument("--output-dir", default="data", help="Directory to write bootstrap train CSVs")
    parser.add_argument("--input-pattern", default="{pair}_chart.csv", help="Input filename pattern")
    parser.add_argument("--output-pattern", default="{pair}_bootstrap_train.csv", help="Output filename pattern")
    parser.add_argument("--horizon-bars", type=int, default=16, help="Future horizon in 15m bars")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    parser.add_argument("--min-samples", type=int, default=300, help="Minimum samples required per pair")
    parser.add_argument("--skip-wfv", action="store_true", help="Skip walk-forward validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    summary: list[dict] = []

    for pair in pairs:
        input_path = input_dir / args.input_pattern.format(pair=pair)
        output_path = output_dir / args.output_pattern.format(pair=pair)

        if not input_path.exists():
            summary.append({"pair": pair, "status": "skip", "reason": f"missing_input:{input_path}"})
            continue

        try:
            built = build_dataset(
                input_path=input_path,
                output_path=output_path,
                pair=pair,
                horizon_bars=args.horizon_bars,
            )

            X, y, dropped, signal_times = _load_xy(output_path, pair=pair)
            if len(X) < args.min_samples:
                summary.append({
                    "pair": pair,
                    "status": "skip",
                    "reason": f"insufficient_samples({len(X)}<{args.min_samples})",
                    "dataset_rows": int(len(built)),
                })
                continue

            wfv_acc = None
            if not args.skip_wfv:
                val = walk_forward_validate(X, y, signal_times=signal_times)
                wfv_acc = float(val.get("accuracy", 0.0))

            train_model(X, y, pair=pair, model_dir=args.model_dir)

            summary.append({
                "pair": pair,
                "status": "trained",
                "dataset_rows": int(len(built)),
                "train_rows": int(len(X)),
                "dropped_nan_rows": int(dropped),
                "wfv_accuracy": None if wfv_acc is None else round(wfv_acc, 4),
                "class_up": int(np.sum(y == 0)),
                "class_flat": int(np.sum(y == 1)),
                "class_down": int(np.sum(y == 2)),
                "model_path": str(Path(args.model_dir) / f"lgbm_{pair}.pkl"),
            })

        except Exception as e:
            summary.append({"pair": pair, "status": "error", "reason": str(e)})

    print("=== bootstrap batch summary ===")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
