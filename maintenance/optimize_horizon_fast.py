"""
短期で利益期待を上げるための、ホライズン高速探索スクリプト。

目的:
  - TradingView CSV から複数 horizon でデータセットを再構築
  - date-based walk-forward 検証で、利益プロキシを比較
  - ペアごとの推奨 horizon (bars) を出力

利益プロキシ:
  pred=long  -> +future_return_pips
  pred=short -> -future_return_pips
  pred=flat  -> 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import get_trading_config
from maintenance.build_bootstrap_dataset import build_dataset
from ml.lgbm_model import FEATURE_NAMES, get_lgbm_params


PAIRS = ("USDJPY", "EURUSD", "GBPJPY")


@dataclass
class HorizonScore:
    pair: str
    horizon_bars: int
    n_folds: int
    avg_accuracy: float
    avg_balanced_accuracy: float
    avg_majority_baseline: float
    avg_proxy_pips_per_signal: float
    avg_proxy_pips_per_trade: float
    avg_trade_rate: float
    total_val_samples: int


def _build_class_weight(y: np.ndarray, directional_boost: float) -> dict[int, float]:
    if len(y) == 0:
        return {0: 1.0, 1: 1.0, 2: 1.0}

    counts = np.bincount(y, minlength=3).astype(float)
    total = float(counts.sum())
    class_weight: dict[int, float] = {}
    for cls in (0, 1, 2):
        denom = max(counts[cls], 1.0)
        class_weight[cls] = total / (3.0 * denom)

    boost = max(0.5, float(directional_boost))
    class_weight[0] *= boost
    class_weight[2] *= boost
    return class_weight


def _evaluate_pair_horizon(
    pair: str,
    dataset_csv_path: Path,
    train_days: int,
    val_days: int,
) -> HorizonScore:
    config = get_trading_config()
    directional_boost = float(config.get("ml", {}).get("directional_class_boost", 1.0))
    params = get_lgbm_params(pair)

    df = pd.read_csv(dataset_csv_path)
    required_cols = [*FEATURE_NAMES, "label", "signal_time", "future_return_pips"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{pair}: missing columns {missing} in {dataset_csv_path}")

    df = df.dropna(subset=FEATURE_NAMES + ["label", "signal_time", "future_return_pips"]).copy()
    if len(df) == 0:
        return HorizonScore(pair, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    times = pd.to_datetime(df["signal_time"], errors="coerce", utc=True)
    valid_mask = ~times.isna()
    df = df.loc[valid_mask].copy()
    times = np.array(pd.to_datetime(df["signal_time"], errors="coerce", utc=True).dt.to_pydatetime())

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy(dtype=np.int32)
    future_return_pips = df["future_return_pips"].astype(float).to_numpy(dtype=np.float64)

    if len(X) == 0:
        return HorizonScore(pair, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    fold_accuracies: list[float] = []
    fold_bal_accs: list[float] = []
    fold_majority_baselines: list[float] = []
    fold_proxy_signal: list[float] = []
    fold_proxy_trade: list[float] = []
    fold_trade_rates: list[float] = []

    total_val_samples = 0
    fold = 0
    t_start = times[0]
    t_end = times[-1]

    while True:
        train_start_dt = t_start + timedelta(days=fold * val_days)
        train_end_dt = train_start_dt + timedelta(days=train_days)
        val_end_dt = train_end_dt + timedelta(days=val_days)
        if val_end_dt > t_end:
            break

        train_mask = (times >= train_start_dt) & (times < train_end_dt)
        val_mask = (times >= train_end_dt) & (times < val_end_dt)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        ret_val = future_return_pips[val_mask]

        if len(X_train) < train_days or len(X_val) == 0:
            fold += 1
            continue

        class_weight = _build_class_weight(y_train, directional_boost)
        model = lgb.LGBMClassifier(**params, class_weight=class_weight)
        model.fit(X_train, y_train, feature_name=FEATURE_NAMES)

        y_pred = model.predict(X_val)
        acc = float(np.mean(y_pred == y_val))

        # balanced_accuracy (sklearn依存を増やさない簡易実装)
        recalls = []
        for cls in (0, 1, 2):
            cls_mask = y_val == cls
            denom = int(cls_mask.sum())
            if denom == 0:
                continue
            recalls.append(float(np.mean(y_pred[cls_mask] == cls)))
        bal_acc = float(np.mean(recalls)) if recalls else 0.0

        class_counts = np.bincount(y_val, minlength=3)
        majority_baseline = float(class_counts.max() / len(y_val))

        proxy = np.where(y_pred == 0, ret_val, np.where(y_pred == 2, -ret_val, 0.0))
        trade_mask = y_pred != 1
        trade_rate = float(np.mean(trade_mask))

        proxy_signal = float(np.mean(proxy))
        proxy_trade = float(np.mean(proxy[trade_mask])) if np.any(trade_mask) else 0.0

        fold_accuracies.append(acc)
        fold_bal_accs.append(bal_acc)
        fold_majority_baselines.append(majority_baseline)
        fold_proxy_signal.append(proxy_signal)
        fold_proxy_trade.append(proxy_trade)
        fold_trade_rates.append(trade_rate)
        total_val_samples += len(y_val)
        fold += 1

    n_folds = len(fold_accuracies)
    if n_folds == 0:
        return HorizonScore(pair, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # horizon は呼び出し側で設定
    return HorizonScore(
        pair=pair,
        horizon_bars=0,
        n_folds=n_folds,
        avg_accuracy=float(np.mean(fold_accuracies)),
        avg_balanced_accuracy=float(np.mean(fold_bal_accs)),
        avg_majority_baseline=float(np.mean(fold_majority_baselines)),
        avg_proxy_pips_per_signal=float(np.mean(fold_proxy_signal)),
        avg_proxy_pips_per_trade=float(np.mean(fold_proxy_trade)),
        avg_trade_rate=float(np.mean(fold_trade_rates)),
        total_val_samples=total_val_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast horizon optimization using CSV bootstrap pipeline")
    parser.add_argument("--input-dir", default="data", help="Input CSV directory")
    parser.add_argument("--work-dir", default="data", help="Intermediate bootstrap CSV directory")
    parser.add_argument("--pairs", default=",".join(PAIRS), help="Comma-separated pairs")
    parser.add_argument("--horizons", default="4,6,8,12,16", help="Comma-separated horizon bars")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {pair: [] for pair in pairs}
    best: dict[str, dict] = {}

    for pair in pairs:
        chart_path = input_dir / f"{pair}_chart.csv"
        if not chart_path.exists():
            continue

        best_score = None
        best_row: dict | None = None

        for horizon in horizons:
            out_path = work_dir / f"{pair}_bootstrap_h{horizon}.csv"
            build_dataset(
                input_path=chart_path,
                output_path=out_path,
                pair=pair,
                horizon_bars=horizon,
            )

            score = _evaluate_pair_horizon(
                pair=pair,
                dataset_csv_path=out_path,
                train_days=args.train_days,
                val_days=args.val_days,
            )
            score.horizon_bars = horizon
            row = {
                "pair": pair,
                "horizon_bars": horizon,
                "horizon_minutes": horizon * 15,
                "n_folds": score.n_folds,
                "avg_accuracy": round(score.avg_accuracy, 6),
                "avg_balanced_accuracy": round(score.avg_balanced_accuracy, 6),
                "avg_majority_baseline": round(score.avg_majority_baseline, 6),
                "avg_proxy_pips_per_signal": round(score.avg_proxy_pips_per_signal, 6),
                "avg_proxy_pips_per_trade": round(score.avg_proxy_pips_per_trade, 6),
                "avg_trade_rate": round(score.avg_trade_rate, 6),
                "total_val_samples": score.total_val_samples,
            }
            results[pair].append(row)

            # 主目的: 期待利益プロキシ最大化。副目的: balanced_accuracy優先。
            key = (score.avg_proxy_pips_per_signal, score.avg_balanced_accuracy)
            if best_score is None or key > best_score:
                best_score = key
                best_row = row

        if best_row is not None:
            best[pair] = best_row

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs,
        "horizons": horizons,
        "results": results,
        "best": best,
    }

    out_path = work_dir / f"horizon_search_results_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
