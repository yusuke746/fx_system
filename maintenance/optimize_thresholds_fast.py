"""
短期で利益期待を上げるための、予測閾値（direction/block）高速最適化。

対象:
  - ml.prediction_thresholds.{PAIR}.long/short

方針:
  - date-based walk-forward で out-of-sample 予測確率を作成
  - 予測方向（argmax）が long/short のときのみ執行対象
  - 各閾値セットで利益プロキシを計算し最大化

利益プロキシ:
  pnl_proxy =
    +future_return_pips (pred=long かつ 閾値通過)
    -future_return_pips (pred=short かつ 閾値通過)
     0 (非執行)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import get_trading_config
from ml.lgbm_model import FEATURE_NAMES, get_lgbm_params


PAIRS = ("USDJPY", "EURUSD", "GBPJPY")


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


def _collect_oos_predictions(
    pair: str,
    csv_path: Path,
    train_days: int,
    val_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      prob_up, prob_down, pred_dir(0 long / 2 short / 1 flat), future_return_pips
    """
    config = get_trading_config()
    directional_boost = float(config.get("ml", {}).get("directional_class_boost", 1.0))
    params = get_lgbm_params(pair)

    df = pd.read_csv(csv_path)
    required = [*FEATURE_NAMES, "signal_time", "label", "future_return_pips"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{pair}: missing columns {missing} in {csv_path}")

    df = df.dropna(subset=required).copy()
    times = np.array(pd.to_datetime(df["signal_time"], errors="coerce", utc=True).dt.to_pydatetime())

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy(dtype=np.int32)
    future_return = df["future_return_pips"].astype(float).to_numpy(dtype=np.float64)

    oos_prob_up: list[np.ndarray] = []
    oos_prob_down: list[np.ndarray] = []
    oos_pred_dir: list[np.ndarray] = []
    oos_ret: list[np.ndarray] = []

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
        X_val = X[val_mask]
        ret_val = future_return[val_mask]

        if len(X_train) < train_days or len(X_val) == 0:
            fold += 1
            continue

        class_weight = _build_class_weight(y_train, directional_boost)
        model = lgb.LGBMClassifier(**params, class_weight=class_weight)
        model.fit(X_train, y_train, feature_name=FEATURE_NAMES)

        proba = model.predict_proba(X_val)
        pred = np.argmax(proba, axis=1)

        oos_prob_up.append(proba[:, 0])
        oos_prob_down.append(proba[:, 2])
        oos_pred_dir.append(pred)
        oos_ret.append(ret_val)
        fold += 1

    if not oos_prob_up:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(oos_prob_up),
        np.concatenate(oos_prob_down),
        np.concatenate(oos_pred_dir),
        np.concatenate(oos_ret),
    )


def _evaluate_thresholds(
    prob_up: np.ndarray,
    prob_down: np.ndarray,
    pred_dir: np.ndarray,
    future_return: np.ndarray,
    dt_long: float,
    bt_long: float,
    dt_short: float,
    bt_short: float,
) -> dict:
    long_exec = (pred_dir == 0) & (prob_up >= dt_long) & (prob_down < bt_long)
    short_exec = (pred_dir == 2) & (prob_down >= dt_short) & (prob_up < bt_short)
    trades = long_exec | short_exec

    pnl = np.where(long_exec, future_return, np.where(short_exec, -future_return, 0.0))

    pips_per_signal = float(np.mean(pnl)) if len(pnl) > 0 else 0.0
    pips_per_trade = float(np.mean(pnl[trades])) if np.any(trades) else 0.0
    trade_rate = float(np.mean(trades)) if len(trades) > 0 else 0.0

    return {
        "dt_long": dt_long,
        "bt_long": bt_long,
        "dt_short": dt_short,
        "bt_short": bt_short,
        "pips_per_signal": pips_per_signal,
        "pips_per_trade": pips_per_trade,
        "trade_rate": trade_rate,
        "n_trades": int(np.sum(trades)),
        "n_samples": int(len(pnl)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast threshold optimization on WFV OOS predictions")
    parser.add_argument("--work-dir", default="data", help="Directory containing *_bootstrap_train.csv")
    parser.add_argument("--pairs", default=",".join(PAIRS), help="Comma-separated pairs")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--dt-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--bt-grid", default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--min-trade-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    dt_grid = [float(x.strip()) for x in args.dt_grid.split(",") if x.strip()]
    bt_grid = [float(x.strip()) for x in args.bt_grid.split(",") if x.strip()]

    cfg = get_trading_config()
    cur_thresholds = cfg.get("ml", {}).get("prediction_thresholds", {})

    work_dir = Path(args.work_dir)

    report: dict[str, dict] = {}
    recommended: dict[str, dict] = {}

    for pair in pairs:
        csv_path = work_dir / f"{pair}_bootstrap_train.csv"
        if not csv_path.exists():
            continue

        prob_up, prob_down, pred_dir, future_return = _collect_oos_predictions(
            pair=pair,
            csv_path=csv_path,
            train_days=args.train_days,
            val_days=args.val_days,
        )
        if len(prob_up) == 0:
            continue

        pair_cfg = cur_thresholds.get(pair, {})
        long_cfg = pair_cfg.get("long", {})
        short_cfg = pair_cfg.get("short", {})

        cur_dt_long = float(long_cfg.get("direction_threshold", 0.45))
        cur_bt_long = float(long_cfg.get("block_threshold", 0.35))
        cur_dt_short = float(short_cfg.get("direction_threshold", 0.45))
        cur_bt_short = float(short_cfg.get("block_threshold", 0.35))

        baseline = _evaluate_thresholds(
            prob_up, prob_down, pred_dir, future_return,
            cur_dt_long, cur_bt_long, cur_dt_short, cur_bt_short,
        )

        best_score = None
        best_row = None
        for dt_long in dt_grid:
            for bt_long in bt_grid:
                for dt_short in dt_grid:
                    for bt_short in bt_grid:
                        row = _evaluate_thresholds(
                            prob_up, prob_down, pred_dir, future_return,
                            dt_long, bt_long, dt_short, bt_short,
                        )
                        if row["trade_rate"] < args.min_trade_rate:
                            continue
                        key = (row["pips_per_signal"], row["pips_per_trade"], row["trade_rate"])
                        if best_score is None or key > best_score:
                            best_score = key
                            best_row = row

        if best_row is None:
            best_row = baseline

        report[pair] = {
            "baseline": {
                **baseline,
                "thresholds": {
                    "long": {"direction_threshold": cur_dt_long, "block_threshold": cur_bt_long},
                    "short": {"direction_threshold": cur_dt_short, "block_threshold": cur_bt_short},
                },
            },
            "best": {
                **best_row,
                "improvement_pips_per_signal": float(best_row["pips_per_signal"] - baseline["pips_per_signal"]),
                "improvement_pips_per_trade": float(best_row["pips_per_trade"] - baseline["pips_per_trade"]),
            },
        }

        recommended[pair] = {
            "long": {
                "direction_threshold": float(best_row["dt_long"]),
                "block_threshold": float(best_row["bt_long"]),
            },
            "short": {
                "direction_threshold": float(best_row["dt_short"]),
                "block_threshold": float(best_row["bt_short"]),
            },
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs,
        "report": report,
        "recommended_thresholds": recommended,
    }

    out_path = work_dir / f"threshold_search_results_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())
    print(json.dumps(recommended, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
