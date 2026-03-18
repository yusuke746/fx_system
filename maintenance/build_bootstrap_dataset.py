"""
TradingView CSV から初期学習用データセットを生成する。

想定入力:
  - Pine の CSVエクスポートモードで出力したチャートCSV
  - CSV_* 列に 35特徴量が含まれる

出力:
  - 学習用CSV（35特徴量 + label + 補助列）

使い方:
  python maintenance/build_bootstrap_dataset.py \
    --input data/USDJPY_chart.csv \
    --output data/USDJPY_bootstrap_train.csv \
    --pair USDJPY \
    --horizon-bars 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLS = [
    "CSV_fvg_4h_zone_active",
    "CSV_ob_4h_zone_active",
    "CSV_liq_sweep_1h",
    "CSV_liq_sweep_qualified",
    "CSV_bos_1h",
    "CSV_choch_1h",
    "CSV_msb_15m_confirmed",
    "CSV_mtf_confluence",
    "CSV_atr_14",
    "CSV_atr_ratio",
    "CSV_bb_width",
    "CSV_close_vs_ema20_4h",
    "CSV_close_vs_ema50_4h",
    "CSV_high_low_range_15m",
    "CSV_trend_direction",
    "CSV_momentum_long",
    "CSV_momentum_short",
    "CSV_macd_histogram",
    "CSV_macd_signal_cross",
    "CSV_rsi_14",
    "CSV_rsi_zone",
    "CSV_stoch_k",
    "CSV_stoch_d",
    "CSV_momentum_3bar",
    "CSV_ob_4h_distance_pips",
    "CSV_fvg_4h_fill_ratio",
    "CSV_liq_sweep_strength",
    "CSV_prior_candle_body_ratio",
    "CSV_consecutive_same_dir",
    "CSV_pivot_proximity",
    "CSV_sweep_pending_bars",
    "CSV_open_positions_count",
    "CSV_max_dd_24h",
    "CSV_calendar_risk_score",
    "CSV_sentiment_score",
]

RENAME_TO_MODEL = {
    "CSV_fvg_4h_zone_active": "fvg_4h_zone_active",
    "CSV_ob_4h_zone_active": "ob_4h_zone_active",
    "CSV_liq_sweep_1h": "liq_sweep_1h",
    "CSV_liq_sweep_qualified": "liq_sweep_qualified",
    "CSV_bos_1h": "bos_1h",
    "CSV_choch_1h": "choch_1h",
    "CSV_msb_15m_confirmed": "msb_15m_confirmed",
    "CSV_mtf_confluence": "mtf_confluence",
    "CSV_atr_14": "atr_14",
    "CSV_atr_ratio": "atr_ratio",
    "CSV_bb_width": "bb_width",
    "CSV_close_vs_ema20_4h": "close_vs_ema20_4h",
    "CSV_close_vs_ema50_4h": "close_vs_ema50_4h",
    "CSV_high_low_range_15m": "high_low_range_15m",
    "CSV_trend_direction": "trend_direction",
    "CSV_momentum_long": "momentum_long",
    "CSV_momentum_short": "momentum_short",
    "CSV_macd_histogram": "macd_histogram",
    "CSV_macd_signal_cross": "macd_signal_cross",
    "CSV_rsi_14": "rsi_14",
    "CSV_rsi_zone": "rsi_zone",
    "CSV_stoch_k": "stoch_k",
    "CSV_stoch_d": "stoch_d",
    "CSV_momentum_3bar": "momentum_3bar",
    "CSV_ob_4h_distance_pips": "ob_4h_distance_pips",
    "CSV_fvg_4h_fill_ratio": "fvg_4h_fill_ratio",
    "CSV_liq_sweep_strength": "liq_sweep_strength",
    "CSV_prior_candle_body_ratio": "prior_candle_body_ratio",
    "CSV_consecutive_same_dir": "consecutive_same_dir",
    "CSV_pivot_proximity": "pivot_proximity",
    "CSV_sweep_pending_bars": "sweep_pending_bars",
    "CSV_open_positions_count": "open_positions_count",
    "CSV_max_dd_24h": "max_dd_24h",
    "CSV_calendar_risk_score": "calendar_risk_score",
    "CSV_sentiment_score": "sentiment_score",
}


def _pip_unit(pair: str) -> float:
    return 0.01 if pair in ("USDJPY", "GBPJPY") else 0.0001


def _label_from_return(return_pips: float, atr_price: float, pair: str) -> int:
    atr_pips = abs(atr_price) / _pip_unit(pair) if atr_price else 0.0
    threshold = max(2.0, atr_pips * 0.30)
    if return_pips > threshold:
        return 0
    if return_pips < -threshold:
        return 2
    return 1


def _find_time_col(df: pd.DataFrame) -> str:
    for c in ("time", "Time", "timestamp", "Timestamp", "date", "Date"):
        if c in df.columns:
            return c
    return ""


def build_dataset(input_path: Path, output_path: Path, pair: str, horizon_bars: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    missing = [c for c in FEATURE_COLS + ["CSV_direction", "CSV_close_price"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    signal_df = df.dropna(subset=["CSV_direction"]).copy()

    time_col = _find_time_col(signal_df)
    if time_col:
        signal_df["signal_time"] = pd.to_datetime(signal_df[time_col], errors="coerce", utc=True)
    else:
        signal_df["signal_time"] = pd.NaT

    signal_df["future_close_price"] = signal_df["close"].shift(-horizon_bars)
    signal_df = signal_df.dropna(subset=["future_close_price"]).copy()

    pu = _pip_unit(pair)
    signal_df["future_return_pips"] = (signal_df["future_close_price"] - signal_df["CSV_close_price"]) / pu

    signal_df["label"] = signal_df.apply(
        lambda r: _label_from_return(
            return_pips=float(r["future_return_pips"]),
            atr_price=float(r["CSV_atr_14"]),
            pair=pair,
        ),
        axis=1,
    )

    signal_df["pair"] = pair
    signal_df["direction"] = signal_df["CSV_direction"].map({1.0: "long", 2.0: "short"})

    model_df = signal_df[["pair", "signal_time", "direction", "CSV_close_price", *FEATURE_COLS, "future_close_price", "future_return_pips", "label"]].copy()
    model_df = model_df.rename(columns=RENAME_TO_MODEL | {"CSV_close_price": "close_price"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_path, index=False)
    return model_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bootstrap training dataset from TradingView CSV")
    parser.add_argument("--input", required=True, help="Path to TradingView exported CSV")
    parser.add_argument("--output", required=True, help="Path to output bootstrap CSV")
    parser.add_argument("--pair", required=True, choices=["USDJPY", "EURUSD", "GBPJPY"], help="Trading pair")
    parser.add_argument("--horizon-bars", type=int, default=16, help="Future horizon in 15m bars (default: 16 = 4h)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_df = build_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        pair=args.pair,
        horizon_bars=args.horizon_bars,
    )

    label_counts = out_df["label"].value_counts().to_dict()
    print(f"rows={len(out_df)}")
    print(f"labels={label_counts}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
