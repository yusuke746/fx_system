"""
TradingView CSV から初期学習用データセットを生成する。

想定入力:
  - Pine の CSVエクスポートモードで出力したチャートCSV
    - CSV_* 列に 38特徴量が含まれる

出力:
    - 学習用CSV（38特徴量 + label + 補助列）

使い方:
  python maintenance/build_bootstrap_dataset.py \
    --input data/USDJPY_chart.csv \
    --output data/USDJPY_bootstrap_train.csv \
    --pair USDJPY \
    --horizon-bars 16
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from core.time_manager import get_session_flag


FEATURE_COLS = [
    "CSV_fvg_4h_zone_active",
    "CSV_ob_4h_zone_active",
    "CSV_liq_sweep_1h",
    "CSV_liq_sweep_qualified",
    "CSV_bos_1h",
    "CSV_choch_1h",
    "CSV_msb_15m_confirmed",
    "CSV_mtf_confluence",
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
    "CSV_fvg_4h_size_pips",
    "CSV_ob_4h_size_pips",
    "CSV_sweep_depth_atr_ratio",
    "CSV_prior_candle_body_ratio",
    "CSV_consecutive_same_dir",
    "CSV_sweep_pending_bars",
    "CSV_open_positions_count",
    "CSV_max_dd_24h",
    "CSV_calendar_risk_score",
    "CSV_sentiment_score",
]

REQUIRED_BASE_COLS = [
    "CSV_fvg_4h_zone_active",
    "CSV_ob_4h_zone_active",
    "CSV_liq_sweep_1h",
    "CSV_liq_sweep_qualified",
    "CSV_bos_1h",
    "CSV_choch_1h",
    "CSV_msb_15m_confirmed",
    "CSV_mtf_confluence",
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
    "CSV_sweep_pending_bars",
    "CSV_open_positions_count",
    "CSV_max_dd_24h",
    "CSV_calendar_risk_score",
    "CSV_sentiment_score",
]

OPTIONAL_NEW_COLS_WITH_DEFAULT = {
    "CSV_fvg_4h_size_pips": 0.0,
    "CSV_ob_4h_size_pips": 0.0,
    "CSV_sweep_depth_atr_ratio": 0.0,
}

RENAME_TO_MODEL = {
    "CSV_fvg_4h_zone_active": "fvg_4h_zone_active",
    "CSV_ob_4h_zone_active": "ob_4h_zone_active",
    "CSV_liq_sweep_1h": "liq_sweep_1h",
    "CSV_liq_sweep_qualified": "liq_sweep_qualified",
    "CSV_bos_1h": "bos_1h",
    "CSV_choch_1h": "choch_1h",
    "CSV_msb_15m_confirmed": "msb_15m_confirmed",
    "CSV_mtf_confluence": "mtf_confluence",
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
    "CSV_fvg_4h_size_pips": "fvg_4h_size_pips",
    "CSV_ob_4h_size_pips": "ob_4h_size_pips",
    "CSV_sweep_depth_atr_ratio": "sweep_depth_atr_ratio",
    "CSV_prior_candle_body_ratio": "prior_candle_body_ratio",
    "CSV_consecutive_same_dir": "consecutive_same_dir",
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


def _calc_sl_tp_pips(atr_value: float, pair: str) -> tuple[float, float]:
    """ATR値からSL/TPのpips幅を計算する（config.jsonのriskパラメータ相当）。"""
    sl_multiplier = 1.5
    sl_min_pips = 15.0
    sl_max_pips = 35.0
    sl_cap_atr_multiplier = 1.0
    tp_min_ratio = 1.5

    if pair in ("USDJPY", "GBPJPY"):
        atr_pips = atr_value * 100
    else:
        atr_pips = atr_value * 10000

    if sl_cap_atr_multiplier > 0:
        sl_upper_cap = max(sl_min_pips, atr_pips * sl_cap_atr_multiplier)
    else:
        sl_upper_cap = sl_max_pips

    sl_pips = max(sl_min_pips, min(atr_pips * sl_multiplier, sl_upper_cap))
    tp_pips = max(sl_pips * tp_min_ratio, sl_pips * 1.5)
    return sl_pips, tp_pips


def _simulate_label(
    row_idx: int,
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
    sl_pips: float,
    tp_pips: float,
    pair: str,
    horizon_bars: int = 16,
) -> int:
    """
    エントリー後のhorizon_bars本分のhigh/lowを参照し、
    SL/TPのどちらが先に当たるかでラベルを決定する。

    Returns:
        0: TP到達（勝ち）
        1: horizon_bars経過しても未決（flat）
        2: SL到達（負け）
    """
    pip_unit = _pip_unit(pair)
    future_bars = df.iloc[row_idx + 1: row_idx + horizon_bars + 1]

    if direction == "long":
        tp_price = entry_price + tp_pips * pip_unit
        sl_price = entry_price - sl_pips * pip_unit
        for _, bar in future_bars.iterrows():
            if bar["high"] >= tp_price:
                return 0
            if bar["low"] <= sl_price:
                return 2
    else:  # short
        tp_price = entry_price - tp_pips * pip_unit
        sl_price = entry_price + sl_pips * pip_unit
        for _, bar in future_bars.iterrows():
            if bar["low"] <= tp_price:
                return 0
            if bar["high"] >= sl_price:
                return 2

    return 1


def _find_time_col(df: pd.DataFrame) -> str:
    for c in ("time", "Time", "timestamp", "Timestamp", "date", "Date"):
        if c in df.columns:
            return c
    return ""


def _session_flag_from_timestamp(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return get_session_flag(ts.to_pydatetime())


def _day_of_week_from_timestamp(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return int(ts.weekday())


def build_dataset(input_path: Path, output_path: Path, pair: str, horizon_bars: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    missing_required = [c for c in REQUIRED_BASE_COLS + ["CSV_direction", "CSV_close_price"] if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    for col, default_value in OPTIONAL_NEW_COLS_WITH_DEFAULT.items():
        if col not in df.columns:
            warnings.warn(
                f"{input_path}: optional column '{col}' not found. filling default={default_value}.",
                UserWarning,
                stacklevel=2,
            )
            df[col] = default_value

    if "high" not in df.columns or "low" not in df.columns:
        warnings.warn(
            f"{input_path}: 'high'/'low' columns not found. Using 'close' as substitute for SL/TP simulation.",
            UserWarning,
            stacklevel=2,
        )
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]

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

    def _label_for_row(i: int, row: pd.Series) -> tuple[int, float]:
        sl_pips, tp_pips = _calc_sl_tp_pips(float(row["CSV_atr_14"]), pair)
        label = _simulate_label(
            row_idx=i,
            df=df,
            direction="long" if row["CSV_direction"] == 1.0 else "short",
            entry_price=float(row["CSV_close_price"]),
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pair=pair,
            horizon_bars=horizon_bars,
        )
        return label

    results = [_label_for_row(i, row) for i, row in signal_df.iterrows()]
    signal_df["label"] = results

    signal_df["pair"] = pair
    signal_df["direction"] = signal_df["CSV_direction"].map({1.0: "long", 2.0: "short"})
    signal_df["session_type"] = signal_df["signal_time"].apply(_session_flag_from_timestamp)
    signal_df["day_of_week"] = signal_df["signal_time"].apply(_day_of_week_from_timestamp)

    model_df = signal_df[["pair", "signal_time", "direction", "CSV_close_price", *FEATURE_COLS, "future_close_price", "future_return_pips", "session_type", "day_of_week", "label"]].copy()
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
