"""
実データ自動再学習モジュール

- training_samples に保存された特徴量を使用
- MT5履歴から将来価格を取得してラベル付け
- 週次メンテで再学習を実行
"""

from datetime import datetime, timedelta, timezone
import sqlite3

import MetaTrader5 as mt5
import numpy as np
from loguru import logger

from config.settings import get_settings, get_trading_config
from core.database import (
    get_unlabeled_training_samples,
    get_labeled_training_samples,
    update_training_label,
)
from core.time_manager import now_utc, utc_to_mt5_server
from ml.lgbm_model import FEATURE_NAMES
from ml.trainer import save_model_metrics, train_model, walk_forward_validate


def _to_datetime_utc(iso_str: str) -> datetime:
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _get_future_close(pair: str, signal_time_utc: datetime, horizon_minutes: int) -> float | None:
    target_time_utc = signal_time_utc + timedelta(minutes=horizon_minutes)
    start_utc = target_time_utc - timedelta(minutes=30)
    end_utc = target_time_utc + timedelta(hours=2)

    target_time_mt5 = utc_to_mt5_server(target_time_utc)
    start_mt5 = utc_to_mt5_server(start_utc)
    end_mt5 = utc_to_mt5_server(end_utc)

    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M15, start_mt5, end_mt5)
    if rates is None or len(rates) == 0:
        return None

    for row in rates:
        bar_time_mt5 = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc).astimezone(start_mt5.tzinfo)
        if bar_time_mt5 >= target_time_mt5:
            return float(row["close"])

    return float(rates[-1]["close"])


def label_unlabeled_samples(
    db_conn: sqlite3.Connection,
    horizon_minutes: int = 240,
    horizon_per_pair: dict | None = None,
    limit: int = 5000,
) -> dict:
    """未ラベルサンプルに将来価格ベースでラベル付けを行う。"""
    settings = get_settings()

    if not mt5.initialize(
        login=settings.mt5_login,
        password=settings.mt5_password.get_secret_value(),
        server=settings.mt5_server,
    ):
        err = mt5.last_error()
        logger.error(f"MT5 initialize failed in label_unlabeled_samples: {err}")
        return {"labeled": 0, "skipped": 0, "error": f"mt5_init_failed: {err}"}

    labeled = 0
    skipped = 0

    try:
        rows = get_unlabeled_training_samples(db_conn, limit=limit)
        for sample in rows:
            try:
                signal_time = _to_datetime_utc(sample["signal_time"])
                pair_horizon = int(
                    (horizon_per_pair or {}).get(sample["pair"], horizon_minutes)
                )
                future_close = _get_future_close(sample["pair"], signal_time, pair_horizon)
                if future_close is None:
                    skipped += 1
                    continue

                close_price = float(sample.get("close_price") or 0.0)
                if close_price <= 0:
                    skipped += 1
                    continue

                pu = _pip_unit(sample["pair"])
                return_pips = (future_close - close_price) / pu
                label = _label_from_return(return_pips, float(sample.get("atr") or 0.0), sample["pair"])

                update_training_label(
                    db_conn,
                    sample_id=int(sample["id"]),
                    label=label,
                    future_close=future_close,
                    future_return_pips=float(return_pips),
                )
                labeled += 1
            except Exception as e:
                logger.warning(f"Labeling skipped for sample id={sample.get('id')}: {e}")
                skipped += 1

    finally:
        mt5.shutdown()

    logger.info(f"Labeling completed: labeled={labeled}, skipped={skipped}")
    return {"labeled": labeled, "skipped": skipped}


def _build_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for row in rows:
        if row.get("label") is None:
            continue

        X.append([
            float(row.get("fvg_4h_zone_active", 0)),
            float(row.get("ob_4h_zone_active", 0)),
            float(row.get("liq_sweep_1h", 0)),
            float(row.get("liq_sweep_qualified", 0)),
            float(row.get("bos_1h", 0)),
            float(row.get("choch_1h", 0)),
            float(row.get("msb_15m_confirmed", 0)),
            float(row.get("mtf_confluence", 0)),
            float(row.get("atr_ratio", 1.0)),
            float(row.get("bb_width", 0.0)),
            float(row.get("close_vs_ema20_4h", 0.0)),
            float(row.get("close_vs_ema50_4h", 0.0)),
            float(row.get("high_low_range_15m", 0.0)),
            float(row.get("trend_direction", 0)),
            float(row.get("momentum_long", 0)),
            float(row.get("momentum_short", 0)),
            float(row.get("macd_histogram", 0.0)),
            float(row.get("macd_signal_cross", 0)),
            float(row.get("rsi_14", 50.0)),
            float(row.get("rsi_zone", 0)),
            float(row.get("stoch_k", 50.0)),
            float(row.get("stoch_d", 50.0)),
            float(row.get("momentum_3bar", 0.0)),
            float(row.get("ob_4h_distance_pips", 0.0)),
            float(row.get("fvg_4h_fill_ratio", 0.0)),
            float(row.get("liq_sweep_strength", 0.0)),
            float(row.get("fvg_4h_size_pips", 0.0)),
            float(row.get("ob_4h_size_pips", 0.0)),
            float(row.get("sweep_depth_atr_ratio", 0.0)),
            float(row.get("prior_candle_body_ratio", 0.5)),
            float(row.get("consecutive_same_dir", 0)),
            float(row.get("sweep_pending_bars", 0)),
            float(row.get("open_positions_count", 0)),
            float(row.get("max_dd_24h", 0.0)),
            float(row.get("calendar_risk_score", 0)),
            float(row.get("sentiment_score", 0.0)),
            float(row.get("risk_appetite_score", 0.0)),
            float(row.get("usd_macro_score", 0.0)),
            float(row.get("jpy_macro_score", 0.0)),
            float(row.get("oil_shock_score", 0.0)),
            float(row.get("geopolitical_risk_score", 0.0)),
            float(row.get("session_type", 0)),
            float(row.get("day_of_week", 0)),
        ])
        y.append(int(row["label"]))

    if not X:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), np.empty((0,), dtype=np.int32)

    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.int32)

    if X_np.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Feature count mismatch: got {X_np.shape[1]}, expected {len(FEATURE_NAMES)}")

    return X_np, y_np


def retrain_models_from_db(
    db_conn: sqlite3.Connection,
    model_dir: str,
    lookback_days: int = 90,
    min_samples_per_pair: int = 300,
) -> dict:
    """DBのラベル済みサンプルから通貨ペア別に再学習する。"""
    config = get_trading_config()
    ml_cfg = config.get("ml", {})
    pairs = config["system"]["pairs"]
    min_directional_samples = int(ml_cfg.get("min_directional_samples", 30))
    min_cv_accuracy = float(ml_cfg.get("min_cv_accuracy", 0.40))

    result = {
        "trained_pairs": [],
        "skipped_pairs": {},
        "validation": {},
    }

    for pair in pairs:
        rows = get_labeled_training_samples(db_conn, pair=pair, days=lookback_days)
        if len(rows) < min_samples_per_pair:
            reason = f"insufficient_samples({len(rows)}<{min_samples_per_pair})"
            result["skipped_pairs"][pair] = reason
            logger.warning(f"Retrain skipped for {pair}: {reason}")
            continue

        X, y = _build_xy(rows)

        class_counts = np.bincount(y, minlength=3)
        up_count = int(class_counts[0])
        flat_count = int(class_counts[1])
        down_count = int(class_counts[2])
        if up_count < min_directional_samples or down_count < min_directional_samples:
            logger.warning(
                f"Directional sample imbalance for {pair}: "
                f"up={up_count}, flat={flat_count}, down={down_count}, "
                f"required={min_directional_samples}. Continue training (ML-priority mode)."
            )

        val = walk_forward_validate(
            X,
            y,
            signal_times=[
                datetime.fromisoformat(r["signal_time"]) for r in rows
                if r.get("signal_time")
            ],
            pair=pair,
        )
        cv_acc = float(val.get("accuracy", 0.0))
        if cv_acc < min_cv_accuracy:
            logger.warning(
                f"Low CV accuracy for {pair}: {cv_acc:.4f}<{min_cv_accuracy:.4f}. "
                f"Continue training (ML-priority mode)."
            )

        train_model(X, y, pair=pair, model_dir=model_dir)
        save_model_metrics(
            pair=pair,
            model_dir=model_dir,
            metrics={
                "accuracy": float(val.get("accuracy", 0.0)),
                "balanced_accuracy": float(val.get("balanced_accuracy", 0.0)),
                "majority_baseline": float(val.get("majority_baseline", 0.0)),
                "samples": len(rows),
                "source": "weekly_retraining",
            },
        )

        result["trained_pairs"].append(pair)
        result["validation"][pair] = val
        logger.info(f"Retrained model for {pair}: samples={len(rows)} acc={val.get('accuracy', 0):.4f}")

    return result


def run_weekly_retraining(db_conn: sqlite3.Connection) -> dict:
    """週次バッチ: ラベル付け→再学習を一括実行。"""
    settings = get_settings()
    config = get_trading_config()

    horizon_minutes = int(config.get("ml", {}).get("label_horizon_minutes", 240))
    horizon_per_pair: dict = config.get("ml", {}).get("label_horizon_minutes_per_pair", {})
    min_samples_per_pair = int(config.get("ml", {}).get("min_samples_per_pair", 300))

    labeling = label_unlabeled_samples(db_conn, horizon_minutes=horizon_minutes, horizon_per_pair=horizon_per_pair)
    retrain = retrain_models_from_db(
        db_conn,
        model_dir=settings.model_dir,
        lookback_days=90,
        min_samples_per_pair=min_samples_per_pair,
    )

    return {
        "labeling": labeling,
        "retrain": retrain,
        "run_at_utc": now_utc().isoformat(),
    }
