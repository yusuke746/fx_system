"""
週末自動最適化モジュール

■ 現状方針
    - 旧 scale-out / 固定TP 倍率ロジックは廃止された。
    - 既存 trade 履歴だけでは、新しいダイナミック・エグジット戦略の
        反実仮想（別パラメータならどう決済されたか）を正しく再現できない。
    - そのため、誤った自動最適化で戦略を壊さないよう、旧パラメータの
        グリッド探索とロールバックは停止する。

■ 時刻基準
    - 最適化実行時刻: 保存基準（UTC）で記録
    - Discord通知: 表示基準（JST）
"""

import sqlite3
from datetime import datetime
from loguru import logger

from config.settings import get_trading_config, save_trading_config
from core.database import get_recent_trades, insert_optimization


def _clamp(v: float, low: float, high: float) -> float:
    return max(low, min(v, high))


def run_weekend_optimization(db_conn: sqlite3.Connection) -> dict:
    """
    週末自動最適化ループの実行。

    Returns:
        {"applied": bool, "reason": str, "params": dict, ...}
    """
    config = get_trading_config()
    opt_cfg = config["optimization"]

    if not opt_cfg.get("auto_optimize", True):
        logger.info("Auto-optimization disabled in config")
        return {"applied": False, "reason": "disabled"}

    optimize_days = opt_cfg["optimize_window_days"]
    validate_days = opt_cfg["validate_window_days"]
    sample_count = len(get_recent_trades(db_conn, days=optimize_days))

    # 新しい出口戦略は trade 結果だけでは再現不能なため、自動最適化を停止する。
    # ここで旧パラメータを動かすと、現行ロジックと整合しない値が復活してしまう。
    logger.info("Weekend optimization skipped: dynamic exit strategy requires manual/feature-level validation")
    insert_optimization(db_conn, {
        "optimize_window_days": optimize_days,
        "validate_window_days": validate_days,
        "sl_multiplier": config["risk"].get("sl_multiplier"),
        "exit_prob_threshold": config["risk"].get("exit_prob_threshold"),
        "time_decay_minutes": config["risk"].get("time_decay_minutes"),
        "time_decay_min_profit_atr": config["risk"].get("time_decay_min_profit_atr"),
        "sample_count": sample_count,
        "was_applied": False,
        "rollback_reason": "dynamic_exit_strategy_active",
    })
    return {
        "applied": False,
        "reason": "dynamic_exit_strategy_active",
        "sample_count": sample_count,
    }


def check_weekly_rollback(db_conn: sqlite3.Connection) -> bool:
    """
    適用翌週の勝率が前週比-10%以上悪化した場合、前週パラメータに自動復元する。

    Returns:
        True if rollback was executed
    """
    logger.info("Weekly rollback skipped: dynamic exit strategy optimization is disabled")
    return False


def auto_tune_execution_noise(
    db_conn: sqlite3.Connection,
    lookback_days: int = 7,
    min_samples: int = 20,
) -> dict:
    """
    執行ノイズ抑制パラメータを週次で微調整する。

    対象は execution layer のみ（AI判定ロジックは変更しない）。
    """
    config = get_trading_config()
    risk = config.setdefault("risk", {})

    trades = get_recent_trades(db_conn, days=lookback_days)
    closed = [t for t in trades if t.get("close_time")]
    sample_count = len(closed)
    if sample_count < min_samples:
        reason = f"insufficient_samples({sample_count}<{min_samples})"
        logger.info(f"Execution noise tuning skipped: {reason}")
        return {"applied": False, "reason": reason, "sample_count": sample_count}

    def _hold_minutes(row: dict) -> float:
        try:
            ot = datetime.fromisoformat(str(row.get("open_time")))
            ct = datetime.fromisoformat(str(row.get("close_time")))
            return max(0.0, (ct - ot).total_seconds() / 60.0)
        except Exception:
            return 0.0

    reason_counts: dict[str, int] = {}
    hold_minutes = []
    early_noise_count = 0
    for row in closed:
        reason = str(row.get("exit_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        hm = _hold_minutes(row)
        hold_minutes.append(hm)
        if hm <= 8 and reason in {"time_decay", "trailing"}:
            early_noise_count += 1

    avg_hold = sum(hold_minutes) / len(hold_minutes) if hold_minutes else 0.0
    trailing_ratio = reason_counts.get("trailing", 0) / sample_count
    structural_tp_ratio = reason_counts.get("structural_tp", 0) / sample_count
    early_noise_ratio = early_noise_count / sample_count

    old = {
        "exit_prob_stale_minutes": float(risk.get("exit_prob_stale_minutes", 30)),
        "trailing_update_cooldown_seconds": float(risk.get("trailing_update_cooldown_seconds", 30)),
        "trailing_min_step_pips": float(risk.get("trailing_min_step_pips", 2.0)),
    }
    new = dict(old)

    # ノイズが多い週は保守側へ。
    if early_noise_ratio >= 0.28:
        new["exit_prob_stale_minutes"] = _clamp(new["exit_prob_stale_minutes"] + 5, 10, 90)
        new["trailing_update_cooldown_seconds"] = _clamp(new["trailing_update_cooldown_seconds"] + 10, 10, 180)
        new["trailing_min_step_pips"] = _clamp(new["trailing_min_step_pips"] + 0.2, 0.5, 6.0)
    # 利確主導でノイズが少ない週はわずかに機敏側へ。
    elif structural_tp_ratio >= 0.30 and early_noise_ratio <= 0.15:
        new["exit_prob_stale_minutes"] = _clamp(new["exit_prob_stale_minutes"] - 5, 10, 90)
        new["trailing_update_cooldown_seconds"] = _clamp(new["trailing_update_cooldown_seconds"] - 5, 10, 180)
        new["trailing_min_step_pips"] = _clamp(new["trailing_min_step_pips"] - 0.1, 0.5, 6.0)

    changed = any(abs(new[k] - old[k]) > 1e-9 for k in old.keys())
    if not changed:
        return {
            "applied": False,
            "reason": "no_change_needed",
            "sample_count": sample_count,
            "metrics": {
                "avg_hold_minutes": round(avg_hold, 2),
                "trailing_ratio": round(trailing_ratio, 3),
                "structural_tp_ratio": round(structural_tp_ratio, 3),
                "early_noise_ratio": round(early_noise_ratio, 3),
            },
        }

    risk["exit_prob_stale_minutes"] = int(round(new["exit_prob_stale_minutes"]))
    risk["trailing_update_cooldown_seconds"] = int(round(new["trailing_update_cooldown_seconds"]))
    risk["trailing_min_step_pips"] = round(float(new["trailing_min_step_pips"]), 2)
    save_trading_config(config)

    logger.info(
        "Execution noise params tuned: "
        f"stale={old['exit_prob_stale_minutes']}→{risk['exit_prob_stale_minutes']}, "
        f"cooldown={old['trailing_update_cooldown_seconds']}→{risk['trailing_update_cooldown_seconds']}, "
        f"step={old['trailing_min_step_pips']}→{risk['trailing_min_step_pips']}"
    )

    return {
        "applied": True,
        "sample_count": sample_count,
        "old": old,
        "new": {
            "exit_prob_stale_minutes": risk["exit_prob_stale_minutes"],
            "trailing_update_cooldown_seconds": risk["trailing_update_cooldown_seconds"],
            "trailing_min_step_pips": risk["trailing_min_step_pips"],
        },
        "metrics": {
            "avg_hold_minutes": round(avg_hold, 2),
            "trailing_ratio": round(trailing_ratio, 3),
            "structural_tp_ratio": round(structural_tp_ratio, 3),
            "early_noise_ratio": round(early_noise_ratio, 3),
        },
    }


def auto_tune_exit_mix(
    db_conn: sqlite3.Connection,
    lookback_days: int = 14,
    min_samples: int = 30,
) -> dict:
    """exit理由の構成比に応じて、time decay 系パラメータを週次調整する。"""
    config = get_trading_config()
    risk = config.setdefault("risk", {})

    trades = get_recent_trades(db_conn, days=lookback_days)
    closed = [t for t in trades if t.get("close_time")]
    sample_count = len(closed)
    if sample_count < min_samples:
        reason = f"insufficient_samples({sample_count}<{min_samples})"
        logger.info(f"Exit mix tuning skipped: {reason}")
        return {"applied": False, "reason": reason, "sample_count": sample_count}

    reason_counts: dict[str, int] = {}
    reason_pips_sum: dict[str, float] = {}
    total_pips = 0.0
    for row in closed:
        reason = str(row.get("exit_reason") or "unknown")
        pips = float(row.get("pnl_pips") or 0.0)
        total_pips += pips
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reason_pips_sum[reason] = reason_pips_sum.get(reason, 0.0) + pips

    def _ratio(name: str) -> float:
        return reason_counts.get(name, 0) / sample_count

    def _avg_pips(name: str) -> float:
        n = reason_counts.get(name, 0)
        if n <= 0:
            return 0.0
        return reason_pips_sum.get(name, 0.0) / n

    time_exit_ratio = _ratio("time_exit")
    atr_sl_ratio = _ratio("atr_sl")
    structural_tp_ratio = _ratio("structural_tp")
    avg_time_exit_pips = _avg_pips("time_exit")
    avg_atr_sl_pips = _avg_pips("atr_sl")
    avg_total_pips = total_pips / sample_count

    old = {
        "time_decay_minutes": float(risk.get("time_decay_minutes", 90)),
        "time_decay_hold_atr_threshold": float(risk.get("time_decay_hold_atr_threshold", 0.15)),
        "time_decay_min_profit_atr": float(risk.get("time_decay_min_profit_atr", 0.5)),
    }
    new = dict(old)

    # time_exit と atr_sl の損失比率が高い週は、時間系エグジットをやや前倒しする。
    # hold_atr_threshold を上げる → より大きい含み益がないと保持しない → time decay が発動しやすくなる。
    if (time_exit_ratio >= 0.42 and avg_time_exit_pips < -1.5) or (
        atr_sl_ratio >= 0.20 and avg_atr_sl_pips < -12.0
    ):
        new["time_decay_minutes"] = _clamp(new["time_decay_minutes"] - 10, 45, 180)
        new["time_decay_hold_atr_threshold"] = _clamp(new["time_decay_hold_atr_threshold"] + 0.02, 0.05, 0.35)
        new["time_decay_min_profit_atr"] = _clamp(new["time_decay_min_profit_atr"] - 0.05, 0.20, 1.00)
    # 利確比率が高く、全体期待値もプラス週はわずかに保持寄りへ。
    # hold_atr_threshold を下げる → 少ない含み益でも保持される → time decay が発動しにくくなる。
    elif structural_tp_ratio >= 0.30 and avg_total_pips > 1.0 and atr_sl_ratio <= 0.10:
        new["time_decay_minutes"] = _clamp(new["time_decay_minutes"] + 10, 45, 180)
        new["time_decay_hold_atr_threshold"] = _clamp(new["time_decay_hold_atr_threshold"] - 0.02, 0.05, 0.35)
        new["time_decay_min_profit_atr"] = _clamp(new["time_decay_min_profit_atr"] + 0.03, 0.20, 1.00)

    changed = any(abs(new[k] - old[k]) > 1e-9 for k in old.keys())
    if not changed:
        return {
            "applied": False,
            "reason": "no_change_needed",
            "sample_count": sample_count,
            "metrics": {
                "time_exit_ratio": round(time_exit_ratio, 3),
                "atr_sl_ratio": round(atr_sl_ratio, 3),
                "structural_tp_ratio": round(structural_tp_ratio, 3),
                "avg_time_exit_pips": round(avg_time_exit_pips, 2),
                "avg_atr_sl_pips": round(avg_atr_sl_pips, 2),
                "avg_total_pips": round(avg_total_pips, 2),
            },
        }

    risk["time_decay_minutes"] = int(round(new["time_decay_minutes"]))
    risk["time_decay_hold_atr_threshold"] = round(float(new["time_decay_hold_atr_threshold"]), 3)
    risk["time_decay_min_profit_atr"] = round(float(new["time_decay_min_profit_atr"]), 3)
    save_trading_config(config)

    logger.info(
        "Exit mix params tuned: "
        f"time_decay_minutes={old['time_decay_minutes']}→{risk['time_decay_minutes']}, "
        f"hold_atr={old['time_decay_hold_atr_threshold']}→{risk['time_decay_hold_atr_threshold']}, "
        f"min_profit_atr={old['time_decay_min_profit_atr']}→{risk['time_decay_min_profit_atr']}"
    )

    return {
        "applied": True,
        "sample_count": sample_count,
        "old": old,
        "new": {
            "time_decay_minutes": risk["time_decay_minutes"],
            "time_decay_hold_atr_threshold": risk["time_decay_hold_atr_threshold"],
            "time_decay_min_profit_atr": risk["time_decay_min_profit_atr"],
        },
        "metrics": {
            "time_exit_ratio": round(time_exit_ratio, 3),
            "atr_sl_ratio": round(atr_sl_ratio, 3),
            "structural_tp_ratio": round(structural_tp_ratio, 3),
            "avg_time_exit_pips": round(avg_time_exit_pips, 2),
            "avg_atr_sl_pips": round(avg_atr_sl_pips, 2),
            "avg_total_pips": round(avg_total_pips, 2),
        },
    }


def auto_tune_directional_allocation(
    db_conn: sqlite3.Connection,
    lookback_days: int = 14,
    min_samples: int = 30,
    min_samples_per_direction: int = 8,
) -> dict:
    """通貨ペア×方向の成績に応じて prediction_thresholds を週次調整する。"""
    config = get_trading_config()
    ml_cfg = config.setdefault("ml", {})
    thresholds = ml_cfg.setdefault("prediction_thresholds", {})

    trades = get_recent_trades(db_conn, days=lookback_days)
    closed = [t for t in trades if t.get("close_time") and str(t.get("direction") or "") in {"long", "short"}]
    sample_count = len(closed)
    if sample_count < min_samples:
        reason = f"insufficient_samples({sample_count}<{min_samples})"
        logger.info(f"Directional allocation tuning skipped: {reason}")
        return {"applied": False, "reason": reason, "sample_count": sample_count}

    agg: dict[tuple[str, str], dict[str, float]] = {}
    for row in closed:
        pair = str(row.get("pair") or "")
        direction = str(row.get("direction") or "")
        key = (pair, direction)
        pips = float(row.get("pnl_pips") or 0.0)
        a = agg.setdefault(key, {"count": 0.0, "pips_sum": 0.0, "wins": 0.0})
        a["count"] += 1
        a["pips_sum"] += pips
        if pips > 0:
            a["wins"] += 1

    changes: list[dict] = []
    for (pair, direction), s in agg.items():
        count = int(s["count"])
        if count < min_samples_per_direction:
            continue

        avg_pips = s["pips_sum"] / s["count"]
        winrate = s["wins"] / s["count"]

        pair_cfg = thresholds.setdefault(pair, {})
        dir_cfg = pair_cfg.setdefault(direction, {})
        old_dt = float(dir_cfg.get("direction_threshold", 0.45))
        old_me = float(dir_cfg.get("min_edge", 0.08))
        new_dt = old_dt
        new_me = old_me

        # 負け方向はフィルタを厳格化する。
        if avg_pips <= -3.0:
            new_dt = _clamp(old_dt + 0.03, 0.35, 0.75)
            new_me = _clamp(old_me + 0.02, 0.04, 0.20)
        # 優位方向はわずかに通しやすくする。
        elif avg_pips >= 2.0 and winrate >= 0.45:
            new_dt = _clamp(old_dt - 0.02, 0.35, 0.75)
            new_me = _clamp(old_me - 0.01, 0.04, 0.20)

        if abs(new_dt - old_dt) > 1e-9 or abs(new_me - old_me) > 1e-9:
            dir_cfg["direction_threshold"] = round(float(new_dt), 3)
            dir_cfg["min_edge"] = round(float(new_me), 3)
            changes.append({
                "pair": pair,
                "direction": direction,
                "count": count,
                "avg_pips": round(avg_pips, 2),
                "winrate": round(winrate, 3),
                "old": {"direction_threshold": old_dt, "min_edge": old_me},
                "new": {
                    "direction_threshold": dir_cfg["direction_threshold"],
                    "min_edge": dir_cfg["min_edge"],
                },
            })

    if not changes:
        return {
            "applied": False,
            "reason": "no_change_needed",
            "sample_count": sample_count,
            "changed": [],
        }

    save_trading_config(config)
    logger.info(f"Directional allocation tuned for {len(changes)} pair-direction buckets")
    return {
        "applied": True,
        "sample_count": sample_count,
        "changed": changes,
    }
