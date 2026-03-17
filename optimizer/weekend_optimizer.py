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
        if hm <= 8 and reason in {"prob_decay", "time_decay", "trailing"}:
            early_noise_count += 1

    avg_hold = sum(hold_minutes) / len(hold_minutes) if hold_minutes else 0.0
    prob_decay_ratio = reason_counts.get("prob_decay", 0) / sample_count
    trailing_ratio = reason_counts.get("trailing", 0) / sample_count
    structural_tp_ratio = reason_counts.get("structural_tp", 0) / sample_count
    early_noise_ratio = early_noise_count / sample_count

    old = {
        "exit_prob_stale_minutes": float(risk.get("exit_prob_stale_minutes", 30)),
        "trailing_update_cooldown_seconds": float(risk.get("trailing_update_cooldown_seconds", 30)),
        "trailing_min_step_pips": float(risk.get("trailing_min_step_pips", 2.0)),
    }
    new = dict(old)

    def _clamp(v: float, low: float, high: float) -> float:
        return max(low, min(v, high))

    # ノイズが多い週は保守側へ。
    if early_noise_ratio >= 0.28 or (prob_decay_ratio >= 0.30 and avg_hold <= 20):
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
                "prob_decay_ratio": round(prob_decay_ratio, 3),
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
            "prob_decay_ratio": round(prob_decay_ratio, 3),
            "trailing_ratio": round(trailing_ratio, 3),
            "structural_tp_ratio": round(structural_tp_ratio, 3),
            "early_noise_ratio": round(early_noise_ratio, 3),
        },
    }
