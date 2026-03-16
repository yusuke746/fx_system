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
from loguru import logger

from config.settings import get_trading_config
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
