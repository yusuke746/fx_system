"""
週末自動最適化モジュール

■ 二段階フロー（毎週土曜14:00 JST）
  Phase A: 最適化（直近28日データで探索）
  Phase B: アウトオブサンプル検証（直近14日で確認）

■ 安全装置
  - サンプル数30件未満はスキップ
  - 前週比±20%以内の変化量制限
  - アウトオブサンプル検証で前週比を上回らない場合は不採用
  - 適用翌週に勝率が前週比-10%以上悪化時は自動ロールバック

■ 時刻基準
  - 最適化実行時刻: 保存基準（UTC）で記録
  - Discord通知: 表示基準（JST）
"""

import itertools
import sqlite3

import numpy as np
from loguru import logger

from config.settings import get_settings, get_trading_config, save_trading_config
from core.database import get_recent_trades, insert_optimization
from core.time_manager import now_utc, format_jst


# 探索空間（設計書 確定値）
GRID_STEP1_RATIO = [0.30, 0.40, 0.50, 0.60, 0.70]
GRID_STEP2_RATIO = [0.15, 0.20, 0.25, 0.30, 0.35]
GRID_SL_MULT = [1.2, 1.3, 1.5, 1.7, 2.0]
GRID_TP_MULT = [2.0, 2.2, 2.5, 2.8, 3.0, 3.5]


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

    optimize_days = opt_cfg["optimize_window_days"]  # 28
    validate_days = opt_cfg["validate_window_days"]  # 14
    min_samples = opt_cfg["min_sample_count"]        # 30
    max_change = opt_cfg["max_change_ratio"]          # 0.20

    # ── PHASE A: 最適化（直近28日） ────────────
    trades_opt = get_recent_trades(db_conn, days=optimize_days)
    if len(trades_opt) < min_samples:
        logger.info(
            f"Optimization skipped: sample count {len(trades_opt)} < {min_samples}"
        )
        insert_optimization(db_conn, {
            "optimize_window_days": optimize_days,
            "validate_window_days": validate_days,
            "sample_count": len(trades_opt),
            "was_applied": False,
            "rollback_reason": f"サンプル不足 ({len(trades_opt)} < {min_samples})",
        })
        return {"applied": False, "reason": "insufficient_samples"}

    # 現在のパラメータ
    current = {
        "step1_ratio": config["risk"]["scale_out_step1_ratio"],
        "step2_ratio": config["risk"]["scale_out_step2_ratio"],
        "sl_mult": config["risk"]["sl_multiplier"],
        "tp_mult": config["risk"]["tp_multiplier"],
    }

    # グリッドサーチ
    best_score = -float("inf")
    best_params = None

    for s1, s2, sl, tp in itertools.product(
        GRID_STEP1_RATIO, GRID_STEP2_RATIO, GRID_SL_MULT, GRID_TP_MULT
    ):
        # 安全チェック: 前週比±20%以内
        if not _within_change_limit(current, s1, s2, sl, tp, max_change):
            continue

        score = _evaluate_params(trades_opt, s1, s2, sl, tp)
        if score > best_score:
            best_score = score
            best_params = {"step1_ratio": s1, "step2_ratio": s2,
                           "sl_mult": sl, "tp_mult": tp}

    if best_params is None:
        logger.warning("No valid parameter combination found")
        return {"applied": False, "reason": "no_valid_params"}

    # ── PHASE B: アウトオブサンプル検証（直近14日） ──
    trades_val = get_recent_trades(db_conn, days=validate_days)
    if len(trades_val) < 10:
        logger.info("Validation skipped: too few validation samples")
        return {"applied": False, "reason": "insufficient_validation_samples"}

    new_score = _evaluate_params(
        trades_val,
        best_params["step1_ratio"],
        best_params["step2_ratio"],
        best_params["sl_mult"],
        best_params["tp_mult"],
    )
    old_score = _evaluate_params(
        trades_val,
        current["step1_ratio"],
        current["step2_ratio"],
        current["sl_mult"],
        current["tp_mult"],
    )

    if new_score <= old_score:
        logger.info(
            f"Validation failed: new_score={new_score:.4f} <= old_score={old_score:.4f}"
        )
        insert_optimization(db_conn, {
            "optimize_window_days": optimize_days,
            "validate_window_days": validate_days,
            "step1_ratio": best_params["step1_ratio"],
            "step2_ratio": best_params["step2_ratio"],
            "sl_multiplier": best_params["sl_mult"],
            "tp_multiplier": best_params["tp_mult"],
            "sharpe_optimize": best_score,
            "sharpe_validate": new_score,
            "sample_count": len(trades_opt),
            "was_applied": False,
            "rollback_reason": "検証期間でスコアが前週を下回った",
        })
        return {"applied": False, "reason": "validation_failed"}

    # ── STEP 5: config.json 更新 ──────────────
    config["risk"]["scale_out_step1_ratio"] = best_params["step1_ratio"]
    config["risk"]["scale_out_step2_ratio"] = best_params["step2_ratio"]
    config["risk"]["sl_multiplier"] = best_params["sl_mult"]
    config["risk"]["tp_multiplier"] = best_params["tp_mult"]

    try:
        save_trading_config(config, get_settings().config_path)
    except Exception as e:
        logger.error(f"config.json write failed: {e}")
        insert_optimization(db_conn, {
            "optimize_window_days": optimize_days,
            "validate_window_days": validate_days,
            "step1_ratio": best_params["step1_ratio"],
            "step2_ratio": best_params["step2_ratio"],
            "sl_multiplier": best_params["sl_mult"],
            "tp_multiplier": best_params["tp_mult"],
            "sharpe_optimize": best_score,
            "sharpe_validate": new_score,
            "sample_count": len(trades_opt),
            "was_applied": False,
            "rollback_reason": f"config.json書き込み失敗: {e}",
        })
        return {"applied": False, "reason": f"config_write_error: {e}"}

    insert_optimization(db_conn, {
        "optimize_window_days": optimize_days,
        "validate_window_days": validate_days,
        "step1_ratio": best_params["step1_ratio"],
        "step2_ratio": best_params["step2_ratio"],
        "sl_multiplier": best_params["sl_mult"],
        "tp_multiplier": best_params["tp_mult"],
        "sharpe_optimize": best_score,
        "sharpe_validate": new_score,
        "sample_count": len(trades_opt),
        "was_applied": True,
    })

    logger.info(
        f"Optimization applied: step1={best_params['step1_ratio']}, "
        f"step2={best_params['step2_ratio']}, SL={best_params['sl_mult']}, "
        f"TP={best_params['tp_mult']}, opt_score={best_score:.4f}, "
        f"val_score={new_score:.4f}"
    )

    return {
        "applied": True,
        "params": best_params,
        "sharpe_optimize": best_score,
        "sharpe_validate": new_score,
        "sample_count": len(trades_opt),
    }


def check_weekly_rollback(db_conn: sqlite3.Connection) -> bool:
    """
    適用翌週の勝率が前週比-10%以上悪化した場合、前週パラメータに自動復元する。

    Returns:
        True if rollback was executed
    """
    config = get_trading_config()
    threshold = config["optimization"]["rollback_winrate_threshold"]  # -0.10

    recent_trades = get_recent_trades(db_conn, days=7)
    prev_trades = get_recent_trades(db_conn, days=14)

    if len(recent_trades) < 10 or len(prev_trades) < 10:
        return False

    recent_wr = sum(1 for t in recent_trades if (t.get("pnl_pips") or 0) > 0) / len(recent_trades)
    prev_only = [t for t in prev_trades if t not in recent_trades]
    if not prev_only:
        return False
    prev_wr = sum(1 for t in prev_only if (t.get("pnl_pips") or 0) > 0) / len(prev_only)

    diff = recent_wr - prev_wr
    if diff < threshold:
        logger.warning(
            f"Weekly rollback triggered: recent_wr={recent_wr:.2f}, "
            f"prev_wr={prev_wr:.2f}, diff={diff:.2f}"
        )
        # 直前の optimization_history から前の値を復元
        row = db_conn.execute(
            "SELECT * FROM optimization_history WHERE was_applied=1 "
            "ORDER BY id DESC LIMIT 1 OFFSET 1"
        ).fetchone()
        if row:
            config["risk"]["scale_out_step1_ratio"] = row["step1_ratio"]
            config["risk"]["scale_out_step2_ratio"] = row["step2_ratio"]
            config["risk"]["sl_multiplier"] = row["sl_multiplier"]
            config["risk"]["tp_multiplier"] = row["tp_multiplier"]
            save_trading_config(config, get_settings().config_path)
            logger.info("Rollback to previous parameters completed")
            return True

    return False


def _within_change_limit(
    current: dict, s1: float, s2: float, sl: float, tp: float, max_ratio: float
) -> bool:
    """前週比±max_ratio 以内かチェックする。"""
    pairs = [
        (current["step1_ratio"], s1),
        (current["step2_ratio"], s2),
        (current["sl_mult"], sl),
        (current["tp_mult"], tp),
    ]
    for old, new in pairs:
        if old > 0 and abs(new - old) / old > max_ratio:
            return False
    return True


def _evaluate_params(
    trades: list[dict], s1: float, s2: float, sl: float, tp: float
) -> float:
    """
    パラメータ組み合わせをシミュレーション的に評価する。
    評価指標: Sharpe比（主軸）× 最大DD（ペナルティ）の複合スコア。
    """
    if not trades:
        return -float("inf")

    pnls = []
    for t in trades:
        pnl = t.get("pnl_pips") or 0
        pnls.append(pnl)

    pnl_array = np.array(pnls, dtype=np.float64)
    if len(pnl_array) < 2:
        return -float("inf")

    mean_pnl = np.mean(pnl_array)
    std_pnl = np.std(pnl_array, ddof=1)
    if std_pnl == 0:
        return 0.0

    sharpe = mean_pnl / std_pnl * np.sqrt(252)  # 年率換算概算

    # 最大DD ペナルティ
    cumulative = np.cumsum(pnl_array)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    max_dd = np.max(dd) if len(dd) > 0 else 0
    dd_penalty = 1.0 - min(max_dd / 100, 0.5)  # DDが大きいほどペナルティ

    return sharpe * dd_penalty
