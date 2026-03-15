"""
DB管理モジュール

- 全タイムスタンプは保存基準（UTC）で保存する。
- 表示時は to_jst() / format_jst() で JST に変換する。
"""

import sqlite3
from pathlib import Path

from loguru import logger

from core.time_manager import now_utc, broker_day_start_utc


def get_connection(db_path: str) -> sqlite3.Connection:
    """SQLite 接続を取得する。WAL モードを有効化。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """DBスキーマを初期化する。テーブルが存在しない場合のみ作成。"""
    conn = get_connection(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        logger.info(f"Database initialized: {db_path}")
    finally:
        conn.close()


_SCHEMA_SQL = """
-- 全タイムスタンプは UTC（保存基準）で記録する。
-- 表示時に JST へ変換すること。

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY,
    pair        TEXT NOT NULL,
    direction   TEXT NOT NULL,          -- 'long', 'short'
    open_time   TIMESTAMP NOT NULL,     -- UTC
    close_time  TIMESTAMP,              -- UTC
    open_price  REAL NOT NULL,
    close_price REAL,
    volume      REAL NOT NULL,
    sl_price    REAL,
    tp_price    REAL,
    pnl_pips    REAL,
    pnl_jpy     REAL,
    exit_reason TEXT,                   -- 'atr_sl'|'scale_out_1'|'scale_out_2'|'doten'|'time_exit'|'calendar_veto'|'trailing'
    mt5_ticket  INTEGER,
    created_at  TIMESTAMP DEFAULT (datetime('now'))  -- UTC
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY,
    pair            TEXT NOT NULL,
    signal_time     TIMESTAMP NOT NULL, -- UTC
    direction       TEXT,
    lgbm_prob_up    REAL,
    lgbm_prob_flat  REAL,
    lgbm_prob_down  REAL,
    gpt_sentiment   REAL,
    mtf_confluence  INTEGER,
    executed        BOOLEAN DEFAULT FALSE,
    veto_reason     TEXT,
    created_at      TIMESTAMP DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS api_call_log (
    id          INTEGER PRIMARY KEY,
    call_time   TIMESTAMP NOT NULL,     -- UTC
    reason      TEXT NOT NULL,          -- 'NEW_ARTICLE'|'PRICE_SPIKE'|'CACHE_EXPIRED'|'CACHE_EXPIRED_LOW'
    model       TEXT NOT NULL,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    REAL,
    created_at  TIMESTAMP DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS optimization_history (
    id                   INTEGER PRIMARY KEY,
    optimized_at         TIMESTAMP NOT NULL, -- UTC
    optimize_window_days INTEGER NOT NULL,
    validate_window_days INTEGER NOT NULL,
    step1_ratio          REAL,
    step2_ratio          REAL,
    sl_multiplier        REAL,
    tp_multiplier        REAL,
    sharpe_optimize      REAL,
    sharpe_validate      REAL,
    sample_count         INTEGER,
    was_applied          BOOLEAN DEFAULT TRUE,
    rollback_reason      TEXT,
    created_at           TIMESTAMP DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_samples (
    id                       INTEGER PRIMARY KEY,
    pair                     TEXT NOT NULL,
    signal_time              TIMESTAMP NOT NULL, -- UTC
    direction                TEXT,
    close_price              REAL,
    atr                      REAL,
    fvg_4h_zone_active       INTEGER,
    ob_4h_zone_active        INTEGER,
    liq_sweep_1h             INTEGER,
    liq_sweep_qualified      INTEGER,
    bos_1h                   INTEGER,
    choch_1h                 INTEGER,
    msb_15m_confirmed        INTEGER,
    mtf_confluence           INTEGER,
    atr_ratio                REAL,
    bb_width                 REAL,
    close_vs_ema20_4h        REAL,
    close_vs_ema50_4h        REAL,
    high_low_range_15m       REAL,
    macd_histogram           REAL,
    macd_signal_cross        INTEGER,
    rsi_14                   REAL,
    rsi_zone                 INTEGER,
    stoch_k                  REAL,
    stoch_d                  REAL,
    momentum_3bar            REAL,
    ob_4h_distance_pips      REAL,
    fvg_4h_fill_ratio        REAL,
    liq_sweep_strength       REAL,
    prior_candle_body_ratio  REAL,
    consecutive_same_dir     INTEGER,
    pivot_proximity          REAL,
    sweep_pending_bars       INTEGER,
    spread_pips              REAL DEFAULT 1.5,
    session_flag             INTEGER,
    hour_of_day              INTEGER,
    day_of_week              INTEGER,
    open_positions_count     INTEGER DEFAULT 0,
    max_dd_24h               REAL DEFAULT 0,
    calendar_risk_score      INTEGER DEFAULT 0,
    sentiment_score          REAL DEFAULT 0,
    label                    INTEGER,            -- 0=up, 1=flat, 2=down
    future_close_price       REAL,
    future_return_pips       REAL,
    labeled_at               TIMESTAMP,
    created_at               TIMESTAMP DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_pair_open ON trades(pair, open_time);
CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time);
CREATE INDEX IF NOT EXISTS idx_signals_pair_time ON signals(pair, signal_time);
CREATE INDEX IF NOT EXISTS idx_api_call_time ON api_call_log(call_time);
CREATE INDEX IF NOT EXISTS idx_training_pair_time ON training_samples(pair, signal_time);
CREATE INDEX IF NOT EXISTS idx_training_unlabeled ON training_samples(label, signal_time);
"""


# ── CRUD ヘルパー ────────────────────────────────────

def insert_trade(conn: sqlite3.Connection, trade: dict) -> int:
    """トレードレコードを挿入し、IDを返す。"""
    cur = conn.execute(
        """INSERT INTO trades
           (pair, direction, open_time, open_price, volume, sl_price, tp_price, mt5_ticket)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade["pair"],
            trade["direction"],
            trade["open_time"].isoformat(),
            trade["open_price"],
            trade["volume"],
            trade.get("sl_price"),
            trade.get("tp_price"),
            trade.get("mt5_ticket"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def close_trade(
    conn: sqlite3.Connection,
    trade_id: int,
    close_price: float,
    pnl_pips: float,
    pnl_jpy: float,
    exit_reason: str,
) -> None:
    """トレードを決済済みとして更新する。"""
    conn.execute(
        """UPDATE trades
           SET close_time=?, close_price=?, pnl_pips=?, pnl_jpy=?, exit_reason=?
           WHERE id=?""",
        (now_utc().isoformat(), close_price, pnl_pips, pnl_jpy, exit_reason, trade_id),
    )
    conn.commit()


def insert_signal(conn: sqlite3.Connection, signal: dict) -> int:
    """シグナルレコードを挿入し、IDを返す。"""
    cur = conn.execute(
        """INSERT INTO signals
           (pair, signal_time, direction, lgbm_prob_up, lgbm_prob_flat,
            lgbm_prob_down, gpt_sentiment, mtf_confluence, executed, veto_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            signal["pair"],
            signal["signal_time"].isoformat(),
            signal.get("direction"),
            signal.get("lgbm_prob_up"),
            signal.get("lgbm_prob_flat"),
            signal.get("lgbm_prob_down"),
            signal.get("gpt_sentiment"),
            signal.get("mtf_confluence"),
            signal.get("executed", False),
            signal.get("veto_reason"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_api_call(conn: sqlite3.Connection, reason: str, model: str,
                    tokens_in: int, tokens_out: int, cost_usd: float) -> None:
    """GPT API呼び出しログを記録する。"""
    conn.execute(
        """INSERT INTO api_call_log (call_time, reason, model, tokens_in, tokens_out, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (now_utc().isoformat(), reason, model, tokens_in, tokens_out, cost_usd),
    )
    conn.commit()


def insert_optimization(conn: sqlite3.Connection, opt: dict) -> int:
    """最適化結果を記録する。"""
    cur = conn.execute(
        """INSERT INTO optimization_history
           (optimized_at, optimize_window_days, validate_window_days,
            step1_ratio, step2_ratio, sl_multiplier, tp_multiplier,
            sharpe_optimize, sharpe_validate, sample_count, was_applied, rollback_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now_utc().isoformat(),
            opt["optimize_window_days"],
            opt["validate_window_days"],
            opt.get("step1_ratio"),
            opt.get("step2_ratio"),
            opt.get("sl_multiplier"),
            opt.get("tp_multiplier"),
            opt.get("sharpe_optimize"),
            opt.get("sharpe_validate"),
            opt.get("sample_count"),
            opt.get("was_applied", True),
            opt.get("rollback_reason"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_recent_trades(conn: sqlite3.Connection, pair: str | None = None,
                      days: int = 28) -> list[dict]:
    """直近 N 日間のクローズ済みトレードを取得する。"""
    from datetime import timedelta
    cutoff = (now_utc() - timedelta(days=days)).isoformat()
    if pair:
        rows = conn.execute(
            "SELECT * FROM trades WHERE pair=? AND close_time IS NOT NULL AND close_time>=? ORDER BY close_time",
            (pair, cutoff),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades WHERE close_time IS NOT NULL AND close_time>=? ORDER BY close_time",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_daily_pnl(conn: sqlite3.Connection, day_start_utc=None) -> float:
    """当日（ブローカー日付の00:00起点）のP&L合計（円）を返す。"""
    if day_start_utc is None:
        day_start_utc = broker_day_start_utc()
    today_start = day_start_utc.isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl_jpy), 0) as total FROM trades WHERE close_time>=?",
        (today_start,),
    ).fetchone()
    return row["total"]


def check_integrity(conn: sqlite3.Connection) -> bool:
    """DB整合性チェック。"""
    result = conn.execute("PRAGMA integrity_check").fetchone()
    return result[0] == "ok"


def insert_training_sample(conn: sqlite3.Connection, sample: dict) -> int:
    """学習用サンプル（特徴量36個）を保存する。"""
    cur = conn.execute(
        """INSERT INTO training_samples (
           pair, signal_time, direction, close_price, atr,
           fvg_4h_zone_active, ob_4h_zone_active, liq_sweep_1h,
           liq_sweep_qualified, bos_1h, choch_1h, msb_15m_confirmed,
           mtf_confluence, atr_ratio, bb_width,
           close_vs_ema20_4h, close_vs_ema50_4h, high_low_range_15m,
           macd_histogram, macd_signal_cross, rsi_14, rsi_zone,
           stoch_k, stoch_d, momentum_3bar,
           ob_4h_distance_pips, fvg_4h_fill_ratio, liq_sweep_strength,
           prior_candle_body_ratio, consecutive_same_dir, pivot_proximity,
           sweep_pending_bars, spread_pips, session_flag, hour_of_day, day_of_week,
           open_positions_count, max_dd_24h, calendar_risk_score, sentiment_score
        ) VALUES (
           ?, ?, ?, ?, ?,
           ?, ?, ?,
           ?, ?, ?, ?,
           ?, ?, ?,
           ?, ?, ?,
           ?, ?, ?, ?,
           ?, ?, ?,
           ?, ?, ?,
           ?, ?, ?,
           ?, ?, ?, ?, ?,
           ?, ?, ?, ?
        )""",
        (
            sample["pair"],
            sample["signal_time"].isoformat(),
            sample.get("direction"),
            sample.get("close_price"),
            sample.get("atr"),
            int(sample.get("fvg_4h_zone_active", False)),
            int(sample.get("ob_4h_zone_active", False)),
            int(sample.get("liq_sweep_1h", False)),
            int(sample.get("liq_sweep_qualified", False)),
            int(sample.get("bos_1h", False)),
            int(sample.get("choch_1h", False)),
            int(sample.get("msb_15m_confirmed", False)),
            sample.get("mtf_confluence", 0),
            sample.get("atr_ratio", 1.0),
            sample.get("bb_width", 0.0),
            sample.get("close_vs_ema20_4h", 0.0),
            sample.get("close_vs_ema50_4h", 0.0),
            sample.get("high_low_range_15m", 0.0),
            sample.get("macd_histogram", 0.0),
            sample.get("macd_signal_cross", 0),
            sample.get("rsi_14", 50.0),
            sample.get("rsi_zone", 0),
            sample.get("stoch_k", 50.0),
            sample.get("stoch_d", 50.0),
            sample.get("momentum_3bar", 0.0),
            sample.get("ob_4h_distance_pips", 0.0),
            sample.get("fvg_4h_fill_ratio", 0.0),
            sample.get("liq_sweep_strength", 0.0),
            sample.get("prior_candle_body_ratio", 0.5),
            sample.get("consecutive_same_dir", 0),
            sample.get("pivot_proximity", 0.0),
            sample.get("sweep_pending_bars", 0),
            sample.get("spread_pips", 1.5),
            sample.get("session_flag", 0),
            sample.get("hour_of_day", 0),
            sample.get("day_of_week", 0),
            sample.get("open_positions_count", 0),
            sample.get("max_dd_24h", 0.0),
            sample.get("calendar_risk_score", 0),
            sample.get("sentiment_score", 0.0),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_unlabeled_training_samples(
    conn: sqlite3.Connection,
    limit: int = 5000,
) -> list[dict]:
    """未ラベル学習サンプルを取得する。"""
    rows = conn.execute(
        "SELECT * FROM training_samples WHERE label IS NULL ORDER BY signal_time ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_training_label(
    conn: sqlite3.Connection,
    sample_id: int,
    label: int,
    future_close: float,
    future_return_pips: float,
) -> None:
    """学習サンプルにラベルを付与する。"""
    conn.execute(
        """UPDATE training_samples
           SET label=?, future_close_price=?, future_return_pips=?, labeled_at=?
           WHERE id=?""",
        (label, future_close, future_return_pips, now_utc().isoformat(), sample_id),
    )
    conn.commit()


def get_labeled_training_samples(
    conn: sqlite3.Connection,
    pair: str,
    days: int = 90,
) -> list[dict]:
    """指定ペアのラベル済み学習サンプルを取得する。"""
    from datetime import timedelta

    cutoff = (now_utc() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT * FROM training_samples
           WHERE pair=? AND label IS NOT NULL AND signal_time>=?
           ORDER BY signal_time ASC""",
        (pair, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]
