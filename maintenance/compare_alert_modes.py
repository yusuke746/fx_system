"""
アラートモード比較レポート（ml_first vs strict）

使い方:
    python maintenance/compare_alert_modes.py
    python maintenance/compare_alert_modes.py --days 14
    python maintenance/compare_alert_modes.py --pair USDJPY
    python maintenance/compare_alert_modes.py --db-path db/trading.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "trading.db"


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _print_table(title: str, rows: list[sqlite3.Row]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return

    headers = list(rows[0].keys())
    widths = [len(h) for h in headers]
    for r in rows:
        for i, h in enumerate(headers):
            widths[i] = max(widths[i], len(str(r[h])))

    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[i]) for i, h in enumerate(headers)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Pine alert modes from SQLite records")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--pair", default=None, choices=["USDJPY", "EURUSD", "GBPJPY"], help="Filter by pair")
    return parser


def query_signal_summary(conn: sqlite3.Connection, days: int, pair: str | None) -> list[sqlite3.Row]:
    pair_filter = "AND pair = ?" if pair else ""
    params: list[object] = [f"-{days} day"]
    if pair:
        params.append(pair)

    sql = f"""
    SELECT
      COALESCE(alert_mode, 'unknown') AS alert_mode,
      COUNT(*) AS signals,
      SUM(CASE WHEN quality_gate_pass = 1 THEN 1 ELSE 0 END) AS quality_pass,
      ROUND(100.0 * SUM(CASE WHEN quality_gate_pass = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS quality_pass_pct,
      SUM(CASE WHEN vol_ok = 1 THEN 1 ELSE 0 END) AS vol_ok_count,
      SUM(CASE WHEN in_session = 1 THEN 1 ELSE 0 END) AS in_session_count,
      SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) AS executed_count
    FROM signals
    WHERE signal_time >= datetime('now', ?)
      {pair_filter}
    GROUP BY COALESCE(alert_mode, 'unknown')
    ORDER BY signals DESC;
    """
    return conn.execute(sql, tuple(params)).fetchall()


def query_training_summary(conn: sqlite3.Connection, days: int, pair: str | None) -> list[sqlite3.Row]:
    pair_filter = "AND pair = ?" if pair else ""
    params: list[object] = [f"-{days} day"]
    if pair:
        params.append(pair)

    sql = f"""
    SELECT
      COALESCE(alert_mode, 'unknown') AS alert_mode,
      COUNT(*) AS samples,
      SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) AS labeled,
      SUM(CASE WHEN direction = 'long' THEN 1 ELSE 0 END) AS long_samples,
      SUM(CASE WHEN direction = 'short' THEN 1 ELSE 0 END) AS short_samples,
      ROUND(AVG(CASE WHEN future_return_pips IS NOT NULL THEN future_return_pips END), 3) AS avg_future_pips,
      ROUND(
        100.0 * SUM(
          CASE
            WHEN label IS NULL THEN 0
            WHEN direction = 'long' AND label = 0 THEN 1
            WHEN direction = 'short' AND label = 2 THEN 1
            ELSE 0
          END
        ) / NULLIF(SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END), 0),
        1
      ) AS directional_hit_rate_pct
    FROM training_samples
    WHERE signal_time >= datetime('now', ?)
      {pair_filter}
    GROUP BY COALESCE(alert_mode, 'unknown')
    ORDER BY samples DESC;
    """
    return conn.execute(sql, tuple(params)).fetchall()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = _connect(str(db_path))
    try:
        signal_rows = query_signal_summary(conn, args.days, args.pair)
        training_rows = query_training_summary(conn, args.days, args.pair)
    finally:
        conn.close()

    target = args.pair if args.pair else "ALL"
    print(f"DB: {db_path}")
    print(f"Window: last {args.days} days / Pair: {target}")

    _print_table("Signals by alert_mode", signal_rows)
    _print_table("Training samples by alert_mode", training_rows)


if __name__ == "__main__":
    main()
