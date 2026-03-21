"""
training_samples の蓄積状況を確認する運用補助スクリプト。

用途:
  - Webhook 受信後に training_samples が増えているか確認する
  - ペア別の未ラベル/ラベル済み件数を確認する
  - 直近に保存されたサンプルの内容をざっと見る

使い方:
  python maintenance/check_training_samples.py
  python maintenance/check_training_samples.py --hours 24 --limit 5
  python maintenance/check_training_samples.py --pair USDJPY --json
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

from config.settings import get_settings
from core.database import get_connection
from core.time_manager import now_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check training_samples accumulation status")
    parser.add_argument("--pair", choices=["USDJPY", "EURUSD", "GBPJPY"], help="Filter by pair")
    parser.add_argument("--hours", type=int, default=24, help="Recent hours window for activity summary")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent rows to show")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def _fetch_scalar(conn, query: str, params: tuple = ()) -> int:
    row = conn.execute(query, params).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def build_report(conn, pair: str | None, hours: int, limit: int) -> dict:
    now = now_utc()
    recent_cutoff = (now - timedelta(hours=hours)).isoformat()

    where_pair = "WHERE pair=?" if pair else ""
    params_pair = (pair,) if pair else ()

    total = _fetch_scalar(
        conn,
        f"SELECT COUNT(*) FROM training_samples {where_pair}",
        params_pair,
    )
    unlabeled = _fetch_scalar(
        conn,
        f"SELECT COUNT(*) FROM training_samples {'WHERE pair=? AND' if pair else 'WHERE'} label IS NULL",
        params_pair,
    )
    labeled = _fetch_scalar(
        conn,
        f"SELECT COUNT(*) FROM training_samples {'WHERE pair=? AND' if pair else 'WHERE'} label IS NOT NULL",
        params_pair,
    )
    recent = _fetch_scalar(
        conn,
        f"SELECT COUNT(*) FROM training_samples {'WHERE pair=? AND' if pair else 'WHERE'} signal_time >= ?",
        params_pair + (recent_cutoff,),
    )

    per_pair_rows = conn.execute(
        """
        SELECT
            pair,
            COUNT(*) AS total,
            SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END) AS unlabeled,
            SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) AS labeled,
            MAX(signal_time) AS latest_signal_time,
            MAX(labeled_at) AS latest_labeled_at
        FROM training_samples
        GROUP BY pair
        ORDER BY pair
        """
    ).fetchall()

    recent_rows = conn.execute(
        f"""
        SELECT pair, signal_time, direction, label, close_price, future_return_pips
        FROM training_samples
        {'WHERE pair=?' if pair else ''}
        ORDER BY signal_time DESC
        LIMIT ?
        """,
        params_pair + (limit,),
    ).fetchall()

    return {
        "db_path": get_settings().db_path,
        "generated_at_utc": now.isoformat(),
        "pair_filter": pair,
        "summary": {
            "total": total,
            "unlabeled": unlabeled,
            "labeled": labeled,
            "recent_last_hours": recent,
            "recent_window_hours": hours,
        },
        "per_pair": [dict(row) for row in per_pair_rows],
        "recent_rows": [dict(row) for row in recent_rows],
    }


def print_human(report: dict) -> None:
    summary = report["summary"]
    print("=== training_samples status ===")
    print(f"db_path={report['db_path']}")
    print(f"generated_at_utc={report['generated_at_utc']}")
    print(f"pair_filter={report['pair_filter'] or 'ALL'}")
    print(f"total={summary['total']}")
    print(f"unlabeled={summary['unlabeled']}")
    print(f"labeled={summary['labeled']}")
    print(f"recent_last_{summary['recent_window_hours']}h={summary['recent_last_hours']}")
    print()
    print("-- per_pair --")
    for row in report["per_pair"]:
        print(
            f"{row['pair']}: total={row['total']} unlabeled={row['unlabeled']} "
            f"labeled={row['labeled']} latest_signal={row['latest_signal_time']} "
            f"latest_labeled={row['latest_labeled_at']}"
        )

    print()
    print("-- recent_rows --")
    for row in report["recent_rows"]:
        print(
            f"{row['signal_time']} {row['pair']} {row['direction']} "
            f"label={row['label']} close={row['close_price']} "
            f"future_return_pips={row['future_return_pips']}"
        )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    conn = get_connection(settings.db_path)
    try:
        report = build_report(conn, pair=args.pair, hours=args.hours, limit=args.limit)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print_human(report)


if __name__ == "__main__":
    main()