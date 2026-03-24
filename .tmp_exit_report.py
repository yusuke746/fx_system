import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

conn = sqlite3.connect('db/trading.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

window_days = 30
params = (f'-{window_days} day',)

summary = {}
summary['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
summary['window_days'] = window_days
summary['closed_total'] = cur.execute(
    "SELECT COUNT(*) FROM trades WHERE close_time IS NOT NULL AND close_time >= datetime('now', ?)",
    params,
).fetchone()[0]

rows = cur.execute(
    """
    SELECT COALESCE(exit_reason, 'unknown') AS exit_reason,
           COUNT(*) AS n,
           ROUND(AVG(pnl_jpy), 2) AS avg_pnl_jpy,
           ROUND(SUM(pnl_jpy), 2) AS sum_pnl_jpy,
           ROUND(AVG((julianday(close_time)-julianday(open_time))*24*60), 2) AS avg_hold_min
    FROM trades
    WHERE close_time IS NOT NULL AND close_time >= datetime('now', ?)
    GROUP BY COALESCE(exit_reason, 'unknown')
    ORDER BY n DESC
    """,
    params,
).fetchall()
summary['by_exit_reason'] = [dict(r) for r in rows]

rows = cur.execute(
    """
    SELECT pair,
           COALESCE(exit_reason, 'unknown') AS exit_reason,
           COUNT(*) AS n,
           ROUND(AVG(pnl_jpy), 2) AS avg_pnl_jpy,
           ROUND(SUM(pnl_jpy), 2) AS sum_pnl_jpy
    FROM trades
    WHERE close_time IS NOT NULL AND close_time >= datetime('now', ?)
    GROUP BY pair, COALESCE(exit_reason, 'unknown')
    ORDER BY pair, n DESC
    """,
    params,
).fetchall()
summary['by_pair_exit_reason'] = [dict(r) for r in rows]

rows = cur.execute(
    """
    SELECT pair,
           COUNT(*) AS n,
           ROUND(AVG((julianday(close_time)-julianday(open_time))*24*60), 2) AS avg_hold_min,
           ROUND(AVG(pnl_jpy), 2) AS avg_pnl_jpy,
           ROUND(SUM(pnl_jpy), 2) AS sum_pnl_jpy
    FROM trades
    WHERE close_time IS NOT NULL AND close_time >= datetime('now', ?)
    GROUP BY pair
    ORDER BY pair
    """,
    params,
).fetchall()
summary['by_pair'] = [dict(r) for r in rows]

conn.close()

out = Path('data') / f"exit_reason_summary_{datetime.now().strftime('%Y-%m-%d')}.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(out.as_posix())
print(json.dumps(summary, ensure_ascii=False, indent=2))
