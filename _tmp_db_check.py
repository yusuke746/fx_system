import sqlite3

conn = sqlite3.connect('db/trading.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print('=== tables ===')
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print([r[0] for r in cur.fetchall()])

print('\n=== trades schema ===')
cur.execute('PRAGMA table_info(trades)')
for r in cur.fetchall():
    print(dict(r))

day = '2026-03-25'

print('\n=== opened today count ===')
cur.execute("SELECT COUNT(*) FROM trades WHERE open_time LIKE ?", (f'{day}%',))
print(cur.fetchone()[0])

print('\n=== closed today count ===')
cur.execute("SELECT COUNT(*) FROM trades WHERE close_time LIKE ?", (f'{day}%',))
print(cur.fetchone()[0])

print('\n=== open trades now (close_time is null) ===')
cur.execute("SELECT COUNT(*) FROM trades WHERE close_time IS NULL")
print(cur.fetchone()[0])

print('\n=== close today pnl summary ===')
cur.execute('''
SELECT
  COUNT(*) as n,
  ROUND(COALESCE(SUM(pnl_jpy),0),2) as pnl_jpy_sum,
  ROUND(COALESCE(SUM(pnl_pips),0),2) as pnl_pips_sum,
  ROUND(COALESCE(AVG(pnl_jpy),0),2) as pnl_jpy_avg
FROM trades
WHERE close_time LIKE ?
''', (f'{day}%',))
print(dict(cur.fetchone()))

print('\n=== close today by exit_reason ===')
cur.execute('''
SELECT COALESCE(exit_reason,'(null)') as reason,
       COUNT(*) as n,
       ROUND(COALESCE(SUM(pnl_jpy),0),2) as pnl_jpy_sum,
       ROUND(COALESCE(AVG(pnl_jpy),0),2) as pnl_jpy_avg
FROM trades
WHERE close_time LIKE ?
GROUP BY COALESCE(exit_reason,'(null)')
ORDER BY pnl_jpy_sum ASC
''', (f'{day}%',))
for r in cur.fetchall():
    print(dict(r))

print('\n=== latest 20 trades ===')
cur.execute('''
SELECT id,pair,direction,open_time,close_time,pnl_jpy,pnl_pips,exit_reason,mt5_ticket
FROM trades
ORDER BY id DESC
LIMIT 20
''')
for r in cur.fetchall():
    print(dict(r))

print('\n=== unresolved candidates (today open but no close) ===')
cur.execute('''
SELECT id,pair,direction,open_time,mt5_ticket
FROM trades
WHERE open_time LIKE ? AND close_time IS NULL
ORDER BY id DESC
''', (f'{day}%',))
rows = cur.fetchall()
print('count=',len(rows))
for r in rows[:20]:
    print(dict(r))

conn.close()
