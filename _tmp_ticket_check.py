import sqlite3

conn = sqlite3.connect('db/trading.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

tickets = [930904921, 931015643, 931015713, 931283628]
for t in tickets:
    cur.execute('''
    SELECT id,pair,direction,open_time,close_time,pnl_jpy,pnl_pips,exit_reason,mt5_ticket
    FROM trades WHERE mt5_ticket=? ORDER BY id DESC LIMIT 1
    ''', (t,))
    row = cur.fetchone()
    print(dict(row) if row else {'mt5_ticket': t, 'status': 'not_found'})

cur.execute("SELECT COUNT(*) FROM trades WHERE close_time IS NULL")
print('open_rows_total=', cur.fetchone()[0])

conn.close()
