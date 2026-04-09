#!/usr/bin/env python
"""本番環境の DB データ状況確認"""

import sqlite3
from pathlib import Path

# 本番 DB パス
db_paths = [
    Path("./db/trading.db"),
    Path("C:\\Users\\MT4ver2-u70-t1jaowH5\\fx_system\\db\\trading.db"),
    Path("./data/trades.db"),
]

db_path = None
for p in db_paths:
    if p.exists():
        db_path = p
        break

if not db_path:
    print("❌ DB見つかりません")
    print(f"  確認範囲: {db_paths}")
else:
    print("=" * 70)
    print(f"本番DB分析: {db_path}")
    print("=" * 70)
    
    with sqlite3.connect(str(db_path)) as conn:
        # training_samples の統計
        cur = conn.execute(
            "SELECT "
            "  COUNT(*) as total, "
            "  SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) as labeled, "
            "  COUNT(DISTINCT pair) as pairs "
            "FROM training_samples"
        )
        total, labeled, pairs = cur.fetchone()
        
        print(f"\ntraining_samples:")
        print(f"  全サンプル: {total}件")
        print(f"  ラベル付き: {labeled}件")
        print(f"  通貨ペア数: {pairs}件")
        
        if labeled and labeled > 100:
            print(f"\n  ✅ 十分なデータ量 ({labeled}件 > 100件)")
            print(f"     → 即座に再学習推奨")
        elif labeled and labeled > 50:
            print(f"\n  🟡 最小限のデータ ({labeled}件)")
            print(f"     → 再学習可能だが、データをもう少し蓄積してからの方が良い")
        else:
            print(f"\n  🔴 不足 ({labeled}件 < 50件)")
            print(f"     → データ蓄積待ち")
        
        # ペア別ラベル数
        print(f"\nペア別ラベル付きサンプル:")
        cur = conn.execute(
            "SELECT pair, COUNT(*) as cnt FROM training_samples "
            "WHERE label IS NOT NULL GROUP BY pair"
        )
        for pair, cnt in cur.fetchall():
            print(f"  {pair}: {cnt}件")
        
        # trades テーブルの状況
        cur = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE close_time IS NOT NULL"
        )
        closed_trades = cur.fetchone()[0]
        print(f"\n決済済みトレード: {closed_trades}件")
    
    print("\n" + "=" * 70)
    print("推奨アクション:")
    print("=" * 70)
    if labeled and labeled > 100:
        print("""
即座に再学習実行:
  python maintenance/run_bootstrap_batch.py

または:
  python -c "from ml.retraining import run_weekly_retraining; 
             import sqlite3; 
             conn = sqlite3.connect('./db/trading.db');
             run_weekly_retraining(conn)"
        """)
    elif labeled and labeled > 50:
        print("""
データはあるが、さらに蓄積待ち:
  - 本番稼働を続行（シグナルは Pine スクリプト方向に従う）
  - 週日曜 23:00 に自動再学習
  - または手動で python maintenance/run_bootstrap_batch.py
        """)
    else:
        print("""
データ蓄積待ち:
  - 本番稼働を続行
  - 定期的にこのスクリプトで確認
  - 週日曜 23:00 の定期メンテナンストリガーで自動再学習
        """)
