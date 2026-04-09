#!/usr/bin/env python
"""深い再学習診断スクリプト"""

import sqlite3
from pathlib import Path
from ml.lgbm_model import FEATURE_NAMES

db_path = Path("data/trades.db")

print("=" * 70)
print("再学習診断: DBとスキーマ詳細確認")
print("=" * 70)

if not db_path.exists():
    print(f"\n✗ DB見つかりません: {db_path}")
    print("  → 再学習するデータが無い状態です")
else:
    print(f"\n✓ DB確認: {db_path}")
    print(f"  サイズ: {db_path.stat().st_size / 1024:.1f}KB")
    
    with sqlite3.connect(str(db_path)) as conn:
        # テーブル一覧
        print("\n" + "=" * 70)
        print("テーブル一覧:")
        print("=" * 70)
        
        cur = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        )
        tables = cur.fetchall()
        
        for tbl_name, tbl_sql in tables:
            cur = conn.execute(f"SELECT COUNT(*) FROM {tbl_name}")
            count = cur.fetchone()[0]
            print(f"\n{tbl_name}: {count}件")
            
            if tbl_name == "training_samples":
                # スキーマ確認
                cur = conn.execute(f"PRAGMA table_info({tbl_name})")
                cols = cur.fetchall()
                print(f"  カラム数: {len(cols)}")
                
                llm_features = ['risk_appetite_score', 'usd_macro_score', 
                               'jpy_macro_score', 'oil_shock_score', 
                               'geopolitical_risk_score']
                
                col_names = [c[1] for c in cols]
                for feat in llm_features:
                    found = "✓" if feat in col_names else "✗"
                    print(f"    {found} {feat}")
                
                # ラベル状況
                if count > 0:
                    cur = conn.execute(
                        "SELECT label, COUNT(*) as cnt "
                        "FROM training_samples GROUP BY label"
                    )
                    for label, cnt in cur.fetchall():
                        print(f"    - label={label}: {cnt}件")
                    
                    # 特徴量サンプル
                    print(f"\n  サンプル1件（最初のラベル付き）:")
                    cur = conn.execute(
                        f"SELECT {', '.join(FEATURE_NAMES[:5])} FROM training_samples "
                        "WHERE label IS NOT NULL LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        for i, val in enumerate(row):
                            print(f"    {FEATURE_NAMES[i]}: {val}")
            
            elif tbl_name == "trades":
                if count > 0:
                    cur = conn.execute(
                        "SELECT COUNT(DISTINCT pair) as pairs, "
                        "MIN(open_time) as first, MAX(close_time) as last "
                        "FROM trades WHERE close_time IS NOT NULL"
                    )
                    row = cur.fetchone()
                    print(f"  通貨ペア数: {row[0]}")
                    print(f"  期間: {row[1]} ～ {row[2]}")

print("\n" + "=" * 70)
print("診断終了")
print("=" * 70)
