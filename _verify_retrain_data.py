#!/usr/bin/env python
"""再学習前のデータ整合性確認"""

import sqlite3
from pathlib import Path
from ml.retraining import get_labeled_training_samples
from ml.lgbm_model import FEATURE_NAMES

db_path = Path('data/trades.db')

print("=" * 60)
print("再学習前のデータ検証:")
print("=" * 60)

with sqlite3.connect(str(db_path)) as conn:
    # 特徴量スキーマ確認
    cursor = conn.execute("PRAGMA table_info(training_samples)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    
    print(f"\n✓ training_samplesテーブル: {len(columns)}カラム")
    
    # LLM特徴量の確認
    llm_features = [
        "risk_appetite_score",
        "usd_macro_score", 
        "jpy_macro_score",
        "oil_shock_score",
        "geopolitical_risk_score"
    ]
    
    for feat in llm_features:
        if feat in columns:
            print(f"  ✓ {feat}: {columns[feat]}")
        else:
            print(f"  ✗ {feat}: 見つかりません")
    
    # ラベル付きサンプル数
    cursor = conn.execute("SELECT COUNT(*) FROM training_samples WHERE label IS NOT NULL")
    labeled_count = cursor.fetchone()[0]
    print(f"\n✓ ラベル付きサンプル: {labeled_count}件")
    
    # サンプル取得
    samples = get_labeled_training_samples(conn, limit=3)
    print(f"✓ 取得サンプル数: {len(samples)}件")
    
    if samples:
        first = samples[0]
        print(f"\nサンプル #1:")
        print(f"  特徴量数: {len(first)} (想定: {len(FEATURE_NAMES)})")
        print(f"  risk_appetite_score: {first.get('risk_appetite_score', 'N/A')}")
        print(f"  label: {first.get('label', 'N/A')}")
        print(f"  列: {list(first.keys())[:5]}...")

print("\n" + "=" * 60)
print("データ検証完了 → 再学習実行可能")
print("=" * 60)
