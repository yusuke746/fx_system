#!/usr/bin/env python
"""再学習実行スクリプト - 43特徴量版"""

import sqlite3
from pathlib import Path
from loguru import logger
from core.database import init_db
from ml.retraining import run_weekly_retraining
from config.settings import get_settings

# ログ設定
logger.add(
    "logs/retrain_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO"
)

print("=" * 70)
print("FX自動売買システム - LightGBM 再学習スクリプト（43特徴量版）")
print("=" * 70)

settings = get_settings()
db_path = Path(settings.db_path)
model_dir = Path(settings.model_dir)

# DB初期化
print("\n[1/3] DB初期化...")
init_db(str(db_path))
print("  ✓ DB初期化完了")

# データ確認
print("\n[2/3] training_samplesデータ確認...")
with sqlite3.connect(str(db_path)) as conn:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT COUNT(*) as total, "
        "SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) as labeled "
        "FROM training_samples"
    )
    row = cur.fetchone()
    total = row[0] if row[0] is not None else 0
    labeled = row[1] if row[1] is not None else 0
    
    print(f"  - 全サンプル: {total}件")
    print(f"  - ラベル付き: {labeled}件")
    
    if labeled < 50:
        print("  ⚠ サンプル不足（最小50件推奨）")
    else:
        print("  ✓ サンプル数OK → 再学習実行")
        print("\n[3/3] 再学習実行中（3ペア、1-3分/ペア）...")
        
        result = run_weekly_retraining(conn)
        
        trained = len(result['retrain'].get('trained_pairs', []))
        print(f"\n  ✓ 完了: {trained}/3ペア")
        
        # 結果詳細
        for pair in ["USDJPY", "EURUSD", "GBPJPY"]:
            if pair in result['retrain'].get('trained_pairs', []):
                val = result['retrain'].get('validation', {}).get(pair, {})
                acc = val.get('balanced_accuracy', val.get('accuracy', 'N/A'))
                print(f"    {pair}: ✓ (balanced_acc={acc})")
            else:
                skip_reason = result['retrain'].get('skipped_pairs', {}).get(pair)
                print(f"    {pair}: ⚠ ({skip_reason})")

# モデルファイル確認
print("\n✓ モデルファイル確認...")
models_dir = Path("models")
models_dir = model_dir
models_dir.mkdir(parents=True, exist_ok=True)

for pair in ["USDJPY", "EURUSD", "GBPJPY"]:
    pkl = models_dir / f"lgbm_{pair}.pkl"
    if pkl.exists():
        size_mb = pkl.stat().st_size / (1024 * 1024)
        print(f"  ✓ lgbm_{pair}.pkl ({size_mb:.2f}MB)")
    else:
        print(f"  - lgbm_{pair}.pkl (未生成)")

print("\n" + "=" * 70)
print("再学習完了")
print("=" * 70)
