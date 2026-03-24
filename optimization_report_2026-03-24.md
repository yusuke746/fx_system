# FX System Optimization Report (2026-03-24)

## 1) 実施内容

### A. CSVベース再学習（3ペア）
実行コマンド:

```bash
python -m maintenance.run_bootstrap_batch --input-dir data --output-dir data --model-dir models
```

処理内容:
- TradingView CSV (`data/*_chart.csv`) から bootstrap 学習データ生成
- Walk-forward validation (date-based)
- LightGBM 再学習
- metrics JSON 更新

結果:
- USDJPY: trained, rows=5518, dropped_nan_rows=0
- EURUSD: trained, rows=5218, dropped_nan_rows=0
- GBPJPY: trained, rows=5759, dropped_nan_rows=0

### B. 最適化ルーチン実行
実行内容:
- `optimizer.weekend_optimizer.run_weekend_optimization`
- `optimizer.weekend_optimizer.auto_tune_execution_noise`

結果:
- weekend optimization: `applied=false`, `reason=dynamic_exit_strategy_active`, `sample_count=0`
- execution noise tuning: `applied=false`, `reason=insufficient_samples(0<20)`, `sample_count=0`

## 2) モデル評価（最新）

| Pair | Accuracy | Balanced Accuracy | Majority Baseline | Samples | Updated(UTC) |
|---|---:|---:|---:|---:|---|
| USDJPY | 0.4405 | 0.4149 | 0.4433 | 5518 | 2026-03-24T11:35:53.354157+00:00 |
| EURUSD | 0.5011 | 0.4969 | 0.4658 | 5218 | 2026-03-24T11:35:59.970553+00:00 |
| GBPJPY | 0.5094 | 0.4740 | 0.4843 | 5759 | 2026-03-24T11:36:03.884404+00:00 |

参照ファイル:
- `models/lgbm_USDJPY_metrics.json`
- `models/lgbm_EURUSD_metrics.json`
- `models/lgbm_GBPJPY_metrics.json`

## 3) 生成/更新された成果物

- 学習データ
  - `data/USDJPY_bootstrap_train.csv`
  - `data/EURUSD_bootstrap_train.csv`
  - `data/GBPJPY_bootstrap_train.csv`
- 学習済みモデル（最新世代）
  - `models/lgbm_USDJPY_20260324_113553.pkl`
  - `models/lgbm_EURUSD_20260324_113559.pkl`
  - `models/lgbm_GBPJPY_20260324_113603.pkl`
- 現行シンボリックモデル
  - `models/lgbm_USDJPY.pkl`
  - `models/lgbm_EURUSD.pkl`
  - `models/lgbm_GBPJPY.pkl`

## 4) 補足

- 現在の `trading.db` は `training_samples`/closed trade が空のため、週次自動最適化（執行ノイズ調整含む）は安全にスキップされる。
- モデル面の最適化は、今回「CSV全量再学習 + WFV再評価」まで完了。

## 5) 次アクション（推奨）

1. 1〜2週間の実運用トレード履歴を蓄積
2. `auto_tune_execution_noise` を再実行して実運用データで最適化
3. 必要ならペア別閾値（`ml.prediction_thresholds`）の再チューニングを実施
