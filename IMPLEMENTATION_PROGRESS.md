# FX自動売買システム v2.3 実装進捗レポート

**更新日:** 2026-03-21  
**対象:** fx_system 物販版  
**対象バージョン:** v2.3

---

## 🎯 実装完了項目

### 1. LightGBM 35特徴量確定 ✅

**ファイル:** [ml/lgbm_model.py](ml/lgbm_model.py#L19-L50)

35特徴量の完全リスト：

- **SMCフラグ (8):** fvg_4h_zone_active, ob_4h_zone_active, liq_sweep_1h, liq_sweep_qualified, bos_1h, choch_1h, msb_15m_confirmed, mtf_confluence
- **価格・ボラティリティ系 (5):** atr_ratio, bb_width, close_vs_ema20_4h, close_vs_ema50_4h, high_low_range_15m
- **トレンド・モメンタム補助 (3):** trend_direction, momentum_long, momentum_short
- **モメンタム系 (7):** macd_histogram, macd_signal_cross, rsi_14, rsi_zone, stoch_k, stoch_d, momentum_3bar
- **構造・パターン系 (6):** ob_4h_distance_pips, fvg_4h_fill_ratio, liq_sweep_strength, prior_candle_body_ratio, consecutive_same_dir, sweep_pending_bars
- **リスク・ポジション系 (4):** open_positions_count, max_dd_24h, calendar_risk_score, sentiment_score
- **セッション補助 (2):** session_type, day_of_week

**確認内容:**
- `assert len(FEATURE_NAMES) == 35` ✅ パス
- `prob_decay` 削除完了 ✅
- 欠落特徴量なし ✅

---

### 2. is_strong_signal ロジック修正（OR → AND）✅

**ファイル:** [ml/lgbm_model.py](ml/lgbm_model.py#L116-L147)

**変更内容:**

```python
# 修正前（バグ）
return self.prob_up >= direction_threshold or self.prob_down < block_threshold

# 修正後（正常）
return self.prob_up >= direction_threshold and self.prob_down < block_threshold
```

**閾値調整:**

| 精度レベル | 修正前 | 修正後 | 理由 |
|----------|--------|--------|------|
| >=0.45 | up≥0.40 OR down<0.55 | up≥0.45 AND down<0.35 | より厳格 |
| >=0.40 | up≥0.35 OR down<0.60 | up≥0.40 AND down<0.45 | 中程度 |
| <0.40 | up≥0.30 OR down<0.65 | up≥0.35 AND down<0.50 | Pine尊重 |

**影響:**  
修正前は「ほぼ全シグナル通過」状態から、修正後は「正方向確率が高く、逆方向確率が低い」場合のみ通過に改善。

---

### 3. ATRスパイク検知実装 ✅

**変更ファイル:**
1. [pinescript/mtf_smc_v2_3.pine](pinescript/mtf_smc_v2_3.pine) - JSON に `atr_20d_avg` を追加
2. [main.py](main.py) - webhook から ATR情報キャッシュ、diff_detection_task で使用

**実装フロー:**

1. **Pine Script側:** バーの確定時に `atr_20d_avg` (20日ATR平均) を JSON ペイロードに追加
2. **Webhook 受信:** `current_atr`, `atr_20d_avg` をメモリキャッシュ (`_last_webhook_atrs`)
3. **15分ごとの差分検知タスク:** 
   - ニュースハッシュ変化 or
   - ATR急変 (`current_atr > avg_atr_20d × 1.5`) or
   - キャッシュ期限切れ（60分）
   - → GPT-5.2呼び出し判定

**効果:**  
カレンダーVeto外の意外なニュースイベントで、LLMが市場変動を分析。エントリー判定の精度向上。

---

### 4. SL二重計算の統一 ✅

**問題:** Orchestrator と Broker で異なるSL値を計算・管理

**修正内容:**

#### 4-1. Trailing Stop のSL更新時の同期確認

**ファイル:** [core/position_manager.py](core/position_manager.py#L260-L289)

```python
# 修正前: 返値チェックなし
await self._broker.modify_sl_tp_async(pos.ticket, sl=round(new_sl, 5))
pos.sl_price = new_sl

# 修正後: 返値チェック + 失敗通知
ok = await self._broker.modify_sl_tp_async(pos.ticket, sl=round(new_sl, 5))
if ok:
    pos.sl_price = new_sl
    logger.info(f"Trailing SL updated: {pos.pair}")
else:
    logger.error(f"Failed to update trailing SL")
    await self._notifier.send("⚠️ Trailing SL更新失敗", AlertLevel.WARNING)
```

#### 4-2. Orchestrator での不要なSL計算削除

**ファイル:** [main.py](main.py#L595-L607) ドテン処理

```python
# 修正前: 計算値を上書きされる
sl_price, tp_price = self._calc_sl_tp_price(...)
ok, new_ticket, sl_price, tp_price = await execute_doten(...)  # ← 上書き

# 修正後: 計算削除、Broker返値をそのまま使用
ok, new_ticket, sl_price, tp_price = await execute_doten(...)
```

**原理:**
- 実SL値（MT5に設定される値）= Broker が**発注時の実tick価格**から計算
- Orchestrator が計算する値（webhook の close_price ベース）= 参考値

Broker返値を正規化し、参考値を削除。トレーリング失敗時も ローカル状態を修正しない（MT5の実状と不一致を防ぐ）。

---

### 5. デッドコード削除 ✅

**削除対象:** [broker/mt5_broker.py](broker/mt5_broker.py) L397-450

**削除内容:**
- `async def partial_close_async()`
- `def _partial_close_sync()`
- スケールアウト機能（~50行）

**理由:**  
Orchestrator から呼び出しなし。設計書では言及あるが、実装されておらず保守負荷のみ。

---

### 6. demo_mode フラグ削除 ✅

**削除ファイル:**

1. [config.json](config.json)  
   - `"demo_mode": true` 削除

2. [config/settings.py](config/settings.py)  
   - `demo_mode: bool = True` 定義削除

3. [main.py](main.py)  
   - L133: `logger.info(f"Demo mode: {self._config['system']['demo_mode']}")` 削除  
   - L183: `f"デモモード: {self._config['system']['demo_mode']}\n"` 削除

**理由:**  
- MT5接続先でデモ/リアル口座を切り替える (`mt5_server` 設定）
- demo_mode フラグは形骸化・機能実装なし
- ユーザーメモ：デモ/リアル切り替えは `.env` で `MT5_SERVER` 環境変数を変更して対応

---

## 📋 確認事項・制限事項

### A. LightGBMモデルの実運用精度

**現状:**
- Bootstrap モデルは合成データで学習
- デフォルト精度値 (USDJPY: 0.3785) はハードコード
- 実データによる学習前

**本番前チェックリスト:**
- [ ] 実運用3ヶ月分のデータ蓄積
- [ ] 実データによるモデル再学習
- [ ] ウォークフォワード検証（PSIドリフト確認）
- [ ] 勝率・Sharpe指標の検証

---

### B. は に関するCoude Evaluation

Claude のレポート（strategy_evaluation_report.md）での指摘を参考。特に注視すべき点：

1. **is_strong_signal の改善度**: 修正でフィルター機能が返ったが、実データでの検証必要
2. **ATRスパイク検知の1.5倍値**: ユーザーの懸念アリ。市場変動に応じて調整検討
3. **SL設定の精度**: Trailing Stop の同期改善で向上（失敗ログがあれば検証可能）

---

### C. 再起動時のポジション復元

**現状:** MT5 から SL値を取得して復元 ✅

**ルーチン:**
```python
# main.py L251-310 _restore_managed_positions_from_broker()
managed = ManagedPosition(
    ...
    sl_price=float(broker_pos.get("sl_price") or 0.0),  # ✅ MT5から正確な値を取得
    ...
)
```

---

### D. Pine Script 更新確認事項

**Pine Script v2.3 での`atr_20d_avg` 追加** ✅

```pinescript
# L266
atr_avg_20 = ta.sma(atr_14, i_atr_avg * (60 / 15))  // 20日平均

# L499: f_build_json内で追加
_p3 = ',...,"atr_20d_avg":' + str.tostring(math.round(atr_avg_20, 5)) + ',...'
```

TradingView チャートに適用後、webhook ペイロードで `atr_20d_avg` が送られて来ることを確認。

---

## 🔧 デプロイチェックリスト

- [x] Python 構文チェック（全修正ファイル） ✅
- [x] インポート確認（未使用import なし） ✅
- [x] デッドコード削除（~50行） ✅
- [ ] Pine Script の TradingView 反映確認（デプロイ時）
- [ ] DB マイグレーション確認（スキーマ変更なし）
- [ ] 環境変数確認 (`.env` に OpenAI/Discord Key 設定)
- [ ] ログディレクトリ作成確認
- [ ] 最初の webhookシグナル受信テスト

---

## 📊 修正前後の比較

### is_strong_signal の実効フィルター率

| シナリオ | 修正前 | 修正後 |
|---------|--------|--------|
| 上昇確実（prob_up=0.75） | ✅ 通過 | ✅ 通過 |
| 下落確実（prob_down=0.75） | ✅ 通過 | ✅ 通過 |
| 不確定（prob_up=0.35, prob_down=0.40） | ✅ **通過（バグ）** | ❌ 除却（正常）|
| 下落示唆（prob_down=0.65）ロングシグナル | ✅ 通過（バグ） | ❌ 除却（正常） |

### SL管理の堅牢性

| 項目 | 修正前 | 修正後状態 |
|-----|--------|--------|
| Trailing更新失敗時 | ローカル状態更新 → MT5とズレ | ログ + 通知、ローカル状態保留 |
| Orchestrator計算の必要性 | 毎回計算（無駄） | ドテン時削除済み |

---

## 📞 問い合わせ・改善提案

本ドキュメントは実装進捗の現況報告です。設計書 (FX_AutoTrading_System_Design_v2.2.md) も参照してください。

**次フェーズ (v2.4):**
- エグジット戦略の付加改善（trailing 活性化条件）
- LightGBMハイパーパラメータ自動最適化
- スケールアウト機能の再実装（オプション）

