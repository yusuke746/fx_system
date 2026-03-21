# FX自動売買システム 詳細設計書 v2.3

> 実装同期メモ: 現行コードは LightGBM 35特徴量、Pine `mtf_smc_v2_3.pine`、動的エグジット（`time_decay` / `structural_tp` / trailing）に移行済みです。本文に旧 v2.2 前提の記述が残る場合は、この注記と実装を優先してください。

**対象通貨:** USD/JPY · EUR/USD · GBP/JPY（GOLD: v3.0以降）
**実行環境:** Windows レンタルサーバー（常時稼働 / VPS）
**ブローカー:** XMTrading KIWA極口座
**ステータス:** 設計凍結 → Phase 1実装へ移行

---

## 目次

1. [システムアーキテクチャ](#1-システムアーキテクチャ)
2. [TradingView Pine Script設計](#2-tradingview-pine-script設計)
3. [GPT-5.2モジュール設計（差分検知）](#3-gpt-52モジュール設計差分検知)
4. [LightGBM設計](#4-lightgbm設計)
5. [エグジット戦略](#5-エグジット戦略)
6. [ブローカー設計](#6-ブローカー設計)
7. [DB・メンテナンス設計](#7-dbメンテナンス設計)
8. [実装ロードマップ](#8-実装ロードマップ)
9. [付録](#付録)
   - [F. カレンダーAPI選定](#f-カレンダーapi選定)
   - [G. 環境変数管理（.env）](#g-環境変数シークレット管理env)
   - [H. リスク管理設計](#h-リスク管理設計)

---

## 1. システムアーキテクチャ

### 1.1 コンポーネント一覧（確定版）

| コンポーネント  | 役割                                                                                 | 実装           |
| --------------- | ------------------------------------------------------------------------------------ | -------------- |
| TradingView     | Pine Script（MTF SMC: 4H+1H+15M 3層構造）→ Webhook発火                              | Pine Script v5 |
| GPT-5.2 Instant | 差分検知タスク（ATR×1.5閾値・低消費モード安全装置付き）→ 15分ごとに判定            | OpenAI API     |
| Python Veto A   | カレンダーIFルール（FOMC/雇用統計等の前後30分を即時ブロック）                        | Python         |
| Python Veto B   | GPT-5.2の突発イベントフラグ（`unexpected_veto: true`）                             | Python         |
| LightGBM        | 35特徴量（MTF SMC版）→ 上昇/横ばい/下落 確率出力                                    | LightGBM 4.x   |
| エグジット      | ATR動的SL / time decay / structural TP / trailing / 金曜クローズ | Python         |
| ブローカー      | XMTrading KIWA極口座 / MT5 Python / asyncio + ThreadPoolExecutor(max_workers=1)      | MetaTrader5    |

### 1.2 クリティカルパス（目標500ms以内）

```
① TradingView Webhook受信（MTF SMC + テクニカル条件成立時のみ）
② Calendar Veto確認（Python IFルール・0ms）
③ GPT-5.2キャッシュ読込（メモリから即時・0ms）
④ LightGBM推論（35特徴量・1ms以下）
⑤ ドテン判定 → インターバル確認（1時間以内の2回目ドテンは禁止）
⑥ 高速連続発注（決済A完了確認後に新規B送信）/ MT5 ThreadPoolExecutor(max_workers=1)
```

### 1.3 バックグラウンドタスク

```
毎15分:  差分検知タスク（ニュースハッシュ比較・ATR×1.5急変確認・60分キャッシュ期限）
          → 差分あり or 急変 or 期限切れ の時のみ GPT-5.2呼び出し
          → 3回連続スキップなら低消費モード（次の呼び出しは90分後に延長）

毎1分:   ポジション監視（time_decay・ structural_tp・トレーリング・金曜Exit）

毎日深夜01:00 JST:  DBメンテ・バックアップ・カレンダーキャッシュ更新

毎週土曜14:00 JST:  週末最適化は旧 scale-out 前提のため安全停止（設定記録のみ）

毎週日曜23:00 JST:  LightGBM再学習 + ウォークフォワード検証 + PSIドリフト検知

毎月月初日曜02:00:  古いモデル削除・月次集計・クラウドバックアップ・APIコストレポート
```

### 1.4 対象銘柄の最終決定

**確定: USD/JPY・EUR/USD・GBP/JPY の3ペアのみ（GOLD = v3.0以降）**

GOLD追加見送り理由:

1. **Liquidity Sweepの規模が別次元** — XAU/USDのストップ狩りは通貨ペアの2〜3倍の振れ幅。現在のATR×1.5 SL設計では狩られる。
2. **USD相関による実質的なリスク重複** — USD/JPYとXAU/USDはDXYを通じて逆相関。GOLDを加えるとポジションサイジングが破綻する可能性がある。
3. **Phase 1で複雑性を増やさない原則** — まず3ペアで安定稼働することが最優先。

---

## 2. TradingView Pine Script設計

### 2.1 MTF SMC 3層構造（確定版）

| 時間足               | SMC概念            | 役割                                | Pine Script実装方針                                                               |
| -------------------- | ------------------ | ----------------------------------- | --------------------------------------------------------------------------------- |
| 4時間足（方向）      | Order Block        | 相場の強固な壁を特定                | 4H OBゾーン（上限・下限）をseries変数で保持。15Mが到達した時のみ `ob_4h_flag=1` |
| 4時間足（方向）      | Fair Value Gap     | 4Hレベルのインバランス              | 4H FVGゾーンを記録。15M終値がゾーン内なら `fvg_4h_flag=1`                       |
| 1時間足（壁・Sweep） | Liquidity Sweep    | 機関投資家の個人SL狩りを検出        | 1H直近高値/安値を1H足で突破→反転を検出。直近5本以内なら `liq_1h_flag=1`        |
| 1時間足（構造）      | BOS                | 1H相場の構造転換を確認              | 1H swing high/lowのブレイクで `bos_1h_flag=1`                                   |
| 15分足（実行）       | エントリートリガー | 上記MTF条件が揃った時のみエントリー | `mtf_confluence >= 2` かつ MACDクロス or RSI反転ゾーンでWebhook発火             |

### 2.2 アラート発火条件（確定版）

Pine Script発火条件（**全て同時成立が必要**）:

- **【必須1】上位TFトレンド:** 4H EMA20 vs EMA50 の方向
- **【必須2】MTF合致スコア:** `mtf_confluence >= 2`（4H OB/FVG + 1H Sweep/BOSのうち2つ以上）
- **【必須3】ボラティリティ:** `ATR(14) > 20日平均ATRの0.8倍` かつ BBスクイーズ中でない
- **【必須4】時間フィルター:** ロンドン・NY・重複セッション中 / 深夜00:00〜07:00 JST除外
- **【推奨】モメンタム:** MACDヒストグラムゼロクロス or RSI(14) が30〜35 or 65〜70ゾーン

**発火時Webhookに含めるSMCデータ:**

```json
{
  "pair": "USDJPY",
  "direction": "long",
  "fvg_4h_zone_active": true,
  "ob_4h_zone_active": false,
  "liq_sweep_1h": true,
  "bos_1h": false,
  "mtf_confluence": 2,
  "atr": 0.45,
  "close": 149.85
}
```

---

## 3. GPTモジュール設計（ハイブリッド差分検知）

### 3.1 採用モデルの選定

**採用モデル: `gpt-5-nano`（一次判定） + `gpt-5.2`（重要時の二次精査）**

GPT-5.2はOpenAIが2025年12月11日にリリースした3バリアント構成のモデル。

| バリアント           | 用途                                                   | コスト感            |
| -------------------- | ------------------------------------------------------ | ------------------- |
| **gpt-5-nano** | 差分検知の一次判定（常時実行・現行既定）               | 最低                |
| **gpt-5.2**    | 重要ニュース時の二次精査（昇格時のみ）                 | 低〜中              |
| gpt-5.2-nano         | 互換候補（環境差分対策）                               | 低                  |
| gpt-5.2-thinking     | 複雑な地政学・政策判断が必要な局面（手動切り替えのみ） | 中                  |
| gpt-5.2-pro          | 使わない（コスト過剰）                                 | 高（$21/Mトークン） |

**モデル名はconfig.jsonで管理し、コードにハードコーディングしない。**

```python
# llm_client.py
with open("config.json") as f:
    config = json.load(f)

LLM_MODEL_DIFF = config["llm"]["model_diff"]      # "gpt-5-nano"
LLM_MODEL_DEEP = config["llm"]["model_instant"]   # "gpt-5.2"
```

### 3.2 v2.2変更点

- **ATR閾値:** ATR×2.0 → **ATR×1.5**（感度を高めて機会損失対策）
- **低消費モード安全装置を追加:** 15分ごとの差分チェックで3回連続「差分なし」の場合、次のGPT呼び出し可能タイミングを90分後に延長

### 3.3 差分検知判定フロー（15分ごと）

```
STEP 1: ニュースフィードから最新10件を取得（コストゼロ）
STEP 2: 前回記事のMD5ハッシュと比較
        → 新しい記事あり: call_gpt = True（理由: NEW_ARTICLE）
STEP 3: 直近15分のATR変化を確認
        → 現在ATR > 20日平均ATR × 1.5: call_gpt = True（理由: PRICE_SPIKE）
STEP 4: Calendar Veto発動中か確認
        → Veto中: call_gpt = False（スキップ）
STEP 5: キャッシュ経過時間を確認
        → 通常モード: キャッシュ > 60分 → call_gpt = True（理由: CACHE_EXPIRED）
        → 低消費モード: キャッシュ > 90分 → call_gpt = True（理由: CACHE_EXPIRED_LOW）
STEP 6: 低消費モードの判定
        → 直近3回連続でSTEP 1〜3がすべてFalseだった場合: 低消費モードに移行
        → 低消費モード中にNEW_ARTICLEまたはPRICE_SPIKEが来たら即座に通常モードに戻る
STEP 7: call_gpt = True の場合、まず model_diff（既定: gpt-5-nano）で一次判定
STEP 8: news_importance_score が閾値以上 or unexpected_veto=True の場合のみ gpt-5.2 へ昇格
STEP 9: 重要ニュース時のみ sentiment_score を特徴量へ反映
```

### 3.4 コスト試算（ATR×1.5・低消費モード込み）

| シナリオ                          | 月次呼び出し数       | 月次コスト推定   | 備考                           |
| --------------------------------- | -------------------- | ---------------- | ------------------------------ |
| v2.0（無条件15分）                | 2,880回              | $15〜90          | 廃止                           |
| v2.1（ATR×2.0）                  | 300〜700回           | $2〜20           | 比較用                         |
| **v2.2（ATR×1.5+低消費）** | **500〜900回** | **$3〜27** | 感度UP・低消費モードで上限抑制 |
| v2.2 低消費モード多用時           | 300〜500回           | $2〜15           | 深夜・静かな相場が長い場合     |

---

## 4. LightGBM設計

### 4.1 確定パラメータ

```python
params = {
    "max_depth": 4,
    "min_child_samples": 50,
    "reg_alpha": 0.2,
    "reg_lambda": 0.3,
    "n_estimators": 300,
    "objective": "multiclass",
    "num_class": 3,  # 上昇/横ばい/下落
}

SL_MULTIPLIER = 1.5   # ATR × 1.5
SL_TP_MIN_PIPS = 15
SL_TP_MAX_PIPS = 35
```

### 4.2 35特徴量リスト（MTF SMC版）

**SMCフラグ（5個）:**

- `fvg_4h_zone_active` — 4H FVGゾーン内かどうか
- `ob_4h_zone_active` — 4H OBゾーン内かどうか
- `liq_sweep_1h` — 1H Liquidity Sweepの有無
- `bos_1h` — 1H BOSの有無
- `mtf_confluence` — MTF合致スコア（0〜4）

**価格・ボラティリティ系（10個）:**

- `atr_14` — ATR(14)の現在値
- `atr_ratio` — 現在ATR / 20日平均ATR
- `bb_width` — ボリンジャーバンド幅（スクイーズ検知用）
- `close_vs_ema20_4h` — 4H EMA20との乖離率
- `close_vs_ema50_4h` — 4H EMA50との乖離率
- `high_low_range_15m` — 直近15M足の高安レンジ
- `trend_direction` — 上位足トレンド方向（-1/0/1）
- `momentum_long` — ロング方向モメンタムの強さ
- `momentum_short` — ショート方向モメンタムの強さ

**モメンタム系（7個）:**

- `macd_histogram` — MACDヒストグラム値
- `macd_signal_cross` — ゼロクロス有無（-1/0/1）
- `rsi_14` — RSI(14)の現在値
- `rsi_zone` — RSIゾーン（oversold=1, overbought=-1, neutral=0）
- `stoch_k` — ストキャスティクス %K
- `stoch_d` — ストキャスティクス %D
- `momentum_3bar` — 直近3本のクローズ変化率

**構造・パターン系（6個）:**

- `ob_4h_distance_pips` — 4H OBゾーンまでの距離（pips）
- `fvg_4h_fill_ratio` — 4H FVGの充填率（0〜1）
- `liq_sweep_strength` — Sweepの強さ（突破幅/ATR）
- `prior_candle_body_ratio` — 直前ローソク足の実体比率
- `consecutive_same_dir` — 同方向ローソク足の連続本数
- `pivot_proximity` — 直近ピボットポイントまでの距離（pips）

**リスク・ポジション系（4個）:**

- `open_positions_count` — 現在のオープンポジション数
- `max_dd_24h` — 直近24時間の最大DD（%）
- `calendar_risk_score` — 経済指標リスクスコア（0=安全/1=注意/2=高危険）
- `sentiment_score` — GPT-5.2出力のセンチメントスコア（-1.0〜1.0）

### 4.3 学習・検証設計

```
学習データ: 直近90日
ウォークフォワード: 訓練60日 / 検証15日
週次再学習: 毎週日曜23:00（直近90日データ）
ドリフト検知: 直近20トレード勝率<40% or PSI>0.2 → 緊急再学習 + Discord通知
モデル管理: 通貨ペアごとに独立モデル（3モデル）
保存世代数: 3世代（月次クリーンアップ）
```

---

## 5. エグジット戦略

### 5.1 エグジット優先順位（v2.3実装）

| 優先度              | 種別                      | 詳細・実装                                               |
| ------------------- | ------------------------- | -------------------------------------------------------- |
| **1（最高）** | Calendar Veto強制クローズ | 重要指標前後の veto 発動時は全ポジション強制クローズ     |
| **2**         | ATR動的SL到達             | MT5のOCO注文として設定済み（SL=ATR×1.5）。MT5が自動執行 |
| **3**         | Time decay                | 一定時間経過後も期待利益が伸びないポジションを解消       |
| **4**         | Structural TP             | 4H OB距離ベースの目標到達で利確                          |
| **5**         | トレーリングストップ      | 高値/安値更新に応じてATRベースで追従                     |
| **6**         | 金曜安全クローズ          | 金曜クローズ前に残ポジションを解消                       |

### 5.2 ドテンインターバル制限（v2.2新規）

```python
# position_manager.py
last_doten_time: dict[str, datetime] = {}

def check_doten_allowed(pair: str) -> bool:
    last = last_doten_time.get(pair)
    if last is None:
        return True
    return (datetime.now() - last).total_seconds() >= 3600
```

### 5.3 「高速連続発注」の正確な定義とThreadPoolExecutorの位置づけ

**重要な前提: MT5 PythonライブラリはAPIレベルでスレッドセーフではない。**

`ThreadPoolExecutor(max_workers=1)` でワーカーを1つに制限し、MT5への発注はシリアル（直列）に処理する。これはMT5 APIの制約に起因する設計上の必須制限であり、変更してはならない。

| 用語                       | 扱い                                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| ~~「asyncio並列発注」~~   | 廃止（MT5への発注は実際にはシリアル処理のため誤解を招く）                                            |
| **「高速連続発注」** | 採用する正確な表現。asyncioのI/O待機重複解消の効果は生かしつつ、決済→確認→新規の順序保証を明示する |

```python
async def execute_doten(pair: str, ticket: int, direction: str,
                        volume: float, sl: float, tp: float) -> bool:
    """
    ドテン高速連続発注。
    asyncio.gather(close, open) を使わない理由:
    - 決済失敗時のロールバック処理が書きにくい
    - 明示的な順序制御（決済→確認→新規）の方が障害時の挙動が予測しやすい
    """
    close_ok = await close_position_async(ticket)
    if not close_ok:
        await notify_slack(f"[DOTEN FAIL] {pair} 決済失敗 ticket={ticket}")
        return False  # 新規エントリーは行わない

    open_ok = await open_position_async(pair, direction, volume, sl, tp)
    if not open_ok:
        await notify_slack(f"[DOTEN FAIL] {pair} 新規建て失敗（決済済み・ノーポジ）")
        return False

    last_doten_time[pair] = datetime.now()
    return True
```

### 5.4 ドテン判定フロー（完全版）

```
LightGBMから逆方向>65%の出力を受信
↓
エントリーから15分（1本）経過しているか？
→ No: 何もしない（エントリー直後の一時的逆行を無視）
↓ Yes
ドテンインターバル確認: last_doten_time[pair] から1時間以上経過しているか？
→ No: 決済のみ実行（新規エントリー禁止）
↓ Yes
execute_doten() を呼び出す
  └ 決済成功 → 新規建て → last_doten_time[pair] = now
  └ 決済失敗 → 新規建てなし → Discord通知 → フォールバック処理へ（付録D参照）
```

---

## 6. ブローカー設計

```
口座:     XMTrading KIWA極口座（コミッションなし・変動スプレッド）
方針:     デモ口座での検証後、本番移行
バックテスト前提: spread_pips=1.5（保守的固定）/ commission_per_lot=0
プラットフォーム: MetaTrader5 + Python
実行環境: Windows VPS（MT5常時起動 / Windowsタスクスケジューラで自動再起動）
セキュリティ: Windows Firewall（Webhookポートのみ開放）+ TradingView IPホワイトリスト + HMAC-SHA256署名検証
```

---

## 7. DB・メンテナンス設計

### 7.1 DBスキーマ（主要テーブル）

```sql
CREATE TABLE trades (
    id          INTEGER PRIMARY KEY,
    pair        TEXT NOT NULL,
    direction   TEXT NOT NULL,         -- 'long', 'short'
    open_time   TIMESTAMP NOT NULL,
    close_time  TIMESTAMP,
    open_price  REAL NOT NULL,
    close_price REAL,
    volume      REAL NOT NULL,
    sl_price    REAL,
    tp_price    REAL,
    pnl_pips    REAL,
    pnl_jpy     REAL,
    exit_reason TEXT,                  -- 'atr_sl'|'doten'|'time_exit'|'calendar_veto'|'trailing'|'time_decay'|'structural_tp'
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE signals (
    id              INTEGER PRIMARY KEY,
    pair            TEXT NOT NULL,
    signal_time     TIMESTAMP NOT NULL,
    direction       TEXT,
    lgbm_prob_up    REAL,
    lgbm_prob_flat  REAL,
    lgbm_prob_down  REAL,
    gpt_sentiment   REAL,
    mtf_confluence  INTEGER,
    executed        BOOLEAN DEFAULT FALSE,
    veto_reason     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_call_log (
    id          INTEGER PRIMARY KEY,
    call_time   TIMESTAMP NOT NULL,
    reason      TEXT NOT NULL,         -- 'NEW_ARTICLE'|'PRICE_SPIKE'|'CACHE_EXPIRED'|'CACHE_EXPIRED_LOW'
    model       TEXT NOT NULL,         -- 'gpt-5-nano' | 'gpt-5.2'
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    REAL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE optimization_history (
    id                   INTEGER PRIMARY KEY,
    optimized_at         TIMESTAMP NOT NULL,
    optimize_window_days INTEGER NOT NULL,   -- 28
    validate_window_days INTEGER NOT NULL,   -- 14
    sl_multiplier        REAL,
    exit_prob_threshold  REAL,
    time_decay_minutes   INTEGER,
    time_decay_min_profit_atr REAL,
    sharpe_optimize      REAL,               -- 最適化期間（28日）内のSharpe比
    sharpe_validate      REAL,               -- 検証期間（14日）内のSharpe比
    sample_count         INTEGER,
    was_applied          BOOLEAN DEFAULT TRUE,
    rollback_reason      TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 7.2 最適化の現行運用（v2.3実装同期）

**方針:** 動的エグジット移行により、旧 scale-out / 固定TP 前提のグリッド最適化は停止。

#### 土曜14:00 JST（`run_weekend_optimization`）

```
STEP 1: 直近28日トレード件数を集計
STEP 2: optimization_history に記録（was_applied=False, reason=dynamic_exit_strategy_active）
STEP 3: Discordへ「不採用（安全停止）」を通知
```

#### 日曜23:00 JST 後（`auto_tune_execution_noise`）

```
STEP 1: 直近7日のクローズ済みトレードを集計
STEP 2: 早期ノイズ比率・exit理由比率・平均保有時間を算出
STEP 3: execution layer のみ微調整
    - exit_prob_stale_minutes（10〜90）
    - trailing_update_cooldown_seconds（10〜180）
    - trailing_min_step_pips（0.5〜6.0）
STEP 4: 変更時のみ config.json に反映
```

#### 安全装置

| 安全装置               | 内容                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| 旧ロジック最適化の停止 | `dynamic_exit_strategy_active` を理由に自動適用を禁止             |
| 対象の限定             | AI判断ロジックは変更せず、執行ノイズ抑制パラメータのみ更新          |
| サンプル数チェック     | 週次チューニングは最低20件未満でスキップ                            |
| 範囲制限               | 各パラメータはクランプ範囲内でのみ変更                              |
| 手動オーバーライド     | `config.json` の `optimization.auto_optimize: false` で停止可能 |

### 7.3 全自動メンテナンススケジュール

| 頻度     | 時刻（JST）     | 実行内容                                                                                                                                                                                        |
| -------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 毎日     | 深夜01:00       | ログローテーション（30日以前を圧縮）/ sentimentキャッシュ切り詰め / DBバックアップ（直近7日保持）/ 翌日カレンダーキャッシュ更新 / Discordにポジションサマリー送信                               |
| 毎週土曜 | **14:00** | 週末最適化タスク（旧グリッド最適化は安全停止・`optimization_history`記録のみ）                                                                                                                |
| 毎週日曜 | 23:00           | LightGBM再学習（直近90日）/ ウォークフォワード検証 / PSIドリフト検知 / signalsアーカイブ移動 / VACUUM ANALYZE / api_call_log週次集計 / 執行ノイズ抑制パラメータ微調整 / 週次レポートDiscord送信 |
| 毎月月初 | 日曜02:00       | 古いモデルファイル削除（3世代保持）/ tradesテーブル月次集計 / クラウドバックアップ / GPT-5.2コストレポート                                                                                      |

---

## 8. 実装ロードマップ

| Phase             | 期間        | 実装内容                                                                                                                                                        | Go判定基準                                                          |
| ----------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Phase 1** | Week 1〜2   | ① KIWA極デモ口座 + MT5（VPS）起動確認 ② Pine Script MTF SMC ③ Webhook受信（FastAPI）+ 署名検証 ④ MT5発注テスト                                              | MT5手動発注OK / Webhook署名検証OK / 4H FVGゾーン正常表示            |
| **Phase 2** | Week 3〜4   | ① GPT-5.2差分検知タスク ② Veto 2層 ③ 経済指標カレンダーAPI連携 ④ api_call_log                                                                               | 1日の呼び出し回数10〜25回 / FOMC前後でVeto A自動発動                |
| **Phase 3** | Week 5、7   | ① 35特徴量エンジニアリング ② LightGBM学習（3ペア独立）③ ウォークフォワード検証 ④ ATR動的SL/TP ⑤ バックテスト                                               | WF検証: 平均Sharpe比>1.0 / 最大DD<15% / SMC特徴量が重要度上位       |
| **Phase 4** | Week 8〜9   | ① ドテン高速連続発注 + インターバル制限 ② スケールアウト ③ 相関リスク管理 ④**フォールバック全パターン実装（付録D参照）** ⑤ DBスキーマ + 全自動メンテ | ドテン動作確認 / インターバル誤動作なし / 1週間の自動メンテ正常動作 |
| **Phase 5** | Week 10〜13 | ① デモ口座1ヶ月フル稼働 ② GPT-5.2コスト週次記録 ③ 週末最適化収束確認 ④ Discord監視・週次レポート                                                            | 勝率>50% / 最大DD<10% / 週末最適化が3週間安定収束                   |
| **Phase 6** | Week 14〜   | ① 本番口座移行（最小ロット0.01から）② 本番スプレッドで再バックテスト ③ 週次レビュー定例化                                                                    | 3ヶ月デモと同等 / 月次純損益（APIコスト差引後）がプラス             |

---

## 付録

### A. 技術スタック

| カテゴリ         | 採用技術                                                      | ステータス                          |
| ---------------- | ------------------------------------------------------------- | ----------------------------------- |
| OS               | Windows Server 2019/2022                                      | 確定                                |
| ブローカー       | XMTrading KIWA極口座                                          | コミッションなし・確定              |
| MT5 Python       | MetaTrader5 + asyncio + ThreadPoolExecutor(max_workers=1)     | シリアル発注・スレッドセーフ対応    |
| 言語             | Python 3.11+                                                  | 確定                                |
| Webサーバー      | FastAPI + uvicorn                                             | 確定                                |
| 機械学習         | LightGBM 4.x + scikit-learn                                   | 確定                                |
| LLM              | OpenAI GPT-5系（`gpt-5-nano` + `gpt-5.2` のハイブリッド） | 確定                                |
| データ           | TradingView（MTF Pine Script: 4H+1H+15M）                     | 確定                                |
| DB（デモ）       | SQLite                                                        | Phase 1〜4                          |
| DB（本番）       | PostgreSQL（月次パーティション）                              | Phase 5以降推奨                     |
| スケジューラ     | Windowsタスクスケジューラ                                     | 確定                                |
| 監視             | Discord Webhook                                               | 確定（ALERT/CRITICAL 専用chを分離） |
| カレンダーAPI    | Forex Factory XML Feed（無料・登録不要）                      | 付録F参照                           |
| シークレット管理 | python-dotenv / pydantic-settings (.env)                      | 付録G参照                           |
| 最適化           | 旧グリッド探索は安全停止 / execution noise のみ週次自動調整   | v2.3実装同期                        |

### B. 設計完成度の推移

| バージョン     | Claude         | Gemini             | 主な変更内容                                                        |
| -------------- | -------------- | ------------------ | ------------------------------------------------------------------- |
| v1.0           | 65点           | 75点               | 基本設計                                                            |
| v2.0           | 75点           | 90点               | 非同期キャッシュ・特徴量削減・エグジット4層・DB設計                 |
| v2.1           | 82点           | 92点（想定）       | 差分検知・ドテン並列・KIWA極・MTF SMC                               |
| **v2.2** | **86点** | **設計凍結** | ATR×1.5+低消費モード・ドテンインターバル・週末自動最適化・GOLD延期 |

### C. config.json スキーマ

```json
{
  "system": {
    "version": "2.2",
    "pairs": ["USDJPY", "EURUSD", "GBPJPY"],
    "demo_mode": true
  },
    "llm": {
        "model_instant": "gpt-5.2",
        "model_diff": "gpt-5-nano",
        "model_thinking": "gpt-5.2",
        "reasoning_effort_instant": "low",
        "reasoning_effort_diff": "low",
        "reasoning_effort_thinking": "medium",
        "hybrid_enabled": true,
        "news_importance_escalation_threshold": 0.65,
        "feature_news_importance_threshold": 0.55,
        "web_search_enabled": true,
        "web_search_tool_type": "web_search_preview",
        "web_search_context_size": "low",
        "llm_enabled": true,
        "atr_threshold_multiplier": 1.5,
        "cache_ttl_normal_minutes": 60,
        "cache_ttl_low_power_minutes": 90,
        "low_power_consecutive_skips": 3
    },
  "risk": {
    "max_risk_per_trade_pct": 2.0,
    "max_daily_drawdown_pct": 10.0,
    "max_total_exposure_pct": 10.0,
    "max_positions": 5,
    "max_usd_exposure": 4,
    "max_jpy_exposure": 2,
    "sl_multiplier": 1.5,
    "sl_min_pips": 15,
    "sl_max_pips": 35,
    "exit_prob_threshold": 0.35,
    "time_decay_minutes": 60,
    "time_decay_min_profit_atr": 0.5,
    "trailing_atr_multiplier": 0.8,
    "exit_prob_stale_minutes": 30,
    "trailing_update_cooldown_seconds": 30,
    "trailing_min_step_pips": 2.0,
    "doten_interval_seconds": 3600
  },
  "optimization": {
    "auto_optimize": true,
    "optimize_window_days": 28,
    "validate_window_days": 14,
    "min_sample_count": 30,
    "max_change_ratio": 0.20,
    "rollback_winrate_threshold": -0.10
  },
  "broker": {
    "spread_assumption_pips": 1.5,
    "commission_per_lot": 0
  }
}
```

### D. フォールバック設計（Phase 4実装必須）

**基本方針: 不確実な状態でのエントリーは禁止。ポジション保護（決済）は最優先。**

#### D.1 LLMモジュール障害

| シナリオ                | 検知方法                             | フォールバック動作                                                                                                 | 復旧条件                                            |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| APIタイムアウト（>5秒） | `asyncio.wait_for` の TimeoutError | キャッシュ済みsentiment値を使用。キャッシュ切れ（>90分）の場合は `sentiment_score=0.0`（ニュートラル）として処理 | 次回差分検知タスク（15分後）で自動復旧              |
| APIレート制限（429）    | HTTPステータスコード429              | exponential backoff（1s→2s→4s→8s）でリトライ×3回。3回失敗でキャッシュ値を使用                                  | バックオフ後に正常応答で自動復旧                    |
| 認証エラー（401/403）   | HTTPステータスコード401/403          | 即座にDiscord緊急通知。以降のGPT呼び出しを全停止（`llm_enabled: false`）。Veto B = 常にFalse（通過）として扱う   | 手動でAPIキーを更新後、`llm_enabled: true` に戻す |
| JSONパースエラー        | `json.JSONDecodeError`             | 当該呼び出しを破棄。キャッシュ値を使用。3回連続失敗でDiscord通知                                                   | 次回正常応答で自動復旧                              |
| Veto B誤連発            | 1時間以内に3回以上Veto B発動         | Discord通知（人間確認を促す）。自動処理は変更なし                                                                  | 手動確認後に必要なら一時停止                        |

#### D.2 MT5 / ブローカー障害

| シナリオ                           | 検知方法                                                    | フォールバック動作                                                                                                         | 復旧条件                                        |
| ---------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| MT5接続断                          | `mt5.initialize()` 失敗 or `mt5.terminal_info()` = None | 1分間隔で最大10回再接続を試みる。再接続中の新規エントリー禁止。既存ポジションのSL/TPはMT5側のOCO注文が維持するため問題なし | 再接続成功で自動復旧。10回失敗でDiscord緊急通知 |
| 発注拒否（資金不足・証拠金エラー） | `retcode != 10009`                                        | 発注キャンセル。Discord通知。当該ペアの新規エントリーを30分停止                                                            | 30分後に自動解除                                |
| 発注拒否（価格乖離）               | `retcode in [10018, 10021]`                               | slippageを広げて1回リトライ（最大1pips）。それでも失敗ならキャンセル                                                       | リトライ成功 or キャンセルで自動復旧            |
| ドテン中の決済失敗                 | `close_position_async()` が False を返す                  | **新規建ては絶対に行わない。** Discord緊急通知（手動確認を促す）。既存ポジションのSL/TPはそのまま維持                | 手動確認後に操作                                |
| MT5プロセスクラッシュ              | Windowsタスクスケジューラの死活監視                         | 1分以内に自動再起動。再起動時に `trades` テーブルの未決済ポジションと実際のMT5ポジションを照合して不整合を検知           | 再起動後に自動復旧。不整合検知時はDiscord通知   |

#### D.3 LightGBM / 推論障害

| シナリオ                            | 検知方法                     | フォールバック動作                                                                                    | 復旧条件                                       |
| ----------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| モデルファイル読み込みエラー        | `joblib.load()` 例外       | 当該ペアの新規エントリーを全停止。Discord緊急通知。既存ポジション管理（SL/TP・スケールアウト）は継続  | 手動でモデルファイルを復元後に再起動           |
| 推論中の例外（NaN・次元不一致）     | 推論関数内のtry-catch        | シグナルを破棄（エントリー禁止）。Discord通知（エラー内容付き）。次の15分足で特徴量を再計算して再試行 | 次回推論成功で自動復旧                         |
| ドリフト検知（勝率<40% or PSI>0.2） | 毎分の監視タスク             | 緊急再学習をトリガー。再学習中も旧モデルで推論継続（安全側）                                          | 再学習完了後に新モデルへ切り替え + Discord通知 |
| 週次再学習失敗                      | 再学習スクリプトの終了コード | 旧モデルで稼働継続。Discord通知。翌週日曜23:00に再試行                                                | 翌週再学習成功で自動復旧                       |

#### D.4 Webhook / ネットワーク障害

| シナリオ                     | 検知方法                                | フォールバック動作                                                                     | 復旧条件                         |
| ---------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------- |
| Webhook署名検証失敗          | HMAC-SHA256の不一致                     | 429を返してリクエスト破棄。5分以内に3回以上でDiscord通知                               | 自動（TradingViewが再送）        |
| FastAPIプロセスクラッシュ    | Windowsタスクスケジューラの死活監視     | 1分以内に自動再起動                                                                    | 再起動後に自動復旧               |
| ニュースフィードAPI取得失敗  | `requests.get()` 例外 or タイムアウト | 差分検知のSTEP 1をスキップし、STEP 3（ATR急変確認）から開始。Discord通知（警告レベル） | 次回差分検知タスクで自動リトライ |
| TradingViewからのWebhook未着 | 検知不可（設計上の限界）                | 対策なし。デモ稼働中に発生頻度を確認し、必要であればポーリング補完をv3.0以降で検討     | —                               |

#### D.5 DB / ストレージ障害

| シナリオ                | 検知方法                                        | フォールバック動作                                                                                           | 復旧条件                                                |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| SQLiteロック競合        | `sqlite3.OperationalError`                    | 500ms待機後にリトライ×3回。失敗の場合はログをCSVファイルに書き出し継続                                      | 次回アクセスで自動復旧。CSVはDB復旧後にバルクインサート |
| DBファイル破損          | `PRAGMA integrity_check`（毎日01:00実行）失敗 | バックアップから復元（直近7日分保持）。Discord緊急通知                                                       | 復元成功後に手動確認                                    |
| config.json書き込み失敗 | `open()` 例外                                 | 最適化結果をDBの `optimization_history` に記録。config.jsonの更新を諦めて前週パラメータで継続。Discord通知 | 手動でconfig.jsonを修正後、次週最適化で適用             |
| ディスク容量不足        | ログローテーション時の `OSError`              | 古いログ（>7日）を強制削除。Discord緊急通知                                                                  | 容量確保後に自動復旧                                    |

#### D.6 カレンダー / 経済指標障害

| シナリオ                   | 検知方法                       | フォールバック動作                                                          | 復旧条件                             |
| -------------------------- | ------------------------------ | --------------------------------------------------------------------------- | ------------------------------------ |
| カレンダーAPI取得失敗      | API呼び出し例外                | 直前のキャッシュを使用（最大24時間）。Discord通知（警告レベル）             | 次回更新（毎日01:00）で自動復旧      |
| キャッシュ切れ（24時間超） | キャッシュのタイムスタンプ確認 | **全ペアの新規エントリーを停止**（最も安全側の動作）。Discord緊急通知 | カレンダーデータ取得成功後に自動解除 |

---

### F. カレンダーAPI選定

**採用: Forex Factory XML Feed（無料・登録不要）**

| 候補                        | 料金           | 特徴                                           | 採用可否       |
| --------------------------- | -------------- | ---------------------------------------------- | -------------- |
| **Forex Factory XML** | **無料** | 登録不要・週次XMLで全通貨の指標を一括取得      | **採用** |
| Investing.com API           | 無料枠あり     | レートリミット厳しい・スクレイピング規約に注意 | 非推奨         |
| TradingEconomics API        | 有料           | 精度高いが月額$100〜                           | v3.0検討       |
| Alpha Vantage               | 無料枠あり     | FX特化の経済指標カバレッジが弱い               | 不採用         |

**Forex Factory XML Feed の利用方法:**

```python
# calendar_client.py
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import hashlib

FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
HIGH_IMPACT_CURRENCIES = {"USD", "JPY", "EUR", "GBP"}
HIGH_IMPACT_LABEL = "High"

def fetch_calendar() -> list[dict]:
    """
    Forex Factory XML から今週の高インパクト指標を取得。
    毎日深夜01:00にキャッシュ更新（daily_maintenance タスク内）。
    """
    resp = requests.get(FOREX_FACTORY_URL, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    events = []
    for event in root.findall("event"):
        currency = event.findtext("country", "")
        impact   = event.findtext("impact", "")
        title    = event.findtext("title", "")
        date_str = event.findtext("date", "")
        time_str = event.findtext("time", "")

        if currency not in HIGH_IMPACT_CURRENCIES:
            continue
        if impact != HIGH_IMPACT_LABEL:
            continue

        # 日時パース（FF形式: "01-14-2026" / "2:30pm"）
        try:
            dt_str = f"{date_str} {time_str}"
            dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
        except ValueError:
            continue  # 時刻未定のイベントはスキップ

        events.append({
            "currency": currency,
            "title": title,
            "datetime_utc": dt,
            "impact": impact,
        })

    return events

def is_veto_active(events: list[dict], pair: str, now: datetime, buffer_minutes: int = 30) -> bool:
    """
    対象ペアの通貨に関わる高インパクト指標の前後 buffer_minutes 分を Veto する。
    USDJPY → USD / JPY 両方をチェック。
    """
    pair_currencies = {
        "USDJPY": {"USD", "JPY"},
        "EURUSD": {"EUR", "USD"},
        "GBPJPY": {"GBP", "JPY"},
    }
    currencies = pair_currencies.get(pair, set())
    buf = timedelta(minutes=buffer_minutes)

    for ev in events:
        if ev["currency"] not in currencies:
            continue
        if (ev["datetime_utc"] - buf) <= now <= (ev["datetime_utc"] + buf):
            return True
    return False
```

**注意事項:**

- Forex Factory XMLは週単位で提供される（月曜〜日曜）。毎週月曜の深夜01:00更新が理想。
- リクエスト過多によるIP制限を避けるため、**1日1回のキャッシュ更新**に留める。
- 翌週分は金曜にプレビュー取得可能（`ff_calendar_nextweek.xml`）。

---

### G. 環境変数・シークレット管理（.env）

**方針: 全APIキー・認証情報は `.env` ファイルで管理し、GitHubには絶対にPushしない。**

#### G.1 .env ファイル構成

```dotenv
# .env （リポジトリには含めない）

# OpenAI
OPENAI_API_KEY=sk-...

# Discord Webhook
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_CRITICAL_URL=https://discord.com/api/webhooks/...  # CRITICALアラート専用ch

# MT5
MT5_LOGIN=123456
MT5_PASSWORD=your_password
MT5_SERVER=XMTrading-MT5

# Forex Factory (認証不要だが将来の有料API切り替えに備えて枠を用意)
# CALENDAR_API_KEY=

# DB
DB_PATH=C:/fx_system/db/trading.db
DB_BACKUP_PATH=C:/fx_system/db/backup/

# システム
LOG_DIR=C:/fx_system/logs/
MODEL_DIR=C:/fx_system/models/
CONFIG_PATH=C:/fx_system/config.json
DEMO_MODE=true
```

#### G.2 .gitignore 設定（必須）

```gitignore
# === シークレット ===
.env
.env.*
!.env.example

# === ローカルDB・ログ ===
*.db
*.db-journal
logs/
*.log

# === モデルファイル ===
models/*.pkl
models/*.txt

# === Python ===
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# === Windows ===
Thumbs.db
desktop.ini
```

#### G.3 .env.example（GitHubにはこちらをコミット）

```dotenv
# .env.example — 実際の値は入れない。コピーして .env を作成する。

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
DISCORD_WEBHOOK_CRITICAL_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
MT5_LOGIN=000000
MT5_PASSWORD=your_password_here
MT5_SERVER=XMTrading-MT5
DB_PATH=C:/fx_system/db/trading.db
DB_BACKUP_PATH=C:/fx_system/db/backup/
LOG_DIR=C:/fx_system/logs/
MODEL_DIR=C:/fx_system/models/
CONFIG_PATH=C:/fx_system/config.json
DEMO_MODE=true
```

#### G.4 Python での読み込み

```python
# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: SecretStr

    # Discord
    discord_webhook_url: str
    discord_webhook_critical_url: str

    # MT5
    mt5_login: int
    mt5_password: SecretStr
    mt5_server: str

    # DB
    db_path: str
    db_backup_path: str

    # System
    log_dir: str
    model_dir: str
    config_path: str
    demo_mode: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 使用例
settings = Settings()
client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
```

> **pydantic-settings** を使うことで、`.env` の読み込み・型チェック・SecretStr によるログ漏洩防止を一括して処理できる。`pip install pydantic-settings` で導入。

#### G.5 セキュリティチェックリスト

- [ ] `.gitignore` に `.env` が含まれていることを確認してからリポジトリ作成
- [ ] `git log --all --full-history -- .env` で過去のコミットに含まれていないか確認
- [ ] 万が一Pushした場合は即座にAPIキーをローテーション（GitHubのシークレットスキャンでも検知される）
- [ ] Windows VPSのファイルパーミッションで `.env` を Owner読み取り専用に設定
- [ ] Discord Webhook URLはチャンネル単位で発行（漏洩した場合に単独で無効化可能）

---

### H. リスク管理設計

#### H.1 リスクパラメータ定義（確定値）

```python
# risk_manager.py
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskConfig:
    MAX_RISK_PER_TRADE_PCT: float = 2.0   # 1トレードで許容する最大損失（口座残高比）
    MAX_DAILY_DRAWDOWN_PCT: float = 10.0  # 1日の最大許容損失（口座残高比）
    MAX_TOTAL_EXPOSURE_PCT: float = 10.0  # 全オープンポジションの証拠金合計上限（口座残高比）
    MAX_POSITIONS: int = 5                # 同時保有ポジション数の上限（全ペア合計）
    MAX_USD_EXPOSURE: int = 4             # USDを含むポジション数の上限（USDJPY + EURUSD）
    MAX_JPY_EXPOSURE: int = 2             # JPYを含むポジション数の上限（USDJPY + GBPJPY）
```

#### H.2 ポジションサイジング計算

```python
def calc_lot_size(
    account_balance: float,
    sl_pips: float,
    pair: str,
    risk_config: RiskConfig,
) -> float:
    """
    1トレードの許容損失額からロットサイズを算出。
    XMTrading KIWA極口座（commission=0）前提。

    例: 残高100万円 / SL=20pips / USDJPY
      → 許容損失 = 1,000,000 × 0.02 = 20,000円
            → 1pip = 0.01lot あたり 100円 (USDJPY mini lot)
            → lot = 20,000 / (20 × 100 × 100) = 0.10 lot
    """
    pip_values = {
        "USDJPY": 100,   # 1pip / 0.01lot = 100円
        "EURUSD": 110,   # 1pip / 0.01lot ≈ 110円（レートにより変動）
        "GBPJPY": 100,   # 1pip / 0.01lot = 100円
    }
    max_loss_jpy = account_balance * (risk_config.MAX_RISK_PER_TRADE_PCT / 100)
    pip_value_per_mini_lot = pip_values.get(pair, 100)
    lot = max_loss_jpy / (sl_pips * pip_value_per_mini_lot * 100)
    return round(max(0.01, min(lot, 10.0)), 2)  # 0.01〜10.0lotでキャップ
```

#### H.3 通貨エクスポージャーチェック

```python
def check_exposure(
    open_positions: list[dict],
    new_pair: str,
    risk_config: RiskConfig,
) -> tuple[bool, str]:
    """
    新規エントリー前に通貨エクスポージャーをチェック。
    True = エントリー許可 / False = ブロック（理由付き）
    """
    # 全ポジション数チェック
    if len(open_positions) >= risk_config.MAX_POSITIONS:
        return False, f"MAX_POSITIONS({risk_config.MAX_POSITIONS})に達しています"

    # USD エクスポージャーチェック
    usd_pairs = {"USDJPY", "EURUSD"}
    usd_count = sum(1 for p in open_positions if p["pair"] in usd_pairs)
    if new_pair in usd_pairs and usd_count >= risk_config.MAX_USD_EXPOSURE:
        return False, f"MAX_USD_EXPOSURE({risk_config.MAX_USD_EXPOSURE})に達しています"

    # JPY エクスポージャーチェック
    jpy_pairs = {"USDJPY", "GBPJPY"}
    jpy_count = sum(1 for p in open_positions if p["pair"] in jpy_pairs)
    if new_pair in jpy_pairs and jpy_count >= risk_config.MAX_JPY_EXPOSURE:
        return False, f"MAX_JPY_EXPOSURE({risk_config.MAX_JPY_EXPOSURE})に達しています"

    # 証拠金合計チェック
    total_exposure = sum(p.get("margin_used", 0) for p in open_positions)
    account_balance = get_account_balance()
    if total_exposure / account_balance * 100 >= risk_config.MAX_TOTAL_EXPOSURE_PCT:
        return False, f"MAX_TOTAL_EXPOSURE({risk_config.MAX_TOTAL_EXPOSURE_PCT}%)に達しています"

    return True, "OK"
```

#### H.4 日次ドローダウン監視

```python
def check_daily_drawdown(
    daily_pnl_jpy: float,
    account_balance: float,
    risk_config: RiskConfig,
) -> bool:
    """
    当日の損失が MAX_DAILY_DRAWDOWN_PCT を超えた場合、全新規エントリーをブロック。
    既存ポジションのSL/TPは維持。深夜00:00 JSTにリセット。
    """
    drawdown_pct = abs(min(daily_pnl_jpy, 0)) / account_balance * 100
    if drawdown_pct >= risk_config.MAX_DAILY_DRAWDOWN_PCT:
        discord.send_alert(
            f"ALERT: 日次ドローダウン {drawdown_pct:.1f}% に達しました。"
            f"本日の新規エントリーを停止します。"
        )
        return False  # エントリー禁止
    return True
```

#### H.5 リスクパラメータの意図と根拠

| パラメータ             | 値    | 根拠                                                         |
| ---------------------- | ----- | ------------------------------------------------------------ |
| MAX_RISK_PER_TRADE_PCT | 2.0%  | 標準的なトレードリスク管理の推奨値（1〜2%）の上限            |
| MAX_DAILY_DRAWDOWN_PCT | 10.0% | 10%超は心理的・資金的にリカバリーが困難になる経験則          |
| MAX_TOTAL_EXPOSURE_PCT | 10.0% | 証拠金維持率の安全マージン確保。XMのストップアウトレベル対策 |
| MAX_POSITIONS          | 5     | 3ペア × 複数方向の上限。監視コストとリスク分散のバランス    |
| MAX_USD_EXPOSURE       | 4     | USDJPY + EURUSD の最大同時保有。USD相関リスクを制限          |
| MAX_JPY_EXPOSURE       | 2     | USDJPY + GBPJPY の最大同時保有。JPY相関リスクを制限          |

---

### E. 免責事項

> **⚠ 重要**
>
> 本設計書は教育・研究・議論目的で作成されたものです。
> FX取引には元本損失のリスクがあります。本設計書を参考に構築したシステムによる損失について設計者は責任を負いません。
> 実際の資金投入前に必ずデモ口座で最低3ヶ月以上の検証を行い、許容範囲内の少額から開始してください。
