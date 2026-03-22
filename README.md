# FX自動売買システム v2.3

**対象通貨:** USD/JPY · EUR/USD · GBP/JPY  
**実行環境:** Windows VPS（常時稼働）  
**ブローカー:** XMTrading KIWA極口座  

## ディレクトリ構成

```
fx_system/
├── main.py                 # メインオーケストレーター（エントリーポイント）
├── config.json             # 動的パラメータ（週次タスクで一部自動更新）
├── .env.example            # 環境変数テンプレート
├── requirements.txt        # Python依存パッケージ
│
├── config/                 # 設定管理
│   └── settings.py         # pydantic-settings による .env + config.json 管理
│
├── core/                   # コアモジュール
│   ├── time_manager.py     # 時刻管理（保存・比較・表示基準の分離）
│   ├── database.py         # SQLite DB操作（スキーマ・CRUD）
│   ├── logger.py           # loguru ロガー設定
│   ├── notifier.py         # Discord Webhook 通知
│   ├── risk_manager.py     # リスク管理（ロットサイズ・エクスポージャー・DD）
│   └── position_manager.py # ポジション管理・エグジット戦略・ドテン
│
├── broker/                 # ブローカー連携
│   └── mt5_broker.py       # MT5 Python + ThreadPoolExecutor(max_workers=1)
│
├── veto/                   # Veto（取引ブロック）
│   └── calendar_veto.py    # Layer A: 経済指標カレンダー（Forex Factory XML）
│
├── llm/                    # LLM連携
│   └── llm_client.py       # GPT-5.2 差分検知 + Layer B Veto
│
├── ml/                     # 機械学習
│   ├── lgbm_model.py       # 35特徴量・推論（通貨ペア独立モデル）
│   ├── retraining.py       # 実データのラベル付け + 週次自動再学習
│   └── trainer.py          # 学習・ウォークフォワード・PSIドリフト検知
│
├── webhook/                # Webhook受信
│   ├── server.py           # FastAPI + JSON共有トークン照合
│   └── signal_queue.py     # 非同期シグナルキュー
│
├── optimizer/              # 自動最適化
│   └── weekend_optimizer.py # 動的エグジット移行中のため自動最適化は安全停止
│
├── maintenance/            # メンテナンス
│   └── scheduler.py        # 日次・週次・月次メンテナンスタスク
│
└── pinescript/             # TradingView Pine Script（コピペ用）
    └── mtf_smc_v2_3.pine   # MTF SMC 3層構造（4H+1H+15M）
```

## 時刻基準の設計

本システムでは時刻を **3つの基準** に分離して管理します。

| 基準 | タイムゾーン | 用途 |
|------|------------|------|
| **保存基準** | UTC | DB保存・ログ記録・API通信のタイムスタンプ |
| **比較基準** | UTC | 時間差計算・キャッシュ有効期限・Veto判定・ドテンインターバル |
| **表示基準** | JST (UTC+9) | Discord通知・ダッシュボード・ユーザー向け表示 |
| **ブローカー基準** | EET (UTC+2/+3) | MT5サーバー時刻（XMTrading）。入力時にUTCへ変換してから利用 |

### ブローカー時刻の注意点

XMTrading MT5サーバーは EET（東欧時間）を使用:
- **冬時間:** UTC+2（11月第1日曜 〜 3月最終日曜）
- **夏時間:** UTC+3（3月最終日曜 〜 11月第1日曜）

MT5から取得した時刻は必ず `mt5_server_to_utc()` でUTCに変換してから保存・比較します。  
MT5に送信する時刻は `utc_to_mt5_server()` でEETに変換します。

### スケジュールの時刻対応表

| タスク | JST | UTC | 備考 |
|--------|-----|-----|------|
| 日次メンテナンス | 01:00 | 16:00 (前日) | ログ・DB・カレンダー更新 |
| 週末最適化 | 土曜 14:00 | 土曜 05:00 | 動的エグジット移行中のため自動適用は停止 |
| 週次メンテナンス | 日曜 23:00 | 日曜 14:00 | 実データ自動ラベル付け + LightGBM自動再学習 + 週次レポート |
| 取引除外時間帯 | 00:00〜07:00 | 15:00〜22:00 (前日) | 深夜帯エントリー禁止 |
| 金曜クローズ | 金曜 22:00 | 金曜 13:00 | 全ポジション強制決済 |

## 本番環境セットアップ手順

### 前提条件

| 項目 | 要件 |
|------|------|
| OS | Windows Server 2019/2022 または Windows 10/11（常時起動 VPS 推奨） |
| Python | 3.11 以上（3.13 以下推奨） |
| MT5 | XMTrading の MetaTrader 5 ターミナルをインストール済み |
| ポート | 外部 → 8000（または任意）のインバウンドを開放 |
| OpenAI | API キー取得済み（GPT-5.2 アクセス権が必要） |
| Discord | Webhook URL 取得済み（通常通知用 + Critical 用の2本推奨） |

---

### ステップ 1: ソースコード配置

```bat
cd C:\
git clone <YOUR_REPO_URL> fx_system
cd fx_system
```

---

### ステップ 2: Python 仮想環境の作成と依存インストール

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> optuna を使う場合（ハイパーパラメータ再チューニング時のみ）:
> ```bat
> pip install optuna
> ```

---

### ステップ 3: 環境変数設定（.env）

```bat
copy .env.example .env
notepad .env
```

`.env` に以下をすべて設定します:

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx          # OpenAI API キー
DISCORD_WEBHOOK_URL=https://discord.com/...  # 通常通知用 Webhook
DISCORD_WEBHOOK_CRITICAL_URL=https://...     # Critical 通知用 Webhook
MT5_LOGIN=123456                             # MT5 口座番号
MT5_PASSWORD=your_password                  # MT5 パスワード
MT5_SERVER=XMTrading-MT5                    # MT5 サーバー名
DB_PATH=./db/trading.db
DB_BACKUP_PATH=./db/backup/
LOG_DIR=./logs/
MODEL_DIR=./models/
CONFIG_PATH=./config.json
DEMO_MODE=false                              # 本番時は false（true=発注なし）
WEBHOOK_SECRET=your_shared_webhook_token    # TradingView と共有するトークン
WEBHOOK_PORT=8000
```

> **DEMO_MODE について**  
> `true` にすると MT5 への発注は行われず、ログとシグナルへの DB 記録のみ行います。  
> 初回は `true` で動作確認してから `false` に切り替えることを推奨します。

---

### ステップ 4: ディレクトリ初期化

```bat
mkdir db db\backup logs models data
```

---

### ステップ 5: 初期モデルの作成（TradingView CSV ブートストラップ）

モデルなしでは起動時に学習済モデル未ロードの状態になります。  
必ず以下のどちらかの方法でモデルを用意してください。

#### 方式 A（推奨）: TradingView CSV から学習

1. `pinescript/mtf_smc_v2_3.pine` を TradingView に追加
2. `i_csv_mode` を ON（CSVエクスポートモード）にする
3. 各ペアの **15分足チャート** で `Export chart data` → CSV 保存  
   （ファイル名: `USDJPY_chart.csv`, `EURUSD_chart.csv`, `GBPJPY_chart.csv`）
4. CSV を `fx_system/data/` に配置
5. 以下で3ペア一括ビルド＋学習:

```bat
.venv\Scripts\activate
python -m maintenance.run_bootstrap_batch --input-dir data --model-dir models
```

> `config.json` の `ml.label_horizon_minutes_per_pair` に従い EURUSD は 300 分ホライズンで自動ビルドします。

WFV 検証をスキップして高速に学習したい場合:

```bat
python -m maintenance.run_bootstrap_batch --input-dir data --model-dir models --skip-wfv
```

#### 方式 B（疎通テスト最速）: ダミーモデル生成

```bat
python bootstrap_models.py
```

> 実運用には不向きです。動作確認後、方式 A に切り替えてください。

**学習完了後の確認:**

```bat
dir models\lgbm_*.pkl
dir models\lgbm_*_metrics.json
```

3ペア分（`lgbm_USDJPY.pkl`, `lgbm_EURUSD.pkl`, `lgbm_GBPJPY.pkl`）が存在することを確認します。

---

### ステップ 6: MT5 ターミナルの設定

1. MT5 を起動し、対象口座にログイン
2. **ツール → オプション → Expert Advisors** で以下を有効化:
   - 「自動売買を許可する」にチェック
   - 「DLL のインポートを許可する」にチェック
3. MT5 を**最小化で常時起動**した状態にしておく（終了不可）

---

### ステップ 7: TradingView アラート設定

1. TradingView で 15 分足チャートを開く
2. Pine Editor に `pinescript/mtf_smc_v2_3.pine` を貼り付けて「チャートに追加」
3. `i_webhook_token` = `.env` の `WEBHOOK_SECRET` と同じ値を設定
4. `i_alert_mode` = **`ml_first`（推奨）**
5. 通貨ペアごとに `i_pair` を変更して **3 つのアラート** を作成:

| ペア | `i_pair` | Webhook URL |
|------|--------|----|
| USDJPY | `usdjpy` | `http://YOUR_VPS_IP:8000/webhook` |
| EURUSD | `eurusd` | `http://YOUR_VPS_IP:8000/webhook` |
| GBPJPY | `gbpjpy` | `http://YOUR_VPS_IP:8000/webhook` |

アラートの設定:
- 条件: **`Any alert() function call`**
- 通知先: **Webhook URL のみ**（他の通知は不要）

> VPS に SSL 証明書（HTTPS）を設定している場合は `https://` を使用してください。  
> TradingView の Webhook は HTTP も受け付けますが、本番では HTTPS 推奨です。

---

### ステップ 8: システム起動

```bat
cd C:\fx_system
.venv\Scripts\activate
python main.py
```

起動後に以下のログが出れば正常です:

```
INFO  | main.py | MT5 connected: XMTrading-MT5
INFO  | main.py | LightGBM models loaded: USDJPY, EURUSD, GBPJPY
INFO  | main.py | Webhook server started on port 8000
INFO  | main.py | Scheduler started
```

Discord に「システム起動」通知が届くことを確認します。

---

### ステップ 9: 動作確認（DEMO_MODE=true で疎通テスト）

```bat
# 別ターミナルでヘルスチェック
curl http://localhost:8000/health

# Webhook 疎通テスト（手動でシグナルを送信）
python test_webhook_post.py
```

`DEMO_MODE=true` の状態で TradingView からシグナルが届き、以下を確認します:

- `signals` テーブルにレコードが追加される
- `training_samples` テーブルにレコードが追加される
- Discord に通知が届く
- LightGBM gate 通過ログが出る

確認が取れたら `.env` の `DEMO_MODE=false` に変更して再起動します。

---

### ステップ 10: 常時起動設定（Windows タスクスケジューラ）

VPS 再起動時に自動起動するよう設定します。

**タスクの作成:**

1. 「タスクスケジューラ」を管理者で開く
2. 「タスクの作成」 → 全般タブ:
   - 名前: `FX AutoTrading System`
   - 「ユーザーがログオンしているかどうかにかかわらず実行する」を選択
   - 「最上位の特権で実行する」にチェック
3. トリガータブ → 「コンピュータのスタートアップ時」
4. 操作タブ:
   - プログラム: `C:\fx_system\.venv\Scripts\python.exe`
   - 引数: `main.py`
   - 開始（フォルダ）: `C:\fx_system`
5. 設定タブ:
   - 「タスクが失敗した場合の再起動の間隔」: 1 分、最大 3 回

または PowerShell でバッチ起動スクリプトを作成する方法もあります:

```bat
REM start_trading.bat
cd /d C:\fx_system
.venv\Scripts\python.exe main.py >> logs\startup.log 2>&1
```

---

### ステップ 11: 起動後の継続監視

**training_samples の蓄積確認（毎日）:**

```bat
python -m maintenance.check_training_samples
python -m maintenance.check_training_samples --hours 24
```

確認ポイント:
- `summary.total` が日々増加している
- `per_pair.latest_signal_time` が更新されている
- 日曜 23:00 JST 以降に `unlabeled` が減り `labeled` が増える

**モデル精度確認（週次）:**

```bat
type models\lgbm_USDJPY_metrics.json
type models\lgbm_EURUSD_metrics.json
type models\lgbm_GBPJPY_metrics.json
```

`balanced_accuracy` が `majority_baseline` を上回っているかを確認します。

---

## 最新実装メモ

- LightGBM入力は 35特徴量です。
- Pineの `i_alert_mode` は `ml_first` を推奨（Pineはトリガー/特徴量送信、最終判断はLightGBM）。
- エグジット戦略は固定ATRスケールアウトではなく、`time_decay` / `structural_tp` / trailing の優先順位で動作します。
- Pine Script は `pinescript/mtf_smc_v2_3.pine` を使用し、`ob_4h_distance_pips` を JSON 送信して構造TPに利用します。
- Pine Webhook は `atr` に加えて `atr_20d_avg` も送信し、LLM差分検知の ATR 急変判定で実値を使います。
- `weekend_optimizer.py` は旧 scale-out 前提のため、グリッド探索の自動適用は安全停止しています（実行時は DB に記録のみ）。
- 代わりに週次メンテ後に execution layer 向けの軽量チューニング（`exit_prob_stale_minutes` / `trailing_update_cooldown_seconds` / `trailing_min_step_pips`）を自動調整します。
- 定期ニュース差分判定の LLM はペア別キャッシュで管理され、`config.json` の `llm.model_diff` で軽量モデルに切り替えできます。
- ハイブリッドLLM: 通常は `llm.model_diff`（既定: `gpt-5-nano`）で低コスト判定し、`news_importance_escalation_threshold` を超える重要ニュース時のみ `llm.model_instant`（既定: `gpt-5.2`）へ昇格して精査します。
- 特徴量反映は `feature_news_importance_threshold` 以上のニュース時のみ `sentiment_score` を有効化し、重要度が低い場合はノイズ混入を避けるため 0 として扱います。
- 週次再学習とブートストラップ学習の両方で `models/lgbm_<PAIR>_metrics.json` を保存し、再起動時はこの精度メタデータを優先して読み込みます。
- TradingView CSV から作る初期モデルは疎通・初期運用向けです。本命は `training_samples` 蓄積後の週次自動再学習です。

### アラートモード比較（ml_first vs strict）

```bash
python maintenance/compare_alert_modes.py --days 7
python maintenance/compare_alert_modes.py --days 14 --pair USDJPY
```

`signals` と `training_samples` から `alert_mode` 別の件数・quality gate通過率・方向一致率を簡易集計します。

初回起動時は学習済みモデルがないため、以下のどちらかで開始します。

- 方式A（推奨・即運用）: 疎通用モデルを作成して起動し、実データ蓄積後に週次自動再学習へ移行
- 方式B: 事前に十分な履歴がある場合のみ、実データ再学習を先に実行

### 疎通テスト用（すぐ動かす）

```bash
python bootstrap_models.py
```

### 自動再学習フロー（実装済み）

以下はシステムが自動で実行します（週次メンテ時）。
1. `training_samples` に保存された未ラベル特徴量を取得
2. MT5履歴（15分足）から `+240分` の将来価格を取得
3. `up/flat/down` ラベルを自動付与
4. 直近90日のラベル済みデータで通貨ペア別に再学習
5. `models/lgbm_<PAIR>.pkl` を更新し、週次タスクで再ロード

### 初回運用の具体手順（最短）

```bash
# 0) 初回のみ: 疎通用モデル作成
python bootstrap_models.py

# 1) システム起動
python main.py

# 2) TradingViewから通常運用（Webhook送信）
#   → training_samples に特徴量が蓄積される

# 3) 日曜23:00 JST を跨ぐと自動ラベル付け + 自動再学習
```

### 自動再学習が動く条件

- `models/` に既存モデルがある（初回は `bootstrap_models.py` 推奨）
- MT5接続が正常
- ペアごとに最小学習件数を満たす（既定: 300件）

### TradingView CSV から初期学習データを作る（ブートストラップ）

1. `pinescript/mtf_smc_v2_3.pine` の `CSVエクスポートモード [NEW]` を ON
2. TradingView の `Export chart data` で CSV を保存
3. 以下で学習用CSVを生成

```bash
python maintenance/build_bootstrap_dataset.py \
    --input data/USDJPY_chart.csv \
    --output data/USDJPY_bootstrap_train.csv \
    --pair USDJPY \
    --horizon-bars 16
```

補足:
- `horizon-bars=16` は 15分足で約4時間先
- Pine単独で取得できない `open_positions_count / max_dd_24h / calendar_risk_score / sentiment_score` は 0 固定で出力
- 生成CSVは 35特徴量 + `label` を含み、初期モデル作成のブートストラップ用途を想定

続けて初期モデルを学習:

```bash
python maintenance/train_bootstrap_model.py \
    --input data/USDJPY_bootstrap_train.csv \
    --pair USDJPY \
    --model-dir models
```

必要に応じて `--skip-wfv` を付けるとウォークフォワード検証を省略できます。

3ペア一括で実行する場合:

```bash
python maintenance/run_bootstrap_batch.py \
    --input-dir data \
    --output-dir data \
    --model-dir models
```

既定では `data/USDJPY_chart.csv`, `data/EURUSD_chart.csv`, `data/GBPJPY_chart.csv` を読み込みます。

### training_samples 保持ポリシー

- `label IS NOT NULL` の学習済みサンプルは **180日保持**
- 180日超の学習済みサンプルは週次メンテで自動削除

### training_samples の蓄積確認

本番投入後は、まず `training_samples` が想定どおり増えているかを確認します。

```bash
python maintenance/check_training_samples.py
python maintenance/check_training_samples.py --hours 24 --limit 5
python maintenance/check_training_samples.py --pair USDJPY --json
```

見るべき点:

- `summary.total` が増加していること
- `recent_last_<hours>h` に直近流入件数が出ていること
- `per_pair` の `latest_signal_time` が更新されていること
- 日曜23:00 JST 以降は `unlabeled` が減り、`labeled` が増え始めること

### モデル精度メタデータ

- 保存先: `models/lgbm_USDJPY_metrics.json`, `models/lgbm_EURUSD_metrics.json`, `models/lgbm_GBPJPY_metrics.json`
- 主な項目: `accuracy`, `balanced_accuracy`, `majority_baseline`, `updated_at_utc`
- 起動時はこのファイルを読んでペア別 `model_accuracy` の初期値に使います
- `balanced_accuracy < majority_baseline` の状態は、まだモデル優位性が弱い目安です

## PineScript の使い方

1. `pinescript/mtf_smc_v2_3.pine` の内容をコピー
2. TradingView で **15分足チャート** を開く
3. Pine Editor に貼り付けて「チャートに追加」
4. アラート設定: 条件 = "Any alert() function call"
5. Webhook URL = `https://YOUR_VPS_IP/webhook`（外部80番）
    - アプリ直接待受ポートは `.env` の `WEBHOOK_PORT`（既定: 8000）
6. `i_webhook_token` に `.env` の `WEBHOOK_SECRET` と同じ値を設定
7. `i_alert_mode` は **`ml_first`（推奨）** を選択（Pine側の過剰ゲートを避け、LightGBM中心で判定）
8. 通貨ペアごとに `i_pair` パラメータを変更して3つのアラートを設定

## プレローンチチェックリスト

明日の常時稼働前に、最低限ここまでは確認してください。

### 1. 設定と接続

- `.env` に OpenAI / Discord / MT5 / Webhook の必須値が入っている
- `config.json` が正しい JSON で、`system.pairs` と `ml` セクションが期待どおり
- MT5 ターミナルに対象口座でログイン済み

### 2. モデルとファイル

- `models/lgbm_<PAIR>.pkl` が 3ペア分存在する
- `models/lgbm_<PAIR>_metrics.json` が 3ペア分存在する
- 疎通用モデルしかない段階であることを理解し、過信しない

### 3. Webhook 動線

- TradingView 側の Webhook URL が VPS に到達する
- Pine の `i_webhook_token` と `.env` の `WEBHOOK_SECRET` が一致する
- `i_alert_mode=ml_first` で3ペア分のアラートが有効

### 4. 起動直後の監視

- `python main.py` 起動後に例外で即落ちしない
- Discord 通知が正常に届く
- 初回シグナル受信後、`signals` と `training_samples` が増える
- `python maintenance/check_training_samples.py --hours 1` で増分確認できる

### 5. 週次再学習の前提

- 日曜23:00 JST までに各ペアで十分な `training_samples` を蓄積できる見込みがある
- MT5 から M15 履歴を取れる
- 初回数週は `balanced_accuracy` と `majority_baseline` を毎週比較する

### 6. リスク面の前提

- いきなり実弾フルサイズで始めない
- 明日開始時点では「モデル改善フェーズ」であり、「完成モデル運用」ではないと認識する
- 異常時は Webhook 停止か MT5 側で手動停止できる体制を用意する

