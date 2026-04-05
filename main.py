"""
FX自動売買システム v2.2 — メインオーケストレーター

■ クリティカルパス（目標500ms以内）
  ① TradingView Webhook受信（MTF SMC + テクニカル条件成立時のみ）
  ② Calendar Veto確認（Python IFルール・0ms）
  ③ GPT-5.2キャッシュ読込（メモリから即時・0ms）
    ④ LightGBM推論（35特徴量・1ms以下）
  ⑤ ドテン判定 → インターバル確認
  ⑥ 高速連続発注 / MT5 ThreadPoolExecutor(max_workers=1)

■ バックグラウンドタスク
  毎15分: 差分検知タスク
  毎1分:  ポジション監視
  スケジュール: 日次・週次・月次メンテナンス

■ 時刻基準
  - 全内部処理: 比較基準（UTC）
  - DB保存: 保存基準（UTC）
  - Discord/ログ表示: 表示基準（JST）
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import get_settings, get_trading_config, reload_trading_config
from core.database import (
    get_connection,
    init_db,
    insert_signal,
    insert_trade,
    close_trade,
    get_daily_pnl,
    insert_training_sample,
    get_open_trade_by_ticket,
)
from core.logger import setup_logger
from core.notifier import DiscordNotifier, AlertLevel
from core.position_manager import ManagedPosition, PositionManager
from core.risk_manager import (
    RiskConfig,
    load_risk_config,
    calc_lot_size,
    calc_sl_tp_pips,
    check_exposure,
    check_daily_drawdown,
)
from core.time_manager import (
    UTC,
    now_utc,
    format_jst,
    is_excluded_hours,
    to_jst,
    is_broker_market_closed,
    broker_day_start_utc,
    get_session_flag,
)
from broker.mt5_broker import MT5Broker
from llm.llm_client import DiffDetector, LLMClient
from ml.lgbm_model import LGBMPredictor, build_features
from ml.trainer import load_model_metrics
from veto.calendar_veto import CalendarVeto
from webhook.server import app as fastapi_app
from webhook.signal_queue import get_queue


class Orchestrator:
    """システム全体の制御・コンポーネント連携。"""

    def __init__(self):
        self._settings = get_settings()
        self._config = get_trading_config()
        self._running = False

        # DB
        self._db_conn = get_connection(self._settings.db_path)
        init_db(self._settings.db_path)

        # 通知
        self._notifier = DiscordNotifier(
            webhook_url=self._settings.discord_webhook_url,
            critical_webhook_url=self._settings.discord_webhook_critical_url,
        )

        # MT5 ブローカー
        self._broker = MT5Broker(
            login=self._settings.mt5_login,
            password=self._settings.mt5_password.get_secret_value(),
            server=self._settings.mt5_server,
        )

        # Veto
        self._calendar_veto = CalendarVeto()

        # LLM
        self._diff_detectors = {
            pair: DiffDetector()
            for pair in self._config["system"]["pairs"]
        }
        self._llm_client = LLMClient(
            api_key=self._settings.openai_api_key.get_secret_value(),
            db_conn=self._db_conn,
        )

        # LightGBM
        self._predictor = LGBMPredictor(model_dir=self._settings.model_dir)

        # ポジション管理
        self._position_manager = PositionManager(self._broker, self._notifier, self._db_conn)

        # リスク設定
        self._risk_config = load_risk_config(self._config)

        # ← [NEW] LLMで使用するATR情報キャッシュ（webhook → diff_detection_task）
        self._last_webhook_atrs: dict[str, tuple[float, float]] = {}  # pair -> (atr_14, atr_20d_avg)

        # MCP EA ポジション管理（起動中のチケットセット）
        self._mcp_position_tickets: set[int] = set()

        # MCP シグナル重複防止: {pair: last_processed_timestamp}
        self._mcp_signal_cooldown: dict[str, float] = {}
        _MCP_SIGNAL_COOLDOWN_SEC = 30  # 同一ペアの連続シグナルを30秒間ブロック

        # スケジューラ
        self._scheduler = AsyncIOScheduler(timezone=UTC)

        # LLMエラー通知のスロットリング
        self._last_llm_model_error_notify = None

    async def start(self) -> None:
        """システムを起動する。"""
        logger.info(f"=== FX Auto-Trading System v{self._config['system']['version']} ===")
        logger.info(f"Pairs: {self._config['system']['pairs']}")
        logger.info(f"Start time: {format_jst(now_utc())}")

        self._running = True

        # MT5 接続
        if not self._broker.connect():
            self._running = False
            message = (
                "MT5接続失敗のためシステムを停止します。\n"
                f"Server: {self._settings.mt5_server} / Login: {self._settings.mt5_login}\n"
                "原因はログのMT5 initialize/login errorを確認してください。"
            )
            await self._notifier.send_critical(message)
            logger.critical("MT5 connection failed. System will stop for fail-safe.")
            raise RuntimeError("MT5 connection failed")

        # カレンダー取得
        try:
            await self._calendar_veto.fetch_events_async()
        except Exception as e:
            logger.warning(f"Initial calendar fetch failed: {e}")

        # LightGBM モデル読み込み
        model_results = self._predictor.load_all_models()
        for pair, ok in model_results.items():
            if not ok:
                logger.warning(f"Model not loaded for {pair} — predictions unavailable")

        # WFV精度をモデルに設定（metricsファイル優先、なければ既定値）
        _default_accuracies = {
            "USDJPY": 0.3785,
            "EURUSD": 0.4501,
            "GBPJPY": 0.4147,
        }
        for pair, acc in _default_accuracies.items():
            metrics = load_model_metrics(pair, self._settings.model_dir)
            if metrics:
                selected_acc = float(metrics.get("balanced_accuracy", metrics.get("accuracy", acc)))
            else:
                selected_acc = acc
            self._predictor.set_model_accuracy(pair, selected_acc)

        # 再起動中に broker 側で消えた建玉を、可能な範囲でDBへクローズ反映する。
        self._reconcile_stale_db_trades_with_broker()

        # MT5 側に残っている建玉をローカル管理へ復元
        self._restore_managed_positions_from_broker()

        # スケジューラ設定
        self._setup_scheduler()
        self._scheduler.start()

        # 起動通知
        await self._notifier.send(
            f"システム起動完了\n"
            f"バージョン: v{self._config['system']['version']}\n"
            f"対象ペア: {', '.join(self._config['system']['pairs'])}"
        )

        # メインループ: Webhook シグナル処理
        await self._signal_processing_loop()

    async def stop(self) -> None:
        """システムを停止する。"""
        self._running = False
        if getattr(self._scheduler, "running", False):
            self._scheduler.shutdown(wait=False)
        self._broker.disconnect()
        self._db_conn.close()
        logger.info("System stopped")

    def _setup_scheduler(self) -> None:
        """APScheduler のジョブを設定する。"""
        # 毎15分: 差分検知タスク
        self._scheduler.add_job(
            self._diff_detection_task,
            CronTrigger(minute="*/15", timezone=UTC),
            id="diff_detection",
            name="Diff Detection (15min)",
        )

        # 毎1分: ポジション監視
        self._scheduler.add_job(
            self._position_monitor_task,
            CronTrigger(minute="*", timezone=UTC),
            id="position_monitor",
            name="Position Monitor (1min)",
        )

        # 毎日 01:00 JST = 16:00 UTC（前日）
        self._scheduler.add_job(
            self._daily_maintenance_task,
            CronTrigger(hour=16, minute=0, timezone=UTC),  # UTC
            id="daily_maintenance",
            name="Daily Maintenance (01:00 JST)",
        )

        # 毎週土曜 14:00 JST = 05:00 UTC
        self._scheduler.add_job(
            self._weekend_optimization_task,
            CronTrigger(day_of_week="sat", hour=5, minute=0, timezone=UTC),  # UTC
            id="weekend_optimization",
            name="Weekend Optimization (Sat 14:00 JST)",
        )

        # 毎週日曜 23:00 JST = 14:00 UTC
        self._scheduler.add_job(
            self._weekly_maintenance_task,
            CronTrigger(day_of_week="sun", hour=14, minute=0, timezone=UTC),  # UTC
            id="weekly_maintenance",
            name="Weekly Maintenance (Sun 23:00 JST)",
        )

        # 毎日 02:00 JST（= 17:00 UTC 前日）に月次条件をガード判定
        self._scheduler.add_job(
            self._monthly_maintenance_guard_task,
            CronTrigger(hour=17, minute=0, timezone=UTC),  # UTC
            id="monthly_maintenance_guard",
            name="Monthly Maintenance Guard (Daily 02:00 JST)",
        )

        logger.info("Scheduler configured with all maintenance jobs")

    def _restore_managed_positions_from_broker(self) -> None:
        """再起動後に MT5 の保有建玉をローカル管理状態へ復元する。"""
        config = get_trading_config()
        risk = config.get("risk", {})
        trailing_mult = float(risk.get("trailing_atr_multiplier", 0.8))
        if trailing_mult <= 0:
            trailing_mult = 0.8

        restored = 0
        for broker_pos in self._broker.get_positions():
            ticket = int(broker_pos["ticket"])
            if ticket in self._position_manager.positions:
                continue

            trade_row = get_open_trade_by_ticket(self._db_conn, ticket)
            if trade_row is None:
                trade_id = insert_trade(self._db_conn, {
                    "pair": broker_pos["pair"],
                    "direction": broker_pos["direction"],
                    "open_time": broker_pos["open_time_utc"],
                    "open_price": broker_pos["open_price"],
                    "volume": broker_pos["volume"],
                    "sl_price": broker_pos.get("sl_price"),
                    "tp_price": broker_pos.get("tp_price"),
                    "mt5_ticket": ticket,
                })
                trade_row = {
                    "id": trade_id,
                    "open_time": broker_pos["open_time_utc"].isoformat(),
                }

            open_price = float(broker_pos["open_price"])
            sl_price = float(broker_pos.get("sl_price") or 0.0)
            tp_price = float(broker_pos.get("tp_price") or 0.0)
            inferred_atr = abs(open_price - sl_price) / trailing_mult if sl_price > 0 else 0.0

            managed = ManagedPosition(
                trade_id=int(trade_row["id"]),
                ticket=ticket,
                pair=broker_pos["pair"],
                direction=broker_pos["direction"],
                volume=float(broker_pos["volume"]),
                open_price=open_price,
                sl_price=sl_price,
                tp_price=tp_price,
                target_tp_price=tp_price,
                target_tp_pips=0.0,
                open_time_utc=broker_pos["open_time_utc"],
                atr_at_entry=max(inferred_atr, 0.0),
            )
            self._position_manager.register_position(managed)
            restored += 1

        if restored > 0:
            logger.warning(f"Recovered {restored} broker positions into local manager state")

    def _reconcile_stale_db_trades_with_broker(self) -> None:
        """DBでは未決済だが、broker側に存在しない建玉を起動時に整合させる。"""
        broker_positions = self._broker.get_positions()
        broker_open_tickets = {int(pos["ticket"]) for pos in broker_positions}
        rows = self._db_conn.execute(
            "SELECT * FROM trades WHERE close_time IS NULL ORDER BY open_time"
        ).fetchall()

        reconciled = 0
        unresolved = 0

        for row in rows:
            ticket = int(row["mt5_ticket"] or 0)
            if ticket <= 0 or ticket in broker_open_tickets:
                continue

            open_time = row["open_time"]
            opened_at = now_utc()
            if isinstance(open_time, str):
                try:
                    opened_at = datetime.fromisoformat(open_time)
                except ValueError:
                    opened_at = now_utc()
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=UTC)

            lookback_hours = max(72, int((now_utc() - opened_at).total_seconds() // 3600) + 24)
            closed_info = self._broker.get_recent_closed_position_info(
                ticket,
                lookback_hours=lookback_hours,
            )
            if not closed_info or not closed_info.get("close_price"):
                unresolved += 1
                continue

            close_price = float(closed_info["close_price"])
            pnl_pips, pnl_jpy = self._estimate_closed_trade_metrics(dict(row), close_price)
            profit = closed_info.get("profit")
            if profit is not None:
                pnl_jpy = round(float(profit), 0)

            close_trade(
                self._db_conn,
                trade_id=int(row["id"]),
                close_price=close_price,
                pnl_pips=pnl_pips,
                pnl_jpy=pnl_jpy,
                exit_reason=self._infer_exit_reason_from_trade_row(dict(row), close_price),
            )
            reconciled += 1

        if reconciled > 0 or unresolved > 0:
            msg = (
                "Startup DB reconciliation: "
                f"reconciled={reconciled}, unresolved={unresolved}, broker_open={len(broker_open_tickets)}"
            )
            if unresolved > 0:
                logger.warning(msg)
            else:
                logger.info(msg)

    def _infer_exit_reason_from_trade_row(self, trade_row: dict, close_price: float) -> str:
        """DB行だけから外部決済の exit_reason を近似推定する。"""
        pair = str(trade_row.get("pair") or "")
        direction = str(trade_row.get("direction") or "")
        open_price = float(trade_row.get("open_price") or 0.0)
        sl_price = float(trade_row.get("sl_price") or 0.0)
        tp_price = float(trade_row.get("tp_price") or 0.0)
        tolerance = 1e-5 if pair not in ("USDJPY", "GBPJPY") else 1e-3

        if tp_price > 0:
            if direction == "long" and close_price >= tp_price - tolerance:
                return "structural_tp"
            if direction == "short" and close_price <= tp_price + tolerance:
                return "structural_tp"

        if sl_price > 0:
            if direction == "long" and close_price <= sl_price + tolerance:
                return "trailing" if sl_price >= open_price else "atr_sl"
            if direction == "short" and close_price >= sl_price - tolerance:
                return "trailing" if sl_price <= open_price else "atr_sl"

        return "time_exit"

    def _estimate_closed_trade_metrics(self, trade_row: dict, close_price: float) -> tuple[float, float]:
        """DB行から決済時の概算 pips / 円損益を求める。"""
        pair = str(trade_row.get("pair") or "")
        direction = str(trade_row.get("direction") or "")
        open_price = float(trade_row.get("open_price") or 0.0)
        volume = float(trade_row.get("volume") or 0.0)
        pip_unit = 0.01 if pair in ("USDJPY", "GBPJPY") else (0.10 if "XAU" in pair or "XAG" in pair or pair == "GOLD" else 0.0001)

        pnl_price = (close_price - open_price) if direction == "long" else (open_price - close_price)
        pnl_pips = pnl_price / pip_unit
        pip_value_per_mini_lot = 100 if pair in ("USDJPY", "GBPJPY") else 110
        pnl_jpy = pnl_pips * pip_value_per_mini_lot * (volume / 0.01)
        return round(pnl_pips, 1), round(pnl_jpy, 0)

    # ── メインシグナル処理ループ ──────────────────
    async def _signal_processing_loop(self) -> None:
        """Webhook キューからシグナルを受け取って処理するメインループ。"""
        signal_queue = get_queue()
        while self._running:
            try:
                payload = await asyncio.wait_for(signal_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal queue error: {e}")
                continue

            try:
                source = payload.get("signal_source")
                if source == "mcp_context":
                    self._handle_mcp_context(payload)
                elif source in ("mcp", "tv_alert"):
                    await self._process_mcp_signal(payload)
                else:
                    await self._process_signal(payload)
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
                await self._notifier.send_alert(f"シグナル処理エラー: {e}")

    # ── MCP EA シグナル処理 ─────────────────────────────────────────────────────

    def _handle_mcp_context(self, payload: dict) -> None:
        """
        tv_mcp_ea からの Market Context を受信し、
        保有中ポジションのエグジット判断用データを更新する。
        """
        pair = payload.get("pair", "")
        # GOLD → XAUUSD の正規化（MT5 シンボルが GOLD の場合）
        # position_manager 内ではペア名でマッチするため、
        # 登録時のペア名と一致させる必要がある
        self._position_manager.update_market_context(pair, payload)

    def _count_mcp_positions(self, category: str | None = None) -> int:
        """現在アクティブな MCP ポジション数を返す。"""
        active_tickets = set(self._position_manager.positions.keys())
        mcp_active = self._mcp_position_tickets & active_tickets
        if category is None:
            return len(mcp_active)
        gold_pairs = {"XAUUSD", "XAUEUR", "XAGUSDUSD", "GOLD"}
        count = 0
        for t in mcp_active:
            pos = self._position_manager.positions[t]
            is_gold = pos.pair in gold_pairs
            if category == "gold" and is_gold:
                count += 1
            elif category == "fx" and not is_gold:
                count += 1
        return count

    def _calc_mcp_sl_tp_pips(self, pair: str, atr: float) -> tuple[float, float]:
        """MCP 設定の ATR 倍率から SL/TP を pips で返す。"""
        mcp_cfg = self._config.get("mcp_ea", {})
        is_gold = "XAU" in pair or "XAG" in pair or pair == "GOLD"
        if is_gold:
            sl_mult = float(mcp_cfg.get("gold_sl_atr_mult", 2.5))
            tp_mult = float(mcp_cfg.get("gold_tp_atr_mult", 2.5))
            gold_pip_unit = 0.10  # GOLD: 1pip = $0.10
            sl_pips = (atr * sl_mult) / gold_pip_unit
            tp_pips = (atr * tp_mult) / gold_pip_unit
        else:
            sl_mult = float(mcp_cfg.get("fx_sl_atr_mult", 1.5))
            tp_mult = float(mcp_cfg.get("fx_tp_atr_mult", 2.0))
            fx_pip_unit = 0.01 if pair in ("USDJPY", "GBPJPY") else 0.0001
            atr_pips = atr / fx_pip_unit
            sl_pips = atr_pips * sl_mult
            tp_pips = atr_pips * tp_mult
        return sl_pips, tp_pips

    async def _process_mcp_signal(self, payload: dict) -> None:
        """
        TV MCP EA からのブレイクアウトシグナルを処理する。

        ① Calendar Veto 確認
        ② 時間フィルター
        ③ MCP ポジション上限確認（FX max 1 / GOLD max 1）
        ④ FX ペアのみ: LightGBM 方向確認（モデル不在時はスキップ）
        ⑤ ATR ベース SL/TP 計算
        ⑥ ロット計算 → MT5 発注
        """
        pair = payload["pair"]
        direction = payload["direction"]
        atr = float(payload["atr"])
        close_price = float(payload["close"])
        breakout_score = int(payload.get("breakout_score", 0))
        pattern = str(payload.get("pattern", "unknown"))

        # ⓪ 同一ペア重複防止（同時アラート発火対策）
        import time as _time
        now_ts = _time.time()
        last_ts = self._mcp_signal_cooldown.get(pair, 0)
        if now_ts - last_ts < 30:
            logger.info(f"[MCP] Signal cooldown active for {pair} ({now_ts - last_ts:.0f}s ago)")
            return
        self._mcp_signal_cooldown[pair] = now_ts

        mcp_cfg = self._config.get("mcp_ea", {})
        if not bool(mcp_cfg.get("enabled", True)):
            logger.info("MCP EA disabled in config")
            return

        logger.info(
            f"[MCP] Processing signal: {pair} {direction} "
            f"score={breakout_score} pattern={pattern}"
        )

        # ① Calendar Veto
        veto_active, veto_reason = self._calendar_veto.is_veto_active(pair)
        if veto_active:
            logger.info(f"[MCP] Signal vetoed by calendar: {veto_reason}")
            return

        # ② 時間フィルター
        if is_excluded_hours():
            logger.info("[MCP] Signal rejected: excluded broker hours")
            return

        # ③ MCP ポジション上限確認
        is_gold = "XAU" in pair or "XAG" in pair or pair == "GOLD"
        category = "gold" if is_gold else "fx"
        max_cat = int(mcp_cfg.get(f"max_{category}_positions", 1))
        if self._count_mcp_positions(category) >= max_cat:
            logger.info(f"[MCP] Position limit reached for {category}")
            return

        # ④ FX ペアのみ LightGBM 確認（モデル不在時はスキップ）
        if not is_gold:
            try:
                from ml.lgbm_model import build_features
                now = now_utc()
                jst_now = to_jst(now)
                # MCP indicators（テクニカル指標）を payload にマージ
                # → build_features() が smc_data / market_data 両方から参照可能にする
                #   (trend_direction, momentum_long/short は smc_data から読まれるため)
                ind = payload.get("indicators", {})
                payload.update(ind)
                market_data = {**payload, "atr_14": atr, "spread_pips": 1.5}
                position_data = {
                    "open_positions_count": len(self._position_manager.positions),
                }
                detector = self._diff_detectors.setdefault(pair, DiffDetector())
                cached = detector.cached_result
                sentiment_score = float(cached.sentiment_score if cached else 0.0)
                calendar_risk_score = 0

                features = build_features(
                    smc_data=payload,
                    market_data=market_data,
                    position_data=position_data,
                    sentiment_score=sentiment_score,
                    calendar_risk_score=calendar_risk_score,
                    session_type=get_session_flag(now),
                    day_of_week=now.weekday(),
                )
                prediction = self._predictor.predict(pair, features)
                if prediction is not None:
                    # LightGBM が逆方向に強い場合のみブロック
                    if direction == "long" and prediction.prob_down > 0.55:
                        logger.info(
                            f"[MCP] Blocked by LightGBM: down={prediction.prob_down:.2f}"
                        )
                        return
                    if direction == "short" and prediction.prob_up > 0.55:
                        logger.info(
                            f"[MCP] Blocked by LightGBM: up={prediction.prob_up:.2f}"
                        )
                        return
            except Exception as e:
                logger.warning(f"[MCP] LightGBM check skipped: {e}")

        # ⑤ SL/TP 計算
        sl_pips, tp_pips = self._calc_mcp_sl_tp_pips(pair, atr)

        # ⑥ ロット計算
        risk_cfg = self._config.get("risk", {})
        if bool(risk_cfg.get("demo_fixed_lot_enabled", False)):
            lot = float(risk_cfg.get("demo_fixed_lot", 0.01))
        else:
            lot = calc_lot_size(self._broker.get_account_balance(), sl_pips, pair, self._risk_config)

        if is_gold:
            gold_scale = float(mcp_cfg.get("gold_lot_scale", 0.5))
            lot = round(lot * gold_scale, 2)
            lot = max(lot, 0.01)

        # 発注
        ok, ticket, sl_price, tp_price = await self._broker.open_position_async(
            pair, direction, lot, sl_pips, tp_pips,
        )

        if ok and ticket:
            # DB 記録
            trade_id = insert_trade(self._db_conn, {
                "pair": pair,
                "direction": direction,
                "open_time": now_utc(),
                "open_price": close_price,
                "volume": lot,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "mt5_ticket": ticket,
            })
            # ポジション登録
            self._position_manager.register_position(ManagedPosition(
                trade_id=trade_id,
                ticket=ticket,
                pair=pair,
                direction=direction,
                volume=lot,
                open_price=close_price,
                sl_price=sl_price,
                tp_price=tp_price,
                target_tp_price=tp_price,
                target_tp_pips=tp_pips,
                open_time_utc=now_utc(),
                atr_at_entry=atr,
                prob_up=0.0,
                prob_flat=0.0,
                prob_down=0.0,
                last_prediction_at_utc=now_utc(),
                is_mcp=True,
                entry_htf_bias=direction,  # ブレイクアウト方向 = HTF バイアス方向と推定
            ))
            self._mcp_position_tickets.add(ticket)

            await self._notifier.send(
                f"[MCP EA] 新規エントリー: {pair} {direction}\n"
                f"Lot: {lot} / SL: {sl_price} / TP: {tp_price}\n"
                f"Pattern: {pattern} (score {breakout_score}/10)"
            )
        elif not ok:
            logger.error(f"[MCP] Order failed for {pair}")

    async def _process_signal(self, payload: dict) -> None:
        """
        Webhook シグナルを処理する（クリティカルパス）。

        ① Calendar Veto確認
        ② GPT-5.2キャッシュ読込
        ③ LightGBM推論
        ④ リスクチェック
        ⑤ ドテン判定
        ⑥ 発注
        """
        pair = payload["pair"]
        signal_direction = payload["direction"]
        atr = payload["atr"]
        atr_20d_avg = payload.get("atr_20d_avg", 0.0)  # ← [NEW] webhook から20日平均ATRを取得
        close_price = payload["close"]

        # ← [NEW] LLM diff_detection_task で使用するATR情報をキャッシュ
        self._last_webhook_atrs[pair] = (atr, atr_20d_avg)

        logger.info(f"Processing signal: {pair} {signal_direction} ATR={atr} avg_atr_20d={atr_20d_avg}")

        # ② Calendar Veto（Layer A）
        veto_active, veto_reason = self._calendar_veto.is_veto_active(pair)
        if veto_active:
            logger.info(f"Signal vetoed by calendar: {veto_reason}")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": signal_direction,
                "mtf_confluence": payload.get("mtf_confluence"),
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
                "executed": False,
                "veto_reason": veto_reason,
            })
            # 高インパクト指標中は新規シグナルだけ拒否する。
            # 既存ポジションの即時全決済は損失を固定化しやすいため、設定時のみ有効化する。
            if bool(self._config.get("risk", {}).get("calendar_veto_force_close", False)):
                await self._position_manager.close_all_positions(reason="calendar_veto")
            return

        # ③ GPT-5.2 キャッシュ読込（Veto Layer B）
        detector = self._diff_detectors.setdefault(pair, DiffDetector())
        cached = detector.cached_result
        llm_cfg = self._config.get("llm", {})
        feature_threshold = float(llm_cfg.get("feature_news_importance_threshold", 0.55))
        importance = cached.news_importance_score if cached else 0.0
        sentiment_score = cached.sentiment_score if cached and importance >= feature_threshold else 0.0
        calendar_risk_score = 2 if (cached and cached.unexpected_veto) else (1 if importance >= feature_threshold else 0)
        if cached and cached.unexpected_veto:
            logger.info("Signal vetoed by GPT unexpected_veto flag")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": signal_direction,
                "gpt_sentiment": sentiment_score,
                "mtf_confluence": payload.get("mtf_confluence"),
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
                "executed": False,
                "veto_reason": "gpt_unexpected_veto",
            })
            return

        # 時間フィルター
        if is_excluded_hours():
            logger.info("Signal rejected: excluded broker hours (00:00-07:00 server time)")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": signal_direction,
                "mtf_confluence": payload.get("mtf_confluence"),
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
                "executed": False,
                "veto_reason": "excluded_hours",
            })
            return

        # ④ LightGBM推論
        # Webhook JSONに全テクニカル指標が含まれるため market_data として共用
        now = now_utc()
        jst_now = to_jst(now)
        market_data = {
            **payload,
            "atr_14": payload.get("atr", 0.0),
            "spread_pips": 1.5,
        }
        position_data = {
            "open_positions_count": len(self._position_manager.positions),
        }

        # 学習用特徴量サンプル保存（未ラベル）
        try:
            insert_training_sample(self._db_conn, {
                "pair": pair,
                "signal_time": now,
                "direction": signal_direction,
                "close_price": payload.get("close"),
                "atr": payload.get("atr"),
                "fvg_4h_zone_active": payload.get("fvg_4h_zone_active", False),
                "ob_4h_zone_active": payload.get("ob_4h_zone_active", False),
                "liq_sweep_1h": payload.get("liq_sweep_1h", False),
                "liq_sweep_qualified": payload.get("liq_sweep_qualified", False),
                "bos_1h": payload.get("bos_1h", False),
                "choch_1h": payload.get("choch_1h", False),
                "msb_15m_confirmed": payload.get("msb_15m_confirmed", False),
                "mtf_confluence": payload.get("mtf_confluence", 0),
                "atr_ratio": payload.get("atr_ratio", 1.0),
                "bb_width": payload.get("bb_width", 0.0),
                "close_vs_ema20_4h": payload.get("close_vs_ema20_4h", 0.0),
                "close_vs_ema50_4h": payload.get("close_vs_ema50_4h", 0.0),
                "high_low_range_15m": payload.get("high_low_range_15m", 0.0),
                "trend_direction": payload.get("trend_direction", 0),
                "momentum_long": payload.get("momentum_long", 0),
                "momentum_short": payload.get("momentum_short", 0),
                "macd_histogram": payload.get("macd_histogram", 0.0),
                "macd_signal_cross": payload.get("macd_signal_cross", 0),
                "rsi_14": payload.get("rsi_14", 50.0),
                "rsi_zone": payload.get("rsi_zone", 0),
                "stoch_k": payload.get("stoch_k", 50.0),
                "stoch_d": payload.get("stoch_d", 50.0),
                "momentum_3bar": payload.get("momentum_3bar", 0.0),
                "ob_4h_distance_pips": payload.get("ob_4h_distance_pips", 0.0),
                "fvg_4h_fill_ratio": payload.get("fvg_4h_fill_ratio", 0.0),
                "liq_sweep_strength": payload.get("liq_sweep_strength", 0.0),
                "fvg_4h_size_pips": payload.get("fvg_4h_size_pips", 0.0),
                "ob_4h_size_pips": payload.get("ob_4h_size_pips", 0.0),
                "sweep_depth_atr_ratio": payload.get("sweep_depth_atr_ratio", 0.0),
                "prior_candle_body_ratio": payload.get("prior_candle_body_ratio", 0.5),
                "consecutive_same_dir": payload.get("consecutive_same_dir", 0),
                "sweep_pending_bars": payload.get("sweep_pending_bars", 0),
                "spread_pips": 1.5,
                "session_flag": 1,
                "hour_of_day": jst_now.hour,
                "day_of_week": now.weekday(),
                "session_type": get_session_flag(now),
                "open_positions_count": len(self._position_manager.positions),
                "max_dd_24h": 0.0,
                "calendar_risk_score": calendar_risk_score,
                "sentiment_score": sentiment_score,
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
            })
        except Exception as e:
            logger.warning(f"Training sample insert failed: {e}")

        features = build_features(
            smc_data=payload,
            market_data=market_data,
            position_data=position_data,
            sentiment_score=sentiment_score,
            calendar_risk_score=calendar_risk_score,
            session_type=get_session_flag(now),
            day_of_week=now.weekday(),
        )
        prediction = self._predictor.predict(pair, features)
        if prediction is None:
            logger.warning(f"LightGBM prediction unavailable for {pair}")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": signal_direction,
                "mtf_confluence": payload.get("mtf_confluence"),
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
                "executed": False,
                "veto_reason": "model_unavailable",
            })
            return

        # 同一ペア保有中のポジションには、最新のLightGBM予測を都度反映する。
        self._position_manager.update_pair_prediction(
            pair,
            prediction.prob_up,
            prediction.prob_flat,
            prediction.prob_down,
        )

        # シグナル記録
        insert_signal(self._db_conn, {
            "pair": pair,
            "signal_time": now_utc(),
            "direction": signal_direction,
            "lgbm_prob_up": prediction.prob_up,
            "lgbm_prob_flat": prediction.prob_flat,
            "lgbm_prob_down": prediction.prob_down,
            "gpt_sentiment": sentiment_score,
            "mtf_confluence": payload.get("mtf_confluence"),
            "alert_mode": payload.get("alert_mode"),
            "quality_gate_pass": payload.get("quality_gate_pass"),
            "vol_ok": payload.get("vol_ok"),
            "in_session": payload.get("in_session"),
            "is_friday_late": payload.get("is_friday_late"),
            "executed": False,
        })

        # LightGBM が強いシグナルを出しているか確認
        model_acc = self._predictor.get_model_accuracy(pair)
        ml_cfg = self._config.get("ml", {})
        threshold_cfg = ml_cfg.get("prediction_thresholds", {})
        execution_direction_mode = str(ml_cfg.get("execution_direction_mode", "signal")).lower()
        direction = signal_direction
        if execution_direction_mode == "model" and prediction.direction != "flat":
            direction = prediction.direction

        pair_thresholds = threshold_cfg.get(pair, {})
        threshold_overrides = pair_thresholds.get(direction)
        if not prediction.is_strong_signal(direction, model_acc, threshold_overrides):
            logger.info(
                f"Signal not strong enough: {prediction.prob_up:.2f}/"
                f"{prediction.prob_flat:.2f}/{prediction.prob_down:.2f}"
            )
            return

        # flat は方向性なし。エントリー/ドテン対象にしない。
        if prediction.direction == "flat":
            logger.info(
                f"Signal skipped: flat direction "
                f"({prediction.prob_up:.2f}/{prediction.prob_flat:.2f}/{prediction.prob_down:.2f})"
            )
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": signal_direction,
                "lgbm_prob_up": prediction.prob_up,
                "lgbm_prob_flat": prediction.prob_flat,
                "lgbm_prob_down": prediction.prob_down,
                "gpt_sentiment": sentiment_score,
                "mtf_confluence": payload.get("mtf_confluence"),
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
                "executed": False,
                "veto_reason": "flat_signal",
            })
            return

        pair_direction_filter = self._config.get("pair_direction_filter", {})
        allowed_directions = pair_direction_filter.get(pair)
        if allowed_directions and direction not in allowed_directions:
            logger.info(f"Signal rejected by pair direction filter: {pair} {direction}")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": signal_direction,
                "lgbm_prob_up": prediction.prob_up,
                "lgbm_prob_flat": prediction.prob_flat,
                "lgbm_prob_down": prediction.prob_down,
                "gpt_sentiment": sentiment_score,
                "mtf_confluence": payload.get("mtf_confluence"),
                "alert_mode": payload.get("alert_mode"),
                "quality_gate_pass": payload.get("quality_gate_pass"),
                "vol_ok": payload.get("vol_ok"),
                "in_session": payload.get("in_session"),
                "is_friday_late": payload.get("is_friday_late"),
                "executed": False,
                "veto_reason": "pair_direction_filter",
            })
            return

        if direction != signal_direction:
            logger.info(
                f"Execution direction overridden by model: signal={signal_direction} -> trade={direction} "
                f"({prediction.prob_up:.2f}/{prediction.prob_flat:.2f}/{prediction.prob_down:.2f})"
            )

        # ⑤ ドテン判定
        existing_positions = [
            p for p in self._position_manager.positions.values()
            if p.pair == pair
        ]

        if existing_positions:
            pos = existing_positions[0]
            if direction != pos.direction:
                # 逆方向シグナル → ドテン判定
                if not self._position_manager.check_entry_age(pos.ticket, min_minutes=15):
                    logger.info(f"Doten ignored: entry too recent (< 15min)")
                    return

                if not self._position_manager.check_doten_allowed(pair):
                    # インターバル内 → 決済のみ
                    await self._position_manager._force_close(pos, "doten_close_only")
                    return

                # ドテン実行
                config = get_trading_config()
                sl_pips, tp_pips = calc_sl_tp_pips(
                    atr,
                    pair,
                    config,
                    ob_4h_distance_pips=payload.get("ob_4h_distance_pips", 0.0),
                    tp_swing_pips=payload.get("tp_swing_pips", 0.0),
                    tp_fvg_pips=payload.get("tp_fvg_pips", 0.0),
                )
                risk_cfg = config.get("risk", {})
                if bool(risk_cfg.get("demo_fixed_lot_enabled", False)):
                    lot = float(risk_cfg.get("demo_fixed_lot", 0.01))
                else:
                    lot = calc_lot_size(
                        self._broker.get_account_balance(),
                        sl_pips, pair, self._risk_config,
                    )
                ok, new_ticket, sl_price, tp_price = await self._position_manager.execute_doten(
                    pair, pos.ticket, direction,
                    lot, sl_pips, tp_pips, close_price,
                )
                if ok and new_ticket:
                    trade_id = insert_trade(self._db_conn, {
                        "pair": pair,
                        "direction": direction,
                        "open_time": now_utc(),
                        "open_price": close_price,
                        "volume": lot,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "mt5_ticket": new_ticket,
                    })
                    self._position_manager.register_position(ManagedPosition(
                        trade_id=trade_id,
                        ticket=new_ticket,
                        pair=pair,
                        direction=direction,
                        volume=lot,
                        open_price=close_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        target_tp_price=tp_price,
                        target_tp_pips=tp_pips,
                        open_time_utc=now_utc(),
                        atr_at_entry=atr,
                        prob_up=prediction.prob_up,
                        prob_flat=prediction.prob_flat,
                        prob_down=prediction.prob_down,
                        last_prediction_at_utc=now_utc(),
                    ))
                return
            else:
                logger.info(f"Same direction already held for {pair}")
                return

        # ⑥ リスクチェック → 新規エントリー
        open_positions = self._broker.get_positions()
        allowed, reason = check_exposure(
            open_positions, pair, self._risk_config,
            self._broker.get_account_balance(),
        )
        if not allowed:
            logger.info(f"Entry blocked by risk check: {reason}")
            return

        daily_pnl = get_daily_pnl(
            self._db_conn,
            day_start_utc=broker_day_start_utc(),
        )
        dd_allowed, dd_pct = check_daily_drawdown(
            daily_pnl, self._broker.get_account_balance(), self._risk_config,
        )
        if not dd_allowed:
            logger.info(f"Entry blocked by daily drawdown: {dd_pct:.1f}%")
            return

        # 発注
        config = get_trading_config()
        sl_pips, tp_pips = calc_sl_tp_pips(
            atr,
            pair,
            config,
            ob_4h_distance_pips=payload.get("ob_4h_distance_pips", 0.0),
            tp_swing_pips=payload.get("tp_swing_pips", 0.0),
            tp_fvg_pips=payload.get("tp_fvg_pips", 0.0),
        )
        risk_cfg = config.get("risk", {})
        if bool(risk_cfg.get("demo_fixed_lot_enabled", False)):
            lot = float(risk_cfg.get("demo_fixed_lot", 0.01))
        else:
            lot = calc_lot_size(
                self._broker.get_account_balance(),
                sl_pips, pair, self._risk_config,
            )
        sl_price, tp_price = self._calc_sl_tp_price(
            pair, direction, close_price, sl_pips, tp_pips,
        )

        ok, ticket, sl_price, tp_price = await self._broker.open_position_async(
            pair, direction, lot, sl_pips, tp_pips,
        )

        if ok and ticket:
            trade_record = {
                "pair": pair,
                "direction": direction,
                "open_time": now_utc(),
                "open_price": close_price,
                "volume": lot,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "mt5_ticket": ticket,
            }
            trade_id = insert_trade(self._db_conn, trade_record)

            managed = ManagedPosition(
                trade_id=trade_id,
                ticket=ticket,
                pair=pair,
                direction=direction,
                volume=lot,
                open_price=close_price,
                sl_price=sl_price,
                tp_price=tp_price,
                target_tp_price=tp_price,
                target_tp_pips=tp_pips,
                open_time_utc=now_utc(),
                atr_at_entry=atr,
                prob_up=prediction.prob_up,
                prob_flat=prediction.prob_flat,
                prob_down=prediction.prob_down,
                last_prediction_at_utc=now_utc(),
            )
            self._position_manager.register_position(managed)

            await self._notifier.send(
                f"新規エントリー: {pair} {direction}\n"
                f"Lot: {lot} / SL: {sl_price} / TP: {tp_price}\n"
                f"LightGBM: up={prediction.prob_up:.2f} flat={prediction.prob_flat:.2f} "
                f"down={prediction.prob_down:.2f}"
            )

        elif not ok:
            logger.error(f"Order failed for {pair}")

    def _calc_sl_tp_price(
        self, pair: str, direction: str, close_price: float,
        sl_pips: float, tp_pips: float,
    ) -> tuple[float, float]:
        """SL/TP の実際の価格を計算する。"""
        if pair in ("USDJPY", "GBPJPY"):
            pip_unit = 0.01
        else:
            pip_unit = 0.0001

        if direction == "long":
            sl_price = round(close_price - sl_pips * pip_unit, 5)
            tp_price = round(close_price + tp_pips * pip_unit, 5)
        else:
            sl_price = round(close_price + sl_pips * pip_unit, 5)
            tp_price = round(close_price - tp_pips * pip_unit, 5)

        return sl_price, tp_price

    # ── バックグラウンドタスク ──────────────────
    def _build_pair_news_articles(self, pair: str) -> list[dict]:
        """Calendar Vetoキャッシュをニュース記事形式に変換する。"""
        now = now_utc()
        recent_cutoff = now - timedelta(hours=2)
        future_cutoff = now + timedelta(hours=12)
        pair_currencies = {pair[:3], pair[3:]}

        pair_events = [
            ev for ev in self._calendar_veto.events
            if ev["currency"] in pair_currencies and recent_cutoff <= ev["datetime_utc"] <= future_cutoff
        ]

        return [
            {
                "title": f"{ev['currency']} {ev['title']}",
                "summary": f"High impact at {format_jst(ev['datetime_utc'])}",
            }
            for ev in pair_events[:10]
        ]

    async def _diff_detection_task(self) -> None:
        """15分ごとの差分検知タスク。"""
        try:
            config = get_trading_config()
            if not config["llm"].get("llm_enabled", True):
                return

            # 市場クローズ中は差分検知不要（APIコスト/エラーノイズ抑制）
            if is_broker_market_closed():
                return

            for pair in config["system"]["pairs"]:
                veto_active, _ = self._calendar_veto.is_veto_active(pair)
                detector = self._diff_detectors.setdefault(pair, DiffDetector())
                news_articles = self._build_pair_news_articles(pair)
                market_context = (
                    f"pair={pair}, high_impact_events_nearby={len(news_articles)}, "
                    f"mode={'low_power' if detector.is_low_power else 'normal'}"
                )

                # ← [NEW] webhook キャッシュから ATR情報を取得
                current_atr, avg_atr_20d = self._last_webhook_atrs.get(pair, (0.0, 0.0))

                should_call, reason = detector.run_diff_check(
                    news_articles=news_articles,
                    current_atr=current_atr,   # ← [FIXED] webhook から取得した値を使用
                    avg_atr_20d=avg_atr_20d,
                    calendar_veto_active=veto_active,
                )

                if should_call:
                    try:
                        result = await self._llm_client.analyze_sentiment_hybrid(
                            pair=pair,
                            news_articles=news_articles,
                            market_context=market_context,
                            reason=reason,
                        )
                        detector._cached_result = result
                        detector._last_call_time = now_utc()
                    except Exception as e:
                        # モデル未提供/アクセス不可時は設定を変えず、通知のみ（スロットリング）
                        msg = str(e)
                        if "model_not_found" in msg or "does not exist" in msg:
                            now = now_utc()
                            should_notify = (
                                self._last_llm_model_error_notify is None
                                or (now - self._last_llm_model_error_notify).total_seconds() >= 3600
                            )
                            if should_notify:
                                await self._notifier.send_alert(
                                    "LLM呼び出し失敗: model_not_found。"
                                    "llm_enabled は変更せず継続します（1時間ごとに通知）。"
                                )
                                self._last_llm_model_error_notify = now
                            logger.warning("LLM model_not_found detected; keeping llm_enabled as-is")
                            continue
                        raise

        except Exception as e:
            logger.error(f"Diff detection task error: {e}")

    async def _position_monitor_task(self) -> None:
        """毎分のポジション監視タスク。"""
        try:
            await self._position_manager.monitor_positions()
        except Exception as e:
            logger.error(f"Position monitor error: {e}")

    async def _daily_maintenance_task(self) -> None:
        """日次メンテナンス。"""
        from maintenance.scheduler import daily_maintenance
        await daily_maintenance(
            self._db_conn, self._notifier,
            self._settings.log_dir, self._settings.db_backup_path,
            self._calendar_veto,
        )

    async def _weekend_optimization_task(self) -> None:
        """週末最適化。"""
        from optimizer.weekend_optimizer import run_weekend_optimization
        result = run_weekend_optimization(self._db_conn)
        if result.get("applied"):
            reload_trading_config()
            self._risk_config = load_risk_config(get_trading_config())
            await self._notifier.send(
                f"週末最適化完了（適用済み）\n"
                f"パラメータ: {result.get('params')}\n"
                f"最適化Sharpe: {result.get('sharpe_optimize', 0):.4f}\n"
                f"検証Sharpe: {result.get('sharpe_validate', 0):.4f}"
            )
        else:
            await self._notifier.send(
                f"週末最適化完了（不採用）\n理由: {result.get('reason', 'N/A')}"
            )

    async def _weekly_maintenance_task(self) -> None:
        """週次メンテナンス（再学習・レポート・DB整備・ロールバック判定）。"""
        from maintenance.scheduler import weekly_maintenance
        retrain_result = await weekly_maintenance(self._db_conn, self._notifier)

        # 執行ノイズ抑制パラメータの週次自動チューニング（AI判定ロジックは変更しない）
        from optimizer.weekend_optimizer import auto_tune_execution_noise
        tune_result = auto_tune_execution_noise(self._db_conn)
        if tune_result.get("applied"):
            self._config = reload_trading_config()
            self._risk_config = load_risk_config(self._config)
            await self._notifier.send(
                "執行ノイズ最適化を適用しました\n"
                f"new: {tune_result.get('new')}\n"
                f"metrics: {tune_result.get('metrics')}"
            )
        else:
            logger.info(f"Execution noise tuning skipped: {tune_result.get('reason', 'N/A')}")

        # Exit比率ベース最適化（time_exit/atr_sl 構成を週次で補正）
        from optimizer.weekend_optimizer import auto_tune_exit_mix
        exit_mix_result = auto_tune_exit_mix(self._db_conn)
        if exit_mix_result.get("applied"):
            self._config = reload_trading_config()
            self._risk_config = load_risk_config(self._config)
            await self._notifier.send(
                "Exit比率最適化を適用しました\n"
                f"new: {exit_mix_result.get('new')}\n"
                f"metrics: {exit_mix_result.get('metrics')}"
            )
        else:
            logger.info(f"Exit mix tuning skipped: {exit_mix_result.get('reason', 'N/A')}")

        # 方向別配分最適化（pair x direction の prediction_thresholds を週次で補正）
        from optimizer.weekend_optimizer import auto_tune_directional_allocation
        directional_result = auto_tune_directional_allocation(self._db_conn)
        if directional_result.get("applied"):
            changes = directional_result.get("changed", [])
            self._config = reload_trading_config()
            await self._notifier.send(
                "方向別配分最適化を適用しました\n"
                f"updated_buckets: {len(changes)}\n"
                f"sample: {changes[:3]}"
            )
        else:
            logger.info(f"Directional allocation tuning skipped: {directional_result.get('reason', 'N/A')}")

        # 週次メンテ後に最新モデルを再読み込み
        reload_result = self._predictor.load_all_models()
        trained_pairs = [p for p, ok in reload_result.items() if ok]
        skipped_pairs = [p for p, ok in reload_result.items() if not ok]

        # 週次再学習後にWFV精度を更新
        validation = retrain_result.get("retrain", {}).get("validation", {})
        for _pair, val_result in validation.items():
            acc = float(val_result.get("balanced_accuracy", val_result.get("accuracy", 0.0)))
            if acc > 0:
                self._predictor.set_model_accuracy(_pair, acc)
                logger.info(f"Model accuracy updated after retraining: {_pair}={acc:.4f}")

        await self._notifier.send(
            f"週次モデル再ロード完了\n"
            f"Loaded: {', '.join(trained_pairs) if trained_pairs else 'none'}\n"
            f"Missing: {', '.join(skipped_pairs) if skipped_pairs else 'none'}"
        )

        # ロールバックチェック
        from optimizer.weekend_optimizer import check_weekly_rollback
        if check_weekly_rollback(self._db_conn):
            reload_trading_config()
            self._risk_config = load_risk_config(get_trading_config())
            await self._notifier.send_alert("週次ロールバック実行: 前週パラメータに復元しました")

    async def _monthly_maintenance_guard_task(self) -> None:
        """月次メンテの実行条件（毎月第1日曜 02:00 JST）を判定して実行する。"""
        now = now_utc()
        jst_now = to_jst(now)

        # 毎月月初の日曜 02:00 JST のみ実行
        if not (jst_now.weekday() == 6 and jst_now.day <= 7 and jst_now.hour == 2):
            return

        from maintenance.scheduler import monthly_maintenance
        await monthly_maintenance(self._db_conn, self._notifier, self._settings.model_dir)


async def main() -> None:
    """エントリーポイント。"""
    setup_logger()
    orchestrator = Orchestrator()

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(orchestrator.stop()))
        except NotImplementedError:
            # Windows では signal handler が制限される
            pass

    # FastAPI サーバーをバックグラウンドで起動
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=orchestrator._settings.webhook_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # 並行実行: Webhook サーバー + Orchestrator
    try:
        await asyncio.gather(
            server.serve(),
            orchestrator.start(),
        )
    except asyncio.CancelledError:
        logger.info("Shutdown requested (task cancelled)")
    except RuntimeError as e:
        logger.error(f"System startup/runtime error: {e}")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
