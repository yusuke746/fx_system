"""
FX自動売買システム v2.2 — メインオーケストレーター

■ クリティカルパス（目標500ms以内）
  ① TradingView Webhook受信（MTF SMC + テクニカル条件成立時のみ）
  ② Calendar Veto確認（Python IFルール・0ms）
  ③ GPT-5.2キャッシュ読込（メモリから即時・0ms）
    ④ LightGBM推論（36特徴量・1ms以下）
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
from core.time_manager import now_utc, format_jst, is_excluded_hours, to_jst
from broker.mt5_broker import MT5Broker
from llm.llm_client import DiffDetector, LLMClient
from ml.lgbm_model import LGBMPredictor, build_features
from veto.calendar_veto import CalendarVeto
from webhook.server import app as fastapi_app
from webhook.signal_queue import signal_queue


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
        self._diff_detector = DiffDetector()
        self._llm_client = LLMClient(
            api_key=self._settings.openai_api_key.get_secret_value(),
            db_conn=self._db_conn,
        )

        # LightGBM
        self._predictor = LGBMPredictor(model_dir=self._settings.model_dir)

        # ポジション管理
        self._position_manager = PositionManager(self._broker, self._notifier)

        # リスク設定
        self._risk_config = load_risk_config(self._config)

        # スケジューラ
        self._scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        """システムを起動する。"""
        logger.info(f"=== FX Auto-Trading System v{self._config['system']['version']} ===")
        logger.info(f"Demo mode: {self._config['system']['demo_mode']}")
        logger.info(f"Pairs: {self._config['system']['pairs']}")
        logger.info(f"Start time: {format_jst(now_utc())}")

        self._running = True

        # MT5 接続
        if not self._broker.connect():
            await self._notifier.send_critical("MT5接続失敗。手動確認が必要です。")
            logger.critical("MT5 connection failed. System will start without broker.")

        # カレンダー取得
        try:
            self._calendar_veto.fetch_events()
        except Exception as e:
            logger.warning(f"Initial calendar fetch failed: {e}")

        # LightGBM モデル読み込み
        model_results = self._predictor.load_all_models()
        for pair, ok in model_results.items():
            if not ok:
                logger.warning(f"Model not loaded for {pair} — predictions unavailable")

        # スケジューラ設定
        self._setup_scheduler()
        self._scheduler.start()

        # 起動通知
        await self._notifier.send(
            f"システム起動完了\n"
            f"バージョン: v{self._config['system']['version']}\n"
            f"デモモード: {self._config['system']['demo_mode']}\n"
            f"対象ペア: {', '.join(self._config['system']['pairs'])}"
        )

        # メインループ: Webhook シグナル処理
        await self._signal_processing_loop()

    async def stop(self) -> None:
        """システムを停止する。"""
        self._running = False
        self._scheduler.shutdown(wait=False)
        self._broker.disconnect()
        self._db_conn.close()
        logger.info("System stopped")

    def _setup_scheduler(self) -> None:
        """APScheduler のジョブを設定する。"""
        # 毎15分: 差分検知タスク
        self._scheduler.add_job(
            self._diff_detection_task,
            CronTrigger(minute="*/15"),
            id="diff_detection",
            name="Diff Detection (15min)",
        )

        # 毎1分: ポジション監視
        self._scheduler.add_job(
            self._position_monitor_task,
            CronTrigger(minute="*"),
            id="position_monitor",
            name="Position Monitor (1min)",
        )

        # 毎日 01:00 JST = 16:00 UTC（前日）
        self._scheduler.add_job(
            self._daily_maintenance_task,
            CronTrigger(hour=16, minute=0),  # UTC
            id="daily_maintenance",
            name="Daily Maintenance (01:00 JST)",
        )

        # 毎週土曜 14:00 JST = 05:00 UTC
        self._scheduler.add_job(
            self._weekend_optimization_task,
            CronTrigger(day_of_week="sat", hour=5, minute=0),  # UTC
            id="weekend_optimization",
            name="Weekend Optimization (Sat 14:00 JST)",
        )

        # 毎週日曜 23:00 JST = 14:00 UTC
        self._scheduler.add_job(
            self._weekly_maintenance_task,
            CronTrigger(day_of_week="sun", hour=14, minute=0),  # UTC
            id="weekly_maintenance",
            name="Weekly Maintenance (Sun 23:00 JST)",
        )

        logger.info("Scheduler configured with all maintenance jobs")

    # ── メインシグナル処理ループ ──────────────────
    async def _signal_processing_loop(self) -> None:
        """Webhook キューからシグナルを受け取って処理するメインループ。"""
        while self._running:
            try:
                payload = await asyncio.wait_for(signal_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal queue error: {e}")
                continue

            try:
                await self._process_signal(payload)
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
                await self._notifier.send_alert(f"シグナル処理エラー: {e}")

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
        direction = payload["direction"]
        atr = payload["atr"]
        close_price = payload["close"]

        logger.info(f"Processing signal: {pair} {direction} ATR={atr}")

        # ② Calendar Veto（Layer A）
        veto_active, veto_reason = self._calendar_veto.is_veto_active(pair)
        if veto_active:
            logger.info(f"Signal vetoed by calendar: {veto_reason}")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": direction,
                "mtf_confluence": payload.get("mtf_confluence"),
                "executed": False,
                "veto_reason": veto_reason,
            })
            # Veto中は全ポジション強制クローズ + SL→BE
            await self._position_manager.close_all_positions(reason="calendar_veto")
            return

        # ③ GPT-5.2 キャッシュ読込（Veto Layer B）
        cached = self._diff_detector.cached_result
        sentiment_score = cached.sentiment_score if cached else 0.0
        if cached and cached.unexpected_veto:
            logger.info("Signal vetoed by GPT unexpected_veto flag")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": direction,
                "gpt_sentiment": sentiment_score,
                "mtf_confluence": payload.get("mtf_confluence"),
                "executed": False,
                "veto_reason": "gpt_unexpected_veto",
            })
            return

        # 時間フィルター
        if is_excluded_hours():
            logger.info("Signal rejected: excluded hours (00:00-07:00 JST)")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": direction,
                "mtf_confluence": payload.get("mtf_confluence"),
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
                "direction": direction,
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
                "prior_candle_body_ratio": payload.get("prior_candle_body_ratio", 0.5),
                "consecutive_same_dir": payload.get("consecutive_same_dir", 0),
                "pivot_proximity": payload.get("pivot_proximity", 0.0),
                "sweep_pending_bars": payload.get("sweep_pending_bars", 0),
                "spread_pips": 1.5,
                "session_flag": 1,
                "hour_of_day": jst_now.hour,
                "day_of_week": now.weekday(),
                "open_positions_count": len(self._position_manager.positions),
                "max_dd_24h": 0.0,
                "calendar_risk_score": 0,
                "sentiment_score": sentiment_score,
            })
        except Exception as e:
            logger.warning(f"Training sample insert failed: {e}")

        features = build_features(
            smc_data=payload,
            market_data=market_data,
            position_data=position_data,
            sentiment_score=sentiment_score,
            calendar_risk_score=0,
        )
        prediction = self._predictor.predict(pair, features)
        if prediction is None:
            logger.warning(f"LightGBM prediction unavailable for {pair}")
            insert_signal(self._db_conn, {
                "pair": pair,
                "signal_time": now_utc(),
                "direction": direction,
                "mtf_confluence": payload.get("mtf_confluence"),
                "executed": False,
                "veto_reason": "model_unavailable",
            })
            return

        # シグナル記録
        insert_signal(self._db_conn, {
            "pair": pair,
            "signal_time": now_utc(),
            "direction": direction,
            "lgbm_prob_up": prediction.prob_up,
            "lgbm_prob_flat": prediction.prob_flat,
            "lgbm_prob_down": prediction.prob_down,
            "gpt_sentiment": sentiment_score,
            "mtf_confluence": payload.get("mtf_confluence"),
            "executed": False,
        })

        # LightGBM が強いシグナルを出しているか確認
        if not prediction.is_strong_signal:
            logger.info(
                f"Signal not strong enough: {prediction.prob_up:.2f}/"
                f"{prediction.prob_flat:.2f}/{prediction.prob_down:.2f}"
            )
            return

        # ⑤ ドテン判定
        existing_positions = [
            p for p in self._position_manager.positions.values()
            if p.pair == pair
        ]

        if existing_positions:
            pos = existing_positions[0]
            if prediction.direction != pos.direction:
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
                sl_pips, tp_pips = calc_sl_tp_pips(atr, pair, config)
                lot = calc_lot_size(
                    self._broker.get_account_balance(),
                    sl_pips, pair, self._risk_config,
                )
                sl_price, tp_price = self._calc_sl_tp_price(
                    pair, prediction.direction, close_price, sl_pips, tp_pips,
                )
                await self._position_manager.execute_doten(
                    pair, pos.ticket, prediction.direction,
                    lot, sl_price, tp_price, atr,
                )
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

        daily_pnl = get_daily_pnl(self._db_conn)
        dd_allowed, dd_pct = check_daily_drawdown(
            daily_pnl, self._broker.get_account_balance(), self._risk_config,
        )
        if not dd_allowed:
            logger.info(f"Entry blocked by daily drawdown: {dd_pct:.1f}%")
            return

        # 発注
        config = get_trading_config()
        sl_pips, tp_pips = calc_sl_tp_pips(atr, pair, config)
        lot = calc_lot_size(
            self._broker.get_account_balance(),
            sl_pips, pair, self._risk_config,
        )
        sl_price, tp_price = self._calc_sl_tp_price(
            pair, prediction.direction, close_price, sl_pips, tp_pips,
        )

        ok, ticket = await self._broker.open_position_async(
            pair, prediction.direction, lot, sl_price, tp_price,
        )

        if ok and ticket:
            managed = ManagedPosition(
                ticket=ticket,
                pair=pair,
                direction=prediction.direction,
                volume=lot,
                open_price=close_price,
                sl_price=sl_price,
                tp_price=tp_price,
                open_time_utc=now_utc(),
                atr_at_entry=atr,
            )
            self._position_manager.register_position(managed)

            trade_record = {
                "pair": pair,
                "direction": prediction.direction,
                "open_time": now_utc(),
                "open_price": close_price,
                "volume": lot,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "mt5_ticket": ticket,
            }
            insert_trade(self._db_conn, trade_record)

            await self._notifier.send(
                f"新規エントリー: {pair} {prediction.direction}\n"
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
    async def _diff_detection_task(self) -> None:
        """15分ごとの差分検知タスク。"""
        try:
            config = get_trading_config()
            if not config["llm"].get("llm_enabled", True):
                return

            for pair in config["system"]["pairs"]:
                veto_active, _ = self._calendar_veto.is_veto_active(pair)

                should_call, reason = self._diff_detector.run_diff_check(
                    news_articles=[],  # ニュースフィード実装時に結合
                    current_atr=0.0,   # MT5からリアルタイム取得時に結合
                    avg_atr_20d=0.0,
                    calendar_veto_active=veto_active,
                )

                if should_call:
                    result = await self._llm_client.analyze_sentiment(
                        pair=pair,
                        news_articles=[],
                        market_context="",
                        reason=reason,
                    )
                    self._diff_detector._cached_result = result
                    self._diff_detector._last_call_time = now_utc()

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
        await weekly_maintenance(self._db_conn, self._notifier)

        # 週次メンテ後に最新モデルを再読み込み
        reload_result = self._predictor.load_all_models()
        trained_pairs = [p for p, ok in reload_result.items() if ok]
        skipped_pairs = [p for p, ok in reload_result.items() if not ok]

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
    await asyncio.gather(
        server.serve(),
        orchestrator.start(),
    )


if __name__ == "__main__":
    asyncio.run(main())
