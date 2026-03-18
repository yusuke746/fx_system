"""
MT5 ブローカー連携モジュール

■ 重要な制約
  - MT5 Python ライブラリは API レベルでスレッドセーフではない。
  - ThreadPoolExecutor(max_workers=1) でワーカーを1つに制限し、
    MT5への発注はシリアル（直列）に処理する。
  - asyncio の I/O 待機重複解消の効果は生かしつつ、
    決済→確認→新規の順序保証を明示する。

■ 時刻基準
  - MT5 サーバー時刻: EET（UTC+2 / 夏時間 UTC+3）
  - MT5 から取得した日時は mt5_server_to_utc() で UTC に変換してから保存する。
  - MT5 に送信する日時は utc_to_mt5_server() で EET に変換する。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from loguru import logger

from core.time_manager import mt5_server_to_utc, utc_to_mt5_server, now_utc, format_jst

# MT5 のインポートはVPS上でのみ利用可能
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 module not available. Broker functions will be disabled.")


# MT5 呼び出しをシリアル化する ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=1)

# MT5 retcode 定義
TRADE_RETCODE_DONE = 10009


class MT5Broker:
    """MT5 ブローカー接続管理。"""

    def __init__(self, login: int, password: str, server: str):
        self._login = login
        self._password = password
        self._server = server
        self._connected = False

    def connect(self) -> bool:
        """MT5ターミナルに接続する。"""
        if not MT5_AVAILABLE:
            logger.error("MT5 module not available")
            return False

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        authorized = mt5.login(
            login=self._login,
            password=self._password,
            server=self._server,
        )
        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self._connected = True
        info = mt5.terminal_info()
        logger.info(f"MT5 connected: {self._server}, build={info.build if info else 'N/A'}")
        return True

    def disconnect(self) -> None:
        """MT5ターミナルから切断する。"""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        if not MT5_AVAILABLE or not self._connected:
            return False
        info = mt5.terminal_info()
        return info is not None

    def reconnect(self, max_retries: int = 10, interval_sec: int = 60) -> bool:
        """再接続を試みる（フォールバック D.2）。"""
        for i in range(max_retries):
            logger.info(f"MT5 reconnect attempt {i + 1}/{max_retries}")
            if self.connect():
                return True
            import time
            time.sleep(interval_sec)
        logger.critical(f"MT5 reconnect failed after {max_retries} attempts")
        return False

    # ── 口座情報 ─────────────────────────────────
    def get_account_balance(self) -> float:
        """口座残高（円）を返す。"""
        if not MT5_AVAILABLE:
            return 0.0
        info = mt5.account_info()
        return info.balance if info else 0.0

    def get_account_equity(self) -> float:
        """口座有効証拠金（円）を返す。"""
        if not MT5_AVAILABLE:
            return 0.0
        info = mt5.account_info()
        return info.equity if info else 0.0

    # ── ポジション ───────────────────────────────
    def get_positions(self, pair: str | None = None) -> list[dict]:
        """
        オープンポジションを取得する。
        MT5の時刻はUTC（保存基準）に変換して返す。
        """
        if not MT5_AVAILABLE:
            return []

        if pair:
            positions = mt5.positions_get(symbol=pair)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "pair": pos.symbol,
                "direction": "long" if pos.type == 0 else "short",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "sl_price": pos.sl,
                "tp_price": pos.tp,
                "profit": pos.profit,
                "open_time_utc": mt5_server_to_utc(
                    datetime.fromtimestamp(pos.time)
                ),
                "margin_used": pos.volume * 100000 * pos.price_open / 1000,  # 概算
            })

        return result

    def get_recent_closed_position_info(
        self,
        ticket: int,
        lookback_hours: int = 48,
    ) -> dict | None:
        """
        直近の約定履歴から、指定ポジションの最終決済情報を取得する。

        MT5側でTP/SLが先に約定した場合でも、ローカル管理状態を同期するために使う。
        """
        if not MT5_AVAILABLE:
            return None

        end = now_utc()
        start = end - timedelta(hours=lookback_hours)

        try:
            deals = mt5.history_deals_get(
                utc_to_mt5_server(start),
                utc_to_mt5_server(end),
                position=ticket,
            )
        except TypeError:
            # 一部のMT5 Python wrapperは position 引数非対応のため全件から絞り込む。
            deals = mt5.history_deals_get(
                utc_to_mt5_server(start),
                utc_to_mt5_server(end),
            )

        if not deals:
            return None

        closing_deals = []
        deal_entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
        deal_entry_out_by = getattr(mt5, "DEAL_ENTRY_OUT_BY", 3)
        for deal in deals:
            if getattr(deal, "position_id", None) != ticket:
                continue
            if getattr(deal, "entry", None) in (deal_entry_out, deal_entry_out_by):
                closing_deals.append(deal)

        if not closing_deals:
            return None

        close_deal = max(closing_deals, key=lambda d: getattr(d, "time", 0))
        return {
            "ticket": ticket,
            "close_price": float(getattr(close_deal, "price", 0.0) or 0.0),
            "profit": float(getattr(close_deal, "profit", 0.0) or 0.0),
            "volume": float(getattr(close_deal, "volume", 0.0) or 0.0),
            "close_time_utc": mt5_server_to_utc(datetime.fromtimestamp(int(close_deal.time))),
            "comment": str(getattr(close_deal, "comment", "") or ""),
        }

    # ── 発注（シリアル実行） ──────────────────────
    async def open_position_async(
        self,
        pair: str,
        direction: str,
        volume: float,
        sl_pips: float,
        tp_pips: float,
    ) -> tuple[bool, int | None, float, float]:
        """
        非同期で新規ポジションを開く（ThreadPoolExecutor経由でシリアル実行）。
        SL/TPは発注時の実tick価格から計算する。

        Returns:
            (success, ticket or None, actual_sl_price, actual_tp_price)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self._open_position_sync,
            pair, direction, volume, sl_pips, tp_pips,
        )

    def _open_position_sync(
        self,
        pair: str,
        direction: str,
        volume: float,
        sl_pips: float,
        tp_pips: float,
    ) -> tuple[bool, int | None, float, float]:
        """新規ポジションを開く（同期版）。SL/TPは発注時の実tick価格から計算する。"""
        if not MT5_AVAILABLE:
            return False, None, 0.0, 0.0

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            logger.error(f"Cannot get tick for {pair}")
            return False, None, 0.0, 0.0

        price = tick.ask if direction == "long" else tick.bid

        # JPY建てペアは 0.01=1pip、その他は 0.0001=1pip
        pip_unit = 0.01 if pair in ("USDJPY", "GBPJPY") else 0.0001

        # ブローカーの最小ストップ距離（ポイント単位）を強制する
        # 環境によって stops_level / trade_stops_level のどちらかになるため両対応する。
        info = mt5.symbol_info(pair)
        stops_level = 0
        point = 0.0
        if info is not None:
            stops_level = int(getattr(info, "stops_level", getattr(info, "trade_stops_level", 0)) or 0)
            point = float(getattr(info, "point", 0.0) or 0.0)

        if stops_level > 0 and point > 0:
            min_stop_pips = stops_level * point / pip_unit
            sl_pips = max(sl_pips, min_stop_pips + 1.0)
            tp_pips = max(tp_pips, min_stop_pips + 1.0)

        if direction == "long":
            sl_price = round(price - sl_pips * pip_unit, 5)
            tp_price = round(price + tp_pips * pip_unit, 5)
        else:
            sl_price = round(price + sl_pips * pip_unit, 5)
            tp_price = round(price - tp_pips * pip_unit, 5)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 202202,
            "comment": "fx_system_v2.2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send failed for {pair}: result is None")
            return False, None, 0.0, 0.0

        if result.retcode == TRADE_RETCODE_DONE:
            logger.info(
                f"Position opened: {pair} {direction} vol={volume} "
                f"price={price} SL={sl_price} TP={tp_price} ticket={result.order}"
            )
            return True, result.order, sl_price, tp_price

        logger.error(
            f"Order rejected for {pair}: retcode={result.retcode} "
            f"comment={result.comment}"
        )
        return False, None, 0.0, 0.0

    async def close_position_async(self, ticket: int) -> bool:
        """非同期でポジションを決済する。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self._close_position_sync,
            ticket,
        )

    def _close_position_sync(self, ticket: int) -> bool:
        """ポジションを決済する（同期版）。"""
        if not MT5_AVAILABLE:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position not found: ticket={ticket}")
            return False

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False

        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 202202,
            "comment": "fx_system_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == TRADE_RETCODE_DONE:
            logger.info(f"Position closed: ticket={ticket} price={price}")
            return True

        logger.error(
            f"Close failed: ticket={ticket} "
            f"retcode={result.retcode if result else 'None'}"
        )
        return False

    async def modify_sl_tp_async(
        self, ticket: int, sl: float | None = None, tp: float | None = None
    ) -> bool:
        """SL/TP を変更する。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self._modify_sl_tp_sync,
            ticket, sl, tp,
        )

    def _modify_sl_tp_sync(
        self, ticket: int, sl: float | None, tp: float | None
    ) -> bool:
        """SL/TP を変更する（同期版）。"""
        if not MT5_AVAILABLE:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == TRADE_RETCODE_DONE:
            logger.debug(f"SL/TP modified: ticket={ticket} SL={sl} TP={tp}")
            return True

        logger.error(f"SL/TP modify failed: ticket={ticket}")
        return False

    async def partial_close_async(self, ticket: int, close_volume: float) -> bool:
        """部分決済（スケールアウト用）。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self._partial_close_sync,
            ticket, close_volume,
        )

    def _partial_close_sync(self, ticket: int, close_volume: float) -> bool:
        """部分決済（同期版）。"""
        if not MT5_AVAILABLE:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False

        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": round(close_volume, 2),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 202202,
            "comment": "fx_system_scale_out",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == TRADE_RETCODE_DONE:
            logger.info(f"Partial close: ticket={ticket} vol={close_volume}")
            return True

        logger.error(f"Partial close failed: ticket={ticket}")
        return False
