"""
ポジション管理・エグジット戦略モジュール

■ エグジット優先順位（v2.2 最終確定版）
  1. Calendar Veto 強制クローズ（最高優先）
  2. ATR動的SL到達（MT5 OCO注文）
  3. 反対シグナルドテン（インターバル制限付き）
  4. スケールアウト Step1（+1.0×ATR → 50%利確 + SL→BE）
  5. スケールアウト Step2（+1.8×ATR → 25%利確 + トレーリングへ）
  6. トレーリングストップ（0.8×ATR追従）
  7. 時間ベース強制クローズ（4h超過/金曜22:00/セッション跨ぎ）

■ ドテン制限
  - エントリーから15分未満は無視（一時的逆行）
  - 同一ペアで1時間以内のドテンは禁止（決済のみ）

■ 時刻基準
  - ドテンインターバル判定: 比較基準（UTC）
  - ポジション保持時間: 比較基準（UTC）
  - 金曜クローズ判定: JST 22:00
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loguru import logger

from broker.mt5_broker import MT5Broker
from config.settings import get_trading_config
from core.notifier import DiscordNotifier, AlertLevel
from core.time_manager import (
    now_utc,
    elapsed_seconds,
    is_friday_close_window,
    get_session,
    format_jst,
)


@dataclass
class ManagedPosition:
    """管理中のポジション情報。"""
    ticket: int
    pair: str
    direction: str
    volume: float
    open_price: float
    sl_price: float
    tp_price: float
    open_time_utc: datetime
    atr_at_entry: float
    # スケールアウト管理
    step1_done: bool = False
    step2_done: bool = False
    trailing_active: bool = False
    trailing_high: float = 0.0  # long の場合の最高値
    trailing_low: float = float("inf")  # short の場合の最安値


class PositionManager:
    """ポジション管理・エグジット戦略の実行。"""

    def __init__(self, broker: MT5Broker, notifier: DiscordNotifier):
        self._broker = broker
        self._notifier = notifier
        self._positions: dict[int, ManagedPosition] = {}  # ticket -> ManagedPosition
        # ドテンインターバル管理（比較基準: UTC）
        self._last_doten_time: dict[str, datetime] = {}

    @property
    def positions(self) -> dict[int, ManagedPosition]:
        return self._positions

    def register_position(self, pos: ManagedPosition) -> None:
        """新規ポジションを管理対象に登録する。"""
        self._positions[pos.ticket] = pos
        logger.info(
            f"Position registered: {pos.pair} {pos.direction} "
            f"ticket={pos.ticket} vol={pos.volume}"
        )

    def unregister_position(self, ticket: int) -> ManagedPosition | None:
        """ポジションを管理対象から除外する。"""
        return self._positions.pop(ticket, None)

    # ── ドテンインターバル判定 ──────────────────
    def check_doten_allowed(self, pair: str) -> bool:
        """同一ペアで1時間以内のドテンを禁止する。比較基準（UTC）で判定。"""
        config = get_trading_config()
        interval = config["risk"]["doten_interval_seconds"]
        last = self._last_doten_time.get(pair)
        if last is None:
            return True
        return elapsed_seconds(last) >= interval

    def check_entry_age(self, ticket: int, min_minutes: float = 15) -> bool:
        """エントリーから min_minutes 分以上経過しているか。"""
        pos = self._positions.get(ticket)
        if pos is None:
            return False
        return elapsed_seconds(pos.open_time_utc) >= min_minutes * 60

    # ── ドテン高速連続発注 ──────────────────────
    async def execute_doten(
        self,
        pair: str,
        ticket: int,
        direction: str,
        volume: float,
        sl_price: float,
        tp_price: float,
        atr: float,
    ) -> bool:
        """
        ドテン高速連続発注。
        決済→確認→新規の順序保証（asyncio.gather は使わない）。
        """
        # 決済
        close_ok = await self._broker.close_position_async(ticket)
        if not close_ok:
            await self._notifier.send(
                f"[DOTEN FAIL] {pair} 決済失敗 ticket={ticket}",
                AlertLevel.CRITICAL,
            )
            return False

        self.unregister_position(ticket)

        # 新規建て
        open_ok, new_ticket = await self._broker.open_position_async(
            pair, direction, volume, sl_price, tp_price,
        )
        if not open_ok:
            await self._notifier.send(
                f"[DOTEN FAIL] {pair} 新規建て失敗（決済済み・ノーポジ）",
                AlertLevel.CRITICAL,
            )
            return False

        # 新ポジション登録
        self.register_position(ManagedPosition(
            ticket=new_ticket,
            pair=pair,
            direction=direction,
            volume=volume,
            open_price=0.0,  # MT5から後で取得
            sl_price=sl_price,
            tp_price=tp_price,
            open_time_utc=now_utc(),
            atr_at_entry=atr,
        ))

        self._last_doten_time[pair] = now_utc()
        logger.info(f"Doten executed: {pair} → {direction} ticket={new_ticket}")
        return True

    # ── 毎分ポジション監視 ────────────────────
    async def monitor_positions(self) -> None:
        """
        毎分実行: 全ポジションのスケールアウト・トレーリング・時間Exit をチェック。
        """
        config = get_trading_config()
        risk = config["risk"]

        for ticket, pos in list(self._positions.items()):
            current_price = self._get_current_price(pos.pair, pos.direction)
            if current_price is None:
                continue

            # 優先1: Calendar Veto 強制クローズは Orchestrator 側で処理

            # 優先4: スケールアウト Step1
            if not pos.step1_done:
                await self._check_scale_out_step1(pos, current_price, risk)

            # 優先5: スケールアウト Step2
            if pos.step1_done and not pos.step2_done:
                await self._check_scale_out_step2(pos, current_price, risk)

            # 優先6: トレーリングストップ
            if pos.trailing_active:
                await self._update_trailing_stop(pos, current_price, risk)

            # 優先7: 時間ベース強制クローズ
            await self._check_time_exit(pos)

    async def _check_scale_out_step1(
        self, pos: ManagedPosition, current_price: float, risk: dict
    ) -> None:
        """スケールアウト Step1: +1.0×ATR → 50%利確 + SL→BE"""
        target = pos.atr_at_entry * 1.0
        if pos.direction == "long":
            pnl = current_price - pos.open_price
        else:
            pnl = pos.open_price - current_price

        if pnl >= target:
            close_ratio = risk.get("scale_out_step1_ratio", 0.50)
            close_vol = round(pos.volume * close_ratio, 2)
            if close_vol < 0.01:
                return

            ok = await self._broker.partial_close_async(pos.ticket, close_vol)
            if ok:
                pos.step1_done = True
                pos.volume = round(pos.volume - close_vol, 2)
                # SL → BE（建値）
                await self._broker.modify_sl_tp_async(pos.ticket, sl=pos.open_price)
                pos.sl_price = pos.open_price
                logger.info(
                    f"Scale-out Step1: {pos.pair} ticket={pos.ticket} "
                    f"closed {close_vol} lots, SL→BE"
                )

    async def _check_scale_out_step2(
        self, pos: ManagedPosition, current_price: float, risk: dict
    ) -> None:
        """スケールアウト Step2: +1.8×ATR → 25%利確 + トレーリング移行"""
        target = pos.atr_at_entry * 1.8
        if pos.direction == "long":
            pnl = current_price - pos.open_price
        else:
            pnl = pos.open_price - current_price

        if pnl >= target:
            close_ratio = risk.get("scale_out_step2_ratio", 0.25)
            original_volume = pos.volume / (1 - risk.get("scale_out_step1_ratio", 0.50))
            close_vol = round(original_volume * close_ratio, 2)
            if close_vol < 0.01:
                return

            close_vol = min(close_vol, pos.volume - 0.01)
            if close_vol < 0.01:
                return

            ok = await self._broker.partial_close_async(pos.ticket, close_vol)
            if ok:
                pos.step2_done = True
                pos.volume = round(pos.volume - close_vol, 2)
                pos.trailing_active = True
                pos.trailing_high = current_price if pos.direction == "long" else 0.0
                pos.trailing_low = current_price if pos.direction == "short" else float("inf")
                logger.info(
                    f"Scale-out Step2: {pos.pair} ticket={pos.ticket} "
                    f"closed {close_vol} lots, trailing activated"
                )

    async def _update_trailing_stop(
        self, pos: ManagedPosition, current_price: float, risk: dict
    ) -> None:
        """トレーリングストップ: 高値更新ごとに 0.8×ATR 追従。"""
        trail_dist = pos.atr_at_entry * risk.get("trailing_atr_multiplier", 0.8)

        if pos.direction == "long":
            if current_price > pos.trailing_high:
                pos.trailing_high = current_price
                new_sl = current_price - trail_dist
                if new_sl > pos.sl_price:
                    await self._broker.modify_sl_tp_async(pos.ticket, sl=round(new_sl, 5))
                    pos.sl_price = new_sl
        else:
            if current_price < pos.trailing_low:
                pos.trailing_low = current_price
                new_sl = current_price + trail_dist
                if new_sl < pos.sl_price:
                    await self._broker.modify_sl_tp_async(pos.ticket, sl=round(new_sl, 5))
                    pos.sl_price = new_sl

    async def _check_time_exit(self, pos: ManagedPosition) -> None:
        """時間ベース強制クローズ（4h超過 / 金曜22:00 / セッション跨ぎ）。"""
        now = now_utc()

        # 4時間超過
        if elapsed_seconds(pos.open_time_utc) > 4 * 3600:
            await self._force_close(pos, "time_exit_4h")
            return

        # 金曜22:00 JST
        if is_friday_close_window(now):
            await self._force_close(pos, "time_exit_friday")
            return

        # セッション跨ぎ: エントリー時と現在でセッションが異なる場合
        entry_session = get_session(pos.open_time_utc)
        current_session = get_session(now)
        if entry_session != current_session and entry_session != "other":
            await self._force_close(pos, f"session_cross_{entry_session}_to_{current_session}")
            return

    async def _force_close(self, pos: ManagedPosition, reason: str) -> None:
        """強制クローズ実行。"""
        ok = await self._broker.close_position_async(pos.ticket)
        if ok:
            self.unregister_position(pos.ticket)
            logger.info(f"Force close: {pos.pair} ticket={pos.ticket} reason={reason}")
            await self._notifier.send(
                f"ポジション強制クローズ: {pos.pair} {pos.direction} "
                f"ticket={pos.ticket} 理由={reason}",
                AlertLevel.ALERT,
            )
        else:
            logger.error(f"Force close failed: {pos.pair} ticket={pos.ticket}")

    async def close_all_positions(self, reason: str = "system") -> int:
        """全ポジション強制クローズ。Calendar Veto 等で使用。"""
        closed = 0
        for ticket, pos in list(self._positions.items()):
            ok = await self._broker.close_position_async(ticket)
            if ok:
                self.unregister_position(ticket)
                closed += 1
        if closed > 0:
            logger.info(f"All positions closed: {closed} positions, reason={reason}")
        return closed

    def _get_current_price(self, pair: str, direction: str) -> float | None:
        """現在価格を取得する（long→bid, short→ask）。"""
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(pair)
            if tick is None:
                return None
            return tick.bid if direction == "long" else tick.ask
        except Exception:
            return None
