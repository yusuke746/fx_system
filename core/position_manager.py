"""
ポジション管理・エグジット戦略モジュール

■ エグジット優先順位（ダイナミック・エグジット版）
  1. Calendar Veto 強制クローズ（最高優先）
  2. ATR動的SL到達（MT5 OCO注文）
  3. time_decay（時間切れ撤退）
  4. structural TP（構造目標到達）
  5. trailing stop（利益追従）
  6. 金曜クローズ（安全弁）

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
import sqlite3

from loguru import logger

from broker.mt5_broker import MT5Broker
from config.settings import get_trading_config
from core.database import close_trade
from core.notifier import DiscordNotifier, AlertLevel
from core.time_manager import (
    now_utc,
    elapsed_seconds,
    elapsed_minutes,
    is_friday_close_window,
)


@dataclass
class MarketContext:
    """tv_mcp_ea から5分毎に受信する市場構造データ。"""
    nearest_resistance: float = 0.0
    nearest_support: float = 0.0
    resistance_strength: int = 0
    support_strength: int = 0
    swing_high: float = 0.0
    swing_low: float = 0.0
    htf_bias: str = "neutral"  # "long" | "short" | "neutral"
    ema20_1h: float = 0.0
    ema50_1h: float = 0.0
    current_atr: float = 0.0
    updated_at: datetime | None = None


@dataclass
class ManagedPosition:
    """管理中のポジション情報。"""
    trade_id: int | None
    ticket: int
    pair: str
    direction: str
    volume: float
    open_price: float
    sl_price: float
    tp_price: float
    target_tp_price: float
    target_tp_pips: float
    open_time_utc: datetime
    atr_at_entry: float
    # 最新のLightGBM予測を同一ペアの新着シグナルで上書きする。
    prob_up: float = 0.0
    prob_flat: float = 0.0
    prob_down: float = 0.0
    last_prediction_at_utc: datetime | None = None
    trailing_active: bool = True
    trailing_high: float = 0.0  # long の場合の最高値
    trailing_low: float = float("inf")  # short の場合の最安値
    last_trailing_update_utc: datetime | None = None
    close_pending_since: datetime | None = None
    # tv_mcp_ea からの市場コンテキスト（5分毎更新）
    market_ctx: MarketContext | None = None
    # MCP EA 経由のポジションかどうか
    is_mcp: bool = False
    # エントリー時の HTF バイアス（トレンド反転検知用）
    entry_htf_bias: str = ""
    # HTF バイアス反転の連続検出カウンタ（ちらつきノイズ除去用）
    bias_reversal_streak: int = 0


class PositionManager:
    """ポジション管理・エグジット戦略の実行。"""

    def __init__(self, broker: MT5Broker, notifier: DiscordNotifier, db_conn: sqlite3.Connection | None = None):
        self._broker = broker
        self._notifier = notifier
        self._db_conn = db_conn
        self._positions: dict[int, ManagedPosition] = {}  # ticket -> ManagedPosition
        # ドテンインターバル管理（比較基準: UTC）
        self._last_doten_time: dict[str, datetime] = {}

    @property
    def positions(self) -> dict[int, ManagedPosition]:
        return self._positions

    def register_position(self, pos: ManagedPosition) -> None:
        """新規ポジションを管理対象に登録する。"""
        # トレーリング起点はエントリー価格で初期化しておく。
        if pos.direction == "long" and pos.trailing_high <= 0:
            pos.trailing_high = pos.open_price
        if pos.direction == "short" and pos.trailing_low == float("inf"):
            pos.trailing_low = pos.open_price

        self._positions[pos.ticket] = pos
        logger.info(
            f"Position registered: {pos.pair} {pos.direction} "
            f"ticket={pos.ticket} vol={pos.volume}"
        )

    def update_market_context(self, pair: str, ctx_data: dict) -> None:
        """
        tv_mcp_ea からの Market Context で保有ポジションを更新する。

        対象ペアの全ポジションに最新の S/R・スイング・EMA 情報を反映し、
        構造的 TP 価格を動的に更新する。
        """
        ctx = MarketContext(
            nearest_resistance=float(ctx_data.get("nearest_resistance", 0)),
            nearest_support=float(ctx_data.get("nearest_support", 0)),
            resistance_strength=int(ctx_data.get("resistance_strength", 0)),
            support_strength=int(ctx_data.get("support_strength", 0)),
            swing_high=float(ctx_data.get("swing_high", 0)),
            swing_low=float(ctx_data.get("swing_low", 0)),
            htf_bias=str(ctx_data.get("htf_bias", "neutral")),
            ema20_1h=float(ctx_data.get("ema20_1h", 0)),
            ema50_1h=float(ctx_data.get("ema50_1h", 0)),
            current_atr=float(ctx_data.get("atr", 0)),
            updated_at=now_utc(),
        )

        updated = 0
        for pos in self._positions.values():
            if pos.pair != pair:
                continue
            pos.market_ctx = ctx

            # 構造的 TP を動的更新: 利益方向の最寄り S/R を目標にする
            structural_tp = self._calc_structural_tp(pos, ctx)
            if structural_tp > 0:
                old_tp = pos.target_tp_price
                pos.target_tp_price = structural_tp
                if abs(old_tp - structural_tp) > 1e-6:
                    logger.info(
                        f"Structural TP updated: {pos.pair} ticket={pos.ticket} "
                        f"{old_tp:.5f} → {structural_tp:.5f}"
                    )
            updated += 1

        if updated > 0:
            logger.debug(f"Market context applied to {updated} position(s) for {pair}")

    def _calc_structural_tp(self, pos: ManagedPosition, ctx: MarketContext) -> float:
        """
        構造的 TP 候補を選出する。

        候補の優先順位:
        1. 利益方向の最寄り S/R レベル (タッチ数 ≥ 2 で信頼性が高い)
        2. 直近スイング高値/安値
        3. フォールバック: 0 (ATR ベースの TP をそのまま使う)

        S/R レベルに ATR の 20% のバッファを入れて、
        レベルの手前で確実に利確できるようにする。
        """
        atr = ctx.current_atr if ctx.current_atr > 0 else pos.atr_at_entry
        buffer = atr * 0.2  # レベル手前にバッファ

        if pos.direction == "long":
            # ロング: 上方向の壁を探す
            candidates = []
            if ctx.nearest_resistance > pos.open_price:
                candidates.append(ctx.nearest_resistance - buffer)
            if ctx.swing_high > pos.open_price:
                candidates.append(ctx.swing_high - buffer)
            # 最も近い (= 届きやすい) 候補を選択
            valid = [c for c in candidates if c > pos.open_price + atr * 0.5]
            return min(valid) if valid else 0.0
        else:
            # ショート: 下方向の壁を探す
            candidates = []
            if ctx.nearest_support > 0 and ctx.nearest_support < pos.open_price:
                candidates.append(ctx.nearest_support + buffer)
            if ctx.swing_low > 0 and ctx.swing_low < pos.open_price:
                candidates.append(ctx.swing_low + buffer)
            valid = [c for c in candidates if c < pos.open_price - atr * 0.5]
            return max(valid) if valid else 0.0

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
    def update_pair_prediction(
        self,
        pair: str,
        prob_up: float,
        prob_flat: float,
        prob_down: float,
    ) -> None:
        """同一ペアの最新予測を保有ポジションへ反映する。"""
        updated_at = now_utc()
        for pos in self._positions.values():
            if pos.pair != pair:
                continue
            pos.prob_up = prob_up
            pos.prob_flat = prob_flat
            pos.prob_down = prob_down
            pos.last_prediction_at_utc = updated_at

    async def execute_doten(
        self,
        pair: str,
        ticket: int,
        direction: str,
        volume: float,
        sl_pips: float,
        tp_pips: float,
        signal_price: float,
        reason: str = "doten",
    ) -> tuple[bool, int | None, float, float]:
        """
        ドテン高速連続発注。
        決済→確認→新規の順序保証（asyncio.gather は使わない）。

        Returns:
            (success, new_ticket or None, actual_sl_price, actual_tp_price)
        """
        pos = self._positions.get(ticket)

        # 決済
        close_ok = await self._broker.close_position_async(ticket)
        if not close_ok:
            await self._notifier.send(
                f"[DOTEN FAIL] {pair} 決済失敗 ticket={ticket}",
                AlertLevel.CRITICAL,
            )
            return False, None, 0.0, 0.0

        if pos is not None:
            self._finalize_closed_position(pos, signal_price, reason)
        else:
            self.unregister_position(ticket)

        # 新規建て（SL/TPは実tick価格から計算）
        open_ok, new_ticket, actual_sl, actual_tp = await self._broker.open_position_async(
            pair, direction, volume, sl_pips, tp_pips,
        )
        if not open_ok:
            await self._notifier.send(
                f"[DOTEN FAIL] {pair} 新規建て失敗（決済済み・ノーポジ）",
                AlertLevel.CRITICAL,
            )
            return False, None, 0.0, 0.0

        self._last_doten_time[pair] = now_utc()
        logger.info(f"Doten executed: {pair} → {direction} ticket={new_ticket}")
        return True, new_ticket, actual_sl, actual_tp

    # ── 毎分ポジション監視 ────────────────────
    async def monitor_positions(self) -> None:
        """
        毎分実行: 全ポジションのダイナミック・エグジットをチェックする。

        エグジット優先順位:
            1. Calendar Veto 強制クローズ（Orchestrator 側で処理）
            2. ATR 動的 SL 到達（MT5 OCO 注文）
            3. HTF バイアス反転（MCP ポジション: EMA クロスが反転した場合）
            4. time_decay（時間切れ撤退）
            5. structural TP（S/R・スイング構造目標到達）
            6. 構造的トレーリング（S/R ベース or ATR トレーリング）
            7. 金曜クローズ（安全弁）
        """
        config = get_trading_config()
        risk = config["risk"]

        await self._sync_closed_positions_with_broker()

        for ticket, pos in list(self._positions.items()):
            # MT5側でクローズ検知済み（履歴反映待ち）の間は、
            # 追加のSL更新や強制クローズを行わない。
            if pos.close_pending_since is not None:
                continue

            current_price = self._get_current_price(pos.pair, pos.direction)
            if current_price is None:
                continue

            # 優先1: Calendar Veto強制クローズ（Orchestrator側で処理）
            # 優先2: ATR動的SL到達（MT5 OCO注文）

            # 優先3: HTFバイアス反転（MCP + market_ctx ありの場合のみ）
            # 複数回の連続検出で初めて発火（ちらつきノイズ除去）
            reversal_now = self._should_exit_bias_reversal(pos)
            if reversal_now:
                pos.bias_reversal_streak += 1
                confirm_count = int(risk.get("bias_reversal_confirm_count", 2))
                if pos.bias_reversal_streak >= confirm_count:
                    await self._force_close(pos, "bias_reversal", current_price)
                    continue
                else:
                    logger.info(
                        f"Bias reversal pending: {pos.pair} ticket={pos.ticket} "
                        f"streak={pos.bias_reversal_streak}/{confirm_count}"
                    )
            else:
                pos.bias_reversal_streak = 0

            # 優先4: time_decay（時間切れ撤退）
            if self._should_exit_time_decay(pos, current_price, risk):
                await self._force_close(pos, "time_decay", current_price)
                continue

            # 優先5: structural TP（構造目標到達）
            if self._has_hit_structural_target(pos, current_price):
                await self._force_close(pos, "structural_tp", current_price)
                continue

            # 優先6: 構造的トレーリング or ATR トレーリング
            if pos.trailing_active:
                await self._update_structural_trailing(pos, current_price, risk)

            # 優先7: 金曜クローズ（安全弁）
            if is_friday_close_window(now_utc()):
                await self._force_close(pos, "time_exit", current_price)
                continue

    def _should_exit_bias_reversal(self, pos: ManagedPosition) -> bool:
        """
        HTF バイアスがエントリー方向と逆転した場合にエグジットする。

        条件:
        - MCP ポジションかつ market_ctx が5分以内に更新されていること
        - エントリー時の HTF バイアスが記録されていること
        - 現在の HTF バイアスがエントリー方向と反対に反転していること
        - 最低 15 分保持（フラッシュ反転を除外）
        """
        if not pos.is_mcp or pos.market_ctx is None:
            return False
        if pos.market_ctx.updated_at is None:
            return False
        # コンテキストが10分以上古い場合は無効（tv_mcp_ea が停止している可能性）
        if elapsed_minutes(pos.market_ctx.updated_at) > 10:
            return False
        # 最低保持時間（config で調整可能）
        config = get_trading_config()
        min_minutes = float(config.get("risk", {}).get("bias_reversal_min_minutes", 30))
        if elapsed_minutes(pos.open_time_utc) < min_minutes:
            return False
        # エントリー時のバイアスが未記録なら判定不能
        if not pos.entry_htf_bias or pos.entry_htf_bias == "neutral":
            return False

        current_bias = pos.market_ctx.htf_bias
        if current_bias == "neutral":
            return False

        # ロングエントリー → ショートバイアスに反転 = 撤退
        if pos.direction == "long" and pos.entry_htf_bias == "long" and current_bias == "short":
            logger.info(
                f"Bias reversal detected: {pos.pair} ticket={pos.ticket} "
                f"entry_bias=long → current_bias=short"
            )
            return True
        # ショートエントリー → ロングバイアスに反転 = 撤退
        if pos.direction == "short" and pos.entry_htf_bias == "short" and current_bias == "long":
            logger.info(
                f"Bias reversal detected: {pos.pair} ticket={pos.ticket} "
                f"entry_bias=short → current_bias=long"
            )
            return True

        return False

    def _should_exit_time_decay(
        self,
        pos: ManagedPosition,
        current_price: float,
        risk: dict,
    ) -> bool:
        """一定時間経っても含み益が伸びなければダマシとして撤退する。"""
        elapsed = elapsed_minutes(pos.open_time_utc)
        min_minutes = float(risk.get("time_decay_minutes", 60))
        if elapsed < min_minutes:
            return False

        # time_decay は「伸びない負け/建値付近の停滞」を切る用途に限定する。
        # ATR × hold_threshold 以上の含み益があれば保持（建値前後の停滞は切る対象）。
        current_profit = self._price_move_in_favor(pos, current_price)
        if bool(risk.get("time_decay_only_on_loser", True)):
            hold_atr = float(risk.get("time_decay_hold_atr_threshold", 0.15))
            if current_profit >= pos.atr_at_entry * hold_atr:
                return False

        min_profit_atr = float(risk.get("time_decay_min_profit_atr", 0.5))
        required_profit = pos.atr_at_entry * min_profit_atr
        return current_profit < required_profit

    def _has_hit_structural_target(self, pos: ManagedPosition, current_price: float) -> bool:
        """Pine由来の構造TP価格に到達したかを判定する。"""
        if pos.target_tp_price <= 0:
            return False
        if pos.direction == "long":
            return current_price >= pos.target_tp_price
        return current_price <= pos.target_tp_price

    async def _update_structural_trailing(
        self, pos: ManagedPosition, current_price: float, risk: dict
    ) -> None:
        """
        構造的トレーリング: S/R レベルとスイング高安を使って SL を構造的に引き上げる。

        Market Context がある場合:
            - ロング: 直近サポート or スイング安値の手前に SL を移動
            - ショート: 直近レジスタンス or スイング高値の向こう側に SL を移動
        Market Context がない場合:
            - フォールバック: 従来の ATR×0.8 固定距離トレーリング
        """
        cooldown_seconds = float(risk.get("trailing_update_cooldown_seconds", 30))
        min_step_pips = float(risk.get("trailing_min_step_pips", 2.0))
        pip_unit = 0.01 if pos.pair in ("USDJPY", "GBPJPY") else (0.10 if pos.pair in ("GOLD", "XAUUSD") else 0.0001)
        min_step_price = max(0.0, min_step_pips * pip_unit)

        if pos.last_trailing_update_utc is not None:
            if elapsed_seconds(pos.last_trailing_update_utc) < cooldown_seconds:
                return

        ctx = pos.market_ctx
        atr = ctx.current_atr if (ctx and ctx.current_atr > 0) else pos.atr_at_entry
        buffer = atr * 0.3  # S/R レベルの少し外側にSLを置く

        # エントリーから最低前進量に達するまでトレーリングを発動しない
        # （エントリー直後の immediate tighten 防止）
        trailing_start_atr_min = float(risk.get("trailing_start_atr_min", 0.3))
        forward_move = (
            (current_price - pos.open_price) if pos.direction == "long"
            else (pos.open_price - current_price)
        )
        if forward_move < atr * trailing_start_atr_min:
            return

        if pos.direction == "long":
            if current_price > pos.trailing_high:
                pos.trailing_high = current_price

            # 構造的 SL 候補を収集
            structural_sl = 0.0
            if ctx and ctx.updated_at and elapsed_minutes(ctx.updated_at) <= 10:
                candidates = []
                # サポートライン (現在価格より下) の上に SL
                if ctx.nearest_support > 0 and ctx.nearest_support < current_price:
                    candidates.append(ctx.nearest_support - buffer)
                # 直近スイング安値の下に SL
                if ctx.swing_low > 0 and ctx.swing_low < current_price:
                    candidates.append(ctx.swing_low - buffer)
                # エントリー価格以上の候補のみ（利益保護）
                valid = [c for c in candidates if c > pos.open_price]
                if valid:
                    structural_sl = max(valid)  # 最も高い（利益を守る）候補

            # ATR トレーリング（フォールバック）
            atr_trail_sl = pos.trailing_high - atr * risk.get("trailing_atr_multiplier", 0.8)

            # 構造的 SL と ATR トレーリングの大きい方を採用
            new_sl = max(structural_sl, atr_trail_sl)

            if new_sl > pos.sl_price + min_step_price:
                ok = await self._broker.modify_sl_tp_async(pos.ticket, sl=round(new_sl, 5))
                if ok:
                    trail_type = "structural" if structural_sl >= atr_trail_sl else "atr"
                    pos.sl_price = new_sl
                    pos.last_trailing_update_utc = now_utc()
                    logger.info(
                        f"Trailing SL updated ({trail_type}): {pos.pair} "
                        f"ticket={pos.ticket} new_sl={round(new_sl, 5)}"
                    )
                else:
                    logger.error(f"Failed to update trailing SL: {pos.pair} ticket={pos.ticket}")
                    await self._notifier.send(
                        f"⚠️ Trailing SL更新失敗: {pos.pair} ticket={pos.ticket}",
                        AlertLevel.WARNING,
                    )
        else:  # short
            if current_price < pos.trailing_low:
                pos.trailing_low = current_price

            structural_sl = float("inf")
            if ctx and ctx.updated_at and elapsed_minutes(ctx.updated_at) <= 10:
                candidates = []
                if ctx.nearest_resistance > 0 and ctx.nearest_resistance > current_price:
                    candidates.append(ctx.nearest_resistance + buffer)
                if ctx.swing_high > 0 and ctx.swing_high > current_price:
                    candidates.append(ctx.swing_high + buffer)
                valid = [c for c in candidates if c < pos.open_price]
                if valid:
                    structural_sl = min(valid)

            atr_trail_sl = pos.trailing_low + atr * risk.get("trailing_atr_multiplier", 0.8)

            new_sl = min(structural_sl, atr_trail_sl)

            if new_sl < pos.sl_price - min_step_price:
                ok = await self._broker.modify_sl_tp_async(pos.ticket, sl=round(new_sl, 5))
                if ok:
                    trail_type = "structural" if structural_sl <= atr_trail_sl else "atr"
                    pos.sl_price = new_sl
                    pos.last_trailing_update_utc = now_utc()
                    logger.info(
                        f"Trailing SL updated ({trail_type}): {pos.pair} "
                        f"ticket={pos.ticket} new_sl={round(new_sl, 5)}"
                    )
                else:
                    logger.error(f"Failed to update trailing SL: {pos.pair} ticket={pos.ticket}")
                    await self._notifier.send(
                        f"⚠️ Trailing SL更新失敗: {pos.pair} ticket={pos.ticket}",
                        AlertLevel.WARNING,
                    )

    async def _force_close(self, pos: ManagedPosition, reason: str, close_price: float | None = None) -> None:
        """強制クローズ実行。"""
        if close_price is None:
            close_price = self._get_current_price(pos.pair, pos.direction)

        ok = await self._broker.close_position_async(pos.ticket)
        if ok:
            self._finalize_closed_position(pos, close_price, reason)
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
            close_price = self._get_current_price(pos.pair, pos.direction)
            ok = await self._broker.close_position_async(ticket)
            if ok:
                self._finalize_closed_position(pos, close_price, reason)
                closed += 1
        if closed > 0:
            logger.info(f"All positions closed: {closed} positions, reason={reason}")
        return closed

    async def _sync_closed_positions_with_broker(self) -> None:
        """MT5側で先にクローズされたポジションを検知し、ローカル状態を同期する。"""
        broker_positions = self._broker.get_positions()
        open_tickets = {int(pos["ticket"]) for pos in broker_positions}

        for ticket, pos in list(self._positions.items()):
            if ticket in open_tickets:
                # まだ開いている → close_pending_since をリセット
                pos.close_pending_since = None
                continue

            # MT5側で決済済み
            closed_info = self._broker.get_recent_closed_position_info(
                ticket, lookback_hours=72
            )

            if closed_info is None:
                if pos.close_pending_since is None:
                    # 初回検知: タイムスタンプを記録して次サイクルへ
                    pos.close_pending_since = now_utc()
                    logger.debug(
                        f"Closed position detected but history not found yet: "
                        f"ticket={ticket}. Will retry next cycle."
                    )
                    continue

                elapsed = elapsed_minutes(pos.close_pending_since)
                if elapsed < 5:
                    # 5分未満: まだ待つ
                    logger.debug(
                        f"Waiting for history: ticket={ticket}, "
                        f"elapsed={elapsed:.1f}min"
                    )
                    continue

                # 5分以上経過しても取得できない: 諦めてunregister
                logger.warning(
                    f"Closed position history not found for ticket={ticket} "
                    f"after {elapsed:.1f}min; unregister now and retry "
                    f"history backfill in background."
                )
                self.unregister_position(ticket)
                continue

            # 履歴取得成功
            reason = self._infer_external_exit_reason(pos, closed_info["close_price"])
            self._finalize_closed_position(
                pos,
                closed_info.get("close_price"),
                reason,
                pnl_jpy_override=closed_info.get("profit"),
            )

        self._retry_unrecorded_closed_trades(open_tickets)

    def _retry_unrecorded_closed_trades(self, open_tickets: set[int]) -> None:
        """
        DB上で未決済のまま残ったトレードを再照合し、
        ブローカー履歴が取得できたものからP&Lをバックフィルする。
        """
        if self._db_conn is None:
            return

        config = get_trading_config()
        risk = config.get("risk", {})
        lookback_hours = int(risk.get("close_history_retry_lookback_hours", 168))
        scan_limit = int(risk.get("close_history_retry_scan_limit", 100))

        try:
            rows = self._db_conn.execute(
                """SELECT
                       id,
                       pair,
                       direction,
                       open_time,
                       open_price,
                       volume,
                       sl_price,
                       tp_price,
                       mt5_ticket
                   FROM trades
                   WHERE close_time IS NULL
                     AND mt5_ticket IS NOT NULL
                   ORDER BY open_time DESC
                   LIMIT ?""",
                (scan_limit,),
            ).fetchall()
        except Exception as e:
            logger.error(f"Failed to scan unrecorded closed trades: {e}")
            return

        recovered = 0
        for row in rows:
            ticket = int(row["mt5_ticket"])
            if ticket in open_tickets:
                continue

            closed_info = self._broker.get_recent_closed_position_info(
                ticket, lookback_hours=lookback_hours
            )
            if closed_info is None:
                continue

            close_price = closed_info.get("close_price")
            if close_price is None:
                logger.warning(
                    f"Close history found but close_price is missing: ticket={ticket}"
                )
                continue

            try:
                open_time_utc = datetime.fromisoformat(row["open_time"])
            except Exception:
                open_time_utc = now_utc()

            pos = ManagedPosition(
                trade_id=int(row["id"]),
                ticket=ticket,
                pair=row["pair"],
                direction=row["direction"],
                volume=float(row["volume"]),
                open_price=float(row["open_price"]),
                sl_price=float(row["sl_price"] or 0.0),
                tp_price=float(row["tp_price"] or 0.0),
                target_tp_price=float(row["tp_price"] or 0.0),
                target_tp_pips=0.0,
                open_time_utc=open_time_utc,
                atr_at_entry=0.0,
            )

            reason = self._infer_external_exit_reason(pos, float(close_price))
            self._finalize_closed_position(
                pos,
                float(close_price),
                reason,
                pnl_jpy_override=closed_info.get("profit"),
            )
            recovered += 1

        if recovered > 0:
            logger.info(
                f"Recovered {recovered} unrecorded closed trade(s) from broker history."
            )

    def _infer_external_exit_reason(self, pos: ManagedPosition, close_price: float) -> str:
        """MT5側で先に決済された場合の理由を価格位置から推定する。"""
        tolerance = 1e-5 if pos.pair not in ("USDJPY", "GBPJPY") else 1e-3

        if self._has_hit_structural_target(pos, close_price):
            return "structural_tp"

        if pos.direction == "long" and close_price <= pos.sl_price + tolerance:
            return "trailing" if pos.sl_price >= pos.open_price else "atr_sl"
        if pos.direction == "short" and close_price >= pos.sl_price - tolerance:
            return "trailing" if pos.sl_price <= pos.open_price else "atr_sl"

        return "time_exit"

    def _finalize_closed_position(
        self,
        pos: ManagedPosition,
        close_price: float | None,
        reason: str,
        pnl_jpy_override: float | None = None,
    ) -> None:
        """クローズ後のローカル状態とDB記録をまとめて更新する。"""
        self.unregister_position(pos.ticket)
        logger.info(f"Force close: {pos.pair} ticket={pos.ticket} reason={reason}")

        if self._db_conn is None or pos.trade_id is None or close_price is None:
            return

        pnl_pips, pnl_jpy = self._estimate_pnl(pos, close_price)
        if pnl_jpy_override is not None:
            pnl_jpy = round(float(pnl_jpy_override), 0)
        try:
            close_trade(
                self._db_conn,
                trade_id=pos.trade_id,
                close_price=close_price,
                pnl_pips=pnl_pips,
                pnl_jpy=pnl_jpy,
                exit_reason=reason,
            )
        except Exception as e:
            logger.error(f"Failed to persist trade close for ticket={pos.ticket}: {e}")

    def _price_move_in_favor(self, pos: ManagedPosition, current_price: float) -> float:
        """ポジション方向に有利な価格差を返す。"""
        if pos.direction == "long":
            return current_price - pos.open_price
        return pos.open_price - current_price

    def _estimate_pnl(self, pos: ManagedPosition, close_price: float) -> tuple[float, float]:
        """決済記録用の pips / 円損益を概算する。"""
        if pos.pair in ("USDJPY", "GBPJPY"):
            pip_unit = 0.01
            pip_value_per_mini_lot = 100
        elif pos.pair in ("GOLD", "XAUUSD"):
            pip_unit = 0.10
            pip_value_per_mini_lot = 100
        else:
            pip_unit = 0.0001
            pip_value_per_mini_lot = 110
        pnl_price = self._price_move_in_favor(pos, close_price)
        pnl_pips = pnl_price / pip_unit

        pnl_jpy = pnl_pips * pip_value_per_mini_lot * (pos.volume / 0.01)
        return round(pnl_pips, 1), round(pnl_jpy, 0)

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
