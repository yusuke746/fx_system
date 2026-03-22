"""
リスク管理モジュール

■ リスクパラメータ（設計書 確定値）
  MAX_RISK_PER_TRADE_PCT = 2.0%
  MAX_DAILY_DRAWDOWN_PCT = 10.0%
  MAX_TOTAL_EXPOSURE_PCT = 10.0%
  MAX_POSITIONS = 5
  MAX_USD_EXPOSURE = 4
  MAX_JPY_EXPOSURE = 2
"""

from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class RiskConfig:
    MAX_RISK_PER_TRADE_PCT: float = 2.0
    MAX_DAILY_DRAWDOWN_PCT: float = 10.0
    MAX_TOTAL_EXPOSURE_PCT: float = 10.0
    MAX_POSITIONS: int = 5
    MAX_USD_EXPOSURE: int = 4
    MAX_JPY_EXPOSURE: int = 2


def load_risk_config(trading_config: dict) -> RiskConfig:
    """config.json の risk セクションから RiskConfig を構築する。"""
    risk = trading_config.get("risk", {})
    return RiskConfig(
        MAX_RISK_PER_TRADE_PCT=risk.get("max_risk_per_trade_pct", 2.0),
        MAX_DAILY_DRAWDOWN_PCT=risk.get("max_daily_drawdown_pct", 10.0),
        MAX_TOTAL_EXPOSURE_PCT=risk.get("max_total_exposure_pct", 10.0),
        MAX_POSITIONS=risk.get("max_positions", 5),
        MAX_USD_EXPOSURE=risk.get("max_usd_exposure", 4),
        MAX_JPY_EXPOSURE=risk.get("max_jpy_exposure", 2),
    )


# ── 通貨エクスポージャーマッピング ───────────────
_USD_PAIRS = {"USDJPY", "EURUSD"}
_JPY_PAIRS = {"USDJPY", "GBPJPY"}

# 1pip あたりの価値（0.01 lot = mini lot 単位、円建て概算）
_PIP_VALUES = {
    "USDJPY": 100,   # 1pip / 0.01lot = 100円
    "EURUSD": 110,   # 1pip / 0.01lot ≈ 110円（レート変動あり）
    "GBPJPY": 100,   # 1pip / 0.01lot = 100円
}


def calc_lot_size(
    account_balance: float,
    sl_pips: float,
    pair: str,
    risk_config: RiskConfig,
) -> float:
    """
    1トレードの許容損失額からロットサイズを算出する。
    XMTrading KIWA極口座（commission=0）前提。

    例: 残高100万円 / SL=20pips / USDJPY
      → 許容損失 = 1,000,000 × 0.02 = 20,000円
      → lot = 20,000 / (20 × 100 × 100) = 0.10 lot
    """
    if sl_pips <= 0:
        logger.error(f"Invalid sl_pips={sl_pips}. Defaulting to minimum lot size 0.01")
        return 0.01

    max_loss_jpy = account_balance * (risk_config.MAX_RISK_PER_TRADE_PCT / 100)
    pip_value_per_mini_lot = _PIP_VALUES.get(pair, 100)
    lot = max_loss_jpy / (sl_pips * pip_value_per_mini_lot * 100)
    lot = round(max(0.01, min(lot, 10.0)), 2)
    logger.debug(
        f"Lot size calculation: balance={account_balance}, SL={sl_pips}pips, "
        f"pair={pair} → lot={lot}"
    )
    return lot


def check_exposure(
    open_positions: list[dict],
    new_pair: str,
    risk_config: RiskConfig,
    account_balance: float,
) -> tuple[bool, str]:
    """
    新規エントリー前に通貨エクスポージャーをチェックする。

    Returns:
        (is_allowed, reason)
    """
    # 全ポジション数チェック
    if len(open_positions) >= risk_config.MAX_POSITIONS:
        return False, f"MAX_POSITIONS({risk_config.MAX_POSITIONS})に達しています"

    # USD エクスポージャーチェック
    usd_count = sum(1 for p in open_positions if p["pair"] in _USD_PAIRS)
    if new_pair in _USD_PAIRS and usd_count >= risk_config.MAX_USD_EXPOSURE:
        return False, f"MAX_USD_EXPOSURE({risk_config.MAX_USD_EXPOSURE})に達しています"

    # JPY エクスポージャーチェック
    jpy_count = sum(1 for p in open_positions if p["pair"] in _JPY_PAIRS)
    if new_pair in _JPY_PAIRS and jpy_count >= risk_config.MAX_JPY_EXPOSURE:
        return False, f"MAX_JPY_EXPOSURE({risk_config.MAX_JPY_EXPOSURE})に達しています"

    # 証拠金合計チェック
    total_margin = sum(p.get("margin_used", 0) for p in open_positions)
    if account_balance > 0:
        exposure_pct = total_margin / account_balance * 100
        if exposure_pct >= risk_config.MAX_TOTAL_EXPOSURE_PCT:
            return False, f"MAX_TOTAL_EXPOSURE({risk_config.MAX_TOTAL_EXPOSURE_PCT}%)に達しています"

    return True, "OK"


def check_daily_drawdown(
    daily_pnl_jpy: float,
    account_balance: float,
    risk_config: RiskConfig,
) -> tuple[bool, float]:
    """
    当日の損失が MAX_DAILY_DRAWDOWN_PCT を超えたかチェックする。

    Returns:
        (is_allowed, drawdown_pct)
    """
    drawdown_pct = abs(min(daily_pnl_jpy, 0)) / account_balance * 100 if account_balance > 0 else 0
    allowed = drawdown_pct < risk_config.MAX_DAILY_DRAWDOWN_PCT
    if not allowed:
        logger.warning(
            f"Daily drawdown limit reached: {drawdown_pct:.1f}% >= "
            f"{risk_config.MAX_DAILY_DRAWDOWN_PCT}%"
        )
    return allowed, drawdown_pct


def calc_sl_tp_pips(
    atr: float,
    pair: str,
    trading_config: dict,
    ob_4h_distance_pips: float | None = None,
    tp_swing_pips: float | None = None,
    tp_fvg_pips: float | None = None,
) -> tuple[float, float]:
    """
    ATR と構造距離からSL/TP（pips）を計算する。

    - SL は従来どおり ATR ベース
    - TP は Pine から渡された3候補（tp_swing_pips / tp_fvg_pips / ob_4h_distance_pips）から選択
      条件: R:R 1.5以上の候補のうち最小値を採用（到達しやすいTPを優先）
      候補が0件の場合: SL × 1.5 をフォールバック

    Returns:
        (sl_pips, tp_pips)
    """
    risk = trading_config["risk"]
    sl_mult = risk["sl_multiplier"]
    sl_min = risk["sl_min_pips"]
    sl_max = risk["sl_max_pips"]
    sl_cap_atr_multiplier = float(risk.get("sl_cap_atr_multiplier", 1.0))

    # ATR → pips変換（USDJPY/GBPJPY は 0.01=1pip, EURUSD は 0.0001=1pip）
    if pair in ("USDJPY", "GBPJPY"):
        atr_pips = atr * 100
    else:
        atr_pips = atr * 10000

    if sl_cap_atr_multiplier > 0:
        sl_upper_cap = max(sl_min, atr_pips * sl_cap_atr_multiplier)
    else:
        sl_upper_cap = sl_max

    sl_pips = max(sl_min, min(atr_pips * sl_mult, sl_upper_cap))

    tp_min_rr = 1.5  # R:R最低保証（SLの1.5倍）
    tp_rr_floor = sl_pips * tp_min_rr

    candidates = []
    for val in [tp_swing_pips, tp_fvg_pips, ob_4h_distance_pips]:
        v = abs(float(val or 0.0))
        if v >= tp_rr_floor:
            candidates.append(v)

    if candidates:
        tp_candidate = min(candidates)
    else:
        logger.warning(
            f"No valid TP candidate for {pair} (all < R:R {tp_min_rr}). "
            f"Using SL×1.5 fallback ({sl_pips:.1f} pips base)"
        )
        tp_candidate = tp_rr_floor

    tp_cap = max(tp_rr_floor, sl_upper_cap * 3.0)
    tp_pips = round(min(tp_candidate, tp_cap), 1)

    return round(sl_pips, 1), tp_pips
