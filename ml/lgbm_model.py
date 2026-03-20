"""
LightGBM 特徴量エンジニアリング・推論モジュール

■ 35特徴量（MTF SMC v2.3 センサー拡張版）

    SMCフラグ(8): fvg_4h_zone_active, ob_4h_zone_active, liq_sweep_1h,
                                 liq_sweep_qualified, bos_1h, choch_1h,
                                 msb_15m_confirmed, mtf_confluence
    価格・ボラティリティ系(6): atr_14, atr_ratio, bb_width,
                                 close_vs_ema20_4h, close_vs_ema50_4h, high_low_range_15m
    トレンド・モメンタム補助(3): trend_direction, momentum_long, momentum_short
    モメンタム系(7): macd_histogram, macd_signal_cross, rsi_14, rsi_zone,
                                 stoch_k, stoch_d, momentum_3bar
  構造・パターン系(7): ob_4h_distance_pips, fvg_4h_fill_ratio,
                 liq_sweep_strength, prior_candle_body_ratio,
                 consecutive_same_dir, pivot_proximity, sweep_pending_bars
  リスク・ポジション系(4): open_positions_count, max_dd_24h,
                 calendar_risk_score, sentiment_score

■ 時刻基準
    - 全特徴量は保存基準（UTC）の signal_time に紐づけて管理
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

from config.settings import get_trading_config


# ── 特徴量名の定義（順番固定） ────────────────────
FEATURE_NAMES = [
    # SMCフラグ (8)
    "fvg_4h_zone_active",
    "ob_4h_zone_active",
    "liq_sweep_1h",
    "liq_sweep_qualified",       # [NEW v2.3]
    "bos_1h",
    "choch_1h",                  # [NEW v2.3]
    "msb_15m_confirmed",         # [NEW v2.3]
    "mtf_confluence",
    # 価格・ボラティリティ系 (6)
    "atr_14",
    "atr_ratio",
    "bb_width",
    "close_vs_ema20_4h",
    "close_vs_ema50_4h",
    "high_low_range_15m",
    # トレンド・モメンタム補助 (3)
    "trend_direction",
    "momentum_long",
    "momentum_short",
    # モメンタム系 (7)
    "macd_histogram",
    "macd_signal_cross",
    "rsi_14",
    "rsi_zone",
    "stoch_k",
    "stoch_d",
    "momentum_3bar",
    # 構造・パターン系 (7)
    "ob_4h_distance_pips",
    "fvg_4h_fill_ratio",
    "liq_sweep_strength",
    "prior_candle_body_ratio",
    "consecutive_same_dir",
    "pivot_proximity",
    "sweep_pending_bars",        # [NEW v2.3]
    # リスク・ポジション系 (4)
    "open_positions_count",
    "max_dd_24h",
    "calendar_risk_score",
    "sentiment_score",
]

assert len(FEATURE_NAMES) == 35, f"Expected 35 features, got {len(FEATURE_NAMES)}"

# LightGBM 学習パラメータ（設計書 確定値）
LGBM_PARAMS = {
    "max_depth": 4,
    "min_child_samples": 50,
    "reg_alpha": 0.2,
    "reg_lambda": 0.3,
    "n_estimators": 300,
    "objective": "multiclass",
    "num_class": 3,   # 上昇 / 横ばい / 下落
    "verbosity": -1,
    "class_weight": "balanced",
}


@dataclass
class PredictionResult:
    """LightGBM の推論結果。"""
    prob_up: float      # 上昇確率
    prob_flat: float    # 横ばい確率
    prob_down: float    # 下落確率

    @property
    def direction(self) -> str:
        """最大確率の方向を返す。"""
        if self.prob_up > self.prob_flat and self.prob_up > self.prob_down:
            return "long"
        if self.prob_down > self.prob_flat and self.prob_down > self.prob_up:
            return "short"
        return "flat"

    @property
    def max_prob(self) -> float:
        return max(self.prob_up, self.prob_flat, self.prob_down)

    @property
    def is_strong_signal(self) -> bool:
        """65% 以上の確率で方向が出ているかどうか。"""
        return self.max_prob >= 0.65

    @property
    def is_reverse_signal(self) -> bool:
        """ドテン判定用: 逆方向が65%以上か。"""
        return self.is_strong_signal and self.direction != "flat"


def build_features(
    smc_data: dict,
    market_data: dict,
    position_data: dict,
    sentiment_score: float = 0.0,
    calendar_risk_score: int = 0,
) -> np.ndarray:
    """
    35特徴量ベクトルを構築する。

    Args:
        smc_data: TradingView Webhook から受け取った SMC データ
        market_data: MT5 から取得した市場データ
        position_data: 現在のポジション情報
        sentiment_score: GPT-5.2 出力のセンチメントスコア
        calendar_risk_score: 経済指標リスクスコア (0/1/2)

    Returns:
        shape=(1, 35) の numpy 配列
    """
    features = [
        # SMC フラグ (8)
        int(smc_data.get("fvg_4h_zone_active", False)),
        int(smc_data.get("ob_4h_zone_active", False)),
        int(smc_data.get("liq_sweep_1h", False)),
        int(smc_data.get("liq_sweep_qualified", False)),       # [NEW v2.3]
        int(smc_data.get("bos_1h", False)),
        int(smc_data.get("choch_1h", False)),                  # [NEW v2.3]
        int(smc_data.get("msb_15m_confirmed", False)),         # [NEW v2.3]
        smc_data.get("mtf_confluence", 0),
        # 価格・ボラティリティ系
        market_data.get("atr_14", 0.0),
        market_data.get("atr_ratio", 1.0),
        market_data.get("bb_width", 0.0),
        market_data.get("close_vs_ema20_4h", 0.0),
        market_data.get("close_vs_ema50_4h", 0.0),
        market_data.get("high_low_range_15m", 0.0),
        # トレンド・モメンタム補助
        float(smc_data.get("trend_direction", 0)),
        float(smc_data.get("momentum_long", 0)),
        float(smc_data.get("momentum_short", 0)),
        # モメンタム系
        market_data.get("macd_histogram", 0.0),
        market_data.get("macd_signal_cross", 0),
        market_data.get("rsi_14", 50.0),
        market_data.get("rsi_zone", 0),
        market_data.get("stoch_k", 50.0),
        market_data.get("stoch_d", 50.0),
        market_data.get("momentum_3bar", 0.0),
        # 構造・パターン系 (7)
        market_data.get("ob_4h_distance_pips", 0.0),
        market_data.get("fvg_4h_fill_ratio", 0.0),
        market_data.get("liq_sweep_strength", 0.0),
        market_data.get("prior_candle_body_ratio", 0.5),
        market_data.get("consecutive_same_dir", 0),
        market_data.get("pivot_proximity", 0.0),
        smc_data.get("sweep_pending_bars", 0),              # [NEW v2.3]
        # リスク・ポジション系
        position_data.get("open_positions_count", 0),
        position_data.get("max_dd_24h", 0.0),
        calendar_risk_score,
        sentiment_score,
    ]

    return np.array(features, dtype=np.float64).reshape(1, -1)


class LGBMPredictor:
    """通貨ペアごとの LightGBM モデルによる推論。"""

    def __init__(self, model_dir: str = "models"):
        self._model_dir = Path(model_dir)
        self._models: dict[str, object] = {}  # pair -> model

    def load_model(self, pair: str) -> bool:
        """指定ペアのモデルを読み込む。"""
        model_path = self._model_dir / f"lgbm_{pair}.pkl"
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        try:
            model = joblib.load(model_path)

            expected_features = len(FEATURE_NAMES)
            model_features = getattr(model, "n_features_in_", None)
            if model_features is not None and int(model_features) != expected_features:
                logger.error(
                    f"Model feature mismatch for {pair}: model={model_features}, expected={expected_features}. "
                    f"Please retrain or regenerate model file: {model_path}"
                )
                return False

            self._models[pair] = model
            logger.info(f"Model loaded for {pair}: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Model load error for {pair}: {e}")
            return False

    def load_all_models(self, pairs: list[str] | None = None) -> dict[str, bool]:
        """全ペアのモデルを読み込む。"""
        if pairs is None:
            config = get_trading_config()
            pairs = config["system"]["pairs"]
        return {pair: self.load_model(pair) for pair in pairs}

    def predict(self, pair: str, features: np.ndarray) -> PredictionResult | None:
        """
        推論を実行する。

        Args:
            pair: 通貨ペア
            features: shape=(1, 35) の特徴量配列

        Returns:
            PredictionResult or None（モデル未読み込み時）
        """
        model = self._models.get(pair)
        if model is None:
            logger.error(f"No model loaded for {pair}")
            return None

        try:
            proba = model.predict_proba(features)[0]  # [prob_up, prob_flat, prob_down]
            result = PredictionResult(
                prob_up=float(proba[0]),
                prob_flat=float(proba[1]),
                prob_down=float(proba[2]),
            )
            logger.debug(
                f"LightGBM prediction for {pair}: "
                f"up={result.prob_up:.3f} flat={result.prob_flat:.3f} "
                f"down={result.prob_down:.3f} → {result.direction}"
            )
            return result
        except Exception as e:
            logger.error(
                f"LightGBM prediction error for {pair}: {e}. "
                f"Hint: verify model was trained with {len(FEATURE_NAMES)} features."
            )
            return None

    def save_model(self, pair: str, model: object) -> None:
        """モデルを保存する。"""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_dir / f"lgbm_{pair}.pkl"
        joblib.dump(model, model_path)
        self._models[pair] = model
        logger.info(f"Model saved for {pair}: {model_path}")
