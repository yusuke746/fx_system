"""
GPT-5.2 差分検知モジュール（Veto Layer B を含む）

■ 差分検知フロー（15分ごと実行）
  STEP 1: ニュースフィード最新10件取得
  STEP 2: MD5ハッシュによる新規記事検出
  STEP 3: ATR急変確認（ATR×1.5閾値）
  STEP 4: Calendar Veto中はスキップ
  STEP 5: キャッシュ有効期限確認
  STEP 6: 低消費モード判定（3回連続差分なし → 90分延長）
  STEP 7: call_gpt=True の場合のみ GPT-5.2 API 呼び出し

■ 時刻基準
  - API呼び出しログ: 保存基準（UTC）で記録
  - キャッシュ有効期限: 比較基準（UTC）で判定
  - Discord通知: 表示基準（JST）で表示
"""

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime

import openai
from loguru import logger

from config.settings import get_trading_config
from core.database import insert_api_call
from core.time_manager import now_utc, elapsed_minutes, format_jst


@dataclass
class SentimentResult:
    """GPT-5.2 出力のセンチメント結果。"""
    sentiment_score: float  # -1.0 〜 1.0
    unexpected_veto: bool   # Veto Layer B フラグ
    summary: str
    news_importance_score: float = 0.0
    risk_appetite_score: float = 0.0
    usd_macro_score: float = 0.0
    jpy_macro_score: float = 0.0
    eur_strength_score: float = 0.0
    gbp_strength_score: float = 0.0
    oil_shock_score: float = 0.0
    geopolitical_risk_score: float = 0.0
    model_used: str = ""
    escalated: bool = False
    timestamp_utc: datetime = field(default_factory=now_utc)

    @property
    def market_features(self) -> dict[str, float]:
        return {
            "risk_appetite_score": float(self.risk_appetite_score),
            "usd_macro_score": float(self.usd_macro_score),
            "jpy_macro_score": float(self.jpy_macro_score),
            "oil_shock_score": float(self.oil_shock_score),
            "geopolitical_risk_score": float(self.geopolitical_risk_score),
        }


class DiffDetector:
    """差分検知タスク管理（15分ごとの実行）。"""

    def __init__(self):
        self._last_article_hashes: set[str] = set()
        self._consecutive_skip_count: int = 0
        self._low_power_mode: bool = False
        self._cached_result: SentimentResult | None = None
        self._last_call_time: datetime | None = None

    @staticmethod
    def _blend(prev: float, cur: float, alpha: float) -> float:
        return (alpha * cur) + ((1.0 - alpha) * prev)

    def _ema_merge(self, new_result: SentimentResult) -> SentimentResult:
        if self._cached_result is None:
            return new_result

        cfg = get_trading_config().get("llm", {})
        alpha = float(cfg.get("state_ema_alpha", 0.35))
        alpha = max(0.0, min(1.0, alpha))
        prev = self._cached_result

        merged = replace(
            new_result,
            sentiment_score=self._blend(prev.sentiment_score, new_result.sentiment_score, alpha),
            news_importance_score=self._blend(prev.news_importance_score, new_result.news_importance_score, alpha),
            risk_appetite_score=self._blend(prev.risk_appetite_score, new_result.risk_appetite_score, alpha),
            usd_macro_score=self._blend(prev.usd_macro_score, new_result.usd_macro_score, alpha),
            jpy_macro_score=self._blend(prev.jpy_macro_score, new_result.jpy_macro_score, alpha),
            eur_strength_score=self._blend(prev.eur_strength_score, new_result.eur_strength_score, alpha),
            gbp_strength_score=self._blend(prev.gbp_strength_score, new_result.gbp_strength_score, alpha),
            oil_shock_score=self._blend(prev.oil_shock_score, new_result.oil_shock_score, alpha),
            geopolitical_risk_score=self._blend(prev.geopolitical_risk_score, new_result.geopolitical_risk_score, alpha),
            unexpected_veto=bool(new_result.unexpected_veto or prev.unexpected_veto),
            summary=new_result.summary or prev.summary,
        )
        return merged

    def update_cached_result(self, new_result: SentimentResult) -> SentimentResult:
        merged = self._ema_merge(new_result)
        self._cached_result = merged
        self._last_call_time = now_utc()
        return merged

    def get_effective_cached_result(self) -> SentimentResult | None:
        if self._cached_result is None:
            return None

        cfg = get_trading_config().get("llm", {})
        if not bool(cfg.get("state_decay_enabled", True)):
            return self._cached_result

        age_minutes = max(0.0, elapsed_minutes(self._cached_result.timestamp_utc))
        max_minutes = float(cfg.get("state_decay_max_minutes", 1440.0))
        if age_minutes >= max_minutes:
            decay = 0.0
        else:
            half_life = max(1.0, float(cfg.get("state_decay_half_life_minutes", 240.0)))
            decay = math.pow(0.5, age_minutes / half_life)

        veto_persist = float(cfg.get("state_veto_persist_minutes", 180.0))
        veto_active = bool(self._cached_result.unexpected_veto and age_minutes <= veto_persist)

        return replace(
            self._cached_result,
            sentiment_score=self._cached_result.sentiment_score * decay,
            news_importance_score=self._cached_result.news_importance_score * decay,
            risk_appetite_score=self._cached_result.risk_appetite_score * decay,
            usd_macro_score=self._cached_result.usd_macro_score * decay,
            jpy_macro_score=self._cached_result.jpy_macro_score * decay,
            eur_strength_score=self._cached_result.eur_strength_score * decay,
            gbp_strength_score=self._cached_result.gbp_strength_score * decay,
            oil_shock_score=self._cached_result.oil_shock_score * decay,
            geopolitical_risk_score=self._cached_result.geopolitical_risk_score * decay,
            unexpected_veto=veto_active,
        )

    @property
    def is_low_power(self) -> bool:
        return self._low_power_mode

    @property
    def cached_result(self) -> SentimentResult | None:
        return self._cached_result

    def run_diff_check(
        self,
        news_articles: list[dict],
        current_atr: float,
        avg_atr_20d: float,
        calendar_veto_active: bool,
    ) -> tuple[bool, str]:
        """
        差分検知ステップを実行し、GPT呼び出しが必要かどうかを返す。

        Returns:
            (should_call_gpt, reason)
        """
        config = get_trading_config()
        llm_cfg = config["llm"]
        atr_threshold = llm_cfg["atr_threshold_multiplier"]
        cache_ttl_normal = llm_cfg["cache_ttl_normal_minutes"]
        cache_ttl_low = llm_cfg["cache_ttl_low_power_minutes"]
        low_power_skips = llm_cfg["low_power_consecutive_skips"]

        has_diff = False

        # STEP 1-2: ニュースハッシュ比較
        new_hashes = set()
        for article in news_articles[:10]:
            h = hashlib.md5(
                json.dumps(article, sort_keys=True, default=str).encode()
            ).hexdigest()
            new_hashes.add(h)

        new_articles = new_hashes - self._last_article_hashes
        if new_articles:
            self._last_article_hashes = new_hashes
            has_diff = True
            reason = "NEW_ARTICLE"
            logger.info(f"Diff detected: {len(new_articles)} new articles")
        else:
            reason = ""

        # STEP 3: ATR急変確認
        price_spike = False
        if avg_atr_20d > 0 and current_atr > avg_atr_20d * atr_threshold:
            price_spike = True
            has_diff = True
            reason = "PRICE_SPIKE"
            logger.info(
                f"ATR spike detected: {current_atr:.5f} > "
                f"{avg_atr_20d:.5f} * {atr_threshold}"
            )

        # STEP 4: Calendar Veto中はスキップ
        if calendar_veto_active:
            logger.debug("Calendar Veto active - skipping GPT call")
            return False, ""

        # STEP 5: キャッシュ有効期限確認
        cache_expired = False
        if self._last_call_time is not None:
            ttl = cache_ttl_low if self._low_power_mode else cache_ttl_normal
            elapsed = elapsed_minutes(self._last_call_time)
            if elapsed >= ttl:
                cache_expired = True
                reason = "CACHE_EXPIRED_LOW" if self._low_power_mode else "CACHE_EXPIRED"
        else:
            cache_expired = True
            reason = "CACHE_EXPIRED"

        # STEP 6: 低消費モード判定
        if not has_diff and not cache_expired:
            self._consecutive_skip_count += 1
            if self._consecutive_skip_count >= low_power_skips and not self._low_power_mode:
                self._low_power_mode = True
                logger.info("Entering low-power mode (3 consecutive skips)")
        else:
            # 差分検出またはキャッシュ切れ → 通常モードに復帰
            if has_diff and self._low_power_mode:
                self._low_power_mode = False
                logger.info("Exiting low-power mode (diff detected)")
            self._consecutive_skip_count = 0

        # STEP 7: 呼び出し判定
        should_call = has_diff or cache_expired
        return should_call, reason


class LLMClient:
    """GPT-5.2 API クライアント（Responses API 使用）。"""

    def __init__(self, api_key: str, db_conn: sqlite3.Connection | None = None):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._db_conn = db_conn
        self._config = get_trading_config()
        llm_cfg = self._config["llm"]
        self._diff_model = llm_cfg.get("model_diff", llm_cfg["model_instant"])
        self._instant_model = llm_cfg.get("model_instant", "gpt-5.2")
        self._analysis_reasoning_effort = llm_cfg.get(
            "reasoning_effort_diff",
            llm_cfg.get("reasoning_effort_instant", "low"),
        )
        self._instant_reasoning_effort = llm_cfg.get("reasoning_effort_instant", "low")
        self._web_search_enabled = bool(llm_cfg.get("web_search_enabled", True))
        self._web_search_tool_type = llm_cfg.get("web_search_tool_type", "web_search_preview")
        self._web_search_context_size = llm_cfg.get("web_search_context_size", "low")

    def _as_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    def _coerce_score(self, value: object, default: float = 0.0) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = default
        return max(-1.0, min(1.0, score))

    def _coerce_importance(self, value: object, default: float = 0.0) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = default
        return max(0.0, min(1.0, score))

    def _first_available(self, parsed: dict, keys: list[str], default: object = 0.0) -> object:
        for key in keys:
            if key in parsed:
                return parsed[key]
        return default

    def _is_tool_not_supported_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "tool" in msg
            and ("not support" in msg or "invalid" in msg or "unknown" in msg)
        ) or "web_search" in msg

    def _build_web_search_tools(self) -> list[dict]:
        return [{
            "type": self._web_search_tool_type,
            "search_context_size": self._web_search_context_size,
        }]

    async def _analyze_with_model(
        self,
        pair: str,
        news_articles: list[dict],
        market_context: str,
        reason: str,
        model: str,
        reasoning_effort: str,
        mode: str,
        use_web_search: bool = False,
    ) -> SentimentResult:
        prompt = self._build_prompt(pair, news_articles, market_context)
        messages = [
            {"role": "system", "content": self._system_prompt(mode=mode)},
            {"role": "user", "content": prompt},
        ]

        request_kwargs = {
            "model": model,
            "input": messages,
            "reasoning": {"effort": reasoning_effort},
        }

        response = None
        if use_web_search and self._web_search_enabled:
            try:
                response = await self._client.responses.create(
                    **request_kwargs,
                    tools=self._build_web_search_tools(),
                )
            except Exception as web_search_error:
                if self._is_tool_not_supported_error(web_search_error):
                    logger.warning(
                        f"web_search unavailable for model={model}. "
                        f"Fallback to no-web-search. err={web_search_error}"
                    )
                else:
                    raise

        if response is None:
            response = await self._client.responses.create(**request_kwargs)

        text = ""
        for item in response.output:
            if getattr(item, "type", "") != "message":
                continue
            content = getattr(item, "content", None)
            if content is None:
                continue
            for block in content:
                if hasattr(block, "text"):
                    text += block.text

        parsed = self._parse_llm_json(text)

        sentiment_raw = self._first_available(
            parsed,
            ["sentiment_score", "target_pair_sentiment"],
            0.0,
        )
        usd_macro_raw = self._first_available(
            parsed,
            ["usd_macro_score", "usd_strength_score"],
            0.0,
        )
        jpy_macro_raw = self._first_available(
            parsed,
            ["jpy_macro_score", "jpy_strength_score"],
            0.0,
        )
        oil_raw = self._first_available(
            parsed,
            ["oil_shock_score", "energy_shock_score"],
            0.0,
        )
        news_importance_raw = self._first_available(
            parsed,
            ["news_importance_score"],
            0.0,
        )

        result = SentimentResult(
            sentiment_score=self._coerce_score(sentiment_raw),
            unexpected_veto=self._as_bool(parsed.get("unexpected_veto", False)),
            summary=parsed.get("summary", ""),
            news_importance_score=self._coerce_importance(news_importance_raw),
            risk_appetite_score=self._coerce_score(parsed.get("risk_appetite_score", 0.0)),
            usd_macro_score=self._coerce_score(usd_macro_raw),
            jpy_macro_score=self._coerce_score(jpy_macro_raw),
            eur_strength_score=self._coerce_score(parsed.get("eur_strength_score", 0.0)),
            gbp_strength_score=self._coerce_score(parsed.get("gbp_strength_score", 0.0)),
            oil_shock_score=self._coerce_score(oil_raw),
            geopolitical_risk_score=self._coerce_score(parsed.get("geopolitical_risk_score", 0.0)),
            model_used=model,
        )

        if "news_importance_score" not in parsed:
            inferred_importance = max(
                abs(result.sentiment_score),
                abs(result.risk_appetite_score),
                abs(result.usd_macro_score),
                abs(result.jpy_macro_score),
                abs(result.oil_shock_score),
                abs(result.geopolitical_risk_score),
            )
            if result.unexpected_veto:
                inferred_importance = max(inferred_importance, 0.9)
            result.news_importance_score = self._coerce_importance(inferred_importance)

        if self._db_conn:
            usage = response.usage
            tokens_in = usage.input_tokens
            tokens_out = usage.output_tokens
            cost = self._estimate_cost(tokens_in, tokens_out, model)
            insert_api_call(
                self._db_conn,
                reason=reason,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )

        return result

    async def analyze_sentiment(
        self,
        pair: str,
        news_articles: list[dict],
        market_context: str,
        reason: str,
    ) -> SentimentResult:
        """
        GPT-5.2 にセンチメント分析を依頼する（Responses API）。

        Args:
            pair: 通貨ペア
            news_articles: 最新ニュース記事リスト
            market_context: 市場コンテキスト文字列
            reason: 呼び出し理由

        Returns:
            SentimentResult
        """
        try:
            result = await self._analyze_with_model(
                pair=pair,
                news_articles=news_articles,
                market_context=market_context,
                reason=reason,
                model=self._instant_model,
                reasoning_effort=self._instant_reasoning_effort,
                mode="deep",
                use_web_search=True,
            )

            logger.info(
                f"GPT sentiment for {pair} via {result.model_used}: score={result.sentiment_score:.2f}, "
                f"veto={result.unexpected_veto}"
            )
            return result

        except openai.APITimeoutError:
            logger.error("GPT API timeout (>5s)")
            return SentimentResult(
                sentiment_score=0.0,
                unexpected_veto=False,
                summary="API timeout",
            )
        except openai.RateLimitError:
            logger.error("GPT API rate limit (429)")
            return SentimentResult(
                sentiment_score=0.0,
                unexpected_veto=False,
                summary="Rate limited",
            )
        except openai.AuthenticationError:
            logger.critical("GPT API authentication error (401/403)")
            raise  # 上位で llm_enabled=false にする

    async def analyze_sentiment_hybrid(
        self,
        pair: str,
        news_articles: list[dict],
        market_context: str,
        reason: str,
    ) -> SentimentResult:
        """nanoで常時判定し、重要ニュース時のみ gpt-5.2 で再判定する。"""
        llm_cfg = self._config.get("llm", {})
        if not llm_cfg.get("hybrid_enabled", True):
            return await self.analyze_sentiment(pair, news_articles, market_context, reason)

        def _is_model_not_found_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return "model_not_found" in msg or "does not exist" in msg

        try:
            try:
                quick = await self._analyze_with_model(
                    pair=pair,
                    news_articles=news_articles,
                    market_context=market_context,
                    reason=f"{reason}:quick",
                    model=self._diff_model,
                    reasoning_effort=self._analysis_reasoning_effort,
                    mode="quick",
                    use_web_search=True,
                )
            except Exception as quick_error:
                if not _is_model_not_found_error(quick_error):
                    raise
                logger.warning(
                    f"LLM quick model unavailable: {self._diff_model}. "
                    f"Falling back to deep model {self._instant_model}."
                )
                quick = await self._analyze_with_model(
                    pair=pair,
                    news_articles=news_articles,
                    market_context=market_context,
                    reason=f"{reason}:quick_fallback",
                    model=self._instant_model,
                    reasoning_effort=self._instant_reasoning_effort,
                    mode="deep",
                    use_web_search=True,
                )
                quick.escalated = True
                return quick

            threshold = float(llm_cfg.get("news_importance_escalation_threshold", 0.65))
            escalate = quick.news_importance_score >= threshold or quick.unexpected_veto
            if not escalate:
                logger.info(
                    f"GPT hybrid quick for {pair}: model={quick.model_used} importance={quick.news_importance_score:.2f}"
                )
                return quick

            deep = await self._analyze_with_model(
                pair=pair,
                news_articles=news_articles,
                market_context=market_context,
                reason=f"{reason}:deep",
                model=self._instant_model,
                reasoning_effort=self._instant_reasoning_effort,
                mode="deep",
                use_web_search=True,
            )
            deep.escalated = True
            deep.news_importance_score = max(deep.news_importance_score, quick.news_importance_score)
            logger.info(
                f"GPT hybrid escalated for {pair}: {quick.model_used} -> {deep.model_used}, "
                f"importance={deep.news_importance_score:.2f}, veto={deep.unexpected_veto}"
            )
            return deep

        except openai.APITimeoutError:
            logger.error("GPT API timeout (>5s)")
            return SentimentResult(
                sentiment_score=0.0,
                unexpected_veto=False,
                summary="API timeout",
                news_importance_score=0.0,
                model_used=self._diff_model,
            )
        except openai.RateLimitError:
            logger.error("GPT API rate limit (429)")
            return SentimentResult(
                sentiment_score=0.0,
                unexpected_veto=False,
                summary="Rate limited",
                news_importance_score=0.0,
                model_used=self._diff_model,
            )
        except openai.AuthenticationError:
            logger.critical("GPT API authentication error (401/403)")
            raise

    async def propose_weekly_optimization(
        self,
        summary: dict,
        reason: str = "WEEKLY_OPTIMIZATION",
    ) -> dict:
        """
        DB集計サマリーを入力として、週次パラメータ調整案をJSONで返す。

        Returns:
            {
              "summary": str,
              "confidence": float(0..1),
              "threshold_adjustments": [
                {
                  "pair": "USDJPY",
                  "direction": "long|short",
                  "direction_threshold": 0.5,
                  "block_threshold": 0.4,
                  "min_edge": 0.1,
                  "reason": "..."
                }
              ]
            }
        """
        system_prompt = (
            "You are a risk-aware FX optimization assistant. "
            "Use only the provided JSON summary from closed-trade statistics. "
            "Do not invent data. Return strict JSON only with keys: "
            "summary (string), confidence (0..1 float), threshold_adjustments (array). "
            "Each threshold_adjustments item must contain: "
            "pair (string), direction (long|short), direction_threshold (float), "
            "block_threshold (float), min_edge (float), reason (string). "
            "If no safe change is needed, return threshold_adjustments as empty array."
        )
        user_prompt = (
            "Weekly optimization input (JSON):\n"
            f"{json.dumps(summary, ensure_ascii=True)}\n\n"
            "Output strict JSON only."
        )

        response = await self._client.responses.create(
            model=self._instant_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reasoning={"effort": self._instant_reasoning_effort},
        )

        text = ""
        for item in response.output:
            if getattr(item, "type", "") != "message":
                continue
            content = getattr(item, "content", None)
            if content is None:
                continue
            for block in content:
                if hasattr(block, "text"):
                    text += block.text

        parsed = self._parse_llm_json(text)

        if self._db_conn:
            usage = response.usage
            tokens_in = usage.input_tokens
            tokens_out = usage.output_tokens
            cost = self._estimate_cost(tokens_in, tokens_out, self._instant_model)
            insert_api_call(
                self._db_conn,
                reason=reason,
                model=self._instant_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )

        confidence = self._coerce_importance(parsed.get("confidence", 0.0))
        adjustments = parsed.get("threshold_adjustments", [])
        if not isinstance(adjustments, list):
            adjustments = []

        return {
            "summary": str(parsed.get("summary", "")),
            "confidence": confidence,
            "threshold_adjustments": adjustments,
        }

    def _system_prompt(self, mode: str = "deep") -> str:
        if mode == "quick":
            return (
                "You are a low-cost forex news triage analyzer. "
                "Return strict JSON only with these fields:\n"
                "- sentiment_score: float from -1.0 to 1.0\n"
                "- unexpected_veto: boolean\n"
                "- news_importance_score: float from 0.0 to 1.0\n"
                "- risk_appetite_score: float from -1.0 (risk-off) to 1.0 (risk-on)\n"
                "- usd_macro_score: float from -1.0 (USD bearish) to 1.0 (USD bullish)\n"
                "- jpy_macro_score: float from -1.0 (JPY bearish) to 1.0 (JPY bullish)\n"
                "- oil_shock_score: float from -1.0 (oil collapse/disinflation) to 1.0 (oil spike/inflation shock)\n"
                "- geopolitical_risk_score: float from -1.0 (risk easing) to 1.0 (risk escalation)\n"
                "- summary: brief summary (max 100 chars)"
            )

        return (
            "You are a forex market sentiment analyzer. "
            "Analyze the provided news and market context. "
            "Respond in JSON format with these fields:\n"
            "- sentiment_score: float from -1.0 (very bearish) to 1.0 (very bullish)\n"
            "- unexpected_veto: boolean, true if there is a sudden geopolitical or "
            "economic event that should halt trading\n"
            "- news_importance_score: float from 0.0 to 1.0\n"
            "- risk_appetite_score: float from -1.0 (risk-off) to 1.0 (risk-on). "
            "Use equity index direction, volatility compression, credit/geopolitical easing, "
            "carry appetite, and broad safe-haven unwind.\n"
            "- usd_macro_score: float from -1.0 (USD bearish) to 1.0 (USD bullish). "
            "Use Fed path, US growth, US yields, tariff/fiscal implications, and USD funding demand.\n"
            "- jpy_macro_score: float from -1.0 (JPY bearish) to 1.0 (JPY bullish). "
            "Use safe-haven demand, BoJ policy expectations, Japan yield/cpi signals, and carry unwind pressure.\n"
            "- oil_shock_score: float from -1.0 (oil collapse/disinflation) to 1.0 (oil spike/inflation shock). "
            "Use crude, shipping, Middle East supply risk, and energy inflation impulse.\n"
            "- geopolitical_risk_score: float from -1.0 (risk easing) to 1.0 (risk escalation). "
            "Use war/ceasefire, sanctions, Strait of Hormuz, shipping lanes, and global conflict spillover.\n"
            "- summary: brief summary of the analysis (max 100 chars)"
        )

    def _build_prompt(self, pair: str, articles: list[dict], context: str) -> str:
        articles_text = "\n".join(
            f"- {a.get('title', 'N/A')}: {a.get('summary', 'N/A')}"
            for a in articles[:10]
        )
        return (
            f"Pair: {pair}\n"
            f"Market Context:\n{context}\n\n"
            f"Recent News:\n{articles_text}\n\n"
            f"Provide sentiment analysis in JSON format."
        )

    def _parse_llm_json(self, raw_text: str) -> dict:
        """LLM出力JSONを安全にパースする（Markdown混入対策つき）。"""
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.error(f"GPT response parse failed. raw={raw_text[:500]}")
        return {
            "sentiment_score": 0.0,
            "unexpected_veto": True,
            "news_importance_score": 1.0,
            "risk_appetite_score": 0.0,
            "usd_macro_score": 0.0,
            "jpy_macro_score": 0.0,
            "oil_shock_score": 0.0,
            "geopolitical_risk_score": 0.0,
            "summary": "Parse error fallback",
        }

    def _estimate_cost(self, tokens_in: int, tokens_out: int, model_name: str) -> float:
        """定期ニュース判定用モデルのコスト概算。未定義モデルは model_instant 相当で保守的に扱う。"""
        pricing = {
            "gpt-5-nano": (0.05, 0.4),
            "gpt-5.2-nano": (0.2, 1.6),
            "gpt-5.2": (1.75, 14.0),
        }
        input_per_million, output_per_million = pricing.get(
            model_name,
            pricing["gpt-5.2"],
        )
        return (tokens_in * input_per_million + tokens_out * output_per_million) / 1_000_000
