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

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
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
    timestamp_utc: datetime = field(default_factory=now_utc)


class DiffDetector:
    """差分検知タスク管理（15分ごとの実行）。"""

    def __init__(self):
        self._last_article_hashes: set[str] = set()
        self._consecutive_skip_count: int = 0
        self._low_power_mode: bool = False
        self._cached_result: SentimentResult | None = None
        self._last_call_time: datetime | None = None

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
    """GPT-5.2 API クライアント。"""

    def __init__(self, api_key: str, db_conn: sqlite3.Connection | None = None):
        self._client = openai.OpenAI(api_key=api_key)
        self._db_conn = db_conn
        self._config = get_trading_config()
        self._model = self._config["llm"]["model_instant"]

    async def analyze_sentiment(
        self,
        pair: str,
        news_articles: list[dict],
        market_context: str,
        reason: str,
    ) -> SentimentResult:
        """
        GPT-5.2 にセンチメント分析を依頼する。

        Args:
            pair: 通貨ペア
            news_articles: 最新ニュース記事リスト
            market_context: 市場コンテキスト文字列
            reason: 呼び出し理由

        Returns:
            SentimentResult
        """
        prompt = self._build_prompt(pair, news_articles, market_context)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            result = SentimentResult(
                sentiment_score=float(parsed.get("sentiment_score", 0.0)),
                unexpected_veto=bool(parsed.get("unexpected_veto", False)),
                summary=parsed.get("summary", ""),
            )

            # API呼び出しログ記録（保存基準: UTC）
            if self._db_conn:
                usage = response.usage
                cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)
                insert_api_call(
                    self._db_conn,
                    reason=reason,
                    model=self._model,
                    tokens_in=usage.prompt_tokens,
                    tokens_out=usage.completion_tokens,
                    cost_usd=cost,
                )

            logger.info(
                f"GPT sentiment for {pair}: score={result.sentiment_score:.2f}, "
                f"veto={result.unexpected_veto}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"GPT response parse error: {e}")
            return SentimentResult(
                sentiment_score=0.0,
                unexpected_veto=False,
                summary=f"Parse error: {e}",
            )
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

    def _system_prompt(self) -> str:
        return (
            "You are a forex market sentiment analyzer. "
            "Analyze the provided news and market context. "
            "Respond in JSON format with these fields:\n"
            "- sentiment_score: float from -1.0 (very bearish) to 1.0 (very bullish)\n"
            "- unexpected_veto: boolean, true if there is a sudden geopolitical or "
            "economic event that should halt trading\n"
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

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """GPT-5.2-instant のコスト概算（$1.75/M tokens）。"""
        return (tokens_in + tokens_out) * 1.75 / 1_000_000
