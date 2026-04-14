"""
過去データ向け LLM マクロ特徴量バックフィル。

入力:
- TradingView の 15分足 CSV（時刻列 + テクニカル列）
- ニュース JSON/JSONL（published_at/time + title + summary）

出力:
- テクニカル CSV に LLM 特徴量を backward merge した CSV

主な仕様:
- 15分ごとに状態を更新
- ATR×1.5 急変時は market_context に明記
- ニュース差分 / ATR急変 / TTL切れ でのみ LLM を再呼び出し
- 呼び出し結果は EMA 平滑化
- 呼び出し間は forward-fill + 指数減衰
- merge_asof(direction='backward') で look-ahead bias を回避

例:
python -m maintenance.backfill_macro_features \
  --pair USDJPY \
  --chart-csv data/USDJPY_chart.csv \
  --news-json data/calendar/news_history.jsonl \
  --output data/USDJPY_backfilled.csv
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import get_settings, get_trading_config
from llm.llm_client import LLMClient, SentimentResult


DEFAULT_FEATURES = {
    "sentiment_score": 0.0,
    "calendar_risk_score": 0,
    "news_importance_score": 0.0,
    "risk_appetite_score": 0.0,
    "usd_macro_score": 0.0,
    "jpy_macro_score": 0.0,
    "oil_shock_score": 0.0,
    "geopolitical_risk_score": 0.0,
    "llm_unexpected_veto": 0,
}


@dataclass
class BackfillState:
    last_result: SentimentResult | None = None
    last_call_time: pd.Timestamp | None = None
    last_news_hash: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill LLM macro features and merge into chart CSV")
    parser.add_argument("--pair", required=True, choices=["USDJPY", "EURUSD", "GBPJPY"])
    parser.add_argument("--chart-csv", required=True, help="Path to TradingView chart CSV")
    parser.add_argument("--news-json", required=True, help="Path to news json/jsonl file")
    parser.add_argument("--output", required=True, help="Output merged CSV")
    parser.add_argument("--lookback-days", type=int, default=180, help="Lookback window from latest bar")
    parser.add_argument("--news-lookback-hours", type=int, default=24, help="News context window in hours")
    parser.add_argument("--max-news", type=int, default=10, help="Max news items per inference")
    parser.add_argument("--cache-ttl-minutes", type=int, default=60, help="TTL for forced LLM refresh")
    parser.add_argument("--atr-threshold", type=float, default=1.5, help="ATR spike threshold multiplier")
    parser.add_argument(
        "--ignore-atr-spike-trigger",
        action="store_true",
        help="Disable ATR spike trigger and call LLM only on TTL/news-diff",
    )
    parser.add_argument(
        "--ignore-news-diff-trigger",
        action="store_true",
        help="Disable news-hash diff trigger and call LLM only on TTL/ATR spike",
    )
    return parser.parse_args()


def _find_time_col(df: pd.DataFrame) -> str:
    for col in ("time", "Time", "timestamp", "Timestamp", "date", "Date"):
        if col in df.columns:
            return col
    raise ValueError("No time column found. expected one of: time/Time/timestamp/Timestamp/date/Date")


def _find_atr_col(df: pd.DataFrame) -> str:
    for col in ("CSV_atr_14", "atr_14", "atr"):
        if col in df.columns:
            return col
    raise ValueError("No ATR column found. expected one of: CSV_atr_14/atr_14/atr")


def _load_chart_df(path: Path, lookback_days: int) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    time_col = _find_time_col(df)
    atr_col = _find_atr_col(df)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    cutoff = df["timestamp"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff].copy().reset_index(drop=True)

    # 過去情報のみで ATR 基準線を作る（shift(1) で未来リーク防止）。
    bars_per_day = 24 * 4
    atr_window = 20 * bars_per_day
    df["atr_current"] = pd.to_numeric(df[atr_col], errors="coerce").fillna(0.0)
    df["atr_20d_avg"] = (
        df["atr_current"]
        .rolling(window=atr_window, min_periods=bars_per_day)
        .mean()
        .shift(1)
        .fillna(0.0)
    )
    return df, atr_col


def _load_news(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame(columns=["published_at", "title", "summary"])

    records: list[dict] = []
    if text.startswith("["):
        raw = json.loads(text)
        if isinstance(raw, list):
            records = [r for r in raw if isinstance(r, dict)]
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)

    norm_rows: list[dict] = []
    for row in records:
        ts = row.get("published_at") or row.get("time") or row.get("timestamp")
        title = str(row.get("title") or "")
        summary = str(row.get("summary") or row.get("body") or "")
        if not ts:
            continue
        dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        norm_rows.append({
            "published_at": dt,
            "title": title,
            "summary": summary,
        })

    if not norm_rows:
        return pd.DataFrame(columns=["published_at", "title", "summary"])

    return pd.DataFrame(norm_rows).sort_values("published_at").reset_index(drop=True)


def _slice_news(news_df: pd.DataFrame, now_ts: pd.Timestamp, hours: int, max_items: int) -> list[dict]:
    if news_df.empty:
        return []
    start_ts = now_ts - pd.Timedelta(hours=hours)
    sliced = news_df[(news_df["published_at"] <= now_ts) & (news_df["published_at"] >= start_ts)]
    if sliced.empty:
        return []
    tail = sliced.tail(max_items)
    return [
        {
            "title": str(r["title"]),
            "summary": str(r["summary"]),
        }
        for _, r in tail.iterrows()
    ]


def _news_hash(news: list[dict]) -> str:
    payload = json.dumps(news, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _ema_result(prev: SentimentResult | None, cur: SentimentResult, alpha: float) -> SentimentResult:
    if prev is None:
        return cur

    alpha = max(0.0, min(1.0, alpha))
    return SentimentResult(
        sentiment_score=(alpha * cur.sentiment_score) + ((1.0 - alpha) * prev.sentiment_score),
        unexpected_veto=bool(cur.unexpected_veto or prev.unexpected_veto),
        summary=cur.summary or prev.summary,
        news_importance_score=(alpha * cur.news_importance_score) + ((1.0 - alpha) * prev.news_importance_score),
        risk_appetite_score=(alpha * cur.risk_appetite_score) + ((1.0 - alpha) * prev.risk_appetite_score),
        usd_macro_score=(alpha * cur.usd_macro_score) + ((1.0 - alpha) * prev.usd_macro_score),
        jpy_macro_score=(alpha * cur.jpy_macro_score) + ((1.0 - alpha) * prev.jpy_macro_score),
        eur_strength_score=(alpha * cur.eur_strength_score) + ((1.0 - alpha) * prev.eur_strength_score),
        gbp_strength_score=(alpha * cur.gbp_strength_score) + ((1.0 - alpha) * prev.gbp_strength_score),
        oil_shock_score=(alpha * cur.oil_shock_score) + ((1.0 - alpha) * prev.oil_shock_score),
        geopolitical_risk_score=(alpha * cur.geopolitical_risk_score) + ((1.0 - alpha) * prev.geopolitical_risk_score),
        model_used=cur.model_used,
        escalated=cur.escalated,
        timestamp_utc=cur.timestamp_utc,
    )


def _apply_decay(result: SentimentResult | None, age_minutes: float, half_life_minutes: float, veto_persist_minutes: float) -> dict[str, float | int]:
    if result is None:
        return dict(DEFAULT_FEATURES)

    half_life = max(1.0, half_life_minutes)
    decay = float(0.5 ** (max(0.0, age_minutes) / half_life))

    veto = 1 if (result.unexpected_veto and age_minutes <= veto_persist_minutes) else 0
    calendar_risk_score = 2 if veto else (1 if result.news_importance_score * decay >= 0.55 else 0)

    return {
        "sentiment_score": float(result.sentiment_score * decay),
        "calendar_risk_score": int(calendar_risk_score),
        "news_importance_score": float(result.news_importance_score * decay),
        "risk_appetite_score": float(result.risk_appetite_score * decay),
        "usd_macro_score": float(result.usd_macro_score * decay),
        "jpy_macro_score": float(result.jpy_macro_score * decay),
        "oil_shock_score": float(result.oil_shock_score * decay),
        "geopolitical_risk_score": float(result.geopolitical_risk_score * decay),
        "llm_unexpected_veto": int(veto),
    }


async def _run_backfill(args: argparse.Namespace) -> pd.DataFrame:
    settings = get_settings()
    cfg = get_trading_config()
    llm_cfg = cfg.get("llm", {})

    chart_df, _ = _load_chart_df(Path(args.chart_csv), lookback_days=args.lookback_days)
    news_df = _load_news(Path(args.news_json))

    if chart_df.empty:
        raise ValueError("No chart rows after lookback filter")

    llm_client = LLMClient(api_key=settings.openai_api_key.get_secret_value())
    state = BackfillState()

    alpha = float(llm_cfg.get("state_ema_alpha", 0.35))
    decay_half_life = float(llm_cfg.get("state_decay_half_life_minutes", 240))
    veto_persist = float(llm_cfg.get("state_veto_persist_minutes", 180))

    rows: list[dict] = []
    llm_calls = 0

    for _, bar in chart_df.iterrows():
        ts = bar["timestamp"]
        atr_current = float(bar.get("atr_current", 0.0) or 0.0)
        atr_20d_avg = float(bar.get("atr_20d_avg", 0.0) or 0.0)

        atr_spike = atr_20d_avg > 0 and atr_current > (atr_20d_avg * float(args.atr_threshold))
        if bool(getattr(args, "ignore_atr_spike_trigger", False)):
            atr_spike = False
        news_items = _slice_news(news_df, ts, args.news_lookback_hours, args.max_news)
        current_hash = _news_hash(news_items)

        cache_expired = (
            state.last_call_time is None
            or (ts - state.last_call_time) >= pd.Timedelta(minutes=args.cache_ttl_minutes)
        )
        news_diff = current_hash != state.last_news_hash
        should_call = state.last_result is None or cache_expired or atr_spike
        if not bool(getattr(args, "ignore_news_diff_trigger", False)):
            should_call = should_call or news_diff

        reason = "CACHE_EXPIRED"
        if atr_spike:
            reason = "PRICE_SPIKE"
        elif current_hash != state.last_news_hash:
            reason = "NEW_ARTICLE"

        if should_call:
            context_text = (
                f"ATR spiked {args.atr_threshold:.2f}x in last 15 mins "
                f"(atr={atr_current:.6f}, avg20d={atr_20d_avg:.6f}). "
                "Infer whether move is news-driven or flow-driven."
                if atr_spike
                else (
                    f"No ATR {args.atr_threshold:.2f}x spike in last 15 mins "
                    f"(atr={atr_current:.6f}, avg20d={atr_20d_avg:.6f})."
                )
            )

            result = await llm_client.analyze_sentiment_hybrid(
                pair=args.pair,
                news_articles=news_items,
                market_context=context_text,
                reason=f"BACKFILL_{reason}",
            )
            state.last_result = _ema_result(state.last_result, result, alpha=alpha)
            state.last_call_time = ts
            state.last_news_hash = current_hash
            llm_calls += 1

        if state.last_call_time is None:
            age_minutes = 0.0
        else:
            age_minutes = max(0.0, (ts - state.last_call_time).total_seconds() / 60.0)

        decayed = _apply_decay(
            result=state.last_result,
            age_minutes=age_minutes,
            half_life_minutes=decay_half_life,
            veto_persist_minutes=veto_persist,
        )
        decayed["timestamp"] = ts
        rows.append(decayed)

    llm_df = pd.DataFrame(rows).sort_values("timestamp")

    merged = pd.merge_asof(
        chart_df.sort_values("timestamp"),
        llm_df,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.drop(columns=["atr_current", "atr_20d_avg"], errors="ignore")
    logger.info(
        "Backfill complete: pair={} rows={} llm_calls={} news_rows={}",
        args.pair,
        len(merged),
        llm_calls,
        len(news_df),
    )
    return merged


def run_backfill_for_pair(
    pair: str,
    chart_csv: Path,
    news_json: Path,
    output: Path,
    lookback_days: int = 180,
    news_lookback_hours: int = 24,
    max_news: int = 10,
    cache_ttl_minutes: int = 60,
    atr_threshold: float = 1.5,
    ignore_news_diff_trigger: bool = False,
    ignore_atr_spike_trigger: bool = False,
) -> pd.DataFrame:
    args = argparse.Namespace(
        pair=pair,
        chart_csv=str(chart_csv),
        news_json=str(news_json),
        output=str(output),
        lookback_days=int(lookback_days),
        news_lookback_hours=int(news_lookback_hours),
        max_news=int(max_news),
        cache_ttl_minutes=int(cache_ttl_minutes),
        atr_threshold=float(atr_threshold),
        ignore_news_diff_trigger=bool(ignore_news_diff_trigger),
        ignore_atr_spike_trigger=bool(ignore_atr_spike_trigger),
    )
    merged = asyncio.run(_run_backfill(args))
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)

    merged = run_backfill_for_pair(
        pair=args.pair,
        chart_csv=Path(args.chart_csv),
        news_json=Path(args.news_json),
        output=output_path,
        lookback_days=args.lookback_days,
        news_lookback_hours=args.news_lookback_hours,
        max_news=args.max_news,
        cache_ttl_minutes=args.cache_ttl_minutes,
        atr_threshold=args.atr_threshold,
        ignore_news_diff_trigger=args.ignore_news_diff_trigger,
        ignore_atr_spike_trigger=args.ignore_atr_spike_trigger,
    )

    print(f"rows={len(merged)}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
