"""
NewsAPI からニュース履歴を取得し、バックフィル用 JSONL を作成する。

要件:
- .env の NEWSAPI_KEY を使用
- published_at, title, summary を必須として保存
- 期間を分割して順次取得

使い方:
python -m maintenance.fetch_news_history_newsapi \
  --days 180 \
  --output data/calendar/news_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
DEFAULT_QUERY = "(forex OR FX OR USDJPY OR EURUSD OR GBPJPY OR Nikkei OR oil OR geopolitics)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical news from NewsAPI")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="NewsAPI query")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    parser.add_argument("--window-days", type=int, default=7, help="Fetch window size in days")
    parser.add_argument("--page-size", type=int, default=100, help="NewsAPI page size (max 100)")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages per window")
    parser.add_argument("--language", default="en", help="Language filter")
    parser.add_argument("--sort-by", default="publishedAt", choices=["publishedAt", "relevancy", "popularity"], help="Sort order")
    parser.add_argument("--retry", type=int, default=3, help="Retry count per request")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between requests")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def _get_api_key() -> str:
    load_dotenv()
    key = os.getenv("NEWSAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("NEWSAPI_KEY is not set in environment/.env")
    return key


def _to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _request_with_retry(params: dict, api_key: str, retry: int, timeout: int, pause_seconds: float) -> dict:
    headers = {"X-Api-Key": api_key}
    attempts = max(1, int(retry))
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = pause_seconds * attempt * 2.0
                logger.warning("NewsAPI rate limited. attempt={}/{} sleep={}s", attempt, attempts, round(wait, 2))
                time.sleep(max(0.0, wait))
                continue

            resp.raise_for_status()
            body = resp.json()
            if body.get("status") != "ok":
                raise RuntimeError(f"NewsAPI error: {body}")
            return body
        except Exception as exc:
            last_exc = exc
            wait = pause_seconds * attempt
            logger.warning("request failed attempt={}/{}: {} (sleep {}s)", attempt, attempts, exc, round(wait, 2))
            time.sleep(max(0.0, wait))

    if last_exc:
        raise last_exc
    raise RuntimeError("request failed without exception")


def _normalize_articles(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        published_at = str(row.get("publishedAt") or "").strip()
        title = str(row.get("title") or "").strip()
        summary = str(row.get("description") or row.get("content") or "").strip()
        url = str(row.get("url") or "").strip()
        source = ""
        src_obj = row.get("source")
        if isinstance(src_obj, dict):
            source = str(src_obj.get("name") or "").strip()

        if not published_at or not title:
            continue

        out.append(
            {
                "published_at": published_at,
                "title": title,
                "summary": summary,
                "url": url,
                "source": source,
            }
        )
    return out


def main() -> None:
    args = _parse_args()
    api_key = _get_api_key()

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=max(1, args.days))
    window_delta = timedelta(days=max(1, args.window_days))

    logger.info("NewsAPI fetch start: {} -> {}", _to_iso_utc(start_utc), _to_iso_utc(now_utc))

    all_rows: list[dict] = []
    cursor = start_utc

    while cursor < now_utc:
        nxt = min(now_utc, cursor + window_delta)

        from_str = _to_iso_utc(cursor)
        to_str = _to_iso_utc(nxt)

        window_count = 0
        for page in range(1, max(1, args.max_pages) + 1):
            params = {
                "q": args.query,
                "language": args.language,
                "sortBy": args.sort_by,
                "from": from_str,
                "to": to_str,
                "pageSize": max(1, min(100, int(args.page_size))),
                "page": page,
            }

            try:
                body = _request_with_retry(
                    params=params,
                    api_key=api_key,
                    retry=args.retry,
                    timeout=args.timeout,
                    pause_seconds=args.pause_seconds,
                )
            except Exception as exc:
                logger.warning("window {} -> {} page={} failed: {}", from_str, to_str, page, exc)
                break

            raw_articles = body.get("articles", [])
            if not isinstance(raw_articles, list) or not raw_articles:
                break

            normalized = _normalize_articles(raw_articles)
            all_rows.extend(normalized)
            window_count += len(normalized)

            logger.info(
                "window {} -> {} page={} raw={} normalized={}",
                from_str,
                to_str,
                page,
                len(raw_articles),
                len(normalized),
            )

            if len(raw_articles) < params["pageSize"]:
                break

            time.sleep(max(0.0, args.pause_seconds))

        logger.info("window {} -> {} total_normalized={}", from_str, to_str, window_count)
        cursor = nxt
        time.sleep(max(0.0, args.pause_seconds))

    dedup: dict[str, dict] = {}
    for row in all_rows:
        key = row.get("url") or f"{row.get('published_at')}|{row.get('title')}"
        if key and key not in dedup:
            dedup[key] = row

    rows = sorted(dedup.values(), key=lambda x: (x.get("published_at") or "", x.get("title") or ""))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows={len(rows)}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
