"""
GDELT Doc API からニュース履歴を取得し、バックフィル用 JSONL を作成する。

- APIキー不要
- UTC時刻で published_at を保存
- 期間を分割して取得し、重複URLを除去

出力フォーマット（1行1JSON）:
{"published_at":"2026-01-01T12:34:56Z","title":"...","summary":"...","url":"...","source":"..."}

使い方:
python -m maintenance.fetch_news_history_gdelt \
  --days 180 \
  --output data/calendar/news_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import requests
from loguru import logger


GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = "(forex OR FX OR USDJPY OR EURUSD OR GBPJPY OR Nikkei OR crude oil OR geopolitics) lang:english"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical news from GDELT Doc API")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="GDELT query string")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    parser.add_argument("--window-hours", type=int, default=24, help="Fetch window size in hours")
    parser.add_argument("--max-records", type=int, default=250, help="Max records per API call")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--retry", type=int, default=4, help="Retry count per window")
    parser.add_argument("--pause-seconds", type=float, default=1.5, help="Base pause between calls")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def _fmt_gdelt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _to_iso_utc(value: str) -> str | None:
    if not value:
        return None

    candidates = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%dT%H%M%SZ",
        "%Y%m%d%H%M%S",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue

    try:
        dt2 = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt2.tzinfo is None:
            dt2 = dt2.replace(tzinfo=timezone.utc)
        return dt2.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _fetch_window_once(
    query: str,
    start: datetime,
    end: datetime,
    max_records: int,
    timeout_sec: int,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max(1, min(250, max_records))),
        "startdatetime": _fmt_gdelt_dt(start),
        "enddatetime": _fmt_gdelt_dt(end),
    }

    url = (
        f"{GDELT_ENDPOINT}?query={quote_plus(params['query'])}"
        f"&mode={params['mode']}&format={params['format']}"
        f"&maxrecords={params['maxrecords']}"
        f"&startdatetime={params['startdatetime']}"
        f"&enddatetime={params['enddatetime']}"
    )

    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    body = resp.json()
    articles = body.get("articles", [])
    if not isinstance(articles, list):
        return []
    return articles


def _fetch_window_with_retry(
    query: str,
    start: datetime,
    end: datetime,
    max_records: int,
    timeout_sec: int,
    retry: int,
    pause_seconds: float,
) -> list[dict]:
    last_error: Exception | None = None
    attempts = max(1, retry)

    for attempt in range(1, attempts + 1):
        try:
            return _fetch_window_once(
                query=query,
                start=start,
                end=end,
                max_records=max_records,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            last_error = exc
            wait = max(0.0, pause_seconds) * attempt
            logger.warning(
                "retry {}/{} failed for {} -> {} : {} (sleep {}s)",
                attempt,
                attempts,
                start.isoformat(),
                end.isoformat(),
                exc,
                round(wait, 2),
            )
            if wait > 0:
                time.sleep(wait)

    if last_error:
        raise last_error
    return []


def _normalize_articles(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        title = str(row.get("title") or "").strip()
        summary = str(row.get("seendate") or "").strip()
        source = str(row.get("sourcecountry") or row.get("domain") or "").strip()

        published_raw = str(
            row.get("seendate")
            or row.get("published")
            or row.get("date")
            or ""
        ).strip()
        published_at = _to_iso_utc(published_raw)

        if not published_at:
            # seendate が取れない場合は除外（時系列整合性優先）。
            continue

        if not title:
            title = "(no title)"

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

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=max(1, args.days))
    window = timedelta(hours=max(1, args.window_hours))

    all_rows: list[dict] = []
    cursor = start_utc

    logger.info("GDELT fetch start: {} -> {}", start_utc.isoformat(), now_utc.isoformat())

    while cursor < now_utc:
        nxt = min(now_utc, cursor + window)
        try:
            raw = _fetch_window_with_retry(
                query=args.query,
                start=cursor,
                end=nxt,
                max_records=args.max_records,
                timeout_sec=args.timeout,
                retry=args.retry,
                pause_seconds=args.pause_seconds,
            )
            normalized = _normalize_articles(raw)
            all_rows.extend(normalized)
            logger.info("window {} -> {} : raw={} normalized={}", cursor.isoformat(), nxt.isoformat(), len(raw), len(normalized))
        except Exception as exc:
            logger.warning("window fetch failed {} -> {} : {}", cursor.isoformat(), nxt.isoformat(), exc)
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)
        cursor = nxt

    dedup: dict[str, dict] = {}
    for row in all_rows:
        key = row.get("url") or f"{row.get('published_at')}|{row.get('title')}"
        if key not in dedup:
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
