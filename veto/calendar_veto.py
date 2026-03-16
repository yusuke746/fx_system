"""
Calendar Veto モジュール（Veto Layer A）

Forex Factory XML から高インパクト経済指標を取得し、
FOMC・雇用統計等の前後30分をブロックする。

■ 時刻基準
  - カレンダーのイベント日時: Forex Factory は EST（UTC-5）で提供。
    XMLの日時を UTC に変換してから保存・比較する。
  - Veto 判定: 比較基準（UTC）で行う。
  - Discord 通知: 表示基準（JST）で表示する。
"""

import importlib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from loguru import logger

from core.time_manager import UTC, now_utc, is_within_buffer, format_jst

ET = importlib.import_module("defusedxml.ElementTree")

FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FOREX_FACTORY_NEXT_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"

HIGH_IMPACT_CURRENCIES = {"USD", "JPY", "EUR", "GBP"}
HIGH_IMPACT_LABEL = "High"

PAIR_CURRENCIES = {
    "USDJPY": {"USD", "JPY"},
    "EURUSD": {"EUR", "USD"},
    "GBPJPY": {"GBP", "JPY"},
}

VETO_BUFFER_MINUTES = 30


class CalendarVeto:
    """経済指標カレンダーに基づく Veto 判定（Layer A）。"""

    def __init__(self, cache_dir: str = "data/calendar"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._last_fetch: datetime | None = None

    def fetch_events(self) -> list[dict]:
        """
        Forex Factory XML から今週の高インパクト指標を取得する。
        取得した日時は UTC（保存基準）に変換して保持する。
        """
        events = []
        for url in [FOREX_FACTORY_URL, FOREX_FACTORY_NEXT_URL]:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                events.extend(self._parse_xml(resp.content))
            except Exception as e:
                logger.warning(f"Calendar fetch failed ({url}): {e}")

        if events:
            self._events = events
            self._last_fetch = now_utc()
            logger.info(f"Calendar events fetched: {len(events)} high-impact events")
            return events

        logger.warning("Calendar fetch returned 0 events. Keeping previous cache and freshness.")
        return self._events

    def _parse_xml(self, content: bytes) -> list[dict]:
        """XML をパースして高インパクトイベントを抽出する。"""
        root = ET.fromstring(content)
        events = []
        ny_tz = ZoneInfo("America/New_York")

        for event in root.findall("event"):
            currency = event.findtext("country", "")
            impact = event.findtext("impact", "")
            title = event.findtext("title", "")
            date_str = event.findtext("date", "")
            time_str = event.findtext("time", "")

            if currency not in HIGH_IMPACT_CURRENCIES:
                continue
            if impact != HIGH_IMPACT_LABEL:
                continue

            # Forex Factory は EST（UTC-5）で日時を提供
            try:
                dt_str = f"{date_str} {time_str}"
                dt_local = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                # ニューヨーク現地時刻として解釈し、UTCへ正規変換（DST考慮）
                dt_utc = dt_local.replace(tzinfo=ny_tz).astimezone(UTC)
            except ValueError:
                continue  # 時刻未定のイベントはスキップ

            events.append({
                "currency": currency,
                "title": title,
                "datetime_utc": dt_utc,
                "impact": impact,
            })

        return events

    @property
    def events(self) -> list[dict]:
        return self._events

    @property
    def cache_age_hours(self) -> float | None:
        """キャッシュ経過時間（時間）。未取得の場合は None。"""
        if self._last_fetch is None:
            return None
        return (now_utc() - self._last_fetch).total_seconds() / 3600

    def is_cache_stale(self, max_hours: float = 24) -> bool:
        """キャッシュが古い（24時間超）かどうか。"""
        age = self.cache_age_hours
        return age is None or age > max_hours

    def is_veto_active(self, pair: str, now: datetime | None = None) -> tuple[bool, str]:
        """
        対象ペアの通貨に関わる高インパクト指標の前後 VETO_BUFFER_MINUTES 以内か判定。

        Args:
            pair: 通貨ペア名（例: "USDJPY"）
            now: 比較基準時刻（UTC）。None の場合は現在時刻。

        Returns:
            (is_blocked, reason_or_empty)
        """
        if now is None:
            now = now_utc()

        currencies = PAIR_CURRENCIES.get(pair, set())
        if not currencies:
            return False, ""

        for ev in self._events:
            if ev["currency"] not in currencies:
                continue
            if is_within_buffer(ev["datetime_utc"], now, VETO_BUFFER_MINUTES):
                reason = (
                    f"Calendar Veto: {ev['title']} ({ev['currency']}) "
                    f"at {format_jst(ev['datetime_utc'])}"
                )
                return True, reason

        return False, ""

    def get_upcoming_events(self, pair: str, hours: int = 4) -> list[dict]:
        """直近 N 時間以内の高インパクトイベントを取得する（表示用）。"""
        now = now_utc()
        cutoff = now + timedelta(hours=hours)
        currencies = PAIR_CURRENCIES.get(pair, set())
        return [
            ev for ev in self._events
            if ev["currency"] in currencies and now <= ev["datetime_utc"] <= cutoff
        ]
