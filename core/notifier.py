"""
Discord 通知モジュール

- 通知メッセージ内の時刻は表示基準（JST）で表示する。
- 保存基準（UTC）のタイムスタンプはフッターに付加。
"""

import asyncio
from enum import Enum

import aiohttp
from loguru import logger

from core.time_manager import format_jst, now_utc


class AlertLevel(Enum):
    INFO = "info"
    ALERT = "alert"
    CRITICAL = "critical"


class DiscordNotifier:
    """Discord Webhook を使った通知送信クラス。"""

    def __init__(self, webhook_url: str, critical_webhook_url: str | None = None):
        self._url = webhook_url
        self._critical_url = critical_webhook_url or webhook_url

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """メッセージを Discord に送信する。"""
        url = self._critical_url if level == AlertLevel.CRITICAL else self._url

        # 表示基準（JST）でタイムスタンプを付加
        timestamp_jst = format_jst(now_utc())
        prefix = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.ALERT: ":warning:",
            AlertLevel.CRITICAL: ":rotating_light: **CRITICAL**",
        }[level]

        payload = {
            "content": f"{prefix} [{timestamp_jst}]\n{message}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status in (200, 204):
                        return True
                    logger.warning(f"Discord notification failed: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False

    async def send_alert(self, message: str) -> bool:
        return await self.send(message, AlertLevel.ALERT)

    async def send_critical(self, message: str) -> bool:
        return await self.send(message, AlertLevel.CRITICAL)

    def send_sync(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """同期コンテキストから呼ぶためのラッパー。"""
        try:
            loop = asyncio.get_running_loop()
            return asyncio.ensure_future(self.send(message, level)) is not None
        except RuntimeError:
            return asyncio.run(self.send(message, level))
