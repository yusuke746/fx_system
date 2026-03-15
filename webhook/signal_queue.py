"""Webhook シグナルキュー（Orchestrator との連携用）。"""

import asyncio

signal_queue: asyncio.Queue = asyncio.Queue()
