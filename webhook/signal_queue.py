"""Webhook シグナルキュー（Orchestrator との連携用）。"""

import asyncio

_signal_queue: asyncio.Queue | None = None


def get_queue() -> asyncio.Queue:
	global _signal_queue
	if _signal_queue is None:
		_signal_queue = asyncio.Queue()
	return _signal_queue
