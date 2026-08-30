"""One-shot, process-local permissions for Codex web research."""

from __future__ import annotations

import threading

from telegram_bot.core.types import ChannelKey


class ResearchGrantStore:
    """Track channels whose next Codex task may use live web search.

    Grants intentionally live only in memory. A bot restart therefore drops
    every unused grant instead of silently carrying an authorization forward.
    """

    def __init__(self) -> None:
        self._pending: set[ChannelKey] = set()
        self._lock = threading.Lock()

    def arm(self, channel_key: ChannelKey) -> None:
        with self._lock:
            self._pending.add(channel_key)

    def is_pending(self, channel_key: ChannelKey) -> bool:
        with self._lock:
            return channel_key in self._pending

    def consume(self, channel_key: ChannelKey) -> bool:
        with self._lock:
            if channel_key not in self._pending:
                return False
            self._pending.remove(channel_key)
            return True

    def clear(self, channel_key: ChannelKey) -> None:
        with self._lock:
            self._pending.discard(channel_key)
