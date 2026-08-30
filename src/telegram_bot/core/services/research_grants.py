"""One-shot, process-local permissions for Codex web research."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import Any

from telegram_bot.core.types import ChannelKey

RESEARCH_PERMISSION_MARKER = "[[RESEARCH_PERMISSION_REQUIRED]]"


def wrap_with_research_policy(prompt: str, *, web_search_enabled: bool) -> str:
    """Tell Codex how to request approval without attempting network work."""
    if web_search_enabled:
        return prompt
    return (
        "<web-research-policy>\n"
        "Live web search is not authorized for this task. Do not use the web-search tool "
        "or network-capable shell commands such as curl or wget. If the task materially "
        "requires current or external information, stop and make your entire final answer "
        f"start with {RESEARCH_PERMISSION_MARKER} on its own line, followed by one concise "
        "sentence explaining what must be researched. Do not guess or continue the task. "
        "If the task can be completed without external information, proceed normally.\n"
        "</web-research-policy>\n\n"
        f"{prompt}"
    )


def research_request_reason(text: str) -> str | None:
    """Return the reason from a strict permission marker, otherwise None."""
    stripped = text.strip()
    if not stripped.startswith(RESEARCH_PERMISSION_MARKER):
        return None
    reason = stripped[len(RESEARCH_PERMISSION_MARKER) :].strip()
    return reason or None


@dataclass(frozen=True)
class PendingResearchRequest:
    token: str
    prompt: str
    source_message: Any


class ResearchGrantStore:
    """Track channels whose next Codex task may use live web search.

    Grants intentionally live only in memory. A bot restart therefore drops
    every unused grant instead of silently carrying an authorization forward.
    """

    def __init__(self) -> None:
        self._pending: set[ChannelKey] = set()
        self._approvals: dict[ChannelKey, PendingResearchRequest] = {}
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
            self._approvals.pop(channel_key, None)

    def request_approval(
        self,
        channel_key: ChannelKey,
        *,
        prompt: str,
        source_message: Any,
    ) -> PendingResearchRequest:
        request = PendingResearchRequest(
            token=token_urlsafe(6),
            prompt=prompt,
            source_message=source_message,
        )
        with self._lock:
            self._approvals[channel_key] = request
        return request

    def take_approval(self, channel_key: ChannelKey, token: str) -> PendingResearchRequest | None:
        with self._lock:
            request = self._approvals.get(channel_key)
            if request is None or request.token != token:
                return None
            return self._approvals.pop(channel_key)

    def decline_approval(self, channel_key: ChannelKey, token: str) -> bool:
        return self.take_approval(channel_key, token) is not None
