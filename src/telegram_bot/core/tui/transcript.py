"""Transcript jsonl parsing + StreamEvent adapter for tmux-TUI mode.

`parse_jsonl_line` normalises one CC transcript line into a `ParsedEvent`.
`parse_transcript_event` is the adapter that turns a ParsedEvent into the
`StreamEvent` shape the bot's stream pipeline already knows how to render,
so tmux-TUI output reuses the same UI path as classic subprocess mode.

Coupling note: `parse_transcript_event` imports `_tool_status` from
`telegram_bot.core.services.claude` (and transitively relies on
`_smart_file_status`, `_smart_bash_status`, `_tool_status_map` inside it).
These are private (underscore-prefixed) but imported across the module
boundary intentionally so the tmux-TUI surface reuses the UX-consistent
status labels from subprocess mode. Any refactor of `_tool_status` or its
internal helpers in `claude.py` must update this module too.
TODO: consider promoting `_tool_status` to public `claude.tool_status`
in a later task.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from telegram_bot.core.messages import t
from telegram_bot.core.services.claude import StreamEvent, _tool_status

logger = logging.getLogger(__name__)

# `system` is intentionally NOT filtered wholesale — compact_boundary /
# status=compacting arrive as type=system,subtype=... and must reach the
# StreamEvent pipeline so the user sees compaction feedback in tmux mode.
# Unknown system subtypes collapse to skip via the explicit branch below.
FILTERED_TYPES = frozenset(
    {
        "permission-mode",
        "attachment",
        "file-history-snapshot",
    }
)

ParsedKind = Literal["text", "tool_use", "tool_result", "thinking", "status", "skip"]


@dataclass(frozen=True)
class ParsedEvent:
    """Normalised event surface for the bot's stream pipeline."""

    kind: ParsedKind
    payload: dict[str, Any]


def parse_jsonl_line(raw: str) -> ParsedEvent | None:
    """Parse one jsonl line. Return None only on malformed json.

    Filtered/system events become `ParsedEvent(kind="skip", ...)` with a
    reason hint so callers can warn-once on unknown types without
    reprocessing them.
    """
    try:
        evt = json.loads(raw)
    except json.JSONDecodeError:
        return None

    etype = evt.get("type")
    if etype in FILTERED_TYPES:
        return ParsedEvent(kind="skip", payload={"reason": etype})

    if etype == "system":
        # CC 2.1.116 writes compact lifecycle as type=system. Surface the
        # user-facing status events here; everything else (file-history,
        # ad-hoc diagnostics) collapses to skip so TmuxManager's warn-once
        # logic sees the reason and stays quiet.
        subtype = evt.get("subtype")
        if subtype == "status" and evt.get("status") == "compacting":
            return ParsedEvent(kind="status", payload={"text": t("ui.compacting")})
        if subtype == "compact_boundary":
            meta = evt.get("compactMetadata") or evt.get("compact_metadata") or {}
            pre = int(meta.get("preTokens", meta.get("pre_tokens", 0)))
            post = int(meta.get("postTokens", meta.get("post_tokens", 0)))
            return ParsedEvent(
                kind="status",
                payload={"text": t("ui.compact_done", pre=pre, post=post)},
            )
        return ParsedEvent(kind="skip", payload={"reason": f"system:{subtype or '?'}"})

    if etype == "user":
        content = evt.get("message", {}).get("content")
        if isinstance(content, str):
            return ParsedEvent(kind="skip", payload={"reason": "user_echo"})
        if isinstance(content, list):
            blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if blocks:
                return ParsedEvent(kind="tool_result", payload={"blocks": blocks})
        return ParsedEvent(kind="skip", payload={"reason": "user_unknown"})

    if etype == "assistant":
        content = evt.get("message", {}).get("content", [])
        # Assistant messages can mix blocks (e.g. `[thinking, text]` on Opus with
        # extended thinking). Scan all blocks and prefer real output — text then
        # tool_use — over thinking, otherwise the text gets silently dropped.
        text_block: dict[str, Any] | None = None
        tool_block: dict[str, Any] | None = None
        thinking_block: dict[str, Any] | None = None
        for block in content:
            btype = block.get("type")
            if btype == "text" and text_block is None:
                text_block = block
            elif btype == "tool_use" and tool_block is None:
                tool_block = block
            elif btype == "thinking" and thinking_block is None:
                thinking_block = block
        if text_block is not None:
            return ParsedEvent(kind="text", payload={"text": text_block.get("text", "")})
        if tool_block is not None:
            return ParsedEvent(
                kind="tool_use",
                payload={
                    "name": tool_block.get("name"),
                    "input": tool_block.get("input", {}),
                    "id": tool_block.get("id"),
                },
            )
        if thinking_block is not None:
            return ParsedEvent(
                kind="thinking",
                payload={"text": thinking_block.get("thinking", "")},
            )
        return ParsedEvent(kind="skip", payload={"reason": "assistant_empty"})

    return ParsedEvent(kind="skip", payload={"reason": f"unknown:{etype}"})


def tail_transcript(path: Path) -> list[ParsedEvent]:
    """One-shot read of a transcript file → list of ParsedEvent.

    Used in tests for deterministic assertions. The production tail is
    incremental (offset-based), but reuses `parse_jsonl_line` per line.
    """
    if not path.exists():
        return []
    out: list[ParsedEvent] = []
    for raw in path.read_text().splitlines():
        parsed = parse_jsonl_line(raw)
        if parsed is not None:
            out.append(parsed)
    return out


class ClaudeTranscriptParser:
    """Stateful Claude JSONL parser with provider-native turn boundaries."""

    _NON_TERMINAL_REASONS: ClassVar[set[str]] = {
        "tool_use",
        "pause_turn",
        "compaction",
    }
    _TERMINAL_REASONS: ClassVar[set[str]] = {
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "model_context_window_exceeded",
        "refusal",
    }

    def __init__(self) -> None:
        self._turn_seq = 0
        self._turn_id: str | None = None
        self._pending_end_turn_request: str | None = None
        self._pending_terminal_request: str | None = None
        self._pending_terminal_text = ""
        self._pending_terminal_reason: str | None = None
        self._ambiguous_completion = False
        self._last_terminal_request: str | None = None

    @property
    def current_turn_id(self) -> str | None:
        return self._turn_id

    @property
    def has_pending_terminal(self) -> bool:
        return bool(self._pending_terminal_text)

    @property
    def has_unfinished_completion(self) -> bool:
        return (
            self.has_pending_terminal
            or self._pending_end_turn_request is not None
            or self._ambiguous_completion
        )

    @staticmethod
    def is_turn_boundary(raw: str) -> bool:
        data = _load_object(raw)
        return data is not None and _is_top_level_user(data)

    def parse(self, raw: str) -> tuple[list[StreamEvent], str | None]:
        data = _load_object(raw)
        if data is None:
            return [], None

        session_id = data.get("sessionId")
        sid = session_id if isinstance(session_id, str) else None
        event_type = data.get("type")
        if event_type == "user":
            if not _is_top_level_user(data):
                return [], sid
            events = self.flush_pending_terminal()
            events.extend(self._close_unfinished_turn())
            self._turn_seq += 1
            self._turn_id = f"claude:{self._turn_seq}"
            self._pending_end_turn_request = None
            self._ambiguous_completion = False
            self._last_terminal_request = None
            events.append(StreamEvent("turn_start", "", turn_id=self._turn_id))
            return events, sid

        if event_type == "system":
            parsed = parse_jsonl_line(raw)
            if parsed is not None and parsed.kind == "status":
                text = str(parsed.payload.get("text", ""))
                return self._with_turn([StreamEvent("status", text)] if text else []), sid
            return [], sid

        if event_type != "assistant":
            return [], sid

        message = data.get("message")
        if not isinstance(message, dict):
            return [], sid
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, list) else []
        stop_reason = message.get("stop_reason")
        request_id = data.get("requestId")
        request = request_id if isinstance(request_id, str) else None
        if (
            stop_reason in self._TERMINAL_REASONS
            and request is not None
            and request == self._last_terminal_request
            and self._turn_id is None
        ):
            logger.warning(
                "Ignoring duplicate Claude terminal fragment request_id=%s",
                request,
            )
            return [], sid

        turn_id = self._ensure_turn()

        text_parts: list[str] = []
        ordered_progress: list[StreamEvent] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                block_text = block.get("text")
                if isinstance(block_text, str) and block_text:
                    text_parts.append(block_text)
                    ordered_progress.append(StreamEvent("text", block_text, turn_id=turn_id))
            elif block_type == "tool_use":
                name = block.get("name")
                tool_input = block.get("input")
                ordered_progress.append(
                    StreamEvent(
                        "status",
                        _tool_status(
                            str(name or ""),
                            tool_input if isinstance(tool_input, dict) else None,
                        ),
                        turn_id=turn_id,
                    )
                )

        text = "".join(text_parts)
        if stop_reason in self._NON_TERMINAL_REASONS:
            return ordered_progress, sid

        if stop_reason in self._TERMINAL_REASONS:
            progress = [event for event in ordered_progress if event.type == "status"]
            if stop_reason == "end_turn" and not text:
                self._pending_end_turn_request = request
                return progress, sid
            if not text:
                self._last_terminal_request = request
                return [*progress, *self._finish_with(_terminal_diagnostic(str(stop_reason)))], sid

            self._pending_end_turn_request = None
            self._ambiguous_completion = False
            if request == self._pending_terminal_request:
                self._pending_terminal_text = _merge_generation_text(
                    self._pending_terminal_text,
                    text,
                )
            else:
                self._pending_terminal_request = request
                self._pending_terminal_text = text
            self._pending_terminal_reason = str(stop_reason)
            return progress, sid

        logger.warning(
            "Unknown Claude stop_reason %r request_id=%s turn_id=%s",
            stop_reason,
            request,
            turn_id,
        )
        self._ambiguous_completion = True
        return ordered_progress, sid

    def finish(self) -> list[StreamEvent]:
        """Close an unfinished turn when the transcript transport stops."""
        events = self.flush_pending_terminal()
        events.extend(self._close_unfinished_turn())
        return events

    def flush_pending_terminal(self) -> list[StreamEvent]:
        """Finalize terminal text after all adjacent generation rows were parsed."""
        if not self._pending_terminal_text:
            return []
        final_text = self._pending_terminal_text
        reason = self._pending_terminal_reason
        request = self._pending_terminal_request
        self._pending_terminal_text = ""
        self._pending_terminal_reason = None
        self._pending_terminal_request = None
        if reason == "max_tokens":
            final_text += t("ui.claude_limit_note")
        elif reason == "model_context_window_exceeded":
            final_text += t("ui.claude_context_note")
        self._last_terminal_request = request
        return self._finish_with(final_text)

    def _ensure_turn(self) -> str:
        if self._turn_id is None:
            self._turn_seq += 1
            self._turn_id = f"claude:{self._turn_seq}"
        return self._turn_id

    def _with_turn(self, events: list[StreamEvent]) -> list[StreamEvent]:
        turn_id = self._ensure_turn()
        for event in events:
            event.turn_id = turn_id
        return events

    def _finish_with(self, content: str) -> list[StreamEvent]:
        turn_id = self._turn_id
        if turn_id is None:
            return []
        self._turn_id = None
        self._pending_end_turn_request = None
        self._ambiguous_completion = False
        return [
            StreamEvent("result_message", content, turn_id=turn_id),
            StreamEvent("turn_end", "", turn_id=turn_id),
        ]

    def _close_unfinished_turn(self) -> list[StreamEvent]:
        if self._turn_id is None:
            return []
        if self._ambiguous_completion:
            return self._finish_with(t("ui.claude_unknown_completion"))
        return self._finish_with(t("ui.claude_no_text"))


def _load_object(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_top_level_user(data: dict[str, Any]) -> bool:
    if data.get("type") != "user":
        return False
    message = data.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return not any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )


def _terminal_diagnostic(stop_reason: str) -> str:
    return {
        "end_turn": t("ui.claude_no_text"),
        "stop_sequence": t("ui.claude_stop_sequence_no_text"),
        "max_tokens": t("ui.claude_limit_no_text"),
        "model_context_window_exceeded": t("ui.claude_context_no_text"),
        "refusal": t("ui.claude_refused_no_text"),
    }[stop_reason]


def _merge_generation_text(current: str, fragment: str) -> str:
    """Merge adjacent Claude snapshots/fragments without duplicating prefixes."""
    if not current:
        return fragment
    if fragment == current or current.endswith(fragment):
        return current
    if fragment.startswith(current):
        return fragment
    return current + fragment


def parse_transcript_event(
    raw: str,
) -> tuple[list[StreamEvent], str | None]:
    """Adapt one transcript jsonl line to the bot's StreamEvent pipeline.

    Returns `(events, session_id)` mirroring `claude.parse_cc_event`'s shape.
    `thinking`, `tool_result`, `skip`, and unknown types collapse to `([], None)`
    — the caller (TmuxManager) owns warn-once bookkeeping for unknown types.
    """
    parsed = parse_jsonl_line(raw)
    if parsed is None:
        return [], None

    # session_id is only surfaced when the raw event carries an explicit field.
    # Not observed in PoC on CC 2.1.114 but the defensive read is cheap.
    session_id: str | None = None
    try:
        evt = json.loads(raw)
        sid = evt.get("sessionId")
        if isinstance(sid, str):
            session_id = sid
    except json.JSONDecodeError:
        pass

    if parsed.kind in ("skip", "thinking", "tool_result"):
        return [], session_id

    if parsed.kind == "text":
        text = parsed.payload.get("text", "")
        if not text:
            return [], session_id
        return [StreamEvent("text", text)], session_id

    if parsed.kind == "tool_use":
        name = parsed.payload.get("name", "")
        tool_input = parsed.payload.get("input")
        status = _tool_status(name, tool_input if isinstance(tool_input, dict) else None)
        return [StreamEvent("status", status)], session_id

    if parsed.kind == "status":
        text = parsed.payload.get("text", "")
        if not text:
            return [], session_id
        return [StreamEvent("status", text)], session_id

    return [], session_id
