"""Incoming Telegram Rich Message normalization contract.

The normalizer converts aiogram ``Message.rich_message`` blocks into one
agent-readable Markdown-like text representation. It is intentionally generic:
handlers decide how to wrap the resulting text for prompts, inbox files, or
reply context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import Mock

from aiogram.utils.text_decorations import HtmlDecoration

from telegram_bot.core.messages import t

if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import Message

    from telegram_bot.core.services.transcriber import Transcriber

logger = logging.getLogger(__name__)
_html_decorator = HtmlDecoration()

ContentSource = Literal["rich", "legacy", "mixed", "media", "empty"]

_MEDIA_PLACEHOLDER_TYPES = {
    "collage",
    "slideshow",
    "map",
    "animation",
    "audio",
    "photo",
    "video",
    "voice_note",
    "unknown",
}
_TEXT_BLOCK_TYPES = {
    "paragraph",
    "heading",
    "pre",
    "footer",
    "divider",
    "mathematical_expression",
    "anchor",
    "list",
    "blockquote",
    "pullquote",
    "table",
    "details",
    "thinking",
}
_KNOWN_BLOCK_TYPES = _TEXT_BLOCK_TYPES | _MEDIA_PLACEHOLDER_TYPES
_FORWARDED_TAG_RE = re.compile(r"</?forwarded-data(?:-[0-9a-f]+)?\b[^>]*>", re.IGNORECASE)
_AUTHOR_HEADER_RE = re.compile(r"^\*\*\d{2}:\d{2} .+?:\*\*")
_SAFE_TOKEN_RE = re.compile(r"^[a-z0-9_]{1,32}$")


@dataclass(slots=True)
class RichContentAttachment:
    type: Literal["photo"]
    block_position: int
    file_id: str
    file_unique_id: str | None
    file_size: int | None


@dataclass(slots=True)
class NormalizedTelegramContent:
    text: str
    source: ContentSource
    warnings: list[str]
    has_rich: bool
    has_media_placeholder: bool
    legacy_text_used: bool
    is_edited: bool
    truncated: bool
    attachments: list[RichContentAttachment] = field(default_factory=list)


@dataclass(slots=True)
class RichContentLimits:
    total_chars: int = 16_000
    block_count: int = 500
    nesting_depth: int = 12
    table_rows: int = 100
    table_cols: int = 20
    per_cell_chars: int = 1_000
    placeholder_count: int = 200
    combined_output_chars: int = 16_000
    raw_payload_bytes: int = 256_000
    raw_payload_blocks: int = 1_000


@dataclass(slots=True)
class _RenderState:
    limits: RichContentLimits
    warnings: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    has_media_placeholder: bool = False
    truncated: bool = False
    stopped: bool = False
    stop_after_current: bool = False
    block_count: int = 0
    placeholder_count: int = 0
    budget_hits: set[str] = field(default_factory=set)
    attachments: list[RichContentAttachment] = field(default_factory=list)

    def current_output_len(self) -> int:
        return sum(len(part) for part in self.parts) + max(len(self.parts) - 1, 0) * 2

    def remaining_output_chars(self) -> int:
        current = self.current_output_len()
        return max(self.output_limit() - current, 0)

    def output_limit(self) -> int:
        return min(self.limits.total_chars, self.limits.combined_output_chars)

    def would_exceed_output(self, text_len: int) -> str | None:
        current = self.current_output_len()
        separator = 2 if self.parts else 0
        projected = current + separator + text_len
        if projected > self.limits.total_chars:
            return "total_chars"
        if projected > self.limits.combined_output_chars:
            return "combined_output_chars"
        return None

    def append(self, text: str, *, block_type: str, position: int) -> None:
        if not text or self.stopped:
            return

        current = self.current_output_len()
        separator = 2 if self.parts else 0
        projected = current + separator + len(text)
        for budget_name, limit in (
            ("total_chars", self.limits.total_chars),
            ("combined_output_chars", self.limits.combined_output_chars),
        ):
            if projected > limit:
                available = max(limit - current - separator, 0)
                if available:
                    self.parts.append(text[:available].rstrip())
                self.add_budget_hit(
                    budget_name,
                    block_type=block_type,
                    position=position,
                    reason="output-ceiling",
                )
                self.stopped = True
                return

        self.parts.append(text)

    def add_placeholder(self, block_type: str, position: int, *, label: str | None = None) -> str:
        self.placeholder_count += 1
        if self.placeholder_count > self.limits.placeholder_count:
            self.add_budget_hit(
                "placeholder_count",
                block_type=block_type,
                position=position,
                reason="too-many-placeholders",
            )
            self.stopped = True
            return ""
        self.has_media_placeholder = True
        return f"[rich block {position}: {label or block_type}]"

    def add_warning(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    def add_budget_hit(
        self,
        budget_name: str,
        *,
        block_type: str,
        position: int,
        reason: str,
    ) -> None:
        if budget_name in self.budget_hits:
            return
        self.record_budget_hit(
            budget_name,
            block_type=block_type,
            position=position,
            reason=reason,
        )
        marker = _budget_marker(budget_name)
        self._append_marker_within_limit(marker)

    def record_budget_hit(
        self,
        budget_name: str,
        *,
        block_type: str,
        position: int,
        reason: str,
    ) -> None:
        if budget_name in self.budget_hits:
            return
        self.budget_hits.add(budget_name)
        self.truncated = True
        warning = f"budget-exceeded:{budget_name}:{block_type}:{position}:{reason}"
        self.add_warning(warning)
        logger.warning(
            "rich-budget-exceeded block_type=%s position=%s budget=%s reason=%s",
            block_type,
            position,
            budget_name,
            reason,
        )

    def _append_marker_within_limit(self, marker: str) -> None:
        limit = self.output_limit()
        if limit <= 0:
            self.parts = []
            return
        if len(marker) >= limit:
            self.parts = [marker[:limit]]
            return

        current_text = "\n\n".join(part for part in self.parts if part)
        if not current_text:
            self.parts = [marker]
            return

        separator = "\n\n"
        max_content_len = limit - len(separator) - len(marker)
        if max_content_len <= 0:
            self.parts = [marker]
            return
        content = current_text[:max_content_len].rstrip()
        self.parts = [content, marker] if content else [marker]


async def normalize_telegram_content(
    message: Message,
    *,
    bot: Bot | None = None,
    transcriber: Transcriber | None = None,
    limits: RichContentLimits | None = None,
) -> NormalizedTelegramContent:
    """Normalize any incoming Telegram message to Markdown-like content."""
    rich_message = _safe_getattr(message, "rich_message")
    is_edited = bool(_safe_getattr(message, "edit_date"))
    if not rich_message:
        return await _normalize_legacy_only(
            message,
            bot=bot,
            transcriber=transcriber,
            is_edited=is_edited,
        )

    return _normalize_rich_message(
        message,
        rich_message,
        is_edited=is_edited,
        limits=limits,
    )


def normalize_telegram_text_content(
    message: Message,
    *,
    limits: RichContentLimits | None = None,
) -> NormalizedTelegramContent:
    """Normalize only text/caption/rich text fields without media I/O.

    Use this for synchronous boundaries such as reply quotes, edited-message
    updates, and media caption formatting. Voice/media transcription remains in
    ``normalize_telegram_content`` / ``extract_content``.
    """
    rich_message = _safe_getattr(message, "rich_message")
    is_edited = bool(_safe_getattr(message, "edit_date"))
    if rich_message:
        return _normalize_rich_message(
            message,
            rich_message,
            is_edited=is_edited,
            limits=limits,
        )

    text = _legacy_text_with_entities(message)
    if not text:
        return NormalizedTelegramContent(
            text="",
            source="empty",
            warnings=[],
            has_rich=False,
            has_media_placeholder=False,
            legacy_text_used=False,
            is_edited=is_edited,
            truncated=False,
        )
    return NormalizedTelegramContent(
        text=text,
        source="legacy",
        warnings=[],
        has_rich=False,
        has_media_placeholder=False,
        legacy_text_used=True,
        is_edited=is_edited,
        truncated=False,
    )


def _normalize_rich_message(
    message: Message,
    rich_message: object,
    *,
    is_edited: bool,
    limits: RichContentLimits | None,
) -> NormalizedTelegramContent:
    active_limits = limits or RichContentLimits()
    legacy_compare_text = extract_plain_text_only_content(message)
    legacy_markup_text = _legacy_text_with_entities(message)

    try:
        blocks = _rich_blocks(rich_message)
    except Exception:
        _log_parse_failure("message", 0, "catastrophic")
        fallback_markup = _legacy_text_with_entities(message)
        fallback_media = _legacy_media_placeholder(message)
        fallback_text = fallback_markup or fallback_media or ""
        fallback_source: ContentSource = "empty"
        if fallback_media and not fallback_markup:
            fallback_source = "media"
        elif fallback_text:
            fallback_source = "legacy"
        text = "[rich parse failed]"
        if fallback_text:
            text = f"{text}\n{fallback_text}"
        return NormalizedTelegramContent(
            text=text,
            source=fallback_source,
            warnings=["parse-failure:catastrophic"],
            has_rich=True,
            has_media_placeholder=bool(fallback_media),
            legacy_text_used=bool(fallback_markup),
            is_edited=is_edited,
            truncated=False,
        )

    state = _RenderState(active_limits)
    if _raw_payload_exceeds(rich_message, blocks, active_limits):
        state.add_budget_hit(
            "raw_payload",
            block_type="message",
            position=0,
            reason="pre-parse-guard",
        )
    else:
        _render_blocks(blocks, state, depth=0)

    rich_text = "\n\n".join(part for part in state.parts if part).strip()
    source: ContentSource = "rich"
    legacy_used = False
    if legacy_compare_text and not _contains_without_duplication(rich_text, legacy_compare_text):
        extra = _legacy_extra_text(rich_text, legacy_markup_text or legacy_compare_text)
        if extra:
            rich_text = f"{rich_text}\n\n{extra}" if rich_text else extra
            source = "mixed"
            legacy_used = True

    return NormalizedTelegramContent(
        text=rich_text,
        source=source,
        warnings=state.warnings,
        has_rich=True,
        has_media_placeholder=state.has_media_placeholder,
        legacy_text_used=legacy_used,
        is_edited=is_edited,
        truncated=state.truncated,
        attachments=state.attachments,
    )


def extract_plain_text_only_content(message: Message) -> str | None:
    """Return only legacy text/caption, ignoring rich-only content."""
    text = _safe_getattr(message, "text")
    if text:
        return str(text)
    caption = _safe_getattr(message, "caption")
    if caption:
        return str(caption)
    return None


def normalize_plain_text_only_content(message: Message) -> NormalizedTelegramContent:
    """Normalized wrapper around the explicit plain-text-only control path."""
    text = extract_plain_text_only_content(message)
    return NormalizedTelegramContent(
        text=text or "",
        source="legacy" if text else "empty",
        warnings=[],
        has_rich=False,
        has_media_placeholder=False,
        legacy_text_used=bool(text),
        is_edited=bool(_safe_getattr(message, "edit_date")),
        truncated=False,
    )


def _legacy_text_with_entities(message: Message) -> str | None:
    text = _safe_getattr(message, "text")
    if text:
        entities = _safe_getattr(message, "entities")
        return _html_decorator.unparse(str(text), entities) if entities else str(text)
    caption = _safe_getattr(message, "caption")
    if caption:
        caption_entities = _safe_getattr(message, "caption_entities")
        return (
            _html_decorator.unparse(str(caption), caption_entities)
            if caption_entities
            else str(caption)
        )
    return None


def sanitize_forwarded_boundary_text(text: str) -> str:
    """Neutralize forwarded-data delimiter tags across multi-line rich bodies."""
    return _FORWARDED_TAG_RE.sub(lambda match: match.group().replace("<", "&lt;"), text)


def sanitize_inbox_boundary_text(text: str) -> str:
    """Neutralize inbox HTML comment markers."""
    return text.replace("<!--", "\\<!--")


def sanitize_reply_boundary_text(text: str) -> str:
    """Neutralize reply-prefix brackets for boundary prefix fields."""
    return text.replace("[", "(").replace("]", ")")


def sanitize_meeting_boundary_text(text: str) -> str:
    """Neutralize meeting transcript wrappers and routing sentinels."""
    return (
        text.replace("<chat-transcript>", "")
        .replace("</chat-transcript>", "")
        .replace("[ГОТОВО]", "[готово]")
        .replace("[ВОПРОС]", "[вопрос]")
    )


def mark_author_header_lines(text: str, *, prefix: str = "> ") -> str:
    """Prefix physical lines that could forge an inbox author header."""
    lines = []
    for line in text.splitlines():
        if _AUTHOR_HEADER_RE.match(line):
            lines.append(f"{prefix}{line}")
        else:
            lines.append(line)
    return "\n".join(lines)


async def _normalize_legacy_only(
    message: Message,
    *,
    bot: Bot | None,
    transcriber: Transcriber | None,
    is_edited: bool,
) -> NormalizedTelegramContent:
    text = _legacy_text_with_entities(message)
    if text and not _has_legacy_media(message):
        return NormalizedTelegramContent(
            text=text,
            source="legacy",
            warnings=[],
            has_rich=False,
            has_media_placeholder=False,
            legacy_text_used=True,
            is_edited=is_edited,
            truncated=False,
        )

    media_text: str | None = None
    if bot is not None and transcriber is not None:
        from telegram_bot.core.services.content import extract_legacy_content

        media_text = await extract_legacy_content(message, bot, transcriber)
    else:
        media_text = _legacy_media_placeholder(message)

    if media_text:
        return NormalizedTelegramContent(
            text=media_text,
            source="media",
            warnings=[],
            has_rich=False,
            has_media_placeholder=True,
            legacy_text_used=False,
            is_edited=is_edited,
            truncated=False,
        )

    return NormalizedTelegramContent(
        text="",
        source="empty",
        warnings=[],
        has_rich=False,
        has_media_placeholder=False,
        legacy_text_used=False,
        is_edited=is_edited,
        truncated=False,
    )


def _legacy_media_placeholder(message: Message) -> str | None:
    if _safe_getattr(message, "photo"):
        caption = _safe_getattr(message, "caption") or ""
        return f"{t('cc.photo')} {caption}".strip()
    if _safe_getattr(message, "video"):
        return t("cc.video")
    document = _safe_getattr(message, "document")
    if document:
        filename = _safe_getattr(document, "file_name")
        if filename:
            from telegram_bot.core.utils.fs import sanitize_filename

            return t("cc.document_named", name=sanitize_filename(str(filename)))
        return t("cc.document")
    sticker = _safe_getattr(message, "sticker")
    if sticker:
        return t("cc.sticker", emoji=_safe_getattr(sticker, "emoji") or "")
    if _safe_getattr(message, "voice"):
        return f"[{t('cc.voice_label')}]"
    if _safe_getattr(message, "video_note"):
        return f"[{t('cc.videomessage_label')}]"
    return None


def _has_legacy_media(message: Message) -> bool:
    return any(
        _safe_getattr(message, field_name)
        for field_name in ("photo", "video", "document", "sticker", "voice", "video_note")
    )


def _render_blocks(blocks: list[object], state: _RenderState, *, depth: int) -> list[str]:
    rendered: list[str] = []
    if depth > state.limits.nesting_depth:
        state.add_budget_hit(
            "nesting_depth",
            block_type="nested",
            position=state.block_count + 1,
            reason="max-depth",
        )
        state.stopped = True
        return rendered

    for block in blocks:
        if state.stopped:
            break
        state.block_count += 1
        position = state.block_count
        block_type = _block_type(block)
        if position > state.limits.block_count:
            state.add_budget_hit(
                "block_count",
                block_type=block_type,
                position=position,
                reason="too-many-blocks",
            )
            state.stopped = True
            break
        text = _render_block(block, state, position=position, depth=depth)
        if text:
            if depth == 0:
                state.append(text, block_type=block_type, position=position)
            else:
                rendered.append(text)
            if state.stop_after_current:
                state.stopped = True
                break
    return rendered


def _render_block(block: object, state: _RenderState, *, position: int, depth: int) -> str:
    block_type = _block_type(block)
    try:
        match block_type:
            case "paragraph":
                return _required_text_block(block, "text", state, position, block_type)
            case "heading":
                text = _required_text_block(block, "text", state, position, block_type)
                size = _safe_getattr(block, "size") or 1
                level = min(max(int(size), 1), 6)
                return f"{'#' * level} {text}"
            case "pre":
                text = _required_text_block(block, "text", state, position, block_type)
                language = _safe_getattr(block, "language") or ""
                return f"```{language}\n{text}\n```"
            case "footer":
                return (
                    f"Footnote: {_required_text_block(block, 'text', state, position, block_type)}"
                )
            case "divider":
                return "---"
            case "anchor":
                name = _safe_getattr(block, "name")
                if not name:
                    return _malformed_placeholder(block_type, state, position)
                return f"[anchor: {name}]"
            case "mathematical_expression":
                expression = _safe_getattr(block, "expression")
                if not expression:
                    return _malformed_placeholder(block_type, state, position)
                return f"```math\n{expression}\n```"
            case "table":
                return _render_table(block, state, position)
            case "list":
                return _render_list(block, state, position, depth)
            case "blockquote":
                return _render_quote(block, state, position, depth, credit_field="credit")
            case "pullquote":
                text = _required_text_block(block, "text", state, position, block_type)
                credit = _safe_getattr(block, "credit")
                suffix = f" — {credit}" if credit else ""
                return "\n".join(
                    f"> {line}{suffix if i == 0 else ''}"
                    for i, line in enumerate(text.splitlines())
                )
            case "details":
                return _render_details(block, state, position, depth)
            case "thinking":
                text = _safe_getattr(block, "text")
                if not text:
                    return ""
                placeholder = state.add_placeholder(block_type, position)
                state.add_warning(f"unsupported-block:{block_type}:{position}")
                return placeholder
            case _ if block_type in _MEDIA_PLACEHOLDER_TYPES:
                placeholder = _media_placeholder_with_caption(block, block_type, state, position)
                state.add_warning(f"unsupported-block:{block_type}:{position}")
                return placeholder
            case _:
                placeholder = state.add_placeholder(block_type, position)
                state.add_warning(f"unsupported-block:{block_type}:{position}")
                return placeholder
    except Exception:
        return _malformed_placeholder(block_type, state, position)


def _media_placeholder_with_caption(
    block: object,
    block_type: str,
    state: _RenderState,
    position: int,
) -> str:
    placeholder = state.add_placeholder(block_type, position)
    caption = _safe_getattr(block, "caption")
    _maybe_add_rich_attachment(
        block,
        block_type=block_type,
        state=state,
        position=position,
    )
    if not caption:
        return placeholder
    prefix = f"{placeholder}\nCaption: "
    current = state.current_output_len()
    separator = 2 if state.parts else 0
    output_limit = min(state.limits.total_chars, state.limits.combined_output_chars)
    available = output_limit - current - separator
    budget_name = (
        "combined_output_chars"
        if state.limits.combined_output_chars <= state.limits.total_chars
        else "total_chars"
    )
    marker = _budget_marker(budget_name)
    max_caption_chars = max(available - len(prefix), 0)
    probe = _render_caption(caption, max_chars=max_caption_chars + 1)
    if len(probe) > max_caption_chars:
        max_caption_chars = max(available - len(prefix) - 1 - len(marker), 0)
        caption_text = _render_caption(caption, max_chars=max_caption_chars).rstrip()
        state.record_budget_hit(
            budget_name,
            block_type=block_type,
            position=position,
            reason="media-caption-output-ceiling",
        )
        state.stop_after_current = True
        if caption_text:
            return f"{prefix}{caption_text}\n{marker}"
        return f"{placeholder}\n{marker}"
    rendered_caption = probe
    if not rendered_caption:
        return placeholder
    return f"{prefix}{rendered_caption}"


def _maybe_add_rich_attachment(
    block: object,
    *,
    block_type: str,
    state: _RenderState,
    position: int,
) -> None:
    if block_type != "photo":
        return
    photo = _safe_getattr(block, "photo")
    if isinstance(photo, list):
        if not photo:
            return
        media = photo[-1]
    else:
        media = photo
    file_id = _safe_getattr(media, "file_id")
    if not file_id:
        return
    file_unique_id = _safe_getattr(media, "file_unique_id")
    file_size = _safe_getattr(media, "file_size")
    state.attachments.append(
        RichContentAttachment(
            type="photo",
            block_position=position,
            file_id=str(file_id),
            file_unique_id=str(file_unique_id) if file_unique_id else None,
            file_size=int(file_size) if isinstance(file_size, int) else None,
        )
    )


def _render_caption(caption: object, *, max_chars: int | None = None) -> str:
    text = _safe_getattr(caption, "text")
    credit = _safe_getattr(caption, "credit")
    if text is None:
        return ""
    rendered = _render_rich_text(text, max_chars=max_chars)
    if credit:
        rendered = f"{rendered} — {_render_rich_text(credit, max_chars=max_chars)}"
    return _cap_text(rendered, max_chars)


def _render_table(block: object, state: _RenderState, position: int) -> str:
    cells = _safe_getattr(block, "cells")
    if not isinstance(cells, list):
        return _malformed_placeholder("table", state, position)
    row_count = len(cells)
    col_count = max((len(row) for row in cells if isinstance(row, list)), default=0)
    if row_count > state.limits.table_rows:
        state.add_budget_hit(
            "table_rows",
            block_type="table",
            position=position,
            reason="too-many-rows",
        )
        state.stopped = True
        return ""
    if col_count > state.limits.table_cols:
        state.add_budget_hit(
            "table_cols",
            block_type="table",
            position=position,
            reason="too-many-columns",
        )
        state.stopped = True
        return ""

    unsupported = bool(_safe_getattr(block, "caption"))
    rows: list[list[str]] = []
    for row in cells:
        if not isinstance(row, list):
            return _malformed_placeholder("table", state, position)
        rendered_row: list[str] = []
        for cell in row:
            text_value = _safe_getattr(cell, "text")
            if text_value is None:
                unsupported = True
                rendered_cell = ""
            else:
                rendered_cell = _render_rich_text(
                    text_value,
                    max_chars=min(
                        state.limits.per_cell_chars + 1, state.remaining_output_chars() + 1
                    ),
                )
                if not isinstance(text_value, str):
                    unsupported = True
            if _safe_getattr(cell, "colspan") or _safe_getattr(cell, "rowspan"):
                unsupported = True
            if len(rendered_cell) > state.limits.per_cell_chars:
                rendered_cell = rendered_cell[: state.limits.per_cell_chars]
                state.add_budget_hit(
                    "per_cell_chars",
                    block_type="table",
                    position=position,
                    reason="cell-too-long",
                )
            rendered_row.append(_escape_table_cell(rendered_cell))
        rows.append(rendered_row)
        estimated_lines = _table_lines(rows, col_count)
        budget_name = state.would_exceed_output(len("\n".join(estimated_lines)))
        if budget_name:
            state.add_budget_hit(
                budget_name,
                block_type="table",
                position=position,
                reason="table-output-ceiling",
            )
            state.stopped = True
            rows.pop()
            break

    if not rows:
        return ""
    lines = _table_lines(rows, col_count)
    caption = _safe_getattr(block, "caption")
    if caption:
        lines.append(
            f"Caption: {_render_rich_text(caption, max_chars=state.remaining_output_chars())}"
        )
    if unsupported:
        state.add_warning(f"unsupported-table-semantics:table:{position}")
        lines.append("Table note: unsupported table semantics rendered best-effort")
    return "\n".join(lines)


def _table_lines(rows: list[list[str]], width_hint: int) -> list[str]:
    width = max([width_hint, *(len(row) for row in rows)], default=0)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    if not padded_rows:
        return []
    lines = [
        "| " + " | ".join(padded_rows[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_list(block: object, state: _RenderState, position: int, depth: int) -> str:
    items = _safe_getattr(block, "items")
    if not isinstance(items, list):
        return _malformed_placeholder("list", state, position)
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        label = str(_safe_getattr(item, "label") or f"{index}.")
        blocks = _safe_getattr(item, "blocks")
        if not isinstance(blocks, list):
            return _malformed_placeholder("list", state, position)
        body = _render_blocks(blocks, state, depth=depth + 1)
        first = body[0] if body else ""
        marker = label if label.endswith((".", ")")) or label == "-" else "-"
        lines.append(f"{marker} {first}".rstrip())
        for extra in body[1:]:
            lines.extend(f"  {line}" for line in extra.splitlines())
    return "\n".join(lines)


def _render_quote(
    block: object,
    state: _RenderState,
    position: int,
    depth: int,
    *,
    credit_field: str,
) -> str:
    blocks = _safe_getattr(block, "blocks")
    if not isinstance(blocks, list):
        return _malformed_placeholder("blockquote", state, position)
    body = "\n".join(_render_blocks(blocks, state, depth=depth + 1))
    credit = _safe_getattr(block, credit_field)
    if credit:
        body = f"{body}\n— {credit}"
    return "\n".join(f"> {line}" for line in body.splitlines())


def _render_details(block: object, state: _RenderState, position: int, depth: int) -> str:
    summary = _safe_getattr(block, "summary")
    blocks = _safe_getattr(block, "blocks")
    if summary is None or not isinstance(blocks, list):
        return _malformed_placeholder("details", state, position)
    body = "\n".join(_render_blocks(blocks, state, depth=depth + 1))
    body = "\n".join(f"  {line}" for line in body.splitlines())
    return f"<details: {_render_rich_text(summary)}>\n{body}".rstrip()


def _required_text_block(
    block: object,
    field_name: str,
    state: _RenderState,
    position: int,
    block_type: str,
) -> str:
    value = _safe_getattr(block, field_name)
    if value is None:
        return _malformed_placeholder(block_type, state, position)
    return _render_rich_text(value)


def _malformed_placeholder(block_type: str, state: _RenderState, position: int) -> str:
    placeholder = state.add_placeholder(block_type, position, label=f"malformed {block_type}")
    state.add_warning(f"malformed-block:{block_type}:{position}")
    _log_parse_failure(block_type, position, "malformed-block")
    return placeholder


def _render_rich_text(value: object, *, max_chars: int | None = None) -> str:
    if max_chars is not None and max_chars <= 0:
        return ""
    if value is None:
        return ""
    if isinstance(value, str):
        return _cap_text(value, max_chars)
    if isinstance(value, list):
        parts: list[str] = []
        remaining = max_chars
        for item in value:
            rendered = _render_rich_text(item, max_chars=remaining)
            parts.append(rendered)
            if remaining is not None:
                remaining -= len(rendered)
                if remaining <= 0:
                    break
        return "".join(parts)
    if isinstance(value, dict):
        text_type = _type_value(value.get("type"))
        text = value.get("text")
        expression = value.get("expression")
        alternative = value.get("alternative_text")
        url = value.get("url")
    else:
        text_type = _type_value(_safe_getattr(value, "type"))
        text = _safe_getattr(value, "text")
        expression = _safe_getattr(value, "expression")
        alternative = _safe_getattr(value, "alternative_text")
        url = _safe_getattr(value, "url")

    if text_type == "bold":
        return _cap_text(f"**{_render_rich_text(text, max_chars=max_chars)}**", max_chars)
    if text_type == "italic":
        return _cap_text(f"*{_render_rich_text(text, max_chars=max_chars)}*", max_chars)
    if text_type == "code":
        return _cap_text(f"`{_render_rich_text(text, max_chars=max_chars)}`", max_chars)
    if text_type == "url" and url:
        return _cap_text(f"[{_render_rich_text(text, max_chars=max_chars)}]({url})", max_chars)
    if text_type == "mathematical_expression":
        return _cap_text(f"${_safe_scalar_text(expression)}$", max_chars)
    if alternative:
        return _cap_text(_safe_scalar_text(alternative), max_chars)
    if expression:
        return _cap_text(_safe_scalar_text(expression), max_chars)
    return _render_rich_text(text, max_chars=max_chars)


def _safe_scalar_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    return ""


def _cap_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _rich_blocks(rich_message: object) -> list[object]:
    blocks = _safe_getattr(rich_message, "blocks")
    if blocks is None:
        return []
    if not isinstance(blocks, list):
        raise TypeError("rich_message.blocks must be a list")
    return blocks


def _raw_payload_exceeds(
    rich_message: object,
    blocks: list[object],
    limits: RichContentLimits,
) -> bool:
    if len(blocks) > limits.raw_payload_blocks:
        return True
    return _estimate_payload_exceeds(rich_message, limits)


def _estimate_payload_exceeds(root: object, limits: RichContentLimits) -> bool:
    """Bounded raw payload estimator that stops as soon as a limit is crossed."""
    total_bytes = 0
    seen_blocks = 0
    stack: list[object] = [root]
    seen_ids: set[int] = set()
    while stack:
        value = stack.pop()
        obj_id = id(value)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        seen_blocks += 1
        if seen_blocks > limits.raw_payload_blocks:
            return True
        if isinstance(value, str):
            total_bytes += len(value.encode("utf-8", errors="ignore"))
        elif isinstance(value, bytes):
            total_bytes += len(value)
        else:
            total_bytes += 16
        if total_bytes > limits.raw_payload_bytes:
            return True

        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list | tuple):
            stack.extend(value)
        else:
            field_names = getattr(type(value), "model_fields", None)
            if isinstance(field_names, dict):
                for field_name in field_names:
                    field_value = _safe_getattr(value, field_name)
                    if field_value is not None:
                        stack.append(field_value)
            elif hasattr(value, "__dict__"):
                stack.extend(vars(value).values())
    return False


def _contains_without_duplication(rich_text: str, legacy_text: str) -> bool:
    rich_normalized = _normalize_compare_text(rich_text)
    legacy_normalized = _normalize_compare_text(legacy_text)
    return bool(legacy_normalized) and legacy_normalized in rich_normalized


def _legacy_extra_text(rich_text: str, legacy_text: str) -> str:
    rich_lines = {_normalize_compare_text(line) for line in rich_text.splitlines() if line.strip()}
    extra = [
        line
        for line in legacy_text.splitlines()
        if line.strip() and _normalize_compare_text(line) not in rich_lines
    ]
    return "\n".join(extra)


def _normalize_compare_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _block_type(block: object) -> str:
    raw = _type_value(_safe_getattr(block, "type")) or "unknown"
    if raw in _KNOWN_BLOCK_TYPES:
        return raw
    if _SAFE_TOKEN_RE.fullmatch(raw):
        return raw
    return "unknown"


def _type_value(value: object) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw)


def _safe_getattr(obj: object, name: str, default: object | None = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, Mock) and name not in obj.__dict__:
        return default
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


def _escape_table_cell(text: str) -> str:
    return text.replace("\n", "<br>").replace("|", "\\|")


def _budget_marker(budget_name: str) -> str:
    return f"[truncated: budget {budget_name} exceeded]"


def _log_parse_failure(block_type: str, position: int, reason: str) -> None:
    logger.warning(
        "rich-parse-failure block_type=%s position=%s reason=%s",
        block_type,
        position,
        reason,
    )
