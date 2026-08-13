"""Final-answer Telegram Rich Message sender with legacy fallback."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.types import InputRichMessage
from markdown_it.token import Token

from telegram_bot.core.services.rich_markdown import markdown_to_rich_html, parse_rich_markdown
from telegram_bot.core.services.telegram_utils import SendOutcome

logger = logging.getLogger(__name__)

RICH_FALLBACK_BAD_REQUEST = "rich-send-fallback-deterministic-bad-request"
RICH_FALLBACK_NO_MESSAGE = "rich-send-fallback-no-message-returned"
RICH_FALLBACK_RETRY_AFTER = "rich-send-fallback-retry-after"
RICH_FALLBACK_AMBIGUOUS_TRANSPORT = "rich-send-fallback-ambiguous-transport"
RICH_FALLBACK_UNSUPPORTED_INPUT = "rich-send-fallback-unsupported-input"
RICH_FALLBACK_UNEXPECTED_ERROR = "rich-send-fallback-unexpected-error"
RICH_FALLBACK_FORBIDDEN = "rich-send-fallback-forbidden-delegate-legacy"

LEGACY_MESSAGE_LIMIT = 4096
RICH_MESSAGE_SPLIT_LIMIT = 32_000

_MAX_TABLE_COLS = 20
_RICH_BLOCK_LIMIT = 500
_TABLE_SEPARATOR_RE = re.compile(r"(?m)^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_FENCE_OPEN_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_RICH_BLOCK_TAG_RE = re.compile(
    r"<(?:p|h[1-6]|blockquote|ul|ol|li|pre|hr|table|thead|tbody|tr|th|td)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RichSendDecision:
    eligible: bool
    reason: str
    input_rich_message: InputRichMessage | None
    rich_html: str | None
    fallback_text: str
    input_rich_messages: tuple[InputRichMessage, ...] = ()
    rich_html_chunks: tuple[str, ...] = ()
    source_chunks: tuple[str, ...] = ()


RichSendCallable = Callable[[InputRichMessage], Awaitable[Any]]
LegacyFallbackCallable = Callable[[], Awaitable[SendOutcome]]
LegacyChunkFallbackCallable = Callable[[str], Awaitable[SendOutcome]]
RichSentCallback = Callable[[int], None]


def _has_table(tokens: Sequence[Token]) -> bool:
    return any(token.type == "table_open" for token in tokens)


def _rich_block_count(text: str) -> int:
    """Count block elements in the actual HTML handed to Telegram.

    Counting Markdown tokens is insufficient because the renderer inserts
    spacer ``<p>&nbsp;</p>`` blocks between top-level prose paragraphs.
    """
    try:
        rich_html = markdown_to_rich_html(text)
    except Exception:
        return _RICH_BLOCK_LIMIT + 1
    return len(_RICH_BLOCK_TAG_RE.findall(rich_html))


def _fits_rich_source(text: str) -> bool:
    return len(text) <= RICH_MESSAGE_SPLIT_LIMIT and _rich_block_count(text) <= _RICH_BLOCK_LIMIT


def _top_level_blocks(text: str) -> list[str]:
    """Return exact source slices for top-level Markdown blocks."""
    lines = text.splitlines(keepends=True)
    try:
        tokens = parse_rich_markdown(text)
    except Exception:
        return [text]

    spans: list[tuple[int, int]] = []
    for token in tokens:
        if token.level != 0 or token.map is None or token.nesting not in {0, 1}:
            continue
        span = (token.map[0], token.map[1])
        if span not in spans:
            spans.append(span)
    spans.sort()
    if not spans:
        return [text]

    blocks: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        blocks.append("".join(lines[cursor:end]))
        cursor = end
    if cursor < len(lines):
        tail = "".join(lines[cursor:])
        if blocks:
            blocks[-1] += tail
        else:
            blocks.append(tail)
    return [block for block in blocks if block]


def _hard_split(text: str, limit: int = RICH_MESSAGE_SPLIT_LIMIT) -> list[str]:
    """Split exact text at a nearby readable boundary, preserving all bytes."""
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        candidates = (
            remaining.rfind("\n\n", 0, limit + 1),
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind(" ", 0, limit + 1),
        )
        boundary = max(candidates)
        cut = boundary + 1 if boundary >= limit // 2 else limit
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        parts.append(remaining)
    return parts


def _table_parts(block: str) -> tuple[list[str], str, str, list[str]] | None:
    """Extract leading lines, header, divider, and body rows from a GFM table block."""
    lines = block.splitlines()
    for divider_index, line in enumerate(lines):
        if divider_index == 0 or _TABLE_SEPARATOR_RE.fullmatch(line) is None:
            continue
        header = lines[divider_index - 1]
        if "|" not in header:
            continue
        return (
            lines[: divider_index - 1],
            header,
            line,
            lines[divider_index + 1 :],
        )
    return None


def _parsed_table_rows(block: str) -> list[list[str]]:
    """Extract Markdown cell sources from the first parsed table in *block*."""
    try:
        tokens = parse_rich_markdown(block)
    except Exception:
        return []

    rows: list[list[str]] = []
    current_row: list[str] | None = None
    current_cell: str | None = None
    in_table = False
    for token in tokens:
        if token.type == "table_open":
            in_table = True
        elif token.type == "table_close":
            break
        elif not in_table:
            continue
        elif token.type == "tr_open":
            current_row = []
        elif token.type in {"th_open", "td_open"}:
            current_cell = ""
        elif token.type == "inline" and current_cell is not None:
            current_cell = token.content
        elif token.type in {"th_close", "td_close"} and current_row is not None:
            current_row.append(current_cell or "")
            current_cell = None
        elif token.type == "tr_close" and current_row is not None:
            rows.append(current_row)
            current_row = None
    return rows


def _markdown_table_row(cells: Sequence[str]) -> str:
    # markdown-it normalizes an escaped table delimiter (``\|``) back to a
    # literal pipe in Token.content. Re-escape it when reconstructing a wide
    # table, otherwise the horizontal slice gains accidental columns.
    escaped_cells = [cell.replace("|", r"\|") for cell in cells]
    return "| " + " | ".join(escaped_cells) + " |"


def _split_wide_table_block(block: str) -> list[str]:
    """Split a >20-column GFM table into native Telegram-sized tables."""
    rows = _parsed_table_rows(block)
    if not rows:
        return [block]
    width = max(len(row) for row in rows)
    if width <= _MAX_TABLE_COLS:
        return [block]

    extracted = _table_parts(block)
    leading = extracted[0] if extracted is not None else []
    tables: list[str] = []
    for start in range(0, width, _MAX_TABLE_COLS):
        end = min(start + _MAX_TABLE_COLS, width)
        table_lines = [_markdown_table_row((row + [""] * width)[start:end]) for row in rows]
        divider = _markdown_table_row(["---"] * (end - start))
        table_lines.insert(1, divider)
        if not tables and leading:
            table_lines = [*leading, *table_lines]
        tables.append("\n".join(table_lines))
    return tables


def _split_table_block(block: str) -> list[str]:
    extracted = _table_parts(block)
    if extracted is None:
        return _split_generic_block(block)
    leading, header, divider, rows = extracted
    prefix = "\n".join(leading)
    first_base = "\n".join([part for part in (prefix, header, divider) if part])
    repeated_base = f"{header}\n{divider}"

    parts: list[str] = []
    current = first_base
    for row in rows:
        candidate = f"{current}\n{row}" if current else row
        if _fits_rich_source(candidate):
            current = candidate
            continue
        if current:
            parts.append(current)
        current = f"{repeated_base}\n{row}"
        if not _fits_rich_source(current):
            # A single pathological row cannot be represented as one native
            # Telegram table. Keep it rich-deliverable rather than emitting
            # raw Markdown or exceeding the API limit.
            row_parts = _hard_split(row, RICH_MESSAGE_SPLIT_LIMIT - len(repeated_base) - 1)
            parts.extend(f"{repeated_base}\n{row_part}" for row_part in row_parts[:-1])
            current = f"{repeated_base}\n{row_parts[-1]}"
    if current:
        parts.append(current)
    return parts


def _split_fenced_block(block: str) -> list[str] | None:
    lines = block.splitlines()
    first_content = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_content is None:
        return None
    match = _FENCE_OPEN_RE.match(lines[first_content])
    if match is None:
        return None
    fence = match.group(1)
    closing_index = next(
        (
            index
            for index in range(len(lines) - 1, first_content, -1)
            if lines[index].strip().startswith(fence[0] * len(fence))
        ),
        None,
    )
    if closing_index is None:
        return None
    opening = "\n".join(lines[: first_content + 1])
    closing = lines[closing_index]
    content = "\n".join(lines[first_content + 1 : closing_index])
    content_limit = RICH_MESSAGE_SPLIT_LIMIT - len(opening) - len(closing) - 2
    if content_limit <= 0:
        return None
    return [f"{opening}\n{part}\n{closing}" for part in _hard_split(content, content_limit)]


def _split_generic_block(block: str) -> list[str]:
    fenced = _split_fenced_block(block)
    if fenced is not None:
        return fenced

    units = block.splitlines(keepends=True) or [block]
    parts: list[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if _fits_rich_source(candidate):
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        if _fits_rich_source(unit):
            current = unit
            continue
        split_unit = _hard_split(unit)
        parts.extend(split_unit[:-1])
        current = split_unit[-1]
    if current:
        parts.append(current)
    return parts


def _split_rich_markdown(text: str) -> list[str]:
    try:
        has_table = _has_table(parse_rich_markdown(text))
    except Exception:
        has_table = False
    if not has_table:
        length_chunks = _hard_split(text)
        if all(_fits_rich_source(chunk) for chunk in length_chunks):
            return length_chunks

    parts: list[str] = []
    for block in _top_level_blocks(text):
        if _table_parts(block) is not None:
            for column_index, column_part in enumerate(_split_wide_table_block(block)):
                table_parts = _split_table_block(column_part)
                if column_index > 0 and table_parts:
                    table_parts[0] = "\n\n" + table_parts[0]
                parts.extend(table_parts)
        elif _fits_rich_source(block):
            parts.append(block)
        else:
            parts.extend(_split_generic_block(block))

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + part
        if _fits_rich_source(candidate):
            current = candidate
            continue
        if current:
            chunks.append(current)
        if _fits_rich_source(part):
            current = part
        else:
            oversized = _hard_split(part)
            chunks.extend(oversized[:-1])
            current = oversized[-1]
    if current:
        chunks.append(current)
    return chunks


def _fallback(reason: str, text: str) -> RichSendDecision:
    return RichSendDecision(
        eligible=False,
        reason=reason,
        input_rich_message=None,
        rich_html=None,
        fallback_text=text,
    )


def detect_rich_send(final_text: str) -> RichSendDecision:
    """Detect whether *final_text* should be delivered via sendRichMessage."""
    fallback_text = final_text
    if not final_text.strip():
        return _fallback("empty", fallback_text)

    try:
        tokens = parse_rich_markdown(final_text)
    except Exception:
        logger.warning("rich-send-detect-failed len=%d", len(final_text), exc_info=True)
        return _fallback("ambiguous-content", fallback_text)

    has_table = _has_table(tokens) or _TABLE_SEPARATOR_RE.search(final_text) is not None
    is_long = len(final_text) > LEGACY_MESSAGE_LIMIT
    if not has_table and not is_long:
        return _fallback("plain-no-rich-structure", fallback_text)

    source_chunks = tuple(_split_rich_markdown(final_text))
    if not source_chunks:
        return _fallback("empty-rich-html", fallback_text)
    try:
        rich_html_chunks = tuple(markdown_to_rich_html(chunk) for chunk in source_chunks)
    except Exception:
        logger.warning("rich-html-render-failed len=%d", len(final_text), exc_info=True)
        return _fallback("ambiguous-content", fallback_text)
    if any(not rich_html.strip() for rich_html in rich_html_chunks):
        return _fallback("empty-rich-html", fallback_text)

    # Telegram's rich renderer ignores <ol start=...>, so an ordered list
    # resuming mid-message (grouped plans, chunks split inside a list) would
    # restart at 1. Legacy delivery keeps the literal source numbers.
    if any('<ol start="' in rich_html for rich_html in rich_html_chunks):
        return _fallback("ordered-list-restart", fallback_text)

    input_rich_messages = tuple(InputRichMessage(html=html) for html in rich_html_chunks)
    reason = "eligible-table" if has_table else "eligible-long"
    if len(input_rich_messages) > 1:
        reason += "-split"

    return RichSendDecision(
        eligible=True,
        reason=reason,
        input_rich_message=input_rich_messages[0],
        rich_html=rich_html_chunks[0],
        fallback_text=fallback_text,
        input_rich_messages=input_rich_messages,
        rich_html_chunks=rich_html_chunks,
        source_chunks=source_chunks,
    )


async def _legacy_with_log(
    *,
    reason: str,
    decision: RichSendDecision,
    legacy_fallback: LegacyFallbackCallable,
    label: str,
) -> SendOutcome:
    log = logger.info if decision.reason == "plain-no-rich-structure" else logger.warning
    log(
        "%s label=%s decision_reason=%s text_len=%d rich_len=%d",
        reason,
        label,
        decision.reason,
        len(decision.fallback_text),
        len(decision.rich_html or ""),
    )
    return await legacy_fallback()


def _message_id(sent: Any) -> int | None:
    message_id = getattr(sent, "message_id", None)
    return message_id if isinstance(message_id, int) else None


async def _try_send_rich(
    *,
    rich_message: InputRichMessage,
    send_rich: RichSendCallable,
    label: str,
    flood_retry_limit: float,
) -> tuple[SendOutcome, str | None]:
    """Attempt one rich chunk and return its outcome plus fallback reason."""
    try:
        sent = await send_rich(rich_message)
    except TelegramRetryAfter as exc:
        logger.warning(
            "%s label=%s retry_after=%ds limit=%.0fs",
            RICH_FALLBACK_RETRY_AFTER,
            label,
            exc.retry_after,
            flood_retry_limit,
        )
        if exc.retry_after > flood_retry_limit:
            return SendOutcome(message_id=None), RICH_FALLBACK_RETRY_AFTER
        await asyncio.sleep(exc.retry_after)
        try:
            sent = await send_rich(rich_message)
        except TelegramRetryAfter:
            return SendOutcome(message_id=None), RICH_FALLBACK_RETRY_AFTER
        except TelegramBadRequest:
            return SendOutcome(message_id=None), RICH_FALLBACK_BAD_REQUEST
        except TelegramForbiddenError:
            return SendOutcome(message_id=None, fatal=True), RICH_FALLBACK_FORBIDDEN
        except (TelegramNetworkError, TimeoutError, OSError, ConnectionError):
            return SendOutcome(message_id=None), RICH_FALLBACK_AMBIGUOUS_TRANSPORT
        except Exception:
            return SendOutcome(message_id=None), RICH_FALLBACK_UNEXPECTED_ERROR
    except TelegramBadRequest:
        return SendOutcome(message_id=None), RICH_FALLBACK_BAD_REQUEST
    except TelegramForbiddenError:
        return SendOutcome(message_id=None, fatal=True), RICH_FALLBACK_FORBIDDEN
    except (TelegramNetworkError, TimeoutError, OSError, ConnectionError):
        return SendOutcome(message_id=None), RICH_FALLBACK_AMBIGUOUS_TRANSPORT
    except Exception:
        return SendOutcome(message_id=None), RICH_FALLBACK_UNEXPECTED_ERROR

    message_id = _message_id(sent)
    if message_id is None:
        return SendOutcome(message_id=None), RICH_FALLBACK_NO_MESSAGE
    return SendOutcome(message_id=message_id), None


async def send_rich_final_answer(
    *,
    final_text: str,
    send_rich: RichSendCallable,
    legacy_fallback: LegacyFallbackCallable,
    label: str,
    flood_retry_limit: float = 300.0,
    legacy_chunk_fallback: LegacyChunkFallbackCallable | None = None,
    on_rich_sent: RichSentCallback | None = None,
) -> SendOutcome:
    """Deliver all eligible rich chunks without duplicating prior successes.

    ``legacy_fallback`` handles an entirely ineligible/single-chunk answer.
    Callers that can receive multi-rich plans provide ``legacy_chunk_fallback``
    so a later rejected chunk degrades on its own instead of resending chunks
    Telegram already accepted. ``on_rich_sent`` records every rich message ID.
    """
    decision = detect_rich_send(final_text)
    if not decision.eligible or not decision.input_rich_messages:
        return await _legacy_with_log(
            reason=RICH_FALLBACK_UNSUPPORTED_INPUT,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )

    last_outcome = SendOutcome(message_id=None)
    total = len(decision.input_rich_messages)
    for index, (rich_message, source_chunk) in enumerate(
        zip(decision.input_rich_messages, decision.source_chunks, strict=True),
        start=1,
    ):
        chunk_label = f"{label} chunk={index}/{total}"
        outcome, failure_reason = await _try_send_rich(
            rich_message=rich_message,
            send_rich=send_rich,
            label=chunk_label,
            flood_retry_limit=flood_retry_limit,
        )
        if outcome.fatal:
            logger.warning("%s label=%s", failure_reason or RICH_FALLBACK_FORBIDDEN, chunk_label)
            return outcome
        if outcome.message_id is not None:
            last_outcome = outcome
            if on_rich_sent is not None:
                try:
                    on_rich_sent(outcome.message_id)
                except Exception:
                    logger.warning(
                        "Rich message-id callback failed label=%s", chunk_label, exc_info=True
                    )
            continue

        if failure_reason == RICH_FALLBACK_AMBIGUOUS_TRANSPORT:
            # Telegram may have accepted the rich message before the client
            # lost the response. Retrying through legacy could duplicate the
            # private answer, so only deterministic failures auto-fallback.
            logger.warning("%s label=%s automatic resend suppressed", failure_reason, chunk_label)
            return SendOutcome(
                message_id=last_outcome.message_id,
                complete=False,
            )

        if total == 1:
            fallback = legacy_fallback
        elif legacy_chunk_fallback is not None:
            fallback = partial(legacy_chunk_fallback, source_chunk)
        elif index == 1:
            fallback = legacy_fallback
        else:
            logger.error(
                "%s label=%s no chunk fallback after prior rich delivery",
                failure_reason or RICH_FALLBACK_UNEXPECTED_ERROR,
                chunk_label,
            )
            return SendOutcome(
                message_id=last_outcome.message_id,
                complete=False,
            )

        fallback_outcome = await _legacy_with_log(
            reason=failure_reason or RICH_FALLBACK_UNEXPECTED_ERROR,
            decision=decision,
            legacy_fallback=fallback,
            label=chunk_label,
        )
        last_outcome = fallback_outcome
        if fallback_outcome.fatal:
            return fallback_outcome
        if fallback_outcome.message_id is None or not fallback_outcome.complete:
            return SendOutcome(
                message_id=last_outcome.message_id,
                complete=False,
            )

    return last_outcome
