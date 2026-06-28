"""Final-answer Telegram Rich Message sender with legacy fallback."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.types import InputRichMessage
from markdown_it.token import Token

from telegram_bot.core.services.rich_markdown import (
    is_safe_rich_link,
    markdown_to_rich_html,
    parse_rich_markdown,
)
from telegram_bot.core.services.telegram_utils import SendOutcome
from telegram_bot.core.utils.telegram_html import split_html_message

logger = logging.getLogger(__name__)

RICH_FALLBACK_BAD_REQUEST = "rich-send-fallback-deterministic-bad-request"
RICH_FALLBACK_NO_MESSAGE = "rich-send-fallback-no-message-returned"
RICH_FALLBACK_RETRY_AFTER = "rich-send-fallback-retry-after"
RICH_FALLBACK_AMBIGUOUS_TRANSPORT = "rich-send-fallback-ambiguous-transport"
RICH_FALLBACK_UNSUPPORTED_INPUT = "rich-send-fallback-unsupported-input"
RICH_FALLBACK_UNEXPECTED_ERROR = "rich-send-fallback-unexpected-error"
RICH_FALLBACK_FORBIDDEN = "rich-send-fallback-forbidden-delegate-legacy"

_MAX_TABLE_ROWS = 20
_MAX_TABLE_COLS = 6
_TASK_LIST_RE = re.compile(r"(?m)^\s{0,3}[-*+]\s+\[[ xX]\]\s+")
_UNSUPPORTED_VISIBLE_MARKDOWN_RE = re.compile(
    r"~~[^\n~]+~~|==[^\n=]+==|^\s{0,3}:\s+\S",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RichSendDecision:
    eligible: bool
    reason: str
    input_rich_message: InputRichMessage | None
    rich_html: str | None
    fallback_text: str


RichSendCallable = Callable[[InputRichMessage], Awaitable[Any]]
LegacyFallbackCallable = Callable[[], Awaitable[SendOutcome]]


def _iter_inline_children(tokens: Sequence[Token]) -> list[Token]:
    children: list[Token] = []
    for token in tokens:
        if token.children:
            children.extend(token.children)
    return children


def _find_unsafe_link(tokens: Sequence[Token]) -> bool:
    for token in [*tokens, *_iter_inline_children(tokens)]:
        if token.type not in {"link_open", "image"}:
            continue
        attr = token.attrGet("href") if token.type == "link_open" else token.attrGet("src")
        href = str(attr) if attr is not None else None
        if href is not None and not is_safe_rich_link(href):
            return True
    return False


def _has_table(tokens: Sequence[Token]) -> bool:
    return any(token.type == "table_open" for token in tokens)


def _has_unsupported(tokens: Sequence[Token], text: str) -> bool:
    if _TASK_LIST_RE.search(text):
        return True
    for token in _iter_inline_children(tokens):
        if token.type == "text" and _UNSUPPORTED_VISIBLE_MARKDOWN_RE.search(token.content):
            return True
    return any(token.type == "image" for token in _iter_inline_children(tokens))


def _table_shape_reason(tokens: Sequence[Token]) -> str | None:
    in_table = False
    row_count = 0
    current_cols = 0
    max_cols = 0

    for token in tokens:
        if token.type == "table_open":
            in_table = True
            row_count = 0
            current_cols = 0
            max_cols = 0
            continue
        if token.type == "table_close" and in_table:
            if row_count > _MAX_TABLE_ROWS:
                return "oversized-table"
            if max_cols > _MAX_TABLE_COLS:
                return "wide-table"
            in_table = False
            continue
        if not in_table:
            continue
        if token.type == "tr_open":
            row_count += 1
            current_cols = 0
        elif token.type in {"th_open", "td_open"}:
            current_cols += 1
            max_cols = max(max_cols, current_cols)
        if row_count > _MAX_TABLE_ROWS:
            return "oversized-table"
        if max_cols > _MAX_TABLE_COLS:
            return "wide-table"
    return None


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
    if len(split_html_message(final_text)) > 1:
        return _fallback("chunking-required", fallback_text)

    try:
        tokens = parse_rich_markdown(final_text)
    except Exception:
        logger.warning("rich-send-detect-failed len=%d", len(final_text), exc_info=True)
        return _fallback("ambiguous-content", fallback_text)

    if not _has_table(tokens):
        return _fallback("plain-no-rich-structure", fallback_text)
    if _find_unsafe_link(tokens):
        return _fallback("unsafe-link", fallback_text)
    table_reason = _table_shape_reason(tokens)
    if table_reason is not None:
        return _fallback(table_reason, fallback_text)
    if _has_unsupported(tokens, final_text):
        return _fallback("unsupported-markdown", fallback_text)
    try:
        rich_html = markdown_to_rich_html(final_text)
    except Exception:
        logger.warning("rich-html-render-failed len=%d", len(final_text), exc_info=True)
        return _fallback("ambiguous-content", fallback_text)
    if not rich_html.strip():
        return _fallback("empty-rich-html", fallback_text)

    return RichSendDecision(
        eligible=True,
        reason="eligible",
        input_rich_message=InputRichMessage(html=rich_html),
        rich_html=rich_html,
        fallback_text=fallback_text,
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


async def send_rich_final_answer(
    *,
    final_text: str,
    send_rich: RichSendCallable,
    legacy_fallback: LegacyFallbackCallable,
    label: str,
    flood_retry_limit: float = 300.0,
) -> SendOutcome:
    """Try rich final delivery, falling back to the caller's legacy sender."""
    decision = detect_rich_send(final_text)
    if not decision.eligible or decision.input_rich_message is None:
        return await _legacy_with_log(
            reason=RICH_FALLBACK_UNSUPPORTED_INPUT,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )

    try:
        sent = await send_rich(decision.input_rich_message)
    except TelegramRetryAfter as exc:
        logger.warning(
            "%s label=%s retry_after=%ds limit=%.0fs text_len=%d rich_len=%d",
            RICH_FALLBACK_RETRY_AFTER,
            label,
            exc.retry_after,
            flood_retry_limit,
            len(decision.fallback_text),
            len(decision.rich_html or ""),
        )
        if exc.retry_after > flood_retry_limit:
            return await _legacy_with_log(
                reason=RICH_FALLBACK_RETRY_AFTER,
                decision=decision,
                legacy_fallback=legacy_fallback,
                label=label,
            )
        await asyncio.sleep(exc.retry_after)
        try:
            sent = await send_rich(decision.input_rich_message)
        except TelegramRetryAfter:
            return await _legacy_with_log(
                reason=RICH_FALLBACK_RETRY_AFTER,
                decision=decision,
                legacy_fallback=legacy_fallback,
                label=label,
            )
        except TelegramBadRequest:
            return await _legacy_with_log(
                reason=RICH_FALLBACK_BAD_REQUEST,
                decision=decision,
                legacy_fallback=legacy_fallback,
                label=label,
            )
        except (TimeoutError, OSError, ConnectionError):
            return await _legacy_with_log(
                reason=RICH_FALLBACK_AMBIGUOUS_TRANSPORT,
                decision=decision,
                legacy_fallback=legacy_fallback,
                label=label,
            )
        except Exception:
            return await _legacy_with_log(
                reason=RICH_FALLBACK_UNEXPECTED_ERROR,
                decision=decision,
                legacy_fallback=legacy_fallback,
                label=label,
            )
    except TelegramBadRequest:
        return await _legacy_with_log(
            reason=RICH_FALLBACK_BAD_REQUEST,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )
    except TelegramForbiddenError:
        return await _legacy_with_log(
            reason=RICH_FALLBACK_FORBIDDEN,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )
    except (TimeoutError, OSError, ConnectionError):
        return await _legacy_with_log(
            reason=RICH_FALLBACK_AMBIGUOUS_TRANSPORT,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )
    except Exception:
        return await _legacy_with_log(
            reason=RICH_FALLBACK_UNEXPECTED_ERROR,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )

    message_id = _message_id(sent)
    if message_id is None:
        return await _legacy_with_log(
            reason=RICH_FALLBACK_NO_MESSAGE,
            decision=decision,
            legacy_fallback=legacy_fallback,
            label=label,
        )
    return SendOutcome(message_id=message_id)
