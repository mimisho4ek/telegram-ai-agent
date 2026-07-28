"""Markdown to Telegram Rich Message HTML conversion."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Any, cast
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.token import Token

_ALLOWED_LINK_SCHEMES = frozenset({"http", "https"})
_LOCAL_FILE_LINE_LINK_RE = re.compile(r"^(?:[^:/?#\s]+/)*[^:/?#\s]*\.[^:/?#\s]+:\d+(?::\d+)?$")


def _normalized_url(raw_url: str) -> str:
    return html.unescape(raw_url).strip()


def is_safe_rich_link(raw_url: str) -> bool:
    """Return whether *raw_url* can be emitted as a Telegram rich link."""
    url = _normalized_url(raw_url)
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() not in _ALLOWED_LINK_SCHEMES:
        return False
    return (
        bool(parsed.netloc)
        and hostname is not None
        and parsed.username is None
        and parsed.password is None
    )


def is_plain_text_rich_link(raw_url: str) -> bool:
    """Return whether a link target can safely degrade to its visible label."""
    url = _normalized_url(raw_url)
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if bool(url) and not parsed.scheme and not parsed.netloc:
        return True
    # RFC 3986 treats ``gdoc.py`` in ``gdoc.py:41`` as a URI scheme, even
    # though agents use this shape as a local file + line reference.
    return _LOCAL_FILE_LINE_LINK_RE.fullmatch(url) is not None


def _build_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark", options_update={"html": False}).enable("table")
    parser.validateLink = lambda _url: True  # type: ignore[method-assign, assignment]
    renderer = cast(Any, parser.renderer)
    rules = renderer.rules

    def _tag(name: str) -> tuple[Any, Any]:
        def _open(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
            return f"<{name}>"

        def _close(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
            return f"</{name}>"

        return _open, _close

    rules["strong_open"], rules["strong_close"] = _tag("b")
    rules["em_open"], rules["em_close"] = _tag("i")
    rules["text_special"] = lambda tokens, idx, options, env: escapeHtml(tokens[idx].content)

    rejected_links: list[bool] = []

    def _link_open(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
        href = str(tokens[idx].attrGet("href") or "")
        if not is_safe_rich_link(href):
            rejected_links.append(True)
            return ""
        rejected_links.append(False)
        return f'<a href="{escapeHtml(_normalized_url(href))}">'

    def _link_close(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
        rejected = rejected_links.pop() if rejected_links else False
        return "" if rejected else "</a>"

    rules["link_open"] = _link_open
    rules["link_close"] = _link_close

    def _image(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
        # Final-answer Markdown is untrusted model output. Never turn image
        # syntax into an automatically fetched remote resource: a prompt-
        # injected URL could otherwise become a tracking/exfiltration beacon.
        return escapeHtml(tokens[idx].content)

    rules["image"] = _image

    def _is_image_only_paragraph(tokens: Sequence[Token], open_idx: int) -> bool:
        inline = tokens[open_idx + 1] if open_idx + 1 < len(tokens) else None
        if inline is None or inline.type != "inline" or not inline.children:
            return False
        has_image = False
        for child in inline.children:
            if child.type == "image":
                has_image = True
            elif child.type in {"softbreak", "hardbreak"} or (
                child.type == "text" and not child.content.strip()
            ):
                continue
            else:
                return False
        return has_image

    def _previous_top_level(tokens: Sequence[Token], idx: int) -> tuple[Token | None, int | None]:
        for prev_idx in range(idx - 1, -1, -1):
            token = tokens[prev_idx]
            if token.hidden or token.level != 0:
                continue
            return token, prev_idx
        return None, None

    def _is_text_paragraph_close(tokens: Sequence[Token], idx: int | None) -> bool:
        return (
            idx is not None
            and idx >= 2
            and tokens[idx].type == "paragraph_close"
            and not _is_image_only_paragraph(tokens, idx - 2)
        )

    base_paragraph_open = rules.get("paragraph_open")
    base_paragraph_close = rules.get("paragraph_close")

    def _paragraph_open(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
        out = ""
        if tokens[idx].level == 0 and not _is_image_only_paragraph(tokens, idx):
            prev, prev_idx = _previous_top_level(tokens, idx)
            preceded_by_heading = prev is not None and prev.type == "heading_close"
            preceded_by_text_paragraph = _is_text_paragraph_close(tokens, prev_idx)
            if preceded_by_heading or preceded_by_text_paragraph:
                out += "<p>&nbsp;</p>"
        if base_paragraph_open is not None:
            return out + str(base_paragraph_open(tokens, idx, options, env))
        return out + str(renderer.renderToken(tokens, idx, options, env))

    def _paragraph_close(tokens: Sequence[Token], idx: int, options: Any, env: Any) -> str:
        if base_paragraph_close is not None:
            return str(base_paragraph_close(tokens, idx, options, env))
        return str(renderer.renderToken(tokens, idx, options, env))

    rules["paragraph_open"] = _paragraph_open
    rules["paragraph_close"] = _paragraph_close
    return parser


def markdown_to_rich_html(markdown: str) -> str:
    """Convert final-answer Markdown to Telegram rich-HTML."""
    return str(_build_parser().render(markdown))


def parse_rich_markdown(markdown: str) -> list[Token]:
    """Parse Markdown using the same rule set as the rich renderer."""
    return _build_parser().parse(markdown)
