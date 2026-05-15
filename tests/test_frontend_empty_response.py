"""Contract tests for the frontend empty-response handling (Task T4).

The chat-response handler in `frontend/app.js` is the single source of truth
for messages shown when the model returns an empty `response` field.
`renderRichText` must never invent placeholder text.

These tests read `frontend/app.js` as source text and assert the expected
literals are (or are not) present.
"""
import re
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
APP_JS = FRONTEND_DIR / "app.js"


def _read_app_js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _extract_render_rich_text_body(source: str) -> str:
    """Return the body of `function renderRichText(...)` from the source.

    Uses a brace-balanced scan starting at the function's opening brace so
    nested braces (object literals, arrow functions) don't terminate early.
    """
    match = re.search(r"function\s+renderRichText\s*\([^)]*\)\s*\{", source)
    assert match is not None, "Could not locate renderRichText function in app.js"
    start = match.end() - 1  # position of the opening '{'
    depth = 0
    for i in range(start, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError("Unbalanced braces while scanning renderRichText body")


def test_render_rich_text_does_not_invent_empty_response_text() -> None:
    """`renderRichText` must not insert the legacy 'No response text returned.' literal.

    The chat-response handler is now the single source of truth for empty-response
    messaging, so this fallback inside renderRichText is removed.
    """
    body = _extract_render_rich_text_body(_read_app_js())
    assert "No response text returned." not in body, (
        "renderRichText must not invent placeholder text for empty wrappers; "
        "the chat-response handler owns empty-response messaging."
    )


def test_legacy_empty_response_literal_is_gone_from_file() -> None:
    """The legacy literal must not appear anywhere in app.js.

    Stronger guarantee than the renderRichText-scoped check: confirms the
    chat-response handler also stopped using the old `data.response || "No response..."`
    pattern.
    """
    assert "No response text returned." not in _read_app_js(), (
        "The legacy 'No response text returned.' literal should not appear "
        "anywhere in app.js after T4."
    )


def test_chat_handler_branch_for_thinking_only_response_exists() -> None:
    """Proves the branch that messages 'reasoning shown below' landed."""
    assert "Model produced no answer; reasoning shown below." in _read_app_js(), (
        "Expected the thinking-only empty-response branch string in app.js."
    )


def test_chat_handler_branch_for_fully_empty_response_exists() -> None:
    """Proves the branch that messages 'please retry' landed."""
    assert "No response from the model. Please retry." in _read_app_js(), (
        "Expected the fully-empty-response branch string in app.js."
    )
