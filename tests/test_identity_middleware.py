"""Tests for src/docai/api/identity.py.

The dependency must (1) generate a fresh UUID + Set-Cookie when the
configured cookie is absent, (2) preserve the value when present, and
(3) honour the cookie name from settings (escape hatch for renaming).
"""
import re
import sys
from unittest.mock import MagicMock


def _make_request(cookies: dict[str, str] | None = None) -> MagicMock:
    request = MagicMock()
    request.cookies = cookies or {}
    return request


def _make_response() -> MagicMock:
    response = MagicMock()
    response.cookies_set: list[dict] = []  # type: ignore[attr-defined]

    def _set_cookie(**kwargs):
        response.cookies_set.append(kwargs)

    response.set_cookie = _set_cookie
    return response


def _import_identity():
    sys.modules.pop("docai.api.identity", None)
    import importlib
    return importlib.import_module("docai.api.identity")


def test_first_request_generates_uuid_and_sets_cookie() -> None:
    identity = _import_identity()
    request = _make_request(cookies={})
    response = _make_response()

    user_id = identity.get_current_user_id(request, response)

    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        user_id,
    )
    assert len(response.cookies_set) == 1
    cookie = response.cookies_set[0]
    assert cookie["key"] == "docai_uid"
    assert cookie["value"] == user_id
    assert cookie["httponly"] is True
    assert cookie["samesite"].lower() == "lax"
    assert cookie["path"] == "/"
    assert cookie["max_age"] > 0


def test_subsequent_request_preserves_cookie() -> None:
    identity = _import_identity()
    existing = "11111111-1111-1111-1111-111111111111"
    request = _make_request(cookies={"docai_uid": existing})
    response = _make_response()

    user_id = identity.get_current_user_id(request, response)

    assert user_id == existing
    assert response.cookies_set == []


def test_respects_configurable_cookie_name(monkeypatch) -> None:
    identity = _import_identity()
    monkeypatch.setattr(identity.settings, "SESSION_USER_ID_COOKIE", "custom_uid", raising=False)

    request = _make_request(cookies={"custom_uid": "abc"})
    response = _make_response()

    user_id = identity.get_current_user_id(request, response)

    assert user_id == "abc"
    assert response.cookies_set == []
