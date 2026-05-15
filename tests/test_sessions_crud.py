"""Tests for src/docai/api/sessions.py.

Covers the full CRUD surface (T5+T6+T7) plus cross-user isolation.
Uses a fake asyncpg connection so no live Postgres is required.
The router is mounted into a fresh FastAPI app per test.

This file starts with T5 endpoints (POST + GET list); T6 and T7 will
extend it.
"""
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _fake_asyncpg_with(rows_by_query: dict[str, list] | None = None,
                      row_by_query: dict[str, dict | None] | None = None,
                      execute_capture: list | None = None):
    """Build a fake asyncpg module + a fake conn that dispatches fetch
    results based on the query string, and captures execute() calls."""
    rows_by_query = rows_by_query or {}
    row_by_query = row_by_query or {}
    execute_capture = execute_capture if execute_capture is not None else []

    conn = MagicMock()

    async def _fetch(sql, *args):
        for needle, rows in rows_by_query.items():
            if needle in sql:
                return rows
        return []

    async def _fetchrow(sql, *args):
        for needle, row in row_by_query.items():
            if needle in sql:
                return row
        return None

    async def _execute(sql, *args):
        execute_capture.append((sql, args))
        return "UPDATE 1" if "UPDATE" in sql else "DELETE 1"

    conn.fetch = AsyncMock(side_effect=_fetch)
    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.execute = AsyncMock(side_effect=_execute)
    conn.close = AsyncMock(return_value=None)

    fake_pg = types.ModuleType("asyncpg")
    fake_pg.connect = AsyncMock(return_value=conn)
    return fake_pg, conn, execute_capture


def _build_app(monkeypatch, fake_pg) -> FastAPI:
    monkeypatch.setitem(sys.modules, "asyncpg", fake_pg)
    sys.modules.pop("docai.api.sessions", None)
    sys.modules.pop("docai.api.identity", None)
    import importlib
    sessions = importlib.import_module("docai.api.sessions")
    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1")
    return app


def test_post_sessions_creates_row(monkeypatch) -> None:
    capture: list = []
    fake_pg, conn, _ = _fake_asyncpg_with(execute_capture=capture)
    app = _build_app(monkeypatch, fake_pg)
    client = TestClient(app)

    response = client.post("/api/v1/sessions", json={"title": "First chat"})
    assert response.status_code == 201
    body = response.json()
    assert "session_id" in body
    assert body["title"] == "First chat"
    assert body["doc_ids"] == []
    # Cookie was set on response.
    assert "docai_uid" in response.cookies

    # The INSERT included user_id from the cookie.
    insert_calls = [(sql, args) for sql, args in capture if "INSERT INTO sessions" in sql]
    assert len(insert_calls) == 1


def test_get_sessions_lists_only_caller_rows(monkeypatch) -> None:
    user_a_session = {
        "session_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "title": "Alice's chat",
        "doc_ids": [],
        "updated_at": "2026-05-14T10:00:00+00:00",
        "message_count": 4,
    }
    fake_pg, _, _ = _fake_asyncpg_with(rows_by_query={"FROM sessions": [user_a_session]})
    app = _build_app(monkeypatch, fake_pg)
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice-uuid")

    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    sessions_list = response.json()
    assert len(sessions_list) == 1
    assert sessions_list[0]["title"] == "Alice's chat"


def test_get_session_by_id_returns_metadata_and_messages(monkeypatch) -> None:
    sid = uuid.UUID("22222222-2222-2222-2222-222222222222")
    session_row = {
        "session_id": sid,
        "user_id": "alice-uuid",
        "title": "About SP500",
        "doc_ids": [],
        "summary": None,
        "created_at": "2026-05-14T09:00:00+00:00",
        "updated_at": "2026-05-14T10:00:00+00:00",
    }
    msg_rows = [
        {"id": 1, "role": "user", "content": "What is SP500?",
         "thinking": None, "citations": None, "created_at": "2026-05-14T09:00:01+00:00"},
        {"id": 2, "role": "assistant", "content": "It is...",
         "thinking": "", "citations": [], "created_at": "2026-05-14T09:00:30+00:00"},
    ]
    fake_pg, _, _ = _fake_asyncpg_with(
        row_by_query={"FROM sessions WHERE session_id": session_row},
        rows_by_query={"FROM session_messages": msg_rows},
    )
    app = _build_app(monkeypatch, fake_pg)
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice-uuid")

    response = client.get(f"/api/v1/sessions/{sid}")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == str(sid)
    assert body["title"] == "About SP500"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"


def test_get_session_returns_404_for_other_user(monkeypatch) -> None:
    sid = uuid.UUID("33333333-3333-3333-3333-333333333333")
    # No row matches because user_id WHERE clause filters it out.
    fake_pg, _, _ = _fake_asyncpg_with(row_by_query={"FROM sessions WHERE session_id": None})
    app = _build_app(monkeypatch, fake_pg)
    client = TestClient(app)
    client.cookies.set("docai_uid", "bob-uuid")

    response = client.get(f"/api/v1/sessions/{sid}")
    assert response.status_code == 404


def test_patch_session_updates_title_and_doc_ids(monkeypatch) -> None:
    sid = uuid.UUID("44444444-4444-4444-4444-444444444444")
    capture: list = []
    fake_pg, _, _ = _fake_asyncpg_with(
        row_by_query={"FROM sessions WHERE session_id": {
            "session_id": sid, "user_id": "alice-uuid",
        }},
        execute_capture=capture,
    )
    app = _build_app(monkeypatch, fake_pg)
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice-uuid")

    new_doc = "55555555-5555-5555-5555-555555555555"
    response = client.patch(
        f"/api/v1/sessions/{sid}",
        json={"title": "Renamed", "doc_ids": [new_doc]},
    )
    assert response.status_code == 200

    # An UPDATE happened with the new values.
    update_calls = [(sql, args) for sql, args in capture if "UPDATE sessions" in sql]
    assert len(update_calls) == 1


def test_delete_session_returns_204(monkeypatch) -> None:
    sid = uuid.UUID("66666666-6666-6666-6666-666666666666")
    capture: list = []
    # DELETE now uses fetchrow with RETURNING (atomic ownership+delete).
    # Match "DELETE FROM sessions" in the _fetchrow dispatcher so it
    # returns a non-None row, indicating the delete succeeded.
    fake_pg, conn, _ = _fake_asyncpg_with(
        row_by_query={"DELETE FROM sessions": {
            "session_id": sid,
        }},
        execute_capture=capture,
    )
    app = _build_app(monkeypatch, fake_pg)
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice-uuid")

    response = client.delete(f"/api/v1/sessions/{sid}")
    assert response.status_code == 204
    # Verify the atomic DELETE+RETURNING fetchrow was called.
    delete_fetchrow_calls = [
        call for call in conn.fetchrow.call_args_list
        if "DELETE FROM sessions" in call.args[0]
    ]
    assert len(delete_fetchrow_calls) == 1
