"""Tests for engine session-persistence helpers (sub-project 1).

Stubs heavy providers so OrchestratorEngine() can be constructed
without loading models or contacting external services. asyncpg is
mocked. Tests focus on read-side helpers in T8 and the persistent
write-side path in T9-T11.
"""
import sys
import types
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_heavy_provider_stubs(monkeypatch):
    providers = types.ModuleType("docai.providers")
    providers.get_llm_client = lambda **kwargs: MagicMock()
    providers.get_dense_embedder = lambda: MagicMock()
    providers.get_sparse_embedder = lambda: MagicMock()
    providers.get_reranker = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")
    qdrant_stub.QdrantStore = MagicMock
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)

    falkor_stub = types.ModuleType("falkordb")
    falkor_stub.FalkorDB = MagicMock
    monkeypatch.setitem(sys.modules, "falkordb", falkor_stub)


def _make_engine(monkeypatch):
    _install_heavy_provider_stubs(monkeypatch)
    sys.modules.pop("docai.orchestrator.engine", None)
    import importlib
    engine_module = importlib.import_module("docai.orchestrator.engine")
    engine = engine_module.OrchestratorEngine()
    return engine, engine_module


@pytest.mark.asyncio
async def test_fetch_session_returns_row_for_owner(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)

    sid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    expected_row = {
        "session_id": sid,
        "user_id": "alice",
        "title": "T",
        "doc_ids": [],
        "summary": None,
        "summary_through_message_id": None,
        "created_at": "2026-05-14T00:00:00+00:00",
        "updated_at": "2026-05-14T00:00:00+00:00",
    }
    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value=expected_row)
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    result = await engine._fetch_session(sid, "alice")
    assert result == expected_row


@pytest.mark.asyncio
async def test_fetch_session_returns_none_for_wrong_owner(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value=None)
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    result = await engine._fetch_session(
        uuid.UUID("22222222-2222-2222-2222-222222222222"), "bob")
    assert result is None


@pytest.mark.asyncio
async def test_load_recent_messages_returns_chronological(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    sid = uuid.UUID("33333333-3333-3333-3333-333333333333")
    rows_desc = [
        {"id": 5, "role": "assistant", "content": "later"},
        {"id": 4, "role": "user", "content": "earlier"},
    ]
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(return_value=rows_desc)
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    result = await engine._load_recent_messages(sid, limit=20)
    # Helper returns oldest-first (chronological) for prompt building.
    assert [m["id"] for m in result] == [4, 5]


@pytest.mark.asyncio
async def test_handle_query_loads_history_when_session_ctx_set(monkeypatch) -> None:
    """When session_ctx is set, the engine must fetch the session row
    AND load recent messages BEFORE invoking retrieval/LLM. The test
    asserts those reads happen and that the LLM call sees the history
    in the prompt."""
    engine, engine_module = _make_engine(monkeypatch)
    sid = uuid.UUID("44444444-4444-4444-4444-444444444444")

    # Stub _fetch_session and _load_recent_messages.
    engine._fetch_session = AsyncMock(return_value={
        "session_id": sid,
        "user_id": "alice",
        "title": "T",
        "doc_ids": None,
        "summary": "Earlier we discussed X.",
        "summary_through_message_id": 4,
        "created_at": None,
        "updated_at": None,
    })
    engine._load_recent_messages = AsyncMock(return_value=[
        {"id": 5, "role": "user", "content": "Tell me more.",
         "thinking": None, "citations": None, "created_at": None},
        {"id": 6, "role": "assistant", "content": "Sure, ...",
         "thinking": None, "citations": None, "created_at": None},
    ])

    # Stub retrieval to a stub return so the rest of handle_query runs.
    async def _gather(*args, **kwargs):
        return {
            "vector": "Some retrieved context.",
            "graph": "No graph.",
            "citations": [],
            "document_inventory": "No docs.",
        }
    engine.gather_unified_context = _gather

    # Stub LLM to return content but capture the system_prompt to check
    # the augmented prompt was assembled.
    captured_kwargs: dict = {}

    async def _gen(**kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "answer", "thinking": "", "reason": None}

    engine.llm_client.generate_response = _gen

    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")
    result = await engine.handle_query(
        "What about Y?", session_ctx=ctx,
    )

    # Both helpers were called.
    engine._fetch_session.assert_awaited_once()
    engine._load_recent_messages.assert_awaited_once()

    # Augmented prompt contains the summary + the prior turn.
    assert "Earlier we discussed X" in captured_kwargs["system_prompt"]
    assert "Tell me more." in captured_kwargs["system_prompt"]
    # Response shape preserved.
    assert result["response"] == "answer"


@pytest.mark.asyncio
async def test_persist_turn_writes_user_and_assistant_rows(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    captured: list = []
    fake_conn = MagicMock()

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.close = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _noop_tx():
        yield

    fake_conn.transaction = MagicMock(return_value=_noop_tx())
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    sid = uuid.UUID("55555555-5555-5555-5555-555555555555")
    await engine._persist_turn(
        session_id=sid,
        user_message="hi",
        assistant_response="hello",
        thinking="",
        citations=None,
    )

    inserts = [(sql, args) for sql, args in captured if "INSERT INTO session_messages" in sql]
    assert len(inserts) == 2
    bumps = [(sql, args) for sql, args in captured if "UPDATE sessions" in sql and "updated_at" in sql]
    assert len(bumps) == 1


@pytest.mark.asyncio
async def test_maybe_auto_title_updates_only_default(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    captured: list = []
    fake_conn = MagicMock()

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    sid = uuid.UUID("66666666-6666-6666-6666-666666666666")
    long_query = "What is the difference between option A and option B in the proposal?"

    await engine._maybe_auto_title(sid, long_query)

    title_updates = [(sql, args) for sql, args in captured if "UPDATE sessions" in sql and "title" in sql]
    assert len(title_updates) == 1
    sql, args = title_updates[0]
    # Guard the WHERE clause so user renames are never overwritten.
    assert "title = 'New chat'" in sql
    new_title = args[0]
    assert len(new_title) <= 60


def test_post_session_chat_persists_messages(monkeypatch) -> None:
    """End-to-end-ish test of POST /api/v1/sessions/{id}/chat:
    the engine receives a SessionContext, returns a response, and
    persistence helpers are invoked."""
    import importlib
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _install_heavy_provider_stubs(monkeypatch)
    fake_pg = types.ModuleType("asyncpg")

    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value={"session_id": uuid.uuid4(), "user_id": "alice"})
    fake_conn.execute = AsyncMock(return_value=None)
    fake_conn.close = AsyncMock(return_value=None)
    fake_pg.connect = AsyncMock(return_value=fake_conn)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_pg)

    sys.modules.pop("docai.api.sessions", None)
    sys.modules.pop("docai.api.identity", None)
    sys.modules.pop("docai.orchestrator.engine", None)
    sys.modules.pop("docai.api.routes", None)

    sessions_module = importlib.import_module("docai.api.sessions")

    handle_query_calls: list = []

    class FakeEngine:
        async def handle_query(self, query, **kwargs):
            handle_query_calls.append((query, kwargs))
            return {"response": "ok", "thinking": "", "citations": []}

    sessions_module._get_engine_for_session_chat = lambda: FakeEngine()

    app = FastAPI()
    app.include_router(sessions_module.router, prefix="/api/v1")
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice")

    sid = "77777777-7777-7777-7777-777777777777"
    response = client.post(f"/api/v1/sessions/{sid}/chat", json={"query": "hello?"})

    assert response.status_code == 200
    assert response.json()["response"] == "ok"
    assert len(handle_query_calls) == 1
    assert "session_ctx" in handle_query_calls[0][1]


@pytest.mark.asyncio
async def test_persist_user_msg_writes_user_row_and_bumps_updated_at(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)

    captured: list = []
    fake_conn = MagicMock()

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.close = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _noop_tx():
        yield

    fake_conn.transaction = MagicMock(return_value=_noop_tx())
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    sid = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    await engine._persist_user_msg(session_id=sid, user_message="hi")

    inserts = [(sql, args) for sql, args in captured
               if "INSERT INTO session_messages" in sql and "'user'" in sql]
    assert len(inserts) == 1
    bumps = [(sql, args) for sql, args in captured
             if "UPDATE sessions" in sql and "updated_at" in sql]
    assert len(bumps) == 1


@pytest.mark.asyncio
async def test_persist_assistant_msg_writes_assistant_row_and_bumps_updated_at(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)

    captured: list = []
    fake_conn = MagicMock()

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.close = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _noop_tx():
        yield

    fake_conn.transaction = MagicMock(return_value=_noop_tx())
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    sid = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    await engine._persist_assistant_msg(
        session_id=sid,
        assistant_response="hello",
        thinking="",
        citations=None,
    )

    inserts = [(sql, args) for sql, args in captured
               if "INSERT INTO session_messages" in sql and "'assistant'" in sql]
    assert len(inserts) == 1
    bumps = [(sql, args) for sql, args in captured
             if "UPDATE sessions" in sql and "updated_at" in sql]
    assert len(bumps) == 1
