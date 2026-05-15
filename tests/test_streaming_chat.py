"""Tests for streaming chat (sub-project 3).

Covers the engine's handle_query_streaming generator (T3) and the
SSE wire-format endpoint (T4). All tests stub heavy providers + asyncpg
+ FalkorDB, so no live infrastructure is required.
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


def _stub_streaming_llm(engine, chunks):
    """Replace engine.llm_client.generate_response_streaming with an
    async generator that yields the supplied chunks in order."""

    async def _gen(**kwargs):
        for c in chunks:
            yield c

    engine.llm_client.generate_response_streaming = _gen


def _stub_engine_helpers(engine, *, citations=None):
    """Mock the read/persist helpers on the engine so the streaming
    generator can run end-to-end without touching asyncpg."""
    engine._fetch_session = AsyncMock(return_value={
        "session_id": uuid.uuid4(),
        "user_id": "alice",
        "title": "T",
        "doc_ids": None,
        "summary": None,
        "summary_through_message_id": None,
        "created_at": None,
        "updated_at": None,
    })
    engine._load_recent_messages = AsyncMock(return_value=[])
    engine._persist_user_msg = AsyncMock(return_value=None)
    engine._persist_assistant_msg = AsyncMock(return_value=None)
    engine._maybe_auto_title = AsyncMock(return_value=None)
    engine._should_resummarize = AsyncMock(return_value=False)
    engine._maybe_summarize = AsyncMock(return_value=None)

    async def _gather(*a, **kw):
        return {
            "vector": "Some retrieved context.",
            "graph": "",
            "citations": citations or [],
            "document_inventory": "",
        }

    engine.gather_unified_context = _gather


@pytest.mark.asyncio
async def test_handle_query_streaming_emits_events_in_order(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine, citations=[
        {"id": "abcd", "doc_id": "d1", "heading": "h", "page_num": 1, "score": 0.9},
    ])
    _stub_streaming_llm(engine, [
        {"kind": "content", "text": "Hi"},
        {"kind": "content", "text": " world"},
    ])

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    events = []
    async for ev in engine.handle_query_streaming("hello?", ctx):
        events.append(ev)

    event_types = [e["event"] for e in events]
    assert event_types == ["citations", "content_delta", "content_delta", "done"]
    assert events[0]["data"][0]["doc_id"] == "d1"
    assert events[1]["data"]["text"] == "Hi"
    assert events[2]["data"]["text"] == " world"
    assert "guard" in events[-1]["data"]


@pytest.mark.asyncio
async def test_handle_query_streaming_persists_user_msg_before_first_event(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)
    _stub_streaming_llm(engine, [{"kind": "content", "text": "ok"}])

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    gen = engine.handle_query_streaming("hello?", ctx)
    # Pull the first event — by then user_msg must already be persisted.
    await gen.__anext__()
    engine._persist_user_msg.assert_awaited_once()
    # And auto-title fires too, also before the first event.
    engine._maybe_auto_title.assert_awaited_once()
    # Drain the rest so the generator cleans up.
    async for _ in gen:
        pass
    engine._persist_assistant_msg.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_query_streaming_emits_thinking_delta_for_thinking_chunks(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)
    _stub_streaming_llm(engine, [
        {"kind": "thinking", "text": "let me think"},
        {"kind": "content", "text": "answer"},
    ])

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    events = []
    async for ev in engine.handle_query_streaming("q?", ctx):
        events.append(ev)

    types_seq = [e["event"] for e in events]
    assert types_seq == ["citations", "thinking_delta", "content_delta", "done"]


def _build_sessions_app(monkeypatch, fake_engine):
    """Build a FastAPI app with the sessions router mounted, swapping
    in a fake engine and a fake asyncpg for the ownership pre-check."""
    import importlib
    from fastapi import FastAPI

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
    sessions_module._get_engine_for_session_chat = lambda: fake_engine

    app = FastAPI()
    app.include_router(sessions_module.router, prefix="/api/v1")
    return app, sessions_module


def _parse_sse_lines(text: str):
    """Parse an SSE response body into [{event, data_str}] dicts."""
    import json
    events = []
    current_event = None
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:") and current_event is not None:
            data_str = line[len("data:"):].strip()
            events.append({"event": current_event, "data": json.loads(data_str)})
            current_event = None
        elif not line:
            current_event = None
    return events


def test_dict_path_unchanged_without_accept_header(monkeypatch):
    """Regression: POST without SSE Accept returns dict."""
    from fastapi.testclient import TestClient

    class FakeEngine:
        async def handle_query(self, query, **kwargs):
            return {"response": "ok", "thinking": "", "citations": []}

    app, _ = _build_sessions_app(monkeypatch, FakeEngine())
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice")

    sid = "11111111-1111-1111-1111-111111111111"
    response = client.post(f"/api/v1/sessions/{sid}/chat", json={"query": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["response"] == "ok"


def test_sse_response_is_text_event_stream_with_named_events(monkeypatch):
    """POST with Accept: text/event-stream returns SSE."""
    from fastapi.testclient import TestClient

    class FakeEngine:
        async def handle_query_streaming(self, query, session_ctx, is_disconnected=None):
            yield {"event": "citations", "data": []}
            yield {"event": "content_delta", "data": {"text": "Hi"}}
            yield {"event": "content_delta", "data": {"text": " world"}}
            yield {"event": "done", "data": {"guard": None}}

    app, _ = _build_sessions_app(monkeypatch, FakeEngine())
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice")

    sid = "22222222-2222-2222-2222-222222222222"
    response = client.post(
        f"/api/v1/sessions/{sid}/chat",
        json={"query": "hi"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_lines(response.text)
    assert [e["event"] for e in events] == [
        "citations", "content_delta", "content_delta", "done",
    ]


def test_sse_emits_error_event_when_engine_yields_error(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeEngine:
        async def handle_query_streaming(self, query, session_ctx, is_disconnected=None):
            yield {"event": "citations", "data": []}
            yield {"event": "error", "data": {"detail": "boom", "stage": "llm"}}

    app, _ = _build_sessions_app(monkeypatch, FakeEngine())
    client = TestClient(app)
    client.cookies.set("docai_uid", "alice")

    sid = "33333333-3333-3333-3333-333333333333"
    response = client.post(
        f"/api/v1/sessions/{sid}/chat",
        json={"query": "hi"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    events = _parse_sse_lines(response.text)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["stage"] == "llm"


@pytest.mark.asyncio
async def test_handle_query_streaming_promotes_empty_content_with_thinking(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)
    # Only thinking, no content → empty_content_with_thinking.
    _stub_streaming_llm(engine, [
        {"kind": "thinking", "text": "I am pondering."},
    ])

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    events = []
    async for ev in engine.handle_query_streaming("q", ctx):
        events.append(ev)

    # Final content_delta carries the spec-locked promoted message.
    content_deltas = [e for e in events if e["event"] == "content_delta"]
    assert len(content_deltas) == 1
    assert "reasoning but no answer" in content_deltas[0]["data"]["text"].lower()

    # Persisted assistant content matches the promoted text.
    persisted = engine._persist_assistant_msg.await_args
    assert "reasoning but no answer" in persisted.kwargs["assistant_response"].lower()


@pytest.mark.asyncio
async def test_handle_query_streaming_promotes_empty_response(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)
    # Zero chunks → empty_response.
    _stub_streaming_llm(engine, [])

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    events = []
    async for ev in engine.handle_query_streaming("q", ctx):
        events.append(ev)

    content_deltas = [e for e in events if e["event"] == "content_delta"]
    assert len(content_deltas) == 1
    assert "produced no output" in content_deltas[0]["data"]["text"].lower()


@pytest.mark.asyncio
async def test_handle_query_streaming_cancellation_skips_assistant_persist(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)
    _stub_streaming_llm(engine, [
        {"kind": "content", "text": "first"},
        {"kind": "content", "text": "second"},
    ])

    # Disconnect after the first chunk arrives.
    state = {"called": 0}

    async def _is_disconnected():
        state["called"] += 1
        return state["called"] >= 2

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    events = []
    async for ev in engine.handle_query_streaming(
        "q", ctx, is_disconnected=_is_disconnected,
    ):
        events.append(ev)

    # User msg DID persist; assistant msg did NOT.
    engine._persist_user_msg.assert_awaited_once()
    engine._persist_assistant_msg.assert_not_awaited()
    # No "done" event at end.
    assert not any(e["event"] == "done" for e in events)


@pytest.mark.asyncio
async def test_handle_query_streaming_emits_error_when_llm_raises(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)

    async def _raises(**kwargs):
        if False:
            yield {}  # make this a generator
        raise RuntimeError("ollama down")

    engine.llm_client.generate_response_streaming = _raises

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    events = []
    async for ev in engine.handle_query_streaming("q", ctx):
        events.append(ev)

    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["stage"] == "llm"
    engine._persist_assistant_msg.assert_not_awaited()


def test_sse_404_returned_as_plain_http_for_wrong_owner(monkeypatch):
    """Cross-user request: ownership check returns 404 BEFORE the stream
    starts, so the response is plain JSON 404 (not text/event-stream)."""
    import importlib
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _install_heavy_provider_stubs(monkeypatch)

    fake_pg = types.ModuleType("asyncpg")
    fake_conn = MagicMock()
    # No row → ownership check returns None → 404.
    fake_conn.fetchrow = AsyncMock(return_value=None)
    fake_conn.execute = AsyncMock(return_value=None)
    fake_conn.close = AsyncMock(return_value=None)
    fake_pg.connect = AsyncMock(return_value=fake_conn)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_pg)

    sys.modules.pop("docai.api.sessions", None)
    sys.modules.pop("docai.api.identity", None)
    sys.modules.pop("docai.orchestrator.engine", None)
    sys.modules.pop("docai.api.routes", None)
    sessions_module = importlib.import_module("docai.api.sessions")

    app = FastAPI()
    app.include_router(sessions_module.router, prefix="/api/v1")
    client = TestClient(app)
    client.cookies.set("docai_uid", "bob")

    sid = "44444444-4444-4444-4444-444444444444"
    response = client.post(
        f"/api/v1/sessions/{sid}/chat",
        json={"query": "hi"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.asyncio
async def test_handle_query_streaming_summarization_trigger_after_assistant_persist(monkeypatch):
    engine, engine_module = _make_engine(monkeypatch)
    _stub_engine_helpers(engine)
    monkeypatch.setattr(engine_module.settings, "ENABLE_SESSION_SUMMARIZATION", True, raising=False)
    engine._should_resummarize = AsyncMock(return_value=True)

    _stub_streaming_llm(engine, [{"kind": "content", "text": "ok"}])

    sid = uuid.uuid4()
    ctx = engine_module.SessionContext(session_id=sid, user_id="alice")

    async for _ in engine.handle_query_streaming("q", ctx):
        pass

    # Summarize was scheduled (the create_task wraps it).
    engine._should_resummarize.assert_awaited()
    # asyncio.create_task call means _maybe_summarize is invoked
    # once the loop runs. Give it a tick.
    import asyncio
    await asyncio.sleep(0)
    engine._maybe_summarize.assert_awaited()
