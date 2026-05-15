"""Regression guard for the stateless POST /api/v1/chat endpoint.

The endpoint must NEVER write to the new sessions or session_messages
tables, regardless of whether session_id is supplied in the request
body. This protects the "stateless debug surface" contract.
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _install_heavy_provider_stubs(monkeypatch):
    providers = types.ModuleType("docai.providers")
    providers.get_llm_client = lambda **kwargs: MagicMock()
    providers.get_dense_embedder = lambda: MagicMock()
    providers.get_sparse_embedder = lambda: MagicMock()
    providers.get_reranker = lambda: MagicMock()
    providers.get_ocr_service = lambda: MagicMock()
    providers.get_captioner = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")
    qdrant_stub.QdrantStore = MagicMock
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)

    falkor_stub = types.ModuleType("falkordb")
    falkor_stub.FalkorDB = MagicMock
    monkeypatch.setitem(sys.modules, "falkordb", falkor_stub)


def test_chat_endpoint_writes_zero_session_rows(monkeypatch) -> None:
    _install_heavy_provider_stubs(monkeypatch)

    captured: list = []
    fake_pg = types.ModuleType("asyncpg")
    fake_conn = MagicMock()

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.fetch = AsyncMock(return_value=[])
    fake_conn.fetchrow = AsyncMock(return_value=None)
    fake_conn.close = AsyncMock(return_value=None)
    fake_pg.connect = AsyncMock(return_value=fake_conn)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_pg)

    # Force fresh module imports so the asyncpg stub is picked up everywhere.
    for m in (
        "docai.api.routes", "docai.api.sessions", "docai.api.identity",
        "docai.orchestrator.engine",
    ):
        sys.modules.pop(m, None)

    import importlib
    routes = importlib.import_module("docai.api.routes")

    # Replace the engine factory with a fake that returns a stub
    # `handle_query` (no real LLM, no real retrieval).
    class FakeEngine:
        async def handle_query(self, query, session_id=None, doc_ids=None, **kwargs):
            return {"response": "ok", "thinking": "", "citations": []}

    async def _fake_get_engine():
        return FakeEngine()

    monkeypatch.setattr(routes, "get_engine", _fake_get_engine)

    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={"query": "hello", "session_id": "11111111-1111-1111-1111-111111111111"},
    )
    assert response.status_code == 200

    # NO writes to session tables — this is the regression guard.
    session_writes = [
        (sql, args) for sql, args in captured
        if "INSERT INTO sessions" in sql
        or "INSERT INTO session_messages" in sql
        or ("UPDATE sessions" in sql and "summary" in sql)
        or ("UPDATE sessions" in sql and "title" in sql)
        or ("UPDATE sessions" in sql and "updated_at" in sql)
    ]
    assert session_writes == [], (
        f"Stateless /chat wrote to session tables: {session_writes}"
    )
