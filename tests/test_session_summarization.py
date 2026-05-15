"""Tests for engine session summarization (sub-project 1, T11).

Cover cadence detection, successful summarization writeback, graceful
fallback when the LLM fails (keep old summary), and the master kill
switch (ENABLE_SESSION_SUMMARIZATION=False).
"""
import sys
import types
import uuid
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
async def test_should_resummarize_triggers_after_cadence(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(engine_module.settings, "SUMMARIZATION_CADENCE_MESSAGES", 10, raising=False)
    monkeypatch.setattr(engine_module.settings, "SESSION_WINDOW_MESSAGES", 20, raising=False)

    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value={
        "max_id": 50, "summary_through_message_id": 10,
    })
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    sid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    assert await engine._should_resummarize(sid) is True


@pytest.mark.asyncio
async def test_should_resummarize_skips_when_below_cadence(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(engine_module.settings, "SUMMARIZATION_CADENCE_MESSAGES", 10, raising=False)
    monkeypatch.setattr(engine_module.settings, "SESSION_WINDOW_MESSAGES", 20, raising=False)

    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value={
        "max_id": 25, "summary_through_message_id": 10,
    })
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    sid = uuid.UUID("22222222-2222-2222-2222-222222222222")
    assert await engine._should_resummarize(sid) is False


@pytest.mark.asyncio
async def test_maybe_summarize_writes_summary(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(engine_module.settings, "ENABLE_SESSION_SUMMARIZATION", True, raising=False)

    captured: list = []
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(return_value=[
        {"id": 1, "role": "user", "content": "Q1"},
        {"id": 2, "role": "assistant", "content": "A1"},
    ])
    fake_conn.fetchrow = AsyncMock(return_value={"summary": None, "summary_through_message_id": None})

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    engine.llm_client.generate_response = AsyncMock(
        return_value={"content": "Summary text.", "thinking": "", "reason": None}
    )

    sid = uuid.UUID("33333333-3333-3333-3333-333333333333")
    await engine._maybe_summarize(sid)

    updates = [(sql, args) for sql, args in captured if "UPDATE sessions" in sql and "summary" in sql]
    assert len(updates) == 1
    sql, args = updates[0]
    assert "Summary text." in args


@pytest.mark.asyncio
async def test_maybe_summarize_keeps_old_on_llm_failure(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(engine_module.settings, "ENABLE_SESSION_SUMMARIZATION", True, raising=False)

    captured: list = []
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(return_value=[
        {"id": 1, "role": "user", "content": "Q"},
    ])
    fake_conn.fetchrow = AsyncMock(return_value={"summary": "OLD SUMMARY", "summary_through_message_id": 0})

    async def _execute(sql, *args):
        captured.append((sql, args))
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.close = AsyncMock(return_value=None)
    monkeypatch.setattr(engine_module.asyncpg, "connect",
                        AsyncMock(return_value=fake_conn))

    engine.llm_client.generate_response = AsyncMock(side_effect=RuntimeError("ollama down"))

    sid = uuid.UUID("44444444-4444-4444-4444-444444444444")
    await engine._maybe_summarize(sid)

    updates = [(sql, args) for sql, args in captured if "UPDATE sessions" in sql and "summary" in sql]
    assert len(updates) == 0
