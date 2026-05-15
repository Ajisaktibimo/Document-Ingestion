"""Tests for the Engine pre-flight helpers (Task T5).

Covers ``OrchestratorEngine._fetch_doc_statuses``: the small private helper
that bulk-fetches document statuses for a list of doc_ids. Used by Task T2
for pre-flight checks before query execution.
"""

import logging
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_heavy_provider_stubs(monkeypatch):
    """Stub out heavy ML providers so OrchestratorEngine() can be instantiated
    without loading models or contacting external services."""

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

    # Force a fresh import so the stubs are picked up.
    sys.modules.pop("docai.orchestrator.engine", None)
    import importlib
    engine_module = importlib.import_module("docai.orchestrator.engine")
    engine = engine_module.OrchestratorEngine()
    return engine, engine_module


class _FakeRow(dict):
    """Mimic asyncpg.Record: dict-like ``row["col"]`` access."""


@pytest.mark.asyncio
async def test_fetch_doc_statuses_returns_list_of_doc_status(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()

    fake_rows = [
        _FakeRow(
            doc_id=uuid1,
            filename="alpha.pdf",
            status="completed",
            error_message=None,
        ),
        _FakeRow(
            doc_id=uuid2,
            filename="beta.pdf",
            status="failed",
            error_message="ocr crashed",
        ),
    ]

    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(return_value=fake_rows)
    fake_conn.close = AsyncMock(return_value=None)

    connect_mock = AsyncMock(return_value=fake_conn)
    monkeypatch.setattr(engine_module.asyncpg, "connect", connect_mock)

    result = await engine._fetch_doc_statuses([uuid1, uuid2])

    assert connect_mock.await_count == 1
    assert fake_conn.fetch.await_count == 1
    fetch_call = fake_conn.fetch.await_args
    # The SQL should reference doc_registry and select error_message.
    sql = fetch_call.args[0]
    assert "doc_registry" in sql
    assert "error_message" in sql
    # Bound parameter should be the list of uuids.
    assert fetch_call.args[1] == [uuid1, uuid2]
    # Connection must be closed.
    assert fake_conn.close.await_count == 1

    DocStatus = engine_module.DocStatus
    assert result == [
        DocStatus(
            doc_id=uuid1,
            filename="alpha.pdf",
            status="completed",
            error_message=None,
        ),
        DocStatus(
            doc_id=uuid2,
            filename="beta.pdf",
            status="failed",
            error_message="ocr crashed",
        ),
    ]
    # Each item is the expected type.
    for item in result:
        assert isinstance(item, DocStatus)


@pytest.mark.asyncio
async def test_fetch_doc_statuses_empty_input_short_circuits(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)

    connect_mock = AsyncMock()
    monkeypatch.setattr(engine_module.asyncpg, "connect", connect_mock)

    result = await engine._fetch_doc_statuses([])

    assert result == []
    # No DB roundtrip on empty input.
    assert connect_mock.await_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# handle_query pre-flight tests (Task T2)
# ─────────────────────────────────────────────────────────────────────────────


def _patch_engine_for_handle_query(engine, engine_module, monkeypatch, *, statuses, llm_result):
    """Wire common mocks for handle_query tests.

    - ``_fetch_doc_statuses`` returns the supplied DocStatus list.
    - ``gather_unified_context`` returns a stub context dict.
    - ``llm_client.generate_response`` returns ``llm_result`` (or raises if None).
    Returns the AsyncMocks so the test can assert on call counts/args.
    """
    fetch_mock = AsyncMock(return_value=statuses)
    monkeypatch.setattr(engine, "_fetch_doc_statuses", fetch_mock)

    gather_mock = AsyncMock(
        return_value={
            "vector": "vector context",
            "graph": "graph context",
            "citations": [],
            "document_inventory": "inventory text",
        }
    )
    monkeypatch.setattr(engine, "gather_unified_context", gather_mock)

    llm_mock = AsyncMock(return_value=llm_result)
    engine.llm_client = MagicMock()
    engine.llm_client.generate_response = llm_mock

    return fetch_mock, gather_mock, llm_mock


@pytest.mark.asyncio
async def test_handle_query_all_scoped_docs_processing_short_circuits(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    statuses = [
        DocStatus(doc_id=uuid1, filename="alpha.pdf", status="processing", error_message=None),
        DocStatus(doc_id=uuid2, filename="beta.pdf", status="processing", error_message=None),
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={"content": "should not be used", "thinking": ""},
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid1), str(uuid2)],
    )

    # Pre-flight should be the only DB call; LLM and retrieval skipped entirely.
    assert fetch_mock.await_count == 1
    assert llm_mock.await_count == 0
    assert gather_mock.await_count == 0

    assert result["guard"] == "all_scoped_docs_unavailable"
    assert "None of the requested documents are ready yet" in result["response"]
    assert "Indexing:" in result["response"]
    assert "alpha.pdf" in result["response"]
    assert "beta.pdf" in result["response"]
    assert result["thinking"] == ""
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_handle_query_mixed_scope_falls_back_to_completed_only(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid_done = uuid.uuid4()
    uuid_pending = uuid.uuid4()
    statuses = [
        DocStatus(doc_id=uuid_done, filename="done.pdf", status="completed", error_message=None),
        DocStatus(doc_id=uuid_pending, filename="pending.pdf", status="processing", error_message=None),
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={"content": "actual answer", "thinking": "some reasoning"},
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid_done), str(uuid_pending)],
    )

    # Retrieval and LLM both invoked.
    assert fetch_mock.await_count == 1
    assert gather_mock.await_count == 1
    assert llm_mock.await_count == 1

    # Retrieval is scoped to the completed doc only.
    gather_kwargs = gather_mock.await_args.kwargs
    scoped = gather_kwargs.get("doc_ids")
    assert scoped == [str(uuid_done)]

    # Response is prefixed with the exclusion note and still carries the LLM answer.
    assert result["response"].startswith("Note: ")
    assert "pending.pdf" in result["response"]
    assert "still indexing" in result["response"]
    assert "actual answer" in result["response"]
    # Thinking is preserved untouched.
    assert result["thinking"] == "some reasoning"


@pytest.mark.asyncio
async def test_handle_query_promotes_empty_content_with_thinking_reason(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid_done = uuid.uuid4()
    statuses = [
        DocStatus(doc_id=uuid_done, filename="done.pdf", status="completed", error_message=None),
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={
            "content": "",
            "thinking": "model reasoned but produced no answer",
            "reason": "empty_content_with_thinking",
        },
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid_done)],
    )

    assert llm_mock.await_count == 1
    assert result["response"] == (
        "The model returned reasoning but no answer. Reasoning is shown below."
    )
    # Thinking is preserved so the UI can render it.
    assert result["thinking"] == "model reasoned but produced no answer"


@pytest.mark.asyncio
async def test_handle_query_short_circuit_includes_failed_docs(monkeypatch) -> None:
    """Pre-flight short-circuit must surface BOTH processing and failed docs.

    A failed doc with ``error_message=None`` must render the ``"unknown error"``
    fallback. Retrieval and LLM must NOT be called when nothing is usable.
    """
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid_processing = uuid.uuid4()
    uuid_failed = uuid.uuid4()
    statuses = [
        DocStatus(
            doc_id=uuid_processing,
            filename="indexing.pdf",
            status="processing",
            error_message=None,
        ),
        DocStatus(
            doc_id=uuid_failed,
            filename="broken.pdf",
            status="failed",
            error_message=None,
        ),
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={"content": "should not be used", "thinking": ""},
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid_processing), str(uuid_failed)],
    )

    # Pre-flight short-circuit: no retrieval, no LLM call.
    assert fetch_mock.await_count == 1
    assert gather_mock.await_count == 0
    assert llm_mock.await_count == 0

    assert result["guard"] == "all_scoped_docs_unavailable"
    response = result["response"]
    assert "Indexing:" in response
    assert "indexing.pdf" in response
    assert "Failed:" in response
    assert "broken.pdf" in response
    # error_message=None must render the fallback string.
    assert "unknown error" in response
    assert result["thinking"] == ""
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_handle_query_promotes_empty_response_reason(monkeypatch) -> None:
    """LLM ``reason="empty_response"`` must be promoted to the spec-locked
    user-facing string, with ``thinking`` preserved as the empty string."""
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid_done = uuid.uuid4()
    statuses = [
        DocStatus(doc_id=uuid_done, filename="done.pdf", status="completed", error_message=None),
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={
            "content": "",
            "thinking": "",
            "reason": "empty_response",
        },
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid_done)],
    )

    assert llm_mock.await_count == 1
    assert result["response"] == (
        "The model produced no output. Please retry, or rephrase your question."
    )
    assert result["thinking"] == ""


@pytest.mark.asyncio
async def test_fetch_doc_statuses_swallows_db_error_returns_empty(
    monkeypatch, caplog
) -> None:
    """If asyncpg.connect raises, the helper must log a warning and return []
    so callers (pre-flight) can degrade to "all unknown" handling."""
    engine, engine_module = _make_engine(monkeypatch)

    async def boom(*args, **kwargs):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(engine_module.asyncpg, "connect", boom)

    with caplog.at_level(logging.WARNING, logger=engine_module.logger.name):
        result = await engine._fetch_doc_statuses([uuid.uuid4()])

    assert result == []
    assert any(
        "Failed to fetch doc statuses" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_handle_query_short_circuit_includes_missing_docs(monkeypatch) -> None:
    """Pre-flight short-circuit must surface BOTH ``processing`` docs and
    requested doc_ids that have no row in ``doc_registry`` at all.

    Retrieval and LLM must not be invoked when nothing is usable.
    """
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid_processing = uuid.uuid4()
    uuid_missing = uuid.uuid4()
    statuses = [
        DocStatus(
            doc_id=uuid_processing,
            filename="indexing.pdf",
            status="processing",
            error_message=None,
        ),
        # uuid_missing intentionally absent from statuses (simulates a
        # doc_id that is not present in doc_registry).
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={"content": "should not be used", "thinking": ""},
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid_processing), str(uuid_missing)],
    )

    # Pre-flight short-circuit: no retrieval, no LLM call.
    assert fetch_mock.await_count == 1
    assert gather_mock.await_count == 0
    assert llm_mock.await_count == 0

    assert result["guard"] == "all_scoped_docs_unavailable"
    response = result["response"]
    assert "Indexing:" in response
    assert "indexing.pdf" in response
    assert "Unknown:" in response
    assert str(uuid_missing) in response
    assert result["thinking"] == ""
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_handle_query_partial_scope_notes_missing_docs(monkeypatch) -> None:
    """One requested doc is ``completed`` and a second requested doc_id has
    NO row in ``doc_registry``. The response must be prefixed with the
    exclusion note (``unknown: <id>``) and the LLM should run with only the
    completed doc in scope.
    """
    engine, engine_module = _make_engine(monkeypatch)
    DocStatus = engine_module.DocStatus

    uuid_done = uuid.uuid4()
    uuid_missing = uuid.uuid4()
    statuses = [
        DocStatus(
            doc_id=uuid_done,
            filename="done.pdf",
            status="completed",
            error_message=None,
        ),
        # uuid_missing intentionally absent from statuses.
    ]
    fetch_mock, gather_mock, llm_mock = _patch_engine_for_handle_query(
        engine, engine_module, monkeypatch,
        statuses=statuses,
        llm_result={"content": "actual answer", "thinking": ""},
    )

    result = await engine.handle_query(
        "any question",
        session_id="s1",
        doc_ids=[str(uuid_done), str(uuid_missing)],
    )

    # Retrieval and LLM both invoked exactly once.
    assert fetch_mock.await_count == 1
    assert gather_mock.await_count == 1
    assert llm_mock.await_count == 1

    # Retrieval is scoped to the completed doc only.
    gather_kwargs = gather_mock.await_args.kwargs
    assert gather_kwargs.get("doc_ids") == [str(uuid_done)]

    # Exact prefix check (Note + 'unknown: <id>' + the LLM answer).
    expected_prefix = (
        f"Note: excluded from this answer — unknown: {uuid_missing}.\n\n"
    )
    assert result["response"].startswith(expected_prefix)
    assert "actual answer" in result["response"]
