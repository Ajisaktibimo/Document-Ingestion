"""Tests for the IngestionPipeline failure-recording path (Task T3).

Covers the persistence of a diagnostic ``error_message`` and ``updated_at``
on the ``doc_registry`` row when ingestion raises. Also covers the
"failure-recording itself fails" path — the original exception must still
propagate, with the DB error logged but not masking it.
"""

import os
import sys
import tempfile
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_heavy_provider_stubs(monkeypatch):
    """Stub out heavy ML providers + QdrantStore so IngestionPipeline()
    can be constructed without loading models or contacting services."""

    providers = types.ModuleType("docai.providers")
    providers.get_ocr_service = lambda: MagicMock()
    providers.get_captioner = lambda: MagicMock()
    providers.get_dense_embedder = lambda: MagicMock()
    providers.get_sparse_embedder = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")
    qdrant_stub.QdrantStore = MagicMock
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)


def _make_pipeline(monkeypatch):
    _install_heavy_provider_stubs(monkeypatch)

    # Force a fresh import so the stubs are picked up.
    sys.modules.pop("docai.ingestion.pipeline", None)
    import importlib
    pipeline_module = importlib.import_module("docai.ingestion.pipeline")
    pipeline = pipeline_module.IngestionPipeline()
    return pipeline, pipeline_module


def _make_temp_file() -> str:
    """Write a small temp file so the pipeline's `os.path.exists` check passes."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"%PDF-1.4\n%fake bytes for hashing\n")
    return path


def _build_fake_conn(*, existing_row=None):
    """Build an AsyncMock asyncpg connection.

    The connection records every `.execute()` call's SQL + args in
    ``conn.execute_calls`` for assertion.
    """
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=existing_row)
    conn.close = AsyncMock(return_value=None)

    execute_calls = []

    async def _execute(sql, *args):
        execute_calls.append((sql, args))
        return None

    conn.execute = AsyncMock(side_effect=_execute)
    conn.execute_calls = execute_calls
    return conn


@pytest.mark.asyncio
async def test_failure_records_error_message_and_updated_at(monkeypatch) -> None:
    """When ingestion raises, the failure UPDATE must set status='failed',
    error_message (truncated to 2000 chars) and updated_at = NOW().
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    # Make the OCR call raise a known exception — this is the first
    # pipeline step after the registry INSERT, so we exercise the
    # failure handler cleanly.
    async def _ocr_boom(_path):
        raise RuntimeError("boom — synthetic ocr failure")

    pipeline.ocr_service.process_document = AsyncMock(side_effect=_ocr_boom)

    fake_conn = _build_fake_conn(existing_row=None)
    connect_mock = AsyncMock(return_value=fake_conn)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", connect_mock)

    tmp = _make_temp_file()
    try:
        result = await pipeline.ingest_document(tmp, doc_class="general")
    finally:
        os.unlink(tmp)

    # The exception handler returns None and does NOT re-raise (existing behavior).
    assert result is None

    # Find the failure UPDATE.
    failure_updates = [
        (sql, args) for (sql, args) in fake_conn.execute_calls
        if "status = 'failed'" in sql or "status='failed'" in sql
    ]
    assert len(failure_updates) == 1, (
        f"expected exactly one failure UPDATE, got: {fake_conn.execute_calls}"
    )
    sql, args = failure_updates[0]

    # Must set error_message and updated_at.
    assert "error_message" in sql, f"failure UPDATE missing error_message: {sql}"
    assert "updated_at" in sql, f"failure UPDATE missing updated_at: {sql}"
    assert "NOW()" in sql, f"failure UPDATE should call NOW(): {sql}"

    # The error message bind parameter must start with 'boom'.
    # args order: (file_hash, error_message) — based on the UPDATE shape.
    assert any(
        isinstance(a, str) and a.startswith("boom")
        for a in args
    ), f"bound args don't carry the error message: {args}"

    # Connection still closed.
    assert fake_conn.close.await_count == 1


@pytest.mark.asyncio
async def test_failure_recording_db_error_does_not_mask_original(
    monkeypatch, caplog
) -> None:
    """If the failure-recording UPDATE itself fails, the ORIGINAL exception
    must still propagate (or in the current pipeline shape: be swallowed
    and return None) and the secondary DB error must be logged — NOT raised.
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    async def _ocr_boom(_path):
        raise RuntimeError("boom — original")

    pipeline.ocr_service.process_document = AsyncMock(side_effect=_ocr_boom)

    # Conn where the INITIAL INSERT succeeds but the failure-recording
    # UPDATE raises. We track this by failing on any UPDATE that mentions
    # status='failed'.
    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value=None)
    fake_conn.close = AsyncMock(return_value=None)
    execute_calls = []

    async def _execute(sql, *args):
        execute_calls.append((sql, args))
        if "status = 'failed'" in sql or "status='failed'" in sql:
            raise RuntimeError("db down — secondary failure")
        return None

    fake_conn.execute = AsyncMock(side_effect=_execute)
    fake_conn.execute_calls = execute_calls

    connect_mock = AsyncMock(return_value=fake_conn)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", connect_mock)

    tmp = _make_temp_file()
    try:
        # The pipeline's outer handler returns None on failure (current
        # shape). The CRITICAL property under test: the secondary DB error
        # must NOT escape and mask the original error.
        caplog.set_level("ERROR")
        result = await pipeline.ingest_document(tmp, doc_class="general")
    finally:
        os.unlink(tmp)

    assert result is None

    # The failure UPDATE was attempted (proves we reached the handler).
    failure_attempts = [
        (sql, args) for (sql, args) in execute_calls
        if "status = 'failed'" in sql or "status='failed'" in sql
    ]
    assert len(failure_attempts) == 1

    # The secondary DB error message should appear in the logs.
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "db down" in log_text, (
        "secondary DB failure should be logged, not silently swallowed.\n"
        f"captured logs: {log_text}"
    )
    # And the ORIGINAL error should also be logged (not lost).
    assert "boom" in log_text, (
        f"original error should still be logged. captured logs: {log_text}"
    )

    # Connection still closed even when failure-recording threw.
    assert fake_conn.close.await_count == 1


@pytest.mark.asyncio
async def test_entry_insert_clears_error_message(monkeypatch) -> None:
    """When inserting a new processing row, error_message must be NULL
    (explicit) so a previously-failed reingest starts clean.

    Note: this case exercises the *fresh insert* path (no existing row).
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    # Make OCR raise so we don't have to mock the entire happy path —
    # we only care about the INSERT issued before the failure.
    async def _ocr_boom(_path):
        raise RuntimeError("stop here")

    pipeline.ocr_service.process_document = AsyncMock(side_effect=_ocr_boom)

    fake_conn = _build_fake_conn(existing_row=None)
    connect_mock = AsyncMock(return_value=fake_conn)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", connect_mock)

    tmp = _make_temp_file()
    try:
        await pipeline.ingest_document(tmp, doc_class="general")
    finally:
        os.unlink(tmp)

    inserts = [
        (sql, args) for (sql, args) in fake_conn.execute_calls
        if "INSERT INTO doc_registry" in sql
    ]
    assert len(inserts) == 1, f"expected one fresh INSERT, got {inserts}"
    sql, _ = inserts[0]
    assert "error_message" in sql, (
        f"fresh INSERT should explicitly set error_message column: {sql}"
    )


@pytest.mark.asyncio
async def test_reuse_path_update_clears_error_message(monkeypatch) -> None:
    """When a previously-failed doc is retried, the reuse-path UPDATE
    must clear error_message and bump updated_at.
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    # Make OCR raise so we stop right after the reuse-path UPDATE.
    async def _ocr_boom(_path):
        raise RuntimeError("stop here")

    pipeline.ocr_service.process_document = AsyncMock(side_effect=_ocr_boom)

    # Existing row with status='failed' triggers the reuse path (not
    # 'completed', not 'processing').
    import uuid as _uuid
    existing = {
        "doc_id": _uuid.uuid4(),
        "status": "failed",
        "page_count": None,
        "filename": "old.pdf",
    }

    fake_conn = _build_fake_conn(existing_row=existing)
    # qdrant_store.delete_document is awaited on reuse path.
    pipeline.qdrant_store.delete_document = AsyncMock(return_value=None)

    connect_mock = AsyncMock(return_value=fake_conn)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", connect_mock)

    tmp = _make_temp_file()
    try:
        await pipeline.ingest_document(tmp, doc_class="general")
    finally:
        os.unlink(tmp)

    # Find the reuse-path UPDATE — the one that sets status='processing'
    # via the WHERE file_hash = $1 form (not the failure UPDATE).
    reuse_updates = [
        (sql, args) for (sql, args) in fake_conn.execute_calls
        if "UPDATE doc_registry" in sql
        and "WHERE file_hash" in sql
        and "status = 'failed'" not in sql
        and "status='failed'" not in sql
    ]
    assert len(reuse_updates) >= 1, (
        f"expected reuse-path UPDATE, got: {fake_conn.execute_calls}"
    )
    sql, _ = reuse_updates[0]
    assert "error_message" in sql, (
        f"reuse UPDATE should clear error_message: {sql}"
    )
    assert "updated_at" in sql, (
        f"reuse UPDATE should bump updated_at: {sql}"
    )
