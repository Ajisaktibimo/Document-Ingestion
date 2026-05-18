"""Tests for the IngestionPipeline stale-processing detection.

When a doc_registry row sits at status='processing' but its updated_at is
either NULL (pre-T3 orphan from a crashed ingestion) or older than
``settings.STALE_PROCESSING_THRESHOLD_SECONDS``, a fresh upload of the same
file_hash must treat it as stale and re-ingest. A row with a recent
updated_at must continue to short-circuit (genuinely in flight elsewhere).
"""

import os
import sys
import tempfile
import types
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_heavy_provider_stubs(monkeypatch):
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
    sys.modules.pop("docai.ingestion.pipeline", None)
    import importlib
    pipeline_module = importlib.import_module("docai.ingestion.pipeline")
    pipeline_module._active_tasks.clear()
    pipeline = pipeline_module.IngestionPipeline()
    return pipeline, pipeline_module


async def _await_background_tasks(pipeline_module):
    tasks = list(pipeline_module._active_tasks)
    if tasks:
        await asyncio.gather(*tasks)


def _make_temp_file() -> str:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b"%PDF-1.4\n%fake bytes for hashing\n")
    return path


def _build_fake_conn(*, existing_row):
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
async def test_stale_null_updated_at_triggers_reingestion(monkeypatch) -> None:
    """A row with status='processing' AND updated_at IS NULL is the
    signature of a pre-T3 orphan. It must NOT short-circuit; instead the
    retry path must run (DELETE old chunks + UPDATE row to processing +
    proceed with full ingestion). We assert the OCR service is called.
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    existing_row = {
        "doc_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "status": "processing",
        "page_count": None,
        "filename": "old-name.pdf",
        "updated_at": None,
    }
    fake_conn = _build_fake_conn(existing_row=existing_row)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    # Make OCR raise so we don't have to mock the entire pipeline downstream;
    # we only care that OCR was called (proving the retry path ran).
    ocr_called = []

    async def _ocr(_path):
        ocr_called.append(_path)
        raise RuntimeError("stop here — proves retry path ran")

    pipeline.ocr_service.process_document = AsyncMock(side_effect=_ocr)
    pipeline.qdrant_store.delete_document = AsyncMock(return_value=None)

    tmp = _make_temp_file()
    try:
        result = await pipeline.ingest_document(tmp, doc_class="general")
        await _await_background_tasks(pipeline_module)
    finally:
        os.unlink(tmp)

    assert ocr_called, "OCR was not called — stale-processing did NOT fall through to retry"
    # Registration still returns immediately; the background task records the
    # OCR failure separately.
    assert result == existing_row["doc_id"]


@pytest.mark.asyncio
async def test_stale_old_updated_at_triggers_reingestion(monkeypatch) -> None:
    """A row with status='processing' and updated_at older than the
    configured threshold must be treated as stale and re-ingested.
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    # 2 hours old — well beyond the default 600s threshold.
    stale_ts = datetime.now(timezone.utc) - timedelta(hours=2)
    existing_row = {
        "doc_id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
        "status": "processing",
        "page_count": None,
        "filename": "old-name.pdf",
        "updated_at": stale_ts,
    }
    fake_conn = _build_fake_conn(existing_row=existing_row)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    ocr_called = []

    async def _ocr(_path):
        ocr_called.append(_path)
        raise RuntimeError("stop here")

    pipeline.ocr_service.process_document = AsyncMock(side_effect=_ocr)
    pipeline.qdrant_store.delete_document = AsyncMock(return_value=None)

    tmp = _make_temp_file()
    try:
        await pipeline.ingest_document(tmp, doc_class="general")
        await _await_background_tasks(pipeline_module)
    finally:
        os.unlink(tmp)

    assert ocr_called, "OCR was not called — stale-old-updated_at did NOT fall through to retry"


@pytest.mark.asyncio
async def test_fresh_processing_returns_existing_doc_id(monkeypatch) -> None:
    """A row with status='processing' and a recent updated_at means a
    genuinely in-flight ingestion (e.g., a parallel request). The
    pipeline must short-circuit and return the existing doc_id WITHOUT
    calling OCR or touching chunks.
    """
    pipeline, pipeline_module = _make_pipeline(monkeypatch)

    fresh_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
    existing_id = uuid.UUID("33333333-3333-3333-3333-333333333333")
    existing_row = {
        "doc_id": existing_id,
        "status": "processing",
        "page_count": None,
        "filename": "in-flight.pdf",
        "updated_at": fresh_ts,
    }
    fake_conn = _build_fake_conn(existing_row=existing_row)
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    ocr_mock = AsyncMock()
    pipeline.ocr_service.process_document = ocr_mock
    pipeline.qdrant_store.delete_document = AsyncMock()

    tmp = _make_temp_file()
    try:
        result = await pipeline.ingest_document(tmp, doc_class="general")
    finally:
        os.unlink(tmp)

    assert result == existing_id, "fresh-processing did not return the existing doc_id"
    assert ocr_mock.await_count == 0, "OCR should NOT be called when the existing row is fresh"
    # No DELETE / UPDATE / INSERT should have run for the registry either.
    assert fake_conn.execute_calls == [], (
        f"expected no execute() calls on fresh-processing, got: {fake_conn.execute_calls}"
    )
