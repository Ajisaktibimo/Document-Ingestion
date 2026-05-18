import uuid
import sys
import asyncio
import os
import tempfile
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_mark_stage_updates_stage_heartbeat_and_updated_at(monkeypatch):
    providers = types.ModuleType("docai.providers")
    providers.get_ocr_service = lambda: MagicMock()
    providers.get_captioner = lambda: MagicMock()
    providers.get_dense_embedder = lambda: MagicMock()
    providers.get_sparse_embedder = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")
    qdrant_stub.QdrantStore = MagicMock
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)
    sys.modules.pop("docai.ingestion.pipeline", None)

    from docai.ingestion import pipeline as pipeline_module

    conn = MagicMock()
    conn.execute = AsyncMock()
    doc_id = uuid.uuid4()

    await pipeline_module.IngestionPipeline._mark_stage(conn, doc_id, "ocr")

    sql, *args = conn.execute.await_args.args
    assert "UPDATE doc_registry" in sql
    assert "ingestion_stage=$2" in sql
    assert "ingestion_heartbeat_at=NOW()" in sql
    assert "updated_at=NOW()" in sql
    assert args == [doc_id, "ocr"]


@pytest.mark.asyncio
async def test_run_with_heartbeat_marks_stage_before_operation_factory(monkeypatch):
    providers = types.ModuleType("docai.providers")
    providers.get_ocr_service = lambda: MagicMock()
    providers.get_captioner = lambda: MagicMock()
    providers.get_dense_embedder = lambda: MagicMock()
    providers.get_sparse_embedder = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")
    qdrant_stub.QdrantStore = MagicMock
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)
    sys.modules.pop("docai.ingestion.pipeline", None)

    from docai.ingestion import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.settings,
        "INGESTION_HEARTBEAT_INTERVAL_SECONDS",
        0,
    )
    pipeline = pipeline_module.IngestionPipeline()
    events = []
    conn = MagicMock()

    async def _execute(*_args):
        events.append("stage")

    conn.execute = AsyncMock(side_effect=_execute)

    async def _operation():
        return "done"

    def _operation_factory():
        events.append("operation")
        return _operation()

    result = await pipeline._run_with_heartbeat(
        conn,
        uuid.uuid4(),
        "ocr",
        _operation_factory,
    )

    assert result == "done"
    assert events == ["stage", "operation"]


@pytest.mark.asyncio
async def test_ingest_document_tracks_background_task_until_done(monkeypatch):
    providers = types.ModuleType("docai.providers")
    providers.get_ocr_service = lambda: MagicMock()
    providers.get_captioner = lambda: MagicMock()
    providers.get_dense_embedder = lambda: MagicMock()
    providers.get_sparse_embedder = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")
    qdrant_stub.QdrantStore = MagicMock
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)
    sys.modules.pop("docai.ingestion.pipeline", None)

    from docai.ingestion import pipeline as pipeline_module

    pipeline_module._active_tasks.clear()
    pipeline = pipeline_module.IngestionPipeline()
    doc_id = uuid.uuid4()
    release = asyncio.Event()

    async def _register(*_args):
        return doc_id, False

    async def _run_pipeline(*_args):
        await release.wait()

    fake_conn = MagicMock()
    fake_conn.close = AsyncMock()
    monkeypatch.setattr(pipeline_module.asyncpg, "connect", AsyncMock(return_value=fake_conn))
    monkeypatch.setattr(pipeline, "_register", _register)
    monkeypatch.setattr(pipeline, "_run_pipeline", _run_pipeline)

    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.close(fd)
        result = await pipeline.ingest_document(path)
        assert result == doc_id
        assert len(pipeline_module._active_tasks) == 1

        task = next(iter(pipeline_module._active_tasks))
        release.set()
        await asyncio.wait_for(task, timeout=1)
        assert pipeline_module._active_tasks == set()
    finally:
        if os.path.exists(path):
            os.unlink(path)
        pipeline_module._active_tasks.clear()
