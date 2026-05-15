import asyncio
import importlib
import io
import sys
import time
import types
import uuid

import pytest
from starlette.datastructures import UploadFile


@pytest.mark.asyncio
async def test_get_engine_coalesces_concurrent_loads(monkeypatch) -> None:
    created = []

    class FakeEngine:
        def __init__(self):
            created.append(self)
            time.sleep(0.05)

    engine_module = types.ModuleType("docai.orchestrator.engine")
    engine_module.OrchestratorEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "docai.orchestrator.engine", engine_module)

    pipeline_module = types.ModuleType("docai.ingestion.pipeline")
    pipeline_module.IngestionPipeline = object
    monkeypatch.setitem(sys.modules, "docai.ingestion.pipeline", pipeline_module)

    sys.modules.pop("docai.api.routes", None)
    routes = importlib.import_module("docai.api.routes")

    first, second = await asyncio.gather(routes.get_engine(), routes.get_engine())

    assert first is second
    assert created == [first]


@pytest.mark.asyncio
async def test_chat_endpoint_passes_doc_scope_to_engine(monkeypatch) -> None:
    calls = []

    class FakeEngine:
        async def handle_query(self, query, session_id=None, doc_ids=None):
            calls.append((query, session_id, doc_ids))
            return {"response": "ok", "thinking": "", "citations": []}

    engine_module = types.ModuleType("docai.orchestrator.engine")
    engine_module.OrchestratorEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "docai.orchestrator.engine", engine_module)

    pipeline_module = types.ModuleType("docai.ingestion.pipeline")
    pipeline_module.IngestionPipeline = object
    monkeypatch.setitem(sys.modules, "docai.ingestion.pipeline", pipeline_module)

    sys.modules.pop("docai.api.routes", None)
    routes = importlib.import_module("docai.api.routes")

    doc_id = "11111111-1111-1111-1111-111111111111"
    request = routes.ChatRequest(query="describe these docs", session_id="session-1", doc_ids=[doc_id])
    response = await routes.chat_endpoint(request)

    assert response.response == "ok"
    assert response.thinking == ""
    assert response.citations == []
    assert calls == [("describe these docs", "session-1", [doc_id])]


@pytest.mark.asyncio
async def test_upload_endpoint_preserves_original_filename(monkeypatch, tmp_path) -> None:
    calls = []
    doc_id = uuid.UUID("22222222-2222-2222-2222-222222222222")

    class FakeEngine:
        pass

    class FakePipeline:
        async def ingest_document(self, file_path, doc_class="general", display_filename=None):
            calls.append((file_path, doc_class, display_filename))
            return doc_id

    engine_module = types.ModuleType("docai.orchestrator.engine")
    engine_module.OrchestratorEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "docai.orchestrator.engine", engine_module)

    pipeline_module = types.ModuleType("docai.ingestion.pipeline")
    pipeline_module.IngestionPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "docai.ingestion.pipeline", pipeline_module)

    monkeypatch.chdir(tmp_path)
    sys.modules.pop("docai.api.routes", None)
    routes = importlib.import_module("docai.api.routes")

    # Stub the post-ingest status query so we don't need a live Postgres.
    async def _fake_final(_doc_id):
        return {"status": "completed", "error_message": None}

    monkeypatch.setattr(routes, "_fetch_doc_final_status", _fake_final)

    file = UploadFile(filename="sp-500-report.pdf", file=io.BytesIO(b"pdf bytes"))
    response = await routes.upload_endpoint(file)

    assert response == {
        "status": "ready",
        "doc_id": str(doc_id),
        "filename": "sp-500-report.pdf",
    }
    assert calls[0][1:] == ("general", "sp-500-report.pdf")
