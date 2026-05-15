"""Tests for the upload endpoint's HTTP status mapping based on the
final doc_registry status after ingestion.

- 200 + body status="ready"      when the row is `completed`.
- 202 + body status="processing" when another upload of the same file_hash
                                 was in-flight and the pipeline short-circuited.
- 500                              when the row is `failed` (with error_message
                                 surfaced in the detail) or pipeline returned None.
"""

import importlib
import io
import os
import sys
import types
import uuid

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile


def _install_routes(monkeypatch, *, ingest_returns, final_status):
    """Install fakes for OrchestratorEngine + IngestionPipeline, import a
    fresh routes module, and patch its ``_fetch_doc_final_status`` to
    return the supplied status payload.

    Returns the freshly-imported routes module.
    """

    class FakeEngine:
        pass

    class FakePipeline:
        async def ingest_document(self, file_path, doc_class="general", display_filename=None):
            return ingest_returns

    engine_module = types.ModuleType("docai.orchestrator.engine")
    engine_module.OrchestratorEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "docai.orchestrator.engine", engine_module)

    pipeline_module = types.ModuleType("docai.ingestion.pipeline")
    pipeline_module.IngestionPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "docai.ingestion.pipeline", pipeline_module)

    sys.modules.pop("docai.api.routes", None)
    routes = importlib.import_module("docai.api.routes")

    async def _fake_final(_doc_id):
        return final_status

    monkeypatch.setattr(routes, "_fetch_doc_final_status", _fake_final)
    return routes


def _upload_file() -> UploadFile:
    return UploadFile(filename="report.pdf", file=io.BytesIO(b"%PDF-1.4 fake"))


def _cleanup_uploads_dir(routes_uploaded_path: str | None = None) -> None:
    """The route writes the upload to `uploads/<uuid>.pdf` from cwd.
    Best-effort cleanup so test runs don't accumulate files."""
    upload_dir = "uploads"
    if not os.path.isdir(upload_dir):
        return
    for name in os.listdir(upload_dir):
        if name.endswith(".pdf"):
            try:
                os.unlink(os.path.join(upload_dir, name))
            except OSError:
                pass


@pytest.mark.asyncio
async def test_upload_returns_ready_when_status_completed(monkeypatch) -> None:
    doc_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    routes = _install_routes(
        monkeypatch,
        ingest_returns=doc_id,
        final_status={"status": "completed", "error_message": None},
    )

    try:
        response = await routes.upload_endpoint(_upload_file())
    finally:
        _cleanup_uploads_dir()

    assert response == {
        "status": "ready",
        "doc_id": str(doc_id),
        "filename": "report.pdf",
    }


@pytest.mark.asyncio
async def test_upload_returns_202_when_status_processing(monkeypatch) -> None:
    doc_id = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    routes = _install_routes(
        monkeypatch,
        ingest_returns=doc_id,
        final_status={"status": "processing", "error_message": None},
    )

    try:
        response = await routes.upload_endpoint(_upload_file())
    finally:
        _cleanup_uploads_dir()

    # JSONResponse — inspect status_code and body.
    assert response.status_code == 202
    import json
    body = json.loads(response.body)
    assert body["status"] == "processing"
    assert body["doc_id"] == str(doc_id)
    assert body["filename"] == "report.pdf"
    assert "indexing" in body["message"].lower()


@pytest.mark.asyncio
async def test_upload_raises_500_when_status_failed_with_error(monkeypatch) -> None:
    doc_id = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    routes = _install_routes(
        monkeypatch,
        ingest_returns=doc_id,
        final_status={"status": "failed", "error_message": "ocr exploded"},
    )

    try:
        with pytest.raises(HTTPException) as exc:
            await routes.upload_endpoint(_upload_file())
    finally:
        _cleanup_uploads_dir()

    assert exc.value.status_code == 500
    assert "ocr exploded" in exc.value.detail


@pytest.mark.asyncio
async def test_upload_raises_500_when_pipeline_returns_none(monkeypatch) -> None:
    routes = _install_routes(
        monkeypatch,
        ingest_returns=None,
        final_status={"status": "failed", "error_message": "irrelevant"},
    )

    try:
        with pytest.raises(HTTPException) as exc:
            await routes.upload_endpoint(_upload_file())
    finally:
        _cleanup_uploads_dir()

    assert exc.value.status_code == 500
    assert "ingestion failed" in exc.value.detail.lower()
