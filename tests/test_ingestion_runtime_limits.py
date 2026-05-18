import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _import_pipeline_with_stubs(monkeypatch):
    calls = {
        "ocr": 0,
        "captioner": 0,
        "dense": 0,
        "sparse": 0,
        "qdrant": 0,
    }

    providers = types.ModuleType("docai.providers")

    def _provider(name):
        def _factory():
            calls[name] += 1
            return MagicMock()

        return _factory

    providers.get_ocr_service = _provider("ocr")
    providers.get_captioner = _provider("captioner")
    providers.get_dense_embedder = _provider("dense")
    providers.get_sparse_embedder = _provider("sparse")
    monkeypatch.setitem(sys.modules, "docai.providers", providers)

    qdrant_stub = types.ModuleType("docai.retrieval.qdrant_client")

    class FakeQdrantStore:
        def __init__(self):
            calls["qdrant"] += 1

    qdrant_stub.QdrantStore = FakeQdrantStore
    monkeypatch.setitem(sys.modules, "docai.retrieval.qdrant_client", qdrant_stub)

    sys.modules.pop("docai.ingestion.pipeline", None)
    module = importlib.import_module("docai.ingestion.pipeline")
    return module, calls


def test_pipeline_constructor_does_not_load_heavy_services(monkeypatch):
    module, calls = _import_pipeline_with_stubs(monkeypatch)

    pipeline = module.IngestionPipeline()

    assert calls == {
        "ocr": 0,
        "captioner": 0,
        "dense": 0,
        "sparse": 0,
        "qdrant": 0,
    }

    _ = pipeline.ocr_service
    _ = pipeline.qdrant_store

    assert calls["ocr"] == 1
    assert calls["qdrant"] == 1


def test_caption_candidates_applies_media_limits(monkeypatch):
    module, _ = _import_pipeline_with_stubs(monkeypatch)
    monkeypatch.setattr(module.settings, "MAX_IMAGES_PER_DOC", 1)
    monkeypatch.setattr(module.settings, "MAX_TABLES_PER_DOC", 2)

    ocr_result = SimpleNamespace(
        images=[SimpleNamespace(media_id="image-1"), SimpleNamespace(media_id="image-2")],
        tables=[
            SimpleNamespace(media_id="table-1"),
            SimpleNamespace(media_id="table-2"),
            SimpleNamespace(media_id="table-3"),
        ],
    )

    selected = module.IngestionPipeline._caption_candidates(ocr_result)

    assert [media.media_id for media in selected] == [
        "image-1",
        "table-1",
        "table-2",
    ]
