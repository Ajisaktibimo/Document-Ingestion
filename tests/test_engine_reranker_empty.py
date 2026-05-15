import sys, types, uuid
import os
import pytest
from unittest.mock import MagicMock, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _install_stubs(monkeypatch):
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


@pytest.mark.asyncio
async def test_rerank_empty_results_no_index_error(monkeypatch):
    """_get_vector_context must not raise IndexError when reranker returns 0 scores."""
    _install_stubs(monkeypatch)
    sys.modules.pop("docai.orchestrator.engine", None)
    import importlib
    eng_mod = importlib.import_module("docai.orchestrator.engine")
    engine = eng_mod.OrchestratorEngine()

    # Mock async methods
    doc_id = str(uuid.uuid4())
    hits = [{"id": "a1b2", "doc_id": doc_id, "text": "hello", "score": 0.9, "heading": "", "heading_path": "", "page_num": 0}]

    engine.dense_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
    engine.sparse_embedder.embed_query = AsyncMock(return_value={"term1": 0.5})
    engine.qdrant_store.search_hybrid = AsyncMock(return_value=hits)

    # Reranker returns empty scores list — this is the bug trigger
    engine.reranker.rerank = AsyncMock(return_value=[])

    ctx, citations = await engine._get_vector_context("test query", doc_ids=None)
    assert citations == []
    assert ctx == ""
