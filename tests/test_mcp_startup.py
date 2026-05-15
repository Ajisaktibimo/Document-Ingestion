import importlib
import json
import sys
import types

import pytest


def _reload_mcp_server(monkeypatch, engine_events=None, pipeline_events=None):
    engine_events = engine_events if engine_events is not None else []
    pipeline_events = pipeline_events if pipeline_events is not None else []

    fastmcp = types.ModuleType("fastmcp")

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name
            self.added_tools = []
            self.run_calls = []

        def tool(self):
            return lambda func: func

        def add_tool(self, func):
            self.added_tools.append(func)

        def run(self, transport, **kwargs):
            self.run_calls.append((transport, kwargs))

    fastmcp.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp)

    asyncpg = types.ModuleType("asyncpg")
    monkeypatch.setitem(sys.modules, "asyncpg", asyncpg)

    engine_module = types.ModuleType("docai.orchestrator.engine")

    class FakeDenseEmbedder:
        async def embed_query(self, query):
            return [0.1, 0.2]

    class FakeSparseEmbedder:
        async def embed_query(self, query):
            return {"indices": [1], "values": [0.5]}

    class FakeQdrantStore:
        async def search_hybrid(self, query_dense, query_sparse, limit):
            return [{"doc_id": "doc-1", "text": "Revenue grew.", "score": 0.8}]

    class FakeReranker:
        async def rerank(self, query, documents, top_k):
            return [(0, 0.99)]

    class FakeOrchestratorEngine:
        def __init__(self):
            engine_events.append("created")
            self.dense_embedder = FakeDenseEmbedder()
            self.sparse_embedder = FakeSparseEmbedder()
            self.qdrant_store = FakeQdrantStore()
            self.reranker = FakeReranker()

    engine_module.OrchestratorEngine = FakeOrchestratorEngine
    monkeypatch.setitem(sys.modules, "docai.orchestrator.engine", engine_module)

    pipeline_module = types.ModuleType("docai.ingestion.pipeline")

    class FakeIngestionPipeline:
        def __init__(self):
            pipeline_events.append("created")

    pipeline_module.IngestionPipeline = FakeIngestionPipeline
    monkeypatch.setitem(sys.modules, "docai.ingestion.pipeline", pipeline_module)

    for module_name, function_names in {
        "docai.mcp.tools.graph": ["graph_query", "entity_link", "check_contradiction"],
        "docai.mcp.tools.validation": ["verify_fact", "ontology_check"],
        "docai.mcp.tools.write": ["graph_write", "quarantine_write", "audit_log"],
    }.items():
        module = types.ModuleType(module_name)
        for function_name in function_names:
            setattr(module, function_name, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, module)

    sys.modules.pop("docai.mcp.server", None)
    return importlib.import_module("docai.mcp.server")


def test_mcp_server_import_does_not_load_heavy_models(monkeypatch) -> None:
    engine_events = []
    pipeline_events = []

    _reload_mcp_server(monkeypatch, engine_events, pipeline_events)

    assert engine_events == []
    assert pipeline_events == []


@pytest.mark.asyncio
async def test_search_documents_lazy_loads_engine(monkeypatch) -> None:
    engine_events = []
    server = _reload_mcp_server(monkeypatch, engine_events)

    response = await server.search_documents("revenue", limit=1)

    assert engine_events == ["created"]
    assert json.loads(response) == {
        "results": [
            {"doc_id": "doc-1", "text": "Revenue grew.", "score": 0.99}
        ]
    }


def test_main_passes_host_and_port_to_fastmcp_run(monkeypatch) -> None:
    server = _reload_mcp_server(monkeypatch)
    monkeypatch.setattr(server.settings, "MCP_HOST", "127.0.0.1")
    monkeypatch.setattr(server.settings, "MCP_PORT", 9876)

    server.main()

    assert server.mcp.run_calls == [("sse", {"host": "127.0.0.1", "port": 9876})]
