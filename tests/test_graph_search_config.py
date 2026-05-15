import pytest
import sys
import types


def _stub_falkordb(monkeypatch):
    falkordb = types.ModuleType("falkordb")
    falkordb.FalkorDB = object
    monkeypatch.setitem(sys.modules, "falkordb", falkordb)


def _stub_qdrant(monkeypatch):
    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = object
    qdrant_http = types.ModuleType("qdrant_client.http")
    qdrant_http.models = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", qdrant_http)


@pytest.mark.asyncio
async def test_gather_unified_context_skips_graph_search_when_disabled(monkeypatch):
    _stub_falkordb(monkeypatch)
    _stub_qdrant(monkeypatch)
    from docai.orchestrator.engine import OrchestratorEngine, settings

    engine = OrchestratorEngine.__new__(OrchestratorEngine)
    calls = {"vector": 0, "graph": 0}

    async def fake_vector_context(query, doc_ids=None):
        calls["vector"] += 1
        return "vector context", []

    async def fake_graph_context(query):
        calls["graph"] += 1
        return "graph context"

    monkeypatch.setattr(settings, "ENABLE_GRAPH_SEARCH", False)
    monkeypatch.setattr(engine, "_get_vector_context", fake_vector_context)
    monkeypatch.setattr(engine, "_get_graph_context_with_timeout", fake_graph_context)

    context = await engine.gather_unified_context("hello")

    assert context["vector"] == "vector context"
    assert context["graph"] == "Graph search disabled."
    assert calls == {"vector": 1, "graph": 0}


@pytest.mark.asyncio
async def test_gather_unified_context_uses_raw_query_for_graph_search(monkeypatch):
    _stub_falkordb(monkeypatch)
    _stub_qdrant(monkeypatch)
    from docai.orchestrator.engine import OrchestratorEngine, settings

    engine = OrchestratorEngine.__new__(OrchestratorEngine)
    calls = {"vector": [], "graph": []}

    async def fake_vector_context(query, doc_ids=None):
        calls["vector"].append(query)
        return "vector context", []

    async def fake_graph_context(query):
        calls["graph"].append(query)
        return "graph context"

    async def fake_document_inventory(*args, **kwargs):
        return "inventory"

    monkeypatch.setattr(settings, "ENABLE_GRAPH_SEARCH", True)
    monkeypatch.setattr(engine, "_get_vector_context", fake_vector_context)
    monkeypatch.setattr(engine, "_get_graph_context_with_timeout", fake_graph_context)
    monkeypatch.setattr(engine, "_get_document_inventory", fake_document_inventory)

    context = await engine.gather_unified_context(
        "user: prior\nassistant: prior answer\nuser: What is my previous chat?",
        graph_query="What is my previous chat?",
    )

    assert context["vector"] == "vector context"
    assert context["graph"] == "graph context"
    assert calls == {
        "vector": ["user: prior\nassistant: prior answer\nuser: What is my previous chat?"],
        "graph": ["What is my previous chat?"],
    }


@pytest.mark.asyncio
async def test_graph_context_skips_invalid_generated_cypher(monkeypatch):
    _stub_falkordb(monkeypatch)
    _stub_qdrant(monkeypatch)
    from docai.orchestrator.engine import OrchestratorEngine

    class FakeLlm:
        async def generate_response(self, **kwargs):
            return "# What is my previous chat?"

    engine = OrchestratorEngine.__new__(OrchestratorEngine)
    engine.llm_client = FakeLlm()

    def fail_if_graph_executes():
        raise AssertionError("invalid Cypher should not be executed")

    monkeypatch.setattr(engine, "_get_falkordb_connection", fail_if_graph_executes)

    result = await engine._get_graph_context("What is my previous chat?")

    assert result == "Graph search skipped: generated query was not valid read-only Cypher."


@pytest.mark.asyncio
async def test_handle_query_uses_recent_session_history_without_followup_keywords(monkeypatch):
    _stub_falkordb(monkeypatch)
    _stub_qdrant(monkeypatch)
    from docai.orchestrator.engine import OrchestratorEngine

    class FakeLlm:
        def __init__(self):
            self.calls = 0

        async def generate_response(self, **kwargs):
            self.calls += 1
            return f"assistant response {self.calls}"

    engine = OrchestratorEngine.__new__(OrchestratorEngine)
    engine.llm_client = FakeLlm()
    calls = []

    async def fake_gather_unified_context(query, doc_ids=None, graph_query=None):
        calls.append((query, doc_ids))
        return {
            "document_inventory": "inventory",
            "vector": "[aaaa] Source: Doc 11111111\nContent: scoped",
            "graph": "Graph search disabled.",
            "citations": [
                {
                    "id": "aaaa",
                    "doc_id": "11111111-1111-1111-1111-111111111111",
                    "page_num": 1,
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(engine, "gather_unified_context", fake_gather_unified_context)

    # Pre-flight (Task T2) is invoked when explicit doc_ids are passed; stub it
    # to mark the requested doc as completed so handle_query proceeds normally.
    from docai.orchestrator.engine import DocStatus
    import uuid as _uuid

    async def fake_fetch_doc_statuses(doc_ids):
        return [
            DocStatus(doc_id=d, filename=f"{str(d)[:8]}.pdf", status="completed", error_message=None)
            for d in doc_ids
        ]

    monkeypatch.setattr(engine, "_fetch_doc_statuses", fake_fetch_doc_statuses)

    await engine.handle_query(
        "S&P 500 underperformance",
        session_id="session-1",
        doc_ids=["11111111-1111-1111-1111-111111111111"],
    )
    await engine.handle_query("summarize risks", session_id="session-1")

    assert calls == [
        ("S&P 500 underperformance", ["11111111-1111-1111-1111-111111111111"]),
        (
            "user: S&P 500 underperformance\n"
            "assistant: assistant response 1\n"
            "user: summarize risks",
            ["11111111-1111-1111-1111-111111111111"],
        ),
    ]


@pytest.mark.asyncio
async def test_handle_query_keeps_only_latest_20_session_messages(monkeypatch):
    _stub_falkordb(monkeypatch)
    _stub_qdrant(monkeypatch)
    from docai.orchestrator.engine import OrchestratorEngine

    class FakeLlm:
        async def generate_response(self, **kwargs):
            return "assistant"

    engine = OrchestratorEngine.__new__(OrchestratorEngine)
    engine.llm_client = FakeLlm()

    async def fake_gather_unified_context(query, doc_ids=None, graph_query=None):
        return {
            "document_inventory": "inventory",
            "vector": "[aaaa] Source: Doc 11111111\nContent: scoped",
            "graph": "Graph search disabled.",
            "citations": [
                {
                    "id": "aaaa",
                    "doc_id": "11111111-1111-1111-1111-111111111111",
                    "page_num": 1,
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(engine, "gather_unified_context", fake_gather_unified_context)

    # Pre-flight (Task T2) is invoked when explicit doc_ids are passed; stub it
    # to mark the requested doc as completed so handle_query proceeds normally.
    from docai.orchestrator.engine import DocStatus

    async def fake_fetch_doc_statuses(doc_ids):
        return [
            DocStatus(doc_id=d, filename=f"{str(d)[:8]}.pdf", status="completed", error_message=None)
            for d in doc_ids
        ]

    monkeypatch.setattr(engine, "_fetch_doc_statuses", fake_fetch_doc_statuses)

    for idx in range(12):
        await engine.handle_query(
            f"question {idx}",
            session_id="session-1",
            doc_ids=["11111111-1111-1111-1111-111111111111"],
        )

    history = engine._session_messages["session-1"]

    assert len(history) == 20
    assert history[0].role == "user"
    assert history[0].content == "question 2"
    assert history[-1].role == "assistant"
    assert history[-1].content == "assistant"


@pytest.mark.asyncio
async def test_vector_context_respects_prompt_context_limits(monkeypatch):
    _stub_falkordb(monkeypatch)
    _stub_qdrant(monkeypatch)
    from docai.orchestrator.engine import OrchestratorEngine, settings

    class FakeEmbedder:
        async def embed_query(self, query):
            return [0.1, 0.2]

    class FakeQdrantStore:
        async def search_hybrid(self, query_dense, query_sparse, limit, doc_ids=None):
            return [
                {
                    "doc_id": "11111111-1111-1111-1111-111111111111",
                    "heading": "",
                    "heading_path": "",
                    "page_num": 1,
                    "text": "A" * 200,
                },
                {
                    "doc_id": "22222222-2222-2222-2222-222222222222",
                    "heading": "",
                    "heading_path": "",
                    "page_num": 2,
                    "text": "B" * 200,
                },
            ]

    class FakeReranker:
        async def rerank(self, query, documents, top_k):
            return [(0, 0.9), (1, 0.8)]

    engine = OrchestratorEngine.__new__(OrchestratorEngine)
    engine.dense_embedder = FakeEmbedder()
    engine.sparse_embedder = FakeEmbedder()
    engine.qdrant_store = FakeQdrantStore()
    engine.reranker = FakeReranker()

    monkeypatch.setattr(settings, "VECTOR_PREFETCH_K", 2)
    monkeypatch.setattr(settings, "RERANKER_OUTPUT_K", 2)
    monkeypatch.setattr(settings, "MIN_RELEVANCE_SCORE", 0.1)
    monkeypatch.setattr(settings, "RAG_CONTEXT_CHUNK_MAX_CHARS", 32)
    monkeypatch.setattr(settings, "RAG_CONTEXT_TOTAL_MAX_CHARS", 180)

    context, citations = await engine._get_vector_context("what is inside?")

    assert len(context) <= 180
    assert "A" * 33 not in context
    assert len(citations) == 1
