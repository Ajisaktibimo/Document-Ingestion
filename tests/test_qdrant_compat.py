import importlib
import sys
import types

import pytest


class FakeDistance:
    COSINE = "Cosine"


class FakeVectorParams:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance


class FakeSparseVectorParams:
    pass


class FakeSparseVector:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class FakePrefetch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFusionQuery:
    def __init__(self, fusion):
        self.fusion = fusion


class FakeFusion:
    RRF = "rrf"


class FakeMatchAny:
    def __init__(self, any):
        self.any = any


class FakeFieldCondition:
    def __init__(self, key, match):
        self.key = key
        self.match = match


class FakeFilter:
    def __init__(self, must=None, should=None, must_not=None, min_should=None):
        self.must = must or []
        self.should = should
        self.must_not = must_not
        self.min_should = min_should


class FakeModels(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            Distance=FakeDistance,
            VectorParams=FakeVectorParams,
            SparseVectorParams=FakeSparseVectorParams,
            SparseVector=FakeSparseVector,
            Prefetch=FakePrefetch,
            FusionQuery=FakeFusionQuery,
            Fusion=FakeFusion,
            MatchAny=FakeMatchAny,
            FieldCondition=FakeFieldCondition,
            Filter=FakeFilter,
        )


def _install_qdrant_stubs(monkeypatch, fake_client_cls):
    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = fake_client_cls
    qdrant_http = types.ModuleType("qdrant_client.http")
    qdrant_http.models = FakeModels()

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", qdrant_http)


@pytest.mark.asyncio
async def test_init_db_creates_document_chunks_with_named_dense_vector(monkeypatch):
    created_collections = {}

    class FakeAsyncQdrantClient:
        def __init__(self, url):
            self.url = url

        async def collection_exists(self, collection_name):
            return False

        async def create_collection(self, collection_name, **kwargs):
            created_collections[collection_name] = kwargs

    _install_qdrant_stubs(monkeypatch, FakeAsyncQdrantClient)
    sys.modules.pop("scripts.init_db", None)
    init_db = importlib.import_module("scripts.init_db")

    await init_db.init_qdrant()

    vectors_config = created_collections["document_chunks"]["vectors_config"]
    assert set(vectors_config) == {"dense"}
    assert vectors_config["dense"].size == 768
    assert set(created_collections["document_chunks"]["sparse_vectors_config"]) == {"sparse"}


@pytest.mark.asyncio
async def test_manual_rrf_uses_query_points_api(monkeypatch):
    class FakeAsyncQdrantClient:
        pass

    _install_qdrant_stubs(monkeypatch, FakeAsyncQdrantClient)
    sys.modules.pop("docai.retrieval.qdrant_client", None)
    qdrant_client = importlib.import_module("docai.retrieval.qdrant_client")

    class FakePoint:
        def __init__(self, point_id, score, text):
            self.id = point_id
            self.score = score
            self.payload = {"doc_id": f"doc-{point_id}", "text": text}

    class FakeQueryResponse:
        def __init__(self, points):
            self.points = points

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def query_points(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["using"] == "dense":
                return FakeQueryResponse(
                    [FakePoint("a", 0.8, "dense first"), FakePoint("b", 0.7, "dense second")]
                )

            return FakeQueryResponse(
                [FakePoint("b", 0.9, "sparse first"), FakePoint("a", 0.6, "sparse second")]
            )

    store = qdrant_client.QdrantStore.__new__(qdrant_client.QdrantStore)
    store.collection_name = "document_chunks"
    store.client = FakeClient()

    results = await store._manual_rrf([0.1, 0.2], {"indices": [1], "values": [0.5]}, limit=1)

    assert [call["using"] for call in store.client.calls] == ["dense", "sparse"]
    assert results[0]["parent_id"] == "a"


@pytest.mark.asyncio
async def test_search_hybrid_preserves_enriched_payload_fields(monkeypatch):
    class FakeAsyncQdrantClient:
        pass

    _install_qdrant_stubs(monkeypatch, FakeAsyncQdrantClient)
    sys.modules.pop("docai.retrieval.qdrant_client", None)
    qdrant_client = importlib.import_module("docai.retrieval.qdrant_client")

    class FakePoint:
        id = "point-a"
        score = 0.8
        payload = {
            "doc_id": "doc-a",
            "text": "chunk text",
            "page_num": 3,
            "heading": "Details",
            "heading_path": "Root > Details",
        }

    class FakeQueryResponse:
        points = [FakePoint()]

    class FakeClient:
        async def query_points(self, **kwargs):
            return FakeQueryResponse()

    store = qdrant_client.QdrantStore.__new__(qdrant_client.QdrantStore)
    store.collection_name = "document_chunks"
    store.client = FakeClient()

    results = await store.search_hybrid(
        [0.1, 0.2],
        {"indices": [1], "values": [0.5]},
        limit=1,
    )

    assert results[0]["page_num"] == 3
    assert results[0]["heading"] == "Details"
    assert results[0]["heading_path"] == "Root > Details"


@pytest.mark.asyncio
async def test_search_hybrid_filters_to_scoped_doc_ids(monkeypatch):
    class FakeAsyncQdrantClient:
        pass

    _install_qdrant_stubs(monkeypatch, FakeAsyncQdrantClient)
    sys.modules.pop("docai.retrieval.qdrant_client", None)
    qdrant_client = importlib.import_module("docai.retrieval.qdrant_client")

    class FakeQueryResponse:
        points = []

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return FakeQueryResponse()

    store = qdrant_client.QdrantStore.__new__(qdrant_client.QdrantStore)
    store.collection_name = "document_chunks"
    store.client = FakeClient()

    await store.search_hybrid(
        [0.1, 0.2],
        {"indices": [1], "values": [0.5]},
        limit=1,
        doc_ids=["doc-a", "doc-b"],
    )

    call = store.client.calls[0]
    assert call["query_filter"].must[0].key == "doc_id"
    assert call["query_filter"].must[0].match.any == ["doc-a", "doc-b"]
    assert [prefetch.kwargs["filter"].must[0].match.any for prefetch in call["prefetch"]] == [
        ["doc-a", "doc-b"],
        ["doc-a", "doc-b"],
    ]


@pytest.mark.asyncio
async def test_init_collection_uses_settings_embedding_dim(monkeypatch):
    """init_collection must use settings.EMBEDDING_DIM, not a hardcoded 768."""
    import asyncio
    from docai.config import settings

    original = settings.EMBEDDING_DIM
    settings.EMBEDDING_DIM = 512

    captured = {}

    class _FakeModels:
        class Distance:
            COSINE = "Cosine"
        @staticmethod
        def VectorParams(size, distance):
            captured["size"] = size
            return {"size": size, "distance": distance}
        @staticmethod
        def SparseVectorParams(**kw):
            return kw

    class _FakeQdrantClient:
        def __init__(self, *a, **kw): pass
        async def collection_exists(self, name): return False
        async def create_collection(self, **kw): pass

    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = _FakeQdrantClient
    qdrant_http = types.ModuleType("qdrant_client.http")
    qdrant_http.models = _FakeModels()

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", qdrant_http)

    try:
        sys.modules.pop("docai.retrieval.qdrant_client", None)
        mod = importlib.import_module("docai.retrieval.qdrant_client")
        store = mod.QdrantStore()
        await store.init_collection()
        assert captured.get("size") == 512, (
            f"Expected EMBEDDING_DIM=512 but VectorParams got size={captured.get('size')}"
        )
    finally:
        settings.EMBEDDING_DIM = original
