"""Tests for schema-aware Cypher generation and self-healing retry.

Covers ``OrchestratorEngine._fetch_graph_schema``, ``_get_cached_graph_schema``,
``_build_cypher_system_prompt``, and ``_generate_cypher_with_retry`` plus
their interaction with the four new settings:

- ``ENABLE_SCHEMA_AWARE_CYPHER``
- ``CYPHER_SCHEMA_CACHE_TTL_SECONDS``
- ``ENABLE_CYPHER_SELF_HEALING``
- ``CYPHER_SELF_HEAL_MAX_ATTEMPTS``
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_heavy_provider_stubs(monkeypatch):
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


def _make_engine(monkeypatch):
    _install_heavy_provider_stubs(monkeypatch)
    sys.modules.pop("docai.orchestrator.engine", None)
    import importlib
    engine_module = importlib.import_module("docai.orchestrator.engine")
    engine = engine_module.OrchestratorEngine()
    return engine, engine_module


def _make_fake_graph(label_rows, rel_rows, prop_rows):
    """Return a fake ``graph`` whose ``query(cypher).result_set`` dispatches
    to the right rows by inspecting the Cypher string.
    """
    def _query(cypher):
        res = MagicMock()
        if "labels(n)" in cypher and "keys" not in cypher:
            res.result_set = label_rows
        elif "type(r)" in cypher:
            res.result_set = rel_rows
        elif "keys(n)" in cypher:
            res.result_set = prop_rows
        else:
            res.result_set = []
        return res

    graph = MagicMock()
    graph.query = _query
    return graph


def _attach_fake_falkordb(engine, *, label_rows, rel_rows, prop_rows):
    fake_graph = _make_fake_graph(label_rows, rel_rows, prop_rows)
    fake_db = MagicMock()
    fake_db.select_graph = MagicMock(return_value=fake_graph)
    engine._get_falkordb_connection = MagicMock(return_value=fake_db)


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_graph_schema
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_graph_schema_happy_path(monkeypatch) -> None:
    engine, _ = _make_engine(monkeypatch)
    _attach_fake_falkordb(
        engine,
        label_rows=[[["Entity"]], [["Document"]]],
        rel_rows=[["MENTIONED_IN"], ["ASSOCIATED_WITH"]],
        prop_rows=[["Entity", ["name", "type"]], ["Document", ["title"]]],
    )

    schema = await engine._fetch_graph_schema()

    assert schema["node_labels"] == ["Document", "Entity"]
    assert schema["rel_types"] == ["ASSOCIATED_WITH", "MENTIONED_IN"]
    assert schema["props_by_label"] == {
        "Entity": ["name", "type"],
        "Document": ["title"],
    }


@pytest.mark.asyncio
async def test_fetch_graph_schema_returns_empty_on_failure(monkeypatch) -> None:
    engine, _ = _make_engine(monkeypatch)

    def _boom():
        raise ConnectionRefusedError("falkordb down")

    fake_db = MagicMock()
    fake_db.select_graph = MagicMock(side_effect=_boom)
    engine._get_falkordb_connection = MagicMock(return_value=fake_db)

    schema = await engine._fetch_graph_schema()
    assert schema == {}


# ─────────────────────────────────────────────────────────────────────────────
# _get_cached_graph_schema
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cached_schema_hits_within_ttl(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "CYPHER_SCHEMA_CACHE_TTL_SECONDS", 300.0, raising=False
    )

    fetch_calls = []

    async def _fake_fetch():
        fetch_calls.append(1)
        return {"node_labels": ["Entity"], "rel_types": [], "props_by_label": {}}

    engine._fetch_graph_schema = _fake_fetch

    # Pin time so two calls fall inside the TTL window.
    fake_now = [1000.0]
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: fake_now[0])

    a = await engine._get_cached_graph_schema()
    fake_now[0] += 5  # 5 seconds later, well inside 300s TTL
    b = await engine._get_cached_graph_schema()

    assert a == b
    assert len(fetch_calls) == 1, "schema was re-fetched even though TTL had not expired"


@pytest.mark.asyncio
async def test_cached_schema_refreshes_after_ttl(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "CYPHER_SCHEMA_CACHE_TTL_SECONDS", 60.0, raising=False
    )

    fetch_calls = []

    async def _fake_fetch():
        fetch_calls.append(1)
        return {"node_labels": ["Entity"], "rel_types": [], "props_by_label": {}}

    engine._fetch_graph_schema = _fake_fetch

    fake_now = [2000.0]
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: fake_now[0])

    await engine._get_cached_graph_schema()
    fake_now[0] += 120  # 2 minutes later — TTL expired
    await engine._get_cached_graph_schema()

    assert len(fetch_calls) == 2, "schema was NOT re-fetched after TTL expired"


# ─────────────────────────────────────────────────────────────────────────────
# _build_cypher_system_prompt
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt_falls_back_to_static_when_flag_off(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_SCHEMA_AWARE_CYPHER", False, raising=False
    )
    schema = {"node_labels": ["Entity"], "rel_types": ["MENTIONED_IN"], "props_by_label": {}}

    prompt = engine_module.OrchestratorEngine._build_cypher_system_prompt(schema)

    assert prompt == engine_module.OrchestratorEngine._STATIC_CYPHER_PROMPT


def test_prompt_falls_back_to_static_when_schema_empty(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_SCHEMA_AWARE_CYPHER", True, raising=False
    )

    prompt = engine_module.OrchestratorEngine._build_cypher_system_prompt({})

    assert prompt == engine_module.OrchestratorEngine._STATIC_CYPHER_PROMPT


def test_prompt_includes_schema_examples_and_constraints(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_SCHEMA_AWARE_CYPHER", True, raising=False
    )
    schema = {
        "node_labels": ["Entity", "Document"],
        "rel_types": ["MENTIONED_IN", "ASSOCIATED_WITH"],
        "props_by_label": {"Entity": ["name", "type"], "Document": ["title"]},
    }

    prompt = engine_module.OrchestratorEngine._build_cypher_system_prompt(schema)

    # Schema block content
    assert "`Entity`" in prompt and "`Document`" in prompt
    assert "`MENTIONED_IN`" in prompt and "`ASSOCIATED_WITH`" in prompt
    assert "Entity(name, type)" in prompt
    assert "Document(title)" in prompt
    # Examples present and anchored to user's actual primary types
    assert "Examples:" in prompt
    assert "(e:Entity)-[:MENTIONED_IN]->" in prompt
    # Constraint enforcement
    assert "READ-ONLY" in prompt
    assert "CREATE, MERGE, DELETE" in prompt
    assert "no markdown fences" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# _generate_cypher_with_retry
# ─────────────────────────────────────────────────────────────────────────────

def _stub_schema(engine, *, with_schema=True):
    """Replace _get_cached_graph_schema so retry tests don't need FalkorDB."""
    schema = (
        {"node_labels": ["Entity"], "rel_types": ["MENTIONED_IN"], "props_by_label": {"Entity": ["name"]}}
        if with_schema
        else {}
    )

    async def _fake_cached():
        return schema

    engine._get_cached_graph_schema = _fake_cached


@pytest.mark.asyncio
async def test_generate_cypher_returns_valid_on_first_attempt(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_CYPHER_SELF_HEALING", False, raising=False
    )
    _stub_schema(engine)

    engine.llm_client.generate_response = AsyncMock(
        return_value={"content": "MATCH (e:Entity) RETURN e LIMIT 5", "thinking": ""}
    )

    result = await engine._generate_cypher_with_retry("any question")

    assert result == "MATCH (e:Entity) RETURN e LIMIT 5"
    assert engine.llm_client.generate_response.await_count == 1


@pytest.mark.asyncio
async def test_generate_cypher_no_retry_when_self_healing_off(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_CYPHER_SELF_HEALING", False, raising=False
    )
    monkeypatch.setattr(
        engine_module.settings, "CYPHER_SELF_HEAL_MAX_ATTEMPTS", 5, raising=False
    )
    _stub_schema(engine)

    # Returns prose — fails the read-only validator.
    engine.llm_client.generate_response = AsyncMock(
        return_value={"content": "Sure! Here is the query: CREATE (n:Entity)", "thinking": ""}
    )

    result = await engine._generate_cypher_with_retry("any question")

    assert result is None
    assert engine.llm_client.generate_response.await_count == 1, (
        "self-healing was off but the LLM was called more than once"
    )


@pytest.mark.asyncio
async def test_generate_cypher_self_heals_then_succeeds(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_CYPHER_SELF_HEALING", True, raising=False
    )
    monkeypatch.setattr(
        engine_module.settings, "CYPHER_SELF_HEAL_MAX_ATTEMPTS", 3, raising=False
    )
    _stub_schema(engine)

    responses = [
        {"content": "Sure, here you go:", "thinking": ""},                 # invalid (prose)
        {"content": "CREATE (n:Entity)", "thinking": ""},                  # invalid (write op)
        {"content": "MATCH (e:Entity) RETURN e LIMIT 5", "thinking": ""},  # valid
    ]
    engine.llm_client.generate_response = AsyncMock(side_effect=responses)

    result = await engine._generate_cypher_with_retry("any question")

    assert result == "MATCH (e:Entity) RETURN e LIMIT 5"
    assert engine.llm_client.generate_response.await_count == 3

    # Retry prompts should include the previous-attempt feedback so the LLM can self-correct.
    third_call_kwargs = engine.llm_client.generate_response.call_args_list[2].kwargs
    assert "Previous attempt was invalid" in third_call_kwargs["system_prompt"]
    assert "CREATE (n:Entity)" in third_call_kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_generate_cypher_returns_none_when_all_attempts_fail(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_CYPHER_SELF_HEALING", True, raising=False
    )
    monkeypatch.setattr(
        engine_module.settings, "CYPHER_SELF_HEAL_MAX_ATTEMPTS", 3, raising=False
    )
    _stub_schema(engine)

    engine.llm_client.generate_response = AsyncMock(
        return_value={"content": "DROP DATABASE", "thinking": ""}
    )

    result = await engine._generate_cypher_with_retry("any question")

    assert result is None
    assert engine.llm_client.generate_response.await_count == 3


@pytest.mark.asyncio
async def test_generate_cypher_returns_none_when_llm_raises(monkeypatch) -> None:
    engine, engine_module = _make_engine(monkeypatch)
    monkeypatch.setattr(
        engine_module.settings, "ENABLE_CYPHER_SELF_HEALING", True, raising=False
    )
    _stub_schema(engine)

    engine.llm_client.generate_response = AsyncMock(side_effect=RuntimeError("ollama timeout"))

    result = await engine._generate_cypher_with_retry("any question")

    assert result is None
    assert engine.llm_client.generate_response.await_count == 1
