import importlib
import sys
import types
from pathlib import Path


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_settings_reads_embedding_runtime_config_from_env(monkeypatch) -> None:
    from docai.config import Settings

    monkeypatch.setenv("DENSE_EMBEDDING_BACKEND", "onnx")
    monkeypatch.setenv("DENSE_EMBEDDING_DEVICE", "cuda:0")
    monkeypatch.setenv("DENSE_EMBEDDING_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
    monkeypatch.setenv("DENSE_EMBEDDING_DEVICE_IDS", "0,1")
    monkeypatch.setenv("DENSE_EMBEDDING_CPU_MODEL", "cpu/model")
    monkeypatch.setenv("DENSE_EMBEDDING_GPU_MODEL", "gpu/model")
    monkeypatch.setenv("DENSE_DOC_PREFIX", "doc: ")
    monkeypatch.setenv("DENSE_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("SPARSE_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("SPARSE_EMBEDDING_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
    monkeypatch.setenv("SPARSE_EMBEDDING_DEVICE_IDS", "0,1")
    monkeypatch.setenv("RERANKER_MODEL", "example/reranker")
    monkeypatch.setenv("RERANKER_BACKEND", "onnx")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "128")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "2m")
    monkeypatch.setenv("OLLAMA_THINK", "False")
    monkeypatch.setenv("CHAT_OLLAMA_THINK", "False")
    monkeypatch.setenv("INGESTION_OLLAMA_THINK", "True")
    monkeypatch.setenv("RAG_CONTEXT_CHUNK_MAX_CHARS", "1234")
    monkeypatch.setenv("RAG_CONTEXT_TOTAL_MAX_CHARS", "5678")
    monkeypatch.setenv("GRAPH_SEARCH_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("ENABLE_GRAPH_SEARCH", "False")

    settings = Settings(_env_file=None)

    assert settings.DENSE_EMBEDDING_BACKEND == "onnx"
    assert settings.DENSE_EMBEDDING_DEVICE == "cuda:0"
    assert settings.DENSE_EMBEDDING_PROVIDERS == "CUDAExecutionProvider,CPUExecutionProvider"
    assert settings.DENSE_EMBEDDING_DEVICE_IDS == "0,1"
    assert settings.DENSE_EMBEDDING_CPU_MODEL == "cpu/model"
    assert settings.DENSE_EMBEDDING_GPU_MODEL == "gpu/model"
    assert settings.DENSE_DOC_PREFIX == "doc: "
    assert settings.DENSE_QUERY_PREFIX == "query: "
    assert settings.SPARSE_EMBEDDING_DEVICE == "cuda"
    assert settings.SPARSE_EMBEDDING_PROVIDERS == "CUDAExecutionProvider,CPUExecutionProvider"
    assert settings.SPARSE_EMBEDDING_DEVICE_IDS == "0,1"
    assert settings.RERANKER_MODEL == "example/reranker"
    assert settings.RERANKER_BACKEND == "onnx"
    assert settings.RERANKER_DEVICE == "cpu"
    assert settings.OLLAMA_REQUEST_TIMEOUT_SECONDS == 12.5
    assert settings.OLLAMA_NUM_PREDICT == 128
    assert settings.OLLAMA_NUM_CTX == 2048
    assert settings.OLLAMA_KEEP_ALIVE == "2m"
    assert settings.OLLAMA_THINK is False
    assert settings.CHAT_OLLAMA_THINK is False
    assert settings.effective_chat_ollama_think is False
    assert settings.INGESTION_OLLAMA_THINK is True
    assert settings.RAG_CONTEXT_CHUNK_MAX_CHARS == 1234
    assert settings.RAG_CONTEXT_TOTAL_MAX_CHARS == 5678
    assert settings.GRAPH_SEARCH_TIMEOUT_SECONDS == 3.5
    assert settings.ENABLE_GRAPH_SEARCH is False


def test_settings_reads_graph_search_enabled_from_env(monkeypatch) -> None:
    from docai.config import Settings

    monkeypatch.setenv("ENABLE_GRAPH_SEARCH", "True")

    settings = Settings(_env_file=None)

    assert settings.ENABLE_GRAPH_SEARCH is True


def test_chat_ollama_think_can_be_true_from_env(monkeypatch) -> None:
    from docai.config import Settings

    monkeypatch.setenv("OLLAMA_THINK", "False")
    monkeypatch.setenv("CHAT_OLLAMA_THINK", "True")

    settings = Settings(_env_file=None)

    assert settings.OLLAMA_THINK is False
    assert settings.CHAT_OLLAMA_THINK is True
    assert settings.effective_chat_ollama_think is True


def test_chat_ollama_think_empty_or_comment_falls_back_to_default(monkeypatch) -> None:
    from docai.config import Settings

    monkeypatch.setenv("OLLAMA_THINK", "True")
    monkeypatch.setenv("CHAT_OLLAMA_THINK", "# can be True or False")

    settings = Settings(_env_file=None)

    assert settings.CHAT_OLLAMA_THINK is None
    assert settings.effective_chat_ollama_think is True


def test_checked_in_env_file_loads_settings() -> None:
    from docai.config import Settings

    settings = Settings()

    assert isinstance(settings.OLLAMA_THINK, bool)
    assert isinstance(settings.CHAT_OLLAMA_THINK, bool)
    assert isinstance(settings.INGESTION_OLLAMA_THINK, bool)
    assert settings.CHAT_OLLAMA_THINK is True


def test_dense_embedder_falls_back_to_model_id_when_local_path_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = kwargs

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    dense_local = _reload_module("docai.embedding.dense_local")
    missing_local_model = tmp_path / "missing-model"
    monkeypatch.setattr(dense_local.settings, "DENSE_MODEL_PATH", str(missing_local_model))
    monkeypatch.setattr(dense_local.settings, "EMBEDDING_MODEL", "example/remote-model")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_BACKEND", "torch")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_DEVICE", "cpu")

    dense_local.LocalDenseEmbedder()

    assert calls["model_name_or_path"] == "example/remote-model"
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["backend"] == "torch"
    assert calls["kwargs"]["device"] == "cpu"
    assert dense_local.LocalDenseEmbedder().doc_prefix == dense_local.settings.DENSE_DOC_PREFIX
    assert dense_local.LocalDenseEmbedder().query_prefix == dense_local.settings.DENSE_QUERY_PREFIX


def test_dense_embedder_prefers_existing_local_path(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = kwargs

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    dense_local = _reload_module("docai.embedding.dense_local")
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    monkeypatch.setattr(dense_local.settings, "DENSE_MODEL_PATH", str(local_model))
    monkeypatch.setattr(dense_local.settings, "EMBEDDING_MODEL", "example/remote-model")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_BACKEND", "torch")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_DEVICE", "cuda:0")

    dense_local.LocalDenseEmbedder()

    assert calls["model_name_or_path"] == str(local_model)
    assert calls["kwargs"]["backend"] == "torch"
    assert calls["kwargs"]["device"] == "cuda:0"


def test_dense_embedder_uses_fastembed_for_onnx_cpu_auto_model(monkeypatch) -> None:
    calls = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    dense_local = _reload_module("docai.embedding.dense_local")
    monkeypatch.setattr(dense_local.settings, "DENSE_MODEL_PATH", "")
    monkeypatch.setattr(dense_local.settings, "EMBEDDING_MODEL", "auto")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_BACKEND", "onnx")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(
        dense_local.settings,
        "DENSE_EMBEDDING_CPU_MODEL",
        "nomic-ai/nomic-embed-text-v1.5-Q",
    )

    dense_local.LocalDenseEmbedder()

    assert calls["model_name"] == "nomic-ai/nomic-embed-text-v1.5-Q"
    assert calls["cuda"] is False
    assert calls["providers"] == ["CPUExecutionProvider"]


def test_dense_embedder_auto_model_keeps_nomic_for_torch_gpu(monkeypatch) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = kwargs

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    dense_local = _reload_module("docai.embedding.dense_local")
    monkeypatch.setattr(dense_local.settings, "DENSE_MODEL_PATH", "")
    monkeypatch.setattr(dense_local.settings, "EMBEDDING_MODEL", "auto")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_BACKEND", "torch")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_DEVICE", "cuda:0")
    monkeypatch.setattr(
        dense_local.settings,
        "DENSE_EMBEDDING_GPU_MODEL",
        "nomic-ai/nomic-embed-text-v2-moe",
    )

    dense_local.LocalDenseEmbedder()

    assert calls["model_name_or_path"] == "nomic-ai/nomic-embed-text-v2-moe"
    assert calls["kwargs"]["backend"] == "torch"
    assert calls["kwargs"]["device"] == "cuda:0"


def test_dense_embedder_missing_local_path_with_auto_model_keeps_nomic_for_gpu(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = kwargs

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    dense_local = _reload_module("docai.embedding.dense_local")
    monkeypatch.setattr(dense_local.settings, "DENSE_MODEL_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(dense_local.settings, "EMBEDDING_MODEL", "auto")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_BACKEND", "torch")
    monkeypatch.setattr(dense_local.settings, "DENSE_EMBEDDING_DEVICE", "cuda:0")
    monkeypatch.setattr(
        dense_local.settings,
        "DENSE_EMBEDDING_GPU_MODEL",
        "nomic-ai/nomic-embed-text-v2-moe",
    )

    dense_local.LocalDenseEmbedder()

    assert calls["model_name_or_path"] == "nomic-ai/nomic-embed-text-v2-moe"


def test_sparse_embedder_uses_configured_cache_dir(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeSparseTextEmbedding:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    fastembed = types.ModuleType("fastembed")
    fastembed.SparseTextEmbedding = FakeSparseTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    sparse_local = _reload_module("docai.embedding.sparse_local")
    monkeypatch.setattr(
        sparse_local.settings,
        "SPARSE_CACHE_DIR",
        str(tmp_path / "sparse-cache"),
    )

    sparse_local.LocalSparseEmbedder()

    assert calls["model_name"] == "Qdrant/bm42-all-minilm-l6-v2-attentions"
    assert calls["cache_dir"] == str(tmp_path / "sparse-cache")
    assert calls["cuda"] is False
    assert calls["providers"] == ["CPUExecutionProvider"]


def test_sparse_embedder_uses_configured_gpu_onnx_providers(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeSparseTextEmbedding:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    fastembed = types.ModuleType("fastembed")
    fastembed.SparseTextEmbedding = FakeSparseTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    sparse_local = _reload_module("docai.embedding.sparse_local")
    monkeypatch.setattr(sparse_local.settings, "SPARSE_CACHE_DIR", str(tmp_path / "sparse-cache"))
    monkeypatch.setattr(sparse_local.settings, "SPARSE_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(
        sparse_local.settings,
        "SPARSE_EMBEDDING_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    )
    monkeypatch.setattr(sparse_local.settings, "SPARSE_EMBEDDING_DEVICE_IDS", "0,2")

    sparse_local.LocalSparseEmbedder()

    assert calls["cuda"] is True
    assert calls["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert calls["device_ids"] == [0, 2]


def test_reranker_falls_back_to_model_id_when_local_path_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    class FakeCrossEncoder:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = kwargs

    sentence_transformers = types.ModuleType("sentence_transformers")
    cross_encoder = types.ModuleType("sentence_transformers.cross_encoder")
    cross_encoder.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "sentence_transformers.cross_encoder", cross_encoder)

    reranker_local = _reload_module("docai.embedding.reranker_local")
    monkeypatch.setattr(reranker_local.settings, "RERANKER_MODEL", "example/reranker")
    monkeypatch.setattr(
        reranker_local.settings,
        "RERANKER_MODEL_PATH",
        str(Path(tmp_path) / "missing-reranker"),
    )
    monkeypatch.setattr(reranker_local.settings, "RERANKER_BACKEND", "onnx")
    monkeypatch.setattr(reranker_local.settings, "RERANKER_DEVICE", "cpu")

    reranker_local.LocalReranker()

    assert calls["model_name_or_path"] == "example/reranker"
    assert calls["kwargs"]["backend"] == "onnx"
    assert calls["kwargs"]["device"] == "cpu"
