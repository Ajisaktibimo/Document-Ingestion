from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Profile
    BACKEND_PROFILE: str = "local"

    # Services
    REDIS_URL: str = "redis://localhost:6381"
    QDRANT_URL: str = "http://localhost:6333"
    POSTGRES_DSN: str = "postgresql+asyncpg://docai:docai@localhost:5432/docai"
    FALKORDB_HOST: str = "localhost"
    FALKORDB_PORT: int = 6380

    # LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:0.8b"
    OLLAMA_REQUEST_TIMEOUT_SECONDS: float = 120.0
    OLLAMA_NUM_PREDICT: int = 256
    OLLAMA_NUM_CTX: int = 1024
    OLLAMA_KEEP_ALIVE: str = "5m"
    OLLAMA_THINK: bool = False
    CHAT_OLLAMA_THINK: bool | None = None
    INGESTION_OLLAMA_THINK: bool = True
    VLLM_BASE_URL: str = "http://vllm-node:8000"

    # Hardware / Docker GPU
    DEVICE: str = "cpu"

    # OCR / Document Parser  (paddle | marker | chandra | mineru | docling)
    DOCUMENT_PARSER: str = "docling"
    CHANDRA_OCR_URL: str = "http://chandra-node:8080"
    CHANDRA_METHOD: str = "hf"  # "hf" for local, "vllm" for remote
    CHANDRA_MODEL_PATH: str = "models/chandra-ocr"
    MINERU_OUTPUT_DIR: str = "data/mineru_output"
    MINERU_USE_GPU: bool = True

    # Captioning (local Ollama – reuses the same OLLAMA_MODEL, no extra VRAM)
    ENABLE_TABLE_CAPTIONING: bool = True
    ENABLE_IMAGE_CAPTIONING: bool = True
    CAPTION_VISION_MODEL: str = ""              # If empty, reuses OLLAMA_MODEL (for multimodal)
    MAX_IMAGES_PER_DOC: int = 10
    MAX_TABLE_MARKDOWN_CHARS: int = 3000

    # Retrieval tuning
    VECTOR_PREFETCH_K: int = 20                 # Over-fetch before reranking
    MIN_RELEVANCE_SCORE: float = 0.10           # Reranker threshold
    RAG_CONTEXT_CHUNK_MAX_CHARS: int = 2200
    RAG_CONTEXT_TOTAL_MAX_CHARS: int = 5000

    # Embedding & Models
    MODELS_DIR: str = "models"
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIM: int = 768
    GPU_EMBED_URL: str = "http://embed-node:8001"
    
    # Local paths (if provided, these take precedence over model IDs for "truly local" mode)
    DENSE_MODEL_PATH: str = "" 
    DENSE_CACHE_DIR: str = "/opt/dense_cache"
    DENSE_EMBEDDING_BACKEND: Literal["torch", "onnx", "openvino"] = "onnx"
    DENSE_EMBEDDING_DEVICE: str = "cpu"
    DENSE_EMBEDDING_PROVIDERS: str = "CPUExecutionProvider"
    DENSE_EMBEDDING_DEVICE_IDS: str = ""
    DENSE_EMBEDDING_CPU_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    DENSE_EMBEDDING_GPU_MODEL: str = "nomic-ai/nomic-embed-text-v2-moe"
    DENSE_DOC_PREFIX: str = "search_document: "
    DENSE_QUERY_PREFIX: str = "search_query: "
    SPARSE_MODEL_PATH: str = ""
    SPARSE_CACHE_DIR: str = "/opt/sparse_cache"
    SPARSE_EMBEDDING_DEVICE: str = "cpu"
    SPARSE_EMBEDDING_PROVIDERS: str = "CPUExecutionProvider"
    SPARSE_EMBEDDING_DEVICE_IDS: str = ""
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_PATH: str = ""
    RERANKER_BACKEND: Literal["torch", "onnx", "openvino"] = "torch"
    RERANKER_DEVICE: str = "cpu"

    # Retrieval constants
    CHUNK_PARENT_TOKENS: int = 1500
    CHUNK_CHILD_TOKENS: int = 400
    CHUNK_OVERLAP: float = 0.20
    RRF_K: int = 60
    DENSE_RECALL_K: int = 50
    SPARSE_RECALL_K: int = 50
    RRF_MERGED_K: int = 20
    RERANKER_OUTPUT_K: int = 5
    SCORE_ALPHA: float = 0.40
    SCORE_BETA: float = 0.30
    SCORE_GAMMA: float = 0.20
    SCORE_DELTA: float = 0.10
    DECAY_LAMBDA: float = 0.01
    DEDUP_THRESHOLD: float = 0.85
    GRAPH_WRITE_CONF_MIN: float = 0.85
    GRAPH_SEARCH_TIMEOUT_SECONDS: float = 10.0
    # If a doc_registry row sits at status='processing' longer than this
    # (or has no updated_at), a fresh upload will treat it as stale, mark
    # it failed, and re-ingest. Prevents orphan rows from a crashed
    # ingestion permanently bricking re-uploads of the same file_hash.
    STALE_PROCESSING_THRESHOLD_SECONDS: float = 600.0

    # ── Session persistence (sub-project 1) ────────────────────────────
    # Master switch. When False, /api/v1/sessions endpoints return 503
    # and the engine ignores any provided SessionContext.
    ENABLE_SESSION_PERSISTENCE: bool = True
    # Re-summarize older turns into sessions.summary on cadence. Off by
    # default because summarization quality on small models is poor;
    # flip True on bigger hardware (per Tier B/C in README.md).
    ENABLE_SESSION_SUMMARIZATION: bool = False
    # Last N messages always passed verbatim to the LLM.
    SESSION_WINDOW_MESSAGES: int = 20
    # Re-summarize when the unsummarized message count past the window
    # exceeds this many.
    SUMMARIZATION_CADENCE_MESSAGES: int = 10
    # Empty string = reuse OLLAMA_MODEL for the summarizer.
    SUMMARIZATION_OLLAMA_MODEL: str = ""
    SUMMARY_MAX_TOKENS: int = 300
    # Cookie name carrying the per-browser anonymous user UUID.
    SESSION_USER_ID_COOKIE: str = "docai_uid"

    # ── Graph-search Cypher generation (effective when ENABLE_GRAPH_SEARCH=True) ──
    # When True, the Cypher-gen prompt is augmented with the live graph
    # schema (node labels, relationship types, properties per label) plus
    # schema-anchored few-shot examples. Improves accuracy on moderate-sized
    # models. Falls back to the static prompt if the schema query fails.
    ENABLE_SCHEMA_AWARE_CYPHER: bool = True
    # Schema introspection runs at most once per TTL window per process.
    CYPHER_SCHEMA_CACHE_TTL_SECONDS: float = 300.0
    # When True, an invalid Cypher generation is re-prompted to the same
    # LLM with the rejection reason, up to MAX_ATTEMPTS times. Default off
    # because each retry is another LLM round-trip — flip on when you have
    # a bigger model and want max graph-recall.
    ENABLE_CYPHER_SELF_HEALING: bool = False
    CYPHER_SELF_HEAL_MAX_ATTEMPTS: int = 3
    ENABLE_GRAPH_SEARCH: bool = Field(
        default=False,
        validation_alias="ENABLE_GRAPH_SEARCH",
    )

    # Networking
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 8010

    @field_validator("CHAT_OLLAMA_THINK", mode="before")
    @classmethod
    def _blank_or_comment_chat_think_as_default(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.startswith("#"):
                return None
        return value

    @property
    def effective_chat_ollama_think(self) -> bool:
        return self.OLLAMA_THINK if self.CHAT_OLLAMA_THINK is None else self.CHAT_OLLAMA_THINK

settings = Settings()

