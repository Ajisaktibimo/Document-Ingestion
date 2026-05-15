# Document Ingestion System

On-premise document-grounded conversational AI platform built with FastAPI, Ollama, Qdrant, PostgreSQL, and optional FalkorDB. Upload documents, ask questions, and get grounded answers with citations.

---

## Architecture

**[Final Architecture Diagram](architecture.mmd)** — single consolidated Mermaid diagram of the current build (v2.3 / 2026-05).

### System Overview

| Tier | Components | Purpose |
|------|------------|---------|
| **Edge** | Nginx + TLS | Reverse proxy, TLS termination |
| **Application** | FastAPI (`:8000`) | Upload, chat, sessions, streaming SSE |
| **MCP** | FastMCP (`:8010`) | Tool registry for document/graph access |
| **AI/Model** | Ollama (`:11434`), Docling/RapidOCR, Embedding Pipeline | LLM, OCR, dense+sparse+reranker embeddings |
| **Data** | Qdrant (`:6333`), PostgreSQL (`:5432`), FalkorDB (`:6380`), Redis (`:6381`) | Vector, relational, graph, cache stores |
| **Frontend** | Vite/React (`:5174`) | Web UI with drag-and-drop, sessions, citations |

### Key Design Decisions

1. **Hybrid Retrieval** — Dense (nomic-embed-text-v2) + Sparse (BM42) → RRF fusion → BGE-reranker-v2-m3
2. **Session Persistence** — Cookie-based identity (`docai_uid`), auto-bootstrap schema, optional summarization
3. **Streaming SSE** — `text/event-stream` with citations, thinking_delta, content_delta, done, error events
4. **Graph Search (Optional)** — FalkorDB with Cypher generation, schema-aware prompts, self-healing retries
5. **Spec Divergences** — Current build uses sync ingestion (not async Celery), Ollama (not vLLM), Docling (not Chandra OCR), direct FalkorDB access (not MCP-only)

### Retrieval Pipeline

```
Stage 1 — Recall (parallel)
├── Dense search: cosine similarity on nomic-embed-text-v2 vectors
└── Sparse search: BM42 lexical match

Stage 2 — Fusion
└── RRF (Reciprocal Rank Fusion) — merge ranked lists, top-K candidates

Stage 3 — Re-rank
└── BGE-reranker-v2-m3 — cross-encoder per-pair relevance score

Stage 4 — Final scoring
└── rerank_score + payload filters (doc_type, language, date range)
```

### MCP Tool Registry

| Tool group | Tools | Access |
|------------|-------|--------|
| Retrieval | `semantic_search`, `get_parent_chunk`, `cross_ref_lookup` | Qdrant, PostgreSQL |
| Graph | `graph_query`, `entity_link`, `check_contradiction` | FalkorDB |
| Validation | `verify_fact`, `ontology_check` | Doc chunks, ontology |
| Write | `graph_write`, `quarantine_write`, `audit_log` | FalkorDB, PostgreSQL |

---

## Quick Start

```bash
# 1. Start all services (schema bootstraps automatically on first start)
docker compose up -d

# 2. Pull the default model into the bundled Ollama container
docker compose exec ollama ollama pull qwen3.5:0.8b

# 3. Open the web UI
open http://localhost:5174
```

Drop a PDF onto the page, wait for the green "Ingested" toast, then ask a question.

---

## Prerequisites

- **Docker Desktop / Docker Engine** with Compose v2
- (Optional) **NVIDIA GPU + nvidia-container-toolkit** for GPU acceleration
- ~12 GB free disk for first run

Ollama is bundled as a Docker service — no host installation required.

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | LLM model to use |
| `OLLAMA_NUM_PREDICT` | `1024` | Output token budget |
| `OLLAMA_NUM_CTX` | `4096` | Context window size |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `1000` | Backend timeout for Ollama |
| `CHAT_OLLAMA_THINK` | `True` | Enable thinking mode for qwen models |
| `DOCUMENT_PARSER` | `docling` | OCR engine (paddle, marker, chandra, mineru, docling) |
| `ENABLE_GRAPH_SEARCH` | `False` | Master switch for FalkorDB graph search |
| `ENABLE_SCHEMA_AWARE_CYPHER` | `True` | Augment Cypher prompt with live schema |
| `ENABLE_CYPHER_SELF_HEALING` | `False` | Retry invalid Cypher generation |
| `CYPHER_SELF_HEAL_MAX_ATTEMPTS` | `3` | Max Cypher generation retries |
| `CYPHER_SCHEMA_CACHE_TTL_SECONDS` | `300` | Schema introspection cache TTL |
| `ENABLE_SESSION_PERSISTENCE` | `True` | Master switch for session persistence |
| `ENABLE_SESSION_SUMMARIZATION` | `False` | Enable session summarization |
| `SESSION_WINDOW_MESSAGES` | `20` | Verbatim turns kept in prompt |
| `SUMMARIZATION_CADENCE_MESSAGES` | `10` | How often to re-summarize |
| `SUMMARY_MAX_TOKENS` | `300` | Summary output token limit |

### Hardware Tier Profiles

**Tier A — Laptop CPU (no GPU)**
```ini
OLLAMA_MODEL=qwen3.5:0.8b
OLLAMA_NUM_PREDICT=1024
CHAT_OLLAMA_THINK=False
ENABLE_GRAPH_SEARCH=False
DENSE_EMBEDDING_DEVICE=cpu
SPARSE_EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

**Tier B — Workstation (single mid-range GPU ≥8GB VRAM)**
```ini
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_NUM_PREDICT=2048
CHAT_OLLAMA_THINK=False
ENABLE_GRAPH_SEARCH=True
ENABLE_SCHEMA_AWARE_CYPHER=True
ENABLE_CYPHER_SELF_HEALING=False
DENSE_EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda
```
Boot with: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`

**Tier C — Server (multi-GPU or ≥24GB VRAM)**
```ini
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_NUM_PREDICT=4096
CHAT_OLLAMA_THINK=True
ENABLE_GRAPH_SEARCH=True
ENABLE_SCHEMA_AWARE_CYPHER=True
ENABLE_CYPHER_SELF_HEALING=True
CYPHER_SELF_HEAL_MAX_ATTEMPTS=3
DENSE_EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda
```

### Graph Search Cypher Flags

| Setting | Default | Purpose |
|---------|---------|---------|
| `ENABLE_GRAPH_SEARCH` | `False` | Master switch for graph-search path |
| `ENABLE_SCHEMA_AWARE_CYPHER` | `True` | Augment prompt with live FalkorDB schema |
| `ENABLE_CYPHER_SELF_HEALING` | `False` | Re-prompt on invalid Cypher (extra LLM calls) |
| `CYPHER_SELF_HEAL_MAX_ATTEMPTS` | `3` | Hard cap on Cypher generation attempts |
| `CYPHER_SCHEMA_CACHE_TTL_SECONDS` | `300` | Schema introspection cache window |

**Recommended profiles:**
- **Small model (≤1B):** `ENABLE_GRAPH_SEARCH=False`
- **Medium model (~7B):** `ENABLE_GRAPH_SEARCH=True`, `ENABLE_SCHEMA_AWARE_CYPHER=True`, `ENABLE_CYPHER_SELF_HEALING=False`
- **Large model (≥14B, GPU):** Add `ENABLE_CYPHER_SELF_HEALING=True`

---

## Operations

### First-Time Setup

```bash
# 1. Start Ollama container
docker compose up -d ollama

# 2. Pull model (default)
docker compose exec ollama ollama pull qwen3.5:0.8b

# Or pull recommended model for production
docker compose exec ollama ollama pull qwen2.5:7b
# Then edit .env: OLLAMA_MODEL=qwen2.5:7b

# 3. Start all services
docker compose up -d

# 4. Initialize database (legacy, auto-bootstrap now handles this)
docker compose exec api python scripts/init_db.py

# 5. Verify health
docker compose ps
curl http://localhost:8000/api/v1/health
curl http://localhost:6333/
docker logs docai-api 2>&1 | tail -30
```

### Document Upload

| Method | Command |
|--------|---------|
| **Web UI** | Drag-and-drop at http://localhost:5174 |
| **CLI** | `docker compose exec api python scripts/ingest_doc.py path/to/file.pdf` |
| **API** | `curl -F "file=@path/to/file.pdf" http://localhost:8000/api/v1/upload` |

**Upload Response Codes:**
| Status | Response | Meaning |
|--------|----------|---------|
| `200` | `{"status": "ready", "doc_id": "...", "filename": "..."}` | Indexing complete |
| `202` | `{"status": "processing", ...}` | Still indexing, wait and retry |
| `500` | `{"detail": "..."}` | Pipeline failed |

### Chat

| Method | Command |
|--------|---------|
| **Web UI** | Type query in chat panel, select document scope |
| **API (JSON)** | `curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d '{"query":"What is X?", "doc_ids":["<uuid>"]}'` |
| **API (Streaming)** | `curl -N -H "Accept: text/event-stream" -X POST http://localhost:8000/api/v1/sessions/$SID/chat -d '{"query":"hello"}'` |

**Streaming Events:**
| Event | Payload | When |
|-------|---------|------|
| `citations` | `[{id, doc_id, heading, page_num, score}, ...]` | After retrieval, before LLM |
| `thinking_delta` | `{text: "..."}` | During chain-of-thought emission |
| `content_delta` | `{text: "..."}` | Per chunk of assistant answer |
| `done` | `{guard, content_chars, thinking_chars}` | Stream complete |
| `error` | `{detail, stage: "retrieval"|"llm"|"persist"}` | Unrecoverable error |

**Streaming notes:**
- User message persisted at stream start (reload-after-error preserves it)
- Assistant message persisted at stream end with fully accumulated content
- Cancellation: closing connection mid-stream stops engine within one chunk
- Reason promotion delivered as final `content_delta` before `done`
- Stateless `POST /api/v1/chat` never streams

### Sessions

Persistent chat sessions per browser with cookie-based identity (`docai_uid`, httpOnly, 1-year expiry).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/sessions` | Create new session |
| GET | `/api/v1/sessions` | List sessions |
| GET | `/api/v1/sessions/{id}` | Get session + last 50 messages |
| GET | `/api/v1/sessions/{id}/messages?limit=50&before_id=N` | Paginate messages |
| PATCH | `/api/v1/sessions/{id}` | Rename or update doc scope |
| DELETE | `/api/v1/sessions/{id}` | Delete session (CASCADE deletes messages) |
| POST | `/api/v1/sessions/{id}/chat` | Send chat turn (persists user + assistant) |

**Session UX Features:**
| Feature | How to use |
|---------|------------|
| Inline rename | Double-click session title in sidebar |
| Avatar initials | 2-letter UID initials on blue background |
| Message animation | 0.18s ease-out slide-in |
| Chat empty state | Call-to-action when no messages |
| Sidebar empty state | Prompt with icon when no sessions |

### Logging & Monitoring

```bash
# Watch live logs
docker logs -f docai-api
docker logs -f docai-api 2>&1 | grep -E "Handling query|Generated Cypher|guard"

# Health checks
docker compose ps
curl http://localhost:8000/api/v1/health
curl http://localhost:6333/
```

### Testing

```bash
docker compose exec api bash -lc "PYTHONPATH=src python -m pytest tests/ -v"
# Expected: 67 passed; 7 errors are pre-existing Windows-tempdir failures
```

### Restart After Config Change

```bash
docker compose restart docai-api docai-mcp
```
No rebuild needed for `.env` changes. Frontend hot-reloads automatically.

---

## Recovery Patterns

### Stuck Indexing
Auto-recovery after 10 minutes via stale-processing detection (`STALE_PROCESSING_THRESHOLD_SECONDS=600`). Just re-upload after 10 minutes.

**Force recovery:**
```bash
docker exec -it docai-postgres psql -U docai -d docai -c \
  "DELETE FROM doc_registry WHERE doc_id = '<doc-uuid>';"
# Then clear Qdrant points for the same doc_id and re-upload
```

### OCR Cache Issues
RapidOCR models cache to `./models/rapidocr_cache/`. Verify:
```bash
ls models/rapidocr_cache/
# Should show: ch_PP-OCRv4_det_mobile.onnx, ch_ppocr_mobile_v2.0_cls_mobile.onnx, ch_PP-OCRv4_rec_mobile.onnx
```

### Frontend Timeout
Either model too slow or Ollama unreachable:
```bash
docker logs docai-api 2>&1 | grep -E "Handling query|Prompting LLM|HTTP Request.*ollama"
```
If generation time > 10 min, increase `CHAT_TIMEOUT_MS` or reduce `OLLAMA_NUM_PREDICT`.

### Volume Reset
```bash
docker compose down -v   # ⚠️ deletes ALL volume data
docker compose up -d
```

---

## File Layout

```
.
├── docker-compose.yml           # 8 services + bind mounts + env wiring
├── docker-compose.gpu.yml       # NVIDIA GPU runtime overlay
├── Dockerfile                   # API + MCP image (PyTorch CUDA base)
├── .env                         # Runtime configuration
├── architecture.mmd             # Current build architecture diagram
├── README-architecture.md       # Spec-level architecture (v2.3 target)
├── scripts/
│   ├── init_db.py               # Postgres schema setup (idempotent)
│   └── ingest_doc.py            # CLI document ingestion
├── src/docai/
│   ├── api/routes.py            # FastAPI endpoints (/upload, /chat, /sessions/*)
│   ├── orchestrator/engine.py   # OrchestratorEngine (chat + retrieval)
│   ├── ingestion/pipeline.py    # IngestionPipeline (sync, stale detection)
│   ├── ocr/                     # Docling, RapidOCR, Paddle, Marker, MinerU
│   ├── embedding/               # Dense, sparse, reranker (local)
│   ├── mcp/server.py            # FastMCP entrypoint
│   └── config.py                # All settings
├── frontend/                    # React/Vite web UI (port 5174)
├── models/                      # Bind-mounted ML caches
└── tests/                       # pytest suite
```

---

## Spec Divergences (v2.3 Target vs Current Build)

| Spec Design | Current Build |
|-------------|---------------|
| Track A async via Celery + Redis | **Synchronous** ingestion with stale-processing detection |
| Track B knowledge extraction (3-agent swarm) | **Not implemented** — FalkorDB stays empty unless populated externally |
| Conversation memory (Redis + Qdrant episodic + composite scoring) | **Not implemented** — session state in-process |
| vLLM + Chandra OCR 2 on GPU node | **Ollama** for LLM, **Docling/RapidOCR** for OCR |
| FalkorDB MCP-only access | **Direct access** from orchestrator |
| Importance/recency/usage scoring | **Not implemented** |

These are deliberate v0.x simplifications, not bugs.

---

## Screenshots

### Welcome Screen
![Welcome Screen](Result/Screenshot%202026-05-15%20215700.png)

### Chat with Document Reader
![Chat with Reader](Result/Screenshot%202026-05-15%20202125.png)

### Document Q&A
![Document Q&A](Result/Screenshot%202026-05-15%202151221.png)

### API — Upload Endpoint
![Upload API](Result/Screenshot%202026-05-15%20214949l.png)

### API — Chat Endpoint Response
![Chat API](Result/Screenshot%202026-05-15%20214750.png)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Upload 500 "Failed to download modelscope" | OCR models not cached + CDN flake | Wait, retry — models cache to `./models/rapidocr_cache/` |
| Upload 202 (not 200) | Same file_hash mid-flight | Wait, retry — doc is still indexing |
| Chat: "None of the requested documents are ready" | All scoped docs processing/failed | Wait for indexing or remove from scope |
| Chat slow but answers | CPU-bound Ollama | Use bigger model, `CHAT_OLLAMA_THINK=False`, reduce `OLLAMA_NUM_PREDICT` |
| `KeyError: 'OLLAMA_MODEL'` at startup | Missing `.env` | Copy `.env` from version control |
| Tests fail with `PermissionError` | Windows tempdir perms | Pre-existing; unrelated to code changes |
| Frontend "Indexing in progress…" forever | Pipeline crashed silently | Check logs, stale-detection allows re-upload after 10 min |
| "Graph search skipped invalid Cypher" | 0.8B model can't emit valid Cypher | Disable graph search or use bigger model |

---

## What's Next

- Original-filename preservation through ingestion
- Schema-with-example-values in Cypher prompt
- Dockerfile bake-time OCR model staging for air-gapped runs
- True async Track A (Celery + Redis)
- Track B knowledge extraction with validation swarm
- Conversation memory tier with composite scoring
