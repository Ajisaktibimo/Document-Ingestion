# Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the architecture, security, API contract, frontend build, and test reliability issues found in the May 18, 2026 audit.

**Architecture:** Keep the current FastAPI + React/Vite + Postgres + Qdrant + Ollama shape, but add explicit ownership boundaries, safer ingestion lifecycle states, hardened MCP exposure, and reproducible verification. Prefer incremental changes that preserve the local MVP workflow while making production risks visible and gated by configuration.

**Tech Stack:** Python 3.11+, FastAPI, asyncpg, Qdrant, FalkorDB, Ollama, React/Vite/TypeScript, pytest, npm, Docker Compose.

---

## Runtime Incident Addendum: API OOMKilled and Vite Proxy DNS Failure

**Observed on May 18, 2026:** `docai-frontend` logged `getaddrinfo EAI_AGAIN api` while proxying `/api/v1/sessions`. Docker inspection confirmed `docai-api` exited with code `137` and `OOMKilled=true` at `2026-05-18T02:49:07Z`.

**Root cause:** The frontend proxy target `http://api:8000` disappeared because the backend container was killed during heavy Docling/OCR ingestion. The backend was also constructing a new OCR/captioning/embedding stack per upload request before the background job queue boundary, which can duplicate large model memory under concurrent uploads.

**Plan impact:** Treat this as part of T9-T11. Add lazy ingestion service loading, a configurable ingestion concurrency limit, media-caption caps, API healthcheck/restart behavior, and frontend `depends_on: condition: service_healthy`.

---

## File Structure

Modify:
- `pyproject.toml`: pytest source-path guard and dependency/tooling metadata.
- `.gitignore`: prevent accidental check-in of `models/`, `uploads/`, `frontend/node_modules/`, and egg-info.
- `README.md`: align documented upload/status behavior, production warnings, and verification commands.
- `docker-compose.yml`: restrict exposed ports, pin/parameterize credentials and image tags, harden MCP defaults.
- `Dockerfile`, `frontend/Dockerfile`: pin mutable tool/image references or document digest update workflow.
- `src/docai/config.py`: add security, upload, MCP, and health settings.
- `src/docai/api/identity.py`: sign anonymous identity cookie and set secure cookie options.
- `src/docai/api/schema_bootstrap.py`: add ownership indexes, ingestion job table if using durable jobs, and file-hash uniqueness migration.
- `src/docai/api/routes.py`: enforce document ownership, upload validation, canonical upload response, safe errors.
- `src/docai/api/sessions.py`: keep session ownership, add secure streaming error semantics if needed.
- `src/docai/ingestion/pipeline.py`: accept user ownership, move completed status after Qdrant success, cleanup failures.
- `src/docai/retrieval/qdrant_client.py`: batch upserts/deletes and guard vector operations.
- `src/docai/orchestrator/engine.py`: enforce user-owned retrieval scope, streaming preflight, sanitized logging, executor offloading where needed.
- `src/docai/embedding/sparse_local.py`: run sparse embedding off the event loop.
- `src/docai/embedding/reranker_local.py`: run reranker prediction off the event loop.
- `src/docai/mcp/server.py`: disable dangerous tools by default and avoid public host-port exposure assumptions.
- `src/docai/mcp/tools/graph.py`: validate raw graph queries as read-only.
- `src/docai/mcp/tools/write.py`, `src/docai/ingestion/track_b/graph_writer.py`: validate graph writes and remove unsafe string interpolation.
- `frontend/src/lib/api.ts`: Vite env typing usage.
- `frontend/src/vite-env.d.ts`: create Vite client type reference.
- `frontend/src/types/index.ts`: align Session/Citation types with backend.
- `frontend/src/hooks/useSessions.ts`: accept undefined session IDs safely.
- `frontend/src/hooks/useRAGChatStream.ts`: include credentials and expose errors.
- `frontend/src/components/rag/MarkdownWithCitations.tsx`: parse 4-character citation IDs.
- `frontend/src/components/rag/ChatPanel.tsx`: display stream errors and add accessible names.
- `frontend/src/components/rag/DataPanel.tsx`: add upload constraints and accessible names.
- `frontend/src/pages/SessionPage.tsx`: fix typing, remove invalid props, add responsive layout.
- `frontend/src/pages/SessionsPage.tsx`: type message count and add accessible names.
- `frontend/eslint.config.js`: create ESLint 9 flat config.

Create or replace tests:
- `tests/test_document_ownership.py`
- `tests/test_upload_validation.py`
- `tests/test_upload_status_mapping.py`
- `tests/test_identity_middleware.py`
- `tests/test_mcp_security.py`
- `tests/test_pipeline_consistency.py`
- `tests/test_ingestion_runtime_limits.py`
- `tests/test_streaming_chat.py`
- `tests/test_frontend_contract.py`

Remove from git tracking:
- `src/docai.egg-info/*` after adding `*.egg-info/` to `.gitignore`.

---

## Task Dependency Graph

| Task ID | Name | depends_on | Files | Agent Scope |
|---|---|---|---|---|
| T0 | Repo and test preflight hygiene | [] | `.gitignore`, `pyproject.toml`, tracked egg-info | repo hygiene |
| T1 | Canonical upload contract | [T0] | `routes.py`, `pipeline.py`, upload tests, README | backend API |
| T2 | Document ownership boundary | [T1] | schema, routes, pipeline, engine, ownership tests | backend security |
| T3 | Upload validation and resource limits | [T1] | config, routes, upload validation tests | backend security |
| T4 | Cross-store ingestion consistency | [T1] | pipeline, qdrant client, consistency tests | backend data lifecycle |
| T5 | MCP hardening | [T0] | compose, mcp server/tools, graph writer, tests | MCP/security |
| T6 | Identity cookie hardening | [T0] | config, identity, tests, README | auth/security |
| T7 | Frontend build and API contract alignment | [T1, T2] | frontend TS files, tests | frontend |
| T8 | Streaming and citation UX fixes | [T7] | stream hook, ChatPanel, citations, tests | frontend/backend contract |
| T9 | Async blocking and DB lifecycle | [T2, T4] | engine, sparse/reranker, db access | performance |
| T10 | Health/readiness and safe errors/logging | [T2, T4, T6] | server, routes, engine, tests | ops/security |
| T11 | Deployment and supply-chain hardening | [T5, T6, T10] | compose, Dockerfiles, lock docs | ops |
| T12 | Final verification and documentation | [T0-T11] | README, tests | release readiness |

---

## Task T0: Repo and Test Preflight Hygiene

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Delete from git tracking: `src/docai.egg-info/*`

- [ ] **Step 1: Add a failing guard for local package import**

Add this test to `tests/test_import_source_path.py`:

```python
from pathlib import Path


def test_imports_docai_from_current_workspace():
    import docai

    repo = Path(__file__).resolve().parents[1]
    assert Path(docai.__file__).resolve().is_relative_to(repo / "src")
```

Run: `python -m pytest tests/test_import_source_path.py -q`
Expected before config fix: FAIL if Python imports the sibling `Document ingestion` checkout.

- [ ] **Step 2: Force pytest to use this repo source tree**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Run: `python -m pytest tests/test_import_source_path.py -q`
Expected: PASS.

- [ ] **Step 3: Expand git ignore rules**

Update `.gitignore` to include:

```gitignore
models/
uploads/
frontend/node_modules/
*.egg-info/
src/*.egg-info/
Result/
```

Run: `git status --short --untracked-files=all | findstr /I "node_modules uploads models egg-info"`
Expected: no `frontend/node_modules`, `uploads`, `models`, or egg-info entries.

- [ ] **Step 4: Stop tracking generated egg-info**

Run:

```powershell
git rm -r --cached src/docai.egg-info
```

Expected: files are staged for removal from git only, not deleted from disk.

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml tests/test_import_source_path.py
git add -u src/docai.egg-info
git commit -m "chore: harden repo hygiene and test import path"
```

---

## Task T1: Canonical Upload Contract

**Files:**
- Modify: `src/docai/api/routes.py`
- Modify: `src/docai/ingestion/pipeline.py`
- Modify: `tests/test_upload_status_mapping.py`
- Modify: `tests/test_api_engine_loading.py`
- Modify: `README.md`

**Decision:** Canonical upload behavior is async-first:
- `202 Accepted` for newly queued/started processing.
- `200 OK` for duplicate already-indexed documents.
- `500` only when registration fails immediately.
- Body is frontend-compatible: `{"id", "original_filename", "status", "message"}`.

- [ ] **Step 1: Write failing contract tests**

Replace upload status expectations with:

```python
assert body == {
    "id": str(doc_id),
    "original_filename": "report.pdf",
    "status": "processing",
    "message": "Document accepted for indexing.",
}
assert response.status_code == 202
```

Add duplicate indexed case:

```python
assert body["status"] == "indexed"
assert response.status_code == 200
```

Run: `python -m pytest tests/test_upload_status_mapping.py -q`
Expected: FAIL against current `routes.py`.

- [ ] **Step 2: Return registration metadata from the pipeline**

In `src/docai/ingestion/pipeline.py`, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionRegistration:
    doc_id: uuid.UUID
    already_done: bool
```

Change `ingest_document()` to return `Optional[IngestionRegistration]` instead of only `uuid.UUID`.

```python
return IngestionRegistration(doc_id=doc_id, already_done=already_done)
```

- [ ] **Step 3: Map upload response status codes**

In `src/docai/api/routes.py`, use `JSONResponse`:

```python
registration = await pipeline.ingest_document(
    temp_file_path,
    display_filename=display_filename,
    user_id=user_id,
)
if not registration:
    raise HTTPException(status_code=500, detail="Document registration failed.")

body = {
    "id": str(registration.doc_id),
    "original_filename": display_filename,
    "status": "indexed" if registration.already_done else "processing",
    "message": (
        "Document already indexed."
        if registration.already_done
        else "Document accepted for indexing."
    ),
}
return JSONResponse(
    status_code=200 if registration.already_done else 202,
    content=body,
)
```

- [ ] **Step 4: Update README upload docs**

Replace the old `ready/doc_id/filename` table with the canonical body above.

- [ ] **Step 5: Verify**

Run:

```bash
python -m pytest tests/test_upload_status_mapping.py tests/test_api_engine_loading.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/docai/api/routes.py src/docai/ingestion/pipeline.py tests/test_upload_status_mapping.py tests/test_api_engine_loading.py README.md
git commit -m "fix: align upload response contract"
```

---

## Task T2: Document Ownership Boundary

**Files:**
- Modify: `src/docai/api/schema_bootstrap.py`
- Modify: `src/docai/ingestion/pipeline.py`
- Modify: `src/docai/api/routes.py`
- Modify: `src/docai/orchestrator/engine.py`
- Create: `tests/test_document_ownership.py`

- [ ] **Step 1: Write failing ownership tests**

Create tests proving:

```python
async def test_documents_list_filters_by_uploaded_owner():
    # Fake rows for alice and bob.
    # Assert list_documents("alice") only returns alice rows.
```

```python
async def test_delete_document_requires_owner_match():
    # Assert DELETE includes "uploaded_by_user_id = $2" and passes user_id.
```

```python
async def test_unscoped_chat_limits_retrieval_to_user_documents():
    # Engine should fetch user completed doc_ids and pass them to Qdrant.
```

Run: `python -m pytest tests/test_document_ownership.py -q`
Expected: FAIL before implementation.

- [ ] **Step 2: Make schema ownership queryable**

Add indexes in `schema_bootstrap.py`:

```python
_INDEX_STATEMENTS.extend([
    "CREATE INDEX IF NOT EXISTS doc_registry_uploaded_by_idx ON doc_registry(uploaded_by_user_id, created_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS doc_registry_owner_file_hash_uidx ON doc_registry(file_hash, uploaded_by_user_id)",
])
```

Add a migration block to drop the old global file-hash unique constraint when present:

```sql
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'doc_registry_file_hash_key'
  ) THEN
    ALTER TABLE doc_registry DROP CONSTRAINT doc_registry_file_hash_key;
  END IF;
END $$
```

- [ ] **Step 3: Store uploader in the registry**

Change `IngestionPipeline.ingest_document()`:

```python
async def ingest_document(
    self,
    file_path: str,
    doc_class: str = "general",
    display_filename: Optional[str] = None,
    user_id: str | None = None,
) -> Optional[IngestionRegistration]:
```

Change `_register()` signature and queries:

```python
existing = await conn.fetchrow(
    """
    SELECT doc_id, status, page_count, filename, updated_at
    FROM doc_registry
    WHERE file_hash = $1 AND uploaded_by_user_id IS NOT DISTINCT FROM $2
    """,
    file_hash,
    user_id,
)
```

Insert:

```sql
INSERT INTO doc_registry
  (doc_id, file_hash, filename, doc_class, status, error_message, updated_at, uploaded_by_user_id)
VALUES ($1, $2, $3, $4, 'processing', NULL, NOW(), $5)
```

- [ ] **Step 4: Filter document endpoints**

In `routes.py`, add ownership filters:

```sql
FROM doc_registry
WHERE uploaded_by_user_id = $1
ORDER BY created_at DESC
```

For markdown:

```sql
SELECT p.full_text
FROM parent_chunks p
JOIN doc_registry d ON d.doc_id = p.doc_id
WHERE p.doc_id = $1 AND d.uploaded_by_user_id = $2
ORDER BY p.parent_id
```

For delete:

```sql
DELETE FROM parent_chunks
WHERE doc_id = $1
  AND EXISTS (
    SELECT 1 FROM doc_registry
    WHERE doc_id = $1 AND uploaded_by_user_id = $2
  )
```

- [ ] **Step 5: Enforce user-owned retrieval**

In `engine.py`, add:

```python
async def _fetch_user_completed_doc_ids(self, user_id: str) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT doc_id
        FROM doc_registry
        WHERE uploaded_by_user_id = $1 AND status = 'completed'
        """,
        user_id,
    )
    return [str(r["doc_id"]) for r in rows]
```

Pass `user_id` through persistent `SessionContext` and stateless `/chat`; when no explicit `doc_ids` exist, fetch the current user's completed docs and use them as Qdrant filters.

- [ ] **Step 6: Verify**

Run:

```bash
python -m pytest tests/test_document_ownership.py tests/test_sessions_crud.py tests/test_streaming_chat.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/docai/api/schema_bootstrap.py src/docai/ingestion/pipeline.py src/docai/api/routes.py src/docai/orchestrator/engine.py tests/test_document_ownership.py
git commit -m "fix: enforce document ownership boundaries"
```

---

## Task T3: Upload Validation and Resource Limits

**Files:**
- Modify: `src/docai/config.py`
- Modify: `src/docai/api/routes.py`
- Create: `tests/test_upload_validation.py`

- [ ] **Step 1: Add failing validation tests**

Test cases:

```python
def test_rejects_file_larger_than_configured_limit():
    assert response.status_code == 413
```

```python
def test_rejects_unsupported_extension():
    assert response.status_code == 415
```

```python
def test_rejects_pdf_extension_without_pdf_magic_bytes():
    assert response.status_code == 415
```

Run: `python -m pytest tests/test_upload_validation.py -q`
Expected: FAIL.

- [ ] **Step 2: Add upload settings**

In `config.py`:

```python
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS: str = ".pdf,.docx,.txt"
DELETE_UPLOAD_AFTER_INGEST: bool = True
```

- [ ] **Step 3: Replace unbounded copy**

Add helper in `routes.py`:

```python
def _allowed_extensions() -> set[str]:
    return {
        item.strip().lower()
        for item in settings.ALLOWED_UPLOAD_EXTENSIONS.split(",")
        if item.strip()
    }


def _validate_magic(extension: str, first_bytes: bytes) -> bool:
    if extension == ".pdf":
        return first_bytes.startswith(b"%PDF-")
    if extension == ".docx":
        return first_bytes.startswith(b"PK\x03\x04")
    if extension == ".txt":
        return True
    return False
```

Copy in chunks and enforce max:

```python
written = 0
first = b""
with open(temp_file_path, "wb") as buffer:
    while chunk := await file.read(1024 * 1024):
        if not first:
            first = chunk[:16]
            if not _validate_magic(file_extension, first):
                raise HTTPException(status_code=415, detail="Unsupported file content.")
        written += len(chunk)
        if written > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file is too large.")
        buffer.write(chunk)
```

- [ ] **Step 4: Verify**

Run:

```bash
python -m pytest tests/test_upload_validation.py tests/test_upload_status_mapping.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/docai/config.py src/docai/api/routes.py tests/test_upload_validation.py
git commit -m "fix: validate uploads and enforce size limits"
```

---

## Task T4: Cross-Store Ingestion Consistency

**Files:**
- Modify: `src/docai/ingestion/pipeline.py`
- Modify: `src/docai/retrieval/qdrant_client.py`
- Modify: `src/docai/api/routes.py`
- Create: `tests/test_pipeline_consistency.py`

- [ ] **Step 1: Write failing consistency tests**

Add tests:

```python
async def test_document_not_completed_until_qdrant_upsert_succeeds():
    # Fake Qdrant upsert raises.
    # Assert final UPDATE sets status='failed', not 'completed'.
```

```python
async def test_delete_does_not_remove_postgres_when_qdrant_delete_fails():
    # Fake Qdrant delete raises.
    # Assert Postgres DELETE statements are not executed.
```

Run: `python -m pytest tests/test_pipeline_consistency.py -q`
Expected: FAIL.

- [ ] **Step 2: Move completed update after Qdrant success**

In `_run_pipeline()`, insert parent chunks first, upsert Qdrant second, then mark complete:

```python
async with conn.transaction():
    for i, chunk in enumerate(chunks):
        await conn.execute(...)

await self.qdrant_store.insert_chunks(...)

await conn.execute(
    """
    UPDATE doc_registry
    SET status='completed', page_count=$2, error_message=NULL, updated_at=NOW()
    WHERE doc_id=$1
    """,
    doc_id,
    ocr_result.page_count,
)
```

- [ ] **Step 3: Cleanup partial state on failure**

In exception handler:

```python
if conn:
    async with conn.transaction():
        await conn.execute("DELETE FROM parent_chunks WHERE doc_id = $1", doc_id)
        await conn.execute(
            """
            UPDATE doc_registry
            SET status='failed', error_message=$2, updated_at=NOW()
            WHERE file_hash=$1
            """,
            file_hash,
            str(e)[:2000],
        )
try:
    await self.qdrant_store.delete_document(doc_id)
except Exception:
    logger.warning("Best-effort Qdrant cleanup failed for doc_id=%s", doc_id)
```

- [ ] **Step 4: Delete Qdrant before Postgres**

In `delete_document()`:

```python
qdrant = QdrantStore()
await qdrant.delete_document(doc_id)

async with conn.transaction():
    deleted = await conn.fetchrow(
        """
        DELETE FROM doc_registry
        WHERE doc_id = $1 AND uploaded_by_user_id = $2
        RETURNING doc_id
        """,
        doc_id,
        user_id,
    )
    if deleted is None:
        raise HTTPException(status_code=404, detail="Document not found.")
```

Rely on FK cascade if added; otherwise delete `parent_chunks` after ownership check.

- [ ] **Step 5: Verify**

Run:

```bash
python -m pytest tests/test_pipeline_consistency.py tests/test_pipeline_failure.py tests/test_pipeline_stale_processing.py -q
```

Expected: all selected tests pass after updating stale-processing tests for async background semantics.

- [ ] **Step 6: Commit**

```bash
git add src/docai/ingestion/pipeline.py src/docai/retrieval/qdrant_client.py src/docai/api/routes.py tests/test_pipeline_consistency.py tests/test_pipeline_failure.py tests/test_pipeline_stale_processing.py
git commit -m "fix: keep ingestion stores consistent"
```

---

## Task T5: MCP Hardening

**Files:**
- Modify: `docker-compose.yml`
- Modify: `src/docai/config.py`
- Modify: `src/docai/mcp/server.py`
- Modify: `src/docai/mcp/tools/graph.py`
- Modify: `src/docai/mcp/tools/write.py`
- Modify: `src/docai/ingestion/track_b/graph_writer.py`
- Create: `tests/test_mcp_security.py`

- [ ] **Step 1: Write failing MCP security tests**

Tests:

```python
def test_raw_graph_query_rejects_write_cypher():
    assert "error" in await graph_query("MATCH (n) DETACH DELETE n")
```

```python
def test_graph_write_rejects_invalid_predicate():
    assert "error" in await graph_write("a", "BAD PREDICATE; MATCH", "b")
```

Run: `python -m pytest tests/test_mcp_security.py -q`
Expected: FAIL.

- [ ] **Step 2: Disable dangerous tools by default**

In `config.py`:

```python
ENABLE_MCP_INGESTION_TOOL: bool = False
ENABLE_MCP_RAW_GRAPH_QUERY: bool = False
ENABLE_MCP_WRITE_TOOLS: bool = False
```

In `mcp/server.py`, register only when enabled:

```python
if settings.ENABLE_MCP_RAW_GRAPH_QUERY:
    mcp.add_tool(graph_query)
if settings.ENABLE_MCP_WRITE_TOOLS:
    mcp.add_tool(graph_write)
    mcp.add_tool(quarantine_write)
    mcp.add_tool(audit_log)
```

- [ ] **Step 3: Reuse read-only Cypher validation**

In `graph.py`:

```python
from docai.orchestrator.engine import OrchestratorEngine

normalised = OrchestratorEngine._normalise_generated_cypher(cypher_query)
if normalised is None:
    return GraphQueryResponse(error="Only read-only Cypher queries are allowed.").model_dump_json()
```

- [ ] **Step 4: Validate graph writes**

In `graph_writer.py`:

```python
import re

RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

def _clean_entity(value: str) -> str:
    value = " ".join(value.split()).strip()
    if not value or len(value) > 200:
        raise ValueError("Invalid entity value.")
    return value

def _clean_predicate(value: str) -> str:
    pred = value.replace(" ", "_").upper()
    if not RELATION_RE.fullmatch(pred):
        raise ValueError("Invalid predicate.")
    return pred
```

Use Falkor parameters if supported by the client; if not supported, reject single quotes and control characters before formatting.

- [ ] **Step 5: Restrict compose exposure**

Change MCP ports to loopback or remove by default:

```yaml
ports:
  - "127.0.0.1:${MCP_PORT:-8010}:${MCP_PORT:-8010}"
```

Do the same for Redis/Qdrant/Postgres/FalkorDB/Ollama unless explicitly needed from host.

- [ ] **Step 6: Verify**

Run:

```bash
python -m pytest tests/test_mcp_security.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml src/docai/config.py src/docai/mcp/server.py src/docai/mcp/tools/graph.py src/docai/mcp/tools/write.py src/docai/ingestion/track_b/graph_writer.py tests/test_mcp_security.py
git commit -m "fix: harden MCP tool exposure"
```

---

## Task T6: Identity Cookie Hardening

**Files:**
- Modify: `src/docai/config.py`
- Modify: `src/docai/api/identity.py`
- Modify: `tests/test_identity_middleware.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing signed-cookie tests**

Add:

```python
def test_tampered_cookie_is_replaced():
    request.cookies = {"docai_uid": "alice.bad-signature"}
    user_id = get_current_user_id(request, response)
    assert user_id != "alice"
    assert response.set_cookie.called
```

Run: `python -m pytest tests/test_identity_middleware.py -q`
Expected: FAIL.

- [ ] **Step 2: Add cookie settings**

In `config.py`:

```python
SESSION_COOKIE_SECRET: str = ""
SESSION_COOKIE_SECURE: bool = False
SESSION_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
```

- [ ] **Step 3: Sign with stdlib HMAC**

In `identity.py`:

```python
import base64
import hashlib
import hmac


def _sign(value: str) -> str:
    secret = settings.SESSION_COOKIE_SECRET or "local-dev-insecure-secret"
    digest = hmac.new(secret.encode(), value.encode(), hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return f"{value}.{sig}"


def _verify(raw: str | None) -> str | None:
    if not raw or "." not in raw:
        return None
    value, _ = raw.rsplit(".", 1)
    expected = _sign(value)
    if hmac.compare_digest(raw, expected):
        return value
    return None
```

Set cookie:

```python
response.set_cookie(
    key=cookie_name,
    value=_sign(user_id),
    httponly=True,
    secure=settings.SESSION_COOKIE_SECURE,
    samesite=settings.SESSION_COOKIE_SAMESITE,
    path="/",
    max_age=COOKIE_MAX_AGE_SECONDS,
)
```

- [ ] **Step 4: Document limits**

README must state anonymous signed cookies are local/dev identity, not user authentication. Production needs real auth before multi-user deployment.

- [ ] **Step 5: Verify**

Run: `python -m pytest tests/test_identity_middleware.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/docai/config.py src/docai/api/identity.py tests/test_identity_middleware.py README.md
git commit -m "fix: sign anonymous identity cookie"
```

---

## Task T7: Frontend Build and API Contract Alignment

**Files:**
- Create: `frontend/src/vite-env.d.ts`
- Create: `frontend/eslint.config.js`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/hooks/useSessions.ts`
- Modify: `frontend/src/pages/SessionPage.tsx`
- Modify: `frontend/src/pages/SessionsPage.tsx`
- Modify: `frontend/src/components/rag/VisualPanel.tsx`

- [ ] **Step 1: Verify current build fails**

Run:

```bash
cd frontend
npm ci
npm run build
```

Expected before fixes: TypeScript errors for `ImportMeta.env`, `sessionId`, `VisualPanel`, and `message_count`.

- [ ] **Step 2: Add Vite typing**

Create `frontend/src/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />
```

- [ ] **Step 3: Align frontend types**

In `types/index.ts`:

```ts
export interface Session {
  session_id: string;
  title: string;
  doc_ids: string[];
  created_at?: string;
  updated_at: string;
  message_count?: number;
  summary?: string | null;
}

export interface Citation {
  index?: number | string;
  id: string;
  doc_id: string;
  heading?: string;
  heading_path?: string | string[];
  page_num?: number;
  score: number;
}
```

- [ ] **Step 4: Accept nullable session IDs**

In `useSessions.ts`:

```ts
export function useSession(sessionId: string | null | undefined) {
  return useQuery({
    queryKey: ["sessions", sessionId],
    queryFn: () => api.get<Session>(`/sessions/${sessionId}`),
    enabled: Boolean(sessionId),
  });
}
```

- [ ] **Step 5: Remove invalid VisualPanel prop**

In `SessionPage.tsx`, replace:

```tsx
<VisualPanel sessionId={sessionId || ""} />
```

with:

```tsx
<VisualPanel />
```

- [ ] **Step 6: Add ESLint 9 config**

Create `frontend/eslint.config.js`:

```js
import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  }
);
```

- [ ] **Step 7: Verify**

Run:

```bash
cd frontend
npm run build
npm run lint
```

Expected: both exit 0.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/vite-env.d.ts frontend/eslint.config.js frontend/src/lib/api.ts frontend/src/types/index.ts frontend/src/hooks/useSessions.ts frontend/src/pages/SessionPage.tsx frontend/src/pages/SessionsPage.tsx frontend/src/components/rag/VisualPanel.tsx
git commit -m "fix: restore frontend build and lint"
```

---

## Task T8: Streaming Credentials, Citation Links, and Error UX

**Files:**
- Modify: `frontend/src/hooks/useRAGChatStream.ts`
- Modify: `frontend/src/components/rag/MarkdownWithCitations.tsx`
- Modify: `frontend/src/components/rag/ChatPanel.tsx`
- Modify: `src/docai/orchestrator/engine.py`
- Modify: `tests/test_streaming_chat.py`

- [ ] **Step 1: Write frontend contract tests**

In `tests/test_frontend_contract.py`, assert source text includes:

```python
assert 'credentials: "include"' in (FRONTEND_DIR / "src/hooks/useRAGChatStream.ts").read_text()
assert r"/\[((?:[a-zA-Z0-9]{4})|\d+)\]/g" in source_or_equivalent
```

Run: `python -m pytest tests/test_frontend_contract.py -q`
Expected: FAIL before fixes.

- [ ] **Step 2: Send cookies on SSE fetch**

In `useRAGChatStream.ts` fetch options:

```ts
credentials: "include",
```

- [ ] **Step 3: Parse backend citation IDs**

In `MarkdownWithCitations.tsx`:

```ts
const CITATION_RE = /\[([a-zA-Z0-9]{4}|\d+)\]/g;
```

Find citation by `id` first:

```ts
const citation = citations.find((c) => c.id === part || String(c.index) === part);
```

- [ ] **Step 4: Display stream errors**

In `ChatPanel.tsx`, destructure:

```ts
const { stream, isStreaming, status, error, sendMessage, stopStream } = useRAGChatStream(sessionId);
```

Render near the input:

```tsx
{status === "error" && error && (
  <div role="alert" className="mb-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
    {error}
  </div>
)}
```

- [ ] **Step 5: Add streaming doc readiness preflight**

In `engine.handle_query_streaming()`, before `gather_unified_context`, reuse `_fetch_doc_statuses()` for session doc IDs and emit an `error` event when all scoped docs are unavailable.

- [ ] **Step 6: Verify**

Run:

```bash
python -m pytest tests/test_streaming_chat.py tests/test_frontend_contract.py -q
cd frontend && npm run build
```

Expected: selected backend tests and frontend build pass.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/useRAGChatStream.ts frontend/src/components/rag/MarkdownWithCitations.tsx frontend/src/components/rag/ChatPanel.tsx src/docai/orchestrator/engine.py tests/test_streaming_chat.py tests/test_frontend_contract.py
git commit -m "fix: align streaming chat frontend contract"
```

---

## Task T9: Async Blocking and Database Lifecycle

**Files:**
- Modify: `src/docai/embedding/sparse_local.py`
- Modify: `src/docai/embedding/reranker_local.py`
- Modify: `src/docai/api/server.py`
- Modify: `src/docai/api/routes.py`
- Modify: `src/docai/api/sessions.py`
- Modify: `src/docai/orchestrator/engine.py`

- [ ] **Step 1: Add executor tests for embedding/reranker**

Add tests that monkeypatch `run_in_executor` and assert it is called for sparse embedding and rerank.

Run: `python -m pytest tests/test_embedding_startup.py tests/test_engine_reranker_empty.py -q`
Expected: FAIL until offloading is added.

- [ ] **Step 2: Offload sparse embedding**

In `sparse_local.py`:

```python
import asyncio

def _embed_many(self, texts: List[str]):
    return list(self.model.embed(texts))

async def embed_documents(self, texts: List[str]) -> List[Dict[str, List]]:
    loop = asyncio.get_running_loop()
    embeddings = await loop.run_in_executor(None, self._embed_many, texts)
    return [self._convert_to_dict(emb) for emb in embeddings]
```

- [ ] **Step 3: Offload reranking**

In `reranker_local.py`:

```python
import asyncio

async def rerank(self, query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
    if not documents:
        return []
    pairs = [[query, doc] for doc in documents]
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, self.model.predict, pairs)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(idx, float(scores[idx])) for idx in sorted_indices[:top_k]]
```

- [ ] **Step 4: Introduce an asyncpg pool in lifespan**

In `server.py` lifespan:

```python
app.state.pg_pool = await asyncpg.create_pool(normalised_dsn, min_size=1, max_size=10)
...
await app.state.pg_pool.close()
```

Then gradually change route/session helpers from `asyncpg.connect()` to `request.app.state.pg_pool.acquire()`.

- [ ] **Step 5: Verify**

Run:

```bash
python -m pytest tests/test_embedding_startup.py tests/test_engine_reranker_empty.py tests/test_sessions_crud.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/docai/embedding/sparse_local.py src/docai/embedding/reranker_local.py src/docai/api/server.py src/docai/api/routes.py src/docai/api/sessions.py src/docai/orchestrator/engine.py
git commit -m "fix: prevent blocking async request paths"
```

---

## Task T10: Health, Safe Errors, and Logging

**Files:**
- Modify: `src/docai/api/server.py`
- Modify: `src/docai/api/routes.py`
- Modify: `src/docai/api/sessions.py`
- Modify: `src/docai/orchestrator/engine.py`
- Modify: `src/docai/mcp/server.py`
- Create: `tests/test_health_readiness.py`
- Create: `tests/test_safe_errors.py`

- [ ] **Step 1: Add readiness tests**

Test:

```python
def test_liveness_does_not_require_models():
    assert client.get("/health/live").json() == {"status": "ok"}
```

```python
def test_readiness_reports_model_load_failure():
    assert client.get("/health/ready").status_code == 503
```

- [ ] **Step 2: Add readiness state**

In `server.py`:

```python
app.state.engine_ready = False
app.state.engine_error = None
```

When preload succeeds, set ready true. When it fails, store sanitized error type.

Add endpoints:

```python
@app.get("/health/live")
async def live():
    return {"status": "ok"}

@app.get("/health/ready")
async def ready():
    if not app.state.engine_ready:
        raise HTTPException(status_code=503, detail="Engine is not ready.")
    return {"status": "ready"}
```

- [ ] **Step 3: Stop returning raw exception details**

In API handlers, replace `detail=str(e)` with stable messages:

```python
logger.exception("Upload failed")
raise HTTPException(status_code=500, detail="Upload failed.")
```

Keep detailed errors in logs only after redaction.

- [ ] **Step 4: Redact sensitive logs**

Replace:

```python
logger.info(f"Handling query: {query}")
```

with:

```python
logger.info("Handling query chars=%d", len(query or ""))
```

Do not log full generated Cypher unless `DEBUG_LOG_PROMPTS=True`.

- [ ] **Step 5: Verify**

Run:

```bash
python -m pytest tests/test_health_readiness.py tests/test_safe_errors.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/docai/api/server.py src/docai/api/routes.py src/docai/api/sessions.py src/docai/orchestrator/engine.py src/docai/mcp/server.py tests/test_health_readiness.py tests/test_safe_errors.py
git commit -m "fix: add readiness checks and safe error handling"
```

---

## Task T11: Deployment and Supply Chain Hardening

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `frontend/Dockerfile`
- Modify: `requirements.txt` or create `requirements.lock`
- Modify: `README.md`

- [ ] **Step 1: Pin exposed services to localhost**

Change host port mappings:

```yaml
ports:
  - "127.0.0.1:6333:6333"
```

Apply to Redis, Qdrant, Postgres, FalkorDB, Ollama, API, MCP, and frontend for local compose. Document how to expose through Nginx/TLS for production.

- [ ] **Step 2: Replace hardcoded default DB password**

In compose:

```yaml
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}
```

Set `POSTGRES_DSN` from the same variable.

- [ ] **Step 3: Remove shell interpolation from ollama-init**

Replace:

```yaml
entrypoint: ["/bin/sh", "-c", "ollama pull ${OLLAMA_MODEL:-qwen3.5:0.8b}"]
```

with:

```yaml
entrypoint: ["ollama", "pull"]
command: ["${OLLAMA_MODEL:-qwen3.5:0.8b}"]
```

- [ ] **Step 4: Pin images**

Use explicit versions:

```yaml
image: qdrant/qdrant:v1.12.6
image: postgres:15.8-alpine
image: ollama/ollama:0.5.7
```

If exact versions differ in your environment, update them deliberately and record the choice in README.

- [ ] **Step 5: Add Python lock workflow**

Pick one:

```bash
uv pip compile pyproject.toml -o requirements.lock
```

Then Docker installs:

```dockerfile
RUN uv pip install --system --no-cache -r requirements.lock
```

- [ ] **Step 6: Verify**

Run:

```bash
docker compose config
python -m pytest tests/test_dependency_constraints.py -q
```

Expected: compose config renders and dependency tests pass.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml Dockerfile frontend/Dockerfile requirements.lock README.md tests/test_dependency_constraints.py
git commit -m "chore: harden deployment and dependency reproducibility"
```

---

## Task T12: Final Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `architecture.mmd`
- Modify: tests as needed after implementation

- [ ] **Step 1: Update architecture diagram**

Update `architecture.mmd` to show:
- user-owned document scope
- async/durable ingestion lifecycle
- MCP disabled/hardened by default
- readiness endpoints
- restricted host exposure

- [ ] **Step 2: Update README operations**

Document:
- local vs production mode
- upload status contract
- identity limitations
- required env vars
- verification commands
- known non-goals

- [ ] **Step 3: Run full backend verification**

Run:

```bash
$env:PYTHONPATH='src'
python -m pytest tests -q
```

Expected: 0 failures.

- [ ] **Step 4: Run frontend verification**

Run:

```bash
cd frontend
npm ci
npm run build
npm run lint
```

Expected: all commands exit 0.

- [ ] **Step 5: Run dependency audits**

Run:

```bash
cd frontend
npm audit --audit-level=moderate
```

For Python, install/use the chosen audit tool and run:

```bash
pip-audit -r requirements.lock
```

Expected: no critical/high vulnerabilities, or documented exceptions with owner and expiry.

- [ ] **Step 6: Final cleanup**

Run:

```bash
git status --short
```

Expected: only intended source/doc/test changes are shown. No `models/`, `uploads/`, `frontend/node_modules/`, `.env`, or generated caches.

- [ ] **Step 7: Commit**

```bash
git add README.md architecture.mmd
git commit -m "docs: document hardened architecture and operations"
```

---

## Acceptance Criteria

- [ ] Python imports `docai` from this workspace during tests.
- [ ] `python -m pytest tests -q` passes with 0 failures.
- [ ] `frontend npm run build` passes.
- [ ] `frontend npm run lint` passes.
- [ ] Documents are listed, read, deleted, and retrieved only within the current user identity boundary.
- [ ] Uploads enforce size/type limits and use the documented async response contract.
- [ ] Postgres never marks a document completed before Qdrant upsert succeeds.
- [ ] MCP write/raw graph tools are disabled by default and validate inputs when enabled.
- [ ] Anonymous identity cookies are signed and production limitations are documented.
- [ ] Streaming chat sends cookies, surfaces errors, and renders backend citation IDs as clickable links.
- [ ] Health endpoints distinguish liveness from readiness.
- [ ] Compose is local-only by default, avoids default production credentials, and avoids mutable image/tool tags where practical.
- [ ] README and architecture diagram match the implemented system.

---

## Execution Notes

Recommended execution order:
1. T0 first, because current tests can import the wrong sibling checkout.
2. T1 through T4 next, because API/data correctness underpins frontend and security work.
3. T5 and T6 can run after T0 in parallel with T1-T4 if workers own separate files carefully.
4. T7 and T8 after the backend contract is stable.
5. T9 through T12 after the functional/security baseline is green.

Use short commits after each task. Do not batch all fixes into one commit; this plan intentionally separates reviewable risk domains.
