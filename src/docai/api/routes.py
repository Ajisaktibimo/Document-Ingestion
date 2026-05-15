import asyncio
import logging
import os
import shutil
import uuid
from typing import List, Optional, Dict

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from docai.config import settings
from docai.orchestrator.engine import OrchestratorEngine
from docai.ingestion.pipeline import IngestionPipeline
from docai.api.identity import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter()

# Sessions router has its own prefixes (/sessions, /sessions/{id}, etc.)
# so we include it under the same /api/v1 mount applied by server.py.
#
# Guard: sessions.py imports SessionContext from docai.orchestrator.engine at
# module level. Unit tests that stub the engine module with a minimal fake
# (OrchestratorEngine only, no SessionContext) would fail this import. The
# try/except lets those tests import routes.py cleanly while the real runtime
# (full engine module) always succeeds.
try:
    from docai.api.sessions import router as sessions_router  # noqa: E402
    router.include_router(sessions_router)
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        "Sessions router unavailable: %s. /api/v1/sessions/* endpoints disabled.",
        e,
    )

_engine = None
_engine_lock = None


async def get_engine():
    global _engine, _engine_lock
    if _engine_lock is None:
        _engine_lock = asyncio.Lock()
        
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                _engine = await asyncio.to_thread(OrchestratorEngine)
    return _engine

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    thinking: Optional[str] = None
    citations: Optional[List[Dict]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Basic Chat Endpoint MVP.
    """
    try:
        # 1. Ensure we have a valid session ID
        current_session_id = request.session_id or str(uuid.uuid4())
        
        # 2. Get engine and handle query
        engine = await get_engine()
        result = await engine.handle_query(
            request.query,
            current_session_id,
            doc_ids=request.doc_ids,
        )
        
        # 3. Return guaranteed session_id
        return ChatResponse(
            response=result["response"],
            thinking=result.get("thinking"),
            citations=result.get("citations"),
            session_id=current_session_id
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    user_id: str = Depends(get_current_user_id),
):
    """List all documents in the registry."""
    dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT doc_id, filename, status, page_count, created_at, updated_at,
                   (SELECT COUNT(*) FROM parent_chunks WHERE doc_id = doc_registry.doc_id) as chunk_count
            FROM doc_registry
            ORDER BY created_at DESC
            """
        )
        return [
            {
                "id": str(r["doc_id"]),
                "original_filename": r["filename"],
                "status": r["status"] if r["status"] != "completed" else "indexed",
                "page_count": r["page_count"],
                "chunk_count": r["chunk_count"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ]
    finally:
        await conn.close()

@router.get("/documents/{doc_id}/markdown")
async def get_document_markdown(
    doc_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Reconstruct document markdown from stored chunks."""
    dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT full_text FROM parent_chunks WHERE doc_id = $1 ORDER BY parent_id",
            doc_id,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Document content not found.")
        
        # Simple join. HierarchicalChunker might need more complex reconstruction 
        # if chunks overlap or are out of order, but sequential insert usually works.
        content = "\n\n".join([r["full_text"] for r in rows])
        return content
    finally:
        await conn.close()

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a document and its associated data."""
    dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            # Delete child table first (parent_chunks), then parent (doc_registry)
            await conn.execute("DELETE FROM parent_chunks WHERE doc_id = $1", doc_id)
            await conn.execute("DELETE FROM doc_registry WHERE doc_id = $1", doc_id)

        # Qdrant delete runs after Postgres commits — cannot be rolled back,
        # but at least Postgres is consistent first.
        from docai.retrieval.qdrant_client import QdrantStore
        qdrant = QdrantStore()
        await qdrant.delete_document(doc_id)

        return {"status": "deleted"}
    finally:
        await conn.close()

@router.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Save the file, register in DB, and return immediately with status=processing.
    The heavy OCR/embed work runs as a background task; the frontend polls
    GET /documents every 3 s until status transitions to 'indexed'.
    """
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(upload_dir, f"{uuid.uuid4()}{file_extension}")

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = IngestionPipeline()
        display_filename = os.path.basename(file.filename or temp_file_path)
        doc_id = await pipeline.ingest_document(
            temp_file_path,
            display_filename=display_filename,
        )

        if not doc_id:
            raise HTTPException(status_code=500, detail="Document registration failed.")

        return {
            "id": str(doc_id),
            "original_filename": file.filename,
            "status": "processing",
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

