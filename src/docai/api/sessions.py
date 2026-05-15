"""Session-aware REST surface (sub-project 1).

All endpoints inject ``get_current_user_id`` and enforce ownership at
the WHERE-clause level. Mismatched ownership returns 404 (don't leak
existence). Mounted under /api/v1 in routes.py.

This file contains the T5 endpoints (POST /sessions + GET /sessions).
T6 adds GET /sessions/{id} + GET /sessions/{id}/messages.
T7 adds PATCH and DELETE.
T12 adds POST /sessions/{id}/chat.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from docai.api.identity import get_current_user_id
from docai.config import settings
from docai.orchestrator.engine import SessionContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])


def _normalise_dsn() -> str:
    """Convert SQLAlchemy DSN prefix to asyncpg's plain form."""
    return settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")


# ── Request / response models ─────────────────────────────────────

class SessionCreate(BaseModel):
    title: Optional[str] = None
    doc_ids: Optional[List[str]] = None


class SessionChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
async def create_session(
    body: SessionCreate,
    fastapi_response: Response,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    session_id = uuid.uuid4()
    title = body.title or "New chat"
    doc_uuids: List[uuid.UUID] = []
    for raw in body.doc_ids or []:
        try:
            doc_uuids.append(uuid.UUID(raw))
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid doc_id: {raw!r}")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, title, doc_ids,
                                  created_at, updated_at)
            VALUES ($1, $2, $3, $4, NOW(), NOW())
            """,
            session_id, user_id, title, doc_uuids or None,
        )
    finally:
        await conn.close()

    return {
        "session_id": str(session_id),
        "title": title,
        "doc_ids": [str(d) for d in doc_uuids],
    }


@router.get("/sessions")
async def list_sessions(
    user_id: str = Depends(get_current_user_id),
) -> List[Dict[str, Any]]:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT s.session_id, s.title, s.doc_ids, s.updated_at,
                   COALESCE((SELECT COUNT(*)::INT
                             FROM session_messages m
                             WHERE m.session_id = s.session_id), 0) AS message_count
            FROM sessions s
            WHERE s.user_id = $1
            ORDER BY s.updated_at DESC
            """,
            user_id,
        )
    finally:
        await conn.close()

    return [
        {
            "session_id": str(r["session_id"]),
            "title": r["title"],
            "doc_ids": [str(d) for d in (r["doc_ids"] or [])],
            "updated_at": (
                r["updated_at"].isoformat()
                if hasattr(r["updated_at"], "isoformat")
                else r["updated_at"]
            ),
            "message_count": int(r["message_count"]),
        }
        for r in rows
    ]


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        session_row = await conn.fetchrow(
            """
            SELECT session_id, user_id, title, doc_ids, summary,
                   created_at, updated_at
            FROM sessions WHERE session_id = $1 AND user_id = $2
            """,
            session_id, user_id,
        )
        if session_row is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        # Last 50 messages, oldest first.
        msg_rows = await conn.fetch(
            """
            SELECT id, role, content, thinking, citations, created_at
            FROM session_messages
            WHERE session_id = $1
            ORDER BY id DESC
            LIMIT 50
            """,
            session_id,
        )
    finally:
        await conn.close()

    return {
        "session_id": str(session_row["session_id"]),
        "title": session_row["title"],
        "doc_ids": [str(d) for d in (session_row["doc_ids"] or [])],
        "summary": session_row["summary"],
        "created_at": str(session_row["created_at"]),
        "updated_at": str(session_row["updated_at"]),
        "messages": [
            {
                "id": int(r["id"]),
                "role": r["role"],
                "content": r["content"],
                "thinking": r["thinking"],
                "citations": r["citations"],
                "created_at": str(r["created_at"]),
            }
            for r in sorted(msg_rows, key=lambda r: r["id"])  # oldest first for the UI
        ],
    }


@router.get("/sessions/{session_id}/messages")
async def list_session_messages(
    session_id: uuid.UUID,
    limit: int = 50,
    before_id: Optional[int] = None,
    user_id: str = Depends(get_current_user_id),
) -> List[Dict[str, Any]]:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200.")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        # Ownership check first.
        owner = await conn.fetchrow(
            "SELECT 1 FROM sessions WHERE session_id = $1 AND user_id = $2",
            session_id, user_id,
        )
        if owner is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        if before_id is None:
            rows = await conn.fetch(
                """
                SELECT id, role, content, thinking, citations, created_at
                FROM session_messages
                WHERE session_id = $1
                ORDER BY id DESC
                LIMIT $2
                """,
                session_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, role, content, thinking, citations, created_at
                FROM session_messages
                WHERE session_id = $1 AND id < $2
                ORDER BY id DESC
                LIMIT $3
                """,
                session_id, before_id, limit,
            )
    finally:
        await conn.close()

    return [
        {
            "id": int(r["id"]),
            "role": r["role"],
            "content": r["content"],
            "thinking": r["thinking"],
            "citations": r["citations"],
            "created_at": str(r["created_at"]),
        }
        for r in sorted(rows, key=lambda r: r["id"])  # oldest first for the UI
    ]


@router.delete("/sessions/{session_id}/messages", status_code=204)
async def clear_session_messages(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
) -> None:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        # Ownership check first.
        owner = await conn.fetchrow(
            "SELECT 1 FROM sessions WHERE session_id = $1 AND user_id = $2",
            session_id, user_id,
        )
        if owner is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        await conn.execute(
            "DELETE FROM session_messages WHERE session_id = $1",
            session_id,
        )
    finally:
        await conn.close()
    return None


class SessionPatch(BaseModel):
    title: Optional[str] = None
    doc_ids: Optional[List[str]] = None


@router.patch("/sessions/{session_id}")
async def patch_session(
    session_id: uuid.UUID,
    body: SessionPatch,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    if body.title is None and body.doc_ids is None:
        raise HTTPException(status_code=400, detail="At least one of title or doc_ids must be provided.")

    doc_uuids: Optional[List[uuid.UUID]] = None
    if body.doc_ids is not None:
        doc_uuids = []
        for raw in body.doc_ids:
            try:
                doc_uuids.append(uuid.UUID(raw))
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid doc_id: {raw!r}")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        # Build the SET clause dynamically; both columns are independent.
        set_clauses: List[str] = ["updated_at = NOW()"]
        args: List[Any] = []
        if body.title is not None:
            args.append(body.title)
            set_clauses.append(f"title = ${len(args)}")
        if doc_uuids is not None:
            args.append(doc_uuids or None)
            set_clauses.append(f"doc_ids = ${len(args)}")
        # Add session_id and user_id to the WHERE — atomic ownership check + update.
        args.append(session_id)
        sid_idx = len(args)
        args.append(user_id)
        uid_idx = len(args)
        result = await conn.execute(
            f"UPDATE sessions SET {', '.join(set_clauses)} WHERE session_id = ${sid_idx} AND user_id = ${uid_idx}",
            *args,
        )
        # asyncpg returns "UPDATE N" — N=0 means nothing matched (not found or wrong owner).
        if result is None or result.endswith(" 0"):
            raise HTTPException(status_code=404, detail="Session not found.")
    finally:
        await conn.close()

    return {
        "session_id": str(session_id),
        "title": body.title,
        "doc_ids": [str(d) for d in (doc_uuids or [])],
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
) -> None:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        # Atomic: ownership check + delete in one statement via RETURNING.
        # CASCADE deletes session_messages.
        deleted = await conn.fetchrow(
            "DELETE FROM sessions WHERE session_id = $1 AND user_id = $2 RETURNING session_id",
            session_id, user_id,
        )
        if deleted is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    finally:
        await conn.close()
    return None


def _format_sse_event(event_name: str, data) -> bytes:
    """Format an SSE event per the protocol."""
    payload = json.dumps(data, default=str)
    return f"event: {event_name}\ndata: {payload}\n\n".encode("utf-8")


def _get_engine_for_session_chat():
    """Lazy indirection so tests can monkeypatch this without importing
    routes.py (which would create a circular import). Production: returns
    routes.get_engine — the cached async-singleton dependency."""
    from docai.api.routes import get_engine
    return get_engine


@router.post("/sessions/{session_id}/chat")
async def session_chat(
    session_id: uuid.UUID,
    body: SessionChatRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    if not settings.ENABLE_SESSION_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Session persistence is disabled.")

    # Ownership check BEFORE the stream begins — delivers 404 as plain HTTP,
    # not a streamed error.
    dsn = _normalise_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        owner = await conn.fetchrow(
            "SELECT 1 FROM sessions WHERE session_id = $1 AND user_id = $2",
            session_id, user_id,
        )
        if owner is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    finally:
        await conn.close()

    engine_provider = _get_engine_for_session_chat()
    # The provider is either the routes.get_engine async dependency
    # (callable returning a coroutine) OR a test fake that returns the
    # engine directly.
    engine = await engine_provider() if callable(engine_provider) else engine_provider
    ctx = SessionContext(session_id=session_id, user_id=user_id)

    accept = (request.headers.get("accept") or "").lower()
    wants_sse = "text/event-stream" in accept

    if wants_sse:
        async def _sse_generator():
            try:
                async for ev in engine.handle_query_streaming(
                    body.query,
                    ctx,
                    is_disconnected=request.is_disconnected,
                ):
                    yield _format_sse_event(ev["event"], ev["data"])
            except Exception as e:
                logger.exception(
                    "SSE stream error for session %s", session_id
                )
                yield _format_sse_event(
                    "error",
                    {"detail": str(e), "stage": "stream"},
                )

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-SSE: existing dict path, unchanged.
    result = await engine.handle_query(body.query, session_ctx=ctx)
    return {
        "response": result.get("response", ""),
        "thinking": result.get("thinking", ""),
        "citations": result.get("citations", []),
        "guard": result.get("guard"),
    }
