"""Consolidated, idempotent Postgres DDL for the docai project.

Single entry point: ``bootstrap_schema(dsn)``. Called by the FastAPI
lifespan hook on every API container startup AND by the
``scripts/init_db.py`` CLI for manual recovery / CI seeding.

All statements use ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF
NOT EXISTS`` / ``ADD COLUMN IF NOT EXISTS`` so re-runs are race-safe
no-ops. The DSN ``postgresql+asyncpg://`` prefix used by SQLAlchemy is
rewritten to plain ``postgresql://`` for the asyncpg driver.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import asyncpg

logger = logging.getLogger(__name__)


_CREATE_STATEMENTS: List[tuple[str, str]] = [
    (
        "doc_registry",
        """
        CREATE TABLE IF NOT EXISTS doc_registry (
            doc_id      UUID PRIMARY KEY,
            file_hash   TEXT UNIQUE NOT NULL,
            filename    TEXT NOT NULL,
            doc_class   TEXT NOT NULL DEFAULT 'general',
            status      TEXT NOT NULL,
            page_count  INT
        )
        """,
    ),
    (
        "parent_chunks",
        """
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_id   UUID PRIMARY KEY,
            doc_id      UUID NOT NULL,
            heading     TEXT,
            full_text   TEXT NOT NULL,
            page_num    INT
        )
        """,
    ),
    (
        "sessions",
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  UUID PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT NOT NULL DEFAULT 'New chat',
            doc_ids     UUID[],
            summary     TEXT,
            summary_through_message_id BIGINT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ),
    (
        "session_messages",
        """
        CREATE TABLE IF NOT EXISTS session_messages (
            id          BIGSERIAL PRIMARY KEY,
            session_id  UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content     TEXT NOT NULL,
            thinking    TEXT,
            citations   JSONB,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ),
    (
        "cross_references",
        """
        CREATE TABLE IF NOT EXISTS cross_references (
            id            SERIAL PRIMARY KEY,
            source_doc_id UUID NOT NULL,
            target_ref    TEXT NOT NULL,
            page_num      INTEGER,
            created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ),
    (
        "audit_log",
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id     SERIAL PRIMARY KEY,
            event_type TEXT NOT NULL,
            payload    JSONB NOT NULL,
            actor      TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ),
]

_INDEX_STATEMENTS: List[str] = [
    "CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_messages_session_id_idx ON session_messages(session_id, id)",
]

_ALTER_STATEMENTS: List[tuple[str, str]] = [
    ("doc_registry.error_message", "ALTER TABLE doc_registry ADD COLUMN IF NOT EXISTS error_message TEXT"),
    ("doc_registry.updated_at", "ALTER TABLE doc_registry ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ"),
    ("doc_registry.uploaded_by_user_id", "ALTER TABLE doc_registry ADD COLUMN IF NOT EXISTS uploaded_by_user_id TEXT"),
    ("doc_registry.created_at", "ALTER TABLE doc_registry ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"),
    ("doc_registry.ingestion_stage", "ALTER TABLE doc_registry ADD COLUMN IF NOT EXISTS ingestion_stage TEXT"),
    ("doc_registry.ingestion_heartbeat_at", "ALTER TABLE doc_registry ADD COLUMN IF NOT EXISTS ingestion_heartbeat_at TIMESTAMPTZ"),
    ("parent_chunks.section_path", "ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS section_path TEXT"),
    ("parent_chunks.created_at", "ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"),
    # ── sessions migration: pre-T1 init_db.py defined a stub `sessions`
    # table with user_id UUID and only 4 columns. Bring it up to the
    # current schema. All operations are idempotent (ADD COLUMN IF NOT
    # EXISTS) except the type conversion, which is safe to re-run
    # (ALTER ... TYPE TEXT USING ... is no-op if column is already TEXT).
    # Conditional cast: only rewrites the column if it isn't already TEXT.
    # Avoids the ACCESS EXCLUSIVE lock that ALTER ... TYPE takes on no-op
    # paths in older Postgres versions.
    ("sessions.user_id_to_text", """
        DO $$
        BEGIN
          IF (SELECT data_type FROM information_schema.columns
              WHERE table_name='sessions' AND column_name='user_id') <> 'text'
          THEN
            ALTER TABLE sessions ALTER COLUMN user_id TYPE TEXT USING user_id::text;
          END IF;
        END $$
    """),
    ("sessions.title", "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'New chat'"),
    ("sessions.doc_ids", "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS doc_ids UUID[]"),
    ("sessions.summary", "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT"),
    ("sessions.summary_through_message_id", "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_through_message_id BIGINT"),
]


def _normalise_dsn(dsn: str) -> str:
    """Convert SQLAlchemy-style ``postgresql+asyncpg://`` to plain ``postgresql://`` for asyncpg.connect."""
    return dsn.replace("postgresql+asyncpg://", "postgresql://")


async def bootstrap_schema(dsn: str) -> Dict[str, List[str]]:
    """Run all DDL idempotently. Returns a report of what was attempted.

    Raises whatever ``asyncpg.connect`` raises so the FastAPI lifespan
    hook can refuse to start the app on connection failure.
    """
    normalised = _normalise_dsn(dsn)
    # 10s connect timeout so a missing/unreachable Postgres surfaces fast
    # in the lifespan hook rather than hanging the API container forever.
    conn = await asyncpg.connect(normalised, timeout=10)
    tables_present: List[str] = []
    alter_columns_applied: List[str] = []
    try:
        for table_name, ddl in _CREATE_STATEMENTS:
            await conn.execute(ddl)
            tables_present.append(table_name)
        for index_ddl in _INDEX_STATEMENTS:
            await conn.execute(index_ddl)
        for column_label, ddl in _ALTER_STATEMENTS:
            await conn.execute(ddl)
            alter_columns_applied.append(column_label)
    finally:
        await conn.close()

    logger.info(
        "Schema bootstrap complete: %d tables, %d ALTER columns.",
        len(tables_present),
        len(alter_columns_applied),
    )
    return {
        "tables_present": tables_present,
        "alter_columns_applied": alter_columns_applied,
    }
