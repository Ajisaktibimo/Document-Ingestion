"""Recovery helpers for process-local ingestion tasks."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

INTERRUPTED_INGESTION_MESSAGE = (
    "Ingestion was interrupted by an API process restart before completion. "
    "Please re-upload the document to retry indexing."
)


def _normalise_dsn(dsn: str) -> str:
    return dsn.replace("postgresql+asyncpg://", "postgresql://")


async def mark_interrupted_ingestions_failed(dsn: str) -> list[dict[str, Any]]:
    """Fail processing rows left behind by dead in-process background tasks."""
    conn = await asyncpg.connect(_normalise_dsn(dsn))
    try:
        rows = await conn.fetch(
            """
            UPDATE doc_registry
            SET status = 'failed',
                error_message = $1,
                ingestion_stage = 'interrupted',
                ingestion_heartbeat_at = NOW(),
                updated_at = NOW()
            WHERE status = 'processing'
            RETURNING doc_id, filename
            """,
            INTERRUPTED_INGESTION_MESSAGE,
        )
        interrupted = [dict(row) for row in rows]
        if interrupted:
            logger.warning(
                "Marked %d interrupted ingestion task(s) as failed after startup.",
                len(interrupted),
            )
        return interrupted
    finally:
        await conn.close()
