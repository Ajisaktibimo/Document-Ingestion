from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_startup_recovery_marks_processing_docs_interrupted(monkeypatch) -> None:
    from docai.ingestion import recovery

    returned_rows = [
        {"doc_id": "11111111-1111-1111-1111-111111111111", "filename": "large.pdf"},
    ]
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=returned_rows)
    conn.close = AsyncMock()

    connect_mock = AsyncMock(return_value=conn)
    monkeypatch.setattr(recovery.asyncpg, "connect", connect_mock)

    rows = await recovery.mark_interrupted_ingestions_failed(
        "postgresql+asyncpg://docai:docai@postgres:5432/docai"
    )

    assert rows == returned_rows
    sql = conn.fetch.await_args.args[0]
    assert "UPDATE doc_registry" in sql
    assert "status = 'failed'" in sql
    assert "ingestion_stage = 'interrupted'" in sql
    assert "ingestion_heartbeat_at = NOW()" in sql
    assert "WHERE status = 'processing'" in sql
    assert conn.close.await_count == 1
