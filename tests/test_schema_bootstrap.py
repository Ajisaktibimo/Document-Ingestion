"""Tests for src/docai/api/schema_bootstrap.py.

The bootstrap function must be idempotent (re-runs are no-ops), report
its actions, and propagate connection errors so the FastAPI lifespan
hook can refuse to start the app on schema failure.
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_asyncpg_stub(monkeypatch, *, conn=None, raise_on_connect=False):
    fake_pg = types.ModuleType("asyncpg")
    if raise_on_connect:
        fake_pg.connect = AsyncMock(side_effect=ConnectionRefusedError("nope"))
    else:
        fake_pg.connect = AsyncMock(return_value=conn)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_pg)


def _make_conn():
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    conn.close = AsyncMock(return_value=None)
    return conn


@pytest.mark.asyncio
async def test_bootstrap_schema_runs_all_ddl(monkeypatch) -> None:
    conn = _make_conn()
    _install_asyncpg_stub(monkeypatch, conn=conn)

    sys.modules.pop("docai.api.schema_bootstrap", None)
    import importlib
    sb = importlib.import_module("docai.api.schema_bootstrap")

    report = await sb.bootstrap_schema("postgresql://x:x@h/db")

    assert isinstance(report["tables_present"], list)
    assert isinstance(report["alter_columns_applied"], list)
    assert conn.execute.await_count >= 6
    sql_seen = " ".join(call.args[0] for call in conn.execute.await_args_list)
    assert "CREATE TABLE IF NOT EXISTS sessions" in sql_seen
    assert "CREATE TABLE IF NOT EXISTS session_messages" in sql_seen
    assert "uploaded_by_user_id" in sql_seen
    assert "CREATE TABLE IF NOT EXISTS cross_references" in sql_seen
    assert "CREATE TABLE IF NOT EXISTS audit_log" in sql_seen
    assert "ADD COLUMN IF NOT EXISTS section_path" in sql_seen
    assert "ALTER COLUMN user_id TYPE TEXT" in sql_seen
    assert "ADD COLUMN IF NOT EXISTS title" in sql_seen
    assert "ADD COLUMN IF NOT EXISTS doc_ids" in sql_seen
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_bootstrap_schema_idempotent(monkeypatch) -> None:
    conn = _make_conn()
    _install_asyncpg_stub(monkeypatch, conn=conn)

    sys.modules.pop("docai.api.schema_bootstrap", None)
    import importlib
    sb = importlib.import_module("docai.api.schema_bootstrap")

    await sb.bootstrap_schema("postgresql://x:x@h/db")
    await sb.bootstrap_schema("postgresql://x:x@h/db")
    assert conn.close.await_count == 2


@pytest.mark.asyncio
async def test_bootstrap_schema_propagates_connect_errors(monkeypatch) -> None:
    _install_asyncpg_stub(monkeypatch, raise_on_connect=True)

    sys.modules.pop("docai.api.schema_bootstrap", None)
    import importlib
    sb = importlib.import_module("docai.api.schema_bootstrap")

    with pytest.raises(ConnectionRefusedError):
        await sb.bootstrap_schema("postgresql://bad")


@pytest.mark.asyncio
async def test_bootstrap_includes_sessions_migration(monkeypatch) -> None:
    """Verify the sessions table gets the full new-shape migration so a
    pre-T1 init_db.py-created stub table is brought forward without
    needing manual intervention."""
    conn = _make_conn()
    _install_asyncpg_stub(monkeypatch, conn=conn)

    sys.modules.pop("docai.api.schema_bootstrap", None)
    import importlib
    sb = importlib.import_module("docai.api.schema_bootstrap")

    report = await sb.bootstrap_schema("postgresql://x:x@h/db")

    sql_seen = " ".join(call.args[0] for call in conn.execute.await_args_list)
    # All five session-shape migrations must run.
    for needle in (
        "ALTER COLUMN user_id TYPE TEXT",
        "ADD COLUMN IF NOT EXISTS title",
        "ADD COLUMN IF NOT EXISTS doc_ids",
        "ADD COLUMN IF NOT EXISTS summary",
        "ADD COLUMN IF NOT EXISTS summary_through_message_id",
    ):
        assert needle in sql_seen, f"missing migration: {needle}"

    # And they appear in the report under their column labels.
    assert any("sessions.user_id_to_text" in label for label in report["alter_columns_applied"])
    assert any("sessions.title" in label for label in report["alter_columns_applied"])
