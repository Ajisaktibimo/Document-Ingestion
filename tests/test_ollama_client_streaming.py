"""Tests for OllamaClient.generate_response_streaming.

The streaming method yields per-chunk dicts of shape
{kind: 'content'|'thinking', text: str}. It does NOT classify empty
responses (the engine does that at stream-end using _classify_empty
on the accumulated buffers).
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_fake_ollama(monkeypatch, fake_chunks):
    """Install a fake `ollama` module whose AsyncClient.chat returns an
    async iterator over the supplied chunks."""

    class _AsyncIter:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def chat(self, **kwargs):
            return _AsyncIter(fake_chunks)

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)


def _make_client(monkeypatch, chunks):
    _install_fake_ollama(monkeypatch, chunks)
    sys.modules.pop("docai.orchestrator.ollama_client", None)
    import importlib
    mod = importlib.import_module("docai.orchestrator.ollama_client")
    return mod.OllamaClient()


@pytest.mark.asyncio
async def test_streaming_yields_content_kind_for_content_chunks(monkeypatch):
    chunks = [
        {"message": {"content": "Hi"}},
        {"message": {"content": " world"}},
    ]
    client = _make_client(monkeypatch, chunks)

    out = []
    async for ev in client.generate_response_streaming(
        messages=[], system_prompt="", temperature=0.0,
    ):
        out.append(ev)

    assert out == [
        {"kind": "content", "text": "Hi"},
        {"kind": "content", "text": " world"},
    ]


@pytest.mark.asyncio
async def test_streaming_yields_thinking_kind_for_thinking_chunks(monkeypatch):
    chunks = [
        {"message": {"thinking": "let me think"}},
        {"message": {"content": "answer"}},
    ]
    client = _make_client(monkeypatch, chunks)

    out = []
    async for ev in client.generate_response_streaming(
        messages=[], system_prompt="", temperature=0.0,
    ):
        out.append(ev)

    assert out == [
        {"kind": "thinking", "text": "let me think"},
        {"kind": "content", "text": "answer"},
    ]


@pytest.mark.asyncio
async def test_streaming_skips_empty_chunk_payloads(monkeypatch):
    """A chunk with no content and no thinking (e.g. final stop frame)
    must NOT yield an event."""
    chunks = [
        {"message": {"content": "real"}},
        {"message": {}},                      # empty payload
        {"message": {"content": ""}},         # empty string
    ]
    client = _make_client(monkeypatch, chunks)

    out = []
    async for ev in client.generate_response_streaming(
        messages=[], system_prompt="", temperature=0.0,
    ):
        out.append(ev)

    assert out == [{"kind": "content", "text": "real"}]
