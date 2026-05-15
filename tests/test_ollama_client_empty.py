"""Tests for empty-response detection in OllamaClient.

Covers Task T1: when the model returns empty content (with or without
thinking), the returned dict should include a ``reason`` field so the
engine can surface a meaningful message instead of a blank one.
"""

import importlib
import sys
import types

import pytest

from docai.orchestrator.base import ChatMessage


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _install_fake_ollama(monkeypatch, message_payload):
    """Install a fake ``ollama`` module whose AsyncClient.chat returns
    ``{"message": message_payload}``.
    """

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def chat(self, **kwargs):
            return {"message": message_payload}

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)


def _configure_settings(monkeypatch, ollama_client_module):
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_BASE_URL", "http://ollama")
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_MODEL", "example/model")
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_REQUEST_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_NUM_PREDICT", 32)
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_NUM_CTX", 1024)
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_KEEP_ALIVE", "1m")
    monkeypatch.setattr(ollama_client_module.settings, "OLLAMA_THINK", False)


@pytest.mark.asyncio
async def test_empty_content_with_thinking_sets_reason(monkeypatch) -> None:
    _install_fake_ollama(monkeypatch, {"content": "", "thinking": "thoughts"})

    ollama_client = _reload_module("docai.orchestrator.ollama_client")
    _configure_settings(monkeypatch, ollama_client)

    client = ollama_client.OllamaClient(think=True)
    response = await client.generate_response(
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.1,
    )

    assert response.get("content") == ""
    assert response.get("thinking") == "thoughts"
    assert response.get("reason") == "empty_content_with_thinking"


@pytest.mark.asyncio
async def test_empty_content_and_thinking_sets_reason(monkeypatch) -> None:
    _install_fake_ollama(monkeypatch, {"content": "", "thinking": ""})

    ollama_client = _reload_module("docai.orchestrator.ollama_client")
    _configure_settings(monkeypatch, ollama_client)

    client = ollama_client.OllamaClient(think=True)
    response = await client.generate_response(
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.1,
    )

    assert response.get("content") == ""
    assert response.get("thinking") == ""
    assert response.get("reason") == "empty_response"
