import importlib
import sys
import types

import pytest

from docai.orchestrator.base import ChatMessage


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_ollama_client_applies_timeout_and_bounded_generation(monkeypatch) -> None:
    calls = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        async def chat(self, **kwargs):
            calls["chat"] = kwargs
            return {"message": {"content": "ok"}}

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    ollama_client = _reload_module("docai.orchestrator.ollama_client")
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_BASE_URL", "http://ollama")
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_MODEL", "example/model")
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_REQUEST_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_NUM_PREDICT", 32)
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_NUM_CTX", 1024)
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_KEEP_ALIVE", "1m")
    monkeypatch.setattr(ollama_client.settings, "OLLAMA_THINK", False)

    client = ollama_client.OllamaClient(think=True)
    response = await client.generate_response(
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.1,
    )

    assert response == {"content": "ok", "thinking": ""}
    assert calls["init"] == {"host": "http://ollama", "timeout": 2.5}
    assert calls["chat"]["model"] == "example/model"
    assert calls["chat"]["options"] == {"temperature": 0.1, "num_predict": 32, "num_ctx": 1024}
    assert calls["chat"]["think"] is True
    assert calls["chat"]["keep_alive"] == "1m"


@pytest.mark.asyncio
async def test_generate_response_works_with_pydantic_style_object(monkeypatch):
    """generate_response must handle object-style (non-dict) responses from newer ollama-python."""
    class _FakeMsg:
        content = "hello from pydantic"
        thinking = ""

    class _FakeResponse:
        message = _FakeMsg()
        # No .get() method — simulates Pydantic model

    class _FakeAsyncClient:
        def __init__(self, *a, **kw): pass
        async def chat(self, **kw): return _FakeResponse()

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    sys.modules.pop("docai.orchestrator.ollama_client", None)
    mod = importlib.import_module("docai.orchestrator.ollama_client")

    monkeypatch.setattr(mod.settings, "OLLAMA_BASE_URL", "http://ollama")
    monkeypatch.setattr(mod.settings, "OLLAMA_MODEL", "example/model")
    monkeypatch.setattr(mod.settings, "OLLAMA_REQUEST_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setattr(mod.settings, "OLLAMA_NUM_PREDICT", 0)
    monkeypatch.setattr(mod.settings, "OLLAMA_NUM_CTX", 0)
    monkeypatch.setattr(mod.settings, "OLLAMA_KEEP_ALIVE", None)
    monkeypatch.setattr(mod.settings, "OLLAMA_THINK", False)

    client = mod.OllamaClient()

    result = await client.generate_response(
        messages=[],
        system_prompt="be helpful",
        temperature=0.0,
    )
    assert result["content"] == "hello from pydantic"
