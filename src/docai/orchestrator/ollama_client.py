import logging
import asyncio
from typing import List, AsyncGenerator
import ollama
from docai.config import settings
from docai.orchestrator.base import LlmClientProtocol, ChatMessage, classify_empty

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, think: bool | None = None):
        self.model = settings.OLLAMA_MODEL
        self.timeout_seconds = settings.OLLAMA_REQUEST_TIMEOUT_SECONDS
        self.think = settings.OLLAMA_THINK if think is None else think
        self.client = ollama.AsyncClient(
            host=settings.OLLAMA_BASE_URL,
            timeout=self.timeout_seconds,
        )
        logger.info(
            "Initialized Ollama Client pointing to %s (Model: %s, timeout=%ss, num_predict=%s, num_ctx=%s, think=%s)",
            settings.OLLAMA_BASE_URL,
            self.model,
            self.timeout_seconds,
            settings.OLLAMA_NUM_PREDICT,
            settings.OLLAMA_NUM_CTX,
            self.think,
        )

    @staticmethod
    def _classify_empty(content: str, thinking: str) -> str | None:
        """Classify an empty Ollama response so callers can surface a
        meaningful message instead of a blank one.

        Delegates to the shared implementation in base.classify_empty to avoid drift.

        Returns:
            "empty_content_with_thinking" if ``content`` is blank but
                ``thinking`` has text (model produced reasoning but no
                final answer).
            "empty_response" if both ``content`` and ``thinking`` are
                blank (model produced nothing).
            ``None`` otherwise (normal non-empty response).
        """
        return classify_empty(content, thinking)

    async def generate_response(
        self,
        messages: List[ChatMessage],
        system_prompt: str = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:

        # Critical: Guard against stream=True before making any network calls
        if stream:
            raise ValueError(
                "generate_response no longer supports stream=True. "
                "Use generate_response_streaming() for SSE flows."
            )

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        try:
            options = {"temperature": temperature}
            if settings.OLLAMA_NUM_PREDICT > 0:
                options["num_predict"] = settings.OLLAMA_NUM_PREDICT
            if settings.OLLAMA_NUM_CTX > 0:
                options["num_ctx"] = settings.OLLAMA_NUM_CTX

            chat_request = self.client.chat(
                model=self.model,
                messages=formatted_messages,
                options=options,
                stream=stream,
                think=self.think,
                keep_alive=settings.OLLAMA_KEEP_ALIVE or None,
            )
            response = await asyncio.wait_for(chat_request, timeout=self.timeout_seconds)

            # Handle both dict (older ollama-python) and Pydantic model (>= 0.4)
            if isinstance(response, dict):
                msg = response.get('message', {})
                content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                thinking = msg.get('thinking', '') if isinstance(msg, dict) else getattr(msg, 'thinking', '')
            else:
                msg = getattr(response, 'message', None) or {}
                content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                thinking = msg.get('thinking', '') if isinstance(msg, dict) else getattr(msg, 'thinking', '')
            result = {
                "content": content,
                "thinking": thinking
            }
            reason = self._classify_empty(content, thinking)
            if reason is not None:
                result["reason"] = reason
            return result
        except asyncio.TimeoutError as e:
            message = (
                f"Ollama generation timed out after {self.timeout_seconds}s "
                f"(model={self.model}, host={settings.OLLAMA_BASE_URL})"
            )
            logger.error(message)
            raise TimeoutError(message) from e

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise e

    async def generate_response_streaming(
        self,
        *,
        messages,
        system_prompt: str,
        temperature: float = 0.0,
    ):
        """Stream LLM output as typed chunks.

        Yields ``{"kind": "content" | "thinking", "text": str}`` per
        non-empty chunk. Does NOT classify empty responses — the engine
        does that at stream-end using the accumulated buffers + the
        existing :meth:`_classify_empty` static method.

        Empty payloads (final stop frames, no-op chunks) are skipped so
        the consumer never sees zero-length text events.
        """
        ollama_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            ollama_messages.append({"role": m.role, "content": m.content})

        # Build options dict with same > 0 guards as generate_response
        options = {"temperature": temperature}
        if settings.OLLAMA_NUM_PREDICT > 0:
            options["num_predict"] = settings.OLLAMA_NUM_PREDICT
        if settings.OLLAMA_NUM_CTX > 0:
            options["num_ctx"] = settings.OLLAMA_NUM_CTX

        # Use self.client instead of creating a new AsyncClient
        # Wrap in asyncio.wait_for to apply consistent timeout
        stream = await asyncio.wait_for(
            self.client.chat(
                model=settings.OLLAMA_MODEL,
                messages=ollama_messages,
                stream=True,
                think=self.think,
                options=options,
                keep_alive=settings.OLLAMA_KEEP_ALIVE or None,
            ),
            timeout=self.timeout_seconds,
        )

        async for chunk in stream:
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            thinking = msg.get("thinking") if isinstance(msg, dict) else getattr(msg, "thinking", "")
            if thinking:
                yield {"kind": "thinking", "text": thinking}
            if content:
                yield {"kind": "content", "text": content}

