from typing import Protocol, List, Dict, Any, AsyncGenerator
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

def classify_empty(content: str, thinking: str) -> str | None:
    """Classify an empty LLM response so callers can surface a meaningful message.

    This function mirrors OllamaClient._classify_empty but is placed here to avoid
    circular dependencies and ollama imports in test contexts.

    Returns:
        "empty_content_with_thinking" if content is blank but thinking has text.
        "empty_response" if both content and thinking are blank.
        None otherwise (normal non-empty response).
    """
    content_blank = (content or "").strip() == ""
    thinking_blank = (thinking or "").strip() == ""
    if content_blank and not thinking_blank:
        return "empty_content_with_thinking"
    if content_blank and thinking_blank:
        return "empty_response"
    return None

class LlmClientProtocol(Protocol):
    async def generate_response(
        self,
        messages: List[ChatMessage],
        system_prompt: str = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Generate response from the LLM"""
        ...

