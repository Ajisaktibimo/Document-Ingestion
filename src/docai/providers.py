"""
Provider Factory
================
Returns concrete implementations based on ``settings.BACKEND_PROFILE``
and ``settings.DOCUMENT_PARSER``.

All providers share the same Pydantic-compatible protocol interfaces
defined in ``src/ocr/base.py``, ``src/embedding/base.py``, and
``src/orchestrator/base.py`` so they are hot-swappable via config.
"""
from typing import Any
from docai.config import settings


# ── OCR / Document Parser ──────────────────────────────────────────────

def get_ocr_service() -> Any:
    """
    Returns the document parser selected by ``DOCUMENT_PARSER``.
    All parsers conform to ``OcrServiceProtocol``.
    """
    parser = settings.DOCUMENT_PARSER.lower()

    if parser == "marker":
        from docai.ocr.marker_service import MarkerService
        return MarkerService()

    elif parser == "mineru":
        from docai.ocr.mineru_service import MinerUService
        return MinerUService()

    elif parser == "docling":
        from docai.ocr.docling_service import DoclingService
        return DoclingService()

    elif parser == "chandra":
        from docai.ocr.chandra_service import ChandraService
        return ChandraService()

    else:  # default: "paddle"
        from docai.ocr.paddle_service import PaddleOcrService
        return PaddleOcrService()


# ── Captioner ──────────────────────────────────────────────────────────

def get_captioner() -> Any:
    """
    Returns the local captioner for images/tables.
    Uses Ollama models configured via OLLAMA_MODEL / CAPTION_VISION_MODEL.
    """
    from docai.ocr.captioner import LocalCaptioner
    return LocalCaptioner()


# ── LLM ────────────────────────────────────────────────────────────────

def get_llm_client(think: bool | None = None) -> Any:
    if settings.BACKEND_PROFILE == "production":
        raise NotImplementedError("Production LLM client not implemented yet.")

    from docai.orchestrator.ollama_client import OllamaClient
    return OllamaClient(think=think)


# ── Embedders ──────────────────────────────────────────────────────────

def get_dense_embedder() -> Any:
    if settings.BACKEND_PROFILE == "production":
        raise NotImplementedError("Production dense embedder not implemented yet.")

    from docai.embedding.dense_local import LocalDenseEmbedder
    return LocalDenseEmbedder()


def get_sparse_embedder() -> Any:
    if settings.BACKEND_PROFILE == "production":
        raise NotImplementedError("Production sparse embedder not implemented yet.")

    from docai.embedding.sparse_local import LocalSparseEmbedder
    return LocalSparseEmbedder()


# ── Reranker ───────────────────────────────────────────────────────────

def get_reranker() -> Any:
    if settings.BACKEND_PROFILE == "production":
        raise NotImplementedError("Production reranker not implemented yet.")

    from docai.embedding.reranker_local import LocalReranker
    return LocalReranker()

