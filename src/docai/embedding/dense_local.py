import asyncio
import logging
from pathlib import Path
from typing import List
from docai.config import settings
from docai.embedding.base import DenseEmbedderProtocol

logger = logging.getLogger(__name__)


def _is_gpu_device(value: str) -> bool:
    device = value.strip().lower()
    return device == "gpu" or device.startswith("cuda")


def _resolve_dense_model_id() -> str:
    configured_model = settings.EMBEDDING_MODEL.strip()
    if configured_model and configured_model.lower() != "auto":
        return configured_model

    if _is_gpu_device(settings.DENSE_EMBEDDING_DEVICE):
        return settings.DENSE_EMBEDDING_GPU_MODEL

    return settings.DENSE_EMBEDDING_CPU_MODEL


def _resolve_dense_model_name_or_path() -> str:
    if not settings.DENSE_MODEL_PATH:
        return _resolve_dense_model_id()

    local_path = Path(settings.DENSE_MODEL_PATH)
    if local_path.exists():
        return str(local_path)

    logger.warning(
        "Configured DENSE_MODEL_PATH does not exist: %s. Falling back to dense model=%s",
        settings.DENSE_MODEL_PATH,
        _resolve_dense_model_id(),
    )
    return _resolve_dense_model_id()


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_device_ids(value: str) -> List[int] | None:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    return ids or None


def _resolve_fastembed_runtime_kwargs() -> dict[str, object]:
    device = settings.DENSE_EMBEDDING_DEVICE.strip().lower()
    kwargs: dict[str, object] = {}

    providers = _parse_csv(settings.DENSE_EMBEDDING_PROVIDERS)
    if providers:
        kwargs["providers"] = providers

    if device in ("", "cpu"):
        kwargs["cuda"] = False
    elif device in ("cuda", "gpu") or device.startswith("cuda:"):
        kwargs["cuda"] = True
    elif device == "auto":
        pass
    else:
        raise ValueError(
            "DENSE_EMBEDDING_DEVICE must be one of: cpu, cuda, cuda:N, gpu, auto"
        )

    device_ids = _parse_device_ids(settings.DENSE_EMBEDDING_DEVICE_IDS)
    if device_ids:
        kwargs["device_ids"] = device_ids

    return kwargs


class LocalDenseEmbedder:
    def __init__(self):
        model_name_or_path = _resolve_dense_model_name_or_path()
        logger.info(
            "Initializing dense embedder from: %s (backend=%s, device=%s)",
            model_name_or_path,
            settings.DENSE_EMBEDDING_BACKEND,
            settings.DENSE_EMBEDDING_DEVICE,
        )
        self.uses_fastembed = settings.DENSE_EMBEDDING_BACKEND == "onnx"
        if self.uses_fastembed:
            from fastembed import TextEmbedding

            self.model = TextEmbedding(
                model_name=model_name_or_path,
                cache_dir=settings.DENSE_CACHE_DIR,
                **_resolve_fastembed_runtime_kwargs(),
            )
        else:
            from sentence_transformers import SentenceTransformer

            # Nomic v2 requires trust_remote_code=True
            self.model = SentenceTransformer(
                model_name_or_path,
                trust_remote_code=True,
                backend=settings.DENSE_EMBEDDING_BACKEND,
                device=settings.DENSE_EMBEDDING_DEVICE,
            )
        self.doc_prefix = settings.DENSE_DOC_PREFIX
        self.query_prefix = settings.DENSE_QUERY_PREFIX

    def _encode(self, texts: List[str]):
        if self.uses_fastembed:
            return list(self.model.embed(texts))

        return self.model.encode(texts)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [self.doc_prefix + t for t in texts]
        loop = asyncio.get_event_loop()
        # run_in_executor avoids blocking the event loop during inference
        embeddings = await loop.run_in_executor(None, self._encode, prefixed_texts)
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        prefixed_query = self.query_prefix + query
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._encode, [prefixed_query])
        return embeddings[0].tolist()

