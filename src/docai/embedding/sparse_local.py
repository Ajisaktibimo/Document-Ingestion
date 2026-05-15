import logging
from typing import List, Dict
import numpy as np
from fastembed import SparseTextEmbedding
from docai.config import settings
from docai.embedding.base import SparseEmbedderProtocol

logger = logging.getLogger(__name__)


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_device_ids(value: str) -> List[int] | None:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    return ids or None


def _resolve_fastembed_runtime_kwargs() -> Dict[str, object]:
    device = settings.SPARSE_EMBEDDING_DEVICE.strip().lower()
    kwargs: Dict[str, object] = {}

    providers = _parse_csv(settings.SPARSE_EMBEDDING_PROVIDERS)
    if providers:
        kwargs["providers"] = providers

    if device in ("", "cpu"):
        kwargs["cuda"] = False
    elif device in ("cuda", "gpu"):
        kwargs["cuda"] = True
    elif device == "auto":
        pass
    else:
        raise ValueError(
            "SPARSE_EMBEDDING_DEVICE must be one of: cpu, cuda, gpu, auto"
        )

    device_ids = _parse_device_ids(settings.SPARSE_EMBEDDING_DEVICE_IDS)
    if device_ids:
        kwargs["device_ids"] = device_ids

    return kwargs


class LocalSparseEmbedder:
    def __init__(self):
        # Using Qdrant's BM42 lightweight sparse model
        # It runs via ONNX runtime locally
        self.model_name = "Qdrant/bm42-all-minilm-l6-v2-attentions"
        logger.info(f"Loading Local Sparse Embedder ({self.model_name}) from cache: {settings.SPARSE_CACHE_DIR}...")
        self.model = SparseTextEmbedding(
            model_name=self.model_name,
            cache_dir=settings.SPARSE_CACHE_DIR,
            **_resolve_fastembed_runtime_kwargs(),
        )
        logger.info("Local Sparse Embedder loaded successfully.")

    def _convert_to_dict(self, sparse_embedding) -> Dict[str, List]:
        # FastEmbed returns a SparseEmbedding object with .indices and .values
        # Qdrant client needs these as lists of ints and floats
        return {
            "indices": sparse_embedding.indices.tolist(),
            "values": sparse_embedding.values.tolist()
        }

    async def embed_documents(self, texts: List[str]) -> List[Dict[str, List]]:
        """
        Embeds a list of documents into sparse vectors.
        Uses fastembed synchronously but we can wrap it if needed.
        """
        # list() forces the generator to compute all embeddings
        embeddings = list(self.model.embed(texts))
        return [self._convert_to_dict(emb) for emb in embeddings]

    async def embed_query(self, query: str) -> Dict[str, List]:
        """
        Embeds a single query into a sparse vector.
        """
        # For fastembed, embed returns a generator
        embeddings = list(self.model.embed([query]))
        return self._convert_to_dict(embeddings[0])

