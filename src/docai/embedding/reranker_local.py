import logging
from pathlib import Path
from typing import List, Tuple
from sentence_transformers.cross_encoder import CrossEncoder
from docai.config import settings
from docai.embedding.base import RerankerProtocol

logger = logging.getLogger(__name__)


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _resolve_reranker_model_name_or_path() -> str:
    if not settings.RERANKER_MODEL_PATH:
        return settings.RERANKER_MODEL

    local_path = Path(settings.RERANKER_MODEL_PATH)
    if local_path.exists():
        return str(local_path)

    logger.warning(
        "Configured RERANKER_MODEL_PATH does not exist: %s. Falling back to %s",
        settings.RERANKER_MODEL_PATH,
        settings.RERANKER_MODEL,
    )
    return settings.RERANKER_MODEL

class LocalReranker:
    def __init__(self):
        # We use a lightweight multilingual BGE cross encoder
        model_name_or_path = _resolve_reranker_model_name_or_path()
        logger.info(f"Loading Local Reranker from: {model_name_or_path}...")
        self.model = CrossEncoder(
            model_name_or_path,
            backend=settings.RERANKER_BACKEND,
            device=settings.RERANKER_DEVICE,
        )
        logger.info("Local Reranker loaded successfully.")

    async def rerank(self, query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
        """
        Reranks documents given a query.
        Returns a sorted list of (document_index, score) pairs.
        """
        if not documents:
            return []

        # CrossEncoder expects pairs of (query, document)
        pairs = [[query, doc] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Sort indices by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Return top_k pairs of (original_index, score)
        return [(idx, float(scores[idx])) for idx in sorted_indices[:top_k]]

