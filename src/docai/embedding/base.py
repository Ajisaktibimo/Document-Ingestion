from typing import Protocol, List, Dict, Tuple

class DenseEmbedderProtocol(Protocol):
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute dense embeddings for indexing."""
        ...
        
    async def embed_query(self, query: str) -> List[float]:
        """Compute dense embedding for search."""
        ...

class SparseEmbedderProtocol(Protocol):
    async def embed_documents(self, texts: List[str]) -> List[Dict[str, List]]:
        """Compute sparse embeddings for indexing. Returns [{'indices': [...], 'values': [...]}, ...]"""
        ...
        
    async def embed_query(self, query: str) -> Dict[str, List]:
        """Compute sparse embedding for search. Returns {'indices': [...], 'values': [...]}"""
        ...

class RerankerProtocol(Protocol):
    async def rerank(self, query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
        """Returns list of (original_index, score) pairs sorted by score."""
        ...

