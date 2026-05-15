import uuid
import logging
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from docai.config import settings

logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.collection_name = "document_chunks"

    @staticmethod
    def _build_doc_filter(doc_ids: List[str] | None):
        if not doc_ids:
            return None

        scoped_ids = [str(doc_id) for doc_id in doc_ids if doc_id]
        if not scoped_ids:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=scoped_ids),
                )
            ]
        )

    async def init_collection(self):
        """
        Initializes the collection with named dense and sparse vectors.
        Call this during app startup or before first ingestion.
        """
        if not await self.client.collection_exists(self.collection_name):
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            try:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(size=settings.EMBEDDING_DIM, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams()
                    }
                )
            except Exception:
                # Another worker may have created the collection concurrently.
                if await self.client.collection_exists(self.collection_name):
                    logger.info(
                        "Collection %s already exists (concurrent creation race) — continuing.",
                        self.collection_name,
                    )
                else:
                    raise

    async def insert_chunks(
        self, 
        parent_ids: List[uuid.UUID], 
        doc_ids: List[uuid.UUID], 
        texts: List[str], 
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, List]] = None,
        extra_payloads: List[Dict[str, Any]] = None,
    ):
        """
        Inserts document chunks into Qdrant for retrieval.
        Supports both dense and sparse vectors.
        Optional extra_payloads merges enriched metadata into the payload.
        """
        await self.init_collection()

        points = []
        for i in range(len(parent_ids)):
            vector = {
                "dense": dense_vectors[i]
            }
            if sparse_vectors and i < len(sparse_vectors):
                vector["sparse"] = models.SparseVector(
                    indices=sparse_vectors[i]["indices"],
                    values=sparse_vectors[i]["values"]
                )

            payload = {
                "doc_id": str(doc_ids[i]),
                "text": texts[i],
            }
            # Merge enriched metadata (heading_path, page_num, image_refs, etc.)
            if extra_payloads and i < len(extra_payloads):
                payload.update(extra_payloads[i])

            points.append(
                models.PointStruct(
                    id=str(parent_ids[i]),
                    vector=vector,
                    payload=payload,
                )
            )

        logger.info(f"Upserting {len(points)} points to Qdrant...")
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def delete_document(self, doc_id: uuid.UUID | str):
        """Delete all indexed chunks for a document before retrying ingestion."""
        if not await self.client.collection_exists(self.collection_name):
            return

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=str(doc_id)),
                        )
                    ]
                )
            ),
        )

    async def search_dense(
        self,
        query_vector: List[float],
        limit: int = 5,
        doc_ids: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense search using named vectors.
        """
        query_filter = self._build_doc_filter(doc_ids)
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            hit.payload
            | {
                "parent_id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "doc_id": hit.payload.get("doc_id", "")
            }
            for hit in results.points
        ]

    async def search_hybrid(
        self,
        query_dense: List[float],
        query_sparse: Dict[str, List],
        limit: int = 5,
        doc_ids: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using Prefetch and Qdrant's native RRF (supported in recent Qdrant versions).
        """
        query_filter = self._build_doc_filter(doc_ids)
        try:
            results = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_dense,
                        using="dense",
                        filter=query_filter,
                        limit=limit * 2
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query_sparse["indices"],
                            values=query_sparse["values"]
                        ),
                        using="sparse",
                        filter=query_filter,
                        limit=limit * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit * 2, # Fetch more so we can rerank locally
                with_payload=True,
            )
            
            return [
                hit.payload
                | {
                    "parent_id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "doc_id": hit.payload.get("doc_id", "")
                }
                for hit in results.points
            ]
        except Exception as e:
            logger.error(f"Hybrid search via query_points failed: {e}. Falling back to manual RRF.")
            return await self._manual_rrf(query_dense, query_sparse, limit, doc_ids=doc_ids)

    async def _manual_rrf(
        self,
        query_dense: List[float],
        query_sparse: Dict[str, List],
        limit: int,
        doc_ids: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Fallback manual RRF if query_points API is not fully compatible with Qdrant server."""
        query_filter = self._build_doc_filter(doc_ids)
        dense_results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            query_filter=query_filter,
            limit=limit*2,
            with_payload=True
        )
        sparse_results = await self.client.query_points(
            collection_name=self.collection_name, 
            query=models.SparseVector(indices=query_sparse["indices"], values=query_sparse["values"]),
            using="sparse",
            query_filter=query_filter,
            limit=limit*2, 
            with_payload=True
        )
        dense_hits = dense_results.points
        sparse_hits = sparse_results.points

        scores = {}
        payloads = {}
        k = settings.RRF_K  # RRF constant
        
        for rank, hit in enumerate(dense_hits):
            scores[hit.id] = scores.get(hit.id, 0) + 1.0 / (k + rank + 1)
            payloads[hit.id] = hit.payload
            
        for rank, hit in enumerate(sparse_hits):
            scores[hit.id] = scores.get(hit.id, 0) + 1.0 / (k + rank + 1)
            payloads[hit.id] = hit.payload
            
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            payloads[doc_id]
            | {
                "parent_id": doc_id,
                "score": scores[doc_id],
                "text": payloads[doc_id].get("text", ""),
                "doc_id": payloads[doc_id].get("doc_id", "")
            }
            for doc_id in sorted_ids[:limit*2]
        ]

