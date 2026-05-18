"""
Ingestion Pipeline
==================
Track A: OCR → Caption → Chunk → Embed → Postgres/Qdrant.

Upload is non-blocking: ingest_document() registers the document and returns
immediately; the heavy OCR/embed work runs as an asyncio background task.
"""
import asyncio
import uuid
import logging
import hashlib
import os
import asyncpg
from datetime import datetime, timezone
from typing import Optional
from docai.config import settings
from docai.providers import get_ocr_service, get_dense_embedder, get_sparse_embedder, get_captioner
from docai.ingestion.chunker import HierarchicalChunker
from docai.retrieval.qdrant_client import QdrantStore

logger = logging.getLogger(__name__)

# Strong references to in-flight background tasks so they can't be GC'd.
_active_tasks: set[asyncio.Task] = set()
_ingestion_semaphore: Optional[asyncio.Semaphore] = None
_ingestion_semaphore_limit: int = 0


def _get_ingestion_semaphore() -> asyncio.Semaphore:
    """Limit memory-heavy OCR/embed work across uploads in this API process."""
    global _ingestion_semaphore, _ingestion_semaphore_limit
    limit = max(1, settings.INGESTION_MAX_CONCURRENT_TASKS)
    if _ingestion_semaphore is None or _ingestion_semaphore_limit != limit:
        _ingestion_semaphore = asyncio.Semaphore(limit)
        _ingestion_semaphore_limit = limit
    return _ingestion_semaphore

class IngestionPipeline:
    def __init__(self):
        # Heavy services are loaded lazily so an upload request can register
        # quickly without duplicating model stacks while another ingest runs.
        self._ocr_service = None
        self._captioner = None
        self._dense_embedder = None
        self._sparse_embedder = None
        self._chunker = None
        self._qdrant_store = None

    @property
    def ocr_service(self):
        if self._ocr_service is None:
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    @ocr_service.setter
    def ocr_service(self, value):
        self._ocr_service = value

    @property
    def captioner(self):
        if self._captioner is None:
            self._captioner = get_captioner()
        return self._captioner

    @captioner.setter
    def captioner(self, value):
        self._captioner = value

    @property
    def dense_embedder(self):
        if self._dense_embedder is None:
            self._dense_embedder = get_dense_embedder()
        return self._dense_embedder

    @dense_embedder.setter
    def dense_embedder(self, value):
        self._dense_embedder = value

    @property
    def sparse_embedder(self):
        if self._sparse_embedder is None:
            self._sparse_embedder = get_sparse_embedder()
        return self._sparse_embedder

    @sparse_embedder.setter
    def sparse_embedder(self, value):
        self._sparse_embedder = value

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = HierarchicalChunker()
        return self._chunker

    @chunker.setter
    def chunker(self, value):
        self._chunker = value

    @property
    def qdrant_store(self):
        if self._qdrant_store is None:
            self._qdrant_store = QdrantStore()
        return self._qdrant_store

    @qdrant_store.setter
    def qdrant_store(self, value):
        self._qdrant_store = value

    @staticmethod
    def _caption_candidates(ocr_result):
        max_images = max(0, settings.MAX_IMAGES_PER_DOC)
        max_tables = max(0, settings.MAX_TABLES_PER_DOC)
        images = list(ocr_result.images[:max_images])
        tables = list(ocr_result.tables[:max_tables])
        skipped = (
            max(0, len(ocr_result.images) - len(images))
            + max(0, len(ocr_result.tables) - len(tables))
        )
        if skipped:
            logger.warning(
                "Skipping captioning for %d media items beyond configured limits "
                "(MAX_IMAGES_PER_DOC=%d, MAX_TABLES_PER_DOC=%d).",
                skipped,
                max_images,
                max_tables,
            )
        return images + tables

    def _get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    async def _register(
        self,
        conn: asyncpg.Connection,
        file_hash: str,
        filename: str,
        doc_class: str,
    ) -> tuple[Optional[uuid.UUID], bool]:
        """Insert or update doc_registry row.

        Returns (doc_id, already_done).
        already_done=True  → doc is completed or a non-stale background task is
                             already running; caller must NOT spawn another task.
        already_done=False → caller should spawn _run_pipeline as a background task.
        """
        existing = await conn.fetchrow(
            'SELECT doc_id, status, page_count, filename, updated_at FROM doc_registry WHERE file_hash = $1',
            file_hash,
        )

        if existing and existing["status"] == "completed" and existing["page_count"]:
            if existing["filename"] != filename:
                await conn.execute(
                    'UPDATE doc_registry SET filename = $2 WHERE file_hash = $1',
                    file_hash, filename,
                )
            logger.info("Document already ingested. doc_id: %s", existing["doc_id"])
            return existing["doc_id"], True

        if existing and existing["status"] == "processing":
            updated_at = existing["updated_at"]
            threshold = settings.STALE_PROCESSING_THRESHOLD_SECONDS
            if updated_at is None:
                logger.warning(
                    "Stale processing (updated_at IS NULL) for doc_id=%s; re-ingesting.",
                    existing["doc_id"],
                )
            else:
                age = (datetime.now(timezone.utc) - updated_at).total_seconds()
                if age <= threshold:
                    logger.info(
                        "Background ingestion already running (age=%.0fs). doc_id: %s",
                        age, existing["doc_id"],
                    )
                    return existing["doc_id"], True
                logger.warning(
                    "Stale processing (age=%.0fs > %.0fs) for doc_id=%s; re-ingesting.",
                    age, threshold, existing["doc_id"],
                )

        if existing:
            doc_id = existing["doc_id"]
            logger.info(
                "Retrying doc_id=%s (previous status=%s)", doc_id, existing["status"]
            )
            await conn.execute('DELETE FROM parent_chunks WHERE doc_id = $1', doc_id)
            await self.qdrant_store.delete_document(doc_id)
            await conn.execute(
                """UPDATE doc_registry
                   SET filename=$2, doc_class=$3, status='processing',
                       error_message=NULL, updated_at=NOW()
                   WHERE file_hash=$1""",
                file_hash, filename, doc_class,
            )
        else:
            doc_id = uuid.uuid4()
            await conn.execute(
                """INSERT INTO doc_registry
                       (doc_id, file_hash, filename, doc_class, status, error_message, updated_at)
                   VALUES ($1, $2, $3, $4, 'processing', NULL, NOW())""",
                doc_id, file_hash, filename, doc_class,
            )

        return doc_id, False

    async def ingest_document(
        self,
        file_path: str,
        doc_class: str = "general",
        display_filename: Optional[str] = None,
    ) -> Optional[uuid.UUID]:
        """Register document and return immediately.

        Spawns _run_pipeline as a background asyncio task so the HTTP response
        is not blocked by OCR / embedding.  The frontend polls GET /documents
        every 3 s while status == 'processing'.
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return None

        file_hash = self._get_file_hash(file_path)
        filename = display_filename or os.path.basename(file_path)
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")

        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            doc_id, already_done = await self._register(conn, file_hash, filename, doc_class)
        except Exception as e:
            logger.error("Failed to register document: %s", e, exc_info=True)
            return None
        finally:
            if conn is not None:
                await conn.close()

        if not already_done:
            asyncio.create_task(
                self._run_pipeline(file_path, doc_id, file_hash, filename)
            )

        return doc_id

    async def _run_pipeline(
        self,
        file_path: str,
        doc_id: uuid.UUID,
        file_hash: str,
        filename: str,
    ) -> None:
        """Background task: OCR → caption → chunk → embed → Postgres + Qdrant."""
        semaphore = _get_ingestion_semaphore()
        if semaphore.locked():
            logger.info(
                "Ingestion queued for doc_id=%s; max concurrent ingestion tasks is %d.",
                doc_id,
                settings.INGESTION_MAX_CONCURRENT_TASKS,
            )

        async with semaphore:
            await self._run_pipeline_unlocked(file_path, doc_id, file_hash, filename)

    async def _run_pipeline_unlocked(
        self,
        file_path: str,
        doc_id: uuid.UUID,
        file_hash: str,
        filename: str,
    ) -> None:
        """Run one ingestion job after concurrency admission."""
        logger.info("Background ingestion started for doc_id=%s (%s)", doc_id, filename)
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = None
        try:
            conn = await asyncpg.connect(dsn)

            # 1. OCR / Structural Extraction
            logger.info("Extracting text via %s parser...", settings.DOCUMENT_PARSER)
            ocr_result = await self.ocr_service.process_document(file_path)

            # 2. Caption images and tables
            all_media = self._caption_candidates(ocr_result)
            if all_media:
                logger.info("Captioning %d media items with local LLM...", len(all_media))
                await self.captioner.caption_all(all_media)

            # 3. Chunk
            logger.info("Chunking document...")
            chunks = self.chunker.chunk_document(ocr_result)

            # 4. Embed
            logger.info("Embedding %d chunks...", len(chunks))
            texts = [c.text for c in chunks]
            dense_vectors = await self.dense_embedder.embed_documents(texts)
            sparse_vectors = await self.sparse_embedder.embed_documents(texts)

            # 5. Postgres insert + mark completed (atomic)
            logger.info("Inserting into Postgres...")
            parent_ids = [uuid.uuid4() for _ in chunks]
            async with conn.transaction():
                for i, chunk in enumerate(chunks):
                    await conn.execute(
                        'INSERT INTO parent_chunks (parent_id, doc_id, heading, full_text, page_num) VALUES ($1,$2,$3,$4,$5)',
                        parent_ids[i], doc_id, chunk.heading, chunk.text, chunk.page_num,
                    )
                await conn.execute(
                    "UPDATE doc_registry SET status='completed', page_count=$2 WHERE doc_id=$1",
                    doc_id, ocr_result.page_count,
                )

            # 6. Qdrant insert (after Postgres commits)
            logger.info("Inserting into Qdrant...")
            payloads = [
                {
                    "heading": chunk.heading,
                    "heading_path": " > ".join(chunk.heading_path) if chunk.heading_path else "",
                    "page_num": chunk.page_num,
                    "image_refs": "|".join(chunk.image_refs),
                    "table_refs": "|".join(chunk.table_refs),
                    "has_table": chunk.has_table,
                    "has_image": chunk.has_image,
                }
                for chunk in chunks
            ]
            await self.qdrant_store.insert_chunks(
                parent_ids=parent_ids,
                doc_ids=[doc_id] * len(chunks),
                texts=texts,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                extra_payloads=payloads,
            )
            logger.info("Ingestion complete for doc_id=%s", doc_id)

        except Exception as e:
            logger.error("Background ingestion failed for doc_id=%s: %s", doc_id, e, exc_info=True)
            try:
                if conn:
                    await conn.execute(
                        "UPDATE doc_registry SET status='failed', error_message=$2, updated_at=NOW() WHERE file_hash=$1",
                        file_hash, str(e)[:2000],
                    )
            except Exception as db_err:
                logger.error("Failed to record failure for hash=%s: %s", file_hash, db_err)
        finally:
            if conn is not None:
                await conn.close()
