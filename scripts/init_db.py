"""CLI entry point for setting up Qdrant collections and Postgres schema.

Postgres DDL is delegated to ``docai.api.schema_bootstrap.bootstrap_schema``
which is also called automatically from the FastAPI lifespan hook on
every API container startup. Running this script manually is only
needed for: (a) standalone Qdrant collection creation when the API
isn't running, (b) manual recovery from a broken state, or (c) CI
seeding for tests.
"""
import asyncio
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from docai.api.schema_bootstrap import bootstrap_schema
from docai.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def init_qdrant() -> None:
    """Create the two Qdrant collections if they don't exist.

    NOTE: schema_bootstrap.py deliberately does NOT touch Qdrant — its
    scope is Postgres DDL only. Qdrant collection setup stays here so
    the lifespan hook can call this function alongside bootstrap_schema.
    """
    client = AsyncQdrantClient(url=settings.QDRANT_URL)

    # 1. episodic_memory collection
    try:
        exists = await client.collection_exists("episodic_memory")
        if not exists:
            await client.create_collection(
                collection_name="episodic_memory",
                vectors_config={
                    "dense": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
            logger.info("Created Qdrant collection: episodic_memory")
        else:
            logger.info("Qdrant collection episodic_memory already exists.")
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection episodic_memory: {e}")

    # 2. document_chunks collection
    try:
        exists = await client.collection_exists("document_chunks")
        if not exists:
            await client.create_collection(
                collection_name="document_chunks",
                vectors_config={
                    "dense": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
            logger.info("Created Qdrant collection: document_chunks")
        else:
            logger.info("Qdrant collection document_chunks already exists.")
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection document_chunks: {e}")


async def main() -> None:
    logger.info("Initializing databases...")
    await init_qdrant()
    report = await bootstrap_schema(settings.POSTGRES_DSN)
    logger.info(
        "Postgres schema bootstrap: %d tables, %d ALTER columns.",
        len(report["tables_present"]),
        len(report["alter_columns_applied"]),
    )
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(main())
