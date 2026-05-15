from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from docai.api.routes import router as chat_router, get_engine
import uvicorn
import logging
import os
import asyncio
from docai.config import settings
from docai.api.schema_bootstrap import bootstrap_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run idempotent schema bootstrap before the app accepts traffic.

    Postgres bootstrap is fatal — if it fails, the app refuses to start
    so the failure is visible in `docker logs docai-api` rather than as
    a confusing per-request 500 once traffic arrives. Qdrant init is
    best-effort: a warning, no crash, since collection auto-creation
    happens on first use anyway.
    """
    # Startup logic
    logger.info("--- Document Ingestion API Starting ---")
    logger.info(f"Profile: {settings.BACKEND_PROFILE}")
    logger.info(f"Listening on: http://{settings.API_HOST}:{settings.API_PORT}")

    logger.info("Bootstrapping Postgres schema...")
    result = await bootstrap_schema(settings.POSTGRES_DSN)
    logger.info(
        "Schema bootstrap complete: %d tables, %d ALTER columns.",
        len(result["tables_present"]),
        len(result["alter_columns_applied"]),
    )

    # Qdrant collections — best effort. The init_qdrant function logs its
    # own status; we just guard against any unexpected raise.
    try:
        import sys
        from pathlib import Path
        # scripts/ isn't a package; add the project root so we can import.
        scripts_root = Path(__file__).resolve().parent.parent.parent.parent
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from scripts.init_db import init_qdrant
        await init_qdrant()
    except Exception as e:
        logger.warning("Qdrant bootstrap failed (non-fatal): %s", e)

    # Pre-load the RAG engine in a background task so the server socket
    # is already bound and health-checks pass while models load.
    async def load_engine():
        try:
            logger.info("Background: Pre-loading RAG Engine (Models) in separate thread...")
            await get_engine()
            logger.info("Background: RAG Engine fully loaded and ready.")
        except Exception as e:
            logger.error(f"Background: Failed to pre-load engine: {e}")

    asyncio.create_task(load_engine())

    logger.info("--- Web Server Process Started ---")
    yield
    # Shutdown logic — connections are per-request; nothing to tear down.
    logger.info("--- API Shutting Down ---")

app = FastAPI(
    title="Document Ingestion API",
    description="Document Grounded Conversational AI MVP",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(chat_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Mount frontend if the directory exists
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

def start():
    """Launch the FastAPI server."""
    uvicorn.run("docai.api.server:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)

if __name__ == "__main__":
    start()

