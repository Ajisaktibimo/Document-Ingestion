import asyncio
import logging
import httpx
import redis.asyncio as redis
import asyncpg
from docai.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def check_redis():
    try:
        client = redis.from_url(settings.REDIS_URL)
        await client.ping()
        await client.aclose()
        logger.info("✅ Redis (Port 6381): OK")
        return True
    except Exception as e:
        logger.error(f"❌ Redis (Port 6381): FAILED - {e}")
        return False

async def check_qdrant():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.QDRANT_URL}/healthz")
            if resp.status_code == 200:
                logger.info("✅ Qdrant (Port 6333): OK")
                return True
            else:
                logger.error(f"❌ Qdrant (Port 6333): FAILED - HTTP {resp.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ Qdrant (Port 6333): FAILED - {e}")
        return False

async def check_postgres():
    try:
        dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn)
        await conn.close()
        logger.info("✅ Postgres (Port 5432): OK")
        return True
    except Exception as e:
        logger.error(f"❌ Postgres (Port 5432): FAILED - {e}")
        return False

async def check_llm():
    try:
        if settings.BACKEND_PROFILE == "production":
            url = f"{settings.VLLM_BASE_URL}/v1/models"
            name = "vLLM"
        else:
            url = f"{settings.OLLAMA_BASE_URL}/api/tags"
            name = "Ollama"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                logger.info(f"✅ {name}: OK")
                return True
            else:
                logger.error(f"❌ {name}: FAILED - HTTP {resp.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ {name}: FAILED - {e}")
        return False

async def main():
    logger.info(f"Running health checks for profile: {settings.BACKEND_PROFILE.upper()}")
    print("-" * 50)
    
    results = await asyncio.gather(
        check_redis(),
        check_qdrant(),
        check_postgres(),
        check_llm()
    )
    
    print("-" * 50)
    if all(results):
        logger.info("🎉 All systems go!")
        exit(0)
    else:
        logger.error("⚠️ Some services failed. Check docker-compose and local endpoints.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
