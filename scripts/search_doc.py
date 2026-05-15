import asyncio
import sys
import argparse
import logging
from docai.providers import get_dense_embedder
from docai.retrieval.qdrant_client import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Test dense retrieval from Qdrant")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--limit", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    
    logger.info("Initializing Dense Embedder...")
    embedder = get_dense_embedder()
    
    logger.info(f"Embedding query: '{args.query}'")
    query_vector = await embedder.embed_query(args.query)
    
    logger.info(f"Searching Qdrant for top {args.limit} results...")
    qdrant_store = QdrantStore()
    results = await qdrant_store.search_dense(query_vector, limit=args.limit)
    
    print("\n--- Search Results ---")
    for i, res in enumerate(results):
        print(f"\nResult {i+1} [Score: {res['score']:.4f}]")
        print(f"Doc ID: {res['doc_id']}")
        print(f"Text: {res['text'][:200]}...")
        
if __name__ == "__main__":
    asyncio.run(main())
