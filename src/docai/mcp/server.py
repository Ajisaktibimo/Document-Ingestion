import logging
import asyncio
from typing import List, Optional
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import asyncpg

from docai.config import settings
from docai.orchestrator.engine import OrchestratorEngine
from docai.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

# Initialize MCP Server
mcp = FastMCP("DocumentGroundingMCP")
_engine: OrchestratorEngine | None = None
_pipeline: IngestionPipeline | None = None


def get_engine() -> OrchestratorEngine:
    global _engine
    if _engine is None:
        _engine = OrchestratorEngine()
    return _engine


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline

# --- Pydantic Models for Tools ---

class SearchResultItem(BaseModel):
    doc_id: str = Field(..., description="Unique ID of the document")
    text: str = Field(..., description="Content of the document chunk")
    score: float = Field(..., description="Relevance score of the chunk")

class SearchResponse(BaseModel):
    results: List[SearchResultItem] = Field(default_factory=list, description="List of search results")

class DocumentStatusResponse(BaseModel):
    doc_id: Optional[str] = Field(None, description="Unique ID of the document")
    filename: Optional[str] = Field(None, description="Original filename")
    status: str = Field(..., description="Current status: processing, completed, failed, or not_found")

class IngestionResponse(BaseModel):
    message: str = Field(..., description="Status message")
    doc_id: Optional[str] = Field(None, description="Document ID if ingestion started successfully")

# --- MCP Tools ---

@mcp.tool()
async def search_documents(query: str, limit: int = 5) -> str:
    """
    Search the document knowledge base for relevant information using Hybrid Search and Reranking.
    Returns a formatted JSON string of the results.
    """
    logger.info(f"MCP Tool 'search_documents' called with query: '{query}'")
    try:
        engine = get_engine()
        # We reuse the dense/sparse/rerank logic from engine but bypass the LLM step
        query_dense = await engine.dense_embedder.embed_query(query)
        query_sparse = await engine.sparse_embedder.embed_query(query)
        
        hits = await engine.qdrant_store.search_hybrid(query_dense, query_sparse, limit=limit)
        
        documents = [hit['text'] for hit in hits]
        reranked_pairs = await engine.reranker.rerank(query, documents, top_k=limit)
        
        results = []
        for idx, score in reranked_pairs:
            hit = hits[idx]
            results.append(SearchResultItem(
                doc_id=hit['doc_id'],
                text=hit['text'],
                score=score
            ))
            
        response = SearchResponse(results=results)
        return response.model_dump_json()
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_document_status(file_hash: str) -> str:
    """
    Check the ingestion status of a document using its SHA-256 file hash.
    """
    logger.info(f"MCP Tool 'get_document_status' called for hash: {file_hash}")
    dsn = settings.POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://")
    
    try:
        conn = await asyncpg.connect(dsn)
        record = await conn.fetchrow(
            'SELECT doc_id, filename, status FROM doc_registry WHERE file_hash = $1', 
            file_hash
        )
        await conn.close()
        
        if record:
            resp = DocumentStatusResponse(
                doc_id=str(record['doc_id']),
                filename=record['filename'],
                status=record['status']
            )
        else:
            resp = DocumentStatusResponse(status="not_found")
            
        return resp.model_dump_json()
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def ingest_document(file_path: str, doc_class: str = "general") -> str:
    """
    Trigger the ingestion of a new document. 
    Warning: This process is synchronous in this MVP and may take a while for large documents.
    """
    logger.info(f"MCP Tool 'ingest_document' called for file: {file_path}")
    try:
        pipeline = get_pipeline()
        # In a production environment, this should be dispatched to Celery.
        # For the MVP, we run it directly.
        doc_id = await pipeline.ingest_document(file_path, doc_class)
        if doc_id:
            resp = IngestionResponse(
                message="Ingestion completed successfully",
                doc_id=str(doc_id)
            )
        else:
            resp = IngestionResponse(message="Ingestion failed or file not found")
            
        return resp.model_dump_json()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return f"Error: {str(e)}"

# --- Register Track B / Graph Tools ---
from docai.mcp.tools.graph import graph_query, entity_link, check_contradiction
from docai.mcp.tools.validation import verify_fact, ontology_check
from docai.mcp.tools.write import graph_write, quarantine_write, audit_log

mcp.add_tool(graph_query)
mcp.add_tool(entity_link)
mcp.add_tool(check_contradiction)

mcp.add_tool(verify_fact)
mcp.add_tool(ontology_check)

mcp.add_tool(graph_write)
mcp.add_tool(quarantine_write)
mcp.add_tool(audit_log)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    from scripts.init_db import main as init_db_main
except ImportError:
    init_db_main = None

def main():
    """Start the FastMCP server."""
    if init_db_main:
        logger.info("Running database initialization (Postgres & Qdrant) before starting MCP...")
        try:
            asyncio.run(init_db_main())
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    else:
        logger.warning("Could not import scripts.init_db to initialize database.")
        
    # Run as an SSE server so it can be exposed via Docker
    mcp.run(transport='sse', host=settings.MCP_HOST, port=settings.MCP_PORT)

if __name__ == "__main__":
    main()

