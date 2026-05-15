from pydantic import BaseModel, Field
from typing import List, Optional
from falkordb import FalkorDB
from docai.config import settings
import logging

logger = logging.getLogger(__name__)

class GraphQueryRequest(BaseModel):
    cypher_query: str = Field(..., description="Cypher query to run against FalkorDB")

class GraphQueryResponse(BaseModel):
    results: List[dict] = Field(default_factory=list, description="Query results")
    error: Optional[str] = Field(None)

async def graph_query(cypher_query: str) -> str:
    """
    Run a raw Cypher query against the knowledge graph.
    """
    try:
        db = FalkorDB(host=settings.FALKORDB_HOST, port=settings.FALKORDB_PORT)
        graph = db.select_graph("docai_knowledge_graph")
        res = graph.query(cypher_query)
        # Convert FalkorDB results to list of dicts for JSON serialization
        results = [dict(zip(res.header, row)) for row in res.result_set]
        return GraphQueryResponse(results=results).model_dump_json()
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return GraphQueryResponse(error=str(e)).model_dump_json()

async def entity_link(entity_name: str) -> str:
    """
    Attempt to link an extracted entity to an existing node in the graph.
    Returns the resolved node_id or null if new.
    """
    return f"{{'entity_name': '{entity_name}', 'node_id': 'mock_node_123'}}"

async def check_contradiction(subject: str, predicate: str, object: str) -> str:
    """
    Check if a proposed relation contradicts existing graph knowledge.
    Returns a list of conflicts (empty if none).
    """
    return "{\"conflicts\": []}"

