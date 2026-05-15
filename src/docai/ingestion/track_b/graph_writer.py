import logging
from typing import Dict, Any
from falkordb import FalkorDB
from docai.config import settings

logger = logging.getLogger(__name__)

class FalkorWriter:
    def __init__(self):
        self.host = settings.FALKORDB_HOST
        self.port = settings.FALKORDB_PORT
        self.graph_name = "docai_knowledge_graph"

    def _get_connection(self):
        # Synchronous FalkorDB client
        return FalkorDB(host=self.host, port=self.port)

    def write_relation(self, relation: Dict[str, Any]):
        """
        Writes a validated relation to FalkorDB.
        """
        try:
            db = self._get_connection()
            graph = db.select_graph(self.graph_name)
            
            subj = relation["subject"].replace("'", "")
            pred = relation["predicate"].replace(" ", "_").upper()
            obj = relation["object"].replace("'", "")
            
            query = f"MERGE (s:Entity {{name: '{subj}'}}) MERGE (o:Entity {{name: '{obj}'}}) MERGE (s)-[:{pred}]->(o)"
            
            logger.info(f"Executing Cypher: {query}")
            graph.query(query)
            logger.info("Relation written to FalkorDB.")
        except Exception as e:
            logger.error(f"FalkorDB write failed: {e}")
            raise

    def write_quarantine(self, relation: Dict[str, Any], reason: str):
        """
        Writes failed relation to quarantine (could be Postgres or separate graph node).
        """
        logger.warning(f"Quarantining relation due to {reason}: {relation}")
        # In a real system, we'd write to a Postgres `quarantine` table.

