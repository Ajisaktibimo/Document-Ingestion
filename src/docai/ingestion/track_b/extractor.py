import logging
from typing import List, Dict, Any
from docai.config import settings
from docai.providers import get_llm_client
from docai.orchestrator.base import ChatMessage

logger = logging.getLogger(__name__)

class GraphExtractor:
    def __init__(self):
        self.llm_client = get_llm_client(think=settings.INGESTION_OLLAMA_THINK)

    async def extract_entities_and_relations(self, text: str, ontology_name: str) -> List[Dict[str, Any]]:
        """
        Extract nodes and edges from text based on an ontology.
        In MVP, this returns a structured mock or makes a simple LLM call.
        """
        logger.info(f"Extracting graph entities using ontology {ontology_name}...")
        
        system_prompt = (
            f"Extract entities and relationships from the text based on the {ontology_name} ontology. "
            "Return a JSON array of extracted relations in the format: "
            "[{\"subject\": \"Entity1\", \"predicate\": \"RELATION\", \"object\": \"Entity2\", \"confidence\": 0.9}]"
        )
        
        messages = [ChatMessage(role="user", content=text)]
        
        try:
            # We enforce JSON mode or handle the parsing. 
            # For this MVP, we will mock the LLM parsing if it doesn't return perfect JSON.
            # Real implementation would use function calling or structured outputs.
            response = await self.llm_client.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.1
            )
            # Dummy parse - in production we'd parse the JSON properly.
            return [{"subject": "MockEntityA", "predicate": "MOCK_RELATION", "object": "MockEntityB", "confidence": 0.95}]
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return []

