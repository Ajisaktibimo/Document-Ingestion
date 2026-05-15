import logging
import httpx
from typing import List
from pathlib import Path
from docai.config import settings
from docai.ocr.base import OcrResult, OcrBlock, OcrServiceProtocol

logger = logging.getLogger(__name__)

class ChandraService(OcrServiceProtocol):
    """
    Service for Chandra OCR, supporting both local and remote execution.
    
    - Local (hf): Runs models directly using transformers/torch.
    - Remote (vllm): Sends files to a remote Chandra API node.
    """

    def __init__(self):
        self.method = settings.CHANDRA_METHOD.lower()
        self.base_url = settings.CHANDRA_OCR_URL.rstrip("/")
        self.timeout = 120.0
        self.manager = None

        if self.method == "hf":
            logger.info(f"Initializing Local Chandra OCR (HF method) using: {settings.CHANDRA_MODEL_PATH}")
            # Lazy import to avoid dependency issues if not installed
            try:
                import os
                from chandra.model import InferenceManager
                
                # Override model checkpoint for truly local mode
                if settings.CHANDRA_MODEL_PATH:
                    os.environ["MODEL_CHECKPOINT"] = settings.CHANDRA_MODEL_PATH
                
                self.manager = InferenceManager(method="hf")
            except ImportError:
                logger.error("chandra-ocr not installed. Run 'pip install chandra-ocr[hf]'")
                raise ImportError("Missing 'chandra-ocr[hf]' for local execution.")
        else:
            logger.info(f"Initialized Remote Chandra OCR client at {self.base_url}")

    async def process_document(self, file_path: str) -> OcrResult:
        """
        Processes a document using either local or remote Chandra backend.
        """
        if self.method == "hf":
            return await self._process_local(file_path)
        else:
            return await self._process_remote(file_path)

    async def _process_local(self, file_path: str) -> OcrResult:
        """Runs chandra-ocr locally."""
        from chandra.model.schema import BatchInputItem
        
        logger.info(f"Processing {file_path} locally with Chandra...")
        
        # Chandra handles images/PDFs. For PDFs, we might need to handle page iteration 
        # or check if InferenceManager.generate handles it directly.
        # Based on docs, BatchInputItem takes the path.
        items = [
            BatchInputItem(
                image=file_path,
                prompt="Extract text and layout",
                prompt_type="ocr"
            )
        ]
        
        # InferenceManager.generate is typically synchronous, we wrap it
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.manager.generate, items)
        
        if not results:
            raise RuntimeError("Chandra produced no results.")
            
        return self._map_local_output(results[0])

    async def _process_remote(self, file_path: str) -> OcrResult:
        """Sends document to remote Chandra node (vLLM)."""
        path = Path(file_path)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (path.name, f, "application/pdf")}
                    response = await client.post(
                        f"{self.base_url}/process",
                        files=files
                    )
                response.raise_for_status()
                return self._map_remote_output(response.json())
            except Exception as e:
                logger.error(f"Chandra Remote Error: {str(e)}")
                raise

    def _map_local_output(self, output) -> OcrResult:
        """Maps BatchOutputItem to OcrResult."""
        blocks = []
        # output.chunks contains the layout segments
        for chunk in getattr(output, "chunks", []):
            blocks.append(OcrBlock(
                text=chunk.get("content", ""),
                type=chunk.get("label", "text"),
                bbox=chunk.get("bbox", [0, 0, 0, 0]),
                page_num=chunk.get("page_idx", 0) + 1
            ))

        return OcrResult(
            blocks=blocks,
            full_text=getattr(output, "raw", ""),
            markdown=getattr(output, "markdown", ""),
            page_count=getattr(output, "page_count", 1),
            metadata={"method": "hf", "model": settings.CHANDRA_MODEL_PATH}
        )

    def _map_remote_output(self, data: dict) -> OcrResult:
        """Maps remote JSON response to OcrResult."""
        blocks = []
        for b in data.get("blocks", []):
            blocks.append(OcrBlock(
                text=b.get("text", ""),
                type=b.get("type", "text"),
                bbox=b.get("bbox", [0, 0, 0, 0]),
                page_num=b.get("page_num", 1)
            ))

        return OcrResult(
            blocks=blocks,
            full_text=data.get("full_text", ""),
            markdown=data.get("markdown", ""),
            page_count=data.get("page_count", 1),
            metadata={"method": "vllm", "url": self.base_url}
        )

