import asyncio
import logging
from typing import Dict, Any
from paddleocr import PPStructure
from docai.ocr.base import OcrServiceProtocol, OcrResult, OcrBlock

logger = logging.getLogger(__name__)

class PaddleOcrService:
    def __init__(self):
        logger.info("Initializing local PaddleOCR (PP-Structure) on CPU...")
        # Using English model for standard layout parsing, disabled GPU
        self.engine = PPStructure(lang='en', show_log=False, use_gpu=False)

    async def process_document(self, file_path: str) -> OcrResult:
        logger.info(f"PaddleOCR processing: {file_path}")
        
        # Run inference in threadpool since PaddleOCR is blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.engine, file_path)
        
        blocks = []
        full_text_parts = []
        
        # PP-Structure returns a list of dictionaries per region.
        # Format usually: [{'type': 'text', 'bbox': [x1, y1, x2, y2], 'res': [{'text': '...', 'confidence': ...}]}]
        for region in result:
            region_type = region.get('type', 'text')
            bbox = region.get('bbox', [0,0,0,0])
            
            res_content = region.get('res', [])
            region_text = ""
            
            if isinstance(res_content, list):
                # For standard text regions
                for line in res_content:
                    if isinstance(line, dict) and 'text' in line:
                        region_text += line['text'] + "\n"
            elif isinstance(res_content, dict):
                # For tables, PP-Structure might return HTML
                if 'html' in res_content:
                    region_text = res_content['html']
            
            region_text = region_text.strip()
            if region_text:
                blocks.append(
                    OcrBlock(
                        text=region_text,
                        type=region_type,
                        bbox=bbox,
                        page_num=1 # PaddleOCR operates per image; assume 1 for basic prototype
                    )
                )
                full_text_parts.append(region_text)
                
        return OcrResult(
            blocks=blocks,
            full_text="\n\n".join(full_text_parts),
            metadata={"engine": "paddleocr-ppstructure"}
        )

