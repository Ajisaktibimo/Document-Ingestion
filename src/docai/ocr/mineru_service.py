"""
MinerU Document Parser (MagicPDF)
================================
High-performance structural document parser using MinerU/MagicPDF.
Recommended for high-VRAM environments.
"""
import asyncio
import logging
import uuid
import os
from pathlib import Path
from typing import Optional

from docai.config import settings
from docai.ocr.base import OcrServiceProtocol, OcrResult, OcrBlock, ExtractedMedia

logger = logging.getLogger(__name__)


class MinerUService:
    """
    Document parser using MinerU (MagicPDF) for state-of-the-art structural extraction.
    
    Conforms to OcrServiceProtocol.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(settings.MINERU_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        self._ensure_config()
        logger.info(f"MinerUService initialized. Output dir: {self.output_dir}")

    def _ensure_config(self):
        """
        Ensures a magic-pdf.json exists in the models directory or user home.
        For 'truly local' mode, we favor a project-specific config.
        """
        home_config = Path.home() / "magic-pdf.json"
        project_config = Path(settings.MODELS_DIR) / "magic-pdf.json"
        
        if not project_config.exists() and not home_config.exists():
            logger.warning(
                f"No magic-pdf.json found. MinerU may attempt to download models. "
                f"Please run 'python scripts/setup_local_models.py' to generate one."
            )
        elif project_config.exists():
            # Set environment variable so magic-pdf uses this config
            os.environ["MAGIC_PDF_CONFIG"] = str(project_config)
            logger.info(f"Using project-local MinerU config: {project_config}")

    async def process_document(self, file_path: str) -> OcrResult:
        """Parse a document with MinerU and return a unified OcrResult."""
        logger.info(f"MinerU parsing (MagicPDF): {file_path}")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._parse_sync, file_path)
            return result
        except Exception as e:
            logger.error(f"MinerU parsing failed: {e}")
            return OcrResult(blocks=[], full_text="", metadata={"error": str(e), "engine": "mineru"})

    def _parse_sync(self, file_path: str) -> OcrResult:
        """Synchronous MinerU parsing using MagicPDF UNIPipe."""
        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.pipe.UNIPipe import UNIPipe
        except ImportError:
            logger.error(
                "magic-pdf is not installed. "
                "Install it with: pip install magic-pdf[full]"
            )
            return OcrResult(blocks=[], full_text="", metadata={"error": "magic-pdf not installed"})

        # 1. Read file bytes
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # 2. Setup Image Writer
        image_writer = FileBasedDataWriter(str(self.image_dir))
        image_dir_name = self.image_dir.name

        # 3. Initialize Pipeline
        # MinerU 2.5 uses UNIPipe for unified processing
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)

        # 4. Execute Pipeline
        pipe.pipe_classify()  # Classify PDF type (text/ocr)
        pipe.pipe_analyze()   # Layout analysis
        pipe.pipe_parse()     # Parse content

        # 5. Generate Markdown and structured data
        # We use drop_mode="none" to ensure we get all elements for our blocks
        markdown_text = pipe.pipe_mk_markdown(image_dir_name, drop_mode="none")
        
        # --- Map MinerU output to our OcrResult Schema ---
        blocks = []
        images: list[ExtractedMedia] = []
        tables: list[ExtractedMedia] = []
        current_heading_path: list[str] = []
        current_page = 1
        
        # Note: MinerU/MagicPDF output parsing is complex. 
        # For now, we extract structural elements from the generated Markdown 
        # to ensure compatibility with our HierarchicalChunker.
        
        for line in markdown_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Page break detection (MinerU often uses horizontal rules or specific markers)
            if stripped == "---" or "page_break" in stripped.lower():
                current_page += 1
                continue

            # Heading detection
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                heading_text = stripped.lstrip("# ").strip()
                current_heading_path = current_heading_path[:level - 1]
                current_heading_path.append(heading_text)

                blocks.append(OcrBlock(
                    text=heading_text,
                    type="heading",
                    page_num=current_page,
                    heading_path=list(current_heading_path),
                ))
                continue

            # Table detection
            if stripped.startswith("|"):
                table_id = str(uuid.uuid4())
                tables.append(ExtractedMedia(
                    media_id=table_id,
                    media_type="table",
                    page_num=current_page,
                    content_markdown=stripped,
                ))
                blocks.append(OcrBlock(
                    text=stripped,
                    type="table",
                    page_num=current_page,
                    heading_path=list(current_heading_path),
                    table_markdown=stripped,
                ))
                continue

            # Image detection
            if stripped.startswith("!["):
                image_id = str(uuid.uuid4())
                alt_text = stripped.split("](")[0].lstrip("![") if "](" in stripped else ""
                images.append(ExtractedMedia(
                    media_id=image_id,
                    media_type="image",
                    page_num=current_page,
                    caption=alt_text,
                ))
                blocks.append(OcrBlock(
                    text=alt_text or "(image)",
                    type="figure",
                    page_num=current_page,
                    heading_path=list(current_heading_path),
                    image_id=image_id,
                ))
                continue

            # Regular text
            blocks.append(OcrBlock(
                text=stripped,
                type="text",
                page_num=current_page,
                heading_path=list(current_heading_path),
            ))

        full_text = "\n".join(b.text for b in blocks if b.type == "text")

        logger.info(
            f"MinerU parsed: {current_page} pages, {len(blocks)} blocks, "
            f"{len(images)} images, {len(tables)} tables"
        )

        return OcrResult(
            blocks=blocks,
            full_text=full_text,
            metadata={"engine": "mineru-magicpdf", "v2.5_compatible": True},
            page_count=current_page,
            images=images,
            tables=tables,
            markdown=markdown_text,
        )

