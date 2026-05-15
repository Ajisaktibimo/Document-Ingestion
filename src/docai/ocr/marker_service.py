"""
Marker Document Parser
======================
Structural document parser using marker-pdf.
Preserves heading hierarchy, page boundaries, images, and tables.
Falls back gracefully if marker is not installed.
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from docai.config import settings
from docai.ocr.base import OcrServiceProtocol, OcrResult, OcrBlock, ExtractedMedia

logger = logging.getLogger(__name__)


class MarkerService:
    """
    Document parser using marker-pdf for high-fidelity structural extraction.

    Outputs:
    - Structured Markdown with heading hierarchy and page markers.
    - Extracted images saved to disk.
    - Extracted tables as Markdown.

    Conforms to OcrServiceProtocol so it is a drop-in replacement for PaddleOcrService.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/marker_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MarkerService initialized.")

    async def process_document(self, file_path: str) -> OcrResult:
        """Parse a document with Marker and return a unified OcrResult."""
        logger.info(f"Marker parsing: {file_path}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._parse_sync, file_path)
        return result

    def _parse_sync(self, file_path: str) -> OcrResult:
        """Synchronous Marker parsing (runs in thread pool)."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.config.parser import ConfigParser
        except ImportError:
            logger.error(
                "marker-pdf is not installed. "
                "Install it with: pip install marker-pdf"
            )
            # Return empty result instead of crashing
            return OcrResult(blocks=[], full_text="", metadata={"error": "marker-pdf not installed"})

        # Configure Marker
        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(config=config_parser.generate_config_dict())

        # Convert
        rendered = converter(file_path)
        markdown_text = rendered.markdown

        # --- Extract blocks from Markdown structure ---
        blocks = []
        images: list[ExtractedMedia] = []
        tables: list[ExtractedMedia] = []
        current_heading_path: list[str] = []
        current_page = 1
        image_count = 0

        for line in markdown_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Page break detection (Marker uses --- or similar)
            if stripped == "---":
                current_page += 1
                continue

            # Heading detection
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                heading_text = stripped.lstrip("# ").strip()
                # Trim heading path to current level
                current_heading_path = current_heading_path[:level - 1]
                current_heading_path.append(heading_text)

                blocks.append(OcrBlock(
                    text=heading_text,
                    type="heading",
                    page_num=current_page,
                    heading_path=list(current_heading_path),
                ))
                continue

            # Table detection (lines starting with |)
            if stripped.startswith("|"):
                # Accumulate contiguous table lines
                table_lines = [stripped]
                # Note: this is simplified — full parsing would track multi-line tables
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

            # Image detection (Markdown image syntax)
            if stripped.startswith("!["):
                if image_count < settings.MAX_IMAGES_PER_DOC:
                    image_id = str(uuid.uuid4())
                    # Extract alt text as initial caption
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
                    image_count += 1
                continue

            # Regular text
            blocks.append(OcrBlock(
                text=stripped,
                type="text",
                page_num=current_page,
                heading_path=list(current_heading_path),
            ))

        # Also extract images from Marker's rendered images dict
        if hasattr(rendered, "images") and rendered.images:
            images_dir = self.output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            for img_name, img_obj in rendered.images.items():
                if image_count >= settings.MAX_IMAGES_PER_DOC:
                    break
                image_id = str(uuid.uuid4())
                image_path = images_dir / f"{image_id}.png"
                try:
                    if hasattr(img_obj, "save"):
                        img_obj.save(str(image_path), format="PNG")
                    width = getattr(img_obj, "width", 0) if hasattr(img_obj, "width") else 0
                    height = getattr(img_obj, "height", 0) if hasattr(img_obj, "height") else 0
                    images.append(ExtractedMedia(
                        media_id=image_id,
                        media_type="image",
                        page_num=1,  # Marker doesn't always track page for images
                        file_path=str(image_path),
                        width=width,
                        height=height,
                    ))
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save Marker image {img_name}: {e}")

        full_text = "\n".join(b.text for b in blocks if b.type == "text")

        logger.info(
            f"Marker parsed: {current_page} pages, {len(blocks)} blocks, "
            f"{len(images)} images, {len(tables)} tables"
        )

        return OcrResult(
            blocks=blocks,
            full_text=full_text,
            metadata={"engine": "marker-pdf"},
            page_count=current_page,
            images=images,
            tables=tables,
            markdown=markdown_text,
        )

