import logging
import asyncio
from typing import Any
from pathlib import Path
from docai.config import settings
from docai.ocr.base import OcrResult, OcrBlock, OcrServiceProtocol, ExtractedMedia

logger = logging.getLogger(__name__)


def _build_converter():
    """Build a DocumentConverter, using CUDA when DEVICE=gpu."""
    from docling.document_converter import DocumentConverter

    if settings.DEVICE.lower() != "gpu":
        return DocumentConverter()

    try:
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            AcceleratorOptions,
            AcceleratorDevice,
        )
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("Docling configured with CUDA acceleration.")
        return converter
    except Exception as e:
        logger.warning(
            "Could not configure Docling with CUDA (%s); falling back to CPU.", e
        )
        return DocumentConverter()


class DoclingService(OcrServiceProtocol):
    """
    Service using IBM's Docling for high-fidelity document conversion.

    Supports complex layout analysis, table extraction, and preserves
    document structure across multiple formats.
    """

    def __init__(self):
        logger.info("Initializing Docling DocumentConverter...")
        try:
            self.converter = _build_converter()
        except ImportError:
            logger.error("docling not installed. Run 'pip install docling'")
            raise ImportError("Missing 'docling' library for DoclingService.")

    async def process_document(self, file_path: str) -> OcrResult:
        """
        Processes a document using Docling and maps it to OcrResult.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing {path.name} with Docling...")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self.converter.convert, str(path))
            doc = result.document
            return self._map_to_ocr_result(doc)
        except Exception as e:
            logger.error(f"Docling conversion failed: {str(e)}")
            raise RuntimeError(f"Docling error: {str(e)}")

    def _map_to_ocr_result(self, doc: Any) -> OcrResult:
        """
        Maps a DoclingDocument to our unified OcrResult schema.
        """
        blocks = []
        tables = []

        for item, _ in doc.iterate_items():
            label = getattr(item, "label", "text").value if hasattr(item.label, "value") else str(item.label)

            bbox = [0, 0, 0, 0]
            page_num = 1
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0]
                page_num = getattr(prov, "page_no", 1)
                if hasattr(prov, "bbox") and prov.bbox:
                    b = prov.bbox
                    bbox = [int(b.l), int(b.t), int(b.r), int(b.b)]

            table_markdown = None
            if label == "table":
                try:
                    table_markdown = item.export_to_markdown(doc)
                    tables.append(ExtractedMedia(
                        media_id=f"table_{len(tables)}",
                        media_type="table",
                        page_num=page_num,
                        content_markdown=table_markdown
                    ))
                except Exception:
                    logger.warning("Failed to export Docling table to markdown.")

            blocks.append(OcrBlock(
                text=getattr(item, "text", ""),
                type=label,
                bbox=bbox,
                page_num=page_num,
                table_markdown=table_markdown
            ))

        return OcrResult(
            blocks=blocks,
            full_text=doc.export_to_markdown(),
            markdown=doc.export_to_markdown(),
            tables=tables,
            page_count=len(doc.pages) if hasattr(doc, "pages") else (doc.page_count() if callable(getattr(doc, "page_count", None)) else (doc.num_pages() if callable(getattr(doc, "num_pages", None)) else 0)),
            metadata={"parser": "docling"}
        )
