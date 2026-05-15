"""
Hierarchical Chunker
====================
Chunks OCR output into parent chunks for embedding.
Now supports structural metadata (heading paths, page numbers)
and visual content injection (image/table captions appended to text).
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from docai.config import settings
from docai.ocr.base import OcrResult, ExtractedMedia


class DocumentChunk(BaseModel):
    """A single chunk ready for embedding and storage."""
    text: str
    heading: str = "Unknown"
    heading_path: List[str] = Field(default_factory=list)
    page_num: int = 1
    char_start: int = 0
    char_end: int = 0

    # Visual content refs (populated by injection step)
    image_refs: List[str] = Field(default_factory=list)     # media_ids of images
    table_refs: List[str] = Field(default_factory=list)     # media_ids of tables
    has_table: bool = False
    has_image: bool = False


class HierarchicalChunker:
    def __init__(self):
        self.parent_size = settings.CHUNK_PARENT_TOKENS
        # Character approximation (1 token ≈ 4 chars)
        self.parent_char_limit = self.parent_size * 4

    def chunk_document(self, ocr_result: OcrResult) -> List[DocumentChunk]:
        """
        Chunks OCR blocks into parent chunks, preserving structural metadata.
        After chunking, injects visual content (image/table captions) from
        the same page into the chunk text for vector searchability.
        """
        chunks = self._split_blocks(ocr_result)

        # Visual content injection
        if ocr_result.images or ocr_result.tables:
            chunks = self._inject_visual_content(
                chunks, ocr_result.images, ocr_result.tables
            )

        return chunks

    def _split_blocks(self, ocr_result: OcrResult) -> List[DocumentChunk]:
        """Split OCR blocks into chunks respecting token limits."""
        chunks = []
        current_text = ""
        current_heading = "Unknown"
        current_heading_path: List[str] = []
        char_start = 0
        current_page = 1
        current_has_table = False
        current_has_image = False

        for block in ocr_result.blocks:
            # Track page number unconditionally (fix: was only set on flush)
            current_page = block.page_num

            # Track heading hierarchy (heading text is metadata only, not body text)
            if block.type in ('title', 'heading'):
                current_heading = block.text
                if hasattr(block, 'heading_path') and block.heading_path:
                    current_heading_path = block.heading_path
                continue  # heading text is in metadata; do not add to chunk body

            # Track table/image presence
            if block.type == "table":
                current_has_table = True
            if block.type == "image":
                current_has_image = True

            # If adding this block would exceed the limit, flush the current chunk first
            if (len(current_text) + len(block.text) > self.parent_char_limit
                    and current_text.strip()):
                stripped = current_text.strip()
                chunks.append(DocumentChunk(
                    text=stripped,
                    heading=current_heading,
                    heading_path=list(current_heading_path),
                    page_num=current_page,
                    char_start=char_start,
                    char_end=char_start + len(stripped),
                    has_table=current_has_table,
                    has_image=current_has_image,
                ))
                char_start += len(current_text)  # advance by raw length (includes separators)
                current_text = ""
                current_has_table = False
                current_has_image = False

            current_text += block.text + "\n\n"

        # Add remainder
        if current_text.strip():
            stripped = current_text.strip()
            chunks.append(DocumentChunk(
                text=stripped,
                heading=current_heading,
                heading_path=list(current_heading_path),
                page_num=current_page,
                char_start=char_start,
                char_end=char_start + len(stripped),
                has_table=current_has_table,
                has_image=current_has_image,
            ))

        return chunks

    @staticmethod
    def _inject_visual_content(
        chunks: List[DocumentChunk],
        images: List[ExtractedMedia],
        tables: List[ExtractedMedia],
    ) -> List[DocumentChunk]:
        """
        Inject captions of images and tables into chunks from the same page.
        Each media item is assigned to the FIRST chunk on its page (avoids duplication).
        This makes visual content searchable via standard vector search.
        """
        # Build page → media lookups
        page_images: dict[int, list[ExtractedMedia]] = {}
        for img in images:
            page_images.setdefault(img.page_num, []).append(img)

        page_tables: dict[int, list[ExtractedMedia]] = {}
        for tbl in tables:
            page_tables.setdefault(tbl.page_num, []).append(tbl)

        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for chunk in chunks:
            page = chunk.page_num

            # Inject image captions
            if page in page_images:
                for img in page_images[page]:
                    if img.media_id not in assigned_images and img.caption:
                        chunk.image_refs.append(img.media_id)
                        chunk.text += f"\n\n[Image on page {img.page_num}]: {img.caption}"
                        chunk.has_image = True
                        assigned_images.add(img.media_id)

            # Inject table captions
            if page in page_tables:
                for tbl in page_tables[page]:
                    if tbl.media_id not in assigned_tables and tbl.caption:
                        chunk.table_refs.append(tbl.media_id)
                        dims = f"({tbl.num_rows}x{tbl.num_cols})" if tbl.num_rows else ""
                        chunk.text += f"\n\n[Table on page {tbl.page_num} {dims}]: {tbl.caption}"
                        chunk.has_table = True
                        assigned_tables.add(tbl.media_id)

        return chunks

