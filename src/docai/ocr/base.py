from typing import Protocol, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class OcrBlock(BaseModel):
    """A single recognized region from a document page."""
    text: str
    type: str                                   # 'text', 'title', 'heading', 'figure', 'table'
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    page_num: int = 1

    # Structural metadata (populated by Marker / Docling parsers)
    heading_path: List[str] = Field(default_factory=list)   # e.g. ["Chapter 1", "Section 1.2"]
    image_id: Optional[str] = None              # UUID for extracted image file
    image_path: Optional[str] = None            # Path to extracted image on disk
    table_markdown: Optional[str] = None        # Raw Markdown for extracted tables


class ExtractedMedia(BaseModel):
    """An image or table extracted during parsing."""
    media_id: str                               # UUID
    media_type: str                             # "image" or "table"
    page_num: int = 1
    file_path: Optional[str] = None             # Disk path (images only)
    content_markdown: Optional[str] = None      # Raw Markdown (tables only)
    caption: str = ""                           # LLM-generated caption
    width: int = 0
    height: int = 0
    num_rows: int = 0                           # Tables only
    num_cols: int = 0                           # Tables only


class OcrResult(BaseModel):
    """Unified output from any OCR / document parser."""
    blocks: List[OcrBlock]
    full_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Structural extras
    page_count: int = 0
    images: List[ExtractedMedia] = Field(default_factory=list)
    tables: List[ExtractedMedia] = Field(default_factory=list)
    markdown: str = ""                          # Full structured Markdown (if available)


class OcrServiceProtocol(Protocol):
    """Interface that all parsers (Paddle, Marker, Chandra, …) must satisfy."""

    async def process_document(self, file_path: str) -> OcrResult:
        """Extracts structured text and layout from a document."""
        ...

