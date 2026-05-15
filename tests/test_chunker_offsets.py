import pytest
from docai.ingestion.chunker import HierarchicalChunker


def _block(text, page=1, btype="text"):
    """Create a minimal Block-like object for testing."""
    from docai.ocr.base import OcrBlock
    return OcrBlock(text=text, type=btype, page_num=page)


def _make_ocr(blocks):
    from docai.ocr.base import OcrResult
    return OcrResult(
        blocks=blocks, images=[], tables=[],
        page_count=len(set(b.page_num for b in blocks)) if blocks else 1,
        full_text="".join(b.text for b in blocks),
        metadata={},
        markdown="",
    )


def test_char_end_matches_stripped_text_length():
    """char_end must equal char_start + len(chunk.text) — no trailing-newline inflation."""
    chunker = HierarchicalChunker()
    ocr = _make_ocr([_block("hello world"), _block("second block")])
    chunks = chunker.chunk_document(ocr)
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.char_end == chunk.char_start + len(chunk.text), (
        f"char_end={chunk.char_end} != char_start({chunk.char_start}) + len(text)({len(chunk.text)})"
    )


def test_has_table_true_when_table_block_present():
    """has_table must be True for chunks that contain a table block."""
    chunker = HierarchicalChunker()
    ocr = _make_ocr([_block("some text"), _block("| col1 | col2 |", btype="table")])
    chunks = chunker.chunk_document(ocr)
    assert len(chunks) >= 1
    assert chunks[0].has_table is True, "chunk containing a table block must have has_table=True"


def test_page_num_reflects_block_page_not_always_one():
    """page_num on a single-chunk document must reflect the actual block page, not always 1."""
    chunker = HierarchicalChunker()
    ocr = _make_ocr([_block("content on page 3", page=3)])
    chunks = chunker.chunk_document(ocr)
    assert len(chunks) >= 1
    assert chunks[0].page_num == 3, f"expected page_num=3, got {chunks[0].page_num}"
