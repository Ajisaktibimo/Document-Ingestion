"""
Local Captioner
===============
Generates captions for images and tables using the local Ollama LLM.
No external API required.

- Table captioning: text-only LLM call (works with any model).
- Image captioning: multimodal call (requires a vision model like llava or gemma4).
"""
import base64
import logging
from pathlib import Path
from typing import List, Optional

import ollama

from docai.config import settings
from docai.ocr.base import ExtractedMedia

logger = logging.getLogger(__name__)

# ── Prompt templates (adapted from NexusRAG, proven effective) ──────────

TABLE_CAPTION_PROMPT = (
    "You are a document analysis assistant. Given a markdown table, "
    "write a concise description that covers:\n"
    "- The purpose/topic of the table\n"
    "- Key column names and what they represent\n"
    "- Notable values, trends, or outliers\n\n"
    "RULES:\n"
    "- Write 2-4 sentences, max 500 characters.\n"
    "- Be factual — describe only what is in the table.\n"
    "- Write in the SAME LANGUAGE as the table content.\n\n"
    "Table:\n"
)

IMAGE_CAPTION_PROMPT = (
    "Describe ONLY what you can directly see in this image. "
    "Do NOT infer, assume, or add any information not visible.\n\n"
    "Include:\n"
    "- Type of visual (chart, table, diagram, photo, screenshot, etc.)\n"
    "- ALL specific numbers, percentages, and labels that are VISIBLE\n"
    "- Axis labels, legend text, and category names exactly as shown\n"
    "- Trends or comparisons that are visually obvious\n\n"
    "RULES:\n"
    "- Write 2-4 concise sentences, max 400 characters.\n"
    "- Do NOT start with 'This image shows' or 'Here is'.\n"
    "- Write in the SAME LANGUAGE as any text visible in the image.\n"
)


class LocalCaptioner:
    """
    Generates captions for extracted media using local Ollama.

    By default, everything reuses the SAME ``OLLAMA_MODEL`` that powers chat.
    If a dedicated vision model is needed (e.g. for better accuracy), it can
    be configured via ``CAPTION_VISION_MODEL``.
    """

    def __init__(
        self,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ):
        self.text_model = text_model or settings.OLLAMA_MODEL
        # Use CAPTION_VISION_MODEL if set, otherwise fallback to text_model
        self.vision_model = vision_model or settings.CAPTION_VISION_MODEL or self.text_model
        self.think = settings.INGESTION_OLLAMA_THINK

        self.client = ollama.AsyncClient(
            host=ollama_host or settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
        logger.info(
            f"LocalCaptioner ready — text: {self.text_model}, "
            f"vision: {self.vision_model} "
            f"{'(shared)' if self.text_model == self.vision_model else '(separate)'}, "
            f"think={self.think}"
        )

    @property
    def supports_vision(self) -> bool:
        # Always True now because we fallback to the main model
        return True

    # ── Table captioning (text-only) ────────────────────────────────────

    async def caption_table(self, table_markdown: str) -> str:
        """Generate a concise caption for a Markdown table."""
        truncated = table_markdown[:settings.MAX_TABLE_MARKDOWN_CHARS]
        if len(table_markdown) > settings.MAX_TABLE_MARKDOWN_CHARS:
            truncated += "\n... (truncated)"

        try:
            response = await self.client.chat(
                model=self.text_model,
                messages=[{
                    "role": "user",
                    "content": TABLE_CAPTION_PROMPT + truncated,
                }],
                options={
                    "temperature": 0.1,
                    "num_predict": settings.OLLAMA_NUM_PREDICT,
                    "num_ctx": settings.OLLAMA_NUM_CTX,
                },
                think=self.think,
                keep_alive=settings.OLLAMA_KEEP_ALIVE or None,
            )
            caption = response["message"]["content"].strip()
            return " ".join(caption.split())[:500]
        except Exception as e:
            logger.warning(f"Table captioning failed: {e}")
            return ""

    # ── Image captioning (multimodal) ───────────────────────────────────

    async def caption_image(self, image_path: str) -> str:
        """Generate a caption for an image using a vision-capable model."""
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image not found for captioning: {path}")
            return ""

        try:
            image_bytes = path.read_bytes()
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            response = await self.client.chat(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": IMAGE_CAPTION_PROMPT,
                    "images": [b64_image],
                }],
                options={
                    "temperature": 0.1,
                    "num_predict": settings.OLLAMA_NUM_PREDICT,
                    "num_ctx": settings.OLLAMA_NUM_CTX,
                },
                think=self.think,
                keep_alive=settings.OLLAMA_KEEP_ALIVE or None,
            )
            caption = response["message"]["content"].strip()
            return " ".join(caption.split())[:500]
        except Exception as e:
            logger.warning(f"Image captioning failed for {path.name}: {e}")
            return ""

    # ── Batch captioning ────────────────────────────────────────────────

    async def caption_all(self, media_list: List[ExtractedMedia]) -> None:
        """
        Caption all extracted media items in-place.
        Skips items that already have a caption.
        """
        for media in media_list:
            if media.caption:
                continue

            if media.media_type == "table" and media.content_markdown:
                if settings.ENABLE_TABLE_CAPTIONING:
                    media.caption = await self.caption_table(media.content_markdown)
                    logger.debug(f"Captioned table {media.media_id}: {media.caption[:80]}...")

            elif media.media_type == "image" and media.file_path:
                if settings.ENABLE_IMAGE_CAPTIONING:
                    media.caption = await self.caption_image(media.file_path)
                    logger.debug(f"Captioned image {media.media_id}: {media.caption[:80]}...")

