"""Transparent word-based chunking with page-level citations."""

import re

from app.models.rag import DocumentChunk, DocumentPage


class TextChunker:
    """Split page text into overlapping word windows."""

    def __init__(self, chunk_size_words: int, overlap_words: int) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive.")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be between 0 and chunk size.")
        self._chunk_size_words = chunk_size_words
        self._overlap_words = overlap_words

    def split(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        """Create stable chunks without mixing text from different pages."""

        chunks = []
        step = self._chunk_size_words - self._overlap_words
        for page in pages:
            words = re.sub(r"\s+", " ", page.text).strip().split(" ")
            for chunk_index, start in enumerate(range(0, len(words), step)):
                text = " ".join(words[start : start + self._chunk_size_words]).strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{page.document_id}:p{page.page_number}:c{chunk_index}"
                        ),
                        document_id=page.document_id,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=text,
                    )
                )
                if start + self._chunk_size_words >= len(words):
                    break
        return chunks
