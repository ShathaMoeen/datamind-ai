"""Tests for page-aware overlapping RAG chunking."""

from app.models.rag import DocumentPage
from app.rag.chunking import TextChunker


def test_chunker_preserves_page_metadata_and_overlap() -> None:
    """Chunks should overlap without mixing citations across pages."""

    page = DocumentPage(
        document_id="doc-1",
        source="report.pdf",
        page_number=3,
        text="one two three four five six seven",
    )

    chunks = TextChunker(chunk_size_words=4, overlap_words=1).split([page])

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert chunks[0].chunk_id == "doc-1:p3:c0"
