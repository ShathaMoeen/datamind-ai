"""End-to-end ingestion test with deterministic local test components."""

import asyncio
from io import BytesIO

import chromadb
from fastapi import UploadFile

from app.models.rag import DocumentPage
from app.rag.chunking import TextChunker
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import ChromaVectorStore
from app.services.document_service import DocumentService


class FakeDocumentLoader:
    """Return known extracted pages without parsing a real PDF."""

    def load(self, document) -> list[DocumentPage]:
        return [
            DocumentPage(
                document_id=document.document_id,
                source=document.original_filename,
                page_number=1,
                text="Shipping delays reduced sales in the western region.",
            )
        ]


class FakeEmbeddingClient:
    """Return deterministic vectors without a model download."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_pipeline_indexes_extracted_chunks(tmp_path) -> None:
    """Upload metadata should flow through extraction into Chroma."""

    document_service = DocumentService(tmp_path / "documents", 1_000)
    upload = UploadFile(
        filename="report.pdf",
        file=BytesIO(b"%PDF-1.4\nmock"),
    )
    uploaded = asyncio.run(document_service.save(upload))
    store = ChromaVectorStore(chromadb.EphemeralClient(), "pipeline_test")
    pipeline = RAGPipeline(
        document_service=document_service,
        document_loader=FakeDocumentLoader(),
        chunker=TextChunker(chunk_size_words=5, overlap_words=1),
        embedding_client=FakeEmbeddingClient(),
        vector_store=store,
    )

    result = pipeline.ingest(uploaded.document_id)

    assert result.page_count == 1
    assert result.chunk_count == 2
    indexed = store.query([10.0, 1.0], top_k=2)
    assert len(indexed) == 2
    assert all(chunk.source == "report.pdf" for chunk in indexed)
