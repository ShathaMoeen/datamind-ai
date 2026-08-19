"""End-to-end PDF extraction, chunking, embedding, and indexing."""

from app.models.rag import RAGIngestionResult
from app.rag.chunking import TextChunker
from app.rag.document_loader import PDFDocumentLoader
from app.rag.embeddings import EmbeddingClient
from app.rag.vector_store import ChromaVectorStore
from app.services.document_service import DocumentService


class RAGPipeline:
    """Coordinate deterministic ingestion components without an LLM call."""

    def __init__(
        self,
        document_service: DocumentService,
        document_loader: PDFDocumentLoader,
        chunker: TextChunker,
        embedding_client: EmbeddingClient,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._document_service = document_service
        self._document_loader = document_loader
        self._chunker = chunker
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def ingest(self, document_id: str) -> RAGIngestionResult:
        """Extract, chunk, embed, and replace one document's index entries."""

        document = self._document_service.get(document_id)
        pages = self._document_loader.load(document)
        chunks = self._chunker.split(pages)
        if not chunks:
            raise ValueError("No indexable chunks were produced from the PDF.")
        embeddings = self._embedding_client.embed_documents(
            [chunk.text for chunk in chunks]
        )
        self._vector_store.delete_document(document_id)
        self._vector_store.upsert(chunks, embeddings)
        return RAGIngestionResult(
            document_id=document_id,
            page_count=len(pages),
            chunk_count=len(chunks),
        )
