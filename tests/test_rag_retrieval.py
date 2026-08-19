"""Integration tests for Chroma storage and semantic retrieval wiring."""

import chromadb

from app.models.rag import DocumentChunk
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore


class FakeEmbeddingClient:
    """Small deterministic embeddings for retrieval tests."""

    _terms = ("sales", "shipping", "policy")

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(term)) for term in self._terms]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_retriever_returns_relevant_chunk_with_citation() -> None:
    """The nearest chunk should retain source and page citation metadata."""

    embedding_client = FakeEmbeddingClient()
    store = ChromaVectorStore(chromadb.EphemeralClient(), "retrieval_test")
    chunks = [
        DocumentChunk(
            chunk_id="doc-1:p1:c0",
            document_id="doc-1",
            source="report.pdf",
            page_number=1,
            chunk_index=0,
            text="Sales decreased after shipping delays.",
        ),
        DocumentChunk(
            chunk_id="doc-1:p2:c0",
            document_id="doc-1",
            source="report.pdf",
            page_number=2,
            chunk_index=0,
            text="The policy describes employee leave.",
        ),
    ]
    store.upsert(chunks, embedding_client.embed_documents([c.text for c in chunks]))
    retriever = RAGRetriever(embedding_client, store)

    results = retriever.retrieve("What caused the sales shipping issue?", top_k=1)

    assert results[0].chunk_id == "doc-1:p1:c0"
    assert results[0].source == "report.pdf"
    assert results[0].page_number == 1
