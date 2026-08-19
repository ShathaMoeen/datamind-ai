"""Semantic retrieval using compatible query embeddings and Chroma."""

from app.models.rag import RetrievedChunk
from app.rag.embeddings import EmbeddingClient
from app.rag.vector_store import ChromaVectorStore


class RAGRetriever:
    """Embed a question and retrieve top-k citation-bearing chunks."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return semantically nearest chunks for a non-empty question."""

        if not question.strip():
            raise ValueError("A retrieval question cannot be empty.")
        query_embedding = self._embedding_client.embed_query(question)
        return self._vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
        )
