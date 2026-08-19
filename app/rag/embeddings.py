"""Provider abstraction and local Sentence Transformers embeddings."""

from typing import Protocol

from sentence_transformers import SentenceTransformer


class EmbeddingClient(Protocol):
    """Interface shared by local and future hosted embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages for storage."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query in the compatible vector space."""
        ...


class SentenceTransformerEmbeddingClient:
    """Local embeddings optimized separately for documents and queries."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return normalized local document embeddings."""

        embeddings = self._model.encode_document(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Return one normalized local query embedding."""

        embedding = self._model.encode_query(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()
