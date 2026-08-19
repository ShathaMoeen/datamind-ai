"""Chroma-backed storage for document chunks and their embeddings."""

from pathlib import Path
from typing import Any

import chromadb

from app.models.rag import DocumentChunk, RetrievedChunk


class ChromaVectorStore:
    """Persist and query embeddings while preserving citation metadata."""

    def __init__(self, client: Any, collection_name: str = "datamind_documents") -> None:
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def persistent(cls, directory: Path) -> "ChromaVectorStore":
        """Create a store backed by a local persistent Chroma directory."""

        directory.mkdir(parents=True, exist_ok=True)
        return cls(chromadb.PersistentClient(path=str(directory.resolve())))

    def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or replace chunks and validated citation metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def delete_document(self, document_id: str) -> None:
        """Remove existing chunks before a document is re-indexed."""

        self._collection.delete(where={"document_id": document_id})

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return nearest chunks, optionally constrained to selected documents."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        where = (
            {"document_id": {"$in": document_ids}}
            if document_ids
            else None
        )
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=str(metadata["document_id"]),
                source=str(metadata["source"]),
                page_number=int(metadata["page_number"]),
                text=document,
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]
