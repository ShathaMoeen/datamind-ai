"""Structured documents, chunks, retrieval results, and citations for RAG."""

from pathlib import Path

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Safe metadata returned after a PDF upload."""

    document_id: str
    original_filename: str
    size_bytes: int = Field(gt=0)


class StoredDocument(BaseModel):
    """Internal metadata needed to locate and cite a stored PDF."""

    document_id: str
    original_filename: str
    size_bytes: int = Field(gt=0)
    path: Path


class DocumentPage(BaseModel):
    """Text extracted from one source page."""

    document_id: str
    source: str
    page_number: int = Field(ge=1)
    text: str


class DocumentChunk(BaseModel):
    """Overlapping text passage with citation metadata."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """One semantically retrieved passage and its citation metadata."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int = Field(ge=1)
    text: str
    distance: float


class RAGIngestionResult(BaseModel):
    """Counts produced after extracting and indexing a document."""

    document_id: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
