"""Grounded answers, citations, and traces returned by the RAG Agent."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models.rag import RetrievedChunk


class RAGCitation(BaseModel):
    """Citation claimed by the generated answer."""

    chunk_id: str
    source: str
    page_number: int = Field(ge=1)


class RAGGeneratedAnswer(BaseModel):
    """Structured answer generated strictly from retrieved evidence."""

    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(min_length=1)
    citations: list[RAGCitation]
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def answered_response_requires_citation(self) -> "RAGGeneratedAnswer":
        """Prevent an apparently grounded answer with no cited evidence."""

        if self.status == "answered" and not self.citations:
            raise ValueError("An answered RAG response requires at least one citation.")
        if self.status == "insufficient_evidence" and self.citations:
            raise ValueError("An insufficient-evidence response cannot cite evidence.")
        return self


class RAGTraceEvent(BaseModel):
    """Observable RAG workflow event without private reasoning."""

    step: int = Field(ge=1)
    action: str
    parameters: dict[str, str] = Field(default_factory=dict)
    status: Literal["success", "failed"]


class RAGAgentResult(BaseModel):
    """Validated RAG answer plus the evidence used to produce it."""

    selected_agent: Literal["rag_agent"] = "rag_agent"
    question: str
    generated: RAGGeneratedAnswer
    retrieved_chunks: list[RetrievedChunk]
    citation_validation_passed: bool
    trace: list[RAGTraceEvent]
