"""Tests for grounded RAG generation and citation validation."""

import asyncio

import pytest

from app.agents.rag_agent import RAGAgent, RAGAnswerValidationError
from app.models.llm import LLMResponse
from app.models.rag import RetrievedChunk
from app.services.fake_llm_client import FakeLLMClient


class FakeRetriever:
    """Return configured evidence without a vector database."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, question, top_k=5, document_ids=None):
        return self._chunks[:top_k]


def _evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="doc-1:p2:c0",
            document_id="doc-1",
            source="sales-report.pdf",
            page_number=2,
            text="Shipping delays contributed to lower western sales.",
            distance=0.1,
        )
    ]


def test_rag_agent_returns_answer_with_verified_citation() -> None:
    """A citation matching retrieved metadata should pass validation."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"status":"answered","answer":"Shipping delays contributed '
                'to the decline.","citations":[{"chunk_id":"doc-1:p2:c0",'
                '"source":"sales-report.pdf","page_number":2}],'
                '"limitations":["The evidence indicates contribution, not proof '
                'of sole causation."]}'
            ),
            model="fake-model",
        )
    )
    agent = RAGAgent(FakeRetriever(_evidence()), llm_client, top_k=3)

    result = asyncio.run(agent.answer("Why did western sales decline?"))

    assert result.generated.status == "answered"
    assert result.generated.citations[0].page_number == 2
    assert result.citation_validation_passed is True
    assert len(result.trace) == 3
    assert "untrusted data" in llm_client.requests[0].messages[0].content


def test_rag_agent_rejects_hallucinated_citation() -> None:
    """A citation not present in retrieval results must be rejected."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"status":"answered","answer":"Unsupported answer.",'
                '"citations":[{"chunk_id":"invented:p99:c0",'
                '"source":"fake.pdf","page_number":99}],"limitations":[]}'
            ),
            model="fake-model",
        )
    )
    agent = RAGAgent(FakeRetriever(_evidence()), llm_client)

    with pytest.raises(RAGAnswerValidationError):
        asyncio.run(agent.answer("What happened?"))


def test_rag_agent_skips_llm_when_no_evidence_is_retrieved() -> None:
    """No evidence should return a deterministic insufficient result."""

    llm_client = FakeLLMClient(
        LLMResponse(content="This must not be used.", model="fake-model")
    )
    agent = RAGAgent(FakeRetriever([]), llm_client)

    result = asyncio.run(agent.answer("What does the document say?"))

    assert result.generated.status == "insufficient_evidence"
    assert result.generated.citations == []
    assert llm_client.requests == []
