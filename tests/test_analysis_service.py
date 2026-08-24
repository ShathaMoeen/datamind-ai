"""Tests for document preparation before integrated analysis."""

import asyncio

from app.models.analysis_request import AnalysisRequest
from app.services.analysis_service import AnalysisService


class FakePipeline:
    def __init__(self) -> None:
        self.ingested: list[str] = []

    def ingest(self, document_id: str) -> None:
        self.ingested.append(document_id)


class FakeWorkflow:
    def __init__(self) -> None:
        self.received = None

    async def run(
        self,
        question: str,
        context: object,
        language: str = "en",
        dashboard_mode: bool = False,
    ) -> str:
        self.received = (question, context, language, dashboard_mode)
        return "completed"


def test_analysis_service_indexes_documents_before_workflow() -> None:
    pipeline = FakePipeline()
    workflow = FakeWorkflow()
    service = AnalysisService(  # type: ignore[arg-type]
        workflow=workflow,
        rag_pipeline=pipeline,
    )

    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                question="Explain the sales trend.",
                dataset_id="dataset-1",
                document_ids=["document-1", "document-2"],
            )
        )
    )

    assert result == "completed"
    assert pipeline.ingested == ["document-1", "document-2"]
    assert workflow.received[0] == "Explain the sales trend."
    assert workflow.received[1].dataset_id == "dataset-1"
    assert workflow.received[2] == "en"
    assert workflow.received[3] is True
