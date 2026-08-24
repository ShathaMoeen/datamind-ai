"""Application service joining document ingestion and agent execution."""

import anyio

from app.models.analysis_request import AnalysisRequest
from app.models.orchestrator import OrchestrationContext
from app.models.workflow import AgentWorkflowResult
from app.rag.pipeline import RAGPipeline
from app.services.agent_workflow import AgentWorkflow


class AnalysisService:
    """Index selected documents, then run the integrated agent workflow."""

    def __init__(self, workflow: AgentWorkflow, rag_pipeline: RAGPipeline) -> None:
        self._workflow = workflow
        self._rag_pipeline = rag_pipeline

    async def analyze(self, request: AnalysisRequest) -> AgentWorkflowResult:
        """Prepare document evidence and execute one user analysis request."""

        for document_id in request.document_ids:
            await anyio.to_thread.run_sync(self._rag_pipeline.ingest, document_id)

        return await self._workflow.run(
            request.question,
            OrchestrationContext(
                dataset_id=request.dataset_id,
                document_ids=request.document_ids,
            ),
            language=request.language,
            dashboard_mode=request.dashboard_mode,
        )
