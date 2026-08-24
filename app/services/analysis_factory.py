"""Construct the production analysis dependency graph once per process."""

from functools import lru_cache

from app.agents.analysis_agent import AnalysisAgent
from app.agents.data_agent import DataAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rag_agent import RAGAgent
from app.agents.report_agent import ReportAgent
from app.agents.visualization_agent import VisualizationAgent
from app.core.config import get_settings
from app.rag.chunking import TextChunker
from app.rag.document_loader import PDFDocumentLoader
from app.rag.embeddings import SentenceTransformerEmbeddingClient
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore
from app.services.agent_workflow import AgentWorkflow
from app.services.analysis_service import AnalysisService
from app.services.dataset_loader import DatasetLoader
from app.services.document_service import DocumentService
from app.services.llm_factory import create_llm_client
from app.tools.analysis_tool_registry import AnalysisToolRegistry
from app.tools.data_tool_registry import DataToolRegistry
from app.tools.visualization_tool_registry import VisualizationToolRegistry


@lru_cache
def get_analysis_service() -> AnalysisService:
    """Create and cache shared LLM, embedding, vector, and agent dependencies."""

    settings = get_settings()
    llm_client = create_llm_client(settings)
    dataset_loader = DatasetLoader(settings.dataset_upload_directory)
    embedding_client = SentenceTransformerEmbeddingClient(settings.embedding_model)
    vector_store = ChromaVectorStore.persistent(settings.vector_store_directory)
    document_service = DocumentService(
        settings.document_upload_directory,
        settings.max_document_upload_size_mb * 1024 * 1024,
    )
    rag_pipeline = RAGPipeline(
        document_service=document_service,
        document_loader=PDFDocumentLoader(),
        chunker=TextChunker(
            settings.rag_chunk_size_words,
            settings.rag_chunk_overlap_words,
        ),
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    workflow = AgentWorkflow(
        orchestrator=OrchestratorAgent(llm_client),
        data_agent=DataAgent(llm_client, DataToolRegistry(dataset_loader)),
        analysis_agent=AnalysisAgent(
            llm_client,
            AnalysisToolRegistry(dataset_loader),
        ),
        visualization_agent=VisualizationAgent(
            llm_client,
            VisualizationToolRegistry(dataset_loader),
        ),
        rag_agent=RAGAgent(
            RAGRetriever(embedding_client, vector_store),
            llm_client,
        ),
        report_agent=ReportAgent(llm_client),
    )
    return AnalysisService(workflow, rag_pipeline)
