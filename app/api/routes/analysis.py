"""HTTP endpoint for the integrated multi-agent analysis workflow."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.analysis_request import AnalysisRequest
from app.models.workflow import AgentWorkflowResult
from app.services.analysis_factory import get_analysis_service
from app.services.analysis_service import AnalysisService
from app.services.ollama_llm_client import OllamaUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])


def provide_analysis_service() -> AnalysisService:
    """Resolve production dependencies with a user-safe configuration error."""

    try:
        return get_analysis_service()
    except ValueError as error:
        if "OPENAI_API_KEY" in str(error):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configure OPENAI_API_KEY in .env to run AI analysis.",
            ) from error
        raise


@router.post("/run", response_model=AgentWorkflowResult)
async def run_analysis(
    request: AnalysisRequest,
    service: Annotated[AnalysisService, Depends(provide_analysis_service)],
) -> AgentWorkflowResult:
    """Run a validated multi-agent request against uploaded resources."""

    try:
        return await service.analyze(request)
    except OllamaUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except ValueError as error:
        if "OPENAI_API_KEY" in str(error):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configure OPENAI_API_KEY in .env to run AI analysis.",
            ) from error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("The integrated analysis workflow failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI analysis service could not complete the request.",
        ) from error
