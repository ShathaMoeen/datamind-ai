from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: Literal["healthy"]
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Confirm that the API process is running."""

    return HealthResponse(
        status="healthy",
        service="DataMind AI",
        version="0.1.0",
    )