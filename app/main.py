from fastapi import FastAPI

from app import __version__
from app.api.routes.health import router as health_router

app = FastAPI(
    title="DataMind AI API",
    description="AI-powered data analysis platform.",
    version=__version__,
)

app.include_router(health_router, prefix="/api/v1")