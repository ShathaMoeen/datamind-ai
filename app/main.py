from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes.analysis import router as analysis_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router

app = FastAPI(
    title="DataMind AI API",
    description="AI-powered data analysis platform.",
    version=__version__,
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")

static_directory = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_directory), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the lightweight DataMind AI browser interface."""

    return FileResponse(static_directory / "index.html")
