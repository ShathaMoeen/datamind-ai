"""API input for the integrated DataMind AI workflow."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalysisRequest(BaseModel):
    """A natural-language question and its uploaded resource identifiers."""

    question: str = Field(min_length=3, max_length=2_000)
    dataset_id: str | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=10)
    language: Literal["ar", "en"] = "en"
    dashboard_mode: bool = True

    @model_validator(mode="after")
    def requires_an_uploaded_resource(self) -> "AnalysisRequest":
        if self.dataset_id is None and not self.document_ids:
            raise ValueError("Upload a dataset or document before running analysis.")
        return self
