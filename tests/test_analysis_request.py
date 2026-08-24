"""Validation tests for integrated analysis API input."""

import pytest
from pydantic import ValidationError

from app.models.analysis_request import AnalysisRequest


def test_analysis_request_requires_uploaded_resource() -> None:
    with pytest.raises(ValidationError, match="Upload a dataset or document"):
        AnalysisRequest(question="Analyze this data.")


def test_analysis_request_accepts_dataset_context() -> None:
    request = AnalysisRequest(
        question="Analyze this data.",
        dataset_id="dataset-1",
    )

    assert request.dataset_id == "dataset-1"


def test_analysis_request_accepts_arabic_output_language() -> None:
    request = AnalysisRequest(
        question="حلل هذه البيانات.",
        dataset_id="dataset-1",
        language="ar",
    )

    assert request.language == "ar"
    assert request.dashboard_mode is True
