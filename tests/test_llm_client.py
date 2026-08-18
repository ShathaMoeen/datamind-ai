"""Tests for the provider-independent LLM client contract."""

import asyncio

import pytest
from pydantic import ValidationError

from app.models.llm import LLMMessage, LLMRequest, LLMResponse, TokenUsage
from app.services.fake_llm_client import FakeLLMClient


def test_fake_client_returns_configured_response() -> None:
    """The fake client should be deterministic and record its request."""

    request = LLMRequest(
        messages=[LLMMessage(role="user", content="Summarize this dataset.")]
    )
    expected = LLMResponse(
        content="The dataset contains a sales trend.",
        model="fake-model",
        usage=TokenUsage(input_tokens=5, output_tokens=7),
    )
    client = FakeLLMClient(response=expected)

    result = asyncio.run(client.generate(request))

    assert result == expected
    assert client.requests == [request]


def test_llm_request_requires_at_least_one_message() -> None:
    """An empty message list should fail validation before any API call."""

    with pytest.raises(ValidationError):
        LLMRequest(messages=[])
