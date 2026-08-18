"""Unit tests for the OpenAI LLM client adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.models.llm import LLMMessage, LLMRequest
from app.services.openai_llm_client import OpenAILLMClient


def test_openai_client_normalizes_response_without_network_call() -> None:
    """The adapter should translate requests and normalize provider responses."""

    api_client = AsyncMock()
    api_client.responses.create.return_value = SimpleNamespace(
        output_text="Sales increased by 12%.",
        model="test-model",
        usage=SimpleNamespace(input_tokens=10, output_tokens=6),
    )
    client = OpenAILLMClient(client=api_client, model="test-model")
    request = LLMRequest(
        messages=[
            LLMMessage(role="developer", content="Use calculated facts only."),
            LLMMessage(role="user", content="Summarize the sales trend."),
        ],
        max_output_tokens=500,
    )

    result = asyncio.run(client.generate(request))

    api_client.responses.create.assert_awaited_once_with(
        model="test-model",
        input=[
            {"role": "developer", "content": "Use calculated facts only."},
            {"role": "user", "content": "Summarize the sales trend."},
        ],
        max_output_tokens=500,
    )
    assert result.content == "Sales increased by 12%."
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 6
