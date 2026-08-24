"""Tests for the free local Ollama LLM provider."""

import asyncio
import json

import httpx
import pytest

from app.models.llm import LLMMessage, LLMRequest
from app.services.ollama_llm_client import OllamaLLMClient, OllamaUnavailableError


def test_ollama_client_requests_non_streaming_json() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen3:4b",
                "message": {"role": "assistant", "content": '{"ok":true}'},
                "prompt_eval_count": 20,
                "eval_count": 5,
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )
    client = OllamaLLMClient(
        base_url="http://ollama.test",
        model="qwen3:4b",
        client=http_client,
    )
    request = LLMRequest(
        messages=[LLMMessage(role="developer", content="Return JSON.")],
        temperature=0.0,
        max_output_tokens=100,
    )

    response = asyncio.run(client.generate(request))
    asyncio.run(http_client.aclose())

    assert captured["format"] == "json"
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["messages"][0]["role"] == "system"
    assert response.content == '{"ok":true}'
    assert response.usage.output_tokens == 5


def test_ollama_client_forwards_json_schema() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"model": "qwen3:1.7b", "message": {"content": '{"ok":true}'}},
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )
    client = OllamaLLMClient(
        base_url="http://ollama.test",
        model="qwen3:1.7b",
        client=http_client,
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    asyncio.run(
        client.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content="Return status.")],
                response_schema=schema,
            )
        )
    )
    asyncio.run(http_client.aclose())

    assert captured["format"] == schema


def test_ollama_client_converts_read_timeout_to_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Local generation timed out.", request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.test",
    )
    client = OllamaLLMClient(
        base_url="http://ollama.test",
        model="qwen3:1.7b",
        client=http_client,
    )

    with pytest.raises(OllamaUnavailableError, match="generation time limit"):
        asyncio.run(
            client.generate(
                LLMRequest(
                    messages=[LLMMessage(role="user", content="Analyze data.")]
                )
            )
        )
    asyncio.run(http_client.aclose())
