"""Local Ollama implementation of the provider-independent LLM contract."""

from typing import Any

import httpx

from app.models.llm import LLMRequest, LLMResponse, TokenUsage


class OllamaUnavailableError(ConnectionError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaLLMClient:
    """Generate structured JSON through Ollama without a paid API key."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Send normalized messages and return one non-streaming JSON response."""

        options: dict[str, Any] = {"num_predict": request.max_output_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": (
                        "system" if message.role == "developer" else message.role
                    ),
                    "content": message.content,
                }
                for message in request.messages
            ],
            "format": request.response_schema or "json",
            "stream": False,
            "think": False,
            "options": options,
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ReadTimeout as error:
            raise OllamaUnavailableError(
                "Ollama exceeded the local generation time limit. Try again or "
                "use a smaller/faster model."
            ) from error
        except httpx.ConnectError as error:
            raise OllamaUnavailableError(
                "Ollama is not running. Start Ollama and pull the configured model."
            ) from error
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:300]
            raise OllamaUnavailableError(
                f"Ollama rejected the request: {detail}"
            ) from error

        body = response.json()
        try:
            content = str(body["message"]["content"])
        except (KeyError, TypeError) as error:
            raise ValueError("Ollama returned an invalid chat response.") from error
        return LLMResponse(
            content=content,
            model=str(body.get("model", self._model)),
            usage=TokenUsage(
                input_tokens=int(body.get("prompt_eval_count", 0)),
                output_tokens=int(body.get("eval_count", 0)),
            ),
        )
