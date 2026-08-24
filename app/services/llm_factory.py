"""Create the configured language-model client."""

from openai import AsyncOpenAI

from app.core.config import Settings
from app.services.llm_client import LLMClient
from app.services.ollama_llm_client import OllamaLLMClient
from app.services.openai_llm_client import OpenAILLMClient


def create_openai_llm_client(settings: Settings) -> OpenAILLMClient:
    """Build an OpenAI client without exposing the API key to other modules."""

    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required to use the OpenAI provider.")

    api_client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return OpenAILLMClient(client=api_client, model=settings.openai_model)


def create_llm_client(settings: Settings) -> LLMClient:
    """Create the configured local or hosted language-model provider."""

    if settings.llm_provider == "ollama":
        return OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    return create_openai_llm_client(settings)
