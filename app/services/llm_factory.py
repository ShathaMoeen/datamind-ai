"""Create the configured language-model client."""

from openai import AsyncOpenAI

from app.core.config import Settings
from app.services.openai_llm_client import OpenAILLMClient


def create_openai_llm_client(settings: Settings) -> OpenAILLMClient:
    """Build an OpenAI client without exposing the API key to other modules."""

    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required to use the OpenAI provider.")

    api_client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return OpenAILLMClient(client=api_client, model=settings.openai_model)
