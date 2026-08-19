"""Environment-based application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load validated configuration from environment variables or `.env`."""

    app_name: str = "DataMind AI"
    app_env: str = "development"
    debug: bool = False
    dataset_upload_directory: Path = Path("data/uploads")
    max_upload_size_mb: int = 10
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-nano"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""

    return Settings()
