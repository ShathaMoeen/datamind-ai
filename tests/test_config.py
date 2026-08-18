"""Tests for application configuration."""

from app.core.config import Settings


def test_settings_do_not_require_api_key_for_local_tests() -> None:
    """Application settings should load without a real external credential."""

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-5.4-nano"
