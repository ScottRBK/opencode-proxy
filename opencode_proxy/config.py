"""Configuration settings for OpenCode Proxy."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Server settings use OPENCODE_PROXY_ prefix.
    Provider API keys use standard env var names (no prefix).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server (OPENCODE_PROXY_ prefix)
    HOST: str = Field(default="0.0.0.0", alias="OPENCODE_PROXY_HOST")
    PORT: int = Field(default=4141, alias="OPENCODE_PROXY_PORT")
    LOG_LEVEL: str = Field(default="INFO", alias="OPENCODE_PROXY_LOG_LEVEL")

    # Provider API keys (standard env var names)
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    TOGETHER_API_KEY: str | None = None
    DEEPINFRA_API_KEY: str | None = None
    PERPLEXITY_API_KEY: str | None = None
    XAI_API_KEY: str | None = None
    OPENCODE_API_KEY: str | None = None

    def get_api_key(self, env_var: str) -> str | None:
        """Get an API key by its env var name."""
        return getattr(self, env_var, None)


settings = Settings()
