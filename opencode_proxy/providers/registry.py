"""Provider registry and model resolution."""

import logging
from dataclasses import dataclass

from opencode_proxy.config import settings
from opencode_proxy.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderDefinition:
    """Static definition of a provider."""

    base_url: str
    env_var: str
    format: str  # "openai" or "anthropic"


PROVIDER_DEFINITIONS: dict[str, ProviderDefinition] = {
    "openai": ProviderDefinition(
        base_url="https://api.openai.com/v1",
        env_var="OPENAI_API_KEY",
        format="openai",
    ),
    "anthropic": ProviderDefinition(
        base_url="https://api.anthropic.com",
        env_var="ANTHROPIC_API_KEY",
        format="anthropic",
    ),
    "groq": ProviderDefinition(
        base_url="https://api.groq.com/openai/v1",
        env_var="GROQ_API_KEY",
        format="openai",
    ),
    "mistral": ProviderDefinition(
        base_url="https://api.mistral.ai/v1",
        env_var="MISTRAL_API_KEY",
        format="openai",
    ),
    "openrouter": ProviderDefinition(
        base_url="https://openrouter.ai/api/v1",
        env_var="OPENROUTER_API_KEY",
        format="openai",
    ),
    "togetherai": ProviderDefinition(
        base_url="https://api.together.xyz/v1",
        env_var="TOGETHER_API_KEY",
        format="openai",
    ),
    "deepinfra": ProviderDefinition(
        base_url="https://api.deepinfra.com/v1/openai",
        env_var="DEEPINFRA_API_KEY",
        format="openai",
    ),
    "perplexity": ProviderDefinition(
        base_url="https://api.perplexity.ai",
        env_var="PERPLEXITY_API_KEY",
        format="openai",
    ),
    "xai": ProviderDefinition(
        base_url="https://api.x.ai/v1",
        env_var="XAI_API_KEY",
        format="openai",
    ),
    "opencode": ProviderDefinition(
        base_url="https://opencode.ai/zen/v1",
        env_var="OPENCODE_API_KEY",
        format="openai",
    ),
}

# Bare model name -> provider/model-id
MODEL_ALIASES: dict[str, str] = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "o3-mini": "openai/o3-mini",
    # Anthropic
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514",
    "claude-haiku-4-20250414": "anthropic/claude-haiku-4-20250414",
    # Groq
    "llama-3.3-70b-versatile": "groq/llama-3.3-70b-versatile",
    # OpenCode Zen (free)
    "big-pickle": "opencode/big-pickle",
}

# Known models per provider (for /v1/models endpoint)
PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-20250414",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
    ],
    "openrouter": [],  # Dynamic - too many to list
    "togetherai": [],
    "deepinfra": [],
    "perplexity": [
        "sonar",
        "sonar-pro",
    ],
    "xai": [
        "grok-2",
        "grok-3-mini",
    ],
    "opencode": [
        "big-pickle",
        "qwen3-coder",
    ],
}


class ProviderRegistry:
    """Manages provider instances and model resolution."""

    def __init__(self) -> None:
        self._instances: dict[str, BaseProvider] = {}

    def resolve_model(self, model_str: str) -> tuple[str, str]:
        """Resolve a model string to (provider_id, model_id).

        Accepts:
          - "provider/model-id" (e.g. "openai/gpt-4o")
          - "provider/org/model-id" (e.g. "openrouter/anthropic/claude-sonnet-4")
          - bare alias (e.g. "gpt-4o" -> "openai/gpt-4o")

        Raises ValueError if provider is unknown.
        """
        # Check aliases first
        if model_str in MODEL_ALIASES:
            model_str = MODEL_ALIASES[model_str]

        provider_id, sep, model_id = model_str.partition("/")
        if not sep or not model_id:
            raise ValueError(
                f"Unknown model '{model_str}'. Use 'provider/model-id' format "
                f"or a known alias. Available aliases: {', '.join(sorted(MODEL_ALIASES))}"
            )

        if provider_id not in PROVIDER_DEFINITIONS:
            raise ValueError(
                f"Unknown provider '{provider_id}'. "
                f"Available: {', '.join(sorted(PROVIDER_DEFINITIONS))}"
            )

        return provider_id, model_id

    def get_provider(self, provider_id: str) -> BaseProvider:
        """Get or create a provider instance.

        Raises ValueError if API key is not set.
        """
        if provider_id in self._instances:
            return self._instances[provider_id]

        defn = PROVIDER_DEFINITIONS.get(provider_id)
        if not defn:
            raise ValueError(f"Unknown provider: {provider_id}")

        api_key = settings.get_api_key(defn.env_var)
        if not api_key:
            raise ValueError(
                f"API key not configured for {provider_id}. "
                f"Set the {defn.env_var} environment variable."
            )

        if defn.format == "anthropic":
            from opencode_proxy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(base_url=defn.base_url, api_key=api_key)
        else:
            from opencode_proxy.providers.openai_compat import OpenAICompatProvider

            provider = OpenAICompatProvider(base_url=defn.base_url, api_key=api_key)

        self._instances[provider_id] = provider
        return provider

    def available_providers(self) -> list[str]:
        """Return provider IDs that have API keys configured."""
        return [
            pid
            for pid, defn in PROVIDER_DEFINITIONS.items()
            if settings.get_api_key(defn.env_var)
        ]

    def list_models(self) -> list[tuple[str, str]]:
        """Return (provider_id/model_id, provider_id) for all available providers."""
        models = []
        for pid in self.available_providers():
            for model_id in PROVIDER_MODELS.get(pid, []):
                models.append((f"{pid}/{model_id}", pid))
        return models

    async def close_all(self) -> None:
        """Close all provider connections."""
        for provider in self._instances.values():
            await provider.close()
        self._instances.clear()
