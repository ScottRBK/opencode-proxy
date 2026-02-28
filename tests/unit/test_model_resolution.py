"""Tests for provider registry and model resolution."""

from unittest.mock import patch

import pytest

from opencode_proxy.providers.registry import (
    MODEL_ALIASES,
    PROVIDER_DEFINITIONS,
    ProviderRegistry,
)


def _patch_settings(**keys):
    """Patch settings so only the specified API keys are set (rest are None)."""
    all_keys = {
        "OPENAI_API_KEY": None,
        "ANTHROPIC_API_KEY": None,
        "GROQ_API_KEY": None,
        "MISTRAL_API_KEY": None,
        "OPENROUTER_API_KEY": None,
        "TOGETHER_API_KEY": None,
        "DEEPINFRA_API_KEY": None,
        "PERPLEXITY_API_KEY": None,
        "XAI_API_KEY": None,
        "OPENCODE_API_KEY": None,
    }
    all_keys.update(keys)
    return patch.multiple("opencode_proxy.providers.registry.settings", **all_keys)


class TestResolveModel:
    def setup_method(self):
        self.registry = ProviderRegistry()

    def test_explicit_provider_model(self):
        provider, model = self.registry.resolve_model("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_anthropic_provider_model(self):
        provider, model = self.registry.resolve_model("anthropic/claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_alias_resolution(self):
        provider, model = self.registry.resolve_model("gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_alias_big_pickle(self):
        provider, model = self.registry.resolve_model("big-pickle")
        assert provider == "opencode"
        assert model == "big-pickle"

    def test_openrouter_multi_slash(self):
        """OpenRouter models use org/model format within the provider prefix."""
        provider, model = self.registry.resolve_model("openrouter/anthropic/claude-sonnet-4")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4"

    def test_unknown_bare_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            self.registry.resolve_model("nonexistent-model-xyz")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            self.registry.resolve_model("fakeprovider/some-model")

    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError):
            self.registry.resolve_model("")

    def test_all_aliases_resolve(self):
        """Every alias should resolve to a valid provider."""
        for alias, full in MODEL_ALIASES.items():
            provider, model = self.registry.resolve_model(alias)
            assert provider in PROVIDER_DEFINITIONS
            assert model


class TestAvailableProviders:
    def setup_method(self):
        self.registry = ProviderRegistry()

    def test_no_keys_no_providers(self):
        """With no API keys set, no providers are available."""
        with _patch_settings():
            available = self.registry.available_providers()
            assert available == []

    def test_single_key_returns_provider(self):
        with _patch_settings(OPENAI_API_KEY="sk-test"):
            available = self.registry.available_providers()
            assert "openai" in available

    def test_multiple_keys(self):
        with _patch_settings(
            OPENAI_API_KEY="sk-test",
            GROQ_API_KEY="gsk-test",
            OPENCODE_API_KEY="oc-test",
        ):
            available = self.registry.available_providers()
            assert "openai" in available
            assert "groq" in available
            assert "opencode" in available
            assert "anthropic" not in available


class TestGetProvider:
    def setup_method(self):
        self.registry = ProviderRegistry()

    def test_missing_api_key_raises(self):
        with _patch_settings():
            with pytest.raises(ValueError, match="API key not configured"):
                self.registry.get_provider("openai")

    def test_creates_openai_compat_provider(self):
        with _patch_settings(OPENAI_API_KEY="sk-test"):
            from opencode_proxy.providers.openai_compat import OpenAICompatProvider

            provider = self.registry.get_provider("openai")
            assert isinstance(provider, OpenAICompatProvider)

    def test_creates_anthropic_provider(self):
        with _patch_settings(ANTHROPIC_API_KEY="sk-ant-test"):
            from opencode_proxy.providers.anthropic_provider import AnthropicProvider

            provider = self.registry.get_provider("anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_provider_instance_caching(self):
        with _patch_settings(OPENAI_API_KEY="sk-test"):
            p1 = self.registry.get_provider("openai")
            p2 = self.registry.get_provider("openai")
            assert p1 is p2

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            self.registry.get_provider("nonexistent")


class TestListModels:
    def setup_method(self):
        self.registry = ProviderRegistry()

    def test_lists_models_for_available_providers(self):
        with _patch_settings(OPENAI_API_KEY="sk-test"):
            models = self.registry.list_models()
            model_ids = [m[0] for m in models]
            assert "openai/gpt-4o" in model_ids
            # Should not include models from providers without keys
            assert not any(m[0].startswith("anthropic/") for m in models)

    def test_no_models_without_keys(self):
        with _patch_settings():
            models = self.registry.list_models()
            assert models == []
