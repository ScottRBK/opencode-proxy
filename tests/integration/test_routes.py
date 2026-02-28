"""Integration tests for routes with mocked upstream providers."""

from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from main import app

# Settings overrides for test providers
TEST_KEYS = {
    "OPENAI_API_KEY": "sk-test-key",
    "ANTHROPIC_API_KEY": "sk-ant-test-key",
    "GROQ_API_KEY": "gsk-test-key",
    "OPENCODE_API_KEY": "oc-test-key",
    # Explicitly None for unconfigured providers
    "MISTRAL_API_KEY": None,
    "OPENROUTER_API_KEY": None,
    "TOGETHER_API_KEY": None,
    "DEEPINFRA_API_KEY": None,
    "PERPLEXITY_API_KEY": None,
    "XAI_API_KEY": None,
}


@pytest.fixture
def client():
    """Create a test client with mocked API keys on settings."""
    with patch.multiple("opencode_proxy.config.settings", **TEST_KEYS):
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "available_providers" in data

    def test_health_lists_providers(self, client):
        response = client.get("/health")
        providers = response.json()["available_providers"]
        assert "openai" in providers
        assert "anthropic" in providers
        assert "groq" in providers
        assert "opencode" in providers
        # Not configured
        assert "mistral" not in providers

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == "opencode-proxy"


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "openai/gpt-4o" in model_ids
        assert "opencode/big-pickle" in model_ids

    def test_get_specific_model(self, client):
        response = client.get("/v1/models/openai/gpt-4o")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "openai/gpt-4o"
        assert data["owned_by"] == "openai"

    def test_get_unknown_model(self, client):
        response = client.get("/v1/models/fakeprovider/no-model")
        assert response.status_code == 404


class TestChatCompletionsPassthrough:
    @respx.mock
    def test_non_streaming_passthrough(self, client):
        """Non-streaming request forwarded to OpenAI and response returned."""
        upstream_response = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=upstream_response)
        )

        response = client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 8

    @respx.mock
    def test_streaming_passthrough(self, client):
        """Streaming request forwards SSE lines verbatim."""
        sse_lines = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_lines.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )

        response = client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        assert response.status_code == 200
        # SSE response should contain our chunks
        text = response.text
        assert "assistant" in text
        assert "Hi" in text
        assert "[DONE]" in text

    @respx.mock
    def test_upstream_401_returns_401(self, client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        response = client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 401

    @respx.mock
    def test_upstream_429_returns_429(self, client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, text="Rate limited")
        )

        response = client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 429

    @respx.mock
    def test_upstream_500_returns_502(self, client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        response = client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 502


class TestChatCompletionsAnthropic:
    @respx.mock
    def test_non_streaming_anthropic(self, client):
        """Anthropic request is transformed and response mapped back."""
        anthropic_response = {
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Bonjour!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 4},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=anthropic_response)
        )

        response = client.post("/v1/chat/completions", json={
            "model": "anthropic/claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Bonjour!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 8
        assert data["model"] == "anthropic/claude-sonnet-4-20250514"


class TestChatCompletionsErrors:
    def test_unknown_model_returns_400(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "nonexistent/fake-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 400
        assert "Unknown provider" in response.json()["detail"]

    def test_missing_api_key_returns_400(self):
        """Provider without API key returns descriptive error."""
        keys = {k: None for k in TEST_KEYS}
        keys["OPENAI_API_KEY"] = "sk-test"
        with patch.multiple("opencode_proxy.config.settings", **keys):
            with TestClient(app) as client:
                response = client.post("/v1/chat/completions", json={
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hi"}],
                })
                assert response.status_code == 400
                assert "API key not configured" in response.json()["detail"]

    def test_empty_messages_returns_400(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [],
        })
        assert response.status_code == 400
