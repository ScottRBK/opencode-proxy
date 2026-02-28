"""End-to-end tests against live providers.

Uses opencode/big-pickle (free model on OpenCode Zen).
Requires OPENCODE_API_KEY environment variable.
"""

import os

import pytest

from opencode_proxy.config import settings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not settings.OPENCODE_API_KEY,
        reason="OPENCODE_API_KEY not set",
    ),
]


class TestNonStreaming:
    def test_non_streaming_completion(self, api_client):
        """Send a simple message to big-pickle and verify response shape."""
        response = api_client.post("/v1/chat/completions", json={
            "model": "opencode/big-pickle",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 50,
        })

        assert response.status_code == 200
        data = response.json()

        # Verify response shape
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1

        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"]  # Non-empty
        assert choice["finish_reason"] in ("stop", "length")

        # Usage should be present
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0


class TestStreaming:
    def test_streaming_completion(self, api_client):
        """Send a streaming request and verify SSE chunk sequence."""
        with api_client.stream("POST", "/v1/chat/completions", json={
            "model": "opencode/big-pickle",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "stream": True,
            "max_tokens": 50,
        }) as response:
            assert response.status_code == 200

            chunks = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    data_str = line.removeprefix("data:").strip()
                    if data_str == "[DONE]":
                        chunks.append("[DONE]")
                    elif data_str:
                        import json
                        chunks.append(json.loads(data_str))

        assert len(chunks) >= 2  # At least one content chunk + [DONE]
        assert "[DONE]" in chunks

        # Filter to only proper chat completion chunks (some providers send extra metadata)
        chat_chunks = [
            c for c in chunks
            if isinstance(c, dict) and c.get("object") == "chat.completion.chunk"
        ]
        assert len(chat_chunks) >= 1

        # Should have some content in the stream
        content_parts = [
            c["choices"][0]["delta"].get("content", "")
            for c in chat_chunks
            if c.get("choices")
        ]
        full_content = "".join(content_parts)
        assert len(full_content) > 0


class TestMultiTurn:
    def test_multi_turn_conversation(self, api_client):
        """Send a multi-turn conversation and verify contextual response."""
        response = api_client.post("/v1/chat/completions", json={
            "model": "opencode/big-pickle",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "My favorite color is blue."},
                {"role": "assistant", "content": "Got it, your favorite color is blue!"},
                {"role": "user", "content": "What is my favorite color?"},
            ],
            "max_tokens": 50,
        })

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()
        assert "blue" in content


class TestEndpoints:
    def test_models_endpoint_lists_providers(self, api_client):
        """Verify /v1/models returns opencode/big-pickle."""
        response = api_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "opencode/big-pickle" in model_ids

    def test_health_endpoint(self, api_client):
        """Verify /health shows opencode as available."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "opencode" in data["available_providers"]


class TestErrors:
    def test_invalid_model_returns_error(self, api_client):
        """Unknown model should return 400."""
        response = api_client.post("/v1/chat/completions", json={
            "model": "nonexistent/fake-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 400

    def test_missing_api_key_returns_error(self, api_client):
        """Provider without configured key should return 400."""
        # Mistral is unlikely to have a key set in test environments
        response = api_client.post("/v1/chat/completions", json={
            "model": "mistral/mistral-large-latest",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        # Should get 400 (API key not configured) unless MISTRAL_API_KEY happens to be set
        if not settings.MISTRAL_API_KEY:
            assert response.status_code == 400
            assert "API key not configured" in response.json()["detail"]
