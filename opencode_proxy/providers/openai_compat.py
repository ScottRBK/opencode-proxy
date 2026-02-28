"""OpenAI-compatible passthrough provider.

Works with any provider that speaks the OpenAI chat completions format:
OpenAI, Groq, Mistral, OpenRouter, Together, DeepInfra, Perplexity, XAI, OpenCode Zen.
"""

import logging
from collections.abc import AsyncGenerator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from opencode_proxy.models.openai_models import ChatCompletionRequest, ChatCompletionResponse
from opencode_proxy.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseProvider):
    """Passthrough provider for OpenAI-compatible APIs."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def chat_completion(
        self, request: ChatCompletionRequest, model_id: str
    ) -> ChatCompletionResponse:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = model_id
        payload["stream"] = False

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())

    async def chat_completion_stream(
        self, request: ChatCompletionRequest, model_id: str
    ) -> AsyncGenerator[str, None]:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = model_id
        payload["stream"] = True

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    yield line
                elif line.startswith("data: "):
                    yield line

    async def close(self) -> None:
        await self._client.aclose()
