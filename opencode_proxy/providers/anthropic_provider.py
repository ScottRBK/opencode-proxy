"""Anthropic provider with bidirectional message format transformation."""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from opencode_proxy.models.openai_models import ChatCompletionRequest, ChatCompletionResponse
from opencode_proxy.providers.base import BaseProvider
from opencode_proxy.transform.anthropic_transform import (
    AnthropicStreamTransformer,
    transform_request,
    transform_response,
)

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Messages API with OpenAI format transformation."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
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
        payload = transform_request(request, model_id)

        response = await self._client.post("/v1/messages", json=payload)
        response.raise_for_status()
        return transform_response(response.json(), f"anthropic/{model_id}")

    async def chat_completion_stream(
        self, request: ChatCompletionRequest, model_id: str
    ) -> AsyncGenerator[str, None]:
        payload = transform_request(request, model_id)
        payload["stream"] = True

        transformer = AnthropicStreamTransformer(model=f"anthropic/{model_id}")

        async with self._client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()

            event_type = None
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:") and event_type:
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Anthropic SSE data: {data_str}")
                        continue

                    for output_line in transformer.process_event(event_type, data):
                        yield output_line
                    event_type = None

    async def close(self) -> None:
        await self._client.aclose()
