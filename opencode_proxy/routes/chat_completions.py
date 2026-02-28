"""OpenAI-compatible chat completions endpoint."""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from opencode_proxy.models.openai_models import ChatCompletionRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat Completions"])


async def _stream_generator(provider, request, model_id):
    """Wrap provider streaming to yield SSE-formatted dicts."""
    async for line in provider.chat_completion_stream(request, model_id):
        # Lines already come as "data: ..." from the provider
        # EventSourceResponse expects {"data": "..."} dicts
        if line.startswith("data: "):
            yield {"data": line[6:]}
        elif line.startswith("data:"):
            yield {"data": line[5:]}


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    """Create a chat completion via the appropriate provider."""
    registry = request.app.state.registry

    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        provider_id, model_id = registry.resolve_model(body.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        provider = registry.get_provider(provider_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Chat completion: provider={provider_id}, model={model_id}, stream={body.stream}")

    try:
        if body.stream:
            return EventSourceResponse(_stream_generator(provider, body, model_id))

        return await provider.chat_completion(body, model_id)

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        detail = e.response.text
        logger.error(f"Upstream error from {provider_id}: {status} - {detail}")

        if status == 401:
            raise HTTPException(status_code=401, detail=f"Authentication failed for {provider_id}")
        elif status == 429:
            raise HTTPException(status_code=429, detail=f"Rate limited by {provider_id}")
        elif status >= 500:
            raise HTTPException(status_code=502, detail=f"Upstream error from {provider_id}: {detail}")
        else:
            raise HTTPException(status_code=status, detail=detail)

    except (httpx.ConnectError, httpx.ReadTimeout) as e:
        logger.error(f"Connection error to {provider_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to {provider_id}: {e}")

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
