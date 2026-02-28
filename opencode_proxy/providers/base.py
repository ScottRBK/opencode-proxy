"""Base provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from opencode_proxy.models.openai_models import ChatCompletionRequest, ChatCompletionResponse


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest, model_id: str
    ) -> ChatCompletionResponse:
        """Handle a non-streaming chat completion request."""

    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest, model_id: str
    ) -> AsyncGenerator[str, None]:
        """Handle a streaming chat completion request. Yields SSE data lines."""

    async def close(self) -> None:
        """Clean up resources. Override if provider holds connections."""
