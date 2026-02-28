"""OpenAI-compatible request and response models."""

from time import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# Tool definition models (for request)
class FunctionDefinition(BaseModel):
    """Function definition in a tool."""

    name: str
    description: str | None = None
    parameters: dict | None = None


class Tool(BaseModel):
    """Tool definition for function calling."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


# Tool call models (for response)
class FunctionCall(BaseModel):
    """Function call details in a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# Request models
class ChatMessage(BaseModel):
    """A single chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None


# Response models
class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


# Streaming models
class DeltaContent(BaseModel):
    """Delta content for streaming responses."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


class StreamChoice(BaseModel):
    """A streaming choice."""

    index: int = 0
    delta: DeltaContent
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: Usage | None = None


# Models endpoint
class Model(BaseModel):
    """A model definition."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "opencode-proxy"


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[Model]
