"""Transform between OpenAI and Anthropic message formats.

Handles:
- Request: OpenAI chat format -> Anthropic Messages API format
- Response: Anthropic Messages API format -> OpenAI chat format
- Streaming: Anthropic SSE events -> OpenAI SSE chunks
"""

import json
import logging
from time import time
from uuid import uuid4

from opencode_proxy.models.openai_models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    DeltaContent,
    FunctionCall,
    StreamChoice,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


def transform_request(request, model_id: str) -> dict:
    """Transform an OpenAI ChatCompletionRequest into an Anthropic Messages API request.

    Key transformations:
    - Extract system messages to top-level `system` field
    - Convert tool_calls in assistant messages to tool_use content blocks
    - Convert tool results to tool_result content blocks in user messages
    - Map tools `parameters` -> `input_schema`
    - Default max_tokens to 4096 if not specified
    - Merge consecutive same-role messages
    """
    system_parts = []
    messages = []

    for msg in request.messages:
        if msg.role == "system":
            # Collect system messages into top-level system field
            if isinstance(msg.content, str):
                system_parts.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append({"type": "text", "text": part["text"]})
            continue

        if msg.role == "assistant" and msg.tool_calls:
            # Assistant message with tool calls -> content blocks with tool_use
            content_blocks = []
            if msg.content:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                content_blocks.append({"type": "text", "text": text})
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": arguments,
                })
            messages.append({"role": "assistant", "content": content_blocks})

        elif msg.role == "tool":
            # Tool result -> tool_result block in a user message
            content_text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": content_text,
            }
            # Anthropic expects tool_result inside a user message
            messages.append({"role": "user", "content": [tool_result]})

        else:
            # Regular user/assistant message
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                messages.append({"role": msg.role, "content": msg.content or ""})

    # Merge consecutive same-role messages (Anthropic requires alternating roles)
    messages = _merge_consecutive_messages(messages)

    payload: dict = {
        "model": model_id,
        "messages": messages,
        "max_tokens": request.max_tokens or 4096,
    }

    if system_parts:
        payload["system"] = system_parts

    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p

    # Transform tools: parameters -> input_schema
    if request.tools:
        anthropic_tools = []
        for tool in request.tools:
            anthropic_tools.append({
                "name": tool.function.name,
                "description": tool.function.description or "",
                "input_schema": tool.function.parameters or {"type": "object", "properties": {}},
            })
        payload["tools"] = anthropic_tools

    if request.tool_choice:
        if isinstance(request.tool_choice, str):
            if request.tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif request.tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif request.tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
        elif isinstance(request.tool_choice, dict):
            fn = request.tool_choice.get("function", {})
            if fn.get("name"):
                payload["tool_choice"] = {"type": "tool", "name": fn["name"]}

    return payload


def _merge_consecutive_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role."""
    if not messages:
        return messages

    merged = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]

            # Normalize both to lists of content blocks
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(curr_content, str):
                curr_content = [{"type": "text", "text": curr_content}]
            if not isinstance(prev_content, list):
                prev_content = [{"type": "text", "text": str(prev_content)}]
            if not isinstance(curr_content, list):
                curr_content = [{"type": "text", "text": str(curr_content)}]

            merged[-1]["content"] = prev_content + curr_content
        else:
            merged.append(msg)

    return merged


def transform_response(response_dict: dict, model: str) -> ChatCompletionResponse:
    """Transform an Anthropic Messages API response to OpenAI format."""
    content_text = ""
    tool_calls = []

    for block in response_dict.get("content", []):
        if block["type"] == "text":
            content_text += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    function=FunctionCall(
                        name=block["name"],
                        arguments=json.dumps(block["input"]),
                    ),
                )
            )

    # Map stop_reason
    stop_reason = response_dict.get("stop_reason", "end_turn")
    finish_reason_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    message = ChatMessage(
        role="assistant",
        content=content_text or None,
        tool_calls=tool_calls or None,
    )

    # Build usage
    usage_data = response_dict.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_data.get("input_tokens", 0),
        completion_tokens=usage_data.get("output_tokens", 0),
        total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{response_dict.get('id', uuid4().hex[:24])}",
        created=int(time()),
        model=model,
        choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=usage,
    )


class AnthropicStreamTransformer:
    """State machine for transforming Anthropic SSE events to OpenAI SSE chunks.

    Anthropic event sequence:
      message_start -> content_block_start -> content_block_delta* ->
      content_block_stop -> ... -> message_delta -> message_stop

    OpenAI expects:
      First chunk: role="assistant"
      Text chunks: content delta
      Tool call chunks: tool_calls delta
      Final chunk: finish_reason set
      Then: data: [DONE]
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._completion_id = f"chatcmpl-{uuid4().hex[:24]}"
        self._created = int(time())
        self._sent_role = False
        self._current_block_type: str | None = None
        self._current_block_index: int = -1
        self._tool_call_id: str | None = None
        self._tool_call_name: str | None = None
        self._tool_call_args: str = ""
        self._tool_call_index: int = -1
        self._usage: Usage | None = None

    def _make_chunk(
        self,
        delta: DeltaContent,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> str:
        chunk = ChatCompletionChunk(
            id=self._completion_id,
            created=self._created,
            model=self._model,
            choices=[StreamChoice(delta=delta, finish_reason=finish_reason)],
            usage=usage,
        )
        return f"data: {chunk.model_dump_json()}"

    def process_event(self, event_type: str, data: dict) -> list[str]:
        """Process an Anthropic SSE event, returning zero or more OpenAI SSE lines."""
        lines = []

        if event_type == "message_start":
            # Emit initial role chunk
            if not self._sent_role:
                lines.append(self._make_chunk(DeltaContent(role="assistant")))
                self._sent_role = True
            # Capture input token usage from message_start
            message = data.get("message", {})
            usage = message.get("usage", {})
            if usage:
                self._usage = Usage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=0,
                    total_tokens=usage.get("input_tokens", 0),
                )

        elif event_type == "content_block_start":
            self._current_block_index += 1
            block = data.get("content_block", {})
            self._current_block_type = block.get("type")

            if self._current_block_type == "tool_use":
                self._tool_call_index += 1
                self._tool_call_id = block.get("id", "")
                self._tool_call_name = block.get("name", "")
                self._tool_call_args = ""
                # Emit tool call start
                lines.append(self._make_chunk(DeltaContent(tool_calls=[{
                    "index": self._tool_call_index,
                    "id": self._tool_call_id,
                    "type": "function",
                    "function": {"name": self._tool_call_name, "arguments": ""},
                }])))

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})

            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    lines.append(self._make_chunk(DeltaContent(content=text)))

            elif delta.get("type") == "input_json_delta":
                partial = delta.get("partial_json", "")
                if partial:
                    self._tool_call_args += partial
                    lines.append(self._make_chunk(DeltaContent(tool_calls=[{
                        "index": self._tool_call_index,
                        "function": {"arguments": partial},
                    }])))

        elif event_type == "content_block_stop":
            self._current_block_type = None

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason", "end_turn")
            finish_map = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
                "max_tokens": "length",
            }
            finish_reason = finish_map.get(stop_reason, "stop")

            # Update usage with output tokens
            usage_data = data.get("usage", {})
            if usage_data and self._usage:
                self._usage.completion_tokens = usage_data.get("output_tokens", 0)
                self._usage.total_tokens = (
                    self._usage.prompt_tokens + self._usage.completion_tokens
                )

            lines.append(self._make_chunk(
                DeltaContent(), finish_reason=finish_reason, usage=self._usage
            ))

        elif event_type == "message_stop":
            lines.append("data: [DONE]")

        return lines
