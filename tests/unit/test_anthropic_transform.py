"""Tests for Anthropic message format transformations."""

import json

import pytest

from opencode_proxy.models.openai_models import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    Tool,
    ToolCall,
)
from opencode_proxy.transform.anthropic_transform import (
    AnthropicStreamTransformer,
    transform_request,
    transform_response,
)


class TestTransformRequest:
    def test_simple_user_message(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["max_tokens"] == 4096
        assert "system" not in payload

    def test_system_message_extracted(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert payload["system"] == [{"type": "text", "text": "You are helpful"}]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_multiple_system_messages(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[
                ChatMessage(role="system", content="Rule 1"),
                ChatMessage(role="system", content="Rule 2"),
                ChatMessage(role="user", content="Hello"),
            ],
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert len(payload["system"]) == 2
        assert payload["system"][0]["text"] == "Rule 1"
        assert payload["system"][1]["text"] == "Rule 2"

    def test_max_tokens_passthrough(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=1000,
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert payload["max_tokens"] == 1000

    def test_tool_calls_transformed(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[
                ChatMessage(role="user", content="What's the weather?"),
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            function=FunctionCall(
                                name="get_weather",
                                arguments='{"location": "London"}',
                            ),
                        )
                    ],
                ),
                ChatMessage(role="tool", content="Sunny, 22C", tool_call_id="call_123"),
                ChatMessage(role="user", content="Thanks"),
            ],
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        messages = payload["messages"]

        # Assistant message should have tool_use block
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0]["type"] == "tool_use"
        assert messages[1]["content"][0]["id"] == "call_123"
        assert messages[1]["content"][0]["name"] == "get_weather"
        assert messages[1]["content"][0]["input"] == {"location": "London"}

        # Tool result should be in a user message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "call_123"

    def test_tools_schema_transformed(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[
                Tool(
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={"type": "object", "properties": {"loc": {"type": "string"}}},
                    )
                )
            ],
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert "tools" in payload
        tool = payload["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"

    def test_consecutive_user_messages_merged(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="user", content="World"),
            ],
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        # Merged into content blocks
        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2

    def test_temperature_and_top_p(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.7,
            top_p=0.9,
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9

    def test_tool_choice_auto(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[ChatMessage(role="user", content="Hi")],
            tool_choice="auto",
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert payload["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self):
        request = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[ChatMessage(role="user", content="Hi")],
            tool_choice="required",
        )
        payload = transform_request(request, "claude-sonnet-4-20250514")
        assert payload["tool_choice"] == {"type": "any"}


class TestTransformResponse:
    def test_simple_text_response(self):
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = transform_response(anthropic_response, "anthropic/claude-sonnet-4-20250514")
        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.model == "anthropic/claude-sonnet-4-20250514"

    def test_tool_use_response(self):
        anthropic_response = {
            "id": "msg_456",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_789",
                    "name": "get_weather",
                    "input": {"location": "London"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 30},
        }
        result = transform_response(anthropic_response, "anthropic/claude-sonnet-4-20250514")
        assert result.choices[0].message.content == "Let me check."
        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "toolu_789"
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"location": "London"}

    def test_max_tokens_stop_reason(self):
        anthropic_response = {
            "id": "msg_max",
            "content": [{"type": "text", "text": "Truncated..."}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }
        result = transform_response(anthropic_response, "anthropic/claude-sonnet-4-20250514")
        assert result.choices[0].finish_reason == "length"


class TestAnthropicStreamTransformer:
    def test_message_start_emits_role(self):
        transformer = AnthropicStreamTransformer(model="anthropic/claude-sonnet-4-20250514")
        lines = transformer.process_event("message_start", {
            "message": {
                "id": "msg_123",
                "role": "assistant",
                "usage": {"input_tokens": 10},
            }
        })
        assert len(lines) == 1
        data = json.loads(lines[0].removeprefix("data: "))
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_text_delta_emits_content(self):
        transformer = AnthropicStreamTransformer(model="anthropic/claude-sonnet-4-20250514")
        # Start message first
        transformer.process_event("message_start", {"message": {"usage": {}}})
        transformer.process_event("content_block_start", {
            "content_block": {"type": "text"}
        })

        lines = transformer.process_event("content_block_delta", {
            "delta": {"type": "text_delta", "text": "Hello"}
        })
        assert len(lines) == 1
        data = json.loads(lines[0].removeprefix("data: "))
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_tool_use_streaming(self):
        transformer = AnthropicStreamTransformer(model="anthropic/claude-sonnet-4-20250514")
        transformer.process_event("message_start", {"message": {"usage": {}}})

        # Start tool use block
        lines = transformer.process_event("content_block_start", {
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
            }
        })
        assert len(lines) == 1
        data = json.loads(lines[0].removeprefix("data: "))
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        assert tc["id"] == "toolu_123"
        assert tc["function"]["name"] == "get_weather"

        # Argument deltas
        lines = transformer.process_event("content_block_delta", {
            "delta": {"type": "input_json_delta", "partial_json": '{"loc'}
        })
        assert len(lines) == 1
        data = json.loads(lines[0].removeprefix("data: "))
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"loc'

    def test_message_delta_emits_finish_reason(self):
        transformer = AnthropicStreamTransformer(model="anthropic/claude-sonnet-4-20250514")
        transformer.process_event("message_start", {
            "message": {"usage": {"input_tokens": 10}}
        })
        lines = transformer.process_event("message_delta", {
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        })
        assert len(lines) == 1
        data = json.loads(lines[0].removeprefix("data: "))
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5

    def test_message_stop_emits_done(self):
        transformer = AnthropicStreamTransformer(model="anthropic/claude-sonnet-4-20250514")
        lines = transformer.process_event("message_stop", {})
        assert lines == ["data: [DONE]"]

    def test_full_text_stream_sequence(self):
        """Test a complete text streaming sequence."""
        transformer = AnthropicStreamTransformer(model="anthropic/claude-sonnet-4-20250514")
        all_lines = []

        # message_start
        all_lines.extend(transformer.process_event("message_start", {
            "message": {"id": "msg_1", "role": "assistant", "usage": {"input_tokens": 5}}
        }))
        # content_block_start (text)
        all_lines.extend(transformer.process_event("content_block_start", {
            "content_block": {"type": "text", "text": ""}
        }))
        # text deltas
        all_lines.extend(transformer.process_event("content_block_delta", {
            "delta": {"type": "text_delta", "text": "Hi "}
        }))
        all_lines.extend(transformer.process_event("content_block_delta", {
            "delta": {"type": "text_delta", "text": "there!"}
        }))
        # content_block_stop
        all_lines.extend(transformer.process_event("content_block_stop", {}))
        # message_delta
        all_lines.extend(transformer.process_event("message_delta", {
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        }))
        # message_stop
        all_lines.extend(transformer.process_event("message_stop", {}))

        # Verify sequence
        assert len(all_lines) == 5  # role, "Hi ", "there!", finish, [DONE]
        assert all_lines[-1] == "data: [DONE]"

        # Check first chunk has role
        first = json.loads(all_lines[0].removeprefix("data: "))
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # Check finish chunk
        finish = json.loads(all_lines[-2].removeprefix("data: "))
        assert finish["choices"][0]["finish_reason"] == "stop"
