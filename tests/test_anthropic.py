"""Anthropic native-adapter tests: Messages-API request build, complete() parse
(text + tool_use), streaming over Anthropic SSE events, and error mapping."""

from __future__ import annotations

import json

import pytest
from helpers import FakeTransport, body_json, path_of, resp

import thinwrap.llm as llm


def _chat(t, **cfg):
    return llm.Chat(llm.ProviderID.ANTHROPIC, llm.AnthropicConfig(**cfg), transport=t)


def test_complete_success_request_and_result():
    body = json.dumps(
        {
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}],
        }
    )
    t = FakeTransport(resp(200, body))
    chat = _chat(t, api_key="sk-ant")
    res = chat.complete(
        llm.ChatInput(
            model="claude-3-5-sonnet-20241022",
            messages=[
                llm.ChatMessage(role="system", content="Be brief."),
                llm.ChatMessage(role="user", content="Hi"),
            ],
        )
    )
    # normalized result
    assert res.message.content == "Hello there"
    assert res.finish_reason == "stop"
    assert res.usage.input_tokens == 11 and res.usage.output_tokens == 7 and res.usage.total_tokens == 18
    assert res.model == "claude-3-5-sonnet-20241022"

    # URL + auth headers
    assert path_of(t.last) == "/v1/messages"
    assert t.last.headers["x-api-key"] == "sk-ant"
    assert t.last.headers["anthropic-version"] == "2023-06-01"

    # request body shape: system split out, default max_tokens, no response_format
    sent = body_json(t.last)
    assert sent["system"] == "Be brief."
    assert sent["max_tokens"] == 4096
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert "response_format" not in sent


def test_complete_tool_call_result():
    body = json.dumps(
        {
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
            ],
        }
    )
    t = FakeTransport(resp(200, body))
    chat = _chat(t, api_key="k")
    res = chat.complete(
        llm.ChatInput(
            model="claude-3-5-sonnet-20241022",
            messages=[llm.ChatMessage(role="user", content="weather?")],
            tools=[llm.ToolDefinition(name="get_weather", description="Get weather", parameters={"type": "object"})],
            tool_choice=llm.ToolChoice(mode="required"),
        )
    )
    assert res.finish_reason == "tool_calls"
    assert len(res.message.tool_calls) == 1
    tc = res.message.tool_calls[0]
    assert tc.id == "toolu_1" and tc.name == "get_weather"
    assert json.loads(tc.arguments) == {"city": "SF"}

    sent = body_json(t.last)
    # tools → input_schema; tool_choice required→any
    assert sent["tools"] == [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}]
    assert sent["tool_choice"] == {"type": "any"}


def test_tool_result_folds_into_user_turn():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    chat = _chat(t, api_key="k")
    chat.complete(
        llm.ChatInput(
            model="m",
            messages=[
                llm.ChatMessage(
                    role="assistant",
                    tool_calls=[llm.ToolCall(id="toolu_1", name="get_weather", arguments='{"city":"SF"}')],
                ),
                llm.ChatMessage(role="tool", tool_call_id="toolu_1", content="sunny"),
                llm.ChatMessage(role="tool", tool_call_id="toolu_2", content="warm"),
            ],
        )
    )
    sent = body_json(t.last)
    assert sent["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "warm"},
            ],
        },
    ]


def test_reasoning_effort_enables_thinking_and_bumps_max_tokens():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    chat = _chat(t, api_key="k")
    chat.complete(
        llm.ChatInput(
            model="m",
            messages=[llm.ChatMessage(role="user", content="think")],
            reasoning=llm.ReasoningOptions(effort="high"),
            temperature=0.7,
            top_p=0.9,
        )
    )
    sent = body_json(t.last)
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 12288}
    # default max_tokens (4096) <= budget → bumped to budget + 4096
    assert sent["max_tokens"] == 12288 + 4096
    # sampling omitted while thinking on
    assert "temperature" not in sent and "top_p" not in sent


def test_image_part_and_stop_sequences_and_passthrough():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    chat = _chat(t, api_key="k", anthropic_version="2099-01-01", headers={"x-extra": "1"})
    chat.complete(
        llm.ChatInput(
            model="m",
            messages=[
                llm.ChatMessage(role="user", content_parts=[llm.image_part("BASE64DATA", "image/png")]),
            ],
            stop=["STOP", "END"],
            passthrough=llm.Passthrough(body={"metadata": {"user_id": "u1"}}, headers={"x-req": "yes"}),
        )
    )
    sent = body_json(t.last)
    assert sent["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BASE64DATA"}}
    ]
    assert sent["stop_sequences"] == ["STOP", "END"]
    assert sent["metadata"] == {"user_id": "u1"}
    # config version override + config header + passthrough header
    assert t.last.headers["anthropic-version"] == "2099-01-01"
    assert t.last.headers["x-extra"] == "1"
    assert t.last.headers["x-req"] == "yes"


def test_stream_text_and_tool_and_usage():
    sse = "\n".join(
        [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":25}}}',
            "",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
            "",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}',
            "",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}',
            "",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_9","name":"f"}}',
            "",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":1}"}}',
            "",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}',
            "",
            'data: {"type":"message_stop"}',
            "",
        ]
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    chat = _chat(t, api_key="k")
    text, finish, usage = "", "", None
    tool_start = None
    args = ""
    for d in chat.stream(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="hi")])):
        text += d.content_delta
        if d.tool_call_delta is not None:
            if d.tool_call_delta.function_name:
                tool_start = d.tool_call_delta
            if d.tool_call_delta.arguments_delta:
                args += d.tool_call_delta.arguments_delta
        if d.finish_reason:
            finish = d.finish_reason
        if d.usage:
            usage = d.usage
    assert text == "Hello"
    # Anthropic's content-block index is 1 (a text block preceded it), but the
    # normalized tool-call index is 0-based tool-call-relative (matches the other
    # languages / the OpenAI-compat surface).
    assert tool_start is not None and tool_start.index == 0 and tool_start.id == "toolu_9" and tool_start.function_name == "f"
    assert args == '{"a":1}'
    assert finish == "stop"
    # input_tokens captured at message_start, output_tokens at message_delta
    assert usage is not None and usage.input_tokens == 25 and usage.output_tokens == 8 and usage.total_tokens == 33
    assert body_json(t.last)["stream"] is True


def test_error_mapping_rate_limited():
    t = FakeTransport(resp(429, '{"error":{"type":"rate_limit_error","message":"slow down"}}', retry_after="30"))
    chat = _chat(t, api_key="k")
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED
    assert ei.value.provider_message == "slow down"
    assert ei.value.cause["retryAfterSeconds"] == 30


def test_error_mapping_auth_and_unavailable_and_context():
    # 401 → auth
    t = FakeTransport(resp(401, '{"error":{"message":"bad key"}}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t, api_key="k").complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.AUTH_FAILED

    # 529 → provider unavailable (Anthropic overloaded)
    t = FakeTransport(resp(529, '{"error":{"message":"overloaded"}}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t, api_key="k").complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE

    # 400 with context-length signal → context_length_exceeded
    t = FakeTransport(resp(400, '{"error":{"message":"prompt is too long: 250000 tokens > max_tokens"}}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t, api_key="k").complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED

    # 400 generic → invalid_request
    t = FakeTransport(resp(400, '{"error":{"message":"unknown field foo"}}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t, api_key="k").complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_stream_error_frame_raises():
    """An `event: error` frame after 200 OK is surfaced as a ConnectorError, not
    silently truncated. Deltas emitted before the error still reach the consumer."""
    sse = "\n".join(
        [
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            "",
            "event: error",
            'data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded mid-stream"}}',
            "",
        ]
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    gen = _chat(t, api_key="k").stream(
        llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="hi")])
    )
    assert next(gen).content_delta == "Hi"
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    assert ei.value.provider_message == "overloaded mid-stream"


def test_requires_anthropic_config():
    with pytest.raises(llm.ConnectorError) as ei:
        llm.Chat(llm.ProviderID.ANTHROPIC, llm.OpenAICompatConfig(api_key="k"))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST
