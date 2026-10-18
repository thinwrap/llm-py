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
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}],
        }
    )
    t = FakeTransport(resp(200, body))
    chat = _chat(t, api_key="sk-ant")
    res = chat.complete(
        llm.ChatInput(
            model="claude-opus-5-5",
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
    assert res.model == "claude-opus-5-5"

    # URL + auth headers
    assert path_of(t.last) == "/v1/messages"
    assert t.last.headers["x-api-key"] == "sk-ant"
    assert t.last.headers["anthropic-version"] == "2023-06-01"

    # request body shape: system split out, default max_tokens (16000), no response_format
    sent = body_json(t.last)
    assert sent["system"] == "Be brief."
    assert sent["max_tokens"] == 16000
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert "response_format" not in sent


def test_complete_tool_call_result():
    body = json.dumps(
        {
            "model": "claude-opus-5-5",
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
            model="claude-opus-5-5",
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


def test_reasoning_effort_sends_adaptive_thinking_and_effort():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    chat = _chat(t, api_key="k")
    chat.complete(
        llm.ChatInput(
            model="claude-opus-5-5",
            messages=[llm.ChatMessage(role="user", content="think")],
            reasoning=llm.ReasoningOptions(effort="high"),
            temperature=0.7,
            top_p=0.9,
        )
    )
    sent = body_json(t.last)
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}
    # Adaptive thinking has no budget, so the default max_tokens is sent as-is.
    assert sent["max_tokens"] == 16000
    assert "budget_tokens" not in json.dumps(sent)
    # sampling omitted while thinking on
    assert "temperature" not in sent and "top_p" not in sent


def test_reasoning_with_small_explicit_max_tokens_is_sent_unchanged():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    _chat(t, api_key="k").complete(
        llm.ChatInput(
            model="claude-opus-5-5",
            messages=[llm.ChatMessage(role="user", content="think")],
            reasoning=llm.ReasoningOptions(effort="low"),
            max_output_tokens=100,
        )
    )
    sent = body_json(t.last)
    assert sent["max_tokens"] == 100
    assert sent["output_config"] == {"effort": "low"}


def test_budget_thinking_via_passthrough_without_reasoning():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    _chat(t, api_key="k").complete(
        llm.ChatInput(
            model="claude-haiku-4-5",
            messages=[llm.ChatMessage(role="user", content="think")],
            max_output_tokens=8000,
            passthrough=llm.Passthrough(body={"thinking": {"type": "enabled", "budget_tokens": 2048}}),
        )
    )
    sent = body_json(t.last)
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in sent


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


@pytest.mark.parametrize("body", ["{}", '{"id":"msg_1","type":"message","stop_reason":"end_turn"}', '{"content":null}'])
def test_complete_200_without_content_array_is_empty_body_error(body):
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(FakeTransport(resp(200, body)), api_key="k").complete(
            llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="hi")])
        )
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.status_code == 200


# ---- provider_metadata: thinking blocks round-trip ----

# A tool-use response with adaptive thinking, in the shape the Messages API
# documents: thinking (with signature) and redacted_thinking blocks precede the
# text and tool_use blocks.
_THINKING_CONTENT = [
    {"type": "thinking", "thinking": "Need the weather tool.", "signature": "sig-abc"},
    {"type": "redacted_thinking", "data": "ENCRYPTED"},
    {"type": "text", "text": "Checking."},
    {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
]


def _tool_turn_result():
    body = json.dumps(
        {
            "model": "claude-opus-5-5",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": _THINKING_CONTENT,
        }
    )
    t = FakeTransport(resp(200, body))
    return _chat(t, api_key="k").complete(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="weather?")])
    )


def _replay(assistant: llm.ChatMessage):
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    _chat(t, api_key="k").complete(
        llm.ChatInput(
            model="claude-opus-5-5",
            messages=[
                llm.ChatMessage(role="user", content="weather?"),
                assistant,
                llm.ChatMessage(role="tool", tool_call_id="toolu_01", content="sunny"),
            ],
        )
    )
    return body_json(t.last)


def test_complete_captures_content_blocks_verbatim():
    res = _tool_turn_result()
    assert res.message.provider_metadata == {"anthropic": {"content": _THINKING_CONTENT}}
    # Decoupled from raw: mutating one does not change the other.
    res.message.provider_metadata["anthropic"]["content"].append({"type": "text", "text": "x"})
    assert len(res.raw["content"]) == 4


def test_replay_sends_content_verbatim_when_message_matches():
    res = _tool_turn_result()
    sent = _replay(
        llm.ChatMessage(
            role="assistant",
            content=res.message.content,
            tool_calls=res.message.tool_calls,
            provider_metadata=res.message.provider_metadata,
        )
    )
    assert sent["messages"][1] == {"role": "assistant", "content": _THINKING_CONTENT}
    assert "provider_metadata" not in json.dumps(sent)


def test_replay_rebuilds_when_text_was_edited():
    res = _tool_turn_result()
    sent = _replay(
        llm.ChatMessage(
            role="assistant",
            content="Checking now.",
            tool_calls=res.message.tool_calls,
            provider_metadata=res.message.provider_metadata,
        )
    )
    assert sent["messages"][1]["content"] == [
        {"type": "text", "text": "Checking now."},
        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
    ]


def test_replay_rebuilds_when_tool_ids_differ():
    res = _tool_turn_result()
    sent = _replay(
        llm.ChatMessage(
            role="assistant",
            content=res.message.content,
            tool_calls=[llm.ToolCall(id="toolu_99", name="get_weather", arguments='{"city":"Paris"}')],
            provider_metadata=res.message.provider_metadata,
        )
    )
    blocks = sent["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]
    assert blocks[1]["id"] == "toolu_99"


def test_replay_ignores_foreign_provider_metadata():
    sent = _replay(
        llm.ChatMessage(
            role="assistant",
            content="Checking.",
            tool_calls=[
                llm.ToolCall(
                    id="toolu_01",
                    name="get_weather",
                    arguments='{"city":"Paris"}',
                    provider_metadata={"gemini": {"thoughtSignature": "g-sig"}},
                )
            ],
            provider_metadata={"gemini": {"content": [{"type": "text", "text": "Checking."}]}},
        )
    )
    assert sent["messages"][1]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    assert "g-sig" not in json.dumps(sent)


def _sse(*events):
    return "\n".join(line for e in events for line in ("data: " + json.dumps(e), ""))


def test_stream_rebuilds_the_same_content_array():
    sse = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Need the "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "weather tool."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "ENCRYPTED"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Checking."}},
        {"type": "content_block_stop", "index": 2},
        {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
        {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
        {"type": "content_block_stop", "index": 3},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    deltas = list(_chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="weather?")])
    ))
    terminal = deltas[-1]
    assert terminal.finish_reason == "tool_calls"
    assert terminal.provider_metadata == {"anthropic": {"content": _THINKING_CONTENT}}
    # Only the terminal delta carries it.
    assert all(d.provider_metadata is None for d in deltas[:-1])
    assert _tool_turn_result().message.provider_metadata == terminal.provider_metadata


def test_stream_server_tool_input_is_not_a_caller_tool_call():
    sse = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query":"x"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    deltas = list(_chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    ))
    tcds = [d.tool_call_delta for d in deltas if d.tool_call_delta is not None]
    assert [(d.index, d.id, d.arguments_delta) for d in tcds] == [(0, "toolu_1", ""), (0, "", "{}")]
    content = deltas[-1].provider_metadata["anthropic"]["content"]
    assert content[0]["input"] == {"query": "x"}


# ---- usage, stop reasons, error classification ----


def test_usage_counts_cache_read_and_write_as_input():
    body = json.dumps({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20, "output_tokens": 7},
    })
    res = _chat(FakeTransport(resp(200, body)), api_key="k").complete(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    )
    assert (res.usage.input_tokens, res.usage.output_tokens, res.usage.total_tokens) == (125, 7, 132)


def test_stream_usage_counts_cache_tokens():
    sse = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20, "output_tokens": 1}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    usage = [d.usage for d in _chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    ) if d.usage][-1]
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (125, 7, 132)


@pytest.mark.parametrize(
    "stop_reason,expected",
    [
        ("pause_turn", "unknown"),
        ("refusal", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("max_tokens", "length"),
    ],
)
def test_stop_reason_mapping(stop_reason, expected):
    body = json.dumps({"content": [{"type": "text", "text": "x"}], "stop_reason": stop_reason, "usage": {}})
    res = _chat(FakeTransport(resp(200, body)), api_key="k").complete(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    )
    assert res.finish_reason == expected


@pytest.mark.parametrize(
    "status,message,expected",
    [
        (400, "prompt is too long: 208310 tokens > 200000 maximum", llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED),
        (400, "input length and `max_tokens` exceed context limit: 188240 + 21333 > 200000, decrease input length or `max_tokens` and try again", llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED),
        (400, "max_tokens: Field required", llm.ProviderCode.INVALID_REQUEST),
        (400, "context_management: Extra inputs are not permitted", llm.ProviderCode.INVALID_REQUEST),
        (400, "tool_choice: type \"tool\" and \"any\" are not supported for this model.", llm.ProviderCode.INVALID_REQUEST),
        (402, "There's an issue with your billing or payment information.", llm.ProviderCode.RATE_LIMITED),
        (413, "Request exceeds the maximum allowed number of bytes.", llm.ProviderCode.INVALID_REQUEST),
        (504, "Request timed out.", llm.ProviderCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_error_classification(status, message, expected):
    body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}})
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(FakeTransport(resp(status, body)), api_key="k").complete(
            llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
        )
    assert ei.value.provider_code is expected


@pytest.mark.parametrize(
    "etype,message,expected",
    [
        ("invalid_request_error", "prompt is too long: 208310 tokens > 200000 maximum", llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED),
        ("invalid_request_error", "max_tokens: Field required", llm.ProviderCode.INVALID_REQUEST),
        ("rate_limit_error", "slow", llm.ProviderCode.RATE_LIMITED),
        ("billing_error", "billing", llm.ProviderCode.RATE_LIMITED),
        ("timeout_error", "timeout", llm.ProviderCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_stream_error_frame_classified_by_type(etype, message, expected):
    sse = _sse({"type": "error", "error": {"type": etype, "message": message}})
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    with pytest.raises(llm.ConnectorError) as ei:
        list(_chat(t, api_key="k").stream(
            llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
        ))
    assert ei.value.provider_code is expected


# ---- review-5 ----


def _replay_history(*assistants):
    """Send a history of alternating user / assistant / tool turns and return the
    assistant turns as sent."""
    msgs = [llm.ChatMessage(role="user", content="go")]
    for a in assistants:
        msgs.append(a)
        for tc in a.tool_calls:
            msgs.append(llm.ChatMessage(role="tool", tool_call_id=tc.id, content="ok"))
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    _chat(t, api_key="k").complete(llm.ChatInput(model="claude-opus-5-5", messages=msgs))
    return [m["content"] for m in body_json(t.last)["messages"] if m["role"] == "assistant"]


def _turn(n, text=None, args=None):
    tid = f"toolu_{n}"
    content = [
        {"type": "thinking", "thinking": f"t{n}", "signature": f"s{n}"},
        {"type": "text", "text": f"step {n}"},
        {"type": "tool_use", "id": tid, "name": "f", "input": {"n": n}},
    ]
    return llm.ChatMessage(
        role="assistant",
        content=f"step {n}" if text is None else text,
        tool_calls=[llm.ToolCall(id=tid, name="f", arguments=json.dumps({"n": n} if args is None else args))],
        provider_metadata={"anthropic": {"content": content}},
    )


def test_every_turn_after_a_rebuilt_one_is_rebuilt():
    sent = _replay_history(_turn(1), _turn(2, text="edited"), _turn(3))
    assert sent[0][0]["type"] == "thinking"  # before the edit: verbatim
    assert [b["type"] for b in sent[1]] == ["text", "tool_use"]  # edited: rebuilt
    assert [b["type"] for b in sent[2]] == ["text", "tool_use"]  # after it: rebuilt too


def test_all_turns_verbatim_when_none_is_edited():
    sent = _replay_history(_turn(1), _turn(2), _turn(3))
    assert all(turn[0]["type"] == "thinking" for turn in sent)


def test_edited_tool_arguments_force_a_rebuild():
    sent = _replay_history(_turn(1, args={"n": 99}))
    assert [b["type"] for b in sent[0]] == ["text", "tool_use"]
    assert sent[0][1]["input"] == {"n": 99}


def test_reformatted_but_equal_arguments_still_replay_verbatim():
    msg = _turn(1)
    msg.tool_calls[0].arguments = '{ "n" : 1 }'
    assert _replay_history(msg)[0][0]["type"] == "thinking"


def test_stream_usage_merges_field_by_field():
    sse = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 20, "output_tokens": 1}}},
        # null and absent fields keep message_start's values.
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": None, "output_tokens": 5}},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    u = [d.usage for d in _chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    ) if d.usage][-1]
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (150, 5, 155)


def test_stream_usage_partial_restatement_keeps_cache_counts():
    sse = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 50}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 110, "output_tokens": 5}},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    u = [d.usage for d in _chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    ) if d.usage][-1]
    assert (u.input_tokens, u.total_tokens) == (160, 165)


def test_stream_keeps_start_input_when_no_deltas_follow():
    sse = _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "q"}}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    last = list(_chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    ))[-1]
    assert last.provider_metadata["anthropic"]["content"][0]["input"] == {"query": "q"}


def test_stream_provider_metadata_only_on_the_stop_delta():
    sse = _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 1}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    )
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    deltas = list(_chat(t, api_key="k").stream(
        llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
    ))
    with_meta = [d for d in deltas if d.provider_metadata is not None]
    assert len(with_meta) == 1 and with_meta[0].finish_reason == "stop"


def test_empty_text_parts_are_dropped():
    t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
    _chat(t, api_key="k").complete(llm.ChatInput(model="claude-opus-5-5", messages=[
        llm.ChatMessage(role="user", content_parts=[llm.text_part(""), llm.text_part("hi"), llm.text_part("")]),
    ]))
    assert body_json(t.last)["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


def test_empty_effort_defaults_to_medium_and_unknown_is_sent_as_given():
    for effort, expected in (("", "medium"), ("max", "max")):
        t = FakeTransport(resp(200, json.dumps({"content": [], "usage": {}, "stop_reason": "end_turn"})))
        _chat(t, api_key="k").complete(llm.ChatInput(
            model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")],
            reasoning=llm.ReasoningOptions(effort=effort),
        ))
        assert body_json(t.last)["output_config"] == {"effort": expected}


@pytest.mark.parametrize(
    "status,message,expected",
    [
        (408, "Request timeout", llm.ProviderCode.PROVIDER_UNAVAILABLE),
        (409, "conflict", llm.ProviderCode.INVALID_REQUEST),
        (400, "context window is too small for this tool", llm.ProviderCode.INVALID_REQUEST),
    ],
)
def test_status_classification_review5(status, message, expected):
    body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}})
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(FakeTransport(resp(status, body)), api_key="k").complete(
            llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
        )
    assert ei.value.provider_code is expected


def test_stream_error_frame_with_unknown_type_is_unknown_and_keeps_stream_status():
    sse = _sse({"type": "error", "error": {"type": "brand_new_error", "message": "?"}})
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    with pytest.raises(llm.ConnectorError) as ei:
        list(_chat(t, api_key="k").stream(
            llm.ChatInput(model="claude-opus-5-5", messages=[llm.ChatMessage(role="user", content="x")])
        ))
    assert ei.value.provider_code is llm.ProviderCode.UNKNOWN and ei.value.status_code == 200
