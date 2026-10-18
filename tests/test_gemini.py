"""Gemini native-adapter tests: complete() (request build + parse), a
functionCall→ToolCall case, streaming SSE, and error mapping."""

from __future__ import annotations

import json

import pytest
from helpers import FakeTransport, body_json, path_of, resp

import thinwrap.llm as llm


def _chat(*responses):
    t = FakeTransport(*responses)
    chat = llm.Chat(llm.ProviderID.GEMINI, llm.GeminiConfig(api_key="AIza-key"), transport=t)
    return chat, t


def test_gemini_complete():
    body = json.dumps(
        {
            "modelVersion": "gemini-3.8-flash",
            "candidates": [
                {"content": {"parts": [{"text": "Hi "}, {"text": "there"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 12, "totalTokenCount": 21},
        }
    )
    chat, t = _chat(resp(200, body))
    res = chat.complete(
        llm.ChatInput(
            model="gemini-3.8-flash",
            messages=[
                llm.ChatMessage(role="system", content="Be terse"),
                llm.ChatMessage(role="user", content="Hello"),
            ],
            temperature=0.5,
            max_output_tokens=256,
            stop=["END"],
            reasoning=llm.ReasoningOptions(effort="medium"),
        )
    )

    # URL + auth
    assert t.last.url.endswith("/models/gemini-3.8-flash:generateContent")
    assert path_of(t.last) == "/v1beta/models/gemini-3.8-flash:generateContent"
    assert t.last.headers["x-goog-api-key"] == "AIza-key"
    assert "Authorization" not in t.last.headers

    # request body shape
    sent = body_json(t.last)
    assert sent["systemInstruction"] == {"parts": [{"text": "Be terse"}]}
    assert sent["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
    gc = sent["generationConfig"]
    assert gc["maxOutputTokens"] == 256
    assert gc["temperature"] == 0.5
    assert gc["stopSequences"] == ["END"]
    assert gc["thinkingConfig"] == {"thinkingBudget": 8192}

    # normalized result
    assert res.message.content == "Hi there"
    assert res.finish_reason == "stop"
    assert res.model == "gemini-3.8-flash"
    assert res.usage.input_tokens == 9 and res.usage.output_tokens == 12 and res.usage.total_tokens == 21


def test_gemini_default_max_tokens_fallback():
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})
    t = FakeTransport(resp(200, body))
    chat = llm.Chat(
        llm.ProviderID.GEMINI, llm.GeminiConfig(api_key="k", default_max_tokens=1000), transport=t
    )
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[llm.ChatMessage(role="user", content="hi")]))
    assert body_json(t.last)["generationConfig"]["maxOutputTokens"] == 1000


def test_gemini_tool_call():
    body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7, "totalTokenCount": 12},
        }
    )
    chat, t = _chat(resp(200, body))
    res = chat.complete(
        llm.ChatInput(
            model="gemini-3.8-flash",
            messages=[llm.ChatMessage(role="user", content="weather?")],
            tools=[
                llm.ToolDefinition(
                    name="get_weather",
                    description="Look up weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
            tool_choice=llm.ToolChoice(mode="required"),
        )
    )

    # request: functionDeclarations + toolConfig
    sent = body_json(t.last)
    decls = sent["tools"][0]["functionDeclarations"]
    assert decls[0]["name"] == "get_weather"
    assert decls[0]["description"] == "Look up weather"
    assert decls[0]["parametersJsonSchema"]["type"] == "object"
    assert "parameters" not in decls[0]
    assert sent["toolConfig"]["functionCallingConfig"] == {"mode": "ANY"}

    # normalized tool call
    assert res.finish_reason == "tool_calls"
    assert len(res.message.tool_calls) == 1
    tc = res.message.tool_calls[0]
    assert tc.id == "call_0"
    assert tc.name == "get_weather"
    assert json.loads(tc.arguments) == {"city": "SF"}


def test_gemini_tool_result_and_function_config():
    """Tool-result message → functionResponse keyed by name via id resolution;
    tool_choice=function → ANY + allowedFunctionNames."""
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]})
    chat, t = _chat(resp(200, body))
    chat.complete(
        llm.ChatInput(
            model="gemini-3.8-flash",
            messages=[
                llm.ChatMessage(role="user", content="weather?"),
                llm.ChatMessage(
                    role="assistant",
                    tool_calls=[llm.ToolCall(id="call_abc", name="get_weather", arguments='{"city":"SF"}')],
                ),
                llm.ChatMessage(role="tool", tool_call_id="call_abc", content='{"temp":70}'),
            ],
            tools=[llm.ToolDefinition(name="get_weather")],
            tool_choice=llm.ToolChoice(mode="function", function_name="get_weather"),
        )
    )
    sent = body_json(t.last)
    contents = sent["contents"]
    # assistant turn → model role with functionCall part
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][-1]["functionCall"]["name"] == "get_weather"
    # tool turn → user role with functionResponse keyed by resolved name
    fr = contents[2]["parts"][0]["functionResponse"]
    assert contents[2]["role"] == "user"
    assert fr["name"] == "get_weather"
    assert fr["response"]["result"] == {"temp": 70}
    # tool_choice=function → ANY + allowedFunctionNames
    assert sent["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["get_weather"],
    }


def test_gemini_stream():
    sse = "\n".join(
        [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
            "",
            'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}',
            "",
            'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}',
            "",
        ]
    )
    chat, t = _chat(resp(200, sse, content_type="text/event-stream"))
    text, finish, usage = "", "", None
    for d in chat.stream(
        llm.ChatInput(model="gemini-3.8-flash", messages=[llm.ChatMessage(role="user", content="hi")])
    ):
        text += d.content_delta
        if d.finish_reason:
            finish = d.finish_reason
        if d.usage:
            usage = d.usage
    assert text == "Hello"
    assert finish == "stop"
    assert usage is not None and usage.total_tokens == 5
    # stream URL carries the SSE query
    assert t.last.url.endswith(":streamGenerateContent?alt=sse")


def test_gemini_stream_tool_call():
    sse = "\n".join(
        [
            'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"x"}}}]}}]}',
            "",
            'data: {"candidates":[{"finishReason":"STOP"}]}',
            "",
        ]
    )
    chat, t = _chat(resp(200, sse, content_type="text/event-stream"))
    tool_delta, finish = None, ""
    for d in chat.stream(
        llm.ChatInput(model="gemini-3.8-flash", messages=[llm.ChatMessage(role="user", content="hi")])
    ):
        if d.tool_call_delta:
            tool_delta = d.tool_call_delta
        if d.finish_reason:
            finish = d.finish_reason
    assert tool_delta is not None
    assert tool_delta.id == "call_0"
    assert tool_delta.function_name == "lookup"
    assert json.loads(tool_delta.arguments_delta) == {"q": "x"}
    # a terminal finishReason after a functionCall maps to tool_calls
    assert finish == "tool_calls"


def test_gemini_rate_limit_retry_delay():
    body = json.dumps(
        {
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}],
            }
        }
    )
    chat, t = _chat(resp(429, body))
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED
    assert ei.value.provider_message == "Resource exhausted"
    assert ei.value.cause["retryAfterSeconds"] == 30


def test_gemini_context_length_error():
    body = json.dumps({"error": {"code": 400, "message": "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", "status": "INVALID_ARGUMENT"}})
    chat, t = _chat(resp(400, body))
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED


def test_gemini_auth_error():
    chat, t = _chat(resp(401, json.dumps({"error": {"message": "API key not valid"}})))
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.AUTH_FAILED


def test_gemini_stream_error_frame_raises():
    """A provider `{"error":...}` frame after 200 OK surfaces as a ConnectorError
    rather than a silently truncated stream. Earlier deltas still reach the consumer."""
    sse = "\n".join(
        [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}',
            "",
            'data: {"error":{"code":500,"message":"internal mid-stream"}}',
            "",
        ]
    )
    chat, t = _chat(resp(200, sse, content_type="text/event-stream"))
    gen = chat.stream(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="hi")]))
    assert next(gen).content_delta == "Hi"
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    assert ei.value.provider_message == "internal mid-stream"


def test_gemini_requires_gemini_config():
    with pytest.raises(llm.ConnectorError) as ei:
        llm.Chat(llm.ProviderID.GEMINI, llm.OpenAICompatConfig(api_key="k"))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_gemini_prompt_block_content_filter():
    # No candidates, only promptFeedback.blockReason → content_filter (not unknown).
    body = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9}})
    chat, _t = _chat(resp(200, body))
    res = chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert res.finish_reason == "content_filter"
    assert res.message.content == ""


def test_gemini_stream_prompt_block_content_filter():
    sse = "\n".join(['data: {"promptFeedback":{"blockReason":"SAFETY"}}', "", ""])
    chat, _t = _chat(resp(200, sse, content_type="text/event-stream"))
    finishes = [d.finish_reason for d in chat.stream(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")])) if d.finish_reason]
    assert finishes == ["content_filter"]


def test_gemini_thought_parts_excluded_from_content():
    body = json.dumps({
        "candidates": [{"content": {"parts": [{"text": "secret cot", "thought": True}, {"text": "the answer"}]}, "finishReason": "STOP"}],
        "usageMetadata": {},
    })
    chat, _t = _chat(resp(200, body))
    res = chat.complete(llm.ChatInput(model="gemini-3-pro", messages=[llm.ChatMessage(role="user", content="hi")]))
    assert res.message.content == "the answer"


@pytest.mark.parametrize("body", ["{}", '{"usageMetadata":{"promptTokenCount":3},"modelVersion":"gemini-3.8-flash"}', '{"promptFeedback":{}}'])
def test_gemini_200_without_candidates_or_block_reason_is_empty_body_error(body):
    chat, _t = _chat(resp(200, body))
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.status_code == 200


# ---- provider_metadata: thought signatures ----

_TOOLS = [llm.ToolDefinition(name="get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})]


def _user(text="weather?"):
    return llm.ChatMessage(role="user", content=text)


def test_signature_and_vendor_id_captured_on_tool_call():
    body = json.dumps({
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"functionCall": {"id": "fc-1", "name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "sig-1"},
                {"functionCall": {"id": "fc-2", "name": "get_weather", "args": {"city": "Rome"}}},
            ]},
            "finishReason": "STOP",
        }],
    })
    chat, _t = _chat(resp(200, body))
    res = chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()], tools=_TOOLS))
    a, b = res.message.tool_calls
    assert (a.id, a.provider_metadata) == ("fc-1", {"gemini": {"thoughtSignature": "sig-1"}})
    # Parallel calls: only the first part is signed.
    assert (b.id, b.provider_metadata) == ("fc-2", None)
    assert res.message.provider_metadata is None


def test_synthetic_ids_are_index_based_for_parallel_same_name_calls():
    body = json.dumps({"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "s"},
        {"functionCall": {"name": "get_weather", "args": {"city": "Rome"}}},
    ]}}]})
    chat, _t = _chat(resp(200, body))
    res = chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()], tools=_TOOLS))
    assert [tc.id for tc in res.message.tool_calls] == ["call_0", "call_1"]


def test_stream_captures_signature_and_index_based_ids():
    sse = "\n".join([
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"sig-s"}]}}]}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Rome"}}}]},"finishReason":"STOP"}]}',
        "",
    ])
    chat, _t = _chat(resp(200, sse, content_type="text/event-stream"))
    tcds = [d.tool_call_delta for d in chat.stream(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()], tools=_TOOLS)) if d.tool_call_delta]
    assert [(d.index, d.id) for d in tcds] == [(0, "call_0"), (1, "call_1")]
    assert tcds[0].provider_metadata == {"gemini": {"thoughtSignature": "sig-s"}}
    assert tcds[1].provider_metadata is None


def _replayed_model_turn(*tool_calls, tool_ids=None):
    chat, t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})))
    msgs = [_user(), llm.ChatMessage(role="assistant", tool_calls=list(tool_calls))]
    for tc in tool_calls:
        msgs.append(llm.ChatMessage(role="tool", tool_call_id=tc.id, content='{"temp":20}'))
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=msgs, tools=_TOOLS))
    return body_json(t.last)["contents"]


def test_replay_puts_stored_signature_back_and_echoes_vendor_ids():
    contents = _replayed_model_turn(
        llm.ToolCall(id="fc-1", name="get_weather", arguments='{"city":"Paris"}', provider_metadata={"gemini": {"thoughtSignature": "sig-1"}}),
        llm.ToolCall(id="fc-2", name="get_weather", arguments='{"city":"Rome"}'),
    )
    assert contents[1] == {"role": "model", "parts": [
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "fc-1"}, "thoughtSignature": "sig-1"},
        {"functionCall": {"name": "get_weather", "args": {"city": "Rome"}, "id": "fc-2"}},
    ]}
    responses = [p["functionResponse"] for p in contents[2]["parts"]]
    assert [(r["name"], r.get("id")) for r in responses] == [("get_weather", "fc-1"), ("get_weather", "fc-2")]


def test_replay_without_signature_uses_dummy_on_first_call_only():
    contents = _replayed_model_turn(
        llm.ToolCall(id="call_0", name="get_weather", arguments='{"city":"Paris"}'),
        llm.ToolCall(id="call_1", name="get_weather", arguments='{"city":"Rome"}'),
    )
    parts = contents[1]["parts"]
    assert parts[0]["thoughtSignature"] == "skip_thought_signature_validator"
    assert "thoughtSignature" not in parts[1]
    # Synthetic ids are never sent back: Gemini only knows the ids it issued.
    assert "id" not in parts[0]["functionCall"] and "id" not in parts[1]["functionCall"]
    assert all("id" not in p["functionResponse"] for p in contents[2]["parts"])


def test_dummy_is_per_model_turn():
    chat, t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})))
    msgs = [
        _user(),
        llm.ChatMessage(role="assistant", tool_calls=[llm.ToolCall(id="fc-1", name="get_weather", arguments="{}", provider_metadata={"gemini": {"thoughtSignature": "sig-1"}})]),
        llm.ChatMessage(role="tool", tool_call_id="fc-1", content="sunny"),
        llm.ChatMessage(role="assistant", content="Next.", tool_calls=[llm.ToolCall(id="call_0", name="get_weather", arguments="{}")]),
        llm.ChatMessage(role="tool", tool_call_id="call_0", content="rainy"),
    ]
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=msgs, tools=_TOOLS))
    contents = body_json(t.last)["contents"]
    assert contents[1]["parts"][0]["thoughtSignature"] == "sig-1"
    assert contents[3]["parts"][0] == {"text": "Next."}
    assert contents[3]["parts"][1]["thoughtSignature"] == "skip_thought_signature_validator"


def test_replay_ignores_foreign_provider_metadata():
    contents = _replayed_model_turn(
        llm.ToolCall(id="toolu_01", name="get_weather", arguments="{}", provider_metadata={"anthropic": {"thoughtSignature": "not-gemini"}}),
    )
    part = contents[1]["parts"][0]
    assert part["thoughtSignature"] == "skip_thought_signature_validator"
    assert "not-gemini" not in json.dumps(contents)


def test_round_trip_from_result_to_next_request():
    body = json.dumps({"candidates": [{"content": {"parts": [
        {"text": "Let me check."},
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "sig-rt"},
    ]}, "finishReason": "STOP"}]})
    chat, t = _chat(resp(200, body))
    res = chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()], tools=_TOOLS))
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", tools=_TOOLS, messages=[
        _user(),
        llm.ChatMessage(role="assistant", content=res.message.content, tool_calls=res.message.tool_calls,
                        provider_metadata=res.message.provider_metadata),
        llm.ChatMessage(role="tool", tool_call_id=res.message.tool_calls[0].id, content="sunny"),
    ]))
    model_turn = body_json(t.last)["contents"][1]
    assert model_turn["parts"][1]["thoughtSignature"] == "sig-rt"


# ---- schemas, usage, finish reasons, errors ----


def test_json_schema_response_format_uses_response_json_schema():
    chat, t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]})))
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()],
                                response_format=llm.ResponseFormat(type="json_schema", json_schema_name="x", json_schema=schema)))
    gc = body_json(t.last)["generationConfig"]
    assert gc["responseJsonSchema"] == schema and "responseSchema" not in gc
    assert gc["responseMimeType"] == "application/json"


def test_unknown_effort_falls_back_to_medium_budget():
    chat, t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}]})))
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()], reasoning=llm.ReasoningOptions(effort="extreme")))
    assert body_json(t.last)["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 8192}


def test_usage_counts_thought_tokens_as_output():
    body = json.dumps({
        "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 40, "totalTokenCount": 55},
    })
    chat, _t = _chat(resp(200, body))
    u = chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])).usage
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (10, 45, 55)


def test_stream_reports_cumulative_usage_once():
    sse = "\n".join([
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"totalTokenCount":11}}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12}}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"thoughtsTokenCount":7,"totalTokenCount":20}}',
        "",
    ])
    chat, _t = _chat(resp(200, sse, content_type="text/event-stream"))
    deltas = list(chat.stream(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])))
    usages = [d.usage for d in deltas if d.usage]
    assert len(usages) == 1
    assert (usages[0].input_tokens, usages[0].output_tokens, usages[0].total_tokens) == (10, 10, 20)
    assert deltas[-1].finish_reason == "stop" and deltas[-1].usage is usages[0]
    assert "".join(d.content_delta for d in deltas) == "Hello!"


@pytest.mark.parametrize(
    "reason,expected",
    [("MALFORMED_FUNCTION_CALL", "unknown"), ("UNEXPECTED_TOOL_CALL", "unknown"), ("IMAGE_SAFETY", "content_filter"), ("OTHER", "unknown")],
)
def test_finish_reason_mapping(reason, expected):
    chat, _t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": reason}]})))
    assert chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])).finish_reason == expected


_API_KEY_INVALID = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
                "metadata": {"service": "generativelanguage.googleapis.com"},
            },
            {"@type": "type.googleapis.com/google.rpc.LocalizedMessage", "locale": "en-US", "message": "API key not valid. Please pass a valid API key."},
        ],
    }
}


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (400, _API_KEY_INVALID, llm.ProviderCode.AUTH_FAILED),
        (400, {"error": {"code": 400, "message": "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", "status": "INVALID_ARGUMENT"}}, llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED),
        (400, {"error": {"code": 400, "message": "Invalid JSON payload received. Unknown name \"context\" at 'generation_config'.", "status": "INVALID_ARGUMENT"}}, llm.ProviderCode.INVALID_REQUEST),
        (400, {"error": {"code": 400, "message": "Function call is missing a thought_signature in functionCall parts.", "status": "INVALID_ARGUMENT"}}, llm.ProviderCode.INVALID_REQUEST),
        (402, {"error": {"code": 402, "message": "Your Prepay credit balance is depleted.", "status": "RESOURCE_EXHAUSTED"}}, llm.ProviderCode.RATE_LIMITED),
        (504, {"error": {"code": 504, "message": "Deadline exceeded", "status": "DEADLINE_EXCEEDED"}}, llm.ProviderCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_error_classification(status, body, expected):
    chat, _t = _chat(resp(status, json.dumps(body)))
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()]))
    assert ei.value.provider_code is expected


def test_stream_error_frame_api_key_invalid_without_code():
    body = {"error": {k: v for k, v in _API_KEY_INVALID["error"].items() if k != "code"}}
    sse = "data: " + json.dumps(body) + "\n\n"
    chat, _t = _chat(resp(200, sse, content_type="text/event-stream"))
    with pytest.raises(llm.ConnectorError) as ei:
        list(chat.stream(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])))
    assert ei.value.provider_code is llm.ProviderCode.AUTH_FAILED


# ---- review-5 ----


def test_tool_result_name_resolves_against_the_preceding_turn():
    chat, t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})))
    msgs = [
        _user(),
        llm.ChatMessage(role="assistant", tool_calls=[llm.ToolCall(id="call_0", name="get_weather", arguments="{}")]),
        llm.ChatMessage(role="tool", tool_call_id="call_0", content='{"temp":20}'),
        llm.ChatMessage(role="assistant", tool_calls=[llm.ToolCall(id="call_0", name="get_time", arguments="{}")]),
        llm.ChatMessage(role="tool", tool_call_id="call_0", content='{"t":20}'),
    ]
    chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=msgs, tools=_TOOLS))
    contents = body_json(t.last)["contents"]
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "get_weather"
    assert contents[4]["parts"][0]["functionResponse"]["name"] == "get_time"


def test_usage_counts_tool_use_prompt_tokens_as_input():
    body = json.dumps({
        "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "toolUsePromptTokenCount": 4, "candidatesTokenCount": 5, "thoughtsTokenCount": 6, "totalTokenCount": 25},
    })
    chat, _t = _chat(resp(200, body))
    u = chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])).usage
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (14, 11, 25)


@pytest.mark.parametrize("reason,expected", [("CONTINUATION", "length"), ("LANGUAGE", "unknown"), ("IMAGE_RECITATION", "content_filter"), ("NO_IMAGE", "unknown")])
def test_finish_reason_enum_values(reason, expected):
    chat, _t = _chat(resp(200, json.dumps({"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": reason}]})))
    assert chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])).finish_reason == expected


@pytest.mark.parametrize(
    "status,message,expected",
    [
        (408, "Request timeout", llm.ProviderCode.PROVIDER_UNAVAILABLE),
        (409, "Aborted", llm.ProviderCode.INVALID_REQUEST),
        (400, "The input token count is required", llm.ProviderCode.INVALID_REQUEST),
    ],
)
def test_status_classification_review5(status, message, expected):
    chat, _t = _chat(resp(status, json.dumps({"error": {"code": status, "message": message}})))
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()]))
    assert ei.value.provider_code is expected


def test_stream_error_frame_without_code_or_known_status_is_unknown():
    sse = 'data: {"error":{"message":"odd","status":"SOMETHING_NEW"}}\n\n'
    chat, _t = _chat(resp(200, sse, content_type="text/event-stream"))
    with pytest.raises(llm.ConnectorError) as ei:
        list(chat.stream(llm.ChatInput(model="gemini-3.8-flash", messages=[_user()])))
    assert ei.value.provider_code is llm.ProviderCode.UNKNOWN and ei.value.status_code == 200
