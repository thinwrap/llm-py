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
            "modelVersion": "gemini-1.5-pro-002",
            "candidates": [
                {"content": {"parts": [{"text": "Hi "}, {"text": "there"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 12, "totalTokenCount": 21},
        }
    )
    chat, t = _chat(resp(200, body))
    res = chat.complete(
        llm.ChatInput(
            model="gemini-1.5-pro",
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
    assert t.last.url.endswith("/models/gemini-1.5-pro:generateContent")
    assert path_of(t.last) == "/v1beta/models/gemini-1.5-pro:generateContent"
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
    assert res.model == "gemini-1.5-pro-002"
    assert res.usage.input_tokens == 9 and res.usage.output_tokens == 12 and res.usage.total_tokens == 21


def test_gemini_default_max_tokens_fallback():
    body = json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})
    t = FakeTransport(resp(200, body))
    chat = llm.Chat(
        llm.ProviderID.GEMINI, llm.GeminiConfig(api_key="k", default_max_tokens=1000), transport=t
    )
    chat.complete(llm.ChatInput(model="gemini-1.5-flash", messages=[llm.ChatMessage(role="user", content="hi")]))
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
            model="gemini-1.5-pro",
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
    assert decls[0]["parameters"]["type"] == "object"
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
            model="gemini-1.5-pro",
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
        llm.ChatInput(model="gemini-1.5-flash", messages=[llm.ChatMessage(role="user", content="hi")])
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
        llm.ChatInput(model="gemini-1.5-pro", messages=[llm.ChatMessage(role="user", content="hi")])
    ):
        if d.tool_call_delta:
            tool_delta = d.tool_call_delta
        if d.finish_reason:
            finish = d.finish_reason
    assert tool_delta is not None
    assert tool_delta.id == "call_lookup"
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
    body = json.dumps({"error": {"message": "The input token count exceeds the maximum context"}})
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
