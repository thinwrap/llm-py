"""Bedrock native-adapter tests: Converse request build + SigV4 signing,
normalized result parse, tool-use, session-token signing, the non-incremental
streaming fallback, and error mapping."""

from __future__ import annotations

import datetime as dt
import json
import re
from urllib.parse import quote

import pytest
from helpers import FakeTransport, body_json, path_of, resp

import thinwrap.llm as llm
from thinwrap.llm._awssigv4 import SigV4Params, sign_aws_v4

_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_HOST = "bedrock-runtime.us-east-1.amazonaws.com"


def _resign(req, *, path, content_type):
    """Recompute the SigV4 Authorization over EXACTLY the bytes/date the connector
    used, varying only `path` / `content_type`, so a test can assert which
    canonical inputs the emitted signature actually committed to."""
    now = dt.datetime.strptime(req.headers["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    return sign_aws_v4(
        SigV4Params(
            method="POST", service="bedrock", region="us-east-1", host=_HOST,
            path=path, payload=req.body, access_key_id="AKIDEXAMPLE",
            secret_access_key="wSecret", signed={"Content-Type": content_type}, now=now,
        )
    )["Authorization"]


def _cfg(**kw):
    base = dict(region="us-east-1", access_key_id="AKIDEXAMPLE", secret_access_key="wSecret")
    base.update(kw)
    return llm.BedrockConfig(**base)


def _chat(t, **kw):
    return llm.Chat(llm.ProviderID.BEDROCK, _cfg(**kw), transport=t)


def test_complete_signing_and_result():
    body = json.dumps(
        {
            "output": {"message": {"content": [{"text": "Hi "}, {"text": "there"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 12, "totalTokens": 21},
        }
    )
    t = FakeTransport(resp(200, body))
    chat = _chat(t)
    res = chat.complete(
        llm.ChatInput(
            model=_MODEL,
            messages=[
                llm.ChatMessage(role="system", content="be brief"),
                llm.ChatMessage(role="user", content="Hello"),
            ],
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=256,
            stop=["STOP"],
        )
    )

    # Normalized result.
    assert res.message.content == "Hi there"
    assert res.finish_reason == "stop"
    assert res.usage.input_tokens == 9 and res.usage.output_tokens == 12 and res.usage.total_tokens == 21
    assert res.model == _MODEL

    # Percent-encoded model id in the Converse path.
    assert path_of(t.last) == f"/model/{_MODEL.replace(':', '%3A')}/converse"

    # SigV4 Authorization header present + well-formed.
    auth = t.last.headers["Authorization"]
    assert auth.startswith(
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"
    ) and "/us-east-1/bedrock/aws4_request" in auth
    m = re.search(r"Credential=AKIDEXAMPLE/(\d{8})/us-east-1/bedrock/aws4_request", auth)
    assert m is not None
    signed_headers = auth.split("SignedHeaders=")[1].split(",")[0]
    assert "content-type" in signed_headers and "host" in signed_headers
    assert "x-amz-content-sha256" in signed_headers and "x-amz-date" in signed_headers
    signature = auth.split("Signature=")[1]
    assert re.fullmatch(r"[0-9a-f]{64}", signature)

    # Amz date header, and the credential-scope date agrees with it.
    amz_date = t.last.headers["X-Amz-Date"]
    assert re.fullmatch(r"\d{8}T\d{6}Z", amz_date)
    assert amz_date[:8] == m.group(1)
    # Host is NOT attached as a request header (the transport sets it).
    assert "Host" not in t.last.headers
    assert t.last.headers["Content-Type"] == "application/json"

    # Converse request body shape.
    sent = body_json(t.last)
    assert sent["system"] == [{"text": "be brief"}]
    assert sent["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]
    assert sent["inferenceConfig"] == {
        "maxTokens": 256,
        "temperature": 0.5,
        "topP": 0.9,
        "stopSequences": ["STOP"],
    }


def test_default_max_tokens_and_config_default():
    t = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"end_turn"}'))
    # No max_output_tokens on the input, no default on the config → 4096.
    _chat(t).complete(llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="hi")]))
    assert body_json(t.last)["inferenceConfig"]["maxTokens"] == 4096

    t2 = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"end_turn"}'))
    # Config default is used when the input omits it.
    _chat(t2, default_max_tokens=1000).complete(
        llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="hi")])
    )
    assert body_json(t2.last)["inferenceConfig"]["maxTokens"] == 1000


def test_complete_tool_use():
    body = json.dumps(
        {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 15, "outputTokens": 7},
        }
    )
    t = FakeTransport(resp(200, body))
    res = _chat(t).complete(
        llm.ChatInput(
            model=_MODEL,
            messages=[llm.ChatMessage(role="user", content="weather?")],
            tools=[
                llm.ToolDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
            tool_choice=llm.ToolChoice(mode="required"),
        )
    )

    # Tool-use parsed into a ToolCall with JSON-stringified input.
    assert res.finish_reason == "tool_calls"
    assert res.message.content == ""
    assert len(res.message.tool_calls) == 1
    tc = res.message.tool_calls[0]
    assert tc.id == "tu_1" and tc.name == "get_weather"
    assert json.loads(tc.arguments) == {"city": "Paris"}
    # totalTokens absent → derived.
    assert res.usage.total_tokens == 22

    # toolConfig on the wire, tool-choice "required" → {"any": {}}.
    sent = body_json(t.last)
    assert sent["toolConfig"]["tools"] == [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}},
            }
        }
    ]
    assert sent["toolConfig"]["toolChoice"] == {"any": {}}


def test_tool_result_folds_into_user_turn():
    t = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}'))
    _chat(t).complete(
        llm.ChatInput(
            model=_MODEL,
            messages=[
                llm.ChatMessage(role="user", content="weather?"),
                llm.ChatMessage(
                    role="assistant",
                    tool_calls=[llm.ToolCall(id="tu_1", name="get_weather", arguments='{"city":"Paris"}')],
                ),
                llm.ChatMessage(role="tool", tool_call_id="tu_1", content="sunny"),
            ],
        )
    )
    sent = body_json(t.last)
    # assistant turn carries the toolUse block (input re-parsed to an object).
    assert sent["messages"][1] == {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}}],
    }
    # tool result becomes a user turn with a toolResult block.
    assert sent["messages"][2] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "tu_1", "content": [{"text": "sunny"}]}}],
    }


def test_session_token_signing():
    t = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"end_turn"}'))
    _chat(t, session_token="sess-tok-123").complete(
        llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="hi")])
    )
    # Session token surfaced as X-Amz-Security-Token and folded into SignedHeaders.
    assert t.last.headers["X-Amz-Security-Token"] == "sess-tok-123"
    auth = t.last.headers["Authorization"]
    signed_headers = auth.split("SignedHeaders=")[1].split(",")[0]
    assert "x-amz-security-token" in signed_headers


def test_stream_fallback_deltas():
    body = json.dumps(
        {
            "output": {
                "message": {
                    "content": [
                        {"text": "Hello world"},
                        {"toolUse": {"toolUseId": "tu_9", "name": "lookup", "input": {"q": "x"}}},
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
        }
    )
    t = FakeTransport(resp(200, body))
    deltas = list(
        _chat(t).stream(llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="hi")]))
    )

    # content delta, then one tool-call delta, then a terminal finish/usage delta.
    assert len(deltas) == 3
    assert deltas[0].content_delta == "Hello world"
    assert deltas[0].tool_call_delta is None
    tcd = deltas[1].tool_call_delta
    assert tcd is not None and tcd.index == 0 and tcd.id == "tu_9" and tcd.function_name == "lookup"
    assert json.loads(tcd.arguments_delta) == {"q": "x"}
    term = deltas[2]
    assert term.finish_reason == "tool_calls"
    assert term.usage is not None and term.usage.total_tokens == 7
    assert term.raw is not None
    # The fallback still issues a single POST to /converse.
    assert path_of(t.last).endswith("/converse")


def test_error_mapping_throttling():
    t = FakeTransport(resp(400, '{"message":"Request was throttled, please slow down","__type":"ThrottlingException"}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t).complete(llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED
    assert ei.value.status_code == 400
    assert "throttled" in ei.value.provider_message


def test_error_mapping_auth():
    t = FakeTransport(resp(403, '{"Message":"The security token included in the request is invalid"}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t).complete(llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.AUTH_FAILED
    assert ei.value.provider_message == "The security token included in the request is invalid"


def test_error_mapping_context_length():
    t = FakeTransport(resp(400, '{"message":"Input is too long for the requested model context"}'))
    with pytest.raises(llm.ConnectorError) as ei:
        _chat(t).complete(llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED


def test_bad_config_type():
    with pytest.raises(llm.ConnectorError) as ei:
        llm.Chat(llm.ProviderID.BEDROCK, llm.OpenAICompatConfig(api_key="k"))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_sigv4_canonical_path_is_double_encoded_for_colon_model():
    """The WIRE URL is single-encoded (%3A); the SIGNED canonical path is
    DOUBLE-encoded (%253A). We prove which one the signature committed to by
    recomputing it over both and asserting the emitted Authorization equals the
    DOUBLE-encoded one (and NOT the single-encoded one)."""
    t = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"end_turn"}'))
    _chat(t).complete(llm.ChatInput(model=_MODEL, messages=[llm.ChatMessage(role="user", content="hi")]))
    req = t.last

    # WIRE path: single-encoded — the colon is %3A, never %253A.
    wire = path_of(req)
    assert "%3A" in wire and "%253A" not in wire
    assert wire == f"/model/{_MODEL.replace(':', '%3A')}/converse"

    single = "/model/" + quote(_MODEL, safe="") + "/converse"
    double = "/model/" + quote(quote(_MODEL, safe=""), safe="") + "/converse"
    # Sanity: for a colon-bearing id the two encodings genuinely differ.
    assert single != double and "%3A" in single and "%253A" in double

    emitted = req.headers["Authorization"]
    assert emitted == _resign(req, path=double, content_type="application/json")
    assert emitted != _resign(req, path=single, content_type="application/json")


def test_sigv4_colon_free_model_unaffected():
    """A colon-free model id double-encodes to itself → no regression: the wire
    and canonical paths coincide and the signature commits to that same path."""
    plain = "amazon.titan-text-express-v1"
    t = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"end_turn"}'))
    _chat(t).complete(llm.ChatInput(model=plain, messages=[llm.ChatMessage(role="user", content="hi")]))
    req = t.last
    path = "/model/" + quote(plain, safe="") + "/converse"
    assert path_of(req) == path  # no percent-encoding to double up
    assert req.headers["Authorization"] == _resign(req, path=path, content_type="application/json")


def test_bedrock_content_type_sign_equals_send():
    """A passthrough Content-Type override must not desync signature from wire:
    post_raw force-sends application/json, so the signature must commit to
    application/json too (NOT to the override)."""
    t = FakeTransport(resp(200, '{"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"end_turn"}'))
    _chat(t).complete(
        llm.ChatInput(
            model=_MODEL,
            messages=[llm.ChatMessage(role="user", content="hi")],
            passthrough=llm.Passthrough(headers={"Content-Type": "application/x-amz-json-1.1"}),
        )
    )
    req = t.last
    # SENT Content-Type is the forced application/json.
    assert req.headers["Content-Type"] == "application/json"
    canonical = "/model/" + quote(quote(_MODEL, safe=""), safe="") + "/converse"
    # SIGNED value == SENT value (application/json), NOT the override.
    assert req.headers["Authorization"] == _resign(req, path=canonical, content_type="application/json")
    assert req.headers["Authorization"] != _resign(req, path=canonical, content_type="application/x-amz-json-1.1")
