"""Core foundation tests: compat chat (complete + stream), embeddings, dispatch,
error mapping, SSE parsing, and SigV4 structure."""

from __future__ import annotations

import datetime as dt
import types

import pytest
from helpers import FakeTransport, body_json, path_of, resp

import thinwrap.llm as llm
from thinwrap.llm import _awssigv4 as sigv4
from thinwrap.llm.sse import parse_sse_lines
from thinwrap.llm.transport import HttpResponse


def _openai(t):
    return llm.Chat(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)


def _one_user(model="m"):
    return llm.ChatInput(model=model, messages=[llm.ChatMessage(role="user", content="hi")])


def test_chat_complete():
    body = '{"model":"gpt-4o-mini","usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21},"choices":[{"message":{"content":"Hi there"},"finish_reason":"stop"}]}'
    t = FakeTransport(resp(200, body))
    chat = llm.Chat(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="sk-x"), transport=t)
    res = chat.complete(llm.ChatInput(model="gpt-4o-mini", messages=[llm.ChatMessage(role="user", content="Hello")]))
    assert res.message.content == "Hi there" and res.finish_reason == "stop"
    assert res.usage.input_tokens == 9 and res.usage.output_tokens == 12 and res.usage.total_tokens == 21
    assert path_of(t.last) == "/v1/chat/completions"
    assert t.last.headers["Authorization"] == "Bearer sk-x"
    assert body_json(t.last)["model"] == "gpt-4o-mini"


def test_base_url_required():
    with pytest.raises(llm.ConnectorError) as ei:
        llm.Chat(llm.ProviderID.AZURE_OPENAI, llm.OpenAICompatConfig(api_key="k"))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_unknown_provider():
    with pytest.raises(llm.ConnectorError):
        llm.Chat("not-a-provider", llm.OpenAICompatConfig(api_key="k"))


def test_chat_stream():
    sse = "\n".join([
        'data: {"choices":[{"delta":{"content":"Hel"}}]}', "",
        'data: {"choices":[{"delta":{"content":"lo"}}]}', "",
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}', "",
        "data: [DONE]", "",
    ])
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    chat = llm.Chat(llm.ProviderID.GROQ, llm.OpenAICompatConfig(api_key="k"), transport=t)
    text, finish, usage = "", "", None
    for d in chat.stream(llm.ChatInput(model="llama", messages=[llm.ChatMessage(role="user", content="hi")])):
        text += d.content_delta
        if d.finish_reason:
            finish = d.finish_reason
        if d.usage:
            usage = d.usage
    assert text == "Hello" and finish == "stop"
    assert usage is not None and usage.total_tokens == 5
    assert body_json(t.last)["stream"] is True


def test_error_mapping():
    t = FakeTransport(resp(429, '{"error":{"message":"slow down"}}'))
    chat = llm.Chat(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED
    assert ei.value.provider_message == "slow down"


def test_context_length_error():
    t = FakeTransport(resp(400, '{"error":{"message":"This model maximum context length is 8192 tokens"}}'))
    chat = llm.Chat(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)
    with pytest.raises(llm.ConnectorError) as ei:
        chat.complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")]))
    assert ei.value.provider_code is llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED


def test_embeddings():
    body = '{"model":"text-embedding-3-small","data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":5,"total_tokens":5}}'
    t = FakeTransport(resp(200, body))
    emb = llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)
    res = emb.create(llm.EmbeddingsInput(model="text-embedding-3-small", input=["a", "b"]))
    assert len(res.embeddings) == 2 and res.embeddings[0][0] == 0.1 and res.embeddings[1][0] == 0.3
    assert res.usage.input_tokens == 5


def test_embeddings_unsupported():
    with pytest.raises(llm.ConnectorError) as ei:
        llm.Embeddings(llm.ProviderID.DEEPSEEK, llm.OpenAICompatConfig(api_key="k"))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_embeddings_passthrough_query():
    body = '{"model":"m","data":[{"index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":1,"total_tokens":1}}'
    t = FakeTransport(resp(200, body))
    emb = llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)
    emb.create(llm.EmbeddingsInput(model="m", input=["a"], passthrough=llm.Passthrough(query={"api-version": "2024-02-01"})))
    assert t.last.url.endswith("/embeddings?api-version=2024-02-01")


def test_together_eos_finish_reason_maps_to_stop():
    body = '{"model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"choices":[{"message":{"content":"ok"},"finish_reason":"eos"}]}'
    res = _openai(FakeTransport(resp(200, body))).complete(_one_user())
    assert res.finish_reason == "stop"


def test_buffered_stream_splits_only_on_sse_terminators():
    # U+2028 (LINE SEPARATOR) inside a JSON string must NOT split the SSE line —
    # str.splitlines() would, corrupting the payload.
    from thinwrap.llm.transport import BufferedStream

    payload = 'data: {"text":"a b"}'
    stream = BufferedStream(HttpResponse(status=200, headers={}, body=(payload + "\n\n").encode("utf-8")))
    lines = list(stream.iter_lines())
    assert payload in lines


def test_parse_sse_lines():
    lines = ['data: {"a":1}', "", ": comment", 'data: {"b":2}', "", "data: [DONE]", ""]
    payloads = list(parse_sse_lines(lines))
    assert payloads == ['{"a":1}', '{"b":2}']


def test_sse_data_strips_exactly_one_leading_space():
    # Per the SSE spec, only ONE optional leading space is stripped; further
    # spaces are significant payload.
    assert list(parse_sse_lines(['data:{"a":1}', ""])) == ['{"a":1}']
    assert list(parse_sse_lines(['data: {"a":1}', ""])) == ['{"a":1}']
    assert list(parse_sse_lines(['data:  {"a":1}', ""])) == [' {"a":1}']


def test_stream_error_frame_raises_mid_stream():
    """A provider `{"error":...}` frame after 200 OK surfaces as a ConnectorError
    rather than a silently truncated stream; earlier deltas still reach the consumer."""
    sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}', "",
            'data: {"error":{"message":"boom mid-stream"}}', "",
        ]
    )
    gen = _openai(FakeTransport(resp(200, sse, content_type="text/event-stream"))).stream(_one_user())
    assert next(gen).content_delta == "Hi"
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    assert ei.value.provider_message == "boom mid-stream"


def test_stream_multiple_tool_calls_in_one_chunk():
    """A single delta carrying several tool_calls emits one ToolCallDelta each
    (the old code read only tool_calls[0])."""
    chunk = (
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"id":"call_a","function":{"name":"f","arguments":"{}"}},'
        '{"index":1,"id":"call_b","function":{"name":"g","arguments":"{}"}}'
        "]}}]}"
    )
    sse = "\n".join([chunk, "", "data: [DONE]", ""])
    gen = _openai(FakeTransport(resp(200, sse, content_type="text/event-stream"))).stream(_one_user())
    tcds = [d.tool_call_delta for d in gen if d.tool_call_delta is not None]
    assert [(tc.index, tc.id, tc.function_name) for tc in tcds] == [(0, "call_a", "f"), (1, "call_b", "g")]


def test_transport_error_does_not_leak_credentials():
    """A BYO transport failure normalizes to PROVIDER_UNAVAILABLE without leaking
    a credential-bearing URL via message/cause/__cause__/__context__."""

    class _Raising:
        def send(self, request):
            raise ConnectionResetError("reset while calling https://api/v1?api_key=SUPERSECRET")

    with pytest.raises(llm.ConnectorError) as ei:
        _openai(_Raising()).complete(_one_user())
    e = ei.value
    assert e.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert "SUPERSECRET" not in str(e)
    assert "SUPERSECRET" not in (e.provider_message or "")
    assert "SUPERSECRET" not in repr(e.cause)
    assert e.cause == {"raw": {"exception": "ConnectionResetError"}}
    # Neither implicit (__context__) nor explicit (__cause__) chaining retains the
    # URL-bearing transport exception.
    assert e.__cause__ is None and e.__context__ is None


def test_mid_stream_transport_read_error_becomes_connector_error():
    """A socket reset WHILE iterating stream lines normalizes to a ConnectorError
    (not a raw exception, not a clean EOF); the stream is still closed and the
    URL-bearing exception is not chained."""

    class _Exploding:
        def __init__(self):
            self.status, self.headers, self.closed = 200, {}, False

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
            yield ""
            raise ConnectionResetError("reset mid-stream https://host/v1?token=LEAKME")

        def read_all(self):
            return b""

        def close(self):
            self.closed = True

    class _StreamTransport:
        def __init__(self, stream):
            self._stream = stream

        def send(self, request):
            return HttpResponse(status=200, headers={}, body=b"")

        def open_stream(self, request):
            return self._stream

    stream = _Exploding()
    gen = _openai(_StreamTransport(stream)).stream(_one_user())
    assert next(gen).content_delta == "Hi"
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    e = ei.value
    assert e.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert "LEAKME" not in str(e) and "LEAKME" not in repr(e.cause)
    assert e.__cause__ is None and e.__context__ is None
    assert stream.closed  # the connector's finally still closed the stream


def test_urllib_stream_caps_line_length(monkeypatch):
    """A newline-less streaming body must not buffer unbounded — a line over the
    cap raises rather than growing memory. Normal short lines still iterate."""
    from thinwrap.llm import transport as tp

    monkeypatch.setattr(tp, "_MAX_SSE_LINE", 8)

    class _Resp:
        def __init__(self, data):
            self._data, self._pos = data, 0
            self.status, self.headers = 200, {}

        def readline(self, limit=-1):
            if self._pos >= len(self._data):
                return b""
            nl = self._data.find(b"\n", self._pos)
            end = len(self._data) if nl == -1 else nl + 1
            if limit >= 0:
                end = min(end, self._pos + limit)
            chunk = self._data[self._pos:end]
            self._pos = end
            return chunk

    with pytest.raises(llm.ConnectorError) as ei:
        list(tp._UrllibStream(_Resp(b"x" * 40)).iter_lines())
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE

    assert list(tp._UrllibStream(_Resp(b"a\nbb\n")).iter_lines()) == ["a", "bb"]


def test_sigv4_structure():
    p = sigv4.SigV4Params(
        method="POST", service="bedrock", region="us-east-1",
        host="bedrock-runtime.us-east-1.amazonaws.com", path="/model/m/converse",
        payload=b'{"x":1}', access_key_id="AKIDEXAMPLE", secret_access_key="wSecret",
        signed={"Content-Type": "application/json"},
        now=dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
    )
    h1 = sigv4.sign_aws_v4(p)
    h2 = sigv4.sign_aws_v4(p)
    auth = h1["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260102/us-east-1/bedrock/aws4_request")
    assert "content-type;host;x-amz-content-sha256;x-amz-date" in auth
    assert h1 == h2
    assert len(auth.split("Signature=")[1]) == 64


# ---------------------------------------------------------------------------
# Transport contract: a BYO transport that RAISES a non-2xx
# ---------------------------------------------------------------------------
#
# `Transport` requires a non-2xx to be returned. `requests`/`httpx` transports
# calling raise_for_status() do not — and without recovery every provider error
# collapses into provider_unavailable with no status. For an LLM caller that is
# the worst possible misreport: a rate limit becomes an outage, so the caller
# fails the request instead of backing off.


class _RaisingTransport:
    """Shape of requests/httpx: the exception carries a `.response`."""

    def __init__(self, status: int, payload: bytes) -> None:
        self._status = status
        self._payload = payload

    def send(self, request):
        error = RuntimeError(f"{self._status} Client Error for url: {request.url}")
        error.response = types.SimpleNamespace(
            status_code=self._status,
            headers={"Content-Type": "application/json"},
            content=self._payload,
        )
        raise error

    def open_stream(self, request):
        return self.send(request)


def test_raised_429_still_classifies_as_rate_limited():
    transport = _RaisingTransport(429, b'{"error":{"message":"Rate limit reached"}}')
    with pytest.raises(llm.ConnectorError) as excinfo:
        _openai(transport).complete(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="hi")]))
    error = excinfo.value
    assert error.status_code == 429
    assert error.provider_code is llm.ProviderCode.RATE_LIMITED
    assert "Rate limit reached" in (error.provider_message or "")


def test_raised_401_on_a_stream_still_classifies_as_auth_failed():
    # A non-2xx never streams, so the buffered wrapper is the faithful shape.
    transport = _RaisingTransport(401, b'{"error":{"message":"Invalid API key"}}')
    with pytest.raises(llm.ConnectorError) as excinfo:
        for _ in _openai(transport).stream(
            llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="hi")])
        ):
            pass
    assert excinfo.value.status_code == 401
    assert excinfo.value.provider_code is llm.ProviderCode.AUTH_FAILED


def test_transport_failure_without_a_status_stays_provider_unavailable():
    class _DeadTransport:
        def send(self, request):
            raise OSError("connection reset by peer")

    with pytest.raises(llm.ConnectorError) as excinfo:
        _openai(_DeadTransport()).complete(
            llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="hi")])
        )
    assert excinfo.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert excinfo.value.status_code is None
