"""Core foundation tests: compat chat (complete + stream), embeddings, dispatch,
error mapping, SSE parsing, and SigV4 structure."""

from __future__ import annotations

import datetime as dt
import io
import types
import urllib.error

import pytest
from helpers import FakeTransport, body_json, path_of, resp

import thinwrap.llm as llm
from thinwrap.llm import _awssigv4 as sigv4
from thinwrap.llm.sse import parse_sse_lines
from thinwrap.llm.transport import HttpRequest, HttpResponse, UrllibTransport


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


# ---- max-tokens field per provider ----


@pytest.mark.parametrize("pid", [llm.ProviderID.OPENAI, llm.ProviderID.AZURE_OPENAI])
def test_openai_and_azure_send_max_completion_tokens(pid):
    ok = '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'
    t = FakeTransport(resp(200, ok))
    base = "https://res.openai.azure.com/openai/v1" if pid is llm.ProviderID.AZURE_OPENAI else ""
    chat = llm.Chat(pid, llm.OpenAICompatConfig(api_key="k", base_url=base), transport=t)
    chat.complete(llm.ChatInput(model="o4-mini", messages=[llm.ChatMessage(role="user", content="hi")], max_output_tokens=256))
    sent = body_json(t.last)
    assert sent["max_completion_tokens"] == 256
    assert "max_tokens" not in sent


def test_other_compat_providers_keep_max_tokens():
    ok = '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'
    t = FakeTransport(resp(200, ok))
    chat = llm.Chat(llm.ProviderID.GROQ, llm.OpenAICompatConfig(api_key="k"), transport=t)
    chat.complete(llm.ChatInput(model="llama", messages=[llm.ChatMessage(role="user", content="hi")], max_output_tokens=64))
    sent = body_json(t.last)
    assert sent["max_tokens"] == 64
    assert "max_completion_tokens" not in sent


# ---- 2xx bodies that are not a result ----


def _openrouter(t):
    return llm.Chat(llm.ProviderID.OPENROUTER, llm.OpenAICompatConfig(api_key="k"), transport=t)


def test_openrouter_200_with_only_error_object_raises_mapped_error():
    # OpenRouter: an error after processing started is HTTP 200 with only `error`.
    body = '{"error":{"code":502,"message":"Provider returned error"}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _openrouter(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    # The status is the one the response arrived with; error.code only classifies.
    assert ei.value.status_code == 200
    assert ei.value.provider_message == "Provider returned error"
    assert ei.value.cause["raw"]["error"]["code"] == 502


def test_openrouter_200_error_code_429_maps_to_rate_limited():
    body = '{"error":{"code":429,"message":"Rate limit exceeded upstream"}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _openrouter(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED
    assert ei.value.status_code == 200


def test_200_error_object_with_string_code_classifies_by_name():
    body = '{"error":{"code":"server_error","message":"boom"}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _openrouter(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.status_code == 200


def test_openrouter_200_with_choice_finish_reason_error_raises():
    # OpenRouter: choices[0] with finish_reason "error" + error beside partial content.
    body = (
        '{"id":"gen-1","model":"openai/gpt-4o","choices":[{"index":0,'
        '"message":{"role":"assistant","content":"Partial answ"},'
        '"finish_reason":"error","error":{"code":502,"message":"Upstream disconnected"}}]}'
    )
    with pytest.raises(llm.ConnectorError) as ei:
        _openrouter(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.status_code == 200
    assert ei.value.provider_message == "Upstream disconnected"


def test_choice_finish_reason_error_without_error_object_raises():
    body = '{"choices":[{"message":{"content":"x"},"finish_reason":"error"}]}'
    with pytest.raises(llm.ConnectorError) as ei:
        _openrouter(FakeTransport(resp(200, body))).complete(_one_user())
    # No error code or type to classify by.
    assert ei.value.provider_code is llm.ProviderCode.UNKNOWN
    assert ei.value.status_code == 200


def test_200_error_object_without_usable_code_is_unknown():
    body = '{"error":{"message":"something failed"}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _openrouter(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.UNKNOWN and ei.value.status_code == 200


def test_embeddings_200_error_object_without_usable_code_is_unknown():
    with pytest.raises(llm.ConnectorError) as ei:
        _emb(FakeTransport(resp(200, '{"error":{"message":"odd"}}'))).create(llm.EmbeddingsInput(model="m", input=["a"]))
    assert ei.value.provider_code is llm.ProviderCode.UNKNOWN and ei.value.status_code == 200


@pytest.mark.parametrize("body", ["{}", '{"choices":[]}', '{"choices":null}', '{"id":"x","model":"m"}'])
def test_compat_200_without_choices_is_empty_body_error(body):
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.status_code == 200


def _emb(t):
    return llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)


def test_embeddings_200_error_object_raises_mapped_error():
    body = '{"error":{"code":503,"message":"No provider available"}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _emb(FakeTransport(resp(200, body))).create(llm.EmbeddingsInput(model="m", input=["a"]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.status_code == 200
    assert ei.value.provider_message == "No provider available"


@pytest.mark.parametrize("body", ["{}", '{"data":null,"model":"m"}'])
def test_embeddings_200_without_data_is_empty_body_error(body):
    with pytest.raises(llm.ConnectorError) as ei:
        _emb(FakeTransport(resp(200, body))).create(llm.EmbeddingsInput(model="m", input=["a"]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


def test_embeddings_count_mismatch_raises():
    body = '{"model":"m","data":[{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":2,"total_tokens":2}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _emb(FakeTransport(resp(200, body))).create(llm.EmbeddingsInput(model="m", input=["a", "b"]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert ei.value.cause["raw"]["model"] == "m"


@pytest.mark.parametrize("cell", ["null", '"0.5"', "true"])
def test_embeddings_non_numeric_cell_raises(cell):
    body = '{"model":"m","data":[{"index":0,"embedding":[0.1,' + cell + ']}],"usage":{"prompt_tokens":1,"total_tokens":1}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _emb(FakeTransport(resp(200, body))).create(llm.EmbeddingsInput(model="m", input=["a"]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


def test_embeddings_row_without_vector_raises():
    body = '{"model":"m","data":[{"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _emb(FakeTransport(resp(200, body))).create(llm.EmbeddingsInput(model="m", input=["a"]))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


# ---- chunk-array content (Mistral reasoning) ----


def _mistral(t):
    return llm.Chat(llm.ProviderID.MISTRAL, llm.OpenAICompatConfig(api_key="k"), transport=t)


def test_complete_with_chunk_array_content_keeps_only_text():
    # docs.mistral.ai reasoning: message.content is a list of ThinkChunk + TextChunk.
    body = (
        '{"model":"magistral-medium-latest","usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14},'
        '"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":['
        '{"type":"thinking","thinking":[{"type":"text","text":"Let me work this out."}]},'
        '{"type":"text","text":"The answer "},{"type":"text","text":"is 4."}]}}]}'
    )
    res = _mistral(FakeTransport(resp(200, body))).complete(_one_user())
    assert res.message.content == "The answer is 4."
    assert res.raw["choices"][0]["message"]["content"][0]["type"] == "thinking"


def test_complete_with_only_thinking_chunks_has_empty_content():
    body = (
        '{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":['
        '{"type":"thinking","thinking":[{"type":"text","text":"hmm"}]}]}}]}'
    )
    res = _mistral(FakeTransport(resp(200, body))).complete(_one_user())
    assert res.message.content == ""


def test_stream_with_chunk_array_deltas_emits_only_answer_text():
    # Thinking phase: list with a ThinkChunk; transition: closing ThinkChunk +
    # first TextChunk; then plain strings.
    sse = "\n".join([
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"Considering"}]}]}}]}', "",
        'data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":" done."}]},{"type":"text","text":"The answer"}]}}]}', "",
        'data: {"choices":[{"index":0,"delta":{"content":" is 4."}}]}', "",
        'data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}', "",
        "data: [DONE]", "",
    ])
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    deltas = list(_mistral(t).stream(_one_user()))
    assert "".join(d.content_delta for d in deltas) == "The answer is 4."
    assert [d.finish_reason for d in deltas if d.finish_reason] == ["stop"]


class _RecordingOpener:
    def __init__(self):
        self.requests = []

    def open(self, req, timeout):
        self.requests.append(req)
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("headers", "expected"), [
    # Cloudflare-fronted vendors (Groq) refuse urllib's own `Python-urllib/3.x`
    # with 403 error 1010.
    ({}, "thinwrap-llm-python"),
    ({"User-Agent": "acme/2"}, "acme/2"),
    ({"user-agent": "acme/2"}, "acme/2"),
])
def test_urllib_transport_user_agent(headers, expected, stream):
    tr = UrllibTransport()
    tr._opener = _RecordingOpener()
    req = HttpRequest(method="POST", url="https://api.example.com/x", headers=headers, body=b"{}")
    tr.open_stream(req) if stream else tr.send(req)
    sent = tr._opener.requests[0]
    assert sent.get_header("User-agent") == expected
    assert sum(1 for k, _ in sent.header_items() if k.lower() == "user-agent") == 1


# ---- stream error body, transport defaults, headers ----


class _ErrorBodyStream:
    """A non-2xx stream whose body read fails (connection reset while draining)."""

    def __init__(self, status):
        self.status, self.headers, self.closed = status, {}, False

    def iter_lines(self):
        return iter(())

    def read_all(self):
        raise ConnectionResetError("reset reading https://host/v1?key=LEAKME")

    def close(self):
        self.closed = True


class _OpenStreamTransport:
    def __init__(self, stream):
        self._stream = stream

    def send(self, request):  # pragma: no cover - stream path only
        raise AssertionError("not used")

    def open_stream(self, request):
        return self._stream


@pytest.mark.parametrize(
    "factory",
    [
        lambda t: _openai(t),
        lambda t: llm.Chat(llm.ProviderID.ANTHROPIC, llm.AnthropicConfig(api_key="k"), transport=t),
        lambda t: llm.Chat(llm.ProviderID.GEMINI, llm.GeminiConfig(api_key="k"), transport=t),
    ],
)
def test_stream_error_body_read_failure_is_guarded_and_closes(factory):
    stream = _ErrorBodyStream(429)
    with pytest.raises(llm.ConnectorError) as ei:
        factory(_OpenStreamTransport(stream)).stream(_one_user())
    e = ei.value
    assert e.provider_code is llm.ProviderCode.RATE_LIMITED and e.status_code == 429
    assert "LEAKME" not in str(e) and "LEAKME" not in repr(e.cause)
    assert stream.closed


@pytest.mark.parametrize("pid", [llm.ProviderID.VLLM, llm.ProviderID.OLLAMA, llm.ProviderID.LMSTUDIO])
def test_default_transport_refuses_http_local_defaults_with_actionable_error(pid):
    chat = llm.Chat(pid, llm.OpenAICompatConfig(), transport=UrllibTransport())
    for call in (chat.complete, chat.stream):
        with pytest.raises(llm.ConnectorError) as ei:
            call(_one_user())
        assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST
        assert "http://" in str(ei.value) and "Transport" in str(ei.value)
        assert "localhost" not in str(ei.value)


def test_default_transport_refuses_file_scheme():
    with pytest.raises(llm.ConnectorError) as ei:
        UrllibTransport().send(HttpRequest(method="GET", url="file:///etc/passwd"))
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_default_timeout_allows_long_non_stream_reasoning_calls():
    assert UrllibTransport()._timeout == 600.0
    assert UrllibTransport(timeout=5)._timeout == 5


@pytest.mark.parametrize("stream", [False, True])
def test_header_with_line_break_is_invalid_request_without_the_value(stream):
    t = FakeTransport(resp(200, "{}"))
    chat = llm.Chat(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="sk-SECRET\n"), transport=t)
    with pytest.raises(llm.ConnectorError) as ei:
        (chat.stream if stream else chat.complete)(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST
    assert "Authorization" in str(ei.value)
    assert "SECRET" not in str(ei.value) and "SECRET" not in (ei.value.provider_message or "")
    assert t.calls == []


@pytest.mark.parametrize("value", ["", "   ", "-5", "1.5", "+5", "soon", "tomorrow", "2015-10-21T07:28:00Z", "Wed, 21 Oct 2015 07:28:00 +0000"])
def test_unusable_retry_after_has_no_seconds(value):
    t = FakeTransport(resp(429, '{"error":{"message":"slow"}}', retry_after=value))
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(t).complete(_one_user())
    assert "retryAfterSeconds" not in ei.value.cause
    if not value.strip():
        assert "retryAfter" not in ei.value.cause


def test_future_retry_after_date_is_parsed():
    from email.utils import format_datetime

    when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=120)
    t = FakeTransport(resp(429, '{"error":{"message":"slow"}}', retry_after=format_datetime(when, usegmt=True)))
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(t).complete(_one_user())
    assert 100 <= ei.value.cause["retryAfterSeconds"] <= 120


@pytest.mark.parametrize(
    "status,expected", [(402, llm.ProviderCode.RATE_LIMITED), (413, llm.ProviderCode.INVALID_REQUEST)]
)
def test_compat_402_is_rate_limited_and_413_invalid_request(status, expected):
    t = FakeTransport(resp(status, '{"error":{"message":"Insufficient credits"}}'))
    with pytest.raises(llm.ConnectorError) as ei:
        llm.Chat(llm.ProviderID.OPENROUTER, llm.OpenAICompatConfig(api_key="k"), transport=t).complete(_one_user())
    assert ei.value.provider_code is expected


@pytest.mark.parametrize(
    "status,expected", [(402, llm.ProviderCode.RATE_LIMITED), (413, llm.ProviderCode.INVALID_REQUEST)]
)
def test_embeddings_402_is_rate_limited_and_413_invalid_request(status, expected):
    t = FakeTransport(resp(status, '{"error":{"message":"nope"}}'))
    emb = llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)
    with pytest.raises(llm.ConnectorError) as ei:
        emb.create(llm.EmbeddingsInput(model="text-embedding-3-small", input=["a"]))
    assert ei.value.provider_code is expected


def test_mid_stream_error_honours_integer_code():
    sse = "\n".join([
        'data: {"choices":[{"delta":{"content":"Hi"}}]}', "",
        'data: {"error":{"code":429,"message":"Upstream busy"},"choices":[{"delta":{},"finish_reason":"error"}]}', "",
    ])
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    chat = llm.Chat(llm.ProviderID.OPENROUTER, llm.OpenAICompatConfig(api_key="k"), transport=t)
    with pytest.raises(llm.ConnectorError) as ei:
        list(chat.stream(_one_user()))
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED


def test_mistral_embeddings_send_output_dimension():
    body = '{"model":"codestral-embed","data":[{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":1,"total_tokens":1}}'
    t = FakeTransport(resp(200, body))
    llm.Embeddings(llm.ProviderID.MISTRAL, llm.OpenAICompatConfig(api_key="k"), transport=t).create(
        llm.EmbeddingsInput(model="codestral-embed", input=["a"], dimensions=256)
    )
    sent = body_json(t.last)
    assert sent["output_dimension"] == 256 and "dimensions" not in sent

    t2 = FakeTransport(resp(200, body))
    llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t2).create(
        llm.EmbeddingsInput(model="text-embedding-3-small", input=["a"], dimensions=256)
    )
    assert body_json(t2.last)["dimensions"] == 256


def test_package_ships_py_typed_marker():
    from importlib import resources

    assert resources.files("thinwrap.llm").joinpath("py.typed").is_file()


def test_provider_metadata_defaults_to_none():
    assert llm.ToolCall(id="a", name="b", arguments="{}").provider_metadata is None
    assert llm.ChatMessage(role="user").provider_metadata is None
    assert llm.AssistantMessage(role="assistant", content="").provider_metadata is None
    assert llm.ToolCallDelta(index=0).provider_metadata is None
    assert llm.ChatStreamDelta().provider_metadata is None


# ---- stream termination: non-SSE bodies and truncated streams ----


def _anthropic(t):
    return llm.Chat(llm.ProviderID.ANTHROPIC, llm.AnthropicConfig(api_key="k"), transport=t)


def _gemini(t):
    return llm.Chat(llm.ProviderID.GEMINI, llm.GeminiConfig(api_key="k"), transport=t)


class _ClosingStream:
    def __init__(self, status, headers, body):
        self.status, self.headers, self._body, self.closed = status, headers, body, False

    def iter_lines(self):
        return iter(self._body.split("\n"))

    def read_all(self):
        return self._body.encode()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("factory", [_openai, _anthropic, _gemini])
@pytest.mark.parametrize("content_type", ["application/json", "text/html; charset=utf-8"])
def test_2xx_stream_with_non_sse_content_type_raises(factory, content_type):
    stream = _ClosingStream(200, {"content-type": content_type}, '{"choices":[]}')
    with pytest.raises(llm.ConnectorError) as ei:
        factory(_OpenStreamTransport(stream)).stream(_one_user())
    e = ei.value
    assert e.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE and e.status_code == 200
    assert content_type.split(";")[0] in str(e)
    assert stream.closed


@pytest.mark.parametrize("content_type", ["", "text/event-stream; charset=utf-8", "TEXT/EVENT-STREAM"])
def test_sse_or_missing_content_type_is_accepted(content_type):
    headers = {"content-type": content_type} if content_type else {}
    sse = 'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    deltas = list(_openai(_OpenStreamTransport(_ClosingStream(200, headers, sse))).stream(_one_user()))
    assert deltas[0].content_delta == "Hi"


@pytest.mark.parametrize(
    "sse",
    [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n',  # [DONE] alone
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',  # finish alone
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]',  # [DONE] at EOF without a blank line
    ],
)
def test_compat_stream_with_terminal_event_completes(sse):
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    assert "".join(d.content_delta for d in _openai(t).stream(_one_user())) == "Hi"


def test_compat_stream_truncated_before_terminal_event_raises():
    t = FakeTransport(resp(200, 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n', content_type="text/event-stream"))
    gen = _openai(t).stream(_one_user())
    assert next(gen).content_delta == "Hi"
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE
    assert "terminal event" in str(ei.value)


def test_compat_empty_stream_body_raises():
    t = FakeTransport(resp(200, "", content_type="text/event-stream"))
    with pytest.raises(llm.ConnectorError) as ei:
        list(_openai(t).stream(_one_user()))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


def test_anthropic_stream_truncated_before_stop_raises():
    sse = "\n".join([
        'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}', "",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}', "",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}', "",
    ])
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    gen = _anthropic(t).stream(_one_user())
    assert next(gen).content_delta == "Hi"
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


def test_anthropic_stream_ending_on_message_stop_completes():
    sse = "\n".join([
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}', "",
        'data: {"type":"message_stop"}', "",
    ])
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    assert [d.content_delta for d in _anthropic(t).stream(_one_user())] == ["Hi"]


def test_gemini_stream_truncated_before_finish_reason_raises():
    sse = 'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":3}}\n\n'
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    gen = _gemini(t).stream(_one_user())
    assert next(gen).content_delta == "Hi"
    # The running usage still reaches the caller before the truncation error.
    assert next(gen).usage.input_tokens == 3
    with pytest.raises(llm.ConnectorError) as ei:
        next(gen)
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


def test_bedrock_stream_without_output_raises():
    t = FakeTransport(resp(200, '{"stopReason":"end_turn","usage":{"inputTokens":1}}'))
    chat = llm.Chat(llm.ProviderID.BEDROCK, llm.BedrockConfig(region="us-east-1", access_key_id="a", secret_access_key="s"), transport=t)
    with pytest.raises(llm.ConnectorError) as ei:
        list(chat.stream(_one_user()))
    assert ei.value.provider_code is llm.ProviderCode.PROVIDER_UNAVAILABLE


@pytest.mark.parametrize(
    "value",
    ["Wed, 21 Oct 2015 07:28:00 GMT", "Wednesday, 21-Oct-15 07:28:00 GMT", "Wed Oct 21 07:28:00 2015"],
)
def test_past_http_date_retry_after_means_retry_now(value):
    t = FakeTransport(resp(429, '{"error":{"message":"slow"}}', retry_after=value))
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(t).complete(_one_user())
    assert ei.value.cause["retryAfterSeconds"] == 0


def test_retry_after_integer_seconds():
    t = FakeTransport(resp(429, '{"error":{"message":"slow"}}', retry_after="30"))
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(t).complete(_one_user())
    assert ei.value.cause["retryAfterSeconds"] == 30


# ---- review-5: error classification, repr ----


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (408, '{"error":{"message":"timeout"}}', llm.ProviderCode.PROVIDER_UNAVAILABLE),
        (409, '{"error":{"message":"conflict"}}', llm.ProviderCode.INVALID_REQUEST),
        (410, '{"error":{"message":"gone"}}', llm.ProviderCode.INVALID_REQUEST),
        (400, '{"error":{"code":"context_length_exceeded","message":"Too many tokens for this model."}}', llm.ProviderCode.CONTEXT_LENGTH_EXCEEDED),
        (400, '{"error":{"code":"content_policy_violation","message":"Your request was rejected as a result of our safety system: content_policy_violation"}}', llm.ProviderCode.CONTENT_FILTERED),
        (400, '{"error":{"message":"Flagged by moderation"}}', llm.ProviderCode.CONTENT_FILTERED),
        (400, '{"error":{"message":"The response was filtered due to the prompt triggering content filter policy"}}', llm.ProviderCode.CONTENT_FILTERED),
        (400, '{"error":{"message":"Invalid value for temperature"}}', llm.ProviderCode.INVALID_REQUEST),
    ],
)
def test_compat_status_classification(status, body, expected):
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(FakeTransport(resp(status, body))).complete(_one_user())
    assert ei.value.provider_code is expected


@pytest.mark.parametrize("status,expected", [(408, llm.ProviderCode.PROVIDER_UNAVAILABLE), (409, llm.ProviderCode.INVALID_REQUEST)])
def test_embeddings_408_and_unlisted_4xx(status, expected):
    emb = llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=FakeTransport(resp(status, "{}")))
    with pytest.raises(llm.ConnectorError) as ei:
        emb.create(llm.EmbeddingsInput(model="text-embedding-3-small", input=["a"]))
    assert ei.value.provider_code is expected


def test_embeddings_in_body_string_code_classifies_like_chat():
    body = '{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}'
    emb = llm.Embeddings(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=FakeTransport(resp(200, body)))
    with pytest.raises(llm.ConnectorError) as ei:
        emb.create(llm.EmbeddingsInput(model="text-embedding-3-small", input=["a"]))
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED
    # A name only classifies; the status is the one the error arrived with.
    assert ei.value.status_code == 200


def test_chat_in_body_string_code_classifies():
    body = '{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}'
    with pytest.raises(llm.ConnectorError) as ei:
        _openai(FakeTransport(resp(200, body))).complete(_one_user())
    assert ei.value.provider_code is llm.ProviderCode.RATE_LIMITED and ei.value.status_code == 200


@pytest.mark.parametrize(
    "frame,expected",
    [
        ('{"error":{"message":"something broke"}}', llm.ProviderCode.UNKNOWN),
        ('{"error":{"code":"rate_limit_exceeded","message":"slow"}}', llm.ProviderCode.RATE_LIMITED),
        ('{"error":{"code":"502","message":"upstream"}}', llm.ProviderCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_compat_mid_stream_frame_classified_by_code_or_type(frame, expected):
    sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: ' + frame + "\n\n"
    t = FakeTransport(resp(200, sse, content_type="text/event-stream"))
    with pytest.raises(llm.ConnectorError) as ei:
        list(_openai(t).stream(_one_user()))
    assert ei.value.provider_code is expected
    # The status is the one the stream arrived with.
    assert ei.value.status_code == 200


def test_http_request_repr_redacts_credentials():
    req = HttpRequest(
        method="POST",
        url="https://user:pw@api.example.com/v1/chat?key=QUERYSECRET",
        headers={"Authorization": "Bearer SECRET1", "x-api-key": "SECRET2", "Content-Type": "application/json",
                 "X-Amz-Security-Token": "SECRET3"},
        body=b'{"prompt":"PRIVATE"}',
    )
    r = repr(req)
    for secret in ("SECRET1", "SECRET2", "SECRET3", "QUERYSECRET", "PRIVATE", "pw"):
        assert secret not in r
    assert "api.example.com" in r and "application/json" in r and "Authorization" in r
    assert "<20 bytes>" in r


def test_http_request_repr_never_raises():
    assert "HttpRequest(" in repr(HttpRequest(method="GET", url="https://h:notaport/x", headers=None))  # type: ignore[arg-type]
