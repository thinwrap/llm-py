"""The single connector shared by every first-class OpenAI-compatible provider."""

from __future__ import annotations

import json
from typing import Any, Iterator, List

from ._jsonpath import extract_error_message, jget, jlist, jnum, jstr
from ._util import decode_json, ok_status
from .base import BaseConnector
from .chat import (
    AssistantMessage,
    ChatInput,
    ChatMessage,
    ChatResult,
    ChatStreamDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from .errors import ProviderCode, embedded_error_class, empty_body_error, invalid_request, vendor_error
from .passthrough import apply_body, merge_headers
from .spec import OpenAICompatSpec


class OpenAICompatConnector(BaseConnector):
    def __init__(self, spec: OpenAICompatSpec, config, transport=None) -> None:
        super().__init__(transport)
        from .config import OpenAICompatConfig

        if not isinstance(config, OpenAICompatConfig):
            raise invalid_request(f"Provider '{spec.id.value}' requires an OpenAICompatConfig")
        self.spec = spec
        self.cfg = config
        base = config.base_url or spec.default_base_url
        if not base:
            raise invalid_request(f"Provider '{spec.id.value}' requires an explicit base_url in config")
        self.base_url = base.rstrip("/")

    def complete(self, msg: ChatInput) -> ChatResult:
        url, body, headers = self._build_request(msg, stream=False)
        resp = self.send_post_json(url, body, headers)
        raw = decode_json(resp.body)
        if not ok_status(resp.status):
            raise self._error(resp.status, resp.headers, raw)
        if raw is None:
            raise empty_body_error(resp.status)
        self._check_success_body(resp.status, resp.headers, raw)
        return self._parse_result(raw, msg.model)

    def stream(self, msg: ChatInput) -> Iterator[ChatStreamDelta]:
        url, body, headers = self._build_request(msg, stream=True)
        stream = self.open_post_stream(url, body, headers)
        if not ok_status(stream.status):
            raw = self._read_error_body(stream)
            raise self._error(stream.status, stream.headers, raw)
        self._require_event_stream(stream)
        return self._iter_stream(stream)

    def _iter_stream(self, stream) -> Iterator[ChatStreamDelta]:
        # A complete stream ends on [DONE], and its last choice chunk carries a
        # finish_reason; either one proves the answer was not cut off.
        state: dict = {"saw_done": False}
        saw_finish = False
        try:
            for payload in self._iter_sse(stream, state):
                try:
                    chunk = json.loads(payload)
                except (ValueError, TypeError):
                    continue
                # A provider error emitted after 200 OK must not be swallowed as
                # a silent truncation — surface it like the non-stream path. The
                # HTTP status is a success code, so classify from the error body.
                if isinstance(chunk, dict) and chunk.get("error") is not None:
                    raise self._error(stream.status, stream.headers, chunk, _stream_error_code(chunk))
                for delta in self._map_stream_chunk(chunk):
                    if delta.finish_reason:
                        saw_finish = True
                    yield delta
            if not state["saw_done"] and not saw_finish:
                raise self._stream_ended_early_error()
        finally:
            stream.close()

    def _build_request(self, msg: ChatInput, stream: bool):
        body: dict[str, Any] = {"model": msg.model, "messages": [_map_message(m) for m in msg.messages]}
        tools_allowed = self.spec.supports_tools
        if tools_allowed and msg.tools:
            body["tools"] = [_map_tool(t) for t in msg.tools]
        if tools_allowed and msg.tool_choice is not None:
            body["tool_choice"] = _map_tool_choice(msg.tool_choice)
        if msg.response_format is not None:
            body["response_format"] = _map_response_format(msg.response_format)
        if msg.temperature is not None:
            body["temperature"] = msg.temperature
        if msg.top_p is not None:
            body["top_p"] = msg.top_p
        if msg.max_output_tokens is not None:
            body[self.spec.max_tokens_field] = msg.max_output_tokens
        if msg.stop:
            body["stop"] = msg.stop[0] if len(msg.stop) == 1 else list(msg.stop)
        if msg.reasoning is not None:
            body["reasoning_effort"] = msg.reasoning.effort
        if stream:
            body["stream"] = True
            if not self.spec.strict_params:
                body["stream_options"] = {"include_usage": True}
        apply_body(body, msg.passthrough)

        headers: dict[str, str] = {}
        headers.update(self.spec.build_auth_headers(self.cfg))
        if self.cfg.headers:
            headers.update(self.cfg.headers)
        headers = merge_headers(headers, msg.passthrough)

        url = self.base_url + "/chat/completions"
        if msg.passthrough and msg.passthrough.query:
            from urllib.parse import urlencode

            url += "?" + urlencode(dict(msg.passthrough.query))
        return url, body, headers

    def _check_success_body(self, status: int, headers, raw: Any) -> None:
        """A 2xx is not proof of success: OpenRouter reports a failure after
        processing started as HTTP 200 with either only an ``error`` object, or
        ``choices[0]`` carrying ``finish_reason: "error"`` + ``error`` beside
        partial content. Without a choice there is no result to return at all."""
        err = jget(raw, "error")
        if err is not None:
            raise self._embedded_error(status, headers, raw, err)
        choices = jlist(jget(raw, "choices"))
        if not choices:
            raise empty_body_error(status)
        choice_err = jget(choices[0], "error")
        if choice_err is not None or jstr(jget(choices[0], "finish_reason")) == "error":
            raise self._embedded_error(status, headers, raw, choice_err)

    def _embedded_error(self, status: int, headers, raw: Any, err: Any):
        msg = extract_error_message({"error": err})
        # The status code is the HTTP status the response arrived with (the 2xx),
        # as for a mid-stream frame; the embedded code or type only classifies.
        # Without one the error says nothing classifiable.
        effective = embedded_error_class(err)
        code = ProviderCode.UNKNOWN if effective is None else _map_code(effective, {"error": err})
        return vendor_error(status, headers, raw, code, msg)

    def _parse_result(self, raw: Any, requested_model: str) -> ChatResult:
        choices = jlist(jget(raw, "choices")) or []
        choice = choices[0] if choices else None
        m = jget(choice, "message")
        tool_calls = []
        for tc in jlist(jget(m, "tool_calls")) or []:
            tool_calls.append(_parse_tool_call(tc))
        return ChatResult(
            message=AssistantMessage(role="assistant", content=_content_text(jget(m, "content")), tool_calls=tool_calls),
            finish_reason=_map_finish_reason(jstr(jget(choice, "finish_reason"))),
            usage=_map_usage(jget(raw, "usage")),
            model=jstr(jget(raw, "model")) or requested_model,
            raw=raw,
        )

    def _map_stream_chunk(self, chunk: Any) -> List[ChatStreamDelta]:
        base = ChatStreamDelta(raw=chunk)
        has = False
        choices = jlist(jget(chunk, "choices")) or []
        choice = choices[0] if choices else None
        delta = jget(choice, "delta")
        content = _content_text(jget(delta, "content"))
        if content:
            base.content_delta = content
            has = True
        fr = jstr(jget(choice, "finish_reason"))
        if fr:
            base.finish_reason = _map_finish_reason(fr)
            has = True
        usage_raw = jget(chunk, "usage")
        if usage_raw is None and self.spec.stream_usage_path == "x_groq":
            usage_raw = jget(jget(chunk, "x_groq"), "usage")
        if usage_raw is not None:
            base.usage = _map_usage(usage_raw)
            has = True

        # A chunk may carry several tool_calls entries at once — emit one delta
        # each (the first rides on the base delta), keyed by the vendor index.
        extra: List[ChatStreamDelta] = []
        for i, tc in enumerate(jlist(jget(delta, "tool_calls")) or []):
            fn = jget(tc, "function")
            tcd = ToolCallDelta(
                index=int(jnum(jget(tc, "index"))),
                id=jstr(jget(tc, "id")),
                function_name=jstr(jget(fn, "name")),
                arguments_delta=jstr(jget(fn, "arguments")),
            )
            if i == 0:
                base.tool_call_delta = tcd
                has = True
            else:
                extra.append(ChatStreamDelta(tool_call_delta=tcd, raw=chunk))

        return ([base] + extra) if has else []

    def _error(self, status: int, headers, raw, code: ProviderCode | None = None):
        if code is None:
            code = _map_code(status, raw)
        return vendor_error(status, headers, raw, code, extract_error_message(raw))


# ---- module-private mappers ----


def _stream_error_code(body: Any) -> ProviderCode:
    """Classify a provider error object delivered mid-stream (after HTTP 200). The
    stream's status is a success code, so the frame's own ``code`` / ``type``
    decides (an HTTP status, or a name such as ``insufficient_quota``). A frame
    with neither says nothing classifiable, so it is ``unknown``."""
    effective = embedded_error_class(jget(body, "error"))
    if effective is None:
        return ProviderCode.UNKNOWN
    return _map_code(effective, body)


def _content_text(content: Any) -> str:
    """Assistant text from ``message.content`` / ``delta.content``. Mistral
    reasoning models return a list of chunks instead of a string — ``thinking``
    chunks plus ``{"type":"text","text":...}`` answer chunks; only the answer is
    content (the reasoning stays visible on ``raw``)."""
    if isinstance(content, str):
        return content
    parts = jlist(content) or []
    return "".join(jstr(jget(p, "text")) for p in parts if jget(p, "type") == "text")


def _map_message(m: ChatMessage) -> dict:
    out: dict[str, Any] = {"role": m.role, "content": _map_content(m)}
    if m.name:
        out["name"] = m.name
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    return out


def _map_content(m: ChatMessage):
    if m.content_parts:
        parts: list[dict[str, Any]] = []
        for p in m.content_parts:
            if p.type == "text":
                parts.append({"type": "text", "text": p.text})
            else:
                parts.append({"type": "image_url", "image_url": {"url": f"data:{p.media_type};base64,{p.base64}"}})
        return parts
    if m.content:
        return m.content
    if m.role == "assistant" and m.tool_calls:
        return None
    return ""


def _map_tool(t) -> dict:
    fn: dict[str, Any] = {"name": t.name}
    if t.description:
        fn["description"] = t.description
    if t.parameters is not None:
        fn["parameters"] = t.parameters
    return {"type": "function", "function": fn}


def _parse_tool_call(tc) -> ToolCall:
    fn = jget(tc, "function")
    return ToolCall(id=jstr(jget(tc, "id")), name=jstr(jget(fn, "name")), arguments=jstr(jget(fn, "arguments")))


def _map_tool_choice(tc):
    if tc.mode == "function":
        return {"type": "function", "function": {"name": tc.function_name}}
    return tc.mode


def _map_response_format(rf) -> dict:
    if rf.type == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": rf.json_schema_name, "schema": rf.json_schema}}
    return {"type": rf.type}


def _map_finish_reason(fr: str) -> str:
    return {
        "stop": "stop",
        # Together AI reports natural end-of-sequence as "eos" rather than "stop".
        "eos": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "content_filter": "content_filter",
    }.get(fr, "unknown")


def _map_usage(u) -> Usage:
    in_tok = int(jnum(jget(u, "prompt_tokens")))
    out_tok = int(jnum(jget(u, "completion_tokens")))
    total = int(jnum(jget(u, "total_tokens"))) or (in_tok + out_tok)
    return Usage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=total)


def _map_code(status: int, body: Any) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    # 402 (out of credits/balance on OpenRouter, DeepSeek, ...): OpenAI reports
    # the same condition as 429 insufficient_quota, so it classifies the same way.
    if status in (402, 429):
        return ProviderCode.RATE_LIMITED
    # 408 is a timeout on the provider's side; a retry can succeed.
    if status >= 500 or status == 408:
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status >= 400:
        if _is_context_overflow(body):
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        if _is_content_filter(extract_error_message(body)):
            return ProviderCode.CONTENT_FILTERED
        # 404, 413, 422, …: the request cannot succeed as sent.
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN


def _is_context_overflow(body: Any) -> bool:
    # OpenAI's documented code first; other compat servers only word it.
    if jget(jget(body, "error"), "code") == "context_length_exceeded":
        return True
    msg = extract_error_message(body).lower()
    return "context length" in msg or "maximum context" in msg or "context_length" in msg


def _is_content_filter(message: str) -> bool:
    # OpenAI words a moderation refusal as content_policy_violation; Azure and
    # others say "content filter"/"content_filter" or name the moderation step.
    msg = message.lower()
    return (
        "content_policy" in msg
        or "content policy" in msg
        or "moderation" in msg
        or ("content" in msg and "filter" in msg)
    )
