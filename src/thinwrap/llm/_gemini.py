"""Native adapter for the Google Gemini (AI Studio) generateContent API.

Gemini uses contents[]/parts[] (roles user/model), a top-level systemInstruction,
functionDeclarations, and keys tool results by function NAME (not id). This
connector translates the normalized surface to that shape and back, emitting the
identical ChatResult / ChatStreamDelta / ConnectorError. Unlike Bedrock, Gemini
streaming IS SSE (:streamGenerateContent?alt=sse), so streaming is truly
incremental.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterator, List, Optional, Tuple
from urllib.parse import quote

from ._jsonpath import extract_error_message, jget, jlist, jnum, jstr
from ._util import decode_json, first_non_empty, ok_status
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
from .errors import ProviderCode, empty_body_error, invalid_request, vendor_error
from .passthrough import apply_body, merge_headers

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiConnector(BaseConnector):
    def __init__(self, config, transport=None) -> None:
        super().__init__(transport)
        from .config import GeminiConfig

        if not isinstance(config, GeminiConfig):
            raise invalid_request("gemini requires a GeminiConfig")
        self.cfg = config
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

    def complete(self, msg: ChatInput) -> ChatResult:
        url = _with_query(
            self.base_url + "/models/" + quote(msg.model, safe="") + ":generateContent", msg
        )
        resp = self.send_post_json(url, self._build_request(msg), self._auth_headers(msg))
        raw = decode_json(resp.body)
        if not ok_status(resp.status):
            raise self._error(resp.status, resp.headers, raw)
        if raw is None:
            raise empty_body_error(resp.status)
        return self._parse_result(raw, msg.model)

    def stream(self, msg: ChatInput) -> Iterator[ChatStreamDelta]:
        url = _with_query(
            self.base_url + "/models/" + quote(msg.model, safe="") + ":streamGenerateContent?alt=sse", msg
        )
        stream = self.open_post_stream(url, self._build_request(msg), self._auth_headers(msg))
        if not ok_status(stream.status):
            raw = decode_json(stream.read_all())
            stream.close()
            raise self._error(stream.status, stream.headers, raw)
        return self._iter_stream(stream)

    def _iter_stream(self, stream) -> Iterator[ChatStreamDelta]:
        saw_function_call = False
        tool_index = 0
        try:
            for payload in self._iter_sse(stream):
                try:
                    chunk = json.loads(payload)
                except (ValueError, TypeError):
                    continue
                # A provider error emitted after 200 OK must surface, not truncate.
                # The HTTP status is 200, so classify from the error body itself.
                if isinstance(chunk, dict) and chunk.get("error") is not None:
                    raise self._error(stream.status, stream.headers, chunk, _stream_error_code(chunk))
                cand = _first_candidate(chunk)
                # A prompt-level safety block arrives as a chunk with no candidate
                # and only promptFeedback.blockReason — emit a content_filter finish
                # so the consumer sees the block rather than a silent, empty end.
                if cand is None and jstr(jget(jget(chunk, "promptFeedback"), "blockReason")):
                    yield ChatStreamDelta(raw=chunk, finish_reason="content_filter")
                    continue
                parts = jlist(jget(jget(cand, "content"), "parts")) or []
                text = ""
                tool_deltas: list[ChatStreamDelta] = []
                for p in parts:
                    # Reasoning/CoT parts (thought=true) are not normalized into
                    # content — they surface only via `raw` (mirrors Anthropic).
                    if jget(p, "thought"):
                        continue
                    tv = jget(p, "text")
                    if isinstance(tv, str) and tv:
                        text += tv
                        continue
                    fc = jget(p, "functionCall")
                    if fc is not None:
                        saw_function_call = True
                        fid = jstr(jget(fc, "id")) or "call_" + jstr(jget(fc, "name"))
                        # Each functionCall part is its own delta with a distinct
                        # index so parallel/multiple calls are never merged.
                        tool_deltas.append(ChatStreamDelta(
                            raw=chunk,
                            tool_call_delta=ToolCallDelta(
                                index=tool_index,
                                id=fid,
                                function_name=jstr(jget(fc, "name")),
                                arguments_delta=_stringify_args(jget(fc, "args")),
                            ),
                        ))
                        tool_index += 1

                # Text precedes the tool-call deltas (Gemini emits prose first).
                if text:
                    yield ChatStreamDelta(raw=chunk, content_delta=text)
                for td in tool_deltas:
                    yield td

                fr = jstr(jget(cand, "finishReason"))
                usage_raw = jget(chunk, "usageMetadata")
                if fr or usage_raw is not None:
                    tail = ChatStreamDelta(raw=chunk)
                    if fr:
                        tail.finish_reason = "tool_calls" if saw_function_call else _map_finish_reason(fr)
                    if usage_raw is not None:
                        tail.usage = _map_usage(usage_raw)
                    yield tail
        finally:
            stream.close()

    def _auth_headers(self, msg: ChatInput) -> dict:
        headers = {"x-goog-api-key": self.cfg.api_key}
        if self.cfg.headers:
            headers.update(self.cfg.headers)
        return merge_headers(headers, msg.passthrough)

    def _build_request(self, msg: ChatInput) -> dict:
        system_instruction, contents = _split_messages(msg.messages)

        body: dict[str, Any] = {"contents": contents}
        if system_instruction is not None:
            body["systemInstruction"] = system_instruction

        if msg.tools:
            decls: List[Any] = []
            for t in msg.tools:
                fn: dict[str, Any] = {"name": t.name}
                if t.description:
                    fn["description"] = t.description
                if t.parameters is not None:
                    fn["parameters"] = t.parameters
                decls.append(fn)
            body["tools"] = [{"functionDeclarations": decls}]
            fc = _function_calling_config(msg.tool_choice)
            if fc is not None:
                body["toolConfig"] = {"functionCallingConfig": fc}

        gc: dict[str, Any] = {}
        if msg.max_output_tokens is not None:
            gc["maxOutputTokens"] = msg.max_output_tokens
        elif self.cfg.default_max_tokens:
            gc["maxOutputTokens"] = self.cfg.default_max_tokens
        if msg.temperature is not None:
            gc["temperature"] = msg.temperature
        if msg.top_p is not None:
            gc["topP"] = msg.top_p
        if msg.stop:
            gc["stopSequences"] = list(msg.stop)
        _apply_response_format(gc, msg.response_format)
        if msg.reasoning is not None:
            gc["thinkingConfig"] = {"thinkingBudget": _effort_budget(msg.reasoning.effort)}
        if gc:
            body["generationConfig"] = gc

        apply_body(body, msg.passthrough)
        return body

    def _parse_result(self, raw: Any, requested_model: str) -> ChatResult:
        cand = _first_candidate(raw)
        parts = jlist(jget(jget(cand, "content"), "parts")) or []

        text = ""
        tool_calls: List[ToolCall] = []
        idx = 0
        for p in parts:
            # Reasoning/CoT parts (thought=true) are not normalized into content
            # — they surface only via `raw` (mirrors Anthropic).
            if jget(p, "thought"):
                continue
            tv = jget(p, "text")
            if isinstance(tv, str):
                text += tv
                continue
            fc = jget(p, "functionCall")
            if fc is not None:
                fid = jstr(jget(fc, "id"))
                if not fid:
                    fid = "call_" + str(idx)
                tool_calls.append(
                    ToolCall(
                        id=fid,
                        name=jstr(jget(fc, "name")),
                        arguments=_stringify_args(jget(fc, "args")),
                    )
                )
                idx += 1

        if tool_calls:
            finish_reason = "tool_calls"
        else:
            fr = jstr(jget(cand, "finishReason"))
            # No candidate came back: a prompt-level safety block reports only
            # promptFeedback.blockReason. Surface it as content_filter rather than
            # a silent empty completion with an unknown finish.
            if not fr and jstr(jget(jget(raw, "promptFeedback"), "blockReason")):
                finish_reason = "content_filter"
            else:
                finish_reason = _map_finish_reason(fr)
        return ChatResult(
            message=AssistantMessage(role="assistant", content=text, tool_calls=tool_calls),
            finish_reason=finish_reason,
            usage=_map_usage(jget(raw, "usageMetadata")),
            model=first_non_empty(jstr(jget(raw, "modelVersion")), requested_model),
            raw=raw,
        )

    def _error(self, status: int, headers, raw, code: Optional[ProviderCode] = None):
        msg = extract_error_message(raw)
        if code is None:
            code = _map_code(status, msg)
        ce = vendor_error(status, headers, raw, code, msg)
        # Gemini has no rate-limit headers — retry timing lives in the 429 body's
        # RetryInfo detail. Fill it in only when the Retry-After header didn't.
        if isinstance(ce.cause, dict) and "retryAfterSeconds" not in ce.cause:
            secs = _retry_delay_seconds(raw)
            if secs is not None:
                ce.cause["retryAfterSeconds"] = secs
        return ce


# ---- module-private mappers ----


def _with_query(url: str, msg: ChatInput) -> str:
    """Append passthrough.query to the request URL, using `&` when the URL
    already carries a query string (e.g. the streaming `?alt=sse`)."""
    pt = msg.passthrough
    if pt and pt.query:
        from urllib.parse import urlencode

        sep = "&" if "?" in url else "?"
        url += sep + urlencode(dict(pt.query))
    return url


def _first_candidate(raw: Any) -> Any:
    candidates = jlist(jget(raw, "candidates"))
    if candidates:
        return candidates[0]
    return None


def _split_messages(messages: List[ChatMessage]) -> Tuple[Optional[dict], List[Any]]:
    """Fold system messages into a single systemInstruction and map the rest into
    Gemini's contents[]. Tool-result messages become functionResponse parts keyed
    by function name and are batched onto a trailing user turn (matching how
    Gemini expects parallel tool results)."""
    name_by_id: dict[str, str] = {}
    for m in messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                name_by_id[tc.id] = tc.name

    system_text = ""
    contents: List[Any] = []
    for m in messages:
        if m.role == "system":
            t = _content_to_text(m)
            system_text = system_text + "\n\n" + t if system_text else t
        elif m.role == "tool":
            name = m.name
            if not name and m.tool_call_id:
                name = name_by_id.get(m.tool_call_id, "")
            if not name:
                name = m.tool_call_id
            if not name:
                name = "tool"
            part = {
                "functionResponse": {
                    "name": name,
                    "response": {"result": _parse_json_or_string(_content_to_text(m))},
                }
            }
            if contents:
                last = contents[-1]
                if (
                    isinstance(last, dict)
                    and last.get("role") == "user"
                    and _is_function_response_parts(last.get("parts"))
                ):
                    last["parts"].append(part)
                    continue
            contents.append({"role": "user", "parts": [part]})
        else:
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": _map_parts(m)})

    system_instruction = None
    if system_text:
        system_instruction = {"parts": [{"text": system_text}]}
    return system_instruction, contents


def _content_to_text(m: ChatMessage) -> str:
    if m.content_parts:
        return "".join(p.text for p in m.content_parts if p.type == "text")
    return m.content


def _map_parts(m: ChatMessage) -> List[Any]:
    parts: List[Any] = []
    if m.content_parts:
        for p in m.content_parts:
            if p.type == "text":
                parts.append({"text": p.text})
            else:
                parts.append({"inlineData": {"mimeType": p.media_type, "data": p.base64}})
    elif m.content:
        parts.append({"text": m.content})
    if m.role == "assistant" and m.tool_calls:
        for tc in m.tool_calls:
            parts.append({"functionCall": {"name": tc.name, "args": _safe_json_parse(tc.arguments)}})
    if not parts:
        parts.append({"text": ""})
    return parts


def _is_function_response_parts(parts: Any) -> bool:
    if not isinstance(parts, list) or not parts:
        return False
    return isinstance(parts[0], dict) and "functionResponse" in parts[0]


def _function_calling_config(tc) -> Optional[dict]:
    if tc is None:
        return None
    if tc.mode == "auto":
        return {"mode": "AUTO"}
    if tc.mode == "required":
        return {"mode": "ANY"}
    if tc.mode == "none":
        return {"mode": "NONE"}
    if tc.mode == "function":
        return {"mode": "ANY", "allowedFunctionNames": [tc.function_name]}
    return None


def _apply_response_format(gc: dict, rf) -> None:
    if rf is None:
        return
    if rf.type == "json_object":
        gc["responseMimeType"] = "application/json"
    elif rf.type == "json_schema":
        gc["responseMimeType"] = "application/json"
        if rf.json_schema is not None:
            gc["responseSchema"] = rf.json_schema


def _map_finish_reason(fr: str) -> str:
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
    }.get(fr, "unknown")


def _map_usage(u: Any) -> Usage:
    inp = int(jnum(jget(u, "promptTokenCount")))
    out = int(jnum(jget(u, "candidatesTokenCount")))
    total = int(jnum(jget(u, "totalTokenCount")))
    if total == 0:
        total = inp + out
    return Usage(input_tokens=inp, output_tokens=out, total_tokens=total)


def _effort_budget(effort: str) -> int:
    return {"low": 1024, "medium": 8192, "high": 24576}.get(effort, 0)


def _stringify_args(args: Any) -> str:
    if args is None:
        args = {}
    try:
        return json.dumps(args, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _safe_json_parse(s: str) -> Any:
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return {}


def _parse_json_or_string(s: str) -> Any:
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def _stream_error_code(body: Any) -> ProviderCode:
    """A Gemini error object delivered after HTTP 200 carries its own numeric
    `code` (an HTTP status) plus a `status` string; classify from those rather
    than the stream's 200 status."""
    err = jget(body, "error")
    code = int(jnum(jget(err, "code")))
    if code:
        return _map_code(code, extract_error_message(body))
    status_str = jstr(jget(err, "status")).upper()
    if status_str == "RESOURCE_EXHAUSTED":
        return ProviderCode.RATE_LIMITED
    if status_str in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return ProviderCode.AUTH_FAILED
    if status_str == "INVALID_ARGUMENT":
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.PROVIDER_UNAVAILABLE


def _map_code(status: int, message: str) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    if status == 429:
        return ProviderCode.RATE_LIMITED
    if status >= 500:
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status in (400, 404, 422):
        msg = message.lower()
        if "too long" in msg or "context" in msg or "token count" in msg:
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN


_RETRY_DELAY_RE = re.compile(r"^(\d+(?:\.\d+)?)s$")


def _retry_delay_seconds(body: Any) -> Optional[int]:
    details = jlist(jget(jget(body, "error"), "details"))
    if details is None:
        return None
    for d in details:
        rd = jstr(jget(d, "retryDelay"))
        if not rd:
            continue
        m = _RETRY_DELAY_RE.match(rd)
        if m:
            return int(math.floor(float(m.group(1)) + 0.5))
    return None
