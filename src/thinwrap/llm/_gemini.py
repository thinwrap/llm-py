"""Native adapter for the Google Gemini (AI Studio) generateContent API.

Gemini uses contents[]/parts[] (roles user/model), a top-level systemInstruction,
functionDeclarations, and keys tool results by function NAME (not id). This
connector translates the normalized surface to that shape and back, emitting the
identical ChatResult / ChatStreamDelta / ConnectorError. Unlike Bedrock, Gemini
streaming IS SSE (:streamGenerateContent?alt=sse), so streaming is truly
incremental.

Gemini 3 rejects a replayed functionCall without its ``thoughtSignature``, so
the signature rides on ``ToolCall.provider_metadata["gemini"]`` and is put back
on the functionCall part when the call is replayed.
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

# Google's documented value for a functionCall whose real signature is not
# available (a hand-built history, or one from another provider): it skips the
# signature check instead of failing the request with a 400.
_SKIP_SIGNATURE = "skip_thought_signature_validator"

# Ids this connector makes up when Gemini sends none. They are never echoed back
# to Gemini, which only knows the ids it issued.
_SYNTHETIC_ID = re.compile(r"call_\d+")


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
        # `candidates` may only be absent on a prompt-level block (reported via
        # promptFeedback.blockReason, mapped to content_filter); otherwise its
        # absence means there is no result to return.
        if raw is None or (
            jlist(jget(raw, "candidates")) is None
            and not jstr(jget(jget(raw, "promptFeedback"), "blockReason"))
        ):
            raise empty_body_error(resp.status)
        return self._parse_result(raw, msg.model)

    def stream(self, msg: ChatInput) -> Iterator[ChatStreamDelta]:
        url = _with_query(
            self.base_url + "/models/" + quote(msg.model, safe="") + ":streamGenerateContent?alt=sse", msg
        )
        stream = self.open_post_stream(url, self._build_request(msg), self._auth_headers(msg))
        if not ok_status(stream.status):
            raw = self._read_error_body(stream)
            raise self._error(stream.status, stream.headers, raw)
        self._require_event_stream(stream)
        return self._iter_stream(stream)

    def _iter_stream(self, stream) -> Iterator[ChatStreamDelta]:
        saw_function_call = False
        tool_index = 0
        # usageMetadata is cumulative (each chunk repeats the running totals), so
        # only the last one is reported, once, on the terminal delta; a consumer
        # summing per-chunk usage would otherwise over-count.
        finish = ""
        usage: Optional[Usage] = None
        tail_raw: Any = None
        # A complete stream ends on a chunk with a finishReason (or a prompt-level
        # block); without one the answer was cut off.
        saw_finish = False
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
                    saw_finish = True
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
                        # The synthetic id is index-based so parallel calls to the
                        # same function stay distinct.
                        fid = jstr(jget(fc, "id")) or "call_" + str(tool_index)
                        # Each functionCall part is its own delta with a distinct
                        # index so parallel/multiple calls are never merged.
                        tool_deltas.append(ChatStreamDelta(
                            raw=chunk,
                            tool_call_delta=ToolCallDelta(
                                index=tool_index,
                                id=fid,
                                function_name=jstr(jget(fc, "name")),
                                arguments_delta=_stringify_args(jget(fc, "args")),
                                provider_metadata=_signature_metadata(p),
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
                if fr:
                    finish = "tool_calls" if saw_function_call else _map_finish_reason(fr)
                    tail_raw = chunk
                if usage_raw is not None:
                    usage = _map_usage(usage_raw)
                    tail_raw = chunk
            if finish or usage is not None:
                yield ChatStreamDelta(raw=tail_raw, finish_reason=finish, usage=usage)
            if not finish and not saw_finish:
                raise self._stream_ended_early_error()
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
                    # parametersJsonSchema takes JSON Schema as written; the older
                    # `parameters` is an OpenAPI subset that rejects keywords such
                    # as additionalProperties with a 400.
                    fn["parametersJsonSchema"] = t.parameters
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
            # thinkingBudget is still accepted by Gemini 3; thinkingLevel is the
            # newer control and belongs with the Interactions API move.
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
                        provider_metadata=_signature_metadata(p),
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
            code = _map_code(status, msg, raw)
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
    # Filled while the history is walked, so a tool result resolves against the
    # assistant turn before it: synthetic ids repeat from turn to turn
    # ("call_0"), so a whole-history map would pair a result with a later
    # turn's function.
    name_by_id: dict[str, str] = {}

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
            fn_response: dict[str, Any] = {
                "name": name,
                "response": {"result": _parse_json_or_string(_content_to_text(m))},
            }
            if _is_vendor_id(m.tool_call_id):
                fn_response["id"] = m.tool_call_id
            part = {"functionResponse": fn_response}
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
            if m.role == "assistant":
                for tc in m.tool_calls:
                    name_by_id[tc.id] = tc.name
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
        call_parts: List[dict] = []
        for tc in m.tool_calls:
            call: dict[str, Any] = {"name": tc.name, "args": _safe_json_parse(tc.arguments)}
            if _is_vendor_id(tc.id):
                call["id"] = tc.id
            part: dict[str, Any] = {"functionCall": call}
            sig = _stored_signature(tc)
            if sig:
                part["thoughtSignature"] = sig
            call_parts.append(part)
        # Gemini 3 400s a model turn whose function calls carry no signature.
        # Only the first call of a turn is signed (parallel calls share it), so
        # the documented bypass goes on the first call only, and only when none
        # of the turn's calls has a real one.
        if not any("thoughtSignature" in p for p in call_parts):
            call_parts[0]["thoughtSignature"] = _SKIP_SIGNATURE
        parts.extend(call_parts)
    if not parts:
        parts.append({"text": ""})
    return parts


def _signature_metadata(part: Any) -> Optional[dict]:
    sig = jstr(jget(part, "thoughtSignature"))
    return {"gemini": {"thoughtSignature": sig}} if sig else None


def _stored_signature(tc: ToolCall) -> str:
    """The signature Gemini issued for this call. Only the `gemini` key is read;
    another provider's state in the same slot is never sent here."""
    meta = tc.provider_metadata.get("gemini") if isinstance(tc.provider_metadata, dict) else None
    return jstr(jget(meta, "thoughtSignature"))


def _is_vendor_id(call_id: str) -> bool:
    return bool(call_id) and not _SYNTHETIC_ID.fullmatch(call_id)


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
            # responseJsonSchema takes JSON Schema as written; responseSchema is
            # the OpenAPI-subset form.
            gc["responseJsonSchema"] = rf.json_schema


def _map_finish_reason(fr: str) -> str:
    # Values from the google-genai FinishReason enum. Everything else
    # (MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS,
    # LANGUAGE, OTHER, IMAGE_OTHER, NO_IMAGE, unspecified, ...) has no normalized
    # counterpart and stays "unknown"; the vendor value is on raw.
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        # Stopped at the per-request token limit before generation finished.
        "CONTINUATION": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "IMAGE_SAFETY": "content_filter",
        "IMAGE_PROHIBITED_CONTENT": "content_filter",
        "IMAGE_RECITATION": "content_filter",
    }.get(fr, "unknown")


def _map_usage(u: Any) -> Usage:
    # Tool-use prompt tokens are billed input and thinking tokens billed output,
    # but Gemini reports both apart from promptTokenCount / candidatesTokenCount.
    # Summing them keeps input + output equal to totalTokenCount.
    inp = int(jnum(jget(u, "promptTokenCount")) + jnum(jget(u, "toolUsePromptTokenCount")))
    out = int(jnum(jget(u, "candidatesTokenCount")) + jnum(jget(u, "thoughtsTokenCount")))
    return Usage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)


def _effort_budget(effort: str) -> int:
    # An unrecognized effort falls back to medium: reasoning was asked for, and a
    # budget of 0 would switch thinking off.
    return {"low": 1024, "medium": 8192, "high": 24576}.get(effort, 8192)


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
        return _map_code(code, extract_error_message(body), body)
    if _is_api_key_invalid(body):
        return ProviderCode.AUTH_FAILED
    status_str = jstr(jget(err, "status")).upper()
    if status_str == "RESOURCE_EXHAUSTED":
        return ProviderCode.RATE_LIMITED
    if status_str in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return ProviderCode.AUTH_FAILED
    if status_str in ("INVALID_ARGUMENT", "FAILED_PRECONDITION", "NOT_FOUND", "OUT_OF_RANGE", "ALREADY_EXISTS"):
        if _is_context_overflow(extract_error_message(body)):
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        return ProviderCode.INVALID_REQUEST
    if status_str in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED"):
        return ProviderCode.PROVIDER_UNAVAILABLE
    # A frame with neither a code nor a known status says nothing classifiable.
    return ProviderCode.UNKNOWN


def _map_code(status: int, message: str, body: Any = None) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    # 402: the prepay credit balance is depleted, which Gemini itself labels
    # RESOURCE_EXHAUSTED like a 429; classified as rate-limited, as OpenAI's
    # 429 insufficient_quota is.
    if status in (402, 429):
        return ProviderCode.RATE_LIMITED
    # 408 is a transient timeout; Google's troubleshooting guide says to retry.
    if status >= 500 or status == 408:
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status >= 400:
        # A bad API key is a 400 INVALID_ARGUMENT; only the ErrorInfo reason
        # tells it apart from a malformed request.
        if _is_api_key_invalid(body):
            return ProviderCode.AUTH_FAILED
        if _is_context_overflow(message):
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        # 404, 413, 422, …: the request cannot succeed as sent.
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN


def _is_api_key_invalid(body: Any) -> bool:
    for d in jlist(jget(jget(body, "error"), "details")) or []:
        if jstr(jget(d, "reason")) == "API_KEY_INVALID":
            return True
    return False


def _is_context_overflow(message: str) -> bool:
    """Gemini reports an oversized prompt as a 400 INVALID_ARGUMENT ("The input
    token count (N) exceeds the maximum number of tokens allowed (M)"), so only
    that wording tells it apart. Looser matches also hit ordinary validation
    errors, and a caller that trims history on this code would loop forever."""
    return "exceeds the maximum number of tokens" in message.lower()


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
