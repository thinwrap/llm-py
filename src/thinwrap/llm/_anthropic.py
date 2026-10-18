"""Native adapter for the Anthropic Messages API (POST <base_url>/messages).

Anthropic is structurally different from OpenAI Chat Completions — different
auth (x-api-key + anthropic-version), a top-level ``system``, typed content
blocks, tool results carried inside a user turn, ``stop_sequences``, and
Anthropic-shaped SSE events — so this is a full translation layer, not an
OpenAI passthrough. It emits the identical normalized ChatResult /
ChatStreamDelta / ConnectorError as every other connector, so it stays drop-in
swappable via the Chat facade.

Not normalized here (surfaced via ``raw`` or ``passthrough``, never emulated):
response_format (Anthropic has none — use tool-based structured output via
passthrough); reasoning CoT output (thinking blocks land in ``raw``, and verbatim
in ``provider_metadata["anthropic"]["content"]`` for replay); prompt caching
(cache_control via passthrough; cache read/write tokens are counted in
``usage.input_tokens``, the breakdown stays in ``raw``).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterator, Optional

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
from .errors import ProviderCode, empty_body_error, invalid_request, vendor_error
from .passthrough import apply_body, merge_headers

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_VERSION = "2023-06-01"
# Used when the caller sets no max_tokens (Anthropic requires one). Current
# models always think and default to medium effort, so a smaller ceiling often
# ends at `length` before any text; max_tokens caps output, it is not billed as
# such. 16000 is Anthropic's recommended non-streaming default.
DEFAULT_MAX_TOKENS = 16000


class AnthropicConnector(BaseConnector):
    def __init__(self, config, transport=None) -> None:
        super().__init__(transport)
        from .config import AnthropicConfig

        if not isinstance(config, AnthropicConfig):
            raise invalid_request("anthropic requires an AnthropicConfig")
        self.cfg = config
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

    def complete(self, msg: ChatInput) -> ChatResult:
        url, body, headers = self._build_request(msg, stream=False)
        resp = self.send_post_json(url, body, headers)
        raw = decode_json(resp.body)
        if not ok_status(resp.status):
            raise self._error(resp.status, resp.headers, raw)
        # A Messages response always carries a `content` array; without it this
        # is not a result, and parsing it would fake an empty completion.
        if raw is None or jlist(jget(raw, "content")) is None:
            raise empty_body_error(resp.status)
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
        # Usage is merged across events field by field, as the official SDK's
        # MessageStream does: message_start reports the counts, and message_delta
        # restates some of them cumulatively; a field it omits or sends as null
        # keeps the earlier value.
        usage_fields: dict[str, float] = {}
        # The response's content blocks, rebuilt from the events in block order
        # so the terminal delta can carry them for verbatim replay (the same
        # array a non-stream call returns).
        blocks: dict[int, dict[str, Any]] = {}
        partial_json: dict[int, str] = {}
        # A complete stream reports its stop_reason on message_delta and closes
        # with message_stop; without either the answer was cut off.
        saw_stop = False
        # Anthropic indexes CONTENT BLOCKS (text + tool_use interleaved), so the
        # first tool call can arrive at block index 1. Normalize to a 0-based
        # tool-call-relative index (matching the OpenAI-compat surface and the other
        # languages); map each block index to its tool ordinal so the argument
        # continuation deltas line up.
        next_tool_index = 0
        tool_index_by_block: dict[int, int] = {}
        try:
            for payload in self._iter_sse(stream):
                try:
                    evt = json.loads(payload)
                except (ValueError, TypeError):
                    continue
                typ = jstr(jget(evt, "type"))
                # An `event: error` frame after 200 OK must surface, not truncate.
                # The HTTP status is 200, so classify from the error frame itself.
                if typ == "error":
                    raise self._error(stream.status, stream.headers, evt, _stream_error_code(evt))
                if typ == "message_start":
                    _merge_usage(usage_fields, jget(jget(evt, "message"), "usage"))
                    continue
                if typ == "content_block_start":
                    cb = jget(evt, "content_block")
                    block = int(jnum(jget(evt, "index")))
                    if isinstance(cb, dict):
                        blocks[block] = copy.deepcopy(cb)
                        if "input" in cb or jstr(cb.get("type")).endswith("tool_use"):
                            # The input arrives as input_json_delta fragments.
                            partial_json[block] = ""
                    if jstr(jget(cb, "type")) == "tool_use":
                        tool_index_by_block[block] = next_tool_index
                        yield ChatStreamDelta(
                            tool_call_delta=ToolCallDelta(
                                index=next_tool_index,
                                id=jstr(jget(cb, "id")),
                                function_name=jstr(jget(cb, "name")),
                            ),
                            raw=evt,
                        )
                        next_tool_index += 1
                    continue
                if typ == "content_block_delta":
                    d = jget(evt, "delta")
                    dtype = jstr(jget(d, "type"))
                    block = int(jnum(jget(evt, "index")))
                    _accumulate_block(blocks.get(block), partial_json, block, d)
                    if dtype == "text_delta":
                        s = jstr(jget(d, "text"))
                        if s:
                            yield ChatStreamDelta(content_delta=s, raw=evt)
                    elif dtype == "input_json_delta":
                        pj = jget(d, "partial_json")
                        # Server-tool blocks (server_tool_use) stream their input
                        # the same way but are not caller tool calls.
                        if isinstance(pj, str) and block in tool_index_by_block:
                            yield ChatStreamDelta(
                                tool_call_delta=ToolCallDelta(index=tool_index_by_block[block], arguments_delta=pj),
                                raw=evt,
                            )
                    # thinking_delta → CoT output not normalized; available via
                    # raw and the terminal provider_metadata.
                    continue
                if typ == "message_delta":
                    out = ChatStreamDelta(raw=evt)
                    sr = jstr(jget(jget(evt, "delta"), "stop_reason"))
                    _merge_usage(usage_fields, jget(evt, "usage"))
                    in_tok = _input_tokens(usage_fields)
                    out_tok = int(usage_fields.get("output_tokens", 0))
                    out.usage = Usage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok)
                    if sr:
                        out.finish_reason = _map_stop_reason(sr)
                        saw_stop = True
                        # Only the terminal delta carries the turn's blocks.
                        out.provider_metadata = {"anthropic": {"content": _finish_blocks(blocks, partial_json)}}
                    yield out
                    continue
                if typ == "message_stop":
                    saw_stop = True
                # content_block_stop / ping → ignored
            if not saw_stop:
                raise self._stream_ended_early_error()
        finally:
            stream.close()

    def _auth_headers(self, msg: ChatInput) -> dict[str, str]:
        version = self.cfg.anthropic_version or DEFAULT_VERSION
        headers: dict[str, str] = {"x-api-key": self.cfg.api_key, "anthropic-version": version}
        if self.cfg.headers:
            headers.update(self.cfg.headers)
        return merge_headers(headers, msg.passthrough)

    def _build_request(self, msg: ChatInput, stream: bool):
        system, messages = _split_messages(msg.messages)

        max_tokens = DEFAULT_MAX_TOKENS
        if self.cfg.default_max_tokens > 0:
            max_tokens = self.cfg.default_max_tokens
        if msg.max_output_tokens is not None:
            max_tokens = msg.max_output_tokens

        body: dict[str, Any] = {"model": msg.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            body["system"] = system

        if msg.tools:
            tools = []
            for t in msg.tools:
                tool: dict[str, Any] = {"name": t.name}
                if t.description:
                    tool["description"] = t.description
                tool["input_schema"] = t.parameters if t.parameters is not None else {"type": "object", "properties": {}}
                tools.append(tool)
            body["tools"] = tools
        if msg.tool_choice is not None:
            tc = _map_tool_choice(msg.tool_choice)
            if tc is not None:
                body["tool_choice"] = tc

        if msg.reasoning is not None:
            # Adaptive thinking is the current-generation control: the model sizes
            # its own thinking from the effort, so there is no budget to fit under
            # max_tokens. Claude 4.5 and older reject it with a 400; those models
            # take budget thinking via passthrough. Thinking also rules out
            # custom sampling, so temperature/top_p are omitted.
            body["thinking"] = {"type": "adaptive"}
            # An empty effort means "unspecified", so the documented default
            # applies; any other value is sent as given for Anthropic to validate.
            body["output_config"] = {"effort": msg.reasoning.effort or "medium"}
        else:
            if msg.temperature is not None:
                body["temperature"] = msg.temperature
            if msg.top_p is not None:
                body["top_p"] = msg.top_p
        if msg.stop:
            body["stop_sequences"] = list(msg.stop)
        # NOTE: response_format is intentionally NOT mapped — Anthropic has no
        # response_format; structured output is via tools / passthrough.

        if stream:
            body["stream"] = True
        apply_body(body, msg.passthrough)

        url = self.base_url + "/messages"
        if msg.passthrough and msg.passthrough.query:
            from urllib.parse import urlencode

            url += "?" + urlencode(dict(msg.passthrough.query))
        return url, body, self._auth_headers(msg)

    def _parse_result(self, raw: Any, requested_model: str) -> ChatResult:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content = jlist(jget(raw, "content")) or []
        for b in content:
            btype = jstr(jget(b, "type"))
            if btype == "text":
                text_parts.append(jstr(jget(b, "text")))
            elif btype == "tool_use":
                inp = jget(b, "input")
                if inp is None:
                    inp = {}
                tool_calls.append(
                    ToolCall(id=jstr(jget(b, "id")), name=jstr(jget(b, "name")), arguments=_dump_json(inp))
                )
            # thinking blocks → raw + provider_metadata only
        usage = jget(raw, "usage")
        in_tok = _input_tokens(usage)
        out_tok = int(jnum(jget(usage, "output_tokens")))
        return ChatResult(
            message=AssistantMessage(
                role="assistant",
                content="".join(text_parts),
                tool_calls=tool_calls,
                provider_metadata={"anthropic": {"content": copy.deepcopy(content)}},
            ),
            finish_reason=_map_stop_reason(jstr(jget(raw, "stop_reason"))),
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok),
            model=jstr(jget(raw, "model")) or requested_model,
            raw=raw,
        )

    def _error(self, status: int, headers, raw, code: ProviderCode | None = None):
        msg = extract_error_message(raw) or f"HTTP {status}"
        if code is None:
            code = _map_code(status, msg)
        return vendor_error(status, headers, raw, code, msg)


# ---- module-private mappers ----


def _stream_error_code(evt: Any) -> ProviderCode:
    """Classify an Anthropic `error` SSE frame delivered after HTTP 200 by its
    error.type (the HTTP status is 200 and can't classify it)."""
    etype = jstr(jget(jget(evt, "error"), "type")).lower()
    if etype == "rate_limit_error":
        return ProviderCode.RATE_LIMITED
    if etype in ("authentication_error", "permission_error"):
        return ProviderCode.AUTH_FAILED
    if etype == "invalid_request_error":
        if _is_context_overflow(extract_error_message(evt)):
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        return ProviderCode.INVALID_REQUEST
    if etype == "billing_error":
        # The streamed form of a 402; classified the same way.
        return ProviderCode.RATE_LIMITED
    if etype in ("not_found_error", "request_too_large", "conflict_error"):
        return ProviderCode.INVALID_REQUEST
    if etype in ("overloaded_error", "api_error", "timeout_error"):
        return ProviderCode.PROVIDER_UNAVAILABLE
    # A frame without a recognizable type says nothing classifiable.
    return ProviderCode.UNKNOWN


def _is_context_overflow(message: str) -> bool:
    """Anthropic reports a context overflow as a plain invalid_request_error, so
    only its own wording ("prompt is too long: N tokens > M maximum", "input
    length and `max_tokens` exceed context limit") tells it apart. Looser
    matches ("context", "max_tokens") also hit ordinary validation errors, and a
    caller that trims history on this code would then loop forever."""
    msg = message.lower()
    return "prompt is too long" in msg or "exceed context limit" in msg


def _merge_usage(fields: dict[str, float], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"):
        v = usage.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fields[key] = v


def _input_tokens(usage: Any) -> int:
    """Every input token billed for the request. Anthropic reports cache reads
    and cache writes apart from ``input_tokens``; the normalized count includes
    them so it means the same thing across providers."""
    return int(
        jnum(jget(usage, "input_tokens"))
        + jnum(jget(usage, "cache_read_input_tokens"))
        + jnum(jget(usage, "cache_creation_input_tokens"))
    )


def _accumulate_block(block: Any, partial_json: dict[int, str], index: int, delta: Any) -> None:
    if not isinstance(block, dict):
        return
    dtype = jstr(jget(delta, "type"))
    if dtype == "text_delta":
        block["text"] = jstr(block.get("text")) + jstr(jget(delta, "text"))
    elif dtype == "thinking_delta":
        block["thinking"] = jstr(block.get("thinking")) + jstr(jget(delta, "thinking"))
    elif dtype == "signature_delta":
        block["signature"] = jstr(jget(delta, "signature"))
    elif dtype == "input_json_delta" and index in partial_json:
        partial_json[index] += jstr(jget(delta, "partial_json"))
    elif dtype == "citations_delta":
        citation = jget(delta, "citation")
        if citation is not None:
            citations = block.get("citations")
            if not isinstance(citations, list):
                citations = []
                block["citations"] = citations
            citations.append(citation)


def _finish_blocks(blocks: dict[int, dict[str, Any]], partial_json: dict[int, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index in sorted(blocks):
        block = blocks[index]
        if index in partial_json:
            pj = partial_json[index]
            if pj:
                block["input"] = _safe_json_parse(pj)
            elif not isinstance(block.get("input"), dict):
                # No deltas followed: the input on content_block_start (if any)
                # is the whole input.
                block["input"] = {}
        out.append(block)
    return out


def _dump_json(v: Any) -> str:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def _safe_json_parse(s: str) -> Any:
    v = decode_json(s.encode("utf-8"))
    return v if v is not None else {}


def _split_messages(messages: list[ChatMessage]):
    system = ""
    out: list[dict[str, Any]] = []
    # Anthropic invalidates every thinking block after a turn whose thinking was
    # edited or stripped, and accounts that enforce preserved thinking reject
    # replaying them with a 400. Once one assistant turn is rebuilt, every later
    # one is rebuilt too.
    rebuilt = False
    for m in messages:
        if m.role == "system":
            text = _content_to_text(m)
            system = system + "\n\n" + text if system else text
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": _content_to_text(m)}
            if out and out[-1].get("role") == "user" and _is_tool_result_content(out[-1].get("content")):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            replay = None
            if m.role == "assistant":
                replay = None if rebuilt else _verbatim_content(m)
                rebuilt = rebuilt or replay is None
            out.append({"role": m.role, "content": replay if replay is not None else _map_content_blocks(m)})
    return system, out


def _verbatim_content(m: ChatMessage) -> Optional[list[Any]]:
    """The assistant turn exactly as Anthropic returned it, thinking blocks
    included, when ``provider_metadata`` still describes this message: the same
    tool_use ids in order with the same inputs, and the same text. A message
    edited since (or built by hand, or from another provider) is rebuilt from
    its normalized fields instead, which drops its thinking blocks; replaying
    the stored blocks would silently undo the edit."""
    meta = m.provider_metadata.get("anthropic") if isinstance(m.provider_metadata, dict) else None
    content = jget(meta, "content")
    if not isinstance(content, list) or not content:
        return None
    if not all(isinstance(b, dict) for b in content):
        return None
    tool_uses = [b for b in content if b.get("type") == "tool_use"]
    if [jstr(b.get("id")) for b in tool_uses] != [tc.id for tc in m.tool_calls]:
        return None
    for block, tc in zip(tool_uses, m.tool_calls):
        try:
            args = json.loads(tc.arguments) if tc.arguments else {}
        except (TypeError, ValueError):
            return None
        if block.get("input") != args:
            return None
    text = "".join(jstr(b.get("text")) for b in content if b.get("type") == "text")
    if text != _content_to_text(m):
        return None
    return copy.deepcopy(content)


def _content_to_text(m: ChatMessage) -> str:
    if m.content_parts:
        return "".join(p.text for p in m.content_parts if p.type == "text")
    return m.content


def _map_content_blocks(m: ChatMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if m.content_parts:
        for p in m.content_parts:
            if p.type == "text":
                # Anthropic rejects an empty text block, so an empty part is dropped.
                if p.text:
                    blocks.append({"type": "text", "text": p.text})
            else:
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": p.media_type, "data": p.base64}}
                )
    elif m.content:
        blocks.append({"type": "text", "text": m.content})
    if m.role == "assistant" and m.tool_calls:
        for tc in m.tool_calls:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": _safe_json_parse(tc.arguments)})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _is_tool_result_content(content: Any) -> bool:
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and content[0].get("type") == "tool_result"
    )


def _map_tool_choice(tc) -> Optional[dict[str, Any]]:
    if tc.mode == "auto":
        return {"type": "auto"}
    if tc.mode == "required":
        return {"type": "any"}
    if tc.mode == "none":
        return {"type": "none"}
    if tc.mode == "function":
        return {"type": "tool", "name": tc.function_name}
    return None


def _map_stop_reason(sr: str) -> str:
    if sr in ("end_turn", "stop_sequence"):
        return "stop"
    # model_context_window_exceeded: the output was cut off by the context
    # window rather than max_tokens; for the caller both mean "truncated".
    if sr in ("max_tokens", "model_context_window_exceeded"):
        return "length"
    if sr == "tool_use":
        return "tool_calls"
    if sr == "refusal":
        return "content_filter"
    # pause_turn (a server-tool loop hit its iteration limit; send the turn back
    # to continue) has no normalized counterpart.
    return "unknown"


def _map_code(status: int, message: str) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    # 402 means the account is out of credit or quota. OpenAI (429
    # insufficient_quota) and Gemini (RESOURCE_EXHAUSTED) already report that as
    # rate-limited, so 402 classifies the same way.
    if status in (402, 429):
        return ProviderCode.RATE_LIMITED
    # Any 5xx is a provider-side failure, including gateway and Cloudflare-origin
    # statuses (504/520/522/524) a fixed whitelist would miss; 408 is a
    # transient timeout.
    if status >= 500 or status == 408:
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status >= 400:
        if _is_context_overflow(message):
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        # 404, 413 request_too_large (a byte limit, not a token one), 422, …:
        # the request cannot succeed as sent.
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN
