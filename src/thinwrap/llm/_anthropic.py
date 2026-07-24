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
passthrough); reasoning CoT output (thinking blocks land in ``raw``); prompt
caching (cache_control via passthrough, cached-token counts in ``raw``).
"""

from __future__ import annotations

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
DEFAULT_MAX_TOKENS = 4096


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
        if raw is None:
            raise empty_body_error(resp.status)
        return self._parse_result(raw, msg.model)

    def stream(self, msg: ChatInput) -> Iterator[ChatStreamDelta]:
        url, body, headers = self._build_request(msg, stream=True)
        stream = self.open_post_stream(url, body, headers)
        if not ok_status(stream.status):
            raw = decode_json(stream.read_all())
            stream.close()
            raise self._error(stream.status, stream.headers, raw)
        return self._iter_stream(stream)

    def _iter_stream(self, stream) -> Iterator[ChatStreamDelta]:
        # input_tokens is captured across events: message_start reports it,
        # message_delta reads it back when it reports output_tokens (Anthropic's
        # SSE is stateful).
        input_tokens = 0
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
                    usage = jget(jget(evt, "message"), "usage")
                    input_tokens = int(jnum(jget(usage, "input_tokens")))
                    continue
                if typ == "content_block_start":
                    cb = jget(evt, "content_block")
                    if jstr(jget(cb, "type")) == "tool_use":
                        block = int(jnum(jget(evt, "index")))
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
                    if dtype == "text_delta":
                        s = jstr(jget(d, "text"))
                        if s:
                            yield ChatStreamDelta(content_delta=s, raw=evt)
                    elif dtype == "input_json_delta":
                        pj = jget(d, "partial_json")
                        if isinstance(pj, str):
                            block = int(jnum(jget(evt, "index")))
                            yield ChatStreamDelta(
                                tool_call_delta=ToolCallDelta(index=tool_index_by_block.get(block, 0), arguments_delta=pj),
                                raw=evt,
                            )
                    # thinking_delta → CoT output not normalized; available only via raw.
                    continue
                if typ == "message_delta":
                    out = ChatStreamDelta(raw=evt)
                    sr = jstr(jget(jget(evt, "delta"), "stop_reason"))
                    if sr:
                        out.finish_reason = _map_stop_reason(sr)
                    out_tok = int(jnum(jget(jget(evt, "usage"), "output_tokens")))
                    out.usage = Usage(input_tokens=input_tokens, output_tokens=out_tok, total_tokens=input_tokens + out_tok)
                    yield out
                    continue
                # message_stop / content_block_stop / ping → ignored
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

        thinking: Optional[dict[str, Any]] = None
        budget = 0
        if msg.reasoning is not None:
            budget = _effort_budget(msg.reasoning.effort)
            thinking = {"type": "enabled", "budget_tokens": budget}

        max_tokens = DEFAULT_MAX_TOKENS
        if self.cfg.default_max_tokens > 0:
            max_tokens = self.cfg.default_max_tokens
        caller_set_max = msg.max_output_tokens is not None
        if caller_set_max:
            max_tokens = msg.max_output_tokens
        if thinking is not None and max_tokens <= budget:
            # Anthropic requires max_tokens > thinking.budget_tokens (max_tokens is
            # the TOTAL, thinking included). If the CALLER set an explicit
            # max_output_tokens below the budget, silently inflating it would spend
            # past their cap — raise instead so they lower reasoning.effort or raise
            # the cap. Only a non-explicit default is inflated to keep the request valid.
            if caller_set_max:
                raise invalid_request(
                    f"max_output_tokens ({max_tokens}) must be greater than the reasoning "
                    f"budget ({budget}) for extended thinking — raise max_output_tokens "
                    f"or lower reasoning.effort"
                )
            max_tokens = budget + DEFAULT_MAX_TOKENS

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

        if thinking is not None:
            # Extended thinking requires temperature=1 and disallows top_p/top_k — omit sampling.
            body["thinking"] = thinking
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
        for b in jlist(jget(raw, "content")) or []:
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
            # thinking blocks → raw only
        usage = jget(raw, "usage")
        in_tok = int(jnum(jget(usage, "input_tokens")))
        out_tok = int(jnum(jget(usage, "output_tokens")))
        return ChatResult(
            message=AssistantMessage(role="assistant", content="".join(text_parts), tool_calls=tool_calls),
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
        msg = extract_error_message(evt).lower()
        if "too long" in msg or "context" in msg or "max_tokens" in msg:
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        return ProviderCode.INVALID_REQUEST
    # overloaded_error / api_error / anything else → availability failure.
    return ProviderCode.PROVIDER_UNAVAILABLE


def _dump_json(v: Any) -> str:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def _safe_json_parse(s: str) -> Any:
    v = decode_json(s.encode("utf-8"))
    return v if v is not None else {}


def _split_messages(messages: list[ChatMessage]):
    system = ""
    out: list[dict[str, Any]] = []
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
            out.append({"role": m.role, "content": _map_content_blocks(m)})
    return system, out


def _content_to_text(m: ChatMessage) -> str:
    if m.content_parts:
        return "".join(p.text for p in m.content_parts if p.type == "text")
    return m.content


def _map_content_blocks(m: ChatMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if m.content_parts:
        for p in m.content_parts:
            if p.type == "text":
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
    if sr == "max_tokens":
        return "length"
    if sr == "tool_use":
        return "tool_calls"
    if sr == "refusal":
        return "content_filter"
    return "unknown"


def _effort_budget(effort: str) -> int:
    if effort == "low":
        return 1024
    if effort == "high":
        return 12288
    return 4096  # "medium" and any unset/unknown value


def _map_code(status: int, message: str) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    if status == 429:
        return ProviderCode.RATE_LIMITED
    # Any 5xx is a provider-side failure. A fixed {500,502,503,529} whitelist
    # dropped 504/520/522/524 (gateway / Cloudflare-origin) to Unknown, so a
    # consumer's retry policy wouldn't fire on them.
    if status >= 500:
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status in (400, 404, 413, 422):
        msg = message.lower()
        if "too long" in msg or "context" in msg or "max_tokens" in msg:
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN
