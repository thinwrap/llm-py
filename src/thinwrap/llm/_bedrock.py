"""Native adapter for AWS Bedrock's unified Converse API
(``POST <origin>/model/{modelId}/converse``).

Requests are signed with the hand-rolled AWS SigV4 helper (``_awssigv4``) — zero
third-party deps. Converse is structurally different from OpenAI Chat Completions
(top-level ``system[]``, typed content blocks, ``inferenceConfig``, ``toolConfig``,
tool results carried in a user turn), so this is a full translation layer emitting
the identical normalized ``ChatResult`` / ``ChatStreamDelta`` / ``ConnectorError``.

Streaming NOTE: Bedrock's ``converse-stream`` uses AWS's binary event-stream
framing (``application/vnd.amazon.eventstream``), not SSE. This adapter does NOT
parse that binary protocol — ``stream()`` falls back to a single non-incremental
Converse call and yields the whole result as deltas (a content delta, one delta
per tool call, then a terminal finish/usage/raw delta). True incremental
streaming (a binary event-stream parser) is a documented follow-up.

Not normalized here (passthrough/raw, never emulated): reasoning control
(model-dependent on Bedrock — pass ``additionalModelRequestFields`` via
``passthrough.body``), response_format, prompt caching, and top_k.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Optional
from urllib.parse import quote, urlsplit

from ._awssigv4 import SigV4Params, build_canonical_query, sign_aws_v4
from ._jsonpath import jget, jlist, jnum, jstr
from ._util import compact_json, decode_json, ok_status
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
from .passthrough import apply_body

# Converse requires inferenceConfig.maxTokens; used when neither ChatInput.max_output_tokens
# nor BedrockConfig.default_max_tokens is set.
_DEFAULT_MAX_TOKENS = 4096
# AWS service name folded into the SigV4 credential scope.
_SERVICE = "bedrock"
# An AWS region is lowercase DNS-label text (us-east-1, us-gov-west-1). It is
# interpolated into the default host, so anything else could point the signed
# request, session token included, at a host of the caller's input's choosing.
_REGION_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class BedrockConnector(BaseConnector):
    def __init__(self, config, transport=None) -> None:
        super().__init__(transport)
        from .config import BedrockConfig

        if not isinstance(config, BedrockConfig):
            raise invalid_request("bedrock requires a BedrockConfig")
        self.cfg = config
        if not isinstance(config.region, str) or not _REGION_RE.fullmatch(config.region):
            raise invalid_request("Bedrock: `region` must be an AWS region name such as 'us-east-1'")
        origin = config.base_url or f"https://bedrock-runtime.{config.region}.amazonaws.com"
        self.origin = origin.rstrip("/")
        parts = urlsplit(self.origin)
        self.host = parts.netloc
        if not self.host:
            raise invalid_request(
                f"Bedrock: invalid `base_url`/`region` — could not derive a host from '{self.origin}'"
            )
        # http.client omits a default port from the Host header it sends, so the
        # signed host must omit it too or the signature never matches.
        default_port = {"https": ":443", "http": ":80"}.get(parts.scheme.lower())
        if default_port and self.host.endswith(default_port):
            self.host = self.host[: -len(default_port)]
        # A base_url may carry a path prefix (a gateway mount point). The wire
        # URL already includes it via the origin; the signed canonical URI must
        # include it too, each segment encoded once more as for the model id.
        segments = [seg for seg in parts.path.split("/") if seg]
        self.canonical_base_path = "".join("/" + quote(seg, safe="-_.~") for seg in segments)

    def complete(self, msg: ChatInput) -> ChatResult:
        url, body_bytes, headers = self._build_signed_request(msg)
        # Send the EXACT signed bytes so the request the transport writes matches
        # the SigV4 payload hash + canonical path.
        resp = self.post_raw(url, "application/json", body_bytes, headers)
        raw = decode_json(resp.body)
        if not ok_status(resp.status):
            raise self._error(resp.status, resp.headers, raw)
        # A Converse response always carries an `output` object; without it this
        # is not a result, and parsing it would fake an empty completion.
        if raw is None or not isinstance(jget(raw, "output"), dict):
            raise empty_body_error(resp.status)
        return self._parse_result(raw, msg.model)

    def stream(self, msg: ChatInput) -> Iterator[ChatStreamDelta]:
        # Non-incremental fallback (see module NOTE): issue one Converse call
        # eagerly, then yield the full result as deltas.
        result = self.complete(msg)
        return self._iter_deltas(result)

    def _iter_deltas(self, result: ChatResult) -> Iterator[ChatStreamDelta]:
        if result.message.content:
            yield ChatStreamDelta(content_delta=result.message.content)
        for i, tc in enumerate(result.message.tool_calls):
            yield ChatStreamDelta(
                tool_call_delta=ToolCallDelta(
                    index=i, id=tc.id, function_name=tc.name, arguments_delta=tc.arguments
                )
            )
        yield ChatStreamDelta(finish_reason=result.finish_reason, usage=result.usage, raw=result.raw)

    # ---- request build ----

    def _build_signed_request(self, msg: ChatInput):
        """Serialize the Converse body ONCE, then SigV4-sign those exact bytes
        together with the canonical path so URL/body/signature all agree."""
        try:
            body_bytes = compact_json(self._build_request(msg))
        except (TypeError, ValueError) as exc:
            raise invalid_request("bedrock: failed to encode request body") from exc

        # The WIRE path is single-encoded; the SIGNED canonical path is
        # DOUBLE-encoded. For every non-S3 AWS service the server URI-encodes the
        # received path a second time before recomputing the signature, so the
        # client's canonical path must be double-encoded relative to the raw
        # model id. Model ids with ':' (on-demand IDs `…-v2:0`, cross-region
        # profiles `us.anthropic.…`, ARNs with ':' / '/') otherwise sign %3A
        # while the server computes %253A → 403 SignatureDoesNotMatch. A
        # colon-free id double-encodes to itself, so this is a no-op for those.
        encoded_model = quote(msg.model, safe="")
        wire_path = "/model/" + encoded_model + "/converse"
        canonical_path = self.canonical_base_path + "/model/" + quote(encoded_model, safe="") + "/converse"
        # passthrough.query must be folded into BOTH the SigV4 canonical query
        # string and the wire URL — identical bytes (RFC3986-encoded, key-sorted)
        # — so the signature the server recomputes from the received query still
        # matches. Appending it unsigned would break the signature.
        canonical_query = ""
        if msg.passthrough and msg.passthrough.query:
            canonical_query = build_canonical_query(dict(msg.passthrough.query))
        url = self.origin + wire_path
        if canonical_query:
            url += "?" + canonical_query

        # Headers folded into the signature: Content-Type + config.headers +
        # per-request passthrough headers. Merged case-insensitively: the signer
        # lowercases names, so "content-type" beside "Content-Type" would sign
        # one value while a transport may send both (→ SignatureDoesNotMatch).
        extra: dict[str, str] = {"Content-Type": "application/json"}
        for k, v in (self.cfg.headers or {}).items():
            _set_header_ci(extra, k, v)
        if msg.passthrough and msg.passthrough.headers:
            for k, v in msg.passthrough.headers.items():
                _set_header_ci(extra, k, v)
        # post_raw force-SENDS application/json; SIGN that same value so a
        # cfg/passthrough Content-Type override can't desync the signature from
        # what actually goes on the wire (→ SignatureDoesNotMatch).
        _set_header_ci(extra, "Content-Type", "application/json")

        signed = sign_aws_v4(
            SigV4Params(
                method="POST",
                service=_SERVICE,
                region=self.cfg.region,
                host=self.host,
                path=canonical_path,
                canonical_query=canonical_query,
                payload=body_bytes,
                access_key_id=self.cfg.access_key_id,
                secret_access_key=self.cfg.secret_access_key,
                session_token=self.cfg.session_token,
                signed=extra,
            )
        )

        headers = dict(extra)
        for k, v in signed.items():
            # Host is derived by the transport from the URL; attaching it as a
            # request header is a no-op, so drop it.
            if k.lower() == "host":
                continue
            _set_header_ci(headers, k, v)
        return url, body_bytes, headers

    def _build_request(self, msg: ChatInput) -> dict[str, Any]:
        system, messages = _split_messages(msg.messages)

        body: dict[str, Any] = {"messages": messages}
        if system:
            body["system"] = system

        max_tokens = _DEFAULT_MAX_TOKENS
        if self.cfg.default_max_tokens:
            max_tokens = self.cfg.default_max_tokens
        if msg.max_output_tokens is not None:
            max_tokens = msg.max_output_tokens
        inference_config: dict[str, Any] = {"maxTokens": max_tokens}
        if msg.temperature is not None:
            inference_config["temperature"] = msg.temperature
        if msg.top_p is not None:
            inference_config["topP"] = msg.top_p
        if msg.stop:
            inference_config["stopSequences"] = list(msg.stop)
        body["inferenceConfig"] = inference_config

        # Converse has no "none" tool choice; withholding the tools is the only
        # way to keep the model from calling one.
        if msg.tools and not (msg.tool_choice is not None and msg.tool_choice.mode == "none"):
            tools = []
            for t in msg.tools:
                spec: dict[str, Any] = {"name": t.name}
                if t.description:
                    spec["description"] = t.description
                schema = t.parameters if t.parameters is not None else {"type": "object", "properties": {}}
                spec["inputSchema"] = {"json": schema}
                tools.append({"toolSpec": spec})
            tool_config: dict[str, Any] = {"tools": tools}
            tc = _map_tool_choice(msg.tool_choice)
            if tc is not None:
                tool_config["toolChoice"] = tc
            body["toolConfig"] = tool_config
        # reasoning control is model-dependent on Bedrock → not auto-mapped;
        # pass additionalModelRequestFields via passthrough.body.

        apply_body(body, msg.passthrough)
        return body

    # ---- response parse ----

    def _parse_result(self, raw: Any, requested_model: str) -> ChatResult:
        blocks = jlist(jget(jget(jget(raw, "output"), "message"), "content")) or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for b in blocks:
            t = jget(b, "text")
            if isinstance(t, str):
                text_parts.append(t)
                continue
            tu = jget(b, "toolUse")
            if tu is not None:
                inp = jget(tu, "input")
                args = compact_json(inp).decode("utf-8") if inp is not None else "{}"
                tool_calls.append(
                    ToolCall(id=jstr(jget(tu, "toolUseId")), name=jstr(jget(tu, "name")), arguments=args)
                )

        usage = jget(raw, "usage")
        # Cache reads and writes are billed input but reported apart from
        # inputTokens; the normalized count includes them.
        in_tok = int(
            jnum(jget(usage, "inputTokens"))
            + jnum(jget(usage, "cacheReadInputTokens"))
            + jnum(jget(usage, "cacheWriteInputTokens"))
        )
        out_tok = int(jnum(jget(usage, "outputTokens")))
        # Derived rather than taken from totalTokens, so it always equals
        # input + output under the cache-inclusive input above.
        total = in_tok + out_tok

        return ChatResult(
            message=AssistantMessage(role="assistant", content="".join(text_parts), tool_calls=tool_calls),
            finish_reason=_map_stop_reason(jstr(jget(raw, "stopReason"))),
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=total),
            model=requested_model,
            raw=raw,
        )

    def _error(self, status: int, headers, raw):
        msg = _extract_error(raw) or f"HTTP {status}"
        return vendor_error(status, headers, raw, _map_code(status, msg, _error_type(headers, raw)), msg)


# ---- module-private mappers ----


def _set_header_ci(headers: dict[str, str], name: str, value: str) -> None:
    """Set a header, first dropping any entry whose name differs only in case,
    so each header appears once whichever casing each source used (last wins)."""
    for existing in [k for k in headers if k.lower() == name.lower()]:
        del headers[existing]
    headers[name] = value


def _split_messages(messages: list[ChatMessage]):
    """Hoist system turns into a top-level system[] and map the rest into Converse
    {role, content[]} turns; a tool result folds into the preceding user turn
    (Converse carries tool results in a user message)."""
    system: list[Any] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            t = _content_to_text(m)
            if t:
                system.append({"text": t})
        elif m.role == "tool":
            block = {
                "toolResult": {
                    "toolUseId": m.tool_call_id,
                    "content": [{"text": _content_to_text(m)}],
                }
            }
            if out:
                last = out[-1]
                if last.get("role") == "user" and _is_tool_result_content(last.get("content")):
                    last["content"].append(block)
                    continue
            out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": m.role, "content": _map_content_blocks(m)})
    return system, out


def _content_to_text(m: ChatMessage) -> str:
    if m.content_parts:
        return "".join(p.text for p in m.content_parts if p.type == "text")
    return m.content


def _map_content_blocks(m: ChatMessage) -> list[Any]:
    blocks: list[Any] = []
    if m.content_parts:
        for p in m.content_parts:
            if p.type == "text":
                # Converse rejects a blank text block, so an empty part is dropped.
                if p.text:
                    blocks.append({"text": p.text})
            else:
                blocks.append(
                    {"image": {"format": _image_format(p.media_type), "source": {"bytes": p.base64}}}
                )
    elif m.content:
        blocks.append({"text": m.content})
    if m.role == "assistant" and m.tool_calls:
        for tc in m.tool_calls:
            blocks.append(
                {"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": _safe_json_parse(tc.arguments)}}
            )
    if not blocks:
        blocks.append({"text": ""})
    return blocks


def _is_tool_result_content(content: Any) -> bool:
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and "toolResult" in content[0]
    )


def _map_tool_choice(tc) -> Optional[dict]:
    if tc is None:
        return None
    if tc.mode == "auto":
        return {"auto": {}}
    if tc.mode == "required":
        return {"any": {}}
    if tc.mode == "function":
        return {"tool": {"name": tc.function_name}}
    return None


def _map_stop_reason(sr: str) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
        # The output was cut off by the context window rather than maxTokens.
        "model_context_window_exceeded": "length",
    }.get(sr, "unknown")


def _image_format(media_type: str) -> str:
    m = (media_type or "").lower()
    if "png" in m:
        return "png"
    if "jpeg" in m or "jpg" in m:
        return "jpeg"
    if "gif" in m:
        return "gif"
    if "webp" in m:
        return "webp"
    return "png"


def _safe_json_parse(s: str) -> Any:
    v = decode_json(s.encode("utf-8") if isinstance(s, str) else s)
    return v if v is not None else {}


def _extract_error(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        for key in ("message", "Message", "__type"):
            v = body.get(key)
            if isinstance(v, str) and v:
                return v
    return ""


def _error_type(headers: Any, body: Any) -> str:
    """The AWS error shape name (``ThrottlingException``). restJson services put
    it in the ``x-amzn-ErrorType`` header, sometimes followed by ``:<uri>``; a
    body ``__type`` may carry a ``namespace#`` prefix."""
    raw = ""
    if headers:
        raw = str(headers.get("x-amzn-errortype", "") or "")
    if not raw and isinstance(body, dict):
        raw = jstr(body.get("__type"))
    return raw.split(":", 1)[0].rsplit("#", 1)[-1].strip()


def _map_code(status: int, message: str, error_type: str = "") -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    if status == 429 or error_type == "ThrottlingException":
        return ProviderCode.RATE_LIMITED
    # 408 ModelTimeoutException and 424 ModelErrorException are failures of the
    # model behind the request, not of the request itself.
    if status >= 500 or status in (408, 424):
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status in (400, 404, 422):
        msg = message.lower()
        if "throttl" in msg:
            return ProviderCode.RATE_LIMITED
        # Tight phrases only: Bedrock reports an oversized prompt as an ordinary
        # ValidationException and relays the model provider's own wording, and a
        # looser match ("token", "max", "too long") also hits validation errors a
        # caller trimming history cannot fix.
        if (
            "input is too long" in msg
            or "prompt is too long" in msg
            or "too many input tokens" in msg
            or "maximum context length" in msg
            or "exceed context limit" in msg
        ):
            return ProviderCode.CONTEXT_LENGTH_EXCEEDED
        return ProviderCode.INVALID_REQUEST
    # 402 means the account is out of credit or quota. OpenAI (429
    # insufficient_quota) and Gemini (RESOURCE_EXHAUSTED) already report that as
    # rate-limited, so 402 classifies the same way.
    if status == 402:
        return ProviderCode.RATE_LIMITED
    if status == 413:
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN
