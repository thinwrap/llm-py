# AWS Bedrock (Python)

Chat via the native adapter for Amazon Bedrock's unified **Converse API**.
Requests are signed with **hand-rolled AWS SigV4** (`_awssigv4.py`, `hmac`-based,
zero dependencies — no `boto3`). Converse is structurally different from OpenAI,
so this is a full translation layer that still emits the identical normalized
`ChatResult` / `ChatStreamDelta` / `ConnectorError`.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, BedrockConfig, ProviderID

chat = Chat(ProviderID.BEDROCK, BedrockConfig(
    region="us-east-1",
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    # session_token=os.environ["AWS_SESSION_TOKEN"],  # for temporary creds
))
res = chat.complete(ChatInput(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.finish_reason, res.usage.total_tokens)
```

## Configuration

`BedrockConfig` (a frozen dataclass):

| Field | Type | Required | Notes |
|---|---|---|---|
| `region` | `str` | yes | e.g. `us-east-1`; folded into the SigV4 credential scope and the default host. No environment inference. |
| `access_key_id` | `str` | yes | SigV4 credential. |
| `secret_access_key` | `str` | yes | SigV4 credential. |
| `session_token` | `str` | no | STS temporary-credential token → `X-Amz-Security-Token`. |
| `base_url` | `str` | no | Overrides `https://bedrock-runtime.<region>.amazonaws.com` (VPC endpoint / proxy). Must yield a host or the constructor raises `ConnectorError(invalid_request)`. |
| `default_max_tokens` | `int` | no | Converse `inferenceConfig.maxTokens` when `max_output_tokens` is unset; default `4096` (Converse requires `maxTokens`). |
| `headers` | `Mapping[str, str]` | no | Merged onto the request **and folded into the SigV4 signature**. |

## Auth setup

Uses standard AWS SigV4 (service `bedrock`) over the stdlib — no AWS SDK. Supply
`access_key_id` / `secret_access_key` (and `session_token` for temporary
credentials). The body is serialized once and those exact bytes are signed
together with the canonical path so URL, body, and signature all agree.

## Chat

### Endpoint

`POST <origin>/model/{modelId}/converse` where `origin` defaults to
`https://bedrock-runtime.<region>.amazonaws.com`. The model id is
percent-encoded for the path (e.g. `...-v2:0` → `...-v2%3A0`) so the SigV4
canonical URI matches the wire URL exactly.

### Notes / quirks

- **System messages** → top-level `system: [{text}]`.
- **`role="tool"` messages** → `toolResult` blocks inside a `user` turn; consecutive tool results coalesce into one turn.
- **Assistant `tool_calls`** → `toolUse` blocks; **`tools`** → `toolConfig.tools[].toolSpec` with the JSON Schema under `inputSchema.json`.
- **`tool_choice`** → `auto`→`{auto:{}}`, `required`→`{any:{}}`, `function`→`{tool:{name}}`; `none` → `toolChoice` omitted (Converse has no "none").
- **Images** (`content_parts`) → `{image:{format, source:{bytes}}}` (format derived from `media_type`).
- **Sampling** → `inferenceConfig` `{maxTokens, temperature, topP, stopSequences}`.
- **`response_format` is NOT mapped** — use tool-based structured output, or pass raw fields via `passthrough.body`.
- **Reasoning is NOT auto-mapped** — reasoning control is model-dependent on Bedrock; pass `additionalModelRequestFields` via `passthrough.body`.
- **Prompt caching** — `cachePoint` blocks via `passthrough.body`. `top_k` is not mapped either.
- **Streaming** (`chat.stream(...)`) is a **non-incremental fallback.** Bedrock's `converse-stream` uses AWS's binary event-stream framing (`application/vnd.amazon.eventstream`), not SSE, and this port does not parse it. `stream()` issues one `/converse` call and yields the whole completion as deltas: an optional content delta, one `tool_call_delta` per tool call, then a terminal delta carrying `finish_reason` + `usage` + `raw`. A true binary-event-stream parser is a documented follow-up.

```python
for delta in chat.stream(ChatInput(model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                                   messages=[ChatMessage(role="user", content="Hi")])):
    if delta.content_delta:
        print(delta.content_delta, end="")
```

- **`stopReason`** → `finish_reason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`, `content_filtered`/`guardrail_intervened`→`content_filter`, else `unknown`. Converse `usage` maps directly; `total_tokens` falls back to input+output when the vendor omits it.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.BEDROCK, ...)` raises
a `ConnectorError` with `provider_code` `invalid_request`. (Amazon Titan
embeddings are not a Converse/OpenAI-float surface.)

## Error mapping & passthrough

`ConnectorError.provider_code`: `401`/`403`→`auth_failed`, `429`→`rate_limited`,
`>=500`→`provider_unavailable`; `400`/`404`/`422`→`invalid_request`, except when
the message mentions "throttl" (→`rate_limited`) or "too long" / "context" /
"token" / "max" (→`context_length_exceeded`); anything else → `unknown`. The raw
decoded body is on `e.cause["raw"]`; a `Retry-After` header (when present) on
`e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`.

```python
from thinwrap.llm import ConnectorError

try:
    chat.complete(input)
except ConnectorError as e:
    print(e.provider_code, e.status_code, e.cause.get("retryAfterSeconds"))
```

Provider-specific request fields go through `Passthrough(body=..., headers=...,
query=...)`; vendor response extras stay in `ChatResult.raw`.
