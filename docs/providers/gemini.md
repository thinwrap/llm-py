# Google Gemini (Python)

Chat via the native adapter for the Google Gemini (**AI Studio**)
`generateContent` API. Gemini uses `contents[]`/`parts[]` (roles `user`/`model`),
a top-level `systemInstruction`, and `functionDeclarations`, so this is a full
translation layer that still emits the identical normalized `ChatResult` /
`ChatStreamDelta` / `ConnectorError`. Unlike the Bedrock fallback, Gemini
streaming is **real SSE** — genuinely incremental.

> This is the AI Studio Gemini API (`gemini` id). Vertex AI Gemini is a separate
> product with different auth (GCP OAuth/ADC) and endpoints — not this connector.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, GeminiConfig, ProviderID

chat = Chat(ProviderID.GEMINI, GeminiConfig(api_key=os.environ["GEMINI_API_KEY"]))
res = chat.complete(ChatInput(
    model="gemini-2.5-flash",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.finish_reason, res.usage.total_tokens)
```

## Configuration

`GeminiConfig` (a frozen dataclass):

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | yes | Sent as the `x-goog-api-key` header. |
| `base_url` | `str` | no | Overrides `https://generativelanguage.googleapis.com/v1beta`. Trailing slash trimmed. |
| `default_max_tokens` | `int` | no | `generationConfig.maxOutputTokens` when `max_output_tokens` is unset. No hard default — if both are unset, `maxOutputTokens` is simply omitted. |
| `headers` | `Mapping[str, str]` | no | Merged onto every request. |

## Auth setup

Create an API key in Google AI Studio. It is sent as the `x-goog-api-key` header
(the key is not placed in the query string). No environment inference — pass
`api_key` explicitly.

## Chat

### Endpoint

`POST <base_url>/models/{model}:generateContent` (non-streaming) and
`POST <base_url>/models/{model}:streamGenerateContent?alt=sse` (streaming). The
model is path-escaped. Default base URL
`https://generativelanguage.googleapis.com/v1beta`.

### Notes / quirks

- **System messages** → top-level `systemInstruction` (multiple concatenated with blank lines).
- **Roles** → `assistant`→`model`, everything else → `user`.
- **`role="tool"` messages** → `functionResponse` parts inside a `user` turn, keyed by function **NAME** (Gemini has no tool-call ids). The name is resolved from the message's `name`, else the matching prior assistant tool call (`tool_call_id`→name), else the raw `tool_call_id`, else `"tool"`. Consecutive tool results coalesce into one turn.
- **Assistant `tool_calls`** → `functionCall` parts; **`tools`** → `[{functionDeclarations: [...]}]`.
- **`tool_choice`** → `toolConfig.functionCallingConfig.mode`: `auto`→`AUTO`, `required`→`ANY`, `none`→`NONE`, `function`→`ANY` + `allowedFunctionNames`.
- **Images** (`content_parts`) → `{inlineData:{mimeType, data}}`.
- **Sampling** → `generationConfig` `{maxOutputTokens, temperature, topP, stopSequences}`.
- **`response_format` IS mapped** (unlike Anthropic/Bedrock): `json_object`→`responseMimeType: application/json`; `json_schema`→ that plus `responseSchema` (when `ResponseFormat.json_schema` is set).
- **`reasoning.effort`** → `generationConfig.thinkingConfig.thinkingBudget`: `low`=1024, `medium`=8192, `high`=24576 (any other value → budget 0).
- **Prompt caching** (`cachedContent`), safety settings, and other model params → `passthrough.body`; reasoning CoT / `thought_signature` round-trip are not persisted by the wrapper (available via `raw`).
- **Streaming** (`chat.stream(...)`) is real SSE. Each chunk collapses to one delta: concatenated text as `content_delta`, a `functionCall` as `tool_call_delta`, plus the trailing `finish_reason` / `usage` when present.

```python
for delta in chat.stream(ChatInput(model="gemini-2.5-flash",
                                   messages=[ChatMessage(role="user", content="Hi")])):
    if delta.content_delta:
        print(delta.content_delta, end="")
```

- **`finishReason`** → `finish_reason`: `STOP`→`stop`, `MAX_TOKENS`→`length`, `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`→`content_filter`, else `unknown`; a `functionCall` part forces `tool_calls`. Usage maps from `usageMetadata` (`promptTokenCount`/`candidatesTokenCount`/`totalTokenCount`), summing the total when omitted.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.GEMINI, ...)` raises
a `ConnectorError` with `provider_code` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.provider_code`: `401`/`403`→`auth_failed`, `429`→`rate_limited`,
`>=500`→`provider_unavailable`, `400`/`404`/`422`→`invalid_request` (or
`context_length_exceeded` when the message mentions "too long" / "context" /
"token count"); anything else → `unknown`. Gemini emits **no rate-limit
headers** — retry timing is parsed from the 429 body's `error.details`
`RetryInfo.retryDelay` (e.g. `"12s"`) into `e.cause["retryAfterSeconds"]` when a
`Retry-After` header did not already populate it. The raw decoded body is on
`e.cause["raw"]`.

```python
from thinwrap.llm import ConnectorError

try:
    chat.complete(input)
except ConnectorError as e:
    print(e.provider_code, e.status_code, e.cause.get("retryAfterSeconds"))
```

Provider-specific request fields go through `Passthrough(body=..., headers=...,
query=...)`; vendor response extras stay in `ChatResult.raw`.
