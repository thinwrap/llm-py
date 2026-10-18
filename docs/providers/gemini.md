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
    model="gemini-3.8-flash",
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
- **`role="tool"` messages** → `functionResponse` parts inside a `user` turn, keyed by function **NAME**. The name is resolved from the message's `name`, else the matching tool call in the nearest preceding assistant turn (`tool_call_id`→name; synthetic ids repeat from turn to turn), else the raw `tool_call_id`, else `"tool"`. Consecutive tool results coalesce into one turn.
- **Tool-call ids** — when Gemini returns a `functionCall.id`, it becomes `ToolCall.id` and is sent back as `functionCall.id` / `functionResponse.id`. Otherwise the id is synthetic and index-based (`call_0`, `call_1`, ... in both `complete` and `stream`, so parallel calls to one function stay distinct), and synthetic ids are never sent to Gemini.
- **Assistant `tool_calls`** → `functionCall` parts; **`tools`** → `[{functionDeclarations: [...]}]`, with `ToolDefinition.parameters` sent as `parametersJsonSchema` (full JSON Schema, e.g. `additionalProperties`; the older OpenAPI-subset `parameters` field rejects that with a 400).
- **Thought signatures (Gemini 3)** — Gemini rejects a replayed `functionCall` without its `thoughtSignature` (400 "Function call is missing a thought_signature"). The signature arrives on `ToolCall.provider_metadata["gemini"]["thoughtSignature"]` (on `tool_call_delta.provider_metadata` when streaming; copy it onto the assembled `ToolCall`). Send the `ToolCall`s back unchanged and the connector puts each signature back on its `functionCall` part. When no call in a model turn has one (a hand-built history, or one from another provider), the first `functionCall` of that turn gets Google's documented `"skip_thought_signature_validator"` value so the loop keeps working. Only the `gemini` key is read; other providers' keys are ignored.
- **`tool_choice`** → `toolConfig.functionCallingConfig.mode`: `auto`→`AUTO`, `required`→`ANY`, `none`→`NONE`, `function`→`ANY` + `allowedFunctionNames`.
- **Images** (`content_parts`) → `{inlineData:{mimeType, data}}`.
- **Sampling** → `generationConfig` `{maxOutputTokens, temperature, topP, stopSequences}`.
- **`response_format` IS mapped** (unlike Anthropic/Bedrock): `json_object`→`responseMimeType: application/json`; `json_schema`→ that plus `responseJsonSchema` (full JSON Schema; when `ResponseFormat.json_schema` is set).
- **`reasoning.effort`** → `generationConfig.thinkingConfig.thinkingBudget`: `low`=1024, `medium`=8192, `high`=24576 (any other value → 8192, since a budget of 0 turns thinking off).
- **Prompt caching** (`cachedContent`), safety settings, and other model params → `passthrough.body`; reasoning CoT (`thought` parts) is not normalized into `content` and stays in `raw`.
- **Streaming** (`chat.stream(...)`) is real SSE. Text arrives as `content_delta`, each `functionCall` as its own `tool_call_delta`. Gemini repeats running usage totals on every chunk, so `finish_reason` and the final `usage` come once, on the terminal delta.

```python
for delta in chat.stream(ChatInput(model="gemini-3.8-flash",
                                   messages=[ChatMessage(role="user", content="Hi")])):
    if delta.content_delta:
        print(delta.content_delta, end="")
```

- **`finishReason`** → `finish_reason`: `STOP`→`stop`, `MAX_TOKENS`→`length`, `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`/`IMAGE_SAFETY`/`IMAGE_PROHIBITED_CONTENT`/`IMAGE_RECITATION`→`content_filter`, `CONTINUATION`→`length`, else (`LANGUAGE`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `OTHER`, ...) `unknown`; a `functionCall` part forces `tool_calls`. Usage maps from `usageMetadata`: `input_tokens`=`promptTokenCount`+`toolUsePromptTokenCount`, `output_tokens`=`candidatesTokenCount`+`thoughtsTokenCount` (thinking is billed as output), `total_tokens`=input + output.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.GEMINI, ...)` raises
a `ConnectorError` with `provider_code` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.provider_code`: `401`/`403`→`auth_failed`, `402` (prepay credit
depleted, `RESOURCE_EXHAUSTED`) / `429`→`rate_limited`, `408` / `>=500`→`provider_unavailable`,
every other `4xx`→`invalid_request` (or `auth_failed` for a 400 whose `ErrorInfo`
reason is `API_KEY_INVALID`, or `context_length_exceeded` when the message is
Gemini's input-token overflow: "exceeds the maximum number of tokens"). An error
object mid-stream is classified by its `code`, else its `status`, else `unknown`,
and carries the HTTP status the stream arrived with. Gemini emits **no rate-limit
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
