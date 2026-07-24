# Anthropic (Python)

Chat via the native Anthropic **Messages API** adapter (Claude models). Anthropic
does not speak OpenAI's Chat Completions shape, so this is a full translation
layer that still emits the identical normalized `ChatResult` / `ChatStreamDelta`
/ `ConnectorError` — switching to/from `anthropic` is only the id + `model`.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, AnthropicConfig, ProviderID

chat = Chat(ProviderID.ANTHROPIC, AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
res = chat.complete(ChatInput(
    model="claude-sonnet-4-5",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.finish_reason, res.usage.total_tokens)
```

## Configuration

`AnthropicConfig` (a frozen dataclass):

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | yes | Sent as the `x-api-key` header (not `Authorization: Bearer`). |
| `base_url` | `str` | no | Overrides `https://api.anthropic.com/v1`. Trailing slash trimmed; `/messages` is appended. |
| `anthropic_version` | `str` | no | `anthropic-version` header; default `2023-06-01`. |
| `default_max_tokens` | `int` | no | Used (when `> 0`) if `ChatInput.max_output_tokens` is unset; otherwise `4096`. Anthropic **requires** `max_tokens`. |
| `headers` | `Mapping[str, str]` | no | Merged onto every request. |

## Auth setup

Create an API key in the Anthropic Console. It is sent as the `x-api-key` header
along with `anthropic-version`. No environment inference — pass `api_key`
explicitly (read it from your own env as above).

## Chat

### Endpoint

`POST <base_url>/messages` (default `https://api.anthropic.com/v1/messages`).

### Notes / quirks

- **System messages** → hoisted into the top-level `system` field (multiple are concatenated with blank lines).
- **`role="tool"` messages** → carried inside a `user` turn as `tool_result` blocks; consecutive tool results coalesce into one turn.
- **Assistant `tool_calls`** → `tool_use` blocks; **`tools`** → `{name, description, input_schema}` (the JSON Schema in `ToolDefinition.parameters` goes directly into `input_schema`).
- **`tool_choice`** → `auto`→`{type:"auto"}`, `required`→`{type:"any"}`, `none`→`{type:"none"}`, `function`→`{type:"tool", name}`.
- **Images** (`content_parts`) → `{type:"image", source:{type:"base64", media_type, data}}`.
- **`stop`** → `stop_sequences`.
- **`reasoning.effort`** → `thinking:{type:"enabled", budget_tokens}` with budget `low`=1024, `medium`/unset=4096, `high`=12288. While thinking is enabled, `temperature` / `top_p` are **omitted** (Anthropic requires temperature=1 and disallows top_p/top_k with extended thinking), and `max_tokens` is bumped to `budget + 4096` if it would otherwise be ≤ the budget.
- **`response_format` is intentionally NOT mapped** — Anthropic has no `response_format`. Use tool-based structured output, supplying the tool via `tools` or raw fields via `passthrough.body`.
- **Reasoning CoT output** — `thinking` blocks are not normalized; read them from `ChatResult.raw`.
- **Prompt caching** — pass `cache_control` markers via `passthrough.body`; cached-token counts appear in `raw`.
- **Streaming** (`chat.stream(...)`) is real Anthropic SSE — genuinely incremental. Text arrives as `content_delta`, tool calls as `tool_call_delta` (concatenate `arguments_delta` per `index`); the terminal `message_delta` carries `finish_reason` + `usage`.

```python
for delta in chat.stream(ChatInput(model="claude-sonnet-4-5",
                                   messages=[ChatMessage(role="user", content="Hi")])):
    if delta.content_delta:
        print(delta.content_delta, end="")
```

- **`stop_reason`** → `finish_reason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`, `refusal`→`content_filter`, else `unknown`. Anthropic reports no total token count; `usage.total_tokens` is summed.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.ANTHROPIC, ...)`
raises a `ConnectorError` with `provider_code` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.provider_code`: `401`/`403`→`auth_failed`, `429`→`rate_limited`,
`500`/`502`/`503`/`529`→`provider_unavailable`, `400`/`404`/`413`/`422`→`invalid_request`
(or `context_length_exceeded` when the message mentions "too long" / "context" /
"max_tokens"); anything else → `unknown`. The raw decoded body is on
`e.cause["raw"]`; a `Retry-After` header (when present) on `e.cause["retryAfter"]`
with parsed seconds on `e.cause["retryAfterSeconds"]`.

```python
from thinwrap.llm import ConnectorError

try:
    chat.complete(input)
except ConnectorError as e:
    print(e.provider_code, e.status_code, e.cause.get("retryAfterSeconds"))
```

Provider-specific request fields go through `Passthrough(body=..., headers=...,
query=...)`; vendor response extras stay in `ChatResult.raw`.
