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
    model="claude-opus-5-5",
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
| `default_max_tokens` | `int` | no | Used (when `> 0`) if `ChatInput.max_output_tokens` is unset; otherwise `16000` (current models always think, so a smaller ceiling often ends at `length` before any text; it caps output and is not a charge). Anthropic **requires** `max_tokens`. |
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
- **`tool_choice`** → `auto`→`{type:"auto"}`, `required`→`{type:"any"}`, `none`→`{type:"none"}`, `function`→`{type:"tool", name}`. Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 reject forced tool use (`required` or a named tool) with a 400; that is the vendor's rule, so it surfaces as `invalid_request` and is never emulated.
- **`temperature` / `top_p`** are sent as given. Claude Opus 4.7 and later, every 5.x model and Fable reject them with a 400 (on Sonnet 5.5 and Haiku 5.5, non-default values), which surfaces as `invalid_request`.
- **Images** (`content_parts`) → `{type:"image", source:{type:"base64", media_type, data}}`. Empty text parts are dropped, since Anthropic rejects an empty text block.
- **`stop`** → `stop_sequences`.
- **`reasoning.effort`** → adaptive thinking: `thinking:{type:"adaptive"}` plus `output_config:{effort}`, for every model. An empty effort sends `medium`; any other value is sent as given for Anthropic to validate. There is no thinking budget, so `max_tokens` is sent unchanged (default `16000`). While reasoning is set, `temperature` / `top_p` are **omitted**.
- A passthrough `output_config` (for example `{"format": ...}` for structured output) **replaces** the connector's whole `output_config`, effort included (the body merge is shallow); put `effort` in your own `output_config` too.
- **Models before Claude 4.6** do not support adaptive thinking (verified on Haiku 4.5, which returns a 400 `invalid_request`). For those budget-mode models, leave `reasoning` unset and pass the thinking config yourself: `Passthrough(body={"thinking": {"type": "enabled", "budget_tokens": N}})`, with `max_output_tokens` above `N`.
- **`response_format` is intentionally NOT mapped** — Anthropic has no `response_format`. Use tool-based structured output, supplying the tool via `tools` or raw fields via `passthrough.body`.
- **Reasoning CoT output** — `thinking` blocks are not normalized into `content`; read them from `ChatResult.raw`.
- **Thinking blocks across a tool loop** — `res.message.provider_metadata["anthropic"]["content"]` holds the response's content blocks verbatim (thinking and `redacted_thinking` included). Copy `provider_metadata` onto every assistant `ChatMessage` you send back; the connector then replays those blocks exactly, but only while they still match the message (the same `tool_use` ids in order with the same inputs, and the same text). Otherwise the turn is rebuilt from `content` / `tool_calls` and loses its earlier reasoning. Anthropic invalidates every thinking block after an edited or stripped turn, and accounts that enforce preserved thinking reject replaying them with a 400, so once one assistant turn is rebuilt, every later assistant turn in the request is rebuilt too.
- **Prompt caching** — pass `cache_control` markers via `passthrough.body`. `usage.input_tokens` includes `cache_read_input_tokens` and `cache_creation_input_tokens`; the breakdown stays in `raw`.
- **Streaming** (`chat.stream(...)`) is real Anthropic SSE — genuinely incremental. Text arrives as `content_delta`, tool calls as `tool_call_delta` (concatenate `arguments_delta` per `index`, the 0-based tool-call ordinal); the terminal delta carries `finish_reason`, `usage` and `provider_metadata` (the content blocks rebuilt from the events, the same array a non-stream call returns).

```python
for delta in chat.stream(ChatInput(model="claude-opus-5-5",
                                   messages=[ChatMessage(role="user", content="Hi")])):
    if delta.content_delta:
        print(delta.content_delta, end="")
```

- **`stop_reason`** → `finish_reason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`/`model_context_window_exceeded`→`length`, `tool_use`→`tool_calls`, `refusal`→`content_filter`, else (`pause_turn`, ...) `unknown`. Anthropic reports no total token count; `usage.total_tokens` is summed.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.ANTHROPIC, ...)`
raises a `ConnectorError` with `provider_code` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.provider_code`: `401`/`403`→`auth_failed`, `402` (billing; out of
credit, like OpenAI's 429 `insufficient_quota`) / `429`→`rate_limited`, `408` and
any `5xx` (including `529` overloaded)→`provider_unavailable`, every other
`4xx` (`400`, `404`, `413` request too large, ...)→`invalid_request` (or
`context_length_exceeded` when the message is Anthropic's context overflow: "prompt
is too long" / "exceed context limit"). An `error` event mid-stream is classified by
its `error.type` (`unknown` for a type it does not know) and carries the HTTP status
the stream arrived with. The raw
decoded body is on `e.cause["raw"]`; a `Retry-After` header (when present) on
`e.cause["retryAfter"]` with parsed seconds on `e.cause["retryAfterSeconds"]`.

```python
from thinwrap.llm import ConnectorError

try:
    chat.complete(input)
except ConnectorError as e:
    print(e.provider_code, e.status_code, e.cause.get("retryAfterSeconds"))
```

Provider-specific request fields go through `Passthrough(body=..., headers=...,
query=...)`; vendor response extras stay in `ChatResult.raw`.
