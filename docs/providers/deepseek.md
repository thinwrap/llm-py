# DeepSeek (Python)

Chat via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from DeepSeek is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.DEEPSEEK, OpenAICompatConfig(api_key=os.environ["DEEPSEEK_API_KEY"]))
res = chat.complete(ChatInput(
    model="deepseek-chat",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.deepseek.com/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.DEEPSEEK, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://platform.deepseek.com/api_keys>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.deepseek.com/v1/chat/completions`

### Notes / quirks

- The OpenAI-compatible endpoint is first-class (the native API).
- **No strict `json_schema`** — `response_format` supports only `text` / `json_object`; a `json_schema` request degrades to best-effort JSON.
- `frequency_penalty` / `presence_penalty` are silently ignored; reasoning is exposed via `reasoning_content` — read it from `res.raw`.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.DEEPSEEK, ...)` raises an `invalid_request` `ConnectorError` ("Provider 'deepseek' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
