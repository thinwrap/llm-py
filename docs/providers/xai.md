# xAI (Grok) (Python)

Chat via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from xAI is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.XAI, OpenAICompatConfig(api_key=os.environ["XAI_API_KEY"]))
res = chat.complete(ChatInput(
    model="grok-4",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.x.ai/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.XAI, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://console.x.ai>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.x.ai/v1/chat/completions`

### Notes / quirks

- Model ids are `grok-*`.
- xAI positions Chat Completions as a **legacy** endpoint (it steers new work to its own Responses API); this connector targets Chat Completions.
- In streaming, a tool/function call arrives **whole in a single chunk**, not as incremental argument deltas — so `delta.tool_call_delta.arguments_delta` carries the full arguments at once.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.XAI, ...)` raises an `invalid_request` `ConnectorError` ("Provider 'xai' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
