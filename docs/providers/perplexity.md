# Perplexity (Sonar) (Python)

Chat via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Perplexity is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.PERPLEXITY, OpenAICompatConfig(api_key=os.environ["PERPLEXITY_API_KEY"]))
res = chat.complete(ChatInput(
    model="sonar",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.perplexity.ai` (no `/v1` segment). |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.PERPLEXITY, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://www.perplexity.ai/settings/api>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.perplexity.ai/chat/completions` (the default base URL has no `/v1` prefix).

### Notes / quirks

- **No tool calling (handled):** the Sonar chat surface has no tool/function calling, so the spec sets `supports_tools=False` — `tools` and `tool_choice` on `ChatInput` are **not sent** (tools live on Perplexity's separate Agent API).
- Sampling surface is stripped upstream (no `top_k` / penalties / `seed` / `n` on Sonar).
- Model ids are `sonar` / `sonar-pro` / `sonar-reasoning-pro` / `sonar-deep-research`.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.PERPLEXITY, ...)` raises an `invalid_request` `ConnectorError` ("Provider 'perplexity' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Search/citation controls + sub-baseline fields go via `Passthrough(body=..., headers=..., query=...)`; citations and other vendor extras on `res.raw`.
