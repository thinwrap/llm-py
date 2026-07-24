# Groq (Python)

Chat via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Groq is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.GROQ, OpenAICompatConfig(api_key=os.environ["GROQ_API_KEY"]))
res = chat.complete(ChatInput(
    model="llama-3.3-70b-versatile",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.groq.com/openai/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.GROQ, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://console.groq.com/keys>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.groq.com/openai/v1/chat/completions`

### Notes / quirks

- **Streaming usage quirk (handled):** in streaming, token usage arrives on `x_groq.usage` of the final chunk, not top-level `usage`. The connector relocates it, so `delta.usage` is populated the same as everywhere else (spec `stream_usage_path="x_groq"`).
- Strict JSON-schema structured output is limited to `openai/gpt-oss-*` models and can't combine with streaming or tools; other models are best-effort.
- Model ids are provider-namespaced (`openai/…`, `meta-llama/…`, `qwen/…`).
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `Embeddings(ProviderID.GROQ, ...)` raises an `invalid_request` `ConnectorError` ("Provider 'groq' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
