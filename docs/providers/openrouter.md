# OpenRouter (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from OpenRouter is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.OPENROUTER, OpenAICompatConfig(api_key=os.environ["OPENROUTER_API_KEY"]))
res = chat.complete(ChatInput(
    model="openai/gpt-4o",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://openrouter.ai/api/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request — set `HTTP-Referer` / `X-Title` here for app attribution. |

Inject a custom transport with `Chat(ProviderID.OPENROUTER, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://openrouter.ai/keys>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://openrouter.ai/api/v1/chat/completions`

### Notes / quirks

- Aggregator: model ids are `{provider}/{model}` (e.g. `anthropic/claude-sonnet-4.5`), with optional `:free` / `:nitro` / `:floor` suffixes.
- Set `HTTP-Referer` / `X-Title` via `headers` for attribution.
- Drops params an underlying provider lacks (unless `provider.require_parameters` is set via `Passthrough(body=...)`); `n` is unsupported.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.OPENROUTER, OpenAICompatConfig(api_key=os.environ["OPENROUTER_API_KEY"]))
res = emb.create(EmbeddingsInput(model="openai/text-embedding-3-small", input=["hello"]))
```

`POST https://openrouter.ai/api/v1/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Provider-routing controls + sub-baseline fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
