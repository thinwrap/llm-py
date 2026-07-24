# Fireworks AI (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Fireworks is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.FIREWORKS, OpenAICompatConfig(api_key=os.environ["FIREWORKS_API_KEY"]))
res = chat.complete(ChatInput(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.fireworks.ai/inference/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.FIREWORKS, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://fireworks.ai/account/api-keys>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.fireworks.ai/inference/v1/chat/completions`

### Notes / quirks

- The OpenAI-compatible endpoint **is** the native API (a superset).
- Model ids are account-namespaced (`accounts/fireworks/models/…`); bare OpenAI names won't resolve.
- Reasoning chain-of-thought is exposed via `reasoning_content` — read it from `res.raw`.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.FIREWORKS, OpenAICompatConfig(api_key=os.environ["FIREWORKS_API_KEY"]))
res = emb.create(EmbeddingsInput(model="nomic-ai/nomic-embed-text-v1.5", input=["hello"]))
```

`POST https://api.fireworks.ai/inference/v1/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
