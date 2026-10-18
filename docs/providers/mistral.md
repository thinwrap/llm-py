# Mistral (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Mistral is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.MISTRAL, OpenAICompatConfig(api_key=os.environ["MISTRAL_API_KEY"]))
res = chat.complete(ChatInput(
    model="mistral-large-latest",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.mistral.ai/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.MISTRAL, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create a key at <https://console.mistral.ai/api-keys>. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.mistral.ai/v1/chat/completions`

### Notes / quirks

- **Strict-params quirk (handled):** Mistral 422s on unknown params, so the connector omits OpenAI-only extras — notably `stream_options` is **not** sent on streaming requests (spec `strict_params=True`). Keep `Passthrough(body=...)` to known Mistral fields.
- Because `stream_options.include_usage` is not sent, streaming `usage` is only present if Mistral includes it natively on the final chunk.
- Model ids are date-stamped (e.g. `-YYMM`).
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- With reasoning, Mistral returns `content` as a list of `thinking` + `text` chunks; `res.message.content` / `delta.content_delta` carry only the `text` chunks, and the reasoning stays on `raw`.
- Streaming via `for delta in chat.stream(input): ...` (a generator).

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.MISTRAL, OpenAICompatConfig(api_key=os.environ["MISTRAL_API_KEY"]))
res = emb.create(EmbeddingsInput(model="mistral-embed", input=["hello"]))
```

`POST https://api.mistral.ai/v1/embeddings` with `encoding_format: "float"`; `dimensions` maps to Mistral's `output_dimension` when set (models that support a configurable size, e.g. `codestral-embed`). Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
