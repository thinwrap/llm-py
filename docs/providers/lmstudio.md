# LM Studio (self-host) (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from LM Studio is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.LMSTUDIO, OpenAICompatConfig())  # api_key optional; defaults to localhost:1234
res = chat.complete(ChatInput(
    model="<loaded-model>",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | No | Optional — sent as `Authorization: Bearer <api_key>` only if set (API-token auth since v0.4.0; off by default). |
| `base_url` | `str` | No* | Overrides the default `http://localhost:1234/v1`. *Required if not on `:1234`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.LMSTUDIO, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Local single-user server; auth is off by default (leave `api_key` empty → no auth header). If you enabled API-token auth, pass the token as `api_key`; it is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST http://localhost:1234/v1/chat/completions` (override `base_url` if not on `:1234`).

### Notes / quirks

- Self-host: with a single model loaded, the server ignores the requested `model` id and serves whatever is loaded.
- Structured output is grammar-enforced (llama.cpp / MLX); small models can be unreliable.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported (an embedding model must be loaded).

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.LMSTUDIO, OpenAICompatConfig())
res = emb.create(EmbeddingsInput(model="<loaded-embedding-model>", input=["hello"]))
```

`POST http://localhost:1234/v1/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
