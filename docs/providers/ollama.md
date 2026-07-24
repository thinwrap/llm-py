# Ollama (self-host) (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Ollama is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.OLLAMA, OpenAICompatConfig())  # no auth; defaults to localhost:11434
res = chat.complete(ChatInput(
    model="llama3.1:8b",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | No | None by default (local server); if set, sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No* | Overrides the default `http://localhost:11434/v1`. *Required for a non-default host. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.OLLAMA, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

No auth by default — leave `api_key` empty and no auth header is sent. Override `base_url` for a remote host.

## Chat

### Endpoint

`POST http://localhost:11434/v1/chat/completions` (override `base_url` for a non-default host).

### Notes / quirks

- Self-host: model ids are host-local `name:tag` (e.g. `llama3.1:8b`) — a 404 means the model isn't pulled (`ollama pull …`).
- The `/v1` compat layer is documented-partial (drops `tool_choice` / `logit_bias` / `n`; base64-only images).
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.OLLAMA, OpenAICompatConfig())
res = emb.create(EmbeddingsInput(model="nomic-embed-text", input=["hello"]))
```

`POST http://localhost:11434/v1/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
