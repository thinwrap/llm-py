# vLLM (self-host) (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from vLLM is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.VLLM, OpenAICompatConfig())  # api_key optional; defaults to localhost:8000
res = chat.complete(ChatInput(
    model="<served-model-name>",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | No | Optional — sent as `Authorization: Bearer <api_key>` only if set (omit if the server has no `--api-key`). |
| `base_url` | `str` | No* | Overrides the default `http://localhost:8000/v1`. *Required for a remote/non-default host. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.VLLM, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Open by default. If the server was launched with `--api-key`, pass that value as `api_key`; it is sent as `Authorization: Bearer <api_key>`. When `api_key` is empty, no auth header is sent.

## Chat

### Endpoint

`POST http://localhost:8000/v1/chat/completions` (override `base_url` for a remote server).

### Notes / quirks

- Self-host: the OpenAI-compatible server is vLLM's first-class interface.
- Model ids are operator-defined (the HF repo id or `--served-model-name`).
- `tool_choice` `"auto"` requires the server launched with `--enable-auto-tool-choice` + a model-matched `--tool-call-parser`.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported (the model must be served with an embeddings task, e.g. `--task embed`).

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.VLLM, OpenAICompatConfig())
res = emb.create(EmbeddingsInput(model="<served-embedding-model>", input=["hello"]))
```

`POST http://localhost:8000/v1/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
