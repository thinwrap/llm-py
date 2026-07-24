# Cloudflare Workers AI (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Cloudflare is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.CLOUDFLARE, OpenAICompatConfig(
    api_key=os.environ["CLOUDFLARE_API_TOKEN"],
    base_url="https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1",  # required
))
res = chat.complete(ChatInput(
    model="@cf/meta/llama-3.1-8b-instruct",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | **Yes** | No public default — construction raises `invalid_request` without it (it embeds your account id). |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.CLOUDFLARE, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create an API token (Workers AI scope) in the Cloudflare dashboard. It is sent as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST <base_url>/chat/completions` — `base_url` is `https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1`. Omitting `base_url` raises an `invalid_request` `ConnectorError`.

### Notes / quirks

- **`base_url` is required** (no fixed public default — it embeds your account id).
- Model ids are `@cf/{provider}/{model}` (some `@hf/…`), required verbatim.
- JSON mode is **not strict** (no schema-adherence guarantee) and can't stream.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.CLOUDFLARE, OpenAICompatConfig(
    api_key=os.environ["CLOUDFLARE_API_TOKEN"],
    base_url="https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1",
))
res = emb.create(EmbeddingsInput(model="@cf/baai/bge-large-en-v1.5", input=["hello"]))
```

`POST <base_url>/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
