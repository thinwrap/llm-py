# Azure OpenAI (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from Azure OpenAI is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.AZURE_OPENAI, OpenAICompatConfig(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://<resource>.openai.azure.com/openai/v1",  # required
))
res = chat.complete(ChatInput(
    model="<your-deployment-name>",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as the `api-key: <api_key>` header (not Bearer). |
| `base_url` | `str` | **Yes** | No public default — construction raises `invalid_request` without it. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request (put an Entra bearer token here for OAuth mode). |

Inject a custom transport with `Chat(ProviderID.AZURE_OPENAI, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Get the key + resource endpoint from the Azure portal. The connector sends it as the `api-key` header. Microsoft Entra OAuth is also accepted on the v1 GA API — supply that bearer token via `headers` instead.

## Chat

### Endpoint

`POST <base_url>/chat/completions` — `base_url` is `https://<resource>.openai.azure.com/openai/v1` (v1 GA API, opt-in from Aug 2025; no `api-version` query needed). Omitting `base_url` raises an `invalid_request` `ConnectorError`.

### Notes / quirks

- **`base_url` is required** (no fixed public default).
- `model` is the per-resource **deployment name**, not a portable global model id.
- `max_output_tokens` maps to `max_tokens`. `reasoning.effort` maps to `reasoning_effort`.
- Streaming via `for delta in chat.stream(input): ...` (a generator). `stream_options.include_usage` is set.

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.AZURE_OPENAI, OpenAICompatConfig(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://<resource>.openai.azure.com/openai/v1",
))
res = emb.create(EmbeddingsInput(model="<embeddings-deployment>", input=["hello"]))
```

`POST <base_url>/embeddings` with `encoding_format: "float"`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` rides on `e.cause["retryAfter"]` / `e.cause["retryAfterSeconds"]`; raw body on `e.cause["raw"]`.
- Sub-baseline request fields (`seed`, penalties, `logit_bias`, `n`, prompt caching) go via `Passthrough(body=..., headers=..., query=...)`; response extras on `res.raw`.
