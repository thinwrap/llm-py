# OpenAI (Python)

Chat + embeddings via the shared OpenAI-compatible connector (a spec row, not a per-provider class). Switching to/from OpenAI is just the provider id + `model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```python
import os
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

chat = Chat(ProviderID.OPENAI, OpenAICompatConfig(api_key=os.environ["OPENAI_API_KEY"]))
res = chat.complete(ChatInput(
    model="gpt-4o",
    messages=[ChatMessage(role="user", content="Hi")],
))
print(res.message.content, res.finish_reason, res.usage.total_tokens)
```

## Configuration

`OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `api_key` | `str` | Yes | Sent as `Authorization: Bearer <api_key>`. |
| `base_url` | `str` | No | Overrides the default `https://api.openai.com/v1`. |
| `headers` | `Mapping[str, str]` | No | Merged onto every request. |

Inject a custom transport with `Chat(ProviderID.OPENAI, cfg, transport=my_transport)`; the default `UrllibTransport` never follows redirects.

## Auth setup

Create an API key at <https://platform.openai.com/api-keys>. It is sent verbatim as `Authorization: Bearer <api_key>`.

## Chat

### Endpoint

`POST https://api.openai.com/v1/chat/completions`

### Notes / quirks

- The reference provider — its Chat Completions shape is the canonical facade shape.
- `max_output_tokens` maps to `max_completion_tokens` (OpenAI deprecates `max_tokens`, which o-series models reject; the cap includes reasoning tokens). `reasoning.effort` maps to `reasoning_effort`.
- Reasoning models (o-series / GPT-5-class) reject `temperature`/`top_p`/penalties with HTTP 400; guard by model family and leave those normalized sampling fields unset.
- OpenAI also ships the newer Responses API; this connector targets Chat Completions.
- **Function tools together with reasoning:** current models (gpt-5.4, gpt-5.4-mini, gpt-5.5, gpt-5.6-*) reject function tools combined with reasoning on Chat Completions with a 400, and gpt-5.6 does so even when `reasoning` is unset. The workaround is `Passthrough(body={"reasoning_effort": "none"})`, accepted on gpt-5.4 and later (gpt-5-mini and o4-mini reject `"none"`). Moving this connector to the Responses API is planned.
- Streaming via `for delta in chat.stream(input): print(delta.content_delta, end="")` (a generator). `stream_options.include_usage` is set so the final delta carries `usage`.

## Embeddings

Supported.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.OPENAI, OpenAICompatConfig(api_key=os.environ["OPENAI_API_KEY"]))
res = emb.create(EmbeddingsInput(model="text-embedding-3-small", input=["hello", "world"]))
print(res.embeddings)  # list[list[float]], one vector per input in input order
```

`POST https://api.openai.com/v1/embeddings` with `encoding_format: "float"`. `dimensions` maps to `dimensions` (Matryoshka) when set. Vectors are returned in input order on `res.embeddings`.

## Error mapping & passthrough

- A non-2xx response raises a `ConnectorError`; catch it and read `e.provider_code` (a `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`) and `e.status_code`.
- `Retry-After` (when present) rides on `e.cause["retryAfter"]` (raw) and `e.cause["retryAfterSeconds"]` (parsed); the raw vendor body is on `e.cause["raw"]`.
- Sub-baseline request fields (`seed`, `frequency`/`presence_penalty`, `logit_bias`, `n`, prompt caching) go via `Passthrough(body=..., headers=..., query=...)`; vendor extras on the response are on `res.raw`.
