# thinwrap-llm (Python)

Unified, **SDK-free**, **zero-runtime-dependency** Python wrapper for **chat
completions** (with streaming) and **embeddings** across 18 LLM providers. Switch
vendor by changing the provider id + model; the input and output shapes stay
identical. Synchronous, stateless, standard-library-only — bring your own HTTP
transport if you want `httpx`/`requests`.

```bash
pip install thinwrap-llm
```

Requires Python ≥ 3.10. No runtime dependencies — Bedrock's SigV4 is `hmac`-based
and SSE is parsed on the stdlib.

## Two-minute chat

```python
from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID, ConnectorError

chat = Chat(ProviderID.OPENAI, OpenAICompatConfig(api_key="..."))
try:
    res = chat.complete(ChatInput(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Say hi in one word.")],
    ))
    print(res.message.content, res.usage.total_tokens)
except ConnectorError as e:
    print(e.provider_code.value, e.provider_message)
```

## Streaming

```python
for delta in chat.stream(ChatInput(model="gpt-4o-mini", messages=msgs)):
    print(delta.content_delta, end="", flush=True)
```

## Providers

The **provider id selects the connector**; `ChatInput` / `ChatResult` are
identical across all of them.

**Chat (18):** the 15 first-class OpenAI-compatible providers —
`ProviderID.OPENAI`, `AZURE_OPENAI`, `OPENROUTER`, `GROQ`, `TOGETHER`,
`FIREWORKS`, `DEEPSEEK`, `XAI`, `MISTRAL`, `PERPLEXITY`, `DEEPINFRA`,
`CLOUDFLARE`, `VLLM`, `OLLAMA`, `LMSTUDIO` — all take `OpenAICompatConfig`; plus 3
natives: `ANTHROPIC` (`AnthropicConfig`), `BEDROCK` (`BedrockConfig`), `GEMINI`
(`GeminiConfig`).

**Embeddings (11):** the OpenAI-float subset — `OPENAI`, `AZURE_OPENAI`,
`OPENROUTER`, `TOGETHER`, `FIREWORKS`, `MISTRAL`, `DEEPINFRA`, `CLOUDFLARE`,
`VLLM`, `OLLAMA`, `LMSTUDIO`.

```python
from thinwrap.llm import Embeddings, EmbeddingsInput, OpenAICompatConfig, ProviderID

emb = Embeddings(ProviderID.OPENAI, OpenAICompatConfig(api_key="..."))
out = emb.create(EmbeddingsInput(model="text-embedding-3-small", input=["hello", "world"]))
# out.embeddings is list[list[float]] in input order
```

Per-provider details live in [`docs/providers/`](docs/providers/).

## The ≥90% baseline & passthrough

Only fields ≥90% of providers support normalizably are first-class on
`ChatInput` / `ChatResult`; everything else rides through `Passthrough` (request)
/ `raw` (response) and is **never emulated**:

```python
from thinwrap.llm import Passthrough

res = chat.complete(ChatInput(model="...", messages=msgs,
                              passthrough=Passthrough(body={"logprobs": True})))
_ = res.raw  # verbatim vendor body — reasoning CoT, cached-token counts, cost, ...
```

## Errors

Every failure raises `ConnectorError`: `provider_code` is one of `rate_limited`,
`auth_failed`, `provider_unavailable`, `invalid_request`,
`context_length_exceeded`, `content_filtered`, `unknown`. The raw `Retry-After`
rides in `cause["retryAfter"]` (+ parsed `cause["retryAfterSeconds"]`) — there is
no top-level retry attribute.

## Bring your own transport

```python
chat = Chat(ProviderID.OPENAI, OpenAICompatConfig(api_key="..."), transport=my_transport)
```

`my_transport` implements `Transport` (`send(HttpRequest) -> HttpResponse`); add
an `open_stream` method for true incremental streaming (otherwise a buffered
fallback is used). The default stdlib `urllib` transport does both and never
follows redirects. The wrapper is stateless — no caching, retries, or
conversation memory.

## Relationship to the siblings

One of four thinwrap llm libraries sharing a normalized surface and
byte-identical provider ids / error codes: TypeScript (`@thinwrap/llm`), PHP
(`thinwrap/llm`), Go (`thinwrap/llm-go`), Python (this package).

## Security

Report vulnerabilities **privately** — please do not open a public issue. Preferred: a
[private security advisory](https://github.com/thinwrap/llm-py/security/advisories/new)
on this repository. Alternatively, email **security@thinwrap.dev**. Include the affected
versions and a minimal reproduction if you have one.

A vulnerability in a *provider's* own API or service belongs to that vendor rather than
to this wrapper — please report those upstream.

## License

MIT © Dmitry Polyanovsky & contributors
