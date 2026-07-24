# Architecture — thinwrap-llm (Python)

## Facade → dispatch → connector

```
Chat(ProviderID.OPENAI, OpenAICompatConfig(...))   →  Chat facade
   .complete(msg) / .stream(msg)                       │  SPECS[id] → OpenAICompatConnector
                                                       ▼      (or _NATIVE[id] → native, lazy import)
                              connector (subclasses BaseConnector)
                                 .complete/.stream → send_post_json / open_post_stream → Transport → Vendor
```

- **One `Chat` facade** (`complete` + generator-based `stream`) and a separate
  `Embeddings` facade (`create`). Both dispatch by provider id.
- **Two dispatch paths in `Chat.__init__`:** a native in the `_NATIVE` registry
  (lazy `importlib`), else a row in `SPECS` → the shared `OpenAICompatConnector`.
  An unknown id → `ConnectorError(invalid_request)`.
- **`BaseConnector`** centralizes the transport seam: the injectable `Transport`
  (default non-redirect-following stdlib `urllib`), `send_post_json` / `post_raw`
  (buffered) and `open_post_stream` (streaming, with a buffered fallback when the
  transport lacks `open_stream`), plus transport-error → `provider_unavailable`.

## Two connector tiers

1. **OpenAI-compat (`_openai_compat.py`, `_openai_compat_embeddings.py`).** 15
   providers share it; `spec.py` parameterizes base URL, auth headers, quirks.
2. **Native adapters (`_anthropic.py`, `_bedrock.py`, `_gemini.py`).** Full wire
   translation to Messages / Converse / generateContent, emitting the identical
   normalized shapes.

## Streaming

`Chat.stream` returns a generator of `ChatStreamDelta`. `sse.py.parse_sse_lines`
turns an iterator of text lines into JSON data payloads (skips comments/blanks,
stops at `[DONE]`). The connector's `stream()` calls `open_post_stream` → a
`StreamResponse` (live urllib response, or a buffered wrap of `send()`), checks
the status, then yields deltas mapped from each decoded chunk, closing the stream
in a `finally`. Stateful streams keep state in generator locals.

## Invariants

- **Canonical shape = OpenAI Chat Completions.** Every connector emits the same
  `ChatInput`/`ChatResult`/`ChatStreamDelta`/`ConnectorError`. That parity is the product.
- **≥90% baseline coverage.** Sub-baseline / provider-specific request fields go
  through `Passthrough` (never emulated); vendor extras in `ChatResult.raw`. The
  one deliberate sub-90% carve-out is `reasoning.effort`.
- **Per-connector locality.** Wire translation + the status→code mapper live in
  the connector; `vendor_error` (Retry-After surfacing), JSON helpers,
  `sign_aws_v4`, and `parse_sse_lines` are building blocks connectors call.
- **`ProviderCode`** — 7 values, byte-identical across siblings. No top-level
  retry attribute (`cause["retryAfter"]` / `["retryAfterSeconds"]`).
- **Stateless / zero-dep.** SigV4 and SSE hand-rolled on the standard library.
- **Python idioms:** synchronous; errors raised as `ConnectorError`; `stream()`
  is a generator; a bring-your-own `Transport` protocol with optional `open_stream`.
