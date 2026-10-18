# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] — 2026-10-18

### Added

- Python port of `@thinwrap/llm` / `thinwrap/llm` / `thinwrap/llm-go`: `Chat` (`complete` + generator `stream`) and `Embeddings` over 18 providers, with a typed `ConnectorError`.
- Fifteen OpenAI-compatible providers share one spec-driven connector; Anthropic, Bedrock and Gemini have native adapters.
- Bring-your-own `Transport` (stdlib `urllib` default, no redirects) with incremental streaming or a buffered fallback.
- Zero runtime dependencies: SigV4 is `hmac`-based, SSE parsed on the stdlib.
- A non-2xx is classified even when a BYO `Transport` raises it instead of returning it.
- `provider_metadata` on `ToolCall`, `ChatMessage`, `AssistantMessage`, `ToolCallDelta` and `ChatStreamDelta` carries Gemini thought signatures and Anthropic thinking blocks across tool loops.
- The wheel ships `py.typed`.

### Changed

- **Breaking:** Anthropic `reasoning` sends adaptive thinking with `output_config.effort`; models before Claude 4.6 need budget thinking via passthrough.
- Anthropic's default `max_tokens` is 16000 (was 4096), so always-thinking models do not stop at `length` before any text.
- Gemini sends tool schemas as `parametersJsonSchema` and `json_schema` response formats as `responseJsonSchema`.
- Usage counts cache reads/writes (Anthropic, Bedrock) and tool-use prompts (Gemini) as input, thinking as output (Gemini), and `total_tokens` is input + output.
- **Breaking:** Bedrock `tool_choice` `none` omits `toolConfig`, which Bedrock may reject once the history holds tool blocks.
- Unlisted 4xx statuses map to `invalid_request` and 408 to `provider_unavailable`.
- Azure OpenAI docs note that `max_completion_tokens` needs the v1 API or `api-version` 2024-09-01-preview or later.
- The default transport's socket timeout is 600 s, so long non-streaming reasoning calls finish.
- Release publishes only tags on `main`, from a `pypi`-environment job with SHA-pinned actions.

### Fixed

- The default transport sends a `thinwrap-llm-python` User-Agent; Cloudflare refused urllib's own with a 403 on Groq.
- OpenAI and Azure OpenAI send `max_output_tokens` as `max_completion_tokens`, so reasoning models accept it.
- A 2xx chat body carrying an `error` (OpenRouter-style) or no `choices` raises a `ConnectorError` instead of an empty success.
- Embeddings raise on a 2xx `error`, a missing `data`, a vector-count mismatch or a non-numeric cell instead of coercing to `0.0`.
- Anthropic, Bedrock and Gemini raise on a 2xx body missing `content` / `output` / `candidates`.
- Array-shaped `message.content` / `delta.content` (Mistral reasoning chunks) yields the answer text, not an empty string.
- Gemini tool loops replay each call's `thoughtSignature` (or Google's bypass value), so Gemini 3 no longer rejects them.
- Gemini echoes its own `functionCall.id`; synthetic ids are index-based, so parallel same-name calls stay distinct.
- Gemini `API_KEY_INVALID` maps to `auth_failed`.
- Context-length errors match only the vendors' overflow wording (Anthropic, Gemini, Bedrock).
- Unmapped finish reasons (`model_context_window_exceeded`, Gemini `CONTINUATION` and image-safety reasons) map to their closest value.
- 402 (out of credit) maps to `rate_limited` and 413 to `invalid_request`; Bedrock 408/424 to `provider_unavailable`; a mid-stream error honours its code or type.
- Mistral embeddings send `dimensions` as `output_dimension`.
- Gemini reports streamed usage once instead of on every chunk.
- An unknown Gemini reasoning effort uses the medium budget instead of disabling thinking.
- Anthropic streaming no longer reports server-tool input as a caller tool call.
- A 2xx stream with a non-SSE body, or one that ends before the provider's terminal event, raises `provider_unavailable` instead of ending silently.
- A failed read of a streaming error body raises a `ConnectorError` and closes the stream.
- The default transport rejects `http://` with an actionable `invalid_request`.
- A header with a line break raises `invalid_request` without echoing its value.
- Bedrock validates `region`, signs the host without a default port and merges header names case-insensitively.
- `Retry-After` accepts only an integer delay or a strict HTTP-date (a past date gives 0); empty, negative, decimal or free-text values yield no `retryAfterSeconds`.
- Gemini resolves a tool result's function name from the preceding assistant turn, so repeated synthetic ids no longer pair results with the wrong function.
- Anthropic rebuilds every assistant turn after a rebuilt one, and replays a turn verbatim only when its tool inputs still match.
- Anthropic streamed usage merges `message_start` and `message_delta` field by field, so a null or partial restatement no longer drops tokens.
- Anthropic streaming keeps a block's start `input` when no deltas follow, and sends `provider_metadata` only on the stop delta.
- Anthropic and Bedrock drop empty text parts, which both vendors reject.
- An empty Anthropic reasoning effort sends `medium`.
- Bedrock signs a `base_url` path prefix.
- OpenAI-compatible errors honour `error.code` `context_length_exceeded` and OpenAI's content-policy and moderation wording.
- An error object inside a 2xx body keeps the 2xx as its status and is classified by its `error.code` (numeric, or a name such as `insufficient_quota`), else `unknown`, in chat and embeddings.
- A mid-stream error frame with no usable code or type is `unknown`.
- `HttpRequest`'s repr redacts credential headers, the URL path and query, and the body.
