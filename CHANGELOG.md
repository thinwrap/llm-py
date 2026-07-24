# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Initial Python port of `@thinwrap/llm` (TypeScript), `thinwrap/llm` (PHP), and
  `thinwrap/llm-go` (Go): a `Chat` facade (`complete` + generator-based `stream`)
  and an `Embeddings` facade over 18 providers behind one normalized `ChatInput`
  / `ChatResult` / `ChatStreamDelta` surface, with a single typed `ConnectorError`.
- Fifteen first-class OpenAI-compatible providers share one connector driven by a
  per-provider spec registry; three natives — Anthropic (Messages), Bedrock
  (Converse, SigV4), Gemini (generateContent) — have bespoke wire adapters.
- Bring-your-own `Transport` (stdlib `urllib` default, never follows redirects)
  with optional incremental streaming (`open_stream`) and a buffered fallback.
- Zero runtime dependencies: SigV4 (Bedrock) is `hmac`-based; SSE is parsed on
  the stdlib.
