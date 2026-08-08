# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Initial Python port of `@thinwrap/llm`, `thinwrap/llm` and `thinwrap/llm-go`: `Chat`
  (`complete` + generator-based `stream`) and `Embeddings` facades over 18 providers, with a
  typed `ConnectorError`.
- Fifteen OpenAI-compatible providers share one connector driven by a spec registry; Anthropic,
  Bedrock and Gemini have bespoke wire adapters.
- Bring-your-own `Transport` (stdlib `urllib` default, never follows redirects) with optional
  incremental streaming and a buffered fallback.
- Zero runtime dependencies: SigV4 is `hmac`-based, SSE parsed on the stdlib.
- A non-2xx is classified even when a BYO `Transport` raises it instead of returning it,
  including on the streaming path.
