# `thinwrap-llm` (Python) — contributor guide

This folder (`.ai/`) is for developers — and the coding agents working alongside
them — who are **changing this library**. It is not usage documentation.

> **Using the package?** See [`../README.md`](../README.md) and the per-provider
> pages under [`../docs/providers/`](../docs/providers). `.ai/` is contributor-only.

## Map of this folder

- **guidelines.md** (this file) — entry point + the two "add a provider" recipes.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — facade → dispatch → connector, the two tiers, invariants.
- [`CONVENTIONS.md`](./CONVENTIONS.md) — layout, naming, error mapping, streaming, testing.

## The shape in one sentence

A consumer constructs the `Chat` facade by provider id + config
(`Chat(ProviderID.OPENAI, OpenAICompatConfig(api_key=...))`); the facade
dispatches to a connector that turns the normalized `ChatInput` into the vendor
wire and the vendor response back into the normalized `ChatResult` /
`ChatStreamDelta` / `ConnectorError`.

## Two connector tiers

1. **First-class OpenAI-compatible (15).** They share ONE connector —
   `_openai_compat.py` — parameterized by a row in the `SPECS` registry
   (`spec.py`). Adding one is a spec row, not a new class.
2. **Native adapters (Anthropic, Bedrock, Gemini).** Structurally different wire.
   Each is its own `_<id>.py` connector class, wired into the `_NATIVE` registry
   in `chat.py` (lazy import).

## Setup & verify

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=thinwrap.llm --cov-fail-under=80
```

Python ≥ 3.10. **Zero runtime dependencies — do not add any.** Bedrock's SigV4 is
`hmac`-based in `_awssigv4.py`; SSE is parsed in `sse.py`. pytest is dev-only.

## Add a first-class OpenAI-compatible provider

1. **Register the id** — add to [`enums.py`](../src/thinwrap/llm/enums.py) `ProviderID`.
2. **Add the spec row** — in [`spec.py`](../src/thinwrap/llm/spec.py): `default_base_url`, `build_auth_headers`, and any quirk (`supports_tools`, `max_tokens_field`, `stream_usage_path`, `strict_params`). Add to `EMBEDDINGS_PROVIDER_IDS` too if it exposes an OpenAI-float `/embeddings`.
3. **Test** — add a case to `tests/`.
4. **Docs** — `docs/providers/<id>.md`.

## Add a native-adapter provider

Copy [`_anthropic.py`](../src/thinwrap/llm/_anthropic.py). Touch-points:

1. **Register the id** — `enums.py`.
2. **Config** — add `<Name>Config` dataclass to [`config.py`](../src/thinwrap/llm/config.py).
3. **Register the connector** — add `ProviderID.X: ("_<id>", "<Name>Connector")` to `_NATIVE` in [`chat.py`](../src/thinwrap/llm/chat.py) (lazy import).
4. **Connector** — `_<id>.py`: `class <Name>Connector(BaseConnector)` with `__init__(self, config, transport=None)` (validate config type), `complete(msg) -> ChatResult`, and `stream(msg) -> Iterator[ChatStreamDelta]` (a generator), plus a module-level status→ProviderCode mapper.
5. **Export** any public config in [`__init__.py`](../src/thinwrap/llm/__init__.py) `__all__`.
6. **Docs** — `docs/providers/<id>.md`.

### Definition of done (the CI gates)

```bash
pytest --cov=thinwrap.llm --cov-fail-under=80   # ≥80% line coverage
python -c "import thinwrap.llm"                 # offline import smoke
```

## Invariants you must not break

- **Zero runtime deps / no vendor SDKs.** SigV4 hand-rolled on `hmac`.
- **Stateless wrapper.** No caching, retries, conversation state, or telemetry.
  Prompt caching is a consumer concern via `Passthrough`.
- **Canonical shape = OpenAI Chat Completions.** One `Chat` facade.
- **≥90% baseline-coverage rule.** A field belongs on `ChatInput`/`ChatResult`
  only if ≥90% of providers support it normalizably; else `Passthrough` (in) /
  `raw` (out) — never emulated. Deliberate sub-90% carve-out: `reasoning.effort`.
- **Every connector emits the identical `ChatInput`/`ChatResult`/`ChatStreamDelta`/`ConnectorError`.** That parity is the whole product.
- **Seven `ProviderCode` values**, raised via `ConnectorError`; retry hints ride
  in `cause["retryAfter"]` / `cause["retryAfterSeconds"]` — never a top-level attribute.
- **Python idioms:** synchronous API; a bring-your-own `Transport` (with optional
  `open_stream` for incremental SSE); `stream()` returns a generator; errors raised.
