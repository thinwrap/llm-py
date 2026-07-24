# Conventions — thinwrap-llm (Python)

## File layout

- One module per **shared concern**: `enums.py` (ProviderID), `errors.py`
  (ProviderCode, ConnectorError, `vendor_error`, `invalid_request`,
  `parse_retry_after`), `config.py` (all config dataclasses), `chat.py` (chat
  dataclasses + `Chat` facade + `_NATIVE` registry), `embeddings.py` (embeddings
  dataclasses + `Embeddings` facade), `passthrough.py`, `transport.py` (Transport
  protocol + UrllibTransport + stream response types), `base.py` (BaseConnector),
  `sse.py` (`parse_sse_lines`), `_awssigv4.py`, `_util.py`, `_jsonpath.py`,
  `spec.py` (the `SPECS` registry + `EMBEDDINGS_PROVIDER_IDS`).
- Shared connectors: `_openai_compat.py` (chat), `_openai_compat_embeddings.py`.
- One module per **native**: `_<id>.py` holds its connector + module-level
  helpers. Tests are `tests/test_<id>.py`.

## Naming

- Public API (re-exported from `__init__.py` `__all__`): `Chat`, `Embeddings`,
  the dataclasses (`ChatInput`, `ChatMessage`, `ChatResult`, `ChatStreamDelta`,
  `ToolCall`, `ToolCallDelta`, `ToolDefinition`, `ToolChoice`, `ResponseFormat`,
  `ReasoningOptions`, `Usage`, `AssistantMessage`, `ContentPart`,
  `EmbeddingsInput`/`Result`/`Usage`), `text_part`/`image_part`, `Passthrough`,
  `ConnectorError`, `ProviderCode`, `ProviderID`, the `<Name>Config` dataclasses,
  `Transport`, `UrllibTransport`, `HttpRequest`, `HttpResponse`.
- Private: connector modules are underscore-prefixed (`_openai_compat.py`,
  `_anthropic.py`); connector classes are `<Name>Connector`; module helpers are
  `_`-prefixed (module-private — no cross-file collision).

## Error mapping

Each connector has a module-level status→ProviderCode mapper porting the sibling
truth table:

| HTTP | ProviderCode |
|---|---|
| 401 / 403 | `AUTH_FAILED` |
| 429 | `RATE_LIMITED` |
| 5xx | `PROVIDER_UNAVAILABLE` |
| 400 / 404 / 422 | `INVALID_REQUEST` (or `CONTEXT_LENGTH_EXCEEDED` / `CONTENT_FILTERED` on a body signal) |
| transport | `PROVIDER_UNAVAILABLE` (status None) |

Raise with `vendor_error(status, headers, raw_body, code, provider_message)`. Pre-flight / unknown provider: `invalid_request(msg)`.

## Streaming

`stream()` is a generator. Get a `StreamResponse` from `self.open_post_stream`,
check `ok_status(stream.status)` (read the error body via `stream.read_all()` on
failure), then `for payload in parse_sse_lines(stream.iter_lines()): ... yield`,
closing the stream in a `finally`. Keep per-stream state in generator locals.

## Testing

- `tests/test_<provider>.py`, standard `pytest`; helpers in `tests/helpers.py`
  (`FakeTransport` implements both `send` and `open_stream`; `resp`, `body_json`,
  `path_of`, `qget`).
- Inject `FakeTransport(...)` via the facade's `transport=` kwarg; assert the
  captured request (URL/headers/body), the normalized result, streaming
  accumulation, and `ConnectorError` (`provider_code`, `status_code`) via
  `pytest.raises`. Never assert a top-level retry attribute.

## Python / build

- `pytest --cov=thinwrap.llm --cov-fail-under=80`; offline import smoke.
- Python ≥ 3.10; no third-party imports. `from __future__ import annotations` at
  the top of every module.
