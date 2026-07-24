"""BaseConnector — the shared HTTP seam (buffered + streaming)."""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional

from ._util import compact_json
from .errors import ConnectorError, ProviderCode
from .transport import (
    BufferedStream,
    HttpRequest,
    HttpResponse,
    StreamResponse,
    Transport,
    default_transport,
)


class BaseConnector:
    """Centralizes the injectable transport and transport-error normalization."""

    def __init__(self, transport: Optional[Transport] = None) -> None:
        self._transport: Transport = transport or default_transport()

    def send_post_json(self, url: str, body: Any, headers: Mapping[str, str]) -> HttpResponse:
        try:
            data = compact_json(body)
        except (TypeError, ValueError) as exc:
            raise ConnectorError(ProviderCode.INVALID_REQUEST, message="Failed to encode request body", provider_message="Failed to encode request body", cause=exc) from exc
        return self._send(HttpRequest(method="POST", url=url, headers=self._with_json(headers), body=data))

    def post_raw(self, url: str, content_type: str, body: bytes, headers: Mapping[str, str]) -> HttpResponse:
        h = dict(headers)
        h["Content-Type"] = content_type
        return self._send(HttpRequest(method="POST", url=url, headers=h, body=body))

    def open_post_stream(self, url: str, body: Any, headers: Mapping[str, str]) -> StreamResponse:
        try:
            data = compact_json(body)
        except (TypeError, ValueError) as exc:
            raise ConnectorError(ProviderCode.INVALID_REQUEST, message="Failed to encode request body", provider_message="Failed to encode request body", cause=exc) from exc
        h = self._with_json(headers)
        h["Accept"] = "text/event-stream"
        req = HttpRequest(method="POST", url=url, headers=h, body=data)
        t = self._transport
        try:
            if hasattr(t, "open_stream"):
                return t.open_stream(req)  # type: ignore[attr-defined]
            return BufferedStream(t.send(req))
        except ConnectorError:
            raise
        except Exception as exc:
            descriptor, cause = _sanitize_transport_error(exc)
        # Raised OUTSIDE the except block on purpose: implicit exception chaining
        # only happens while an exception is handled, so neither __cause__ nor
        # __context__ ends up referencing the URL-bearing transport exception
        # (a BYO transport's error can embed a credential-bearing request URL).
        raise ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, message=descriptor, provider_message=descriptor, cause=cause)

    def _send(self, request: HttpRequest) -> HttpResponse:
        try:
            return self._transport.send(request)
        except ConnectorError:
            raise
        except Exception as exc:
            descriptor, cause = _sanitize_transport_error(exc)
        # Raised OUTSIDE the except block — see open_post_stream for why.
        raise ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, message=descriptor, provider_message=descriptor, cause=cause)

    def _iter_sse(self, stream: Any) -> Iterator[str]:
        """Drive a streaming response through the SSE parser, normalizing a
        mid-stream transport read error (e.g. a BYO transport raising
        IncompleteRead/ConnectionReset while lines are pulled) into a
        ConnectorError — never a raw socket exception, and without chaining the
        (URL-bearing) transport exception via __cause__/__context__."""
        from .sse import parse_sse_lines

        return parse_sse_lines(self._guard_lines(stream))

    def _guard_lines(self, stream: Any) -> Iterator[str]:
        it = iter(stream.iter_lines())
        while True:
            try:
                line = next(it)
            except StopIteration:
                return
            except ConnectorError:
                raise
            except Exception as exc:
                descriptor, cause = _sanitize_transport_error(exc)
                break
            yield line
        # Raised OUTSIDE the except block (via break) so __context__ is not set.
        raise ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, message=descriptor, provider_message=descriptor, cause=cause)

    @staticmethod
    def _with_json(headers: Mapping[str, str]) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(headers)
        return h


def _sanitize_transport_error(exc: BaseException) -> tuple[str, dict[str, Any]]:
    """Capture ONLY the exception class name — never str(exc). A BYO
    requests/httpx exception embeds the full request URL, which for a
    query-credential provider would leak the secret via the message and cause."""
    descriptor = f"Upstream transport error ({type(exc).__name__})"
    return descriptor, {"raw": {"exception": type(exc).__name__}}
