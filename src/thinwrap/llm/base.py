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
        _check_header_values(h)
        req = HttpRequest(method="POST", url=url, headers=h, body=data)
        t = self._transport
        try:
            if hasattr(t, "open_stream"):
                return t.open_stream(req)  # type: ignore[attr-defined]
            return BufferedStream(t.send(req))
        except ConnectorError:
            raise
        except Exception as exc:
            # …unless the provider answered and the transport raised the non-2xx
            # instead of returning it. A non-2xx never streams anyway, so a
            # buffered wrapper is the faithful shape for the connector to map.
            rebuilt = _response_from_raised_http_error(exc)
            if rebuilt is not None:
                return BufferedStream(rebuilt)
            descriptor, cause = _sanitize_transport_error(exc)
        # Raised OUTSIDE the except block on purpose: implicit exception chaining
        # only happens while an exception is handled, so neither __cause__ nor
        # __context__ ends up referencing the URL-bearing transport exception
        # (a BYO transport's error can embed a credential-bearing request URL).
        raise ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, message=descriptor, provider_message=descriptor, cause=cause)

    def _send(self, request: HttpRequest) -> HttpResponse:
        _check_header_values(request.headers)
        try:
            return self._transport.send(request)
        except ConnectorError:
            raise
        except Exception as exc:
            rebuilt = _response_from_raised_http_error(exc)
            if rebuilt is not None:
                return rebuilt
            descriptor, cause = _sanitize_transport_error(exc)
        # Raised OUTSIDE the except block — see open_post_stream for why.
        raise ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, message=descriptor, provider_message=descriptor, cause=cause)

    @staticmethod
    def _read_error_body(stream: Any) -> Any:
        """Decode the body of a non-2xx stream, always closing it. A failed read
        yields ``None`` so the caller still classifies by status: a raw socket
        exception would escape the typed surface and can carry the request URL."""
        from ._util import decode_json

        try:
            return decode_json(stream.read_all())
        except Exception:
            return None
        finally:
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _require_event_stream(stream: Any) -> None:
        """Guard a 2xx streaming response before reading it as SSE. A body
        labelled as anything but ``text/event-stream`` (a JSON completion from a
        proxy that dropped ``stream: true``, an HTML gateway page) holds no
        events, so reading it would end the stream with nothing and no error. A
        missing content-type is let through; the end-of-stream check still
        catches an empty body."""
        headers = getattr(stream, "headers", None) or {}
        content_type = ""
        for k, v in headers.items():
            if str(k).lower() == "content-type":
                content_type = str(v or "").strip()
                break
        if not content_type or "text/event-stream" in content_type.lower():
            return
        media_type = content_type.split(";")[0].strip()
        try:
            stream.close()
        except Exception:
            pass
        msg = f"Provider returned a successful status with a non-SSE body ({media_type}) for a streaming request"
        raise ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, status_code=stream.status, message=msg, provider_message=msg)

    @staticmethod
    def _stream_ended_early_error() -> ConnectorError:
        """The error for a stream whose body ended before the provider's terminal
        event. Every supported provider closes a complete stream with one, so an
        EOF before it is a truncated answer (dropped connection, proxy timeout),
        not a short one."""
        msg = "Stream ended before the provider sent its terminal event"
        return ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, message=msg, provider_message=msg)

    def _iter_sse(self, stream: Any, state: Optional[dict] = None) -> Iterator[str]:
        """Drive a streaming response through the SSE parser, normalizing a
        mid-stream transport read error (e.g. a BYO transport raising
        IncompleteRead/ConnectionReset while lines are pulled) into a
        ConnectorError — never a raw socket exception, and without chaining the
        (URL-bearing) transport exception via __cause__/__context__."""
        from .sse import parse_sse_lines

        return parse_sse_lines(self._guard_lines(stream), state)

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


def _check_header_values(headers: Mapping[str, str]) -> None:
    """Refuse a header carrying a line break or NUL before it reaches the
    transport. It is a configuration error (typically an API key pasted with a
    trailing newline), not an unreachable provider, and http.client's own
    ValueError quotes the value, which for a credential header is the secret.
    The message names only the header."""
    for name, value in headers.items():
        if any(c in str(value) for c in "\r\n\x00") or any(c in str(name) for c in "\r\n\x00:"):
            raise ConnectorError(
                ProviderCode.INVALID_REQUEST,
                message=f"Header '{_printable(name)}' contains a line break or NUL character; check the configured credential and headers",
                provider_message="Invalid header value",
            )


def _printable(name: Any) -> str:
    return "".join(c if c.isprintable() else "?" for c in str(name))


def _sanitize_transport_error(exc: BaseException) -> tuple[str, dict[str, Any]]:
    """Capture ONLY the exception class name — never str(exc). A BYO
    requests/httpx exception embeds the full request URL, which for a
    query-credential provider would leak the secret via the message and cause."""
    descriptor = f"Upstream transport error ({type(exc).__name__})"
    return descriptor, {"raw": {"exception": type(exc).__name__}}


def _response_from_raised_http_error(exc: BaseException) -> Optional[HttpResponse]:
    """Rebuild an :class:`HttpResponse` from an exception that carries one, or
    return ``None`` when the failure is a genuine transport failure.

    :class:`Transport` requires a non-2xx to be RETURNED, and the bundled
    urllib transport obeys that. A BYO transport may not: ``requests`` and
    ``httpx`` both raise from ``raise_for_status()``, and a thin urllib wrapper
    can let :class:`urllib.error.HTTPError` escape. The provider answered, so the
    answer is what the caller must see — without this, every 401/429/503
    collapses into ``provider_unavailable`` with no status, and a rate-limited
    caller cannot tell that it should back off.

    Duck-typed on the shape (``.response.status_code``, ``.code`` + ``.read()``),
    never on an import: the library has zero runtime dependencies."""
    for candidate in (getattr(exc, "response", None), exc):
        if candidate is None:
            continue
        status = _read_http_status(candidate)
        if status is None:
            continue
        return HttpResponse(
            status=status,
            headers=_read_http_headers(candidate),
            body=_read_http_body(candidate),
        )
    return None


def _read_http_status(candidate: Any) -> Optional[int]:
    """The HTTP status a response-like object carries, if it plausibly is one."""
    status = getattr(candidate, "status_code", None)
    if status is None:
        status = getattr(candidate, "status", None)
    if status is None and (hasattr(candidate, "read") or hasattr(candidate, "headers")):
        # `code` alone is too generic to trust — urllib.error.HTTPError pairs it
        # with a readable body, an unrelated exception with a `.code` does not.
        status = getattr(candidate, "code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status if 200 <= status <= 599 else None


def _read_http_headers(candidate: Any) -> dict:
    headers = getattr(candidate, "headers", None)
    if headers is None:
        return {}
    try:
        items = headers.items()
    except Exception:
        return {}
    return {str(k).lower(): str(v) for k, v in items}


def _read_http_body(candidate: Any) -> bytes:
    for attribute in ("content", "text", "body"):
        try:
            value = getattr(candidate, attribute, None)
        except Exception:
            # httpx raises on `.content` for an unread streaming response.
            continue
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8", "replace")
    read = getattr(candidate, "read", None)
    if callable(read):
        try:
            value = read()
        except Exception:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8", "replace")
    return b""
