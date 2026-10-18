"""The bring-your-own HTTP seam, with optional streaming support.

A ``Transport`` need only implement ``send`` (buffered request/response). A
transport that also implements ``open_stream`` gets true incremental SSE
streaming; otherwise the connector falls back to buffering the whole response
and parsing it as SSE (functionally identical, just not token-by-token). The
default stdlib ``urllib`` transport implements both. No third-party dependency
is forced on anyone.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional, Protocol, runtime_checkable

from .errors import ConnectorError, ProviderCode, invalid_request

# Hard cap on a single SSE line. http.client's readline() is otherwise
# unbounded, so a newline-less streaming body would grow memory without limit
# (a DoS). Mirrors the 16 MiB cap the sibling libs use.
_MAX_SSE_LINE = 16 * 1024 * 1024

# SSE line terminators are CR, LF, or CRLF ONLY. `str.splitlines()` additionally
# breaks on U+2028/U+2029/U+0085(NEL)/VT/FF, any of which can legitimately appear
# INSIDE a JSON string value in a `data:` payload — splitting there would corrupt
# the event. Split on the three real terminators only.
_SSE_LINE_RE = re.compile(r"\r\n|\r|\n")


def _split_sse_lines(text: str) -> list[str]:
    lines = _SSE_LINE_RE.split(text)
    # A trailing terminator yields a final empty element; drop it to match
    # `str.splitlines()` semantics (no phantom blank line at end of body).
    if lines and lines[-1] == "":
        lines.pop()
    return lines

# The default transport is HTTPS-only. This closes the stdlib opener's file://,
# ftp:// and data: handlers (so a config-/consumer-supplied base_url cannot turn
# into a local-file read) AND refuses cleartext http. A consumer with a
# self-hosted http endpoint (e.g. a local Ollama/vLLM) injects their own
# Transport rather than relaxing TLS by default.
_ALLOWED_SCHEMES = ("https",)

# Default socket timeout (seconds) so a blackholed connection cannot hang the
# calling thread indefinitely. It bounds each blocking read, not the whole call,
# but a non-streaming request sends nothing until the model is done, so a long
# reasoning call needs the full budget: 600 s matches the official OpenAI and
# Anthropic SDK defaults. Consumers needing a different budget pass
# UrllibTransport(timeout=...) or inject their own Transport.
_DEFAULT_TIMEOUT_SECONDS = 600.0


def _assert_http_scheme(url: str) -> None:
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        # A configuration problem, not an unreachable provider, so a typed
        # invalid_request that says what to do. The URL is never echoed: a
        # self-hosted base URL may carry userinfo credentials.
        raise _scheme_refused(scheme)


def _scheme_refused(scheme: str) -> ConnectorError:
    if scheme == "http":
        return invalid_request(
            "The default transport is HTTPS-only and refuses http:// URLs. To reach a "
            "cleartext http endpoint (e.g. a local vLLM, Ollama or LM Studio server), "
            "inject your own Transport via the facade's transport= argument."
        )
    return invalid_request(
        f"The default transport only sends https:// URLs (got scheme: {scheme or '(none)'}). "
        "Inject your own Transport via the facade's transport= argument for anything else."
    )


# Header values a repr may show; every other value (Authorization, x-api-key,
# x-goog-api-key, X-Amz-Security-Token, …) is a credential or may be.
_REPR_VISIBLE_HEADERS = frozenset({
    "accept", "content-type", "content-length", "user-agent", "host",
    "x-amz-date", "x-amz-content-sha256", "anthropic-version",
})


@dataclass(frozen=True, repr=False)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None

    def __repr__(self) -> str:
        # A BYO transport that logs its request must not log credentials: header
        # values hold API keys and session tokens, a passthrough query can hold a
        # token, and the body is the caller's prompt. Only the origin, header
        # names and safe header values, and the body size are shown.
        # Never raises: a repr runs inside logging, so a malformed URL (bad port,
        # non-string) must still render.
        try:
            parts = urllib.parse.urlsplit(str(self.url))
            # netloc minus any userinfo; the raw port text, never parsed.
            host = parts.netloc.rpartition("@")[2]
            url = f"{parts.scheme}://{host}" + ("/<redacted>" if (parts.path not in ("", "/") or parts.query) else "")
        except Exception:
            url = "<redacted>"
        try:
            headers = {
                k: (v if str(k).lower() in _REPR_VISIBLE_HEADERS else "<redacted>") for k, v in self.headers.items()
            }
        except Exception:
            headers = {"<unreadable>": "<redacted>"}
        try:
            size = "None" if self.body is None else f"<{len(self.body)} bytes>"
        except Exception:
            size = "<unknown size>"
        return f"HttpRequest(method={self.method!r}, url={url!r}, headers={headers!r}, body={size})"


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # keys lower-cased
    body: bytes


@runtime_checkable
class Transport(Protocol):
    """Send one request, return one response. Raise on transport failure; return
    (never raise) non-2xx HTTP responses."""

    def send(self, request: HttpRequest) -> HttpResponse: ...


class StreamResponse:
    """A live (or buffered) streaming response. ``iter_lines`` yields decoded
    text lines (newline-stripped); ``read_all`` drains the remaining body (used
    to read an error body on a non-2xx stream)."""

    status: int
    headers: Mapping[str, str]

    def iter_lines(self) -> Iterator[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def read_all(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _lower_headers(h) -> dict[str, str]:
    return {k.lower(): v for k, v in (h or {}).items()}


class _UrllibStream(StreamResponse):
    def __init__(self, resp) -> None:
        self._resp = resp
        self.status = resp.status
        self.headers = _lower_headers(resp.headers)

    def iter_lines(self) -> Iterator[str]:
        # readline(cap+1) bounds each read: a line longer than the cap comes back
        # at cap+1 bytes without ever having hit a newline → refuse rather than
        # buffer unbounded.
        while True:
            raw = self._resp.readline(_MAX_SSE_LINE + 1)
            if not raw:
                break
            if len(raw) > _MAX_SSE_LINE:
                raise ConnectorError(
                    ProviderCode.PROVIDER_UNAVAILABLE,
                    message="SSE line exceeds the 16 MiB cap",
                    provider_message="SSE line exceeds the 16 MiB cap",
                )
            yield raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")

    def read_all(self) -> bytes:
        return self._resp.read()

    def close(self) -> None:
        try:
            self._resp.close()
        except Exception:
            pass


# Cloudflare-fronted vendor APIs (Groq, among others) answer urllib's
# default `Python-urllib/3.x` User-Agent with 403 "error code: 1010", so the
# default transport names itself. A User-Agent already on the request wins.
_USER_AGENT = "thinwrap-llm-python"


def _with_user_agent(headers: Mapping[str, str]) -> dict:
    out = dict(headers)
    if not any(k.lower() == "user-agent" for k in out):
        out["User-Agent"] = _USER_AGENT
    return out


class UrllibTransport:
    """Default synchronous transport built on the standard library."""

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._timeout = timeout

    def send(self, request: HttpRequest) -> HttpResponse:
        _assert_http_scheme(request.url)
        req = urllib.request.Request(request.url, data=request.body, method=request.method, headers=_with_user_agent(request.headers))
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                return HttpResponse(status=resp.status, headers=_lower_headers(resp.headers), body=resp.read())
        except urllib.error.HTTPError as exc:
            return HttpResponse(status=exc.code, headers=_lower_headers(exc.headers), body=exc.read())

    def open_stream(self, request: HttpRequest) -> StreamResponse:
        _assert_http_scheme(request.url)
        req = urllib.request.Request(request.url, data=request.body, method=request.method, headers=_with_user_agent(request.headers))
        try:
            return _UrllibStream(self._opener.open(req, timeout=self._timeout))
        except urllib.error.HTTPError as exc:
            return _ErrorStream(exc.code, _lower_headers(exc.headers), exc.read())


class _ErrorStream(StreamResponse):
    """A non-2xx stream whose body was already read (from urllib's HTTPError)."""

    def __init__(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self._body = body

    def iter_lines(self) -> Iterator[str]:
        for line in _split_sse_lines(self._body.decode("utf-8", "replace")):
            yield line

    def read_all(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


class BufferedStream(StreamResponse):
    """Wraps a buffered HttpResponse as a StreamResponse — the fallback when a
    BYO transport implements only ``send``."""

    def __init__(self, resp: HttpResponse) -> None:
        self.status = resp.status
        self.headers = resp.headers
        self._body = resp.body

    def iter_lines(self) -> Iterator[str]:
        for line in _split_sse_lines(self._body.decode("utf-8", "replace")):
            yield line

    def read_all(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


_default_transport: UrllibTransport | None = None


def default_transport() -> Transport:
    global _default_transport
    if _default_transport is None:
        _default_transport = UrllibTransport()
    return _default_transport
