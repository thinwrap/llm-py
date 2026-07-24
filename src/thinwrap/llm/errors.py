"""The typed error surface: ProviderCode + ConnectorError."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ProviderCode(str, Enum):
    """Normalized, cross-provider error classification. Byte-identical across the
    thinwrap llm siblings (TypeScript, PHP, Go, Python)."""

    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    CONTENT_FILTERED = "content_filtered"
    UNKNOWN = "unknown"


class ConnectorError(Exception):
    """The single typed error every operation can raise. For a vendor HTTP error
    the raw Retry-After header rides in ``cause['retryAfter']`` and its parsed
    seconds in ``cause['retryAfterSeconds']`` — there is no ``retry_after_seconds``
    attribute."""

    def __init__(
        self,
        provider_code: ProviderCode,
        *,
        status_code: int | None = None,
        provider_message: str | None = None,
        message: str | None = None,
        cause: Any = None,
    ) -> None:
        self.provider_code = provider_code
        self.status_code = status_code
        self.provider_message = provider_message
        self.cause = cause
        super().__init__(message or provider_message or "Connector error")


def invalid_request(msg: str) -> ConnectorError:
    return ConnectorError(ProviderCode.INVALID_REQUEST, message=msg, provider_message=msg)


def empty_body_error(status: int) -> ConnectorError:
    """Raised when a 2xx response carries no decodable JSON body (a proxy/
    captive-portal HTML page, or a truncated/aborted read). Without this guard
    the connector would parse ``None`` into an empty "successful" result."""
    msg = "provider returned a successful status with an empty or non-JSON body"
    return ConnectorError(ProviderCode.PROVIDER_UNAVAILABLE, status_code=status, message=msg, provider_message=msg)


def parse_retry_after(value: str) -> int | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        secs = int(s)
        return secs if secs >= 0 else None
    except ValueError:
        pass
    try:
        import datetime as _dt
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(s)
        if when is None:
            return None
        return max(int((when - _dt.datetime.now(tz=when.tzinfo)).total_seconds()), 0)
    except (TypeError, ValueError):
        return None


def vendor_error(
    status: int,
    headers: Mapping[str, str],
    raw_body: Any,
    code: ProviderCode,
    provider_message: str | None,
) -> ConnectorError:
    """Build a ConnectorError for a non-2xx response; the shared Retry-After
    surfacing lives here (cause['retryAfter'] + cause['retryAfterSeconds'])."""
    cause: dict[str, Any] = {"raw": raw_body}
    retry_after = headers.get("retry-after", "") if headers else ""
    if retry_after:
        cause["retryAfter"] = retry_after
        secs = parse_retry_after(retry_after)
        if secs is not None:
            cause["retryAfterSeconds"] = secs
    msg = provider_message or f"HTTP {status}"
    return ConnectorError(code, status_code=status, provider_message=msg, message=msg, cause=cause)
