"""The typed error surface: ProviderCode + ConnectorError."""

from __future__ import annotations

import datetime as _dt
import re
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


def malformed_body_error(status: int, raw_body: Any, msg: str) -> ConnectorError:
    """Raised when a 2xx JSON body decodes but cannot be a real result (e.g. one
    embedding vector per input is missing, or a vector cell is not a number).
    Coercing it would hand the caller silently wrong data."""
    return ConnectorError(
        ProviderCode.PROVIDER_UNAVAILABLE,
        status_code=status,
        message=msg,
        provider_message=msg,
        cause={"raw": raw_body},
    )


# HTTP status for the OpenAI-style error `code` / `type` names an embedded error
# can carry instead of a number (`insufficient_quota` is OpenAI's out-of-credit
# code, which it also sends as HTTP 429).
_ERROR_NAME_STATUS = {
    "invalid_request_error": 400,
    "context_length_exceeded": 400,
    "authentication_error": 401,
    "invalid_api_key": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "rate_limit_error": 429,
    "rate_limit_exceeded": 429,
    "insufficient_quota": 429,
    "server_error": 500,
    "api_error": 500,
    "overloaded_error": 529,
}


def embedded_error_status(err: Any) -> int | None:
    """The HTTP status an error object delivered inside a 2xx body or a stream
    frame carries: OpenRouter's post-processing errors put the real status in
    ``error.code`` (e.g. 502, as a number or a numeric string). Anything else is
    not a status."""
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if isinstance(code, str) and len(code) == 3 and code.isdigit():
        code = int(code)
    if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
        return code
    return None


def embedded_error_class(err: Any) -> int | None:
    """The status to classify an embedded error object by: its HTTP status when
    it carries one, else the status its ``code`` / ``type`` name stands for
    (``insufficient_quota`` → 429). ``None`` when it names nothing known."""
    status = embedded_error_status(err)
    if status is not None:
        return status
    if isinstance(err, dict):
        for name in (err.get("code"), err.get("type")):
            if isinstance(name, str) and name in _ERROR_NAME_STATUS:
                return _ERROR_NAME_STATUS[name]
    return None


_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
# The three HTTP-date forms RFC 9110 §5.6.7 allows, matched strictly so that
# free text ("tomorrow") never reads as a date.
_IMF_FIXDATE = re.compile(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) ([A-Za-z]{3}) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT")
_RFC850_DATE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), (\d{2})-([A-Za-z]{3})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) GMT"
)
_ASCTIME_DATE = re.compile(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) ([A-Za-z]{3}) ([ \d]\d) (\d{2}):(\d{2}):(\d{2}) (\d{4})")


def _parse_http_date(s: str) -> _dt.datetime | None:
    m = _IMF_FIXDATE.fullmatch(s)
    if m:
        day, mon, year, hh, mm, ss = m.groups()
    else:
        m = _RFC850_DATE.fullmatch(s)
        if m:
            day, mon, yy, hh, mm, ss = m.groups()
            # RFC 9110: a two-digit year more than 50 years ahead is in the past.
            now_year = _dt.datetime.now(tz=_dt.timezone.utc).year
            year = str(now_year // 100 * 100 + int(yy))
            if int(year) > now_year + 50:
                year = str(int(year) - 100)
        else:
            m = _ASCTIME_DATE.fullmatch(s)
            if not m:
                return None
            mon, day, hh, mm, ss, year = m.groups()
    month = _MONTHS.get(mon.lower())
    if month is None:
        return None
    try:
        return _dt.datetime(int(year), month, int(day.strip()), int(hh), int(mm), int(ss), tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def parse_retry_after(value: str) -> int | None:
    """Seconds to wait per RFC 9110: delay-seconds is a non-negative integer, so
    "1.5" or "-5" is ignored; an HTTP-date gives the time until it, and a date
    already past means "retry now" (0). Anything else is ignored."""
    s = (value or "").strip()
    if not s:
        return None
    if s.isascii() and s.isdigit():
        return int(s)
    when = _parse_http_date(s)
    if when is None:
        return None
    return max(int((when - _dt.datetime.now(tz=_dt.timezone.utc)).total_seconds()), 0)


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
    retry_after = (headers.get("retry-after", "") if headers else "").strip()
    if retry_after:
        cause["retryAfter"] = retry_after
        secs = parse_retry_after(retry_after)
        if secs is not None:
            cause["retryAfterSeconds"] = secs
    msg = provider_message or f"HTTP {status}"
    return ConnectorError(code, status_code=status, provider_message=msg, message=msg, cause=cause)
