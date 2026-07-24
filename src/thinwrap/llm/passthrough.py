"""The per-request escape hatch for sub-baseline / provider-specific fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Passthrough:
    """Body is shallow-merged onto (and overrides) the connector-built request
    body; headers are merged onto request headers; query is appended to the URL.
    Never emulated — forwarded verbatim."""

    body: Optional[Mapping[str, Any]] = None
    headers: Optional[Mapping[str, str]] = None
    query: Optional[Mapping[str, str]] = None


def apply_body(body: dict, pt: Optional[Passthrough]) -> None:
    if pt and pt.body:
        for k, v in pt.body.items():
            body[k] = v


def merge_headers(base: Mapping[str, str], pt: Optional[Passthrough]) -> dict[str, str]:
    out = dict(base)
    if pt and pt.headers:
        out.update(pt.headers)
    return out
