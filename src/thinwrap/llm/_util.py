"""Small shared helpers: JSON encode/decode, status checks."""

from __future__ import annotations

import json
from typing import Any


def compact_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode_json(data: bytes | None) -> Any:
    if not data:
        return None
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        return None


def ok_status(status: int) -> bool:
    return 200 <= status < 300


def first_non_empty(*vals: str) -> str:
    for v in vals:
        if v:
            return v
    return ""
