"""Nil/absence-tolerant helpers for navigating decoded JSON on the error/raw paths."""

from __future__ import annotations

from typing import Any, List, Optional


def jget(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else None


def jstr(v: Any) -> str:
    return v if isinstance(v, str) else ""


def jnum(v: Any) -> float:
    if isinstance(v, bool):
        return 0.0
    return float(v) if isinstance(v, (int, float)) else 0.0


def jlist(v: Any) -> Optional[List[Any]]:
    return v if isinstance(v, list) else None


def extract_error_message(body: Any) -> str:
    """Pull a human message out of a decoded vendor error body across the common
    shapes (string, {error:string}, {error:{message}}, {message}, {errors:[{message}]})."""
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(body.get("message"), str):
        return body["message"]
    errs = body.get("errors")
    if isinstance(errs, list) and errs and isinstance(errs[0], dict) and isinstance(errs[0].get("message"), str):
        return errs[0]["message"]
    return ""
