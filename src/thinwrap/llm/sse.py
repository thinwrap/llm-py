"""Server-Sent-Events parsing over an iterator of (newline-stripped) text lines."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional

from .errors import ConnectorError, ProviderCode
from .transport import _MAX_SSE_LINE


def parse_sse_lines(lines: Iterable[str], state: Optional[dict] = None) -> Iterator[str]:
    """Yield the JSON payload of each SSE event's data: line(s). Skips
    comments/blank lines, stops at the [DONE] sentinel, joins multi-line data.
    When ``state`` is given, ``state["saw_done"]`` records whether [DONE] arrived,
    so a caller can tell a complete stream from one cut off before it."""
    data_parts: list[str] = []
    event_size = 0  # cumulative data: bytes buffered for the current event
    for line in lines:
        if line == "":
            if data_parts:
                payload = "\n".join(data_parts)
                data_parts = []
                event_size = 0
                if payload == "[DONE]":
                    if state is not None:
                        state["saw_done"] = True
                    return
                yield payload
            continue
        if line.startswith("data:"):
            # Per the SSE spec, strip ONLY one optional leading space after the
            # colon — not all leading whitespace (which is significant payload).
            seg = line[5:]
            if seg.startswith(" "):
                seg = seg[1:]
            # The transport caps a SINGLE line at 16 MiB, but an adversarial
            # upstream could stream many data: lines with no event boundary and
            # grow this buffer without limit. Cap the cumulative event too.
            event_size += len(seg) + 1
            if event_size > _MAX_SSE_LINE:
                raise ConnectorError(
                    ProviderCode.PROVIDER_UNAVAILABLE,
                    message="SSE event exceeds the 16 MiB cap",
                    provider_message="SSE event exceeds the 16 MiB cap",
                )
            data_parts.append(seg)
        # comments (":" prefix) and other fields ignored
    if data_parts:
        payload = "\n".join(data_parts)
        if payload != "[DONE]":
            yield payload
        elif state is not None:
            state["saw_done"] = True
