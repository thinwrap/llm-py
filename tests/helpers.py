from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

from thinwrap.llm.transport import BufferedStream, HttpRequest, HttpResponse


class FakeTransport:
    """Test transport implementing both send and open_stream: returns queued
    responses in order (repeating the last) and records every request."""

    def __init__(self, *responses: HttpResponse) -> None:
        self.queue = list(responses)
        self.calls: list[HttpRequest] = []

    def _next(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        idx = len(self.calls) - 1
        if idx < len(self.queue):
            return self.queue[idx]
        if self.queue:
            return self.queue[-1]
        return HttpResponse(status=200, headers={}, body=b"")

    def send(self, request: HttpRequest) -> HttpResponse:
        return self._next(request)

    def open_stream(self, request: HttpRequest):
        return BufferedStream(self._next(request))

    @property
    def last(self) -> HttpRequest:
        return self.calls[-1]


def resp(status: int, body: str = "", **headers: str) -> HttpResponse:
    hdrs = {k.replace("_", "-").lower(): v for k, v in headers.items()}
    return HttpResponse(status=status, headers=hdrs, body=body.encode("utf-8"))


def body_json(req: HttpRequest):
    return json.loads(req.body.decode("utf-8"))


def path_of(req: HttpRequest) -> str:
    return urlsplit(req.url).path


def qget(req: HttpRequest, key: str) -> str:
    return parse_qs(urlsplit(req.url).query).get(key, [""])[0]
