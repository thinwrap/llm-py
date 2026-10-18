"""The shared embeddings connector for OpenAI-float-shaped providers."""

from __future__ import annotations

import math
from typing import Any, List

from ._jsonpath import extract_error_message, jget, jlist, jnum, jstr
from ._util import decode_json, ok_status
from .base import BaseConnector
from .embeddings import EmbeddingsInput, EmbeddingsResult, EmbeddingsUsage
from .errors import (
    ProviderCode,
    embedded_error_class,
    empty_body_error,
    invalid_request,
    malformed_body_error,
    vendor_error,
)
from .passthrough import apply_body, merge_headers
from .spec import OpenAICompatSpec


class OpenAICompatEmbeddingsConnector(BaseConnector):
    def __init__(self, spec: OpenAICompatSpec, config, transport=None) -> None:
        super().__init__(transport)
        from .config import OpenAICompatConfig

        if not isinstance(config, OpenAICompatConfig):
            raise invalid_request(f"Provider '{spec.id.value}' requires an OpenAICompatConfig")
        self.spec = spec
        self.cfg = config
        base = config.base_url or spec.default_base_url
        if not base:
            raise invalid_request(f"Provider '{spec.id.value}' requires an explicit base_url in config")
        self.base_url = base.rstrip("/")

    def create(self, msg: EmbeddingsInput) -> EmbeddingsResult:
        body: dict[str, Any] = {"model": msg.model, "input": list(msg.input), "encoding_format": "float"}
        if msg.dimensions is not None:
            body[self.spec.dimensions_field] = msg.dimensions
        apply_body(body, msg.passthrough)

        headers: dict[str, str] = {}
        headers.update(self.spec.build_auth_headers(self.cfg))
        if self.cfg.headers:
            headers.update(self.cfg.headers)
        headers = merge_headers(headers, msg.passthrough)

        url = self.base_url + "/embeddings"
        # Honor passthrough.query on the embeddings path too (e.g. Azure OpenAI's
        # required api-version), mirroring the chat connector.
        if msg.passthrough and msg.passthrough.query:
            from urllib.parse import urlencode

            url += "?" + urlencode(dict(msg.passthrough.query))
        resp = self.send_post_json(url, body, headers)
        raw = decode_json(resp.body)
        if not ok_status(resp.status):
            raise vendor_error(resp.status, resp.headers, raw, _map_code(resp.status), extract_error_message(raw))
        if raw is None:
            raise empty_body_error(resp.status)
        err = jget(raw, "error")
        if err is not None:
            # An error object on a 2xx (OpenRouter-style) keeps the 2xx it arrived
            # with as its status; its `error.code` (a number, or a name such as
            # insufficient_quota) classifies it the way chat does, and without one
            # it says nothing classifiable.
            effective = embedded_error_class(err)
            msg_text = extract_error_message({"error": err})
            code = ProviderCode.UNKNOWN if effective is None else _map_code(effective)
            raise vendor_error(resp.status, resp.headers, raw, code, msg_text)
        data = jlist(jget(raw, "data"))
        if data is None:
            raise empty_body_error(resp.status)
        # The result promises one vector per input in input order; a short or
        # padded `data` cannot be re-aligned to the inputs.
        if len(data) != len(msg.input):
            raise malformed_body_error(
                resp.status, raw, f"provider returned {len(data)} embeddings for {len(msg.input)} inputs"
            )

        rows = list(data)
        rows.sort(key=lambda r: jnum(jget(r, "index")))
        embeddings = [_vector(r, resp.status, raw) for r in rows]

        usage = jget(raw, "usage")
        in_tok = int(jnum(jget(usage, "prompt_tokens")))
        total = int(jnum(jget(usage, "total_tokens"))) or in_tok
        return EmbeddingsResult(
            embeddings=embeddings,
            usage=EmbeddingsUsage(input_tokens=in_tok, total_tokens=total),
            model=jstr(jget(raw, "model")) or msg.model,
            raw=raw,
        )


def _vector(row: Any, status: int, raw: Any) -> List[float]:
    # A null/string cell is not a coordinate; coercing it to 0.0 would return a
    # plausible-looking but wrong vector.
    cells = jlist(jget(row, "embedding"))
    if cells is None:
        raise malformed_body_error(status, raw, "provider returned an embedding row without a vector")
    out: List[float] = []
    for c in cells:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
            raise malformed_body_error(status, raw, "provider returned a non-numeric embedding value")
        out.append(float(c))
    return out


def _map_code(status: int) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    # 402 is out of credit/quota, which OpenAI reports as 429 insufficient_quota.
    if status in (402, 429):
        return ProviderCode.RATE_LIMITED
    # 408 is a timeout on the provider's side; a retry can succeed.
    if status >= 500 or status == 408:
        return ProviderCode.PROVIDER_UNAVAILABLE
    # 400, 404, 413, 422, …: the request cannot succeed as sent.
    if status >= 400:
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN
