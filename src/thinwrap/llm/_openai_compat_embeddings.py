"""The shared embeddings connector for OpenAI-float-shaped providers."""

from __future__ import annotations

from typing import Any

from ._jsonpath import extract_error_message, jget, jlist, jnum, jstr
from ._util import decode_json, ok_status
from .base import BaseConnector
from .embeddings import EmbeddingsInput, EmbeddingsResult, EmbeddingsUsage
from .errors import ProviderCode, empty_body_error, invalid_request, vendor_error
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
            body["dimensions"] = msg.dimensions
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

        rows = list(jlist(jget(raw, "data")) or [])
        rows.sort(key=lambda r: jnum(jget(r, "index")))
        embeddings = [[jnum(f) for f in (jlist(jget(r, "embedding")) or [])] for r in rows]

        usage = jget(raw, "usage")
        in_tok = int(jnum(jget(usage, "prompt_tokens")))
        total = int(jnum(jget(usage, "total_tokens"))) or in_tok
        return EmbeddingsResult(
            embeddings=embeddings,
            usage=EmbeddingsUsage(input_tokens=in_tok, total_tokens=total),
            model=jstr(jget(raw, "model")) or msg.model,
            raw=raw,
        )


def _map_code(status: int) -> ProviderCode:
    if status in (401, 403):
        return ProviderCode.AUTH_FAILED
    if status == 429:
        return ProviderCode.RATE_LIMITED
    if status >= 500:
        return ProviderCode.PROVIDER_UNAVAILABLE
    if status in (400, 404, 422):
        return ProviderCode.INVALID_REQUEST
    return ProviderCode.UNKNOWN
