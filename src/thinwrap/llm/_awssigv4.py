"""Hand-rolled AWS Signature Version 4 (HMAC-SHA256) on the standard library —
no third-party dependency. Used by the Bedrock native adapter."""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional
from urllib.parse import quote

_UNSIGNABLE = {
    "authorization", "cache-control", "connection", "content-length", "expect",
    "from", "keep-alive", "max-forwards", "pragma", "proxy-authenticate",
    "proxy-authorization", "referer", "te", "trailer", "transfer-encoding",
    "upgrade", "user-agent", "x-amzn-trace-id",
}
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SigV4Params:
    method: str
    service: str
    region: str
    host: str
    path: str
    payload: bytes
    access_key_id: str
    secret_access_key: str
    session_token: str = ""
    canonical_query: str = ""
    signed: Mapping[str, str] = field(default_factory=dict)
    now: Optional[_dt.datetime] = None


def sign_aws_v4(p: SigV4Params) -> dict[str, str]:
    ts = (p.now or _dt.datetime.now(tz=_dt.timezone.utc)).astimezone(_dt.timezone.utc)
    amz_date = ts.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = ts.strftime("%Y%m%d")
    hashed_payload = hashlib.sha256(p.payload).hexdigest()

    canonical: dict[str, str] = {}
    for k, v in p.signed.items():
        canonical[k.lower()] = _WS.sub(" ", v.strip())
    canonical["host"] = _WS.sub(" ", p.host.strip())
    canonical["x-amz-date"] = amz_date
    canonical["x-amz-content-sha256"] = hashed_payload
    if p.session_token:
        canonical["x-amz-security-token"] = p.session_token
    for h in list(canonical):
        if h in _UNSIGNABLE:
            del canonical[h]

    names = sorted(canonical)
    canonical_headers = "".join(f"{n}:{canonical[n]}\n" for n in names)
    signed_headers = ";".join(names)
    canonical_request = "\n".join([p.method, p.path, p.canonical_query, canonical_headers, signed_headers, hashed_payload])
    credential_scope = f"{date_stamp}/{p.region}/{p.service}/aws4_request"
    string_to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, credential_scope, hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()])

    def _hmac(key: bytes, data: str) -> bytes:
        return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()

    k_date = _hmac(("AWS4" + p.secret_access_key).encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, p.region)
    k_service = _hmac(k_region, p.service)
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={p.access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    out = {
        "Host": p.host,
        "X-Amz-Date": amz_date,
        "X-Amz-Content-Sha256": hashed_payload,
        "Authorization": authorization,
    }
    if p.session_token:
        out["X-Amz-Security-Token"] = p.session_token
    return out


def build_canonical_query(query: Mapping[str, str]) -> str:
    if not query:
        return ""
    return "&".join(f"{_rfc3986(k)}={_rfc3986(query[k])}" for k in sorted(query))


def _rfc3986(s: str) -> str:
    return quote(s, safe="-._~")
