"""Embeddings: normalized types + the Embeddings facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .config import OpenAICompatConfig
from .enums import ProviderID
from .errors import invalid_request
from .passthrough import Passthrough
from .transport import Transport


@dataclass
class EmbeddingsInput:
    model: str
    input: List[str]
    dimensions: Optional[int] = None
    passthrough: Optional[Passthrough] = None


@dataclass
class EmbeddingsUsage:
    input_tokens: int
    total_tokens: int


@dataclass
class EmbeddingsResult:
    embeddings: List[List[float]]  # one vector per input, in input order
    usage: EmbeddingsUsage
    model: str
    raw: Any


class Embeddings:
    """Embeddings facade — a separate operation from Chat. Only the OpenAI-float
    subset of providers is supported."""

    def __init__(self, provider_id, config: OpenAICompatConfig, *, transport: Optional[Transport] = None) -> None:
        try:
            self.provider_id = ProviderID(provider_id)
        except ValueError:
            raise invalid_request(f"Unsupported LLM provider: {provider_id}") from None
        from .spec import EMBEDDINGS_PROVIDER_IDS, SPECS

        if self.provider_id not in EMBEDDINGS_PROVIDER_IDS:
            raise invalid_request(f"Provider '{self.provider_id.value}' has no OpenAI-compatible embeddings surface")
        from ._openai_compat_embeddings import OpenAICompatEmbeddingsConnector

        self._c = OpenAICompatEmbeddingsConnector(SPECS[self.provider_id], config, transport)

    def create(self, message: EmbeddingsInput) -> EmbeddingsResult:
        return self._c.create(message)
