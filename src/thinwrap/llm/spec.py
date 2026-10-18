"""Per-provider parameterization of the one shared OpenAI-compatible connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import OpenAICompatConfig
from .enums import ProviderID


@dataclass(frozen=True)
class OpenAICompatSpec:
    id: ProviderID
    default_base_url: str
    build_auth_headers: Callable[[OpenAICompatConfig], dict]
    supports_tools: bool = True
    max_tokens_field: str = "max_tokens"
    stream_usage_path: str = ""  # "x_groq" relocates streaming usage
    strict_params: bool = False
    dimensions_field: str = "dimensions"  # embeddings output-size field


def _bearer(cfg: OpenAICompatConfig) -> dict:
    return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}


def _azure(cfg: OpenAICompatConfig) -> dict:
    return {"api-key": cfg.api_key} if cfg.api_key else {}


SPECS: dict = {
    # OpenAI deprecates `max_tokens` in favor of `max_completion_tokens` (the
    # only one o-series/reasoning models accept); Azure's reasoning models on the
    # v1 API likewise reject `max_tokens`.
    ProviderID.OPENAI: OpenAICompatSpec(
        ProviderID.OPENAI, "https://api.openai.com/v1", _bearer, max_tokens_field="max_completion_tokens"
    ),
    ProviderID.AZURE_OPENAI: OpenAICompatSpec(
        ProviderID.AZURE_OPENAI, "", _azure, max_tokens_field="max_completion_tokens"
    ),
    ProviderID.OPENROUTER: OpenAICompatSpec(ProviderID.OPENROUTER, "https://openrouter.ai/api/v1", _bearer),
    ProviderID.GROQ: OpenAICompatSpec(ProviderID.GROQ, "https://api.groq.com/openai/v1", _bearer, stream_usage_path="x_groq"),
    ProviderID.TOGETHER: OpenAICompatSpec(ProviderID.TOGETHER, "https://api.together.ai/v1", _bearer),
    ProviderID.FIREWORKS: OpenAICompatSpec(ProviderID.FIREWORKS, "https://api.fireworks.ai/inference/v1", _bearer),
    ProviderID.DEEPSEEK: OpenAICompatSpec(ProviderID.DEEPSEEK, "https://api.deepseek.com/v1", _bearer),
    ProviderID.XAI: OpenAICompatSpec(ProviderID.XAI, "https://api.x.ai/v1", _bearer),
    # Mistral's embeddings API has no `dimensions`; its output size is
    # `output_dimension`.
    ProviderID.MISTRAL: OpenAICompatSpec(
        ProviderID.MISTRAL, "https://api.mistral.ai/v1", _bearer, strict_params=True, dimensions_field="output_dimension"
    ),
    ProviderID.PERPLEXITY: OpenAICompatSpec(ProviderID.PERPLEXITY, "https://api.perplexity.ai", _bearer, supports_tools=False),
    ProviderID.DEEPINFRA: OpenAICompatSpec(ProviderID.DEEPINFRA, "https://api.deepinfra.com/v1/openai", _bearer),
    ProviderID.CLOUDFLARE: OpenAICompatSpec(ProviderID.CLOUDFLARE, "", _bearer),
    ProviderID.VLLM: OpenAICompatSpec(ProviderID.VLLM, "http://localhost:8000/v1", _bearer),
    ProviderID.OLLAMA: OpenAICompatSpec(ProviderID.OLLAMA, "http://localhost:11434/v1", _bearer),
    ProviderID.LMSTUDIO: OpenAICompatSpec(ProviderID.LMSTUDIO, "http://localhost:1234/v1", _bearer),
}

# First-class providers exposing an OpenAI-float /embeddings surface.
EMBEDDINGS_PROVIDER_IDS = {
    ProviderID.OPENAI, ProviderID.AZURE_OPENAI, ProviderID.OPENROUTER, ProviderID.TOGETHER,
    ProviderID.FIREWORKS, ProviderID.MISTRAL, ProviderID.DEEPINFRA, ProviderID.CLOUDFLARE,
    ProviderID.VLLM, ProviderID.OLLAMA, ProviderID.LMSTUDIO,
}
