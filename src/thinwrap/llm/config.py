"""Provider configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class OpenAICompatConfig:
    """Configures any of the 15 first-class OpenAI-compatible providers (the
    provider id selects which). api_key is optional for self-host providers
    (vllm / ollama / lmstudio); base_url is required for providers with no fixed
    public default (azure-openai, cloudflare)."""

    # repr=False keeps credentials out of repr()/str() (and thus logs/tracebacks).
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    headers: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class AnthropicConfig:
    """Anthropic Messages API native adapter. api_key is sent as x-api-key."""

    api_key: str = field(repr=False)
    base_url: str = ""
    anthropic_version: str = ""
    default_max_tokens: int = 0
    headers: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class BedrockConfig:
    """AWS Bedrock Converse native adapter (SigV4)."""

    region: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    session_token: str = field(default="", repr=False)
    base_url: str = ""
    default_max_tokens: int = 0
    headers: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class GeminiConfig:
    """Google Gemini generateContent native adapter. api_key is sent as
    x-goog-api-key."""

    api_key: str = field(repr=False)
    base_url: str = ""
    default_max_tokens: int = 0
    headers: Optional[Mapping[str, str]] = None
