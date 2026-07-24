"""User-facing typed vocabulary. Values are byte-identical across siblings."""

from __future__ import annotations

from enum import Enum


class ProviderID(str, Enum):
    # First-class OpenAI-compatible providers (one shared connector).
    OPENAI = "openai"
    AZURE_OPENAI = "azure-openai"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"
    DEEPINFRA = "deepinfra"
    CLOUDFLARE = "cloudflare"
    VLLM = "vllm"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    # Native adapters.
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    GEMINI = "gemini"
