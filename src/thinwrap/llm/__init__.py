"""thinwrap.llm — unified, SDK-free, zero-dependency Python wrapper for chat
completions (with streaming) and embeddings across 18 LLM providers behind one
normalized surface.

Switching vendor is a change of provider id + model only::

    from thinwrap.llm import Chat, ChatInput, ChatMessage, OpenAICompatConfig, ProviderID

    chat = Chat(ProviderID.OPENAI, OpenAICompatConfig(api_key="..."))
    res = chat.complete(ChatInput(model="gpt-4o-mini",
                                  messages=[ChatMessage(role="user", content="Hi")]))
    print(res.message.content, res.usage.total_tokens)
"""

from __future__ import annotations

from .chat import (
    AssistantMessage,
    Chat,
    ChatInput,
    ChatMessage,
    ChatResult,
    ChatStreamDelta,
    ContentPart,
    ReasoningOptions,
    ResponseFormat,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolDefinition,
    Usage,
    image_part,
    text_part,
)
from .config import AnthropicConfig, BedrockConfig, GeminiConfig, OpenAICompatConfig
from .embeddings import Embeddings, EmbeddingsInput, EmbeddingsResult, EmbeddingsUsage
from .enums import ProviderID
from .errors import ConnectorError, ProviderCode
from .passthrough import Passthrough
from .transport import HttpRequest, HttpResponse, Transport, UrllibTransport

__all__ = [
    "Chat", "Embeddings",
    "ChatInput", "ChatMessage", "ChatResult", "ChatStreamDelta", "ContentPart",
    "ToolCall", "ToolCallDelta", "ToolDefinition", "ToolChoice", "ResponseFormat",
    "ReasoningOptions", "Usage", "AssistantMessage", "text_part", "image_part",
    "EmbeddingsInput", "EmbeddingsResult", "EmbeddingsUsage",
    "OpenAICompatConfig", "AnthropicConfig", "BedrockConfig", "GeminiConfig",
    "ProviderID", "ConnectorError", "ProviderCode", "Passthrough",
    "Transport", "UrllibTransport", "HttpRequest", "HttpResponse",
]
