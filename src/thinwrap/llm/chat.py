"""Chat completions: normalized types + the Chat facade (complete + stream)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Mapping, Optional

from .enums import ProviderID
from .errors import invalid_request
from .passthrough import Passthrough
from .transport import Transport

# --------------------------------------------------------------------------- #
# Normalized types (identical surface across every provider)
# --------------------------------------------------------------------------- #


@dataclass
class ContentPart:
    """One part of a multimodal message. type is "text" or "image"."""

    type: str
    text: str = ""
    base64: str = ""
    media_type: str = ""


def text_part(text: str) -> ContentPart:
    return ContentPart(type="text", text=text)


def image_part(base64: str, media_type: str) -> ContentPart:
    return ContentPart(type="image", base64=base64, media_type=media_type)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON-encoded arguments string


@dataclass
class ChatMessage:
    """One turn. Use content for plain text, or content_parts for multimodal
    (content_parts wins when non-empty)."""

    role: str
    content: str = ""
    content_parts: List[ContentPart] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""


@dataclass
class ToolDefinition:
    name: str
    description: str = ""
    parameters: Optional[Mapping[str, Any]] = None


@dataclass
class ToolChoice:
    """mode is "auto", "none", "required", or "function" (with function_name)."""

    mode: str
    function_name: str = ""


@dataclass
class ResponseFormat:
    """type is "text", "json_object", or "json_schema" (with json_schema_name + json_schema)."""

    type: str
    json_schema_name: str = ""
    json_schema: Optional[Mapping[str, Any]] = None


@dataclass
class ReasoningOptions:
    effort: str  # "low" | "medium" | "high"


@dataclass
class ChatInput:
    model: str
    messages: List[ChatMessage]
    tools: List[ToolDefinition] = field(default_factory=list)
    tool_choice: Optional[ToolChoice] = None
    response_format: Optional[ResponseFormat] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    reasoning: Optional[ReasoningOptions] = None
    passthrough: Optional[Passthrough] = None


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class AssistantMessage:
    role: str
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class ChatResult:
    message: AssistantMessage
    finish_reason: str
    usage: Usage
    model: str
    raw: Any


@dataclass
class ToolCallDelta:
    index: int
    id: str = ""
    function_name: str = ""
    arguments_delta: str = ""


@dataclass
class ChatStreamDelta:
    content_delta: str = ""
    tool_call_delta: Optional[ToolCallDelta] = None
    finish_reason: str = ""
    usage: Optional[Usage] = None
    raw: Any = None


# --------------------------------------------------------------------------- #
# Facade + dispatch
# --------------------------------------------------------------------------- #

_NATIVE = {
    ProviderID.ANTHROPIC: ("_anthropic", "AnthropicConnector"),
    ProviderID.BEDROCK: ("_bedrock", "BedrockConnector"),
    ProviderID.GEMINI: ("_gemini", "GeminiConnector"),
}


class Chat:
    """Chat-completion facade. Construct by provider id + config; the input and
    output shapes are identical across every provider. Switching vendor is a
    change of id + model only."""

    def __init__(self, provider_id, config, *, transport: Optional[Transport] = None) -> None:
        try:
            self.provider_id = ProviderID(provider_id)
        except ValueError:
            raise invalid_request(f"Unsupported LLM provider: {provider_id}") from None
        entry = _NATIVE.get(self.provider_id)
        if entry is not None:
            module = importlib.import_module(f".{entry[0]}", __package__)
            self._c = getattr(module, entry[1])(config, transport)
            return
        from .spec import SPECS

        spec = SPECS.get(self.provider_id)
        if spec is None:
            raise invalid_request(f"Unsupported LLM provider: {self.provider_id.value}")
        from ._openai_compat import OpenAICompatConnector

        self._c = OpenAICompatConnector(spec, config, transport)

    def complete(self, message: ChatInput) -> ChatResult:
        return self._c.complete(message)

    def stream(self, message: ChatInput) -> Iterator[ChatStreamDelta]:
        return self._c.stream(message)
