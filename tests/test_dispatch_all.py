"""Dispatch guards: every chat provider builds; the embeddings subset is
enforced; a buffered-only transport still streams; native config-type mismatch
raises."""

from __future__ import annotations

import pytest
from helpers import resp

import thinwrap.llm as llm
from thinwrap.llm.transport import HttpRequest, HttpResponse


def _cfg(pid: llm.ProviderID):
    if pid is llm.ProviderID.ANTHROPIC:
        return llm.AnthropicConfig(api_key="k")
    if pid is llm.ProviderID.BEDROCK:
        return llm.BedrockConfig(region="us-east-1", access_key_id="a", secret_access_key="s")
    if pid is llm.ProviderID.GEMINI:
        return llm.GeminiConfig(api_key="k")
    # compat: cloudflare/azure need a base_url
    base = "https://x/v1" if pid in (llm.ProviderID.AZURE_OPENAI, llm.ProviderID.CLOUDFLARE) else ""
    return llm.OpenAICompatConfig(api_key="k", base_url=base)


def test_all_chat_providers_build():
    for pid in llm.ProviderID:
        chat = llm.Chat(pid, _cfg(pid))
        assert chat.provider_id is pid


def test_embeddings_subset():
    from thinwrap.llm.spec import EMBEDDINGS_PROVIDER_IDS

    for pid in llm.ProviderID:
        if pid in EMBEDDINGS_PROVIDER_IDS:
            base = "https://x/v1" if pid in (llm.ProviderID.AZURE_OPENAI, llm.ProviderID.CLOUDFLARE) else ""
            emb = llm.Embeddings(pid, llm.OpenAICompatConfig(api_key="k", base_url=base))
            assert emb.provider_id is pid
        elif pid not in (llm.ProviderID.ANTHROPIC, llm.ProviderID.BEDROCK, llm.ProviderID.GEMINI):
            with pytest.raises(llm.ConnectorError) as ei:
                llm.Embeddings(pid, llm.OpenAICompatConfig(api_key="k"))
            assert ei.value.provider_code is llm.ProviderCode.INVALID_REQUEST


def test_native_wrong_config_type():
    with pytest.raises(llm.ConnectorError):
        llm.Chat(llm.ProviderID.ANTHROPIC, llm.OpenAICompatConfig(api_key="k"))
    with pytest.raises(llm.ConnectorError):
        llm.Chat(llm.ProviderID.OPENAI, llm.AnthropicConfig(api_key="k"))


class _BufferedOnlyTransport:
    """A BYO transport implementing only send() — exercises the buffered stream fallback."""

    def __init__(self, response: HttpResponse) -> None:
        self._response = response
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        return self._response


def test_buffered_stream_fallback():
    sse = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    t = _BufferedOnlyTransport(resp(200, sse))
    chat = llm.Chat(llm.ProviderID.OPENAI, llm.OpenAICompatConfig(api_key="k"), transport=t)
    text = "".join(d.content_delta for d in chat.stream(llm.ChatInput(model="m", messages=[llm.ChatMessage(role="user", content="x")])))
    assert text == "hi"
