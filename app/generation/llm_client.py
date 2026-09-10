"""Cliente de LLM para geração da resposta final (Fase 5).

Só expõe streaming (`.stream()`) porque é isso que a API usa (ver
app/api/main.py) — o endpoint /query devolve a resposta via
Server-Sent Events à medida que o modelo gera, em vez de esperar o
texto inteiro. Imports dos SDKs são lazy pelo mesmo motivo dos outros
módulos com dependências pesadas/opcionais (ver
app/indexing/embeddings.py): não acoplar o resto do projeto a uma lib
que pode não estar instalada (ex.: testes de chunking/parsing).

Dois clientes:
  - AnthropicLLMClient: API oficial da Anthropic.
  - OpenAICompatibleLLMClient: qualquer proxy compatível com o formato
    de chat completions da OpenAI (base_url configurável) — usado aqui
    pra apontar pro 9Router local do usuário, que expõe vários modelos
    (combo) atrás de um único endpoint OpenAI-compatible em
    http://localhost:20128/v1. Ver app/api/dependencies.py::build_llm_client
    pra qual dos dois é escolhido, com base em LLM_PROVIDER.
"""

from __future__ import annotations

from typing import Iterator, Protocol


class LLMClient(Protocol):
    def stream(self, system: str, user: str) -> Iterator[str]: ...


class AnthropicLLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY não configurada. Defina no .env antes de "
                "subir a API (LLM_PROVIDER=anthropic)."
            )
        from anthropic import Anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic(api_key=api_key, timeout=60.0, max_retries=1)

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream


class OpenAICompatibleLLMClient:
    """Cliente pra qualquer proxy/gateway que fale o protocolo de chat
    completions da OpenAI — LiteLLM, Ollama, vLLM, ou (o caso de uso
    daqui) o 9Router rodando localmente. `base_url` deve incluir o
    sufixo /v1 (ex.: http://localhost:20128/v1)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        if not base_url:
            raise RuntimeError(
                "LLM_BASE_URL não configurada. Defina no .env antes de subir "
                "a API (LLM_PROVIDER=openai_compatible)."
            )
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Alguns proxies locais não exigem API key mesmo com "Require API
        # key" desligado — o SDK da OpenAI exige uma string não-vazia de
        # qualquer forma, por isso o fallback. timeout curto + sem retry
        # automático: se o proxy/modelo travar, falha rápido em vez de
        # ficar parado sem feedback nenhum.
        self._client = OpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=60.0, max_retries=1
        )

    def stream(self, system: str, user: str) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
