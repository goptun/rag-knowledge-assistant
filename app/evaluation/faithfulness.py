"""Answer Faithfulness via RAGAS (Fase 5) — avalia se a resposta
*gerada* está fundamentada nos chunks recuperados (detecta alucinação).
Diferente das métricas da Fase 4 (Precision/Recall/MRR em
app/evaluation/metrics.py), que avaliam só a recuperação, esta métrica
avalia a geração: o RAGAS decompõe a resposta em afirmações atômicas e
checa, uma a uma, se cada afirmação pode ser inferida do contexto
recuperado.

Não precisa de resposta de referência (ground truth) — só de
question + answer + contexts —, o que combina bem com o
qa_dataset.json existente, que só tem ground truth de *relevância*
(quais chunks são relevantes), não de resposta esperada.

O "juiz" (evaluator_llm) usa o mesmo provider já configurado no projeto
(Anthropic via LLM_PROVIDER/ANTHROPIC_API_KEY) — evita depender de uma
chave da OpenAI só pra rodar avaliação, quando o resto do pipeline já é
100% Anthropic + modelos locais.
"""

from __future__ import annotations

import asyncio
from typing import Protocol


class AnswerProvider(Protocol):
    def answer(self, query: str, mode: str | None, top_k: int | None) -> dict: ...


def build_samples(
    generator: AnswerProvider,
    dataset: list[dict],
    mode: str | None = None,
    top_k: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Gera resposta + contexto pra cada pergunta do dataset (ou pras
    primeiras `limit`) e monta os samples no formato que o RAGAS espera:
    user_input, response, retrieved_contexts.

    Cada pergunta aqui é uma chamada real ao LLM de geração — por isso
    `limit` existe: rodar as 26 perguntas do dataset completo tem custo
    e tempo não-triviais quando você só quer validar a métrica."""
    questions = dataset[:limit] if limit else dataset
    samples = []
    for i, item in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] gerando resposta: {item['question'][:70]}...", flush=True)
        result = generator.answer(item["question"], mode=mode, top_k=top_k)
        contexts = [c["payload"]["text"] for c in result.get("chunks", [])]
        samples.append(
            {
                "question": item["question"],
                "user_input": item["question"],
                "response": result.get("answer", ""),
                "retrieved_contexts": contexts,
            }
        )
    return samples


def evaluate_faithfulness(samples: list[dict], evaluator_llm) -> list[dict]:
    """Roda a métrica Faithfulness do RAGAS sobre os samples, um a um.

    single_turn_ascore é assíncrono (é a API documentada do RAGAS pra
    essa métrica) — como o resto do projeto é síncrono, cada chamada é
    resolvida individualmente via asyncio.run() em vez de introduzir
    async em toda a cadeia de chamadas só por causa disso.

    Uma falha do juiz numa pergunta (o LLM não termina a geração
    estruturada dentro de max_tokens, comum com modelos menores/locais
    tentando produzir JSON) não derruba a avaliação inteira — fica
    registrada como faithfulness=None com o erro, e as outras perguntas
    continuam normalmente. Uma métrica de avaliação que quebra no
    primeiro outlier não serve pra nada."""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

    scorer = Faithfulness(llm=evaluator_llm)

    results = []
    for i, sample in enumerate(samples, start=1):
        print(f"  [{i}/{len(samples)}] juiz do RAGAS avaliando: {sample['question'][:70]}...", flush=True)
        ragas_sample = SingleTurnSample(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
        try:
            score = asyncio.run(scorer.single_turn_ascore(ragas_sample))
            print(f"      -> faithfulness={score:.2f}", flush=True)
            results.append({**sample, "faithfulness": score, "error": None})
        except Exception as exc:  # noqa: BLE001 — outlier de uma pergunta não pode derrubar a avaliação inteira
            print(f"      -> falhou: {exc}", flush=True)
            results.append({**sample, "faithfulness": None, "error": str(exc)})
    return results


def build_evaluator_llm(
    model: str,
    provider: str = "anthropic",
    api_key: str = "",
    base_url: str = "",
):
    """Wrapper do RAGAS em torno de um chat model do LangChain — forma
    documentada de plugar um LLM que não seja OpenAI/Azure/Bedrock/
    Vertex como juiz do RAGAS. Só é usado pro *julgamento*; o cliente
    usado pra gerar a resposta em si é app.generation.llm_client, que
    não depende de LangChain.

    provider="openai_compatible" usa ChatOpenAI apontado pra um
    base_url customizado (ex.: o 9Router local do usuário) — evita
    depender de crédito na API oficial da Anthropic só pra rodar o
    juiz do RAGAS."""
    from ragas.llms import LangchainLLMWrapper

    # max_tokens generoso: o RAGAS pede uma resposta *estruturada* (a
    # lista de afirmações atômicas extraídas da resposta, formatada como
    # JSON) — sem isso o default de alguns proxies corta a geração no
    # meio e o parsing falha com LLMDidNotFinishException.
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            model=model,
            api_key=api_key or "not-needed",
            base_url=base_url,
            temperature=0,
            timeout=60,
            max_retries=1,
            max_tokens=4096,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(
            model=model, api_key=api_key, temperature=0, timeout=60, max_retries=1, max_tokens=4096
        )

    return LangchainLLMWrapper(chat)
