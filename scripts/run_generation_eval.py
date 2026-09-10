"""CLI de avaliação de Answer Faithfulness via RAGAS (Fase 5).

Diferente de run_evaluation.py (Fase 4, avalia só retrieval), este
script gera respostas de verdade via LLM e mede se elas são fiéis ao
contexto recuperado — por isso tem custo real de API a cada execução
(2 chamadas de LLM por pergunta: 1 pra gerar a resposta, 1+ pro juiz do
RAGAS decompor e checar as afirmações). Use --limit pra controlar isso.

Uso:
    python scripts/run_generation_eval.py --dry-run
    python scripts/run_generation_eval.py --limit 5
    python scripts/run_generation_eval.py --limit 5 --mode hybrid_rerank
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.dataset import DEFAULT_PATH, load_dataset
from app.evaluation.faithfulness import build_evaluator_llm, build_samples, evaluate_faithfulness
from app.retrieval.pipeline import VALID_MODES

ROOT = Path(__file__).parent.parent
DEFAULT_BM25_PATH = str(ROOT / "data" / "bm25_index.pkl")


class _DryRunRetriever:
    """Fake de retrieval só pra este script: devolve sempre o mesmo
    chunk de exemplo, independente da pergunta — o objetivo do --dry-run
    aqui é validar a orquestração (dataset -> generator.answer ->
    build_samples), não simular relevância real."""

    def retrieve(self, query: str, mode: str, top_k: int) -> list[dict]:
        return [
            {
                "id": "1",
                "payload": {
                    "text": "Texto de exemplo.",
                    "source": "doc.md",
                    "section": "Exemplo",
                    "page": None,
                },
            }
        ]


def print_results(results: list[dict]) -> None:
    print(f"\n=== Answer Faithfulness (RAGAS), n={len(results)} ===")
    for r in results:
        score_label = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "ERRO"
        print(f"\n[{score_label}] {r['question']}")
        print(f"  resposta: {r['response'][:200]}")
        if r.get("error"):
            print(f"  erro do juiz: {r['error']}")
    scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    failed = len(results) - len(scores)
    if scores:
        print(f"\nMédia: {sum(scores) / len(scores):.3f} ({len(scores)}/{len(results)} perguntas avaliadas)")
    if failed:
        print(f"{failed} pergunta(s) falharam no juiz do RAGAS (ver 'erro do juiz' acima).")


def build_dry_run_generator():
    from app.generation.pipeline import AnswerGenerator
    from tests.fakes import FakeLLMClient

    llm = FakeLLMClient("Resposta de exemplo, fundamentada no contexto [1].")
    return AnswerGenerator(_DryRunRetriever(), llm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia Answer Faithfulness via RAGAS (Fase 5)")
    parser.add_argument("--dataset", default=str(DEFAULT_PATH))
    parser.add_argument("--mode", choices=list(VALID_MODES), default="hybrid_rerank_blend")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=5, help="Quantas perguntas rodar (custo de API por pergunta)"
    )
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Usa fakes: só valida a orquestração, sem custo de API nem RAGAS",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if args.dry_run:
        generator = build_dry_run_generator()
        samples = build_samples(
            generator, dataset, mode=args.mode, top_k=args.top_k, limit=args.limit
        )
        print(f"\n=== Dry-run: {len(samples)} respostas geradas (fakes, sem custo de API) ===")
        for s in samples:
            print(f"\n[{s['question']}]\n  resposta: {s['response']}\n  contextos: {len(s['retrieved_contexts'])}")
        print("\nPra rodar a métrica de verdade (RAGAS + Anthropic), remova --dry-run.")
        return

    from app.api.dependencies import build_generator
    from app.config.settings import settings

    generator = build_generator()
    samples = build_samples(
        generator, dataset, mode=args.mode, top_k=args.top_k, limit=args.limit
    )

    evaluator_llm = build_evaluator_llm(
        model=settings.llm_model,
        provider=settings.llm_provider,
        api_key=settings.llm_api_key or settings.anthropic_api_key,
        base_url=settings.llm_base_url,
    )
    results = evaluate_faithfulness(samples, evaluator_llm)
    print_results(results)


if __name__ == "__main__":
    main()
