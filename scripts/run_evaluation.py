"""CLI da Fase 4: avalia e compara os 3 modos de retrieval (Precision@K,
Recall@K, MRR) sobre o dataset de perguntas em data/eval/qa_dataset.json.

Uso real (precisa do Qdrant indexado + BM25 salvo, ver Fases 2 e 3):
    python scripts/run_evaluation.py

Com detalhe pergunta-a-pergunta (o que cada modo recuperou, e quais
perguntas tiveram o recall piorado ao trocar de modo):
    python scripts/run_evaluation.py --verbose

Smoke test sem dependências pesadas (componentes falsos, valida só a
orquestração e a matemática das métricas):
    python scripts/run_evaluation.py --dry-run

Nota: Answer Faithfulness (RAGAS) não está aqui de propósito — essa
métrica avalia respostas *geradas*, e a geração via LLM é Fase 5. Este
script avalia só a qualidade do retrieval, que já é suficiente pra
comparar vector-only vs hybrid vs hybrid+rerank.
"""

import argparse
import sys
from pathlib import Path

# garante que a raiz do projeto está no sys.path, independente de
# como o script é chamado (python scripts/x.py em vez de -m scripts.x)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.dataset import load_dataset
from app.evaluation.runner import evaluate_all_modes, find_regressions
from app.retrieval.pipeline import VALID_MODES, HybridRetriever

ROOT = Path(__file__).parent.parent
DEFAULT_BM25_PATH = str(ROOT / "data" / "bm25_index.pkl")
DEFAULT_DATASET_PATH = str(ROOT / "data" / "eval" / "qa_dataset.json")


def _fmt_relevant(rel: dict) -> str:
    loc = f"section={rel['section']!r}" if "section" in rel else f"page={rel['page']}"
    return f"{rel['source']} ({loc})"


def _fmt_relevant_list(relevant: list[dict]) -> str:
    return "; ".join(_fmt_relevant(r) for r in relevant) if relevant else "(nenhum)"


def print_comparison(results: list[dict]) -> None:
    k = results[0]["k"] if results else "-"
    print(f"\n=== Comparação de retrieval (k={k}, n={results[0]['n_questions'] if results else 0}) ===\n")
    header = f"{'modo':<16} {'precision@k':>12} {'recall@k':>12} {'mrr':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['mode']:<16} {r['precision']:>12.3f} {r['recall']:>12.3f} {r['mrr']:>8.3f}")


def print_regressions(results: list[dict]) -> None:
    regressions = find_regressions(results)
    print(f"\n=== Regressões de recall entre modos ({len(regressions)} encontradas) ===")
    if not regressions:
        print("Nenhuma — o recall nunca piorou ao trocar de modo, em nenhuma pergunta.")
        return

    for reg in regressions:
        print(f"\n[{reg['id']}] {reg['question']}")
        print(
            f"  {reg['from_mode']} -> {reg['to_mode']}: "
            f"recall {reg['recall_before']:.2f} -> {reg['recall_after']:.2f}"
        )
        for rel in reg["lost"]:
            print(f"    chunk perdido: {_fmt_relevant(rel)}")


def print_verbose_details(results: list[dict]) -> None:
    for result in results:
        print(f"\n\n{'=' * 60}\nDetalhe por pergunta — modo: {result['mode']}\n{'=' * 60}")
        for d in result["details"]:
            print(f"\n[{d['id']}] {d['question']}")
            print(f"  relevantes esperados: {_fmt_relevant_list(d['relevant'])}")
            if not d["retrieved"]:
                print("  (nenhum resultado retornado)")
            for r in d["retrieved"]:
                loc = f"section={r['section']!r}" if r["section"] else f"page={r['page']}"
                mark = "OK " if r["hit"] else "-- "
                print(f"    [{mark}] rank={r['rank']} {r['source']} ({loc})")
            print(
                f"  precision={d['precision']:.2f} "
                f"recall={d['recall']:.2f} "
                f"rr={d['reciprocal_rank']:.2f}"
            )
            if d["relevant_missed"]:
                print(f"  perdidos: {_fmt_relevant_list(d['relevant_missed'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia retrieval (Fase 4)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra o retrieval pergunta-a-pergunta pra cada modo, além do resumo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Roda com componentes falsos, sem precisar de deps pesadas nem Qdrant no ar",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Dataset carregado: {len(dataset)} perguntas")

    if args.dry_run:
        from tests.fakes import (
            FakeDenseSearcher,
            FakeEmbedder,
            FakeReranker,
            FakeSparseSearcher,
        )

        embedder = FakeEmbedder()
        dense = FakeDenseSearcher([])  # sem hits reais — só valida que roda sem crashar
        sparse = FakeSparseSearcher([])
        reranker = FakeReranker()
        retriever = HybridRetriever(
            embedder=embedder, vector_store=dense, sparse_index=sparse, reranker=reranker
        )
        print(
            "[dry-run] componentes falsos sem hits reais — métricas ficarão em 0.0, "
            "isso só valida que a orquestração roda sem erro\n"
        )
    else:
        from app.config.settings import settings
        from app.indexing.bm25_store import BM25Index
        from app.indexing.embeddings import HuggingFaceEmbedder
        from app.indexing.qdrant_store import QdrantStore
        from app.retrieval.reranker import CrossEncoderReranker

        if not Path(args.bm25_path).exists():
            print(
                f"Índice BM25 não encontrado em {args.bm25_path}. "
                "Rode scripts/run_indexing.py primeiro."
            )
            return

        print(f"Carregando embedder ({settings.embedding_model}) e reranker ({settings.reranker_model})...")
        embedder = HuggingFaceEmbedder(settings.embedding_model)
        dense = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
        sparse = BM25Index.load(args.bm25_path)
        reranker = CrossEncoderReranker(settings.reranker_model)
        retriever = HybridRetriever(
            embedder=embedder, vector_store=dense, sparse_index=sparse, reranker=reranker
        )

    results = evaluate_all_modes(retriever, dataset, modes=list(VALID_MODES), k=args.k)
    print_comparison(results)
    print_regressions(results)

    if args.verbose:
        print_verbose_details(results)


if __name__ == "__main__":
    main()
