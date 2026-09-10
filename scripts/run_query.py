"""CLI da Fase 3: consulta com retrieval híbrido + reranking.

Uso real (precisa do Qdrant indexado via run_indexing.py, com BM25
salvo em data/bm25_index.pkl):
    python scripts/run_query.py --query "Como funciona o rate limiting da API?"
    python scripts/run_query.py --query "..." --mode vector_only
    python scripts/run_query.py --query "..." --mode hybrid

Smoke test sem dependências pesadas (embedder/dense/sparse/reranker
falsos, só valida a orquestração):
    python scripts/run_query.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# garante que a raiz do projeto está no sys.path, independente de
# como o script é chamado (python scripts/x.py em vez de -m scripts.x)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.pipeline import VALID_MODES, HybridRetriever

ROOT = Path(__file__).parent.parent
DEFAULT_BM25_PATH = str(ROOT / "data" / "bm25_index.pkl")


def print_results(results: list[dict], mode: str, full_text: bool = False) -> None:
    print(f"\n=== Resultados ({mode}) ===")
    if not results:
        print("Nenhum resultado.")
        return

    for i, r in enumerate(results, start=1):
        payload = r["payload"]
        score_key = next(
            (k for k in ("blended_score", "rerank_score", "rrf_score", "score") if k in r), None
        )
        score = r.get(score_key, 0.0)
        location = f"page={payload.get('page')}" if payload.get("page") else f"section={payload.get('section')!r}"
        print(f"\n[{i}] {payload.get('source')} ({location}) — {score_key}={score:.4f}")
        text = payload["text"].replace("\n", " ")
        if full_text:
            print(f"    {text}")
        else:
            print(f"    {text[:200]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta com retrieval híbrido (Fase 3)")
    parser.add_argument("--query", default="pergunta de teste")
    parser.add_argument(
        "--mode", choices=list(VALID_MODES), default="hybrid_rerank"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Roda com componentes falsos, sem precisar de deps pesadas nem Qdrant no ar",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Imprime o texto completo de cada chunk, sem truncar em 200 chars (debug)",
    )
    args = parser.parse_args()

    if args.dry_run:
        from tests.fakes import (
            FakeDenseSearcher,
            FakeEmbedder,
            FakeReranker,
            FakeSparseSearcher,
        )

        def fake_item(doc_id, score, source, text):
            return {"id": doc_id, "score": score, "payload": {"source": source, "section": None, "page": None, "text": text}}

        embedder = FakeEmbedder()
        dense = FakeDenseSearcher([
            fake_item("1", 0.92, "api_documentation.md", "Rate limiting é de 100 requisições por minuto por token."),
            fake_item("2", 0.81, "company_policy.html", "Funcionários remotos devem estar disponíveis das 10h às 16h."),
        ])
        sparse = FakeSparseSearcher([
            fake_item("1", 8.2, "api_documentation.md", "Rate limiting é de 100 requisições por minuto por token."),
            fake_item("3", 5.1, "product_manual.pdf", "Resistência de terminação da rede RS-485 deve ser 120 ohms."),
        ])
        reranker = FakeReranker(score_by_id={"1": 5.0, "2": 1.0, "3": 0.5})
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=dense,
            sparse_index=sparse,
            reranker=reranker,
            candidate_pool_size=args.candidate_pool,
        )
        print("[dry-run] usando embedder, dense/sparse searchers e reranker falsos")
    else:
        # imports pesados só acontecem aqui, no caminho real
        from app.config.settings import settings
        from app.indexing.bm25_store import BM25Index
        from app.indexing.embeddings import HuggingFaceEmbedder
        from app.indexing.qdrant_store import QdrantStore
        from app.retrieval.reranker import CrossEncoderReranker

        print(f"Carregando embedder ({settings.embedding_model}) e reranker ({settings.reranker_model})...")
        embedder = HuggingFaceEmbedder(settings.embedding_model)
        dense = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
        sparse = BM25Index.load(args.bm25_path) if Path(args.bm25_path).exists() else None
        reranker = CrossEncoderReranker(settings.reranker_model)

        if sparse is None and args.mode != "vector_only":
            print(
                f"Aviso: índice BM25 não encontrado em {args.bm25_path}. "
                "Rode run_indexing.py sem --no-bm25 primeiro, ou use --mode vector_only."
            )
            return

        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=dense,
            sparse_index=sparse,
            reranker=reranker,
            candidate_pool_size=args.candidate_pool,
        )

    results = retriever.retrieve(args.query, mode=args.mode, top_k=args.top_k)
    print_results(results, args.mode, full_text=args.full_text)


if __name__ == "__main__":
    main()
