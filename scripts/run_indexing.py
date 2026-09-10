"""CLI da Fase 2: parse -> chunk -> embed -> upsert no Qdrant.

Uso real (precisa de sentence-transformers, qdrant-client, pydantic-settings
instalados e Qdrant rodando via docker compose):
    python scripts/run_indexing.py --dir data/test_docs

Smoke test sem nenhuma dependência pesada (usa embedder/store falsos,
só valida a orquestração e os números de parsing/chunking — roda até
sem pydantic instalado):
    python scripts/run_indexing.py --dir data/test_docs --dry-run
"""

import argparse
from pathlib import Path
import sys

# garante que a raiz do projeto está no sys.path, independente de
# como o script é chamado (python scripts/x.py em vez de -m scripts.x)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.pipeline import build_index

ROOT = Path(__file__).parent.parent

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 50
DEFAULT_BM25_PATH = str(ROOT / "data" / "bm25_index.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos no Qdrant (Fase 2)")
    parser.add_argument("--dir", default=str(ROOT / "data" / "test_docs"))
    parser.add_argument("--strategy", choices=["recursive", "semantic"], default="recursive")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--bm25-path",
        default=DEFAULT_BM25_PATH,
        help="Onde salvar o índice BM25 (sparse) — usado no hybrid retrieval da Fase 3",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Pula a construção do índice BM25 (só indexa dense no Qdrant)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Roda com embedder/store falsos, sem precisar de deps pesadas nem Qdrant no ar",
    )
    args = parser.parse_args()

    if args.dry_run:
        from tests.fakes import FakeEmbedder, InMemoryStore

        embedder = FakeEmbedder(dimension=16)
        store = InMemoryStore()
        print("[dry-run] usando embedder e vector store falsos\n")
    else:
        # imports pesados (pydantic-settings, sentence-transformers,
        # qdrant-client) só acontecem aqui, no caminho real
        from app.config.settings import settings
        from app.indexing.embeddings import HuggingFaceEmbedder
        from app.indexing.qdrant_store import QdrantStore

        print(f"Carregando modelo de embedding: {settings.embedding_model}")
        embedder = HuggingFaceEmbedder(settings.embedding_model)
        store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)

    stats = build_index(
        args.dir,
        embedder,
        store,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        bm25_index_path=None if (args.no_bm25 or args.dry_run) else args.bm25_path,
    )

    print("\n=== Resultado da indexação ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if not args.dry_run:
        print(f"\nTotal de pontos na collection '{store.collection}': {store.count()}")


if __name__ == "__main__":
    main()
