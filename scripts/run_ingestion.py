"""CLI de validação da Fase 1: parsing + chunking, ponta a ponta.

Uso:
    python scripts/run_ingestion.py --dir data/test_docs --strategy both
    python scripts/run_ingestion.py --dir data/test_docs --strategy recursive --dump out.json

Não faz embedding nem indexação (isso é Fase 2/3) — o objetivo aqui é
validar que os documentos são parseados corretamente e que os chunks
resultantes têm tamanho e metadata sensatos antes de gastar tempo/custo
com embeddings.
"""

import argparse
import json
import statistics
from pathlib import Path
import sys

# garante que a raiz do projeto está no sys.path, independente de
# como o script é chamado (python scripts/x.py em vez de -m scripts.x)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_units
from app.parsing import parse_directory

ROOT = Path(__file__).parent.parent


def summarize(chunks, strategy: str) -> None:
    print(f"\n=== Estratégia: {strategy} ===")
    print(f"Total de chunks: {len(chunks)}")

    if not chunks:
        print("Nenhum chunk gerado.")
        return

    token_counts = [c.token_count for c in chunks]
    print(
        f"Tokens por chunk — min: {min(token_counts)}, "
        f"max: {max(token_counts)}, "
        f"média: {statistics.mean(token_counts):.1f}, "
        f"mediana: {statistics.median(token_counts):.1f}"
    )

    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print("Chunks por documento:")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count} chunks")

    print("\nExemplo (primeiro chunk):")
    first = chunks[0]
    preview = first.text[:200].replace("\n", " ")
    print(f"  source={first.source} section={first.section!r} page={first.page}")
    print(f"  tokens={first.token_count}")
    print(f"  texto: {preview}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida parsing + chunking (Fase 1)")
    parser.add_argument(
        "--dir",
        default=str(ROOT / "data" / "test_docs"),
        help="Diretório com documentos de origem (pdf/html/md)",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic", "both"],
        default="both",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--dump",
        help="Se informado, salva os chunks (da última estratégia rodada) em JSON",
    )
    args = parser.parse_args()

    source_dir = Path(args.dir)
    print(f"Parseando documentos em: {source_dir}")
    units = parse_directory(source_dir)
    print(f"Total de unidades parseadas (seções/páginas): {len(units)}")

    sources = sorted({u.source for u in units})
    print(f"Documentos encontrados: {sources}")

    strategies = ["recursive", "semantic"] if args.strategy == "both" else [args.strategy]

    last_chunks = None
    for strategy in strategies:
        chunks = chunk_units(
            units,
            strategy=strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        summarize(chunks, strategy)
        last_chunks = chunks

    if args.dump and last_chunks is not None:
        dump_path = Path(args.dump)
        payload = [
            {
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "section": c.section,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "strategy": c.strategy,
            }
            for c in last_chunks
        ]
        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nChunks salvos em: {dump_path}")


if __name__ == "__main__":
    main()
