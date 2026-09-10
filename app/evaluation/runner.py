"""Roda a avaliação de retrieval: para cada modo (vector_only, hybrid,
hybrid_rerank), computa Precision@K, Recall@K e MRR agregados sobre o
dataset de perguntas — a comparação central da Fase 4.

Cada resultado de evaluate_mode() também carrega "details": um registro
por pergunta com o que foi recuperado, quais itens relevantes foram
encontrados/perdidos e as métricas individuais — é sempre calculado
(custo desprezível), mas só é impresso pela CLI quando --verbose é
passado. Isso é o que permite descobrir EXATAMENTE qual pergunta
regrediu entre dois modos, em vez de só ver o número agregado.

Answer Faithfulness (RAGAS) fica de fora daqui de propósito: essa
métrica avalia se uma resposta *gerada* é fiel ao contexto recuperado,
então depende do LLM de geração (Fase 5) já existir.
"""

from __future__ import annotations

from typing import Protocol

from app.evaluation.metrics import is_relevant, precision_at_k, recall_at_k, reciprocal_rank


class Retriever(Protocol):
    def retrieve(self, query: str, mode: str, top_k: int) -> list[dict]: ...


def _question_detail(item: dict, retrieved: list[dict], k: int) -> dict:
    top_k = retrieved[:k]

    matched_indices = set()
    for r in top_k:
        for idx, rel in enumerate(item["relevant"]):
            if is_relevant(r["payload"], rel):
                matched_indices.add(idx)

    relevant_found = [item["relevant"][i] for i in sorted(matched_indices)]
    relevant_missed = [
        rel for i, rel in enumerate(item["relevant"]) if i not in matched_indices
    ]

    retrieved_view = [
        {
            "rank": rank,
            "source": r["payload"].get("source"),
            "section": r["payload"].get("section"),
            "page": r["payload"].get("page"),
            "hit": any(is_relevant(r["payload"], rel) for rel in item["relevant"]),
        }
        for rank, r in enumerate(top_k, start=1)
    ]

    return {
        "id": item.get("id"),
        "question": item["question"],
        "relevant": item["relevant"],
        "relevant_found": relevant_found,
        "relevant_missed": relevant_missed,
        "retrieved": retrieved_view,
        "precision": precision_at_k(retrieved, item["relevant"], k),
        "recall": recall_at_k(retrieved, item["relevant"], k),
        "reciprocal_rank": reciprocal_rank(retrieved, item["relevant"]),
    }


def evaluate_mode(retriever: Retriever, dataset: list[dict], mode: str, k: int = 5) -> dict:
    details = [
        _question_detail(item, retriever.retrieve(item["question"], mode=mode, top_k=k), k)
        for item in dataset
    ]

    n = len(dataset)
    return {
        "mode": mode,
        "k": k,
        "n_questions": n,
        "precision": sum(d["precision"] for d in details) / n if n else 0.0,
        "recall": sum(d["recall"] for d in details) / n if n else 0.0,
        "mrr": sum(d["reciprocal_rank"] for d in details) / n if n else 0.0,
        "details": details,
    }


def evaluate_all_modes(
    retriever: Retriever, dataset: list[dict], modes: list[str], k: int = 5
) -> list[dict]:
    return [evaluate_mode(retriever, dataset, mode, k=k) for mode in modes]


def find_regressions(results: list[dict]) -> list[dict]:
    """Compara modos consecutivos (na ordem de `results`) e aponta
    perguntas onde o recall caiu de um modo pro próximo — é o jeito
    direto de achar em qual pergunta o reranking (ou o RRF) atrapalhou
    em vez de ajudar."""
    regressions = []

    for prev_result, curr_result in zip(results, results[1:]):
        prev_by_id = {d["id"]: d for d in prev_result["details"]}

        for curr_detail in curr_result["details"]:
            prev_detail = prev_by_id.get(curr_detail["id"])
            if prev_detail is None:
                continue
            if curr_detail["recall"] < prev_detail["recall"]:
                regressions.append(
                    {
                        "id": curr_detail["id"],
                        "question": curr_detail["question"],
                        "from_mode": prev_result["mode"],
                        "to_mode": curr_result["mode"],
                        "recall_before": prev_detail["recall"],
                        "recall_after": curr_detail["recall"],
                        "lost": [
                            rel
                            for rel in prev_detail["relevant_found"]
                            if rel in curr_detail["relevant_missed"]
                        ],
                    }
                )

    return regressions
