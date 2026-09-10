"""ID compartilhado entre Qdrant (dense) e BM25 (sparse).

Os dois índices precisam concordar sobre o "id" de um chunk pra que a
fusão RRF (Fase 3) consiga casar um resultado dense com o mesmo chunk
retornado pelo sparse. point_id() é a função única de conversão — usada
tanto no upsert do Qdrant quanto na construção do índice BM25.
"""

from __future__ import annotations

import uuid


def point_id(chunk_id: str) -> str:
    """UUID determinístico a partir do id lógico do chunk (Chunk.id).

    uuid5 garante que o mesmo chunk sempre gera o mesmo id — reindexar
    o mesmo documento substitui o ponto em vez de duplicar.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
