"""Geração de embeddings via modelo local (sentence-transformers).

O import de sentence_transformers é feito de forma lazy (dentro do
construtor) para não quebrar o resto do projeto quando a lib não está
instalada — por exemplo, os testes de parsing/chunking não precisam
dela.
"""

from __future__ import annotations


class HuggingFaceEmbedder:
    """Embedder local baseado em sentence-transformers.

    Uso típico: HuggingFaceEmbedder(settings.embedding_model), que por
    padrão é BAAI/bge-base-en-v1.5.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Retorna um vetor normalizado (L2) por texto de entrada.

        Normalizar embeddings deixa a similaridade de cosseno equivalente
        ao produto interno, o que é mais barato de computar no Qdrant.
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
