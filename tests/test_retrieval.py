import unittest

from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.pipeline import HybridRetriever
from tests.fakes import FakeDenseSearcher, FakeEmbedder, FakeReranker, FakeSparseSearcher


def _item(doc_id: str, score: float, text: str = "texto") -> dict:
    return {"id": doc_id, "score": score, "payload": {"text": text}}


class TestReciprocalRankFusion(unittest.TestCase):
    def test_document_in_both_rankings_ranks_above_single_ranking(self):
        dense = [_item("a", 0.9), _item("b", 0.8)]
        sparse = [_item("b", 5.0), _item("c", 4.0)]

        fused = reciprocal_rank_fusion([dense, sparse])

        # "b" aparece nas duas listas (rank 2 no dense, rank 1 no sparse):
        # deve vir na frente de "a" e "c", que aparecem em só uma lista.
        self.assertEqual(fused[0]["id"], "b")

    def test_respects_top_k(self):
        dense = [_item("a", 1.0), _item("b", 0.9), _item("c", 0.8)]
        fused = reciprocal_rank_fusion([dense], top_k=2)
        self.assertEqual(len(fused), 2)

    def test_empty_rankings_returns_empty(self):
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_preserves_payload(self):
        dense = [_item("a", 0.9, text="conteúdo do chunk a")]
        fused = reciprocal_rank_fusion([dense])
        self.assertEqual(fused[0]["payload"]["text"], "conteúdo do chunk a")


class TestHybridRetriever(unittest.TestCase):
    def _make_retriever(self, dense_results=None, sparse_results=None, reranker=None):
        embedder = FakeEmbedder(dimension=4)
        dense = FakeDenseSearcher(dense_results or [_item("a", 0.9), _item("b", 0.8)])
        sparse = FakeSparseSearcher(sparse_results or [_item("b", 5.0), _item("c", 4.0)])
        return HybridRetriever(
            embedder=embedder,
            vector_store=dense,
            sparse_index=sparse,
            reranker=reranker,
        ), embedder, dense, sparse

    def test_vector_only_does_not_call_sparse_index(self):
        retriever, _, dense, sparse = self._make_retriever()

        results = retriever.retrieve("pergunta", mode="vector_only", top_k=2)

        self.assertEqual(len(sparse.calls), 0)
        self.assertEqual(len(dense.calls), 1)
        self.assertEqual([r["id"] for r in results], ["a", "b"])

    def test_hybrid_fuses_dense_and_sparse(self):
        retriever, _, dense, sparse = self._make_retriever()

        results = retriever.retrieve("pergunta", mode="hybrid", top_k=3)

        self.assertEqual(len(dense.calls), 1)
        self.assertEqual(len(sparse.calls), 1)
        # "b" está nas duas listas — deve ficar em primeiro no fused
        self.assertEqual(results[0]["id"], "b")

    def test_hybrid_without_sparse_index_raises(self):
        embedder = FakeEmbedder()
        dense = FakeDenseSearcher([_item("a", 0.9)])
        retriever = HybridRetriever(embedder=embedder, vector_store=dense)

        with self.assertRaises(RuntimeError):
            retriever.retrieve("pergunta", mode="hybrid")

    def test_hybrid_rerank_calls_reranker_with_fused_candidates(self):
        reranker = FakeReranker(score_by_id={"c": 10.0, "b": 1.0, "a": 0.5})
        retriever, _, dense, sparse = self._make_retriever(reranker=reranker)

        results = retriever.retrieve("pergunta", mode="hybrid_rerank", top_k=3)

        self.assertEqual(len(reranker.calls), 1)
        # o FakeReranker força "c" pro topo, mesmo não sendo o melhor no RRF
        self.assertEqual(results[0]["id"], "c")

    def test_hybrid_rerank_without_reranker_raises(self):
        retriever, _, _, _ = self._make_retriever(reranker=None)
        with self.assertRaises(RuntimeError):
            retriever.retrieve("pergunta", mode="hybrid_rerank")

    def test_invalid_mode_raises_value_error(self):
        retriever, _, _, _ = self._make_retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("pergunta", mode="nao_existe")


if __name__ == "__main__":
    unittest.main()


class TestBlendScores(unittest.TestCase):
    def test_empty_returns_empty(self):
        from app.retrieval.blend import blend_scores
        self.assertEqual(blend_scores([]), [])

    def test_pure_rerank_when_alpha_is_one(self):
        from app.retrieval.blend import blend_scores

        candidates = [
            {"id": "a", "rrf_score": 0.03, "rerank_score": -5.0},
            {"id": "b", "rrf_score": 0.01, "rerank_score": 5.0},
        ]
        blended = blend_scores(candidates, alpha=1.0)
        self.assertEqual(blended[0]["id"], "b")  # só rerank_score importa

    def test_pure_rrf_when_alpha_is_zero(self):
        from app.retrieval.blend import blend_scores

        candidates = [
            {"id": "a", "rrf_score": 0.03, "rerank_score": -5.0},
            {"id": "b", "rrf_score": 0.01, "rerank_score": 5.0},
        ]
        blended = blend_scores(candidates, alpha=0.0)
        self.assertEqual(blended[0]["id"], "a")  # só rrf_score importa

    def test_rrf_consensus_can_rescue_low_rerank_score(self):
        from app.retrieval.blend import blend_scores

        # "b" tem RRF muito mais alto (achado por dense+sparse) e o
        # rerank_score é só um pouco menor que o de "a" — com alpha
        # pesando também o RRF, "b" supera "a" mesmo perdendo no
        # cross-encoder puro. É exatamente o cenário do caso q26.
        candidates = [
            {"id": "a", "rrf_score": 0.001, "rerank_score": 1.0},
            {"id": "b", "rrf_score": 0.03, "rerank_score": 0.95},
        ]
        blended = blend_scores(candidates, alpha=0.3)
        self.assertEqual(blended[0]["id"], "b")

    def test_all_equal_scores_does_not_crash(self):
        from app.retrieval.blend import blend_scores

        candidates = [
            {"id": "a", "rrf_score": 0.02, "rerank_score": 1.0},
            {"id": "b", "rrf_score": 0.02, "rerank_score": 1.0},
        ]
        blended = blend_scores(candidates, alpha=0.7)
        self.assertEqual(len(blended), 2)


class TestHybridRetrieverBlendMode(unittest.TestCase):
    def test_hybrid_rerank_blend_calls_reranker_without_truncating(self):
        embedder = FakeEmbedder()
        dense = FakeDenseSearcher([_item("a", 0.9), _item("b", 0.8), _item("c", 0.7)])
        sparse = FakeSparseSearcher([_item("b", 5.0), _item("c", 4.0)])
        reranker = FakeReranker(score_by_id={"a": 0.1, "b": 0.2, "c": 0.9})

        retriever = HybridRetriever(
            embedder=embedder, vector_store=dense, sparse_index=sparse, reranker=reranker
        )
        results = retriever.retrieve("pergunta", mode="hybrid_rerank_blend", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("blended_score", results[0])

    def test_hybrid_rerank_blend_without_reranker_raises(self):
        embedder = FakeEmbedder()
        dense = FakeDenseSearcher([_item("a", 0.9)])
        sparse = FakeSparseSearcher([_item("a", 5.0)])
        retriever = HybridRetriever(embedder=embedder, vector_store=dense, sparse_index=sparse)

        with self.assertRaises(RuntimeError):
            retriever.retrieve("pergunta", mode="hybrid_rerank_blend")
