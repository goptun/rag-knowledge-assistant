import unittest

from app.evaluation.metrics import is_relevant, precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.runner import evaluate_all_modes, evaluate_mode, find_regressions
from tests.fakes import FakeRetriever


def _hit(source: str, section: str | None = None, page: int | None = None) -> dict:
    return {"id": f"{source}:{section}:{page}", "payload": {"source": source, "section": section, "page": page}}


class TestIsRelevant(unittest.TestCase):
    def test_matches_by_section(self):
        payload = {"source": "doc.md", "section": "Autenticação", "page": None}
        self.assertTrue(is_relevant(payload, {"source": "doc.md", "section": "Autenticação"}))
        self.assertFalse(is_relevant(payload, {"source": "doc.md", "section": "Outra"}))

    def test_matches_by_page(self):
        payload = {"source": "manual.pdf", "section": None, "page": 3}
        self.assertTrue(is_relevant(payload, {"source": "manual.pdf", "page": 3}))
        self.assertFalse(is_relevant(payload, {"source": "manual.pdf", "page": 4}))

    def test_different_source_never_matches(self):
        payload = {"source": "a.md", "section": "X", "page": None}
        self.assertFalse(is_relevant(payload, {"source": "b.md", "section": "X"}))


class TestRetrievalMetrics(unittest.TestCase):
    def setUp(self):
        self.relevant = [{"source": "api.md", "section": "Rate limiting"}]

    def test_precision_at_k_all_relevant(self):
        retrieved = [_hit("api.md", section="Rate limiting")] * 3
        self.assertEqual(precision_at_k(retrieved, self.relevant, k=3), 1.0)

    def test_precision_at_k_partial(self):
        retrieved = [
            _hit("api.md", section="Rate limiting"),
            _hit("api.md", section="Webhooks"),
        ]
        self.assertAlmostEqual(precision_at_k(retrieved, self.relevant, k=2), 0.5)

    def test_recall_at_k_found(self):
        retrieved = [_hit("api.md", section="Webhooks"), _hit("api.md", section="Rate limiting")]
        self.assertEqual(recall_at_k(retrieved, self.relevant, k=2), 1.0)

    def test_recall_at_k_not_found_within_k(self):
        retrieved = [_hit("api.md", section="Webhooks")]
        self.assertEqual(recall_at_k(retrieved, self.relevant, k=1), 0.0)

    def test_reciprocal_rank_first_position(self):
        retrieved = [_hit("api.md", section="Rate limiting"), _hit("api.md", section="Webhooks")]
        self.assertEqual(reciprocal_rank(retrieved, self.relevant), 1.0)

    def test_reciprocal_rank_second_position(self):
        retrieved = [_hit("api.md", section="Webhooks"), _hit("api.md", section="Rate limiting")]
        self.assertEqual(reciprocal_rank(retrieved, self.relevant), 0.5)

    def test_reciprocal_rank_no_match(self):
        retrieved = [_hit("api.md", section="Webhooks")]
        self.assertEqual(reciprocal_rank(retrieved, self.relevant), 0.0)

    def test_empty_retrieved_does_not_crash(self):
        self.assertEqual(precision_at_k([], self.relevant, k=5), 0.0)
        self.assertEqual(recall_at_k([], self.relevant, k=5), 0.0)
        self.assertEqual(reciprocal_rank([], self.relevant), 0.0)


class TestEvaluateMode(unittest.TestCase):
    def test_perfect_retriever_scores_one(self):
        dataset = [
            {"question": "q1", "relevant": [{"source": "a.md", "section": "S1"}]},
            {"question": "q2", "relevant": [{"source": "b.md", "section": "S2"}]},
        ]
        retriever = FakeRetriever({
            "q1": [_hit("a.md", section="S1")],
            "q2": [_hit("b.md", section="S2")],
        })

        result = evaluate_mode(retriever, dataset, mode="hybrid", k=5)

        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["n_questions"], 2)

    def test_retriever_with_no_hits_scores_zero(self):
        dataset = [{"question": "q1", "relevant": [{"source": "a.md", "section": "S1"}]}]
        retriever = FakeRetriever({"q1": [_hit("a.md", section="outra secao")]})

        result = evaluate_mode(retriever, dataset, mode="vector_only", k=5)

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["mrr"], 0.0)

    def test_passes_correct_mode_to_retriever(self):
        dataset = [{"question": "q1", "relevant": [{"source": "a.md", "section": "S1"}]}]
        retriever = FakeRetriever({"q1": [_hit("a.md", section="S1")]})

        evaluate_mode(retriever, dataset, mode="hybrid_rerank", k=3)

        self.assertEqual(retriever.calls, [("q1", "hybrid_rerank", 3)])


class TestEvaluateAllModes(unittest.TestCase):
    def test_returns_one_result_per_mode(self):
        dataset = [{"question": "q1", "relevant": [{"source": "a.md", "section": "S1"}]}]
        retriever = FakeRetriever({"q1": [_hit("a.md", section="S1")]})

        results = evaluate_all_modes(
            retriever, dataset, modes=["vector_only", "hybrid", "hybrid_rerank"], k=5
        )

        self.assertEqual([r["mode"] for r in results], ["vector_only", "hybrid", "hybrid_rerank"])


class TestDatasetFile(unittest.TestCase):
    """Sanity check no dataset real (não nos fixtures sintéticos acima)."""

    def test_dataset_loads_and_is_well_formed(self):
        from app.evaluation.dataset import load_dataset

        dataset = load_dataset()
        self.assertGreaterEqual(len(dataset), 19)

        for item in dataset:
            self.assertIn("question", item)
            self.assertIn("relevant", item)
            self.assertGreater(len(item["relevant"]), 0)
            for rel in item["relevant"]:
                self.assertIn("source", rel)
                self.assertTrue("section" in rel or "page" in rel)

    def test_dataset_has_multi_relevant_questions(self):
        from app.evaluation.dataset import load_dataset

        dataset = load_dataset()
        multi_relevant = [item for item in dataset if len(item["relevant"]) > 1]
        self.assertGreaterEqual(len(multi_relevant), 5)


class TestEvaluateModeDetails(unittest.TestCase):
    def test_details_report_found_and_missed_relevant_items(self):
        dataset = [{
            "id": "q1",
            "question": "q1",
            "relevant": [
                {"source": "a.md", "section": "S1"},
                {"source": "a.md", "section": "S2"},
            ],
        }]
        # só acha S1, perde S2
        retriever = FakeRetriever({"q1": [_hit("a.md", section="S1")]})

        result = evaluate_mode(retriever, dataset, mode="hybrid", k=5)
        detail = result["details"][0]

        self.assertEqual(detail["relevant_found"], [{"source": "a.md", "section": "S1"}])
        self.assertEqual(detail["relevant_missed"], [{"source": "a.md", "section": "S2"}])
        self.assertTrue(detail["retrieved"][0]["hit"])


class TestFindRegressions(unittest.TestCase):
    def test_detects_recall_drop_between_modes(self):
        dataset = [{
            "id": "q1",
            "question": "q1",
            "relevant": [
                {"source": "a.md", "section": "S1"},
                {"source": "a.md", "section": "S2"},
            ],
        }]
        # "hybrid" acha as duas seções; "hybrid_rerank" perde S2
        retriever = FakeRetriever({
            "q1": {
                "hybrid": [_hit("a.md", section="S1"), _hit("a.md", section="S2")],
                "hybrid_rerank": [_hit("a.md", section="S1")],
            }
        })

        results = [
            evaluate_mode(retriever, dataset, mode="hybrid", k=5),
            evaluate_mode(retriever, dataset, mode="hybrid_rerank", k=5),
        ]
        regressions = find_regressions(results)

        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]["id"], "q1")
        self.assertEqual(regressions[0]["from_mode"], "hybrid")
        self.assertEqual(regressions[0]["to_mode"], "hybrid_rerank")
        self.assertEqual(regressions[0]["lost"], [{"source": "a.md", "section": "S2"}])

    def test_no_regression_when_recall_stable_or_improves(self):
        dataset = [{"id": "q1", "question": "q1", "relevant": [{"source": "a.md", "section": "S1"}]}]
        retriever = FakeRetriever({"q1": [_hit("a.md", section="S1")]})

        results = [
            evaluate_mode(retriever, dataset, mode="vector_only", k=5),
            evaluate_mode(retriever, dataset, mode="hybrid", k=5),
        ]
        self.assertEqual(find_regressions(results), [])


if __name__ == "__main__":
    unittest.main()
