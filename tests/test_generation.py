import unittest

from app.generation.citations import build_citations, extract_cited_indices
from app.generation.pipeline import AnswerGenerator
from app.generation.prompt import build_context_block, build_user_prompt
from tests.fakes import FakeLLMClient, FakeRetriever


def _chunk(idx_source, text="texto", section=None, page=None):
    return {
        "id": idx_source,
        "payload": {"text": text, "source": idx_source, "section": section, "page": page},
    }


class TestPrompt(unittest.TestCase):
    def test_context_block_numbers_chunks_in_order(self):
        chunks = [_chunk("a.md", section="Um"), _chunk("b.md", page=3)]
        block = build_context_block(chunks)
        self.assertIn("[1] Fonte: a.md (seção: Um)", block)
        self.assertIn("[2] Fonte: b.md (página: 3)", block)

    def test_user_prompt_includes_question_and_context(self):
        chunks = [_chunk("a.md", text="conteúdo relevante")]
        prompt = build_user_prompt("Qual a política de X?", chunks)
        self.assertIn("Qual a política de X?", prompt)
        self.assertIn("conteúdo relevante", prompt)


class TestCitations(unittest.TestCase):
    def test_extract_cited_indices_deduplicates_and_preserves_order(self):
        text = "Fato A [2]. Fato B [1]. Fato C [2] de novo."
        self.assertEqual(extract_cited_indices(text), [2, 1])

    def test_extract_cited_indices_accepts_fullwidth_brackets(self):
        text = "Fato A 【2】. Fato B [1]. Fato C 【2】."
        self.assertEqual(extract_cited_indices(text), [2, 1])

    def test_build_citations_resolves_to_real_metadata(self):
        chunks = [_chunk("a.md", section="Auth"), _chunk("b.md", page=5)]
        citations = build_citations("Usa OAuth2 [1] e expira em 5 min [2].", chunks)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0], {"index": 1, "source": "a.md", "section": "Auth", "page": None})
        self.assertEqual(citations[1], {"index": 2, "source": "b.md", "section": None, "page": 5})

    def test_build_citations_ignores_out_of_range_index(self):
        chunks = [_chunk("a.md")]
        citations = build_citations("Alguma coisa [7].", chunks)
        self.assertEqual(citations, [])


class TestAnswerGenerator(unittest.TestCase):
    def test_stream_emits_retrieved_then_deltas_then_done_with_citations(self):
        chunks = [_chunk("a.md", text="A API usa OAuth2.", section="Autenticação")]
        retriever = FakeRetriever(results_by_question={"pergunta": chunks})
        llm = FakeLLMClient("A API usa OAuth2 [1].", chunk_size=6)
        generator = AnswerGenerator(retriever, llm)

        events = list(generator.stream("pergunta", mode="hybrid_rerank_blend", top_k=5))

        self.assertEqual(events[0]["type"], "retrieved")
        self.assertEqual(events[0]["chunks"], chunks)
        self.assertTrue(all(e["type"] == "delta" for e in events[1:-1]))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["answer"], "A API usa OAuth2 [1].")
        self.assertEqual(len(done["citations"]), 1)
        self.assertEqual(done["citations"][0]["source"], "a.md")

    def test_retriever_and_llm_receive_correct_arguments(self):
        chunks = [_chunk("a.md")]
        retriever = FakeRetriever(results_by_question={"pergunta": chunks})
        llm = FakeLLMClient("resposta")
        generator = AnswerGenerator(retriever, llm)

        list(generator.stream("pergunta", mode="hybrid", top_k=3))

        self.assertEqual(retriever.calls, [("pergunta", "hybrid", 3)])
        self.assertEqual(len(llm.calls), 1)

    def test_empty_retrieval_short_circuits_without_calling_llm(self):
        retriever = FakeRetriever(results_by_question={})
        llm = FakeLLMClient("não deveria ser chamado")
        generator = AnswerGenerator(retriever, llm)

        result = generator.answer("pergunta sem chunks")

        self.assertEqual(result["citations"], [])
        self.assertEqual(llm.calls, [])
        self.assertIn("Não encontrei", result["answer"])

    def test_answer_aggregates_stream_into_final_result(self):
        chunks = [_chunk("a.md", section="Rate limiting")]
        retriever = FakeRetriever(results_by_question={"pergunta": chunks})
        llm = FakeLLMClient("Limite de 100 req/min [1].")
        generator = AnswerGenerator(retriever, llm)

        result = generator.answer("pergunta")

        self.assertEqual(result["answer"], "Limite de 100 req/min [1].")
        self.assertEqual(result["mode"], "hybrid_rerank_blend")
        self.assertEqual(result["chunks"], chunks)
        self.assertEqual(len(result["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
