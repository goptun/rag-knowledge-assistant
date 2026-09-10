import json
import unittest

from fastapi.testclient import TestClient

from app.api.dependencies import get_generator
from app.api.main import app
from app.generation.pipeline import AnswerGenerator
from tests.fakes import FakeLLMClient, FakeRetriever


def _chunk(source, text="texto", section=None, page=None):
    return {"id": source, "payload": {"text": text, "source": source, "section": section, "page": page}}


def _parse_sse(body: str) -> list[dict]:
    """Faz o parse mínimo do formato SSE usado em app/api/main.py:
    blocos separados por linha em branco, cada um com 'event: X' e
    'data: {...}'."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.strip().splitlines()
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event_type is not None:
            events.append({"type": event_type, "data": data})
    return events


def _build_fake_generator(chunks, response_text) -> AnswerGenerator:
    retriever = FakeRetriever(results_by_question={"pergunta": chunks})
    llm = FakeLLMClient(response_text)
    return AnswerGenerator(retriever, llm)


class TestQueryEndpoint(unittest.TestCase):
    def setUp(self):
        self.chunks = [_chunk("a.md", text="A API usa OAuth2.", section="Autenticação")]
        self.generator = _build_fake_generator(self.chunks, "Usa OAuth2 [1].")
        app.dependency_overrides[get_generator] = lambda: self.generator
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_query_streams_retrieved_delta_and_done_events(self):
        response = self.client.post("/query", json={"question": "pergunta"})

        self.assertEqual(response.status_code, 200)
        events = _parse_sse(response.text)

        self.assertEqual(events[0]["type"], "retrieved")
        self.assertEqual(events[0]["data"]["retrieved"][0]["source"], "a.md")

        delta_events = [e for e in events if e["type"] == "delta"]
        self.assertTrue(delta_events)
        self.assertEqual("".join(e["data"]["text"] for e in delta_events), "Usa OAuth2 [1].")

        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["data"]["answer"], "Usa OAuth2 [1].")
        self.assertEqual(len(done["data"]["citations"]), 1)
        self.assertEqual(done["data"]["citations"][0]["source"], "a.md")

    def test_query_rejects_invalid_mode(self):
        response = self.client.post("/query", json={"question": "pergunta", "mode": "modo_inexistente"})
        self.assertEqual(response.status_code, 422)

    def test_query_rejects_empty_question(self):
        response = self.client.post("/query", json={"question": ""})
        self.assertEqual(response.status_code, 422)


class TestHealthEndpoint(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    def test_health_reports_bm25_loaded_true_when_sparse_index_present(self):
        generator = _build_fake_generator([_chunk("a.md")], "ok")
        generator.retriever.sparse_index = object()
        app.dependency_overrides[get_generator] = lambda: generator
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["bm25_loaded"])

    def test_health_reports_bm25_loaded_false_when_sparse_index_absent(self):
        generator = _build_fake_generator([_chunk("a.md")], "ok")
        generator.retriever.sparse_index = None
        app.dependency_overrides[get_generator] = lambda: generator
        client = TestClient(app)

        response = client.get("/health")

        self.assertFalse(response.json()["bm25_loaded"])


class _FailingLLMClient:
    """Simula uma falha do provedor de LLM (rate limit, billing, etc.)
    no meio do streaming."""

    def stream(self, system: str, user: str):
        raise RuntimeError("Your credit balance is too low to access the Anthropic API.")
        yield  # pragma: no cover - torna a função um generator; nunca alcançado


class TestQueryEndpointErrorHandling(unittest.TestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    def test_llm_failure_mid_stream_emits_error_event_instead_of_dropping_connection(self):
        chunks = [_chunk("a.md", text="A API usa OAuth2.", section="Autenticação")]
        retriever = FakeRetriever(results_by_question={"pergunta": chunks})
        generator = AnswerGenerator(retriever, _FailingLLMClient())
        app.dependency_overrides[get_generator] = lambda: generator
        client = TestClient(app)

        response = client.post("/query", json={"question": "pergunta"})

        self.assertEqual(response.status_code, 200)
        events = _parse_sse(response.text)

        self.assertEqual(events[0]["type"], "retrieved")
        error_events = [e for e in events if e["type"] == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertIn("credit balance", error_events[0]["data"]["message"])


if __name__ == "__main__":
    unittest.main()
