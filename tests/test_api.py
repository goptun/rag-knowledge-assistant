import json
import unittest

from fastapi.testclient import TestClient

from app.api import rate_limit as rate_limit_module
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
        rate_limit_module.reset_for_tests()
        self.chunks = [_chunk("a.md", text="A API usa OAuth2.", section="Autenticação")]
        self.generator = _build_fake_generator(self.chunks, "Usa OAuth2 [1].")
        app.dependency_overrides[get_generator] = lambda: self.generator
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        rate_limit_module.reset_for_tests()

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
    def setUp(self):
        rate_limit_module.reset_for_tests()

    def tearDown(self):
        app.dependency_overrides.clear()
        rate_limit_module.reset_for_tests()

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


class TestRateLimiting(unittest.TestCase):
    def setUp(self):
        rate_limit_module.reset_for_tests()
        chunks = [_chunk("a.md", text="A API usa OAuth2.", section="Autenticação")]
        self.generator = _build_fake_generator(chunks, "Usa OAuth2 [1].")
        app.dependency_overrides[get_generator] = lambda: self.generator
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        rate_limit_module.reset_for_tests()

    def test_exceeding_limit_within_window_returns_429(self):
        limit = rate_limit_module.MAX_REQUESTS_PER_WINDOW
        for _ in range(limit):
            response = self.client.post("/query", json={"question": "pergunta"})
            self.assertEqual(response.status_code, 200)

        response = self.client.post("/query", json={"question": "pergunta"})

        self.assertEqual(response.status_code, 429)
        self.assertIn("detail", response.json())

    @staticmethod
    def _request(host="10.0.0.1", **headers):
        """Requisição falsa: `headers` é um dict (o código real usa .get em minúsculas), como no Starlette."""
        from unittest.mock import Mock

        return Mock(headers={k.lower().replace("_", "-"): v for k, v in headers.items()}, client=Mock(host=host))

    def test_limit_is_scoped_per_client_ip_when_there_is_no_proxy_header(self):
        from app.api.rate_limit import rate_limit

        limit = rate_limit_module.MAX_REQUESTS_PER_WINDOW
        req_a, req_b = self._request(host="1.1.1.1"), self._request(host="2.2.2.2")

        for _ in range(limit):
            rate_limit(req_a)  # não deve levantar
        with self.assertRaises(Exception):
            rate_limit(req_a)
        rate_limit(req_b)  # IP diferente, cota independente

    def test_visitors_behind_the_same_proxy_peer_have_independent_quotas(self):
        """Regressão: atrás do Nginx o peer TCP é sempre o gateway do Docker. Chaveando por client.host,
        todos os visitantes dividiam UM bucket (8 perguntas/min no total em vez de por pessoa)."""
        from app.api.rate_limit import rate_limit

        limit = rate_limit_module.MAX_REQUESTS_PER_WINDOW
        gateway = "172.18.0.1"
        alice = self._request(host=gateway, CF_Connecting_IP="203.0.113.10")
        bob = self._request(host=gateway, CF_Connecting_IP="203.0.113.20")

        for _ in range(limit):
            rate_limit(alice)
        with self.assertRaises(Exception):
            rate_limit(alice)
        for _ in range(limit):  # o Bob, no mesmo gateway, ainda tem a cota inteira
            rate_limit(bob)

    def test_cloudflare_header_wins_over_forwarded_for_and_first_hop_is_used(self):
        from app.api.rate_limit import visitor_key

        both = self._request(CF_Connecting_IP="203.0.113.1", X_Forwarded_For="198.51.100.9, 10.0.0.1")
        only_xff = self._request(X_Forwarded_For="198.51.100.9, 10.0.0.1")
        neither = self._request(host="192.0.2.7")
        self.assertEqual(visitor_key(both), "203.0.113.1")
        self.assertEqual(visitor_key(only_xff), "198.51.100.9")
        self.assertEqual(visitor_key(neither), "192.0.2.7")

    def test_global_cap_limits_the_total_across_visitors(self):
        from unittest.mock import patch

        from app.api.rate_limit import rate_limit

        with patch.object(rate_limit_module, "GLOBAL_MAX_REQUESTS_PER_WINDOW", 5):
            for i in range(5):
                rate_limit(self._request(CF_Connecting_IP=f"203.0.113.{i}"))
            with self.assertRaises(Exception):  # 6º visitante distinto: o teto global já foi atingido
                rate_limit(self._request(CF_Connecting_IP="203.0.113.99"))

    def test_429_carries_retry_after_and_per_visitor_isolation_over_http(self):
        limit = rate_limit_module.MAX_REQUESTS_PER_WINDOW
        vis = lambda ip: {"CF-Connecting-IP": ip}
        for _ in range(limit):
            self.assertEqual(self.client.post("/query", json={"question": "q"}, headers=vis("203.0.113.1")).status_code, 200)
        blocked = self.client.post("/query", json={"question": "q"}, headers=vis("203.0.113.1"))
        self.assertEqual(blocked.status_code, 429)
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)
        # outro visitante (mesmo peer "testclient") segue sendo atendido
        self.assertEqual(self.client.post("/query", json={"question": "q"}, headers=vis("203.0.113.2")).status_code, 200)

    def test_tracked_visitors_are_bounded(self):
        from unittest.mock import patch

        from app.api.rate_limit import rate_limit

        clock = {"t": 1000.0}
        with patch.object(rate_limit_module, "MAX_TRACKED_VISITORS", 20), patch.object(rate_limit_module.time, "monotonic", lambda: clock["t"]), patch.object(
            rate_limit_module, "GLOBAL_MAX_REQUESTS_PER_WINDOW", 10**9
        ):
            for i in range(60):
                rate_limit(self._request(CF_Connecting_IP=f"198.51.100.{i}"))
                clock["t"] += 5  # cada visitante expira depois de 60 s
            self.assertLess(len(rate_limit_module._hits), 60)  # entradas expiradas foram descartadas


if __name__ == "__main__":
    unittest.main()
