import unittest
from pathlib import Path

from app.indexing.pipeline import build_index
from tests.fakes import FakeEmbedder, InMemoryStore

TEST_DOCS = Path(__file__).parent.parent / "data" / "test_docs"


class TestBuildIndex(unittest.TestCase):
    def test_indexes_all_chunks(self):
        embedder = FakeEmbedder(dimension=8)
        store = InMemoryStore()

        stats = build_index(TEST_DOCS, embedder, store, strategy="recursive")

        self.assertEqual(stats["documents_parsed"], 3)
        self.assertGreater(stats["chunks_indexed"], 0)
        self.assertEqual(stats["chunks_indexed"], store.count())

    def test_ensure_collection_called_once_with_correct_dimension(self):
        embedder = FakeEmbedder(dimension=16)
        store = InMemoryStore()

        build_index(TEST_DOCS, embedder, store, batch_size=2)

        self.assertEqual(store.ensure_collection_calls, 1)
        self.assertEqual(store.vector_size, 16)

    def test_respects_batch_size(self):
        embedder = FakeEmbedder()
        store = InMemoryStore()

        stats = build_index(TEST_DOCS, embedder, store, batch_size=3)

        expected_batches = -(-stats["chunks_indexed"] // 3)  # ceil division
        self.assertEqual(len(embedder.calls), expected_batches)
        for call in embedder.calls[:-1]:
            self.assertEqual(len(call), 3)

    def test_empty_directory_returns_zero_stats(self):
        embedder = FakeEmbedder()
        store = InMemoryStore()
        empty_dir = Path(__file__).parent  # não tem pdf/html/md

        stats = build_index(empty_dir, embedder, store)

        self.assertEqual(stats["chunks_indexed"], 0)
        self.assertEqual(store.ensure_collection_calls, 0)


if __name__ == "__main__":
    unittest.main()
