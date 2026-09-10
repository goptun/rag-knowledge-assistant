import unittest

from app.chunking import chunk_units
from app.chunking.token_utils import count_tokens
from app.parsing.models import ParsedUnit


class TestRecursiveChunking(unittest.TestCase):
    def test_respects_chunk_size_upper_bound(self):
        long_text = "Esta é uma frase de teste repetida. " * 100
        units = [ParsedUnit(text=long_text, source="synthetic.txt")]

        chunks = chunk_units(units, strategy="recursive", chunk_size=50, overlap=10)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.token_count, 50)

    def test_overlap_carries_content_between_chunks(self):
        long_text = " ".join(f"palavra{i}" for i in range(200))
        units = [ParsedUnit(text=long_text, source="synthetic.txt")]

        chunks = chunk_units(units, strategy="recursive", chunk_size=50, overlap=10)

        self.assertGreaterEqual(len(chunks), 2)
        tail_of_first = chunks[0].text.split()[-5:]
        head_of_second = chunks[1].text.split()[:15]
        overlap_found = any(tok in head_of_second for tok in tail_of_first)
        self.assertTrue(overlap_found)

    def test_small_text_is_a_single_chunk(self):
        units = [ParsedUnit(text="Texto curto.", source="synthetic.txt")]
        chunks = chunk_units(units, strategy="recursive", chunk_size=512, overlap=50)
        self.assertEqual(len(chunks), 1)


class TestSemanticChunking(unittest.TestCase):
    def test_keeps_sections_separate(self):
        units = [
            ParsedUnit(text="Conteúdo da seção um.", source="doc.md", section="Um"),
            ParsedUnit(text="Conteúdo da seção dois.", source="doc.md", section="Dois"),
        ]
        chunks = chunk_units(units, strategy="semantic", chunk_size=512)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section, "Um")
        self.assertEqual(chunks[1].section, "Dois")

    def test_splits_oversized_section(self):
        big_section_text = "Frase de teste. " * 300  # bem acima do limite
        units = [ParsedUnit(text=big_section_text, source="doc.md", section="Grande")]
        chunks = chunk_units(units, strategy="semantic", chunk_size=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.section == "Grande" for c in chunks))


class TestTokenUtils(unittest.TestCase):
    def test_count_tokens_basic(self):
        self.assertEqual(count_tokens("um dois tres"), 3)
        self.assertEqual(count_tokens("ola, mundo!"), 4)  # "ola" "," "mundo" "!"


if __name__ == "__main__":
    unittest.main()
