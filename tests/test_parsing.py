import unittest
from pathlib import Path

from app.parsing import parse_directory, parse_document

TEST_DOCS = Path(__file__).parent.parent / "data" / "test_docs"


class TestMarkdownParser(unittest.TestCase):
    def test_splits_by_heading(self):
        units = parse_document(TEST_DOCS / "api_documentation.md")
        self.assertGreater(len(units), 1)
        sections = [u.section for u in units if u.section]
        self.assertIn("Autenticação", sections)
        self.assertIn("Rate limiting", sections)

    def test_all_units_have_source(self):
        units = parse_document(TEST_DOCS / "api_documentation.md")
        self.assertTrue(all(u.source == "api_documentation.md" for u in units))


class TestHtmlParser(unittest.TestCase):
    def test_splits_by_heading_and_ignores_tags(self):
        units = parse_document(TEST_DOCS / "company_policy.html")
        sections = [u.section for u in units if u.section]
        self.assertIn("Elegibilidade", sections)
        for u in units:
            self.assertNotIn("<p>", u.text)


class TestPdfParser(unittest.TestCase):
    def test_one_unit_per_page(self):
        units = parse_document(TEST_DOCS / "product_manual.pdf")
        self.assertEqual(len(units), 5)  # PAGES no generate_test_pdf.py
        self.assertEqual(units[0].page, 1)
        self.assertEqual(units[-1].page, 5)


class TestParseDirectory(unittest.TestCase):
    def test_parses_all_supported_files(self):
        units = parse_directory(TEST_DOCS)
        sources = {u.source for u in units}
        self.assertEqual(
            sources,
            {"api_documentation.md", "company_policy.html", "product_manual.pdf"},
        )


if __name__ == "__main__":
    unittest.main()
