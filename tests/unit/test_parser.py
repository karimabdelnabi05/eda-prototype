"""Unit tests for the document parser."""


import pytest

from eda.compiler.parser import DocumentContent, DocumentParser


class TestDocumentParser:
    """Tests for DocumentParser — no LLM calls needed."""

    def setup_method(self):
        self.parser = DocumentParser()

    def test_parse_text_file(self, fixtures_dir):
        """Parser can read a .txt file and extract content."""
        path = fixtures_dir / "sample_financial_report.txt"
        result = self.parser.parse(path)

        assert isinstance(result, DocumentContent)
        assert result.char_count > 0
        assert result.source_hash != ""
        assert len(result.sections) > 0
        assert "Acme" in result.full_text

    def test_parse_raw_text(self):
        """Parser can parse raw text strings."""
        text = "# My Report\n\nThis is the content.\n\n## Section 2\n\nMore content."
        result = self.parser.parse_text(text, title="Test Report")

        assert result.title == "Test Report"
        assert result.char_count == len(text)
        assert len(result.sections) >= 2
        assert result.source_hash != ""

    def test_sections_extracted(self, fixtures_dir):
        """Parser correctly splits document into sections."""
        path = fixtures_dir / "sample_financial_report.txt"
        result = self.parser.parse(path)

        headings = [s.heading for s in result.sections if s.heading]
        assert len(headings) >= 3  # Should find multiple section headings

    def test_compilation_text_output(self):
        """to_compilation_text() produces a string for the LLM."""
        text = "# Title\n\nContent here.\n\n## Section A\n\nMore content."
        result = self.parser.parse_text(text, "Test")

        compilation_text = result.to_compilation_text()
        assert isinstance(compilation_text, str)
        assert len(compilation_text) > 0
        assert "Content" in compilation_text

    def test_unsupported_extension_raises(self, tmp_path):
        """Parser raises ValueError for unsupported file types."""
        bad_file = tmp_path / "report.xlsx"
        bad_file.write_text("data")

        with pytest.raises(ValueError, match="Unsupported file type"):
            self.parser.parse(bad_file)

    def test_missing_file_raises(self):
        """Parser raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/file.txt")

    def test_source_hash_deterministic(self):
        """Same content produces the same hash."""
        text = "Hello, World!"
        r1 = self.parser.parse_text(text)
        r2 = self.parser.parse_text(text)
        assert r1.source_hash == r2.source_hash

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        r1 = self.parser.parse_text("Document A")
        r2 = self.parser.parse_text("Document B")
        assert r1.source_hash != r2.source_hash
