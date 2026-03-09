"""
Document Parser — Extracts text and tables from PDFs and text files.

This is the first stage of the EDA compiler pipeline. It takes raw documents
and produces structured DocumentContent that the synthesizer can compile.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TableData(BaseModel):
    """A single extracted table from a document."""

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    page_number: int | None = None
    caption: str = ""

    def to_markdown(self) -> str:
        """Convert table to markdown format for LLM consumption."""
        if not self.headers and not self.rows:
            return ""
        lines = []
        if self.caption:
            lines.append(f"**Table: {self.caption}**\n")
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


class DocumentSection(BaseModel):
    """A section of a document with its heading and content."""

    heading: str = ""
    content: str = ""
    level: int = 1  # Heading level (1 = top-level)
    tables: list[TableData] = Field(default_factory=list)


class DocumentContent(BaseModel):
    """Structured representation of a parsed document."""

    source_path: str = ""
    source_hash: str = ""  # SHA-256 of the raw content for change detection
    title: str = ""
    full_text: str = ""
    sections: list[DocumentSection] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    total_pages: int = 0
    char_count: int = 0

    def to_compilation_text(self) -> str:
        """Produce the full text representation for the LLM compiler."""
        parts = []
        if self.title:
            parts.append(f"# {self.title}\n")
        if self.sections:
            for section in self.sections:
                prefix = "#" * (section.level + 1)
                if section.heading:
                    parts.append(f"{prefix} {section.heading}\n")
                if section.content:
                    parts.append(section.content)
                for table in section.tables:
                    parts.append(table.to_markdown())
        else:
            parts.append(self.full_text)
        for table in self.tables:
            parts.append(table.to_markdown())
        return "\n\n".join(parts)


class DocumentParser:
    """Parse documents into structured DocumentContent.

    Supports:
    - PDF files (via PyMuPDF)
    - Plain text files (.txt)
    - Markdown files (.md)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}

    def parse(self, path: str | Path) -> DocumentContent:
        """Parse a document file and return structured content.

        Args:
            path: Path to the document file.

        Returns:
            DocumentContent with extracted text, sections, and tables.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is not supported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".pdf":
            return self._parse_pdf(path)
        else:
            return self._parse_text(path)

    def parse_text(self, text: str, title: str = "Untitled") -> DocumentContent:
        """Parse raw text string directly (no file needed).

        Args:
            text: The raw text content.
            title: Optional title for the document.

        Returns:
            DocumentContent with extracted sections.
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        sections = self._extract_sections_from_text(text)

        return DocumentContent(
            source_path="<raw_text>",
            source_hash=content_hash,
            title=title,
            full_text=text,
            sections=sections,
            char_count=len(text),
        )

    def _parse_pdf(self, path: Path) -> DocumentContent:
        """Extract text and tables from a PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install it with: pip install PyMuPDF"
            )

        doc = fitz.open(str(path))
        all_text_parts = []
        all_tables = []

        for page_num, page in enumerate(doc, 1):
            # Extract text
            text = page.get_text("text")
            if text.strip():
                all_text_parts.append(text)

            # Extract tables (PyMuPDF has built-in table finder)
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    table_data = tab.extract()
                    if table_data and len(table_data) > 1:
                        headers = [str(h) for h in table_data[0]]
                        rows = [[str(cell) for cell in row] for row in table_data[1:]]
                        all_tables.append(
                            TableData(
                                headers=headers,
                                rows=rows,
                                page_number=page_num,
                            )
                        )
            except Exception:
                pass  # Table extraction is best-effort

        full_text = "\n\n".join(all_text_parts)
        raw_bytes = path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        sections = self._extract_sections_from_text(full_text)

        doc.close()

        return DocumentContent(
            source_path=str(path),
            source_hash=content_hash,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            full_text=full_text,
            sections=sections,
            tables=all_tables,
            total_pages=len(doc) if hasattr(doc, "__len__") else 0,
            char_count=len(full_text),
        )

    def _parse_text(self, path: Path) -> DocumentContent:
        """Parse a plain text or markdown file."""
        text = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        sections = self._extract_sections_from_text(text)

        return DocumentContent(
            source_path=str(path),
            source_hash=content_hash,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            full_text=text,
            sections=sections,
            char_count=len(text),
        )

    def _extract_sections_from_text(self, text: str) -> list[DocumentSection]:
        """Split text into sections based on markdown-style headings or paragraph breaks."""
        lines = text.split("\n")
        sections: list[DocumentSection] = []
        current_heading = ""
        current_level = 1
        current_content: list[str] = []

        for line in lines:
            stripped = line.strip()
            # Detect markdown headings
            if stripped.startswith("#"):
                # Save previous section
                if current_content or current_heading:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            content="\n".join(current_content).strip(),
                            level=current_level,
                        )
                    )
                # Parse new heading
                level = len(stripped) - len(stripped.lstrip("#"))
                current_heading = stripped.lstrip("#").strip()
                current_level = level
                current_content = []
            # Detect ALL-CAPS headings (common in PDFs and reports)
            elif stripped.isupper() and len(stripped) > 3 and len(stripped) < 100:
                if current_content or current_heading:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            content="\n".join(current_content).strip(),
                            level=current_level,
                        )
                    )
                current_heading = stripped.title()
                current_level = 2
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content or current_heading:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    content="\n".join(current_content).strip(),
                    level=current_level,
                )
            )

        return sections
