"""
Document Differ — Detects changes between document versions.

Compares two versions of a source document to identify which sections
changed, enabling incremental recompilation.
"""

from __future__ import annotations

import difflib

from pydantic import BaseModel, Field

from eda.compiler.parser import DocumentContent, DocumentParser


class SectionDiff(BaseModel):
    """A change detected in a specific section."""

    section_heading: str = ""
    change_type: str = ""  # "added", "removed", "modified"
    old_content: str = ""
    new_content: str = ""
    similarity_ratio: float = 0.0


class DocumentDiff(BaseModel):
    """Complete diff between two document versions."""

    old_hash: str = ""
    new_hash: str = ""
    has_changes: bool = False
    section_diffs: list[SectionDiff] = Field(default_factory=list)
    full_text_diff: str = ""
    change_summary: str = ""

    @property
    def changed_section_count(self) -> int:
        return len(self.section_diffs)


class Differ:
    """Compare two document versions and identify changes."""

    def __init__(self):
        self.parser = DocumentParser()

    def diff_documents(
        self,
        old_doc: DocumentContent,
        new_doc: DocumentContent,
    ) -> DocumentDiff:
        """Compare two parsed documents and return their differences.

        Args:
            old_doc: The previous version of the document.
            new_doc: The updated version of the document.

        Returns:
            DocumentDiff with section-level change details.
        """
        if old_doc.source_hash == new_doc.source_hash:
            return DocumentDiff(
                old_hash=old_doc.source_hash,
                new_hash=new_doc.source_hash,
                has_changes=False,
                change_summary="No changes detected.",
            )

        # Full text diff
        text_diff = difflib.unified_diff(
            old_doc.full_text.splitlines(keepends=True),
            new_doc.full_text.splitlines(keepends=True),
            fromfile="old",
            tofile="new",
        )
        full_text_diff = "".join(text_diff)

        # Section-level diff
        section_diffs = self._diff_sections(old_doc.sections, new_doc.sections)

        # Summary
        added = sum(1 for d in section_diffs if d.change_type == "added")
        removed = sum(1 for d in section_diffs if d.change_type == "removed")
        modified = sum(1 for d in section_diffs if d.change_type == "modified")
        summary_parts = []
        if added:
            summary_parts.append(f"{added} section(s) added")
        if removed:
            summary_parts.append(f"{removed} section(s) removed")
        if modified:
            summary_parts.append(f"{modified} section(s) modified")
        change_summary = ", ".join(summary_parts) or "Minor text changes"

        return DocumentDiff(
            old_hash=old_doc.source_hash,
            new_hash=new_doc.source_hash,
            has_changes=True,
            section_diffs=section_diffs,
            full_text_diff=full_text_diff,
            change_summary=change_summary,
        )

    def diff_files(self, old_path: str, new_path: str) -> DocumentDiff:
        """Compare two document files directly."""
        old_doc = self.parser.parse(old_path)
        new_doc = self.parser.parse(new_path)
        return self.diff_documents(old_doc, new_doc)

    def _diff_sections(self, old_sections, new_sections) -> list[SectionDiff]:
        """Compare sections between old and new documents."""
        diffs = []
        old_by_heading = {s.heading: s for s in old_sections if s.heading}
        new_by_heading = {s.heading: s for s in new_sections if s.heading}

        all_headings = set(old_by_heading.keys()) | set(new_by_heading.keys())

        for heading in sorted(all_headings):
            old_section = old_by_heading.get(heading)
            new_section = new_by_heading.get(heading)

            if old_section and not new_section:
                diffs.append(SectionDiff(
                    section_heading=heading,
                    change_type="removed",
                    old_content=old_section.content,
                ))
            elif new_section and not old_section:
                diffs.append(SectionDiff(
                    section_heading=heading,
                    change_type="added",
                    new_content=new_section.content,
                ))
            elif old_section and new_section:
                ratio = difflib.SequenceMatcher(
                    None, old_section.content, new_section.content
                ).ratio()
                if ratio < 0.95:  # More than 5% change
                    diffs.append(SectionDiff(
                        section_heading=heading,
                        change_type="modified",
                        old_content=old_section.content,
                        new_content=new_section.content,
                        similarity_ratio=ratio,
                    ))

        return diffs
