"""
Evaluation Dataset — Load and manage test fixtures for benchmarking.

Each fixture contains a source document and a set of Q&A pairs
with expected answers for evaluating compiled artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single question-answer pair for evaluation."""

    question: str
    expected_answer: Any
    method_hint: str = ""  # Optional: expected method name
    category: str = ""  # e.g., "numerical", "policy", "comparison"
    difficulty: str = "medium"  # easy, medium, hard


class EvalFixture(BaseModel):
    """A complete evaluation fixture: document + Q&A pairs."""

    name: str
    document_path: str = ""
    document_text: str = ""
    qa_pairs: list[QAPair] = Field(default_factory=list)
    ground_truth_facts: dict[str, Any] = Field(default_factory=dict)

    @property
    def query_count(self) -> int:
        return len(self.qa_pairs)


class EvalDataset:
    """Load and manage evaluation fixtures."""

    def __init__(self, fixtures_dir: str | Path | None = None):
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir else None

    def load_fixture(self, name: str) -> EvalFixture:
        """Load a fixture by name from the fixtures directory.

        Expects:
        - {name}.txt or {name}.pdf — the source document
        - ground_truth_{name}.json — the Q&A pairs and facts
        """
        if not self.fixtures_dir:
            raise ValueError("No fixtures directory configured.")

        # Find the document file
        doc_path = None
        for ext in [".txt", ".md", ".pdf"]:
            candidate = self.fixtures_dir / f"{name}{ext}"
            if candidate.exists():
                doc_path = candidate
                break

        # Load ground truth
        gt_path = self.fixtures_dir / f"ground_truth_{name}.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")

        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

        # Build fixture
        fixture = EvalFixture(
            name=name,
            document_path=str(doc_path) if doc_path else "",
            qa_pairs=[QAPair(**qa) for qa in ground_truth.get("qa_pairs", [])],
            ground_truth_facts=ground_truth.get("facts", {}),
        )

        # Load document text if it's a text file
        if doc_path and doc_path.suffix in (".txt", ".md"):
            fixture.document_text = doc_path.read_text(encoding="utf-8")

        return fixture

    def list_fixtures(self) -> list[str]:
        """List available fixture names."""
        if not self.fixtures_dir or not self.fixtures_dir.exists():
            return []

        names = set()
        for path in self.fixtures_dir.glob("ground_truth_*.json"):
            name = path.stem.replace("ground_truth_", "")
            names.add(name)
        return sorted(names)

    @staticmethod
    def create_fixture_from_inline(
        name: str,
        document_text: str,
        qa_pairs: list[dict],
        facts: dict | None = None,
    ) -> EvalFixture:
        """Create a fixture programmatically (for testing)."""
        return EvalFixture(
            name=name,
            document_text=document_text,
            qa_pairs=[QAPair(**qa) for qa in qa_pairs],
            ground_truth_facts=facts or {},
        )
