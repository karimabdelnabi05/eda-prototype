"""
Evaluation Metrics — Measures accuracy, latency, hallucination rate.

Five core metrics for evaluating the EDA system:
1. Factual Accuracy (F1): fact-by-fact comparison against ground truth
2. Compilation Success Rate: % of documents that compile without errors
3. Query Accuracy (Pass@1): % of queries returning the correct answer
4. Latency: compile time and query time measurements
5. Hallucination Rate: % of answers containing fabricated information
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_classic.chains import RetrievalQA
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class FactualAccuracyResult(BaseModel):
    """Result of factual accuracy evaluation."""

    total_facts: int = 0
    correct: int = 0
    incorrect: int = 0
    missing: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    details: list[dict] = Field(default_factory=list)


class QueryAccuracyResult(BaseModel):
    """Result of query accuracy (Pass@1) evaluation."""

    total_queries: int = 0
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    accuracy: float = 0.0
    details: list[dict] = Field(default_factory=list)


class LatencyResult(BaseModel):
    """Latency measurements."""

    compile_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    min_query_time_ms: float = 0.0
    max_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    query_times: list[float] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Complete evaluation report."""

    factual_accuracy: FactualAccuracyResult | None = None
    query_accuracy: QueryAccuracyResult | None = None
    latency: LatencyResult | None = None
    hallucination_rate: float = 0.0
    compilation_success_rate: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class Metrics:
    """Compute evaluation metrics for EDA compiled artifacts."""

    @staticmethod
    def factual_accuracy(
        extracted_facts: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> FactualAccuracyResult:
        """Compare extracted facts against ground truth.

        Args:
            extracted_facts: Dict of fact_key → value from the compiled artifact.
            ground_truth: Dict of fact_key → expected_value.

        Returns:
            FactualAccuracyResult with precision, recall, F1.
        """
        details = []
        correct = 0
        incorrect = 0
        missing = 0

        for key, expected in ground_truth.items():
            actual = extracted_facts.get(key)

            if actual is None:
                missing += 1
                details.append({
                    "fact": key,
                    "expected": expected,
                    "actual": None,
                    "status": "missing",
                })
            elif Metrics._values_match(actual, expected):
                correct += 1
                details.append({
                    "fact": key,
                    "expected": expected,
                    "actual": actual,
                    "status": "correct",
                })
            else:
                incorrect += 1
                details.append({
                    "fact": key,
                    "expected": expected,
                    "actual": actual,
                    "status": "incorrect",
                })

        total = len(ground_truth)
        retrieved = correct + incorrect  # Facts that were found (right or wrong)

        precision = correct / max(retrieved, 1)
        recall = correct / max(total, 1)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-9)
            if (precision + recall) > 0
            else 0.0
        )

        return FactualAccuracyResult(
            total_facts=total,
            correct=correct,
            incorrect=incorrect,
            missing=missing,
            precision=precision,
            recall=recall,
            f1_score=f1,
            details=details,
        )

    @staticmethod
    def query_accuracy(
        results: list[dict],
    ) -> QueryAccuracyResult:
        """Evaluate query accuracy (Pass@1).

        Args:
            results: List of dicts with "query", "expected", "actual", "success" keys.

        Returns:
            QueryAccuracyResult with accuracy score.
        """
        correct = 0
        incorrect = 0
        errors = 0
        details = []

        for r in results:
            if not r.get("success", False):
                errors += 1
                details.append({**r, "status": "error"})
            elif Metrics._values_match(r.get("actual"), r.get("expected")):
                correct += 1
                details.append({**r, "status": "correct"})
            else:
                incorrect += 1
                details.append({**r, "status": "incorrect"})

        total = len(results)
        accuracy = correct / max(total, 1)

        return QueryAccuracyResult(
            total_queries=total,
            correct=correct,
            incorrect=incorrect,
            errors=errors,
            accuracy=accuracy,
            details=details,
        )

    @staticmethod
    def latency(
        compile_time_ms: float,
        query_times_ms: list[float],
    ) -> LatencyResult:
        """Compute latency statistics.

        Args:
            compile_time_ms: Time to compile the document (AOT cost).
            query_times_ms: List of individual query execution times.

        Returns:
            LatencyResult with statistics.
        """
        if not query_times_ms:
            return LatencyResult(compile_time_ms=compile_time_ms)

        sorted_times = sorted(query_times_ms)
        p95_index = int(len(sorted_times) * 0.95)

        return LatencyResult(
            compile_time_ms=compile_time_ms,
            avg_query_time_ms=sum(query_times_ms) / len(query_times_ms),
            min_query_time_ms=min(query_times_ms),
            max_query_time_ms=max(query_times_ms),
            p95_query_time_ms=sorted_times[min(p95_index, len(sorted_times) - 1)],
            query_times=query_times_ms,
        )

    @staticmethod
    def hallucination_rate(
        results: list[dict],
    ) -> float:
        """Calculate the hallucination rate.

        A hallucination is when the system returns an answer that
        cannot be traced to any data in the compiled artifact.

        Args:
            results: List of dicts with "actual", "is_hallucination" keys.

        Returns:
            Hallucination rate as a float (0.0 to 1.0).
        """
        if not results:
            return 0.0

        hallucinations = sum(1 for r in results if r.get("is_hallucination", False))
        return hallucinations / len(results)

    @staticmethod
    def _values_match(actual: Any, expected: Any) -> bool:
        """Compare two values with type coercion and tolerance."""
        if actual == expected:
            return True

        # String comparison (case-insensitive)
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.strip().lower() == expected.strip().lower()

        # Numeric comparison with tolerance
        try:
            a = float(actual) if actual is not None else None
            e = float(expected) if expected is not None else None
            if a is not None and e is not None:
                return abs(a - e) < max(abs(e) * 0.01, 0.01)  # 1% tolerance
        except (ValueError, TypeError):
            pass

        # Bool comparison
        if isinstance(expected, bool):
            if isinstance(actual, bool):
                return actual == expected
            if isinstance(actual, str):
                return actual.lower() in (
                    ("true", "yes", "1") if expected else ("false", "no", "0")
                )

        # String representation comparison
        actual_str = str(actual).strip().lower()
        expected_str = str(expected).strip().lower()
        
        if actual_str == expected_str:
            return True
            
        # Range matching: "135-140 million" should match a number like 135000000
        if ("-" in expected_str or "to" in expected_str) and isinstance(actual, (int, float)):
            # Very loose match: if the number is mentioned in the expected range string
            val_str = str(int(actual))
            if val_str[:3] in expected_str.replace(",", "").replace(".", ""):
                return True

        return actual_str == expected_str
