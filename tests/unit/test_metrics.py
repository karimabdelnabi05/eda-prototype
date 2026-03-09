"""Unit tests for evaluation metrics."""

import pytest

from eda.evaluation.metrics import Metrics


class TestMetrics:
    """Tests for evaluation metrics — pure computation, no LLM needed."""

    def test_factual_accuracy_perfect(self):
        """100% accuracy when all facts match."""
        facts = {"revenue": 128500000, "employees": 2847}
        ground_truth = {"revenue": 128500000, "employees": 2847}

        result = Metrics.factual_accuracy(facts, ground_truth)
        assert result.f1_score == 1.0
        assert result.correct == 2
        assert result.missing == 0

    def test_factual_accuracy_partial(self):
        """Partial accuracy with some correct and some missing."""
        facts = {"revenue": 128500000}
        ground_truth = {"revenue": 128500000, "employees": 2847}

        result = Metrics.factual_accuracy(facts, ground_truth)
        assert result.correct == 1
        assert result.missing == 1
        assert result.recall == 0.5

    def test_factual_accuracy_incorrect(self):
        """Incorrect values are counted."""
        facts = {"revenue": 999999}
        ground_truth = {"revenue": 128500000}

        result = Metrics.factual_accuracy(facts, ground_truth)
        assert result.incorrect == 1
        assert result.f1_score == 0.0

    def test_query_accuracy(self):
        """Query accuracy with mixed results."""
        results = [
            {"query": "Q1", "expected": 100, "actual": 100, "success": True},
            {"query": "Q2", "expected": 200, "actual": 200, "success": True},
            {"query": "Q3", "expected": 300, "actual": 999, "success": True},
            {"query": "Q4", "expected": 400, "actual": None, "success": False},
        ]

        result = Metrics.query_accuracy(results)
        assert result.accuracy == 0.5  # 2/4 correct
        assert result.correct == 2
        assert result.incorrect == 1
        assert result.errors == 1

    def test_latency_stats(self):
        """Latency statistics computed correctly."""
        result = Metrics.latency(
            compile_time_ms=5000.0,
            query_times_ms=[0.1, 0.2, 0.15, 0.3, 0.05],
        )

        assert result.compile_time_ms == 5000.0
        assert result.min_query_time_ms == 0.05
        assert result.max_query_time_ms == 0.3
        assert result.avg_query_time_ms == pytest.approx(0.16, abs=0.01)

    def test_hallucination_rate(self):
        """Hallucination rate computed correctly."""
        results = [
            {"actual": "fact", "is_hallucination": False},
            {"actual": "fact", "is_hallucination": False},
            {"actual": "made up", "is_hallucination": True},
        ]

        rate = Metrics.hallucination_rate(results)
        assert rate == pytest.approx(1 / 3)

    def test_values_match_string_case_insensitive(self):
        """String matching is case-insensitive."""
        assert Metrics._values_match("Hello", "hello") is True

    def test_values_match_numeric_tolerance(self):
        """Numeric matching has 1% tolerance."""
        assert Metrics._values_match(100.5, 100) is True
        assert Metrics._values_match(200, 100) is False

    def test_values_match_bool(self):
        """Boolean matching works with string representations."""
        assert Metrics._values_match(True, True) is True
        assert Metrics._values_match("yes", True) is True
        assert Metrics._values_match(False, True) is False
