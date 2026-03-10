"""Unit tests for the usage tracker."""

import pytest

from eda.tracker import UsageTracker, LLMCall, reset_tracker, get_tracker


class MockUsageMetadata:
    """Mock Gemini usage metadata for testing."""

    def __init__(self, prompt_tokens: int = 100, candidates_tokens: int = 200):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens


class MockResponse:
    """Mock Gemini API response."""

    def __init__(self, prompt_tokens: int = 100, candidates_tokens: int = 200):
        self.usage_metadata = MockUsageMetadata(prompt_tokens, candidates_tokens)
        self.text = "generated code here"


class TestUsageTracker:
    """Tests for UsageTracker — pure computation, no LLM needed."""

    def setup_method(self):
        self.tracker = UsageTracker()

    def test_record_single_call(self):
        """Records a single LLM call with token counts."""
        response = MockResponse(prompt_tokens=500, candidates_tokens=1000)
        call = self.tracker.record_call(
            call_type="compile",
            model="gemini-2.0-flash",
            response=response,
            latency_ms=2500.0,
        )

        assert call.input_tokens == 500
        assert call.output_tokens == 1000
        assert call.total_tokens == 1500
        assert call.latency_ms == 2500.0
        assert call.call_type == "compile"

    def test_cost_calculation(self):
        """Cost is calculated correctly based on Gemini pricing."""
        response = MockResponse(prompt_tokens=1_000_000, candidates_tokens=1_000_000)
        call = self.tracker.record_call(
            call_type="compile",
            model="gemini-2.0-flash",
            response=response,
            latency_ms=5000.0,
        )

        # gemini-2.0-flash: $0.10/1M input, $0.40/1M output
        assert call.input_cost_usd == pytest.approx(0.10)
        assert call.output_cost_usd == pytest.approx(0.40)
        assert call.total_cost_usd == pytest.approx(0.50)

    def test_aggregate_metrics(self):
        """Metrics aggregate across multiple calls."""
        # Compile call
        self.tracker.record_call(
            call_type="compile",
            model="gemini-2.0-flash",
            response=MockResponse(500, 1000),
            latency_ms=3000.0,
        )
        # Route call
        self.tracker.record_call(
            call_type="route",
            model="gemini-2.0-flash-lite",
            response=MockResponse(100, 50),
            latency_ms=200.0,
        )

        metrics = self.tracker.get_metrics()

        assert metrics.total_input_tokens == 600
        assert metrics.total_output_tokens == 1050
        assert metrics.total_tokens == 1650
        assert metrics.total_compile_time_ms == 3000.0
        assert metrics.total_route_time_ms == 200.0
        assert len(metrics.llm_calls) == 2

    def test_query_accuracy_tracking(self):
        """Tracks query accuracy correctly."""
        self.tracker.record_query_result("Q1", 100, 100, correct=True, latency_ms=0.5)
        self.tracker.record_query_result("Q2", 200, 200, correct=True, latency_ms=0.3)
        self.tracker.record_query_result("Q3", 300, 999, correct=False, latency_ms=0.4)

        metrics = self.tracker.get_metrics()

        assert metrics.query_accuracy == pytest.approx(2 / 3)
        assert metrics.queries_correct == 2
        assert metrics.queries_total == 3

    def test_export_json(self):
        """Export produces valid JSON structure."""
        self.tracker.record_call(
            call_type="compile",
            model="gemini-2.0-flash",
            response=MockResponse(500, 1000),
            latency_ms=3000.0,
        )

        export = self.tracker.export_json()

        assert "summary" in export
        assert "tokens" in export
        assert "cost" in export
        assert "calls" in export
        assert export["tokens"]["total"] == 1500
        assert isinstance(export["cost"]["total_usd"], float)

    def test_reset_tracker(self):
        """Global tracker resets correctly."""
        tracker = reset_tracker()
        tracker.record_call(
            call_type="compile",
            model="gemini-2.0-flash",
            response=MockResponse(100, 100),
            latency_ms=1000.0,
        )

        assert len(get_tracker().calls) == 1

        # Reset
        new_tracker = reset_tracker()
        assert len(new_tracker.calls) == 0

    def test_unknown_model_uses_default_pricing(self):
        """Unknown models get default pricing, don't crash."""
        response = MockResponse(1_000_000, 1_000_000)
        call = self.tracker.record_call(
            call_type="compile",
            model="some-future-model-name",
            response=response,
            latency_ms=1000.0,
        )

        # Should use default pricing ($0.50/1M in, $2.00/1M out)
        assert call.input_cost_usd == pytest.approx(0.50)
        assert call.output_cost_usd == pytest.approx(2.00)

    def test_separate_compile_and_route_costs(self):
        """Compile and route costs are tracked separately."""
        self.tracker.record_call(
            call_type="compile",
            model="gemini-2.0-flash",
            response=MockResponse(1000, 2000),
            latency_ms=3000.0,
        )
        self.tracker.record_call(
            call_type="route",
            model="gemini-2.0-flash-lite",
            response=MockResponse(500, 100),
            latency_ms=200.0,
        )

        metrics = self.tracker.get_metrics()

        assert metrics.compile_cost_usd > 0
        assert metrics.route_cost_usd > 0
        assert metrics.compile_cost_usd != metrics.route_cost_usd
