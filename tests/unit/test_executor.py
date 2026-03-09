"""Unit tests for the code executor."""

import pytest

from eda.runtime.executor import Executor


class TestExecutor:
    """Tests for Executor — uses pre-built compiled class, no LLM needed."""

    def setup_method(self):
        self.executor = Executor()

    def test_load_artifact(self, sample_compiled_class):
        """Executor can load and instantiate a compiled class."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")
        assert instance is not None
        assert hasattr(instance, "get_total_revenue")

    def test_execute_simple_method(self, sample_compiled_class):
        """Executor can call a method with no arguments."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")
        result = self.executor.execute(instance, "get_total_revenue")

        assert result.success is True
        assert result.data == 128500000
        assert result.execution_time_ms < 100  # Should be sub-millisecond

    def test_execute_with_arguments(self, sample_compiled_class):
        """Executor can call a method with keyword arguments."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")
        result = self.executor.execute(
            instance, "get_capex", {"quarter": "Q3"}
        )

        assert result.success is True
        assert result.data["amount"] == 45000000
        assert result.data["driver"] == "Cloud Infrastructure"

    def test_execute_policy_check(self, sample_compiled_class):
        """Executor can run boolean policy checks."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")

        # Standard employee, $1500 flight — should fail
        result = self.executor.execute(
            instance,
            "check_travel_compliance",
            {"role": "Standard", "flight_class": "Business", "cost": 1500.0},
        )
        assert result.success is True
        assert result.data is False

        # Executive, $4000 flight — should pass
        result = self.executor.execute(
            instance,
            "check_travel_compliance",
            {"role": "Executive", "flight_class": "Business", "cost": 4000.0},
        )
        assert result.success is True
        assert result.data is True

    def test_execute_nonexistent_method(self, sample_compiled_class):
        """Executor returns error for non-existent methods."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")
        result = self.executor.execute(instance, "nonexistent_method")

        assert result.success is False
        assert "not found" in result.error

    def test_get_available_methods(self, sample_compiled_class):
        """Executor can list available methods."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")
        methods = self.executor.get_available_methods(instance)

        assert "get_total_revenue" in methods
        assert "get_capex" in methods
        assert "check_travel_compliance" in methods
        assert "get_summary" in methods

    def test_execution_timing(self, sample_compiled_class):
        """Execution time is sub-millisecond for data retrieval."""
        instance = self.executor.load_artifact(sample_compiled_class, "TestFinancialReport")
        result = self.executor.execute(instance, "get_total_revenue")

        assert result.execution_time_ms < 10  # Well under a millisecond

    def test_load_invalid_code_raises(self):
        """Executor raises RuntimeError for invalid code."""
        with pytest.raises(RuntimeError):
            self.executor.load_artifact("this is not python code!!!", "FakeClass")

    def test_load_missing_class_raises(self):
        """Executor raises RuntimeError if class not found in code."""
        code = "class WrongName:\n    pass\n"
        with pytest.raises(RuntimeError, match="not found"):
            self.executor.load_artifact(code, "ExpectedClass")
