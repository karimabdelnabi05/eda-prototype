"""Unit tests for the response formatter."""


from eda.runtime.executor import ExecutionResult
from eda.runtime.formatter import Formatter


class TestFormatter:
    """Tests for Formatter — no LLM calls needed."""

    def setup_method(self):
        self.formatter = Formatter()

    def test_format_boolean_true(self):
        """Formats True as Yes."""
        result = ExecutionResult(success=True, data=True, method_name="check_policy")
        output = self.formatter.format(result, "Can I do this?")
        assert "Yes" in output

    def test_format_boolean_false(self):
        """Formats False as No."""
        result = ExecutionResult(success=True, data=False, method_name="check_policy")
        output = self.formatter.format(result, "Can I do this?")
        assert "No" in output

    def test_format_dict(self):
        """Formats dict data readably."""
        result = ExecutionResult(
            success=True,
            data={"amount": 45000000, "driver": "Cloud Infrastructure"},
            method_name="get_capex",
        )
        output = self.formatter.format(result)
        assert "Amount" in output or "amount" in output.lower()
        assert "Cloud Infrastructure" in output

    def test_format_number(self):
        """Formats large numbers with commas."""
        result = ExecutionResult(
            success=True, data=128500000, method_name="get_revenue"
        )
        output = self.formatter.format(result)
        assert "128,500,000" in output or "128500000" in output

    def test_format_error(self):
        """Formats error results with error marker."""
        result = ExecutionResult(
            success=False,
            error="Method not found",
            method_name="bad_method",
            method_call="obj.bad_method()",
        )
        output = self.formatter.format(result)
        assert "❌" in output
        assert "Method not found" in output

    def test_format_includes_provenance(self):
        """Output includes source method call info."""
        result = ExecutionResult(
            success=True,
            data=42,
            method_name="get_answer",
            method_call="obj.get_answer()",
            execution_time_ms=0.05,
        )
        output = self.formatter.format(result)
        assert "obj.get_answer()" in output

    def test_format_list(self):
        """Formats list data as bullet points."""
        result = ExecutionResult(
            success=True,
            data=["Item A", "Item B", "Item C"],
            method_name="list_items",
        )
        output = self.formatter.format(result)
        assert "Item A" in output
        assert "Item C" in output

    def test_format_list_of_dicts(self):
        """Formats list of dicts as a table."""
        result = ExecutionResult(
            success=True,
            data=[
                {"quarter": "Q1", "amount": 32000000},
                {"quarter": "Q2", "amount": 41000000},
            ],
            method_name="get_capex_table",
        )
        output = self.formatter.format(result)
        assert "Q1" in output
        assert "Q2" in output
