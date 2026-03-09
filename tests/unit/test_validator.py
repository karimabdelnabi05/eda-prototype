"""Unit tests for the code validator."""


from eda.compiler.validator import Validator


class TestValidator:
    """Tests for Validator — uses subprocess sandbox, no LLM needed."""

    def setup_method(self):
        self.validator = Validator(max_retries=0, sandbox_timeout=10)

    def test_valid_code_passes(self, sample_compiled_class):
        """Valid Python code passes validation."""
        result = self.validator.validate(sample_compiled_class, "TestFinancialReport")

        assert result.success is True
        assert "get_total_revenue" in result.methods_found
        assert "get_summary" in result.methods_found
        assert result.retries_used == 0

    def test_syntax_error_fails(self):
        """Code with syntax errors fails validation."""
        bad_code = "class Broken:\n    def oops(self)\n        return 42\n"
        result = self.validator.validate(bad_code, "Broken")

        assert result.success is False
        assert len(result.errors) > 0

    def test_missing_class_fails(self):
        """Code without the expected class fails validation."""
        code = "class WrongName:\n    pass\n"
        result = self.validator.validate(code, "ExpectedClass")

        assert result.success is False
        assert any("ExpectedClass" in e for e in result.errors)

    def test_runtime_error_fails(self):
        """Code that crashes at instantiation fails validation."""
        code = (
            "class CrashOnInit:\n"
            "    def __init__(self):\n"
            "        raise ValueError('Intentional crash')\n"
        )
        result = self.validator.validate(code, "CrashOnInit")

        assert result.success is False

    def test_method_discovery(self, sample_compiled_class):
        """Validator discovers all public methods."""
        result = self.validator.validate(sample_compiled_class, "TestFinancialReport")

        assert result.success is True
        assert "check_travel_compliance" in result.methods_found
        assert "get_capex" in result.methods_found
        assert result.method_count >= 5

    def test_timeout_handling(self):
        """Code that runs too long fails with timeout."""
        slow_code = (
            "import time\n"
            "class SlowClass:\n"
            "    def __init__(self):\n"
            "        time.sleep(100)\n"
        )
        validator = Validator(max_retries=0, sandbox_timeout=2)
        result = validator.validate(slow_code, "SlowClass")

        assert result.success is False
        assert any("timed out" in e.lower() for e in result.errors)
