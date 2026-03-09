"""
Integration test — End-to-end compile + query pipeline.
Requires LLM API key. Marked as slow.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.slow
class TestCompileAndQuery:
    """End-to-end integration tests — requires API key."""

    def test_full_pipeline(self, sample_financial_text):
        """Full pipeline: parse → compile → validate → query."""
        from eda.compiler.parser import DocumentParser
        from eda.compiler.synthesizer import Synthesizer
        from eda.compiler.validator import Validator
        from eda.runtime.executor import Executor

        parser = DocumentParser()
        synthesizer = Synthesizer()
        validator = Validator()
        executor = Executor()

        # Phase 1: Parse + Compile
        doc = parser.parse_text(sample_financial_text, "Financial Report Q3 2024")
        compilation = synthesizer.compile(doc, "FinancialReportQ32024")

        assert compilation.source_code != ""
        assert "class FinancialReportQ32024" in compilation.source_code

        # Validate
        validation = validator.validate(
            compilation.source_code,
            "FinancialReportQ32024",
            synthesizer=synthesizer,
        )
        assert validation.success, f"Validation failed: {validation.errors}"
        assert len(validation.methods_found) >= 3

        # Phase 3: Load + Query
        instance = executor.load_artifact(validation.source_code, "FinancialReportQ32024")
        assert len(executor.get_available_methods(instance)) >= 3
        assert isinstance(executor.get_method_descriptions(instance), dict)

        # Query: get summary
        summary_result = executor.execute(instance, "get_summary")
        assert summary_result.success
        assert summary_result.execution_time_ms < 10  # Sub-millisecond

    @pytest.mark.slow
    def test_compilation_success_rate(self, sample_financial_text):
        """Compilation should succeed on first attempt or after retries."""
        from eda.compiler.parser import DocumentParser
        from eda.compiler.synthesizer import Synthesizer
        from eda.compiler.validator import Validator

        parser = DocumentParser()
        synthesizer = Synthesizer()
        validator = Validator(max_retries=3)

        doc = parser.parse_text(sample_financial_text, "Financial Report")
        compilation = synthesizer.compile(doc, "TestCompilation")
        validation = validator.validate(
            compilation.source_code,
            "TestCompilation",
            synthesizer=synthesizer,
        )

        assert validation.success, f"Failed after retries: {validation.errors}"
