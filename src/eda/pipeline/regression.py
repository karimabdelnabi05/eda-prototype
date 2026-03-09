"""
Regression Tester — Validates that document recompilation didn't break things.

Runs a suite of queries against both old and new artifact versions
and flags any semantic regressions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from eda.compiler.artifacts import ArtifactManager
from eda.runtime.executor import Executor


class RegressionCase(BaseModel):
    """A single regression test case."""

    query_method: str
    query_args: dict = Field(default_factory=dict)
    old_result: str = ""
    new_result: str = ""
    passed: bool = True
    regression_type: str = ""  # "value_changed", "method_missing", "error"


class RegressionReport(BaseModel):
    """Report from running regression tests."""

    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    regressions: list[RegressionCase] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total_cases, 1)

    @property
    def has_regressions(self) -> bool:
        return self.failed > 0


class RegressionTester:
    """Run regression tests comparing old vs new artifact versions."""

    def __init__(self):
        self.artifacts = ArtifactManager()
        self.executor = Executor()

    def run_regression(
        self,
        artifact_id: str,
        test_queries: list[dict],
        old_version: str = "v1",
        new_version: str = "latest",
    ) -> RegressionReport:
        """Run regression tests comparing two artifact versions.

        Args:
            artifact_id: The artifact to test.
            test_queries: List of dicts with "method" and optional "args" keys.
            old_version: Version string for the baseline (e.g., "v1").
            new_version: Version string for the new version (e.g., "latest").

        Returns:
            RegressionReport with pass/fail details.
        """
        # Load both versions
        old_code, old_meta = self.artifacts.load(artifact_id, old_version)
        new_code, new_meta = self.artifacts.load(artifact_id, new_version)

        old_instance = self.executor.load_artifact(old_code, old_meta.class_name)
        new_instance = self.executor.load_artifact(new_code, new_meta.class_name)

        cases = []
        for query in test_queries:
            method = query["method"]
            args = query.get("args", {})

            case = RegressionCase(
                query_method=method,
                query_args=args,
            )

            # Execute on old version
            old_result = self.executor.execute(old_instance, method, args)
            case.old_result = str(old_result.data) if old_result.success else f"ERROR: {old_result.error}"

            # Execute on new version
            new_result = self.executor.execute(new_instance, method, args)
            case.new_result = str(new_result.data) if new_result.success else f"ERROR: {new_result.error}"

            # Compare
            if not new_result.success and old_result.success:
                case.passed = False
                case.regression_type = "method_broken"
            elif old_result.success and new_result.success:
                if str(old_result.data) != str(new_result.data):
                    case.passed = False
                    case.regression_type = "value_changed"

            cases.append(case)

        passed = sum(1 for c in cases if c.passed)
        failed = sum(1 for c in cases if not c.passed)

        return RegressionReport(
            total_cases=len(cases),
            passed=passed,
            failed=failed,
            regressions=[c for c in cases if not c.passed],
        )
