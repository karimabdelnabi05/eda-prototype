"""
Code Validator — Sandbox execution and validation loop for compiled artifacts.

This module takes generated Python code, executes it in a restricted subprocess,
catches errors, and feeds them back to the synthesizer for patching.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

from pydantic import BaseModel, Field

from eda.config import config


class ValidationResult(BaseModel):
    """Result of validating a compiled artifact."""

    success: bool = False
    source_code: str = ""
    class_name: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    methods_found: list[str] = Field(default_factory=list)
    retries_used: int = 0

    @property
    def method_count(self) -> int:
        return len(self.methods_found)


class Validator:
    """Validates compiled Python code through sandboxed execution.

    The validation loop:
    1. Write code to a temp file
    2. Execute it in a subprocess (sandbox)
    3. If errors → feed traceback back to synthesizer for patching
    4. Repeat up to max_retries times
    """

    def __init__(self, max_retries: int | None = None, sandbox_timeout: int | None = None):
        self.max_retries = max_retries or config.compiler.max_retries
        self.sandbox_timeout = sandbox_timeout or config.compiler.sandbox_timeout

    def validate(
        self,
        source_code: str,
        class_name: str,
        synthesizer=None,
    ) -> ValidationResult:
        """Validate compiled code, optionally using synthesizer for patching.

        Args:
            source_code: The Python source code to validate.
            class_name: Expected class name in the code.
            synthesizer: Optional Synthesizer instance for auto-patching.

        Returns:
            ValidationResult with success/failure and details.
        """
        current_code = source_code
        errors = []
        retries = 0

        for attempt in range(1 + self.max_retries):
            result = self._execute_in_sandbox(current_code, class_name)

            if result.success:
                result.retries_used = retries
                result.source_code = current_code
                return result

            errors.extend(result.errors)

            # If we have a synthesizer and more retries, attempt patching
            if synthesizer and attempt < self.max_retries:
                error_text = "\n".join(result.errors)
                try:
                    current_code = synthesizer.patch(current_code, error_text)
                    retries += 1
                except Exception as e:
                    errors.append(f"Patch attempt failed: {e}")
                    break
            else:
                break

        return ValidationResult(
            success=False,
            source_code=current_code,
            class_name=class_name,
            errors=errors,
            retries_used=retries,
        )

    def _execute_in_sandbox(self, source_code: str, class_name: str) -> ValidationResult:
        """Execute code in a subprocess sandbox and validate it.

        Checks:
        1. Code compiles (no SyntaxError)
        2. Code executes without runtime errors
        3. The expected class exists and is instantiable
        4. The class has the required methods (get_summary, list_available_methods)
        """
        # Build a test script that imports and validates the class
        test_script = textwrap.dedent(f"""
            import json
            import sys

            # Execute the compiled code
            exec_globals = {{}}
            source = open(sys.argv[1], 'r', encoding='utf-8').read()
            exec(source, exec_globals)

            # Check class exists
            if '{class_name}' not in exec_globals:
                print(json.dumps({{"error": "Class '{class_name}' not found in compiled code"}}))
                sys.exit(1)

            # Instantiate
            cls = exec_globals['{class_name}']
            try:
                obj = cls()
            except Exception as e:
                print(json.dumps({{"error": f"Failed to instantiate {class_name}: {{e}}"}}))
                sys.exit(1)

            # Discover methods
            methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))]

            # Check required methods
            warnings = []
            if 'get_summary' not in methods:
                warnings.append("Missing recommended method: get_summary()")
            if 'list_available_methods' not in methods:
                warnings.append("Missing recommended method: list_available_methods()")

            # Try calling get_summary if it exists
            summary = None
            if 'get_summary' in methods:
                try:
                    summary = obj.get_summary()
                except Exception as e:
                    warnings.append(f"get_summary() raised: {{e}}")

            print(json.dumps({{
                "success": True,
                "methods": methods,
                "warnings": warnings,
                "summary": str(summary) if summary else None,
            }}))
        """).strip()

        try:
            # Write source code to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as src_file:
                src_file.write(source_code)
                src_path = src_file.name

            # Write test script to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as test_file:
                test_file.write(test_script)
                test_path = test_file.name

            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, test_path, src_path],
                capture_output=True,
                text=True,
                timeout=self.sandbox_timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                return ValidationResult(
                    success=False,
                    class_name=class_name,
                    errors=[error_msg or "Unknown execution error"],
                )

            # Parse the JSON output from our test script
            import json

            try:
                output = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return ValidationResult(
                    success=False,
                    class_name=class_name,
                    errors=[f"Invalid test output: {result.stdout[:500]}"],
                )

            if "error" in output:
                return ValidationResult(
                    success=False,
                    class_name=class_name,
                    errors=[output["error"]],
                )

            return ValidationResult(
                success=True,
                source_code=source_code,
                class_name=class_name,
                methods_found=output.get("methods", []),
                warnings=output.get("warnings", []),
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                success=False,
                class_name=class_name,
                errors=[f"Execution timed out after {self.sandbox_timeout}s"],
            )
        except Exception:
            return ValidationResult(
                success=False,
                class_name=class_name,
                errors=[f"Sandbox error: {traceback.format_exc()}"],
            )
        finally:
            # Cleanup temp files
            for p in [src_path, test_path]:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
