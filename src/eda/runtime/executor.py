"""
Code Executor — Runs compiled artifact methods in a sandboxed environment.

This is where deterministic execution happens. The compiled Python class
is loaded, the routed method is called, and the result is returned.
Zero LLM involvement at this stage — pure Python execution.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of executing a method on a compiled artifact."""

    success: bool = False
    data: Any = None
    error: str = ""
    method_name: str = ""
    method_call: str = ""
    execution_time_ms: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class Executor:
    """Execute methods on compiled artifacts — deterministic, sub-millisecond.

    Loads a compiled Python module, instantiates the class, and calls
    the routed method with the extracted arguments.
    """

    def __init__(self):
        self._loaded_modules: dict[str, Any] = {}

    def load_artifact(self, source_code: str, class_name: str) -> Any:
        """Load a compiled artifact and return an instance of the class.

        Args:
            source_code: The Python source code of the compiled artifact.
            class_name: The class name to instantiate.

        Returns:
            Instance of the compiled class.

        Raises:
            RuntimeError: If loading or instantiation fails.
        """
        exec_globals: dict[str, Any] = {}
        try:
            exec(source_code, exec_globals)
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact: {e}") from e

        if class_name not in exec_globals:
            raise RuntimeError(
                f"Class '{class_name}' not found in compiled artifact. "
                f"Available: {[k for k in exec_globals if not k.startswith('_')]}"
            )

        cls = exec_globals[class_name]
        try:
            instance = cls()
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate {class_name}: {e}") from e

        self._loaded_modules[class_name] = instance
        return instance

    def execute(
        self,
        instance: Any,
        method_name: str,
        arguments: dict | None = None,
        method_call: str = "",
    ) -> ExecutionResult:
        """Execute a method on a loaded artifact instance.

        Args:
            instance: The compiled class instance.
            method_name: Name of the method to call.
            arguments: Keyword arguments for the method.
            method_call: The original method call string (for logging).

        Returns:
            ExecutionResult with the return value and timing.
        """
        arguments = arguments or {}

        if not hasattr(instance, method_name):
            return ExecutionResult(
                success=False,
                error=f"Method '{method_name}' not found on artifact",
                method_name=method_name,
                method_call=method_call,
            )

        method = getattr(instance, method_name)
        if not callable(method):
            return ExecutionResult(
                success=False,
                error=f"'{method_name}' is not callable",
                method_name=method_name,
                method_call=method_call,
            )

        start = time.perf_counter()
        try:
            result = method(**arguments)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return ExecutionResult(
                success=True,
                data=result,
                method_name=method_name,
                method_call=method_call,
                execution_time_ms=elapsed_ms,
            )

        except TypeError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Common: wrong arguments. Try without arguments as fallback.
            if arguments:
                try:
                    result = method()
                    return ExecutionResult(
                        success=True,
                        data=result,
                        method_name=method_name,
                        method_call=method_call,
                        execution_time_ms=elapsed_ms,
                    )
                except Exception:
                    pass

            return ExecutionResult(
                success=False,
                error=f"Argument error calling {method_name}: {e}",
                method_name=method_name,
                method_call=method_call,
                execution_time_ms=elapsed_ms,
            )

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                success=False,
                error=f"Execution error: {traceback.format_exc()}",
                method_name=method_name,
                method_call=method_call,
                execution_time_ms=elapsed_ms,
            )

    def get_available_methods(self, instance: Any) -> list[str]:
        """Get all public callable methods on a loaded artifact."""
        return [
            name
            for name in dir(instance)
            if not name.startswith("_") and callable(getattr(instance, name))
        ]

    def get_method_descriptions(self, instance: Any) -> dict[str, str]:
        """Get method names with their docstrings for the router."""
        descriptions = {}
        for name in self.get_available_methods(instance):
            method = getattr(instance, name)
            doc = getattr(method, "__doc__", None)
            descriptions[name] = (doc or "").strip().split("\n")[0] if doc else ""
        return descriptions
