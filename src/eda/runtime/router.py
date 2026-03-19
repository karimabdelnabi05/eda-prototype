"""
Query Router — Translates natural language questions into Python method calls.

This is a fast, cheap LLM call that maps user intent to a specific method
on the compiled artifact. Think of it as the "instruction decoder" in a CPU.
"""

from __future__ import annotations

import re

from google import genai

from eda.compiler.prompts import ROUTER_SYSTEM_PROMPT
from eda.config import config


class RoutingResult:
    """Result of routing a natural language query to a function call."""

    def __init__(
        self,
        original_query: str,
        method_call: str,
        method_name: str,
        arguments: dict,
        confidence: float = 1.0,
    ):
        self.original_query = original_query
        self.method_call = method_call  # e.g. "obj.get_capex(quarter='Q3')"
        self.method_name = method_name  # e.g. "get_capex"
        self.arguments = arguments  # e.g. {"quarter": "Q3"}
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"RoutingResult({self.method_call})"


class Router:
    """Routes natural language queries to compiled artifact method calls.

    Uses a fast, cheap LLM to translate user intent into a Python expression.
    Falls back to fuzzy method name matching if LLM routing fails.
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.router.model
        self.client = genai.Client(api_key=config.google_api_key)

    def route(
        self,
        query: str,
        available_methods: list[str],
        method_descriptions: dict[str, str] | None = None,
    ) -> RoutingResult:
        """Translate a natural language query into a method call.

        Args:
            query: The user's natural language question.
            available_methods: List of method names on the compiled artifact.
            method_descriptions: Optional dict of method_name → description.

        Returns:
            RoutingResult with the translated method call.
        """
        # Build method listing for the LLM
        method_listing = self._format_methods(available_methods, method_descriptions)

        user_prompt = (
            f"## Available Methods:\n{method_listing}\n\n"
            f"## User Question:\n{query}\n\n"
            f"Translate this question into a Python method call using `obj.method_name(...)`."
        )

        import time as _time
        from google.genai.errors import APIError
        from eda.tracker import get_tracker

        for attempt in range(3):
            try:
                start = _time.perf_counter()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
                    config={
                        "system_instruction": ROUTER_SYSTEM_PROMPT,
                        "temperature": config.router.temperature,
                    },
                )
                latency_ms = (_time.perf_counter() - start) * 1000

                # Record token usage
                tracker = get_tracker()
                tracker.record_call(
                    call_type="route",
                    model=self.model,
                    response=response,
                    latency_ms=latency_ms,
                )

                method_call = (response.text or "").strip()
                method_call = self._clean_method_call(method_call)

                # Parse the method name and arguments
                method_name, arguments = self._parse_method_call(method_call)

                return RoutingResult(
                    original_query=query,
                    method_call=method_call,
                    method_name=method_name,
                    arguments=arguments,
                )
            except APIError as e:
                import logging
                logging.warning(f"API Error during routing (attempt {attempt+1}/3): {e}. Sleeping for 15s...")
                _time.sleep(15)
            except Exception as e:
                import logging
                logging.error(f"Routing exception: {e}")
                break

        # Fallback: fuzzy match method name from query keywords
        import logging
        logging.warning("Falling back to fuzzy routing...")
        return self._fallback_route(query, available_methods)

    def _format_methods(
        self,
        methods: list[str],
        descriptions: dict[str, str] | None = None,
    ) -> str:
        """Format method listing for the LLM prompt."""
        lines = []
        for method in methods:
            desc = (descriptions or {}).get(method, "")
            if desc:
                lines.append(f"- `obj.{method}(...)` — {desc}")
            else:
                lines.append(f"- `obj.{method}(...)`")
        return "\n".join(lines)

    def _clean_method_call(self, call: str) -> str:
        """Clean LLM output to ensure it's a valid method call."""
        # Remove markdown fences
        call = re.sub(r"```(?:python)?\s*", "", call)
        call = call.replace("```", "").strip()
        # Remove any leading text before obj.
        if "obj." in call:
            call = call[call.index("obj."):]
        # Remove trailing text after the closing parenthesis
        paren_depth = 0
        for i, ch in enumerate(call):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    call = call[:i + 1]
                    break
        return call.strip()

    def _parse_method_call(self, call: str) -> tuple[str, dict]:
        """Extract method name and arguments from a method call string."""
        # Match: obj.method_name(arg1="val", arg2=123)
        match = re.match(r'obj\.(\w+)\((.*)\)', call, re.DOTALL)
        if not match:
            return "get_summary", {}

        method_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse keyword arguments
        arguments = {}
        if args_str:
            # Use a simple regex to extract key=value pairs
            for arg_match in re.finditer(
                r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\d.]+|True|False|None)',
                args_str,
            ):
                key = arg_match.group(1)
                value = arg_match.group(2)
                # Convert value types
                if value.startswith(("'", '"')):
                    arguments[key] = value.strip("'\"")
                elif value in ("True", "False"):
                    arguments[key] = value == "True"
                elif value == "None":
                    arguments[key] = None
                elif "." in value:
                    arguments[key] = float(value)
                else:
                    try:
                        arguments[key] = int(value)
                    except ValueError:
                        arguments[key] = value

        return method_name, arguments

    def _fallback_route(self, query: str, methods: list[str]) -> RoutingResult:
        """Fallback: fuzzy match method name from query keywords."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        best_match = "get_summary"
        best_score = 0

        for method in methods:
            method_words = set(re.findall(r'\w+', method.lower()))
            score = len(query_words & method_words)
            if score > best_score:
                best_score = score
                best_match = method

        return RoutingResult(
            original_query=query,
            method_call=f"obj.{best_match}()",
            method_name=best_match,
            arguments={},
            confidence=0.5,
        )
