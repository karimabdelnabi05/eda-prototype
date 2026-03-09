"""
Response Formatter — Converts raw execution results into user-friendly responses.

Takes the deterministic output from code execution and wraps it in
natural language for delivery to the end user.
"""

from __future__ import annotations

import json

from eda.runtime.executor import ExecutionResult


class Formatter:
    """Format execution results into natural language responses.

    Uses template-based formatting by default (no LLM needed).
    Can optionally use an LLM for richer natural language responses.
    """

    def format(
        self,
        result: ExecutionResult,
        original_query: str = "",
        use_llm: bool = False,
    ) -> str:
        """Format an execution result into a user-friendly response.

        Args:
            result: The execution result to format.
            original_query: The original user question.
            use_llm: Whether to use LLM for richer formatting (default: template-based).

        Returns:
            Formatted string response.
        """
        if not result.success:
            return self._format_error(result)

        if use_llm:
            return self._format_with_llm(result, original_query)

        return self._format_template(result, original_query)

    def _format_template(self, result: ExecutionResult, query: str) -> str:
        """Template-based formatting — fast, no API calls."""
        data = result.data
        parts = []

        # Format based on return type
        if isinstance(data, bool):
            answer = "**Yes**" if data else "**No**"
            parts.append(answer)

        elif isinstance(data, dict):
            if "error" in data:
                parts.append(f"⚠️ {data['error']}")
            else:
                parts.append(self._format_dict(data))

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                parts.append(self._format_table(data))
            else:
                for item in data:
                    parts.append(f"• {item}")

        elif isinstance(data, (int, float)):
            parts.append(self._format_number(data))

        elif isinstance(data, str):
            parts.append(data)

        else:
            parts.append(str(data))

        # Add provenance info
        parts.append(
            f"\n_Source: `{result.method_call}` "
            f"({result.execution_time_ms:.2f}ms)_"
        )

        return "\n".join(parts)

    def _format_error(self, result: ExecutionResult) -> str:
        """Format an error result."""
        return (
            f"❌ Could not answer this query.\n"
            f"**Error:** {result.error}\n"
            f"**Method attempted:** `{result.method_call}`"
        )

    def _format_dict(self, data: dict, indent: int = 0) -> str:
        """Format a dictionary into readable text."""
        lines = []
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}**{self._humanize_key(key)}:**")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}**{self._humanize_key(key)}:** {', '.join(str(v) for v in value)}")
            elif isinstance(value, (int, float)):
                lines.append(f"{prefix}**{self._humanize_key(key)}:** {self._format_number(value)}")
            else:
                lines.append(f"{prefix}**{self._humanize_key(key)}:** {value}")
        return "\n".join(lines)

    def _format_table(self, rows: list[dict]) -> str:
        """Format a list of dicts as a markdown table."""
        if not rows:
            return "No data."

        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(self._humanize_key(h) for h in headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_number(self, value: int | float) -> str:
        """Format a number with appropriate formatting."""
        if isinstance(value, float):
            if value < 1:
                return f"{value:.2%}"  # Looks like a percentage
            return f"{value:,.2f}"
        if isinstance(value, int) and abs(value) >= 1_000_000:
            return f"${value:,.0f}" if value > 0 else f"{value:,.0f}"
        return f"{value:,}"

    def _humanize_key(self, key: str) -> str:
        """Convert snake_case or camelCase to human-readable."""
        # snake_case → Title Case
        return key.replace("_", " ").replace("-", " ").title()

    def _format_with_llm(self, result: ExecutionResult, query: str) -> str:
        """Use LLM for richer natural language formatting."""
        from google import genai

        from eda.config import config

        client = genai.Client(api_key=config.google_api_key)

        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"The system returned this data: {json.dumps(result.data, default=str)}\n\n"
            f"Write a clear, concise 1-2 sentence answer based on this data. "
            f"Be direct and factual. Do not add information not in the data."
        )

        response = client.models.generate_content(
            model=config.router.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.0},
        )

        answer = (response.text or "").strip()
        answer += f"\n\n_Source: `{result.method_call}` ({result.execution_time_ms:.2f}ms)_"
        return answer
