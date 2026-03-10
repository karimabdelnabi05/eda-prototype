"""
Usage Tracker — Tracks token usage, cost, timing, and accuracy across the EDA pipeline.

Every LLM call (compilation, routing, patching) is logged with token counts,
cost estimates, and timing. Provides a full report at the end of each pipeline run.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()

# Gemini pricing per 1M tokens (as of March 2025)
# https://ai.google.dev/pricing
PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},           # per 1M tokens
    "gemini-2.0-flash-lite": {"input": 0.025, "output": 0.10},     # per 1M tokens
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-pro": {"input": 1.25, "output": 10.00},
}

# Fallback for unknown models
DEFAULT_PRICING = {"input": 0.50, "output": 2.00}


class LLMCall(BaseModel):
    """Record of a single LLM API call."""

    timestamp: str = ""
    call_type: str = ""  # "compile", "compile_section", "merge", "patch", "route", "format"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


class PipelineMetrics(BaseModel):
    """Aggregated metrics for a full pipeline run."""

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Cost
    total_cost_usd: float = 0.0
    compile_cost_usd: float = 0.0
    route_cost_usd: float = 0.0

    # Timing
    total_compile_time_ms: float = 0.0
    total_route_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0

    # Accuracy
    compilation_success: bool = False
    compilation_retries: int = 0
    methods_compiled: int = 0
    query_accuracy: float = 0.0
    queries_correct: int = 0
    queries_total: int = 0
    hallucination_rate: float = 0.0

    # Individual calls
    llm_calls: list[LLMCall] = Field(default_factory=list)


class UsageTracker:
    """Tracks all LLM usage across a pipeline run.

    Usage:
        tracker = UsageTracker()

        # In synthesizer._call_llm:
        tracker.record_call("compile", model, response, latency)

        # In router.route:
        tracker.record_call("route", model, response, latency)

        # At the end:
        tracker.print_report()
    """

    def __init__(self):
        self.calls: list[LLMCall] = []
        self._query_results: list[dict] = []

    def record_call(
        self,
        call_type: str,
        model: str,
        response: Any,
        latency_ms: float,
        success: bool = True,
        error: str = "",
    ) -> LLMCall:
        """Record a single LLM API call.

        Args:
            call_type: Type of call (compile, route, patch, merge, format).
            model: Model name used.
            response: The raw Gemini API response object.
            latency_ms: Time taken for the call in milliseconds.
            success: Whether the call succeeded.
            error: Error message if failed.

        Returns:
            The recorded LLMCall.
        """
        # Extract token usage from Gemini response
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        total_tokens = input_tokens + output_tokens

        # Calculate cost
        pricing = PRICING.get(model, DEFAULT_PRICING)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        call = LLMCall(
            timestamp=datetime.now(timezone.utc).isoformat(),
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

        self.calls.append(call)
        return call

    def record_query_result(
        self,
        query: str,
        expected: Any,
        actual: Any,
        correct: bool,
        latency_ms: float,
    ) -> None:
        """Record a query result for accuracy tracking."""
        self._query_results.append({
            "query": query,
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "latency_ms": latency_ms,
        })

    def get_metrics(self) -> PipelineMetrics:
        """Aggregate all recorded calls into pipeline metrics."""
        metrics = PipelineMetrics()

        for call in self.calls:
            metrics.total_input_tokens += call.input_tokens
            metrics.total_output_tokens += call.output_tokens
            metrics.total_tokens += call.total_tokens
            metrics.total_cost_usd += call.total_cost_usd
            metrics.llm_calls.append(call)

            if call.call_type in ("compile", "compile_section", "merge", "patch"):
                metrics.total_compile_time_ms += call.latency_ms
                metrics.compile_cost_usd += call.total_cost_usd
            elif call.call_type in ("route", "format"):
                metrics.total_route_time_ms += call.latency_ms
                metrics.route_cost_usd += call.total_cost_usd

        # Query accuracy
        if self._query_results:
            metrics.queries_total = len(self._query_results)
            metrics.queries_correct = sum(1 for q in self._query_results if q["correct"])
            metrics.query_accuracy = metrics.queries_correct / metrics.queries_total
            query_times = [q["latency_ms"] for q in self._query_results]
            metrics.avg_query_time_ms = sum(query_times) / len(query_times) if query_times else 0

        return metrics

    def print_report(self) -> None:
        """Print a formatted usage report to console."""
        metrics = self.get_metrics()

        # Header
        console.print("\n")
        console.print("━" * 60, style="bold blue")
        console.print("📊  EDA Pipeline Usage Report", style="bold blue")
        console.print("━" * 60, style="bold blue")

        # Token Usage Table
        token_table = Table(title="Token Usage", show_header=True, header_style="bold cyan")
        token_table.add_column("Category", style="bold")
        token_table.add_column("Value", justify="right")

        token_table.add_row("Input Tokens", f"{metrics.total_input_tokens:,}")
        token_table.add_row("Output Tokens", f"{metrics.total_output_tokens:,}")
        token_table.add_row("Total Tokens", f"[bold]{metrics.total_tokens:,}[/bold]")
        token_table.add_row("", "")
        token_table.add_row("Total Cost", f"[bold green]${metrics.total_cost_usd:.6f}[/bold green]")
        token_table.add_row("  ↳ Compilation", f"${metrics.compile_cost_usd:.6f}")
        token_table.add_row("  ↳ Routing", f"${metrics.route_cost_usd:.6f}")

        console.print(token_table)

        # Timing Table
        timing_table = Table(title="Timing", show_header=True, header_style="bold cyan")
        timing_table.add_column("Phase", style="bold")
        timing_table.add_column("Time", justify="right")

        timing_table.add_row("Compilation", f"{metrics.total_compile_time_ms:,.0f}ms")
        timing_table.add_row("Avg Query (routing)", f"{metrics.avg_query_time_ms:.2f}ms")

        console.print(timing_table)

        # Accuracy Table
        if metrics.queries_total > 0:
            acc_table = Table(title="Accuracy", show_header=True, header_style="bold cyan")
            acc_table.add_column("Metric", style="bold")
            acc_table.add_column("Value", justify="right")

            acc_table.add_row(
                "Query Accuracy",
                f"[bold]{metrics.query_accuracy:.1%}[/bold] "
                f"({metrics.queries_correct}/{metrics.queries_total})"
            )
            acc_table.add_row("Hallucination Rate", f"{metrics.hallucination_rate:.1%}")

            console.print(acc_table)

        # Individual Calls Table
        calls_table = Table(title="LLM Calls Breakdown", show_header=True, header_style="bold cyan")
        calls_table.add_column("#", justify="right", style="dim")
        calls_table.add_column("Type")
        calls_table.add_column("Model")
        calls_table.add_column("In Tokens", justify="right")
        calls_table.add_column("Out Tokens", justify="right")
        calls_table.add_column("Cost", justify="right")
        calls_table.add_column("Time", justify="right")

        for i, call in enumerate(self.calls, 1):
            calls_table.add_row(
                str(i),
                call.call_type,
                call.model.split("/")[-1],
                f"{call.input_tokens:,}",
                f"{call.output_tokens:,}",
                f"${call.total_cost_usd:.6f}",
                f"{call.latency_ms:.0f}ms",
            )

        console.print(calls_table)
        console.print("━" * 60, style="bold blue")

    def export_json(self) -> dict:
        """Export all metrics as a JSON-serializable dict."""
        metrics = self.get_metrics()
        return {
            "summary": {
                "total_tokens": metrics.total_tokens,
                "total_cost_usd": round(metrics.total_cost_usd, 6),
                "compile_time_ms": round(metrics.total_compile_time_ms, 2),
                "query_accuracy": round(metrics.query_accuracy, 4) if metrics.queries_total else None,
                "queries": f"{metrics.queries_correct}/{metrics.queries_total}" if metrics.queries_total else None,
            },
            "tokens": {
                "input": metrics.total_input_tokens,
                "output": metrics.total_output_tokens,
                "total": metrics.total_tokens,
            },
            "cost": {
                "total_usd": round(metrics.total_cost_usd, 6),
                "compile_usd": round(metrics.compile_cost_usd, 6),
                "route_usd": round(metrics.route_cost_usd, 6),
            },
            "calls": [call.model_dump() for call in self.calls],
        }


# Global tracker instance — shared across modules within a pipeline run
_active_tracker: UsageTracker | None = None


def get_tracker() -> UsageTracker:
    """Get or create the active tracker for the current pipeline run."""
    global _active_tracker
    if _active_tracker is None:
        _active_tracker = UsageTracker()
    return _active_tracker


def reset_tracker() -> UsageTracker:
    """Reset and return a fresh tracker for a new pipeline run."""
    global _active_tracker
    _active_tracker = UsageTracker()
    return _active_tracker
