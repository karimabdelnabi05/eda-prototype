"""
Example: Query a compiled artifact with natural language.

Usage:
    python examples/query_report.py

Requires: Run examples/compile_report.py first to generate the artifact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from eda.compiler.artifacts import ArtifactManager
from eda.runtime.executor import Executor
from eda.runtime.router import Router
from eda.runtime.formatter import Formatter

console = Console()

SAMPLE_QUERIES = [
    "What was the total revenue in Q3 2024?",
    "Can a standard employee book a $1,500 business class flight?",
    "What was the Q3 capital expenditure and what drove it?",
    "How many engineers work at the company?",
    "What is the operating margin?",
]


def main():
    console.print("[bold]EDA — Querying Compiled Financial Report[/bold]\n")

    # Load the compiled artifact
    artifacts = ArtifactManager()
    try:
        source_code, metadata = artifacts.load("financialreport2024")
    except FileNotFoundError:
        console.print("[red]❌ No compiled artifact found. Run compile_report.py first.[/red]")
        return

    console.print(f"📚 Loaded: {metadata.class_name} v{metadata.version}\n")

    # Set up runtime
    executor = Executor()
    instance = executor.load_artifact(source_code, metadata.class_name)
    methods = executor.get_available_methods(instance)
    descriptions = executor.get_method_descriptions(instance)
    router = Router()
    formatter = Formatter()

    console.print(f"🔧 Available methods: {', '.join(methods)}\n")
    console.print("─" * 60)

    # Run queries
    for query in SAMPLE_QUERIES:
        console.print(f"\n❓ [bold]{query}[/bold]")

        # Route
        routing = router.route(query, methods, descriptions)
        console.print(f"   🔀 → {routing.method_call}")

        # Execute (deterministic!)
        result = executor.execute(
            instance, routing.method_name, routing.arguments, routing.method_call
        )

        # Format
        response = formatter.format(result, query)
        console.print(f"   💬 {response.split(chr(10))[0]}")
        console.print(f"   ⏱️  {result.execution_time_ms:.3f}ms")

    console.print("\n" + "─" * 60)
    console.print("[green]All queries executed deterministically — zero LLM at runtime.[/green]")


if __name__ == "__main__":
    main()
