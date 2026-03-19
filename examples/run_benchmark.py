"""
Example: Run a head-to-head benchmark comparing EDA vs RAG.

Usage:
    python examples/run_benchmark.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from eda.evaluation.benchmark import Benchmark
from eda.evaluation.dataset import DatasetManager
from eda.tracker import reset_tracker

console = Console()


def main():
    console.print("[bold]🏆 EDA vs RAG Benchmark Runner[/bold]\n")

    # Load the financial report fixture
    datasets = DatasetManager()
    fixture = datasets.get_fixture("financial_report_q3")
    
    if not fixture:
        console.print("[red]❌ Could not find the financial_report_q3 fixture.[/red]")
        return

    console.print(f"📄 Loaded fixture: {fixture.name}")
    console.print(f"❓ {len(fixture.qa_pairs)} specific test questions")

    # Start tracking
    tracker = reset_tracker()

    # Run the benchmark
    benchmark = Benchmark()
    try:
        report = benchmark.run_full_comparison(fixture)
        
        # Print results side-by-side
        console.print("\n")
        benchmark.print_comparison(report)
        
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        import traceback
        traceback.print_exc()

    # Finally, print the usage report (cost & tokens) for this entire run
    tracker.print_report()


if __name__ == "__main__":
    main()
