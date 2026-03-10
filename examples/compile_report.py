"""
Example: Compile a financial report into an executable API.

Usage:
    python examples/compile_report.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.syntax import Syntax

from eda.compiler.parser import DocumentParser
from eda.compiler.synthesizer import Synthesizer
from eda.compiler.validator import Validator
from eda.compiler.artifacts import ArtifactManager

console = Console()


def main():
    from eda.tracker import reset_tracker

    tracker = reset_tracker()  # Start fresh tracking for this run
    console.print("[bold]EDA — Compiling Financial Report[/bold]\n")

    # 1. Parse the sample document
    doc_path = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_financial_report.txt"
    parser = DocumentParser()
    doc = parser.parse(doc_path)
    console.print(f"📄 Parsed: {doc.char_count:,} chars, {len(doc.sections)} sections\n")

    # 2. Compile with LLM
    console.print("⚡ Compiling document → Python class...\n")
    synthesizer = Synthesizer()
    compilation = synthesizer.compile(doc, "FinancialReport2024")

    # 3. Validate
    console.print("✅ Validating in sandbox...\n")
    validator = Validator()
    validation = validator.validate(
        compilation.source_code,
        "FinancialReport2024",
        synthesizer=synthesizer,
    )

    if validation.success:
        console.print(f"[green]✅ Compilation successful![/green]")
        console.print(f"   Methods: {', '.join(validation.methods_found)}\n")

        # Show the generated code
        console.print("[bold]Generated Code:[/bold]")
        syntax = Syntax(validation.source_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Save artifact
        artifacts = ArtifactManager()
        metadata = artifacts.save(
            source_code=validation.source_code,
            class_name="FinancialReport2024",
            source_path=str(doc_path),
            source_hash=doc.source_hash,
            methods=validation.methods_found,
        )
        console.print(f"\n💾 Saved as: artifacts/{metadata.artifact_id}/v{metadata.version}.py")
    else:
        console.print(f"[red]❌ Compilation failed:[/red]")
        for error in validation.errors:
            console.print(f"   {error}")

    # 4. Print usage report — tokens, cost, timing
    tracker.print_report()


if __name__ == "__main__":
    main()
