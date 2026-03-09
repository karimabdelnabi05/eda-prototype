"""
EDA CLI — Command-line interface for compiling and querying documents.
"""

import sys
import time

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """EDA — Executable Document Architecture.

    Compile documents into deterministic, executable Python APIs.
    """
    pass


@main.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option("--class-name", "-c", default=None, help="Class name for the compiled artifact")
@click.option("--output", "-o", default=None, help="Output file path (default: artifacts/)")
@click.option("--show-code", is_flag=True, help="Print the generated code")
def compile(document_path: str, class_name: str, output: str, show_code: bool):
    """Compile a document into an executable Python class."""
    from eda.compiler.artifacts import ArtifactManager
    from eda.compiler.parser import DocumentParser
    from eda.compiler.synthesizer import Synthesizer
    from eda.compiler.validator import Validator

    console.print(Panel("🔧 EDA Compiler", subtitle="Ahead-of-Time Document Synthesis"))

    # Parse
    console.print(f"📄 Parsing: {document_path}")
    parser = DocumentParser()
    doc = parser.parse(document_path)
    console.print(f"   → {doc.char_count:,} characters, {len(doc.sections)} sections")

    # Compile
    console.print("⚡ Compiling with LLM...")
    start = time.perf_counter()
    synthesizer = Synthesizer()
    compilation = synthesizer.compile(doc, class_name)
    compile_time = (time.perf_counter() - start) * 1000

    console.print(f"   → Generated class: {compilation.class_name}")
    console.print(f"   → Compile time: {compile_time:.0f}ms")

    # Validate
    console.print("✅ Validating in sandbox...")
    validator = Validator()
    validation = validator.validate(
        compilation.source_code,
        compilation.class_name,
        synthesizer=synthesizer,
    )

    if not validation.success:
        console.print("[red]❌ Validation failed:[/red]")
        for error in validation.errors:
            console.print(f"   {error}")
        sys.exit(1)

    console.print(
        f"   → {validation.method_count} methods discovered "
        f"(retries: {validation.retries_used})"
    )

    # Save
    artifacts = ArtifactManager()
    metadata = artifacts.save(
        source_code=validation.source_code,
        class_name=compilation.class_name,
        source_path=document_path,
        source_hash=doc.source_hash,
        methods=validation.methods_found,
        retries=validation.retries_used,
    )

    console.print(
        f"[green]✅ Saved: {metadata.artifact_id}/v{metadata.version}.py[/green]"
    )

    if show_code:
        console.print("\n")
        syntax = Syntax(validation.source_code, "python", theme="monokai")
        console.print(syntax)


@main.command()
@click.argument("artifact_id")
@click.argument("query")
@click.option("--llm-format", is_flag=True, help="Use LLM for richer formatting")
def query(artifact_id: str, query: str, llm_format: bool):
    """Query a compiled artifact with natural language."""
    from eda.compiler.artifacts import ArtifactManager
    from eda.runtime.executor import Executor
    from eda.runtime.formatter import Formatter
    from eda.runtime.router import Router

    console.print(Panel(f"🔍 Querying: {artifact_id}"))

    # Load artifact
    artifacts = ArtifactManager()
    source_code, metadata = artifacts.load(artifact_id)

    # Execute
    executor = Executor()
    instance = executor.load_artifact(source_code, metadata.class_name)
    methods = executor.get_available_methods(instance)
    descriptions = executor.get_method_descriptions(instance)

    # Route
    console.print(f"📝 Query: \"{query}\"")
    router = Router()
    routing = router.route(query, methods, descriptions)
    console.print(f"🔀 Routed to: {routing.method_call}")

    # Execute
    result = executor.execute(
        instance, routing.method_name, routing.arguments, routing.method_call
    )

    # Format
    formatter = Formatter()
    response = formatter.format(result, query, use_llm=llm_format)
    console.print(f"\n{response}")


@main.command()
def list():
    """List all compiled artifacts."""
    from eda.compiler.artifacts import ArtifactManager

    artifacts = ArtifactManager()
    all_artifacts = artifacts.list_artifacts()

    if not all_artifacts:
        console.print("No compiled artifacts found.")
        return

    console.print(Panel("📚 Compiled Artifacts"))
    for a in all_artifacts:
        console.print(
            f"  • [bold]{a.artifact_id}[/bold] v{a.version} "
            f"({len(a.methods)} methods) — {a.compiled_at[:10]}"
        )


if __name__ == "__main__":
    main()
