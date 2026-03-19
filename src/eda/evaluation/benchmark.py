"""
Benchmark — Head-to-head comparison of EDA vs RAG.

Runs both pipelines on the same documents and queries,
measuring accuracy, latency, and hallucination rate.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from eda.evaluation.dataset import EvalFixture
from eda.evaluation.metrics import Metrics

console = Console()


class PipelineResult(BaseModel):
    """Result from running a single pipeline."""

    pipeline_name: str = ""
    accuracy: float = 0.0
    avg_query_latency_ms: float = 0.0
    compile_time_ms: float = 0.0
    hallucination_rate: float = 0.0
    total_queries: int = 0
    correct_queries: int = 0
    query_results: list[dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class BenchmarkReport(BaseModel):
    """Side-by-side comparison of EDA vs RAG."""

    eda_result: PipelineResult | None = None
    rag_result: PipelineResult | None = None
    winner_accuracy: str = ""
    winner_latency: str = ""
    speedup_factor: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class Benchmark:
    """Run head-to-head EDA vs RAG benchmarks."""

    def run_eda_benchmark(
        self,
        fixture: EvalFixture,
    ) -> PipelineResult:
        """Run the EDA pipeline on a fixture.

        Args:
            fixture: Evaluation fixture with document and Q&A pairs.

        Returns:
            PipelineResult with accuracy and latency.
        """
        from eda.compiler.parser import DocumentParser
        from eda.compiler.synthesizer import Synthesizer
        from eda.compiler.validator import Validator
        from eda.runtime.executor import Executor
        from eda.runtime.router import Router

        parser = DocumentParser()
        synthesizer = Synthesizer()
        validator = Validator()
        executor = Executor()
        router = Router()

        # Phase 1: Compile
        compile_start = time.perf_counter()

        if fixture.document_text:
            doc = parser.parse_text(fixture.document_text, fixture.name)
        else:
            doc = parser.parse(fixture.document_path)

        compilation = synthesizer.compile(doc)
        validation = validator.validate(
            compilation.source_code,
            compilation.class_name,
            synthesizer=synthesizer,
        )

        compile_time_ms = (time.perf_counter() - compile_start) * 1000

        if not validation.success:
            from rich.console import Console
            c = Console()
            c.print(f"[red]❌ EDA Validation Failed![/red]")
            c.print(f"[yellow]Issues:[/yellow] {validation.issues}")
            c.print(f"[cyan]Source Code Generated:[/cyan]\n{validation.source_code}")
            return PipelineResult(
                pipeline_name="EDA",
                compile_time_ms=compile_time_ms,
                total_queries=len(fixture.qa_pairs),
            )

        # Load the artifact
        instance = executor.load_artifact(validation.source_code, compilation.class_name)
        methods = executor.get_available_methods(instance)
        descriptions = executor.get_method_descriptions(instance)

        # Phase 3: Query
        query_results = []
        query_times = []

        for qa in fixture.qa_pairs:
            q_start = time.perf_counter()

            # Route the query
            routing = router.route(qa.question, methods, descriptions)

            # Execute
            exec_result = executor.execute(
                instance,
                routing.method_name,
                routing.arguments,
                routing.method_call,
            )

            q_time = (time.perf_counter() - q_start) * 1000
            query_times.append(q_time)

            actual_val = exec_result.data if exec_result.success else None
            is_correct = Metrics._values_match(actual_val, qa.expected_answer)
            
            console.print(f"  [cyan]Q:[/cyan] {qa.question}")
            console.print(f"  [yellow]Expected:[/yellow] {qa.expected_answer} ({type(qa.expected_answer).__name__})")
            console.print(f"  [magenta]Actual:[/magenta]   {actual_val} ({type(actual_val).__name__})")
            if is_correct:
                console.print(f"  [green]✅ Correct[/green]")
            else:
                console.print(f"  [red]❌ Incorrect[/red]")
            console.print("-" * 20)

            query_results.append({
                "query": qa.question,
                "expected": qa.expected_answer,
                "actual": actual_val,
                "success": exec_result.success,
                "latency_ms": q_time,
            })

        # Compute metrics
        accuracy_result = Metrics.query_accuracy(query_results)
        latency_result = Metrics.latency(compile_time_ms, query_times)

        return PipelineResult(
            pipeline_name="EDA",
            accuracy=accuracy_result.accuracy,
            avg_query_latency_ms=latency_result.avg_query_time_ms,
            compile_time_ms=compile_time_ms,
            total_queries=accuracy_result.total_queries,
            correct_queries=accuracy_result.correct,
            query_results=query_results,
        )

    def run_rag_benchmark(self, fixture: EvalFixture) -> PipelineResult:
        """Run the RAG baseline on a fixture."""
        from eda.evaluation.rag_baseline import RAGBaseline
        
        console.print("[dim]Initializing RAG Baseline (ChromaDB + LangChain)...[/dim]")
        rag = RAGBaseline()
        return rag.run_benchmark(fixture)

    def run_full_comparison(self, fixture: EvalFixture) -> BenchmarkReport:
        """Run both pipelines and return a comparison report."""
        console.print(f"\n[bold blue]=== Starting Benchmark: {fixture.name} ===[/bold blue]")
        
        console.print("\n[bold]1. Running EDA Pipeline...[/bold]")
        eda_res = self.run_eda_benchmark(fixture)
        
        console.print("\n[bold]2. Running RAG Baseline...[/bold]")
        try:
            rag_res = self.run_rag_benchmark(fixture)
        except Exception as e:
            console.print(f"[red]RAG Pipeline failed: {e}[/red]")
            rag_res = PipelineResult(pipeline_name="RAG failed")

        report = BenchmarkReport(eda_result=eda_res, rag_result=rag_res)

        # Compute winners
        if eda_res.accuracy >= rag_res.accuracy:
            report.winner_accuracy = "EDA"
        else:
            report.winner_accuracy = "RAG"

        if eda_res.avg_query_latency_ms <= rag_res.avg_query_latency_ms or not rag_res.avg_query_latency_ms:
            report.winner_latency = "EDA"
            if rag_res.avg_query_latency_ms and eda_res.avg_query_latency_ms > 0:
                report.speedup_factor = rag_res.avg_query_latency_ms / eda_res.avg_query_latency_ms
        else:
            report.winner_latency = "RAG"

        return report

    def print_comparison(self, report: BenchmarkReport) -> None:
        """Print a formatted comparison table."""
        table = Table(title="EDA vs RAG Benchmark Results", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("EDA", justify="right")
        table.add_column("RAG", justify="right")
        table.add_column("Winner", justify="center")

        eda = report.eda_result or PipelineResult()
        rag = report.rag_result or PipelineResult()

        # Accuracy
        acc_winner = "EDA" if eda.accuracy >= (rag.accuracy or 0) else "RAG"
        table.add_row(
            "Query Accuracy",
            f"{eda.accuracy:.1%}",
            f"{rag.accuracy:.1%}" if rag.accuracy else "N/A",
            f"🏆 {acc_winner}",
        )

        # Query Latency
        lat_winner = "EDA" if eda.avg_query_latency_ms <= (rag.avg_query_latency_ms or float("inf")) else "RAG"
        table.add_row(
            "Avg Query Latency",
            f"{eda.avg_query_latency_ms:.2f}ms",
            f"{rag.avg_query_latency_ms:.2f}ms" if rag.avg_query_latency_ms else "N/A",
            f"🏆 {lat_winner}",
        )

        # Compile Time
        table.add_row(
            "Compile/Index Time",
            f"{eda.compile_time_ms:.0f}ms",
            f"{rag.compile_time_ms:.0f}ms" if rag.compile_time_ms else "N/A",
            "",
        )

        # Hallucination Rate
        table.add_row(
            "Hallucination Rate",
            f"{eda.hallucination_rate:.1%}",
            f"{rag.hallucination_rate:.1%}" if rag.hallucination_rate else "N/A",
            "",
        )

        console.print(table)

        if report.speedup_factor > 0:
            console.print(
                f"\n⚡ EDA is **{report.speedup_factor:.0f}x faster** at query time"
            )
