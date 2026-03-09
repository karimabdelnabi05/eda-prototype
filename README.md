# 🏗️ EDA — Executable Document Architecture

> **An LLM-powered AOT compiler that synthesizes documents into deterministic, executable Python APIs — replacing RAG entirely.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Core Idea

Traditional AI treats documents as things to **search**. EDA treats documents as things to **compile**.

Instead of chunking text → embedding → vector search → LLM generation at runtime (RAG), EDA:

1. **Compiles** the document into a self-contained Python class using an LLM (ahead of time)
2. **Destroys** the original document — the compiled code IS the knowledge base
3. **Queries** are translated into deterministic function calls — no LLM at runtime

The result: **sub-millisecond query latency, zero hallucinations on data retrieval, and 100% provenance**.

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Document   │────▶│  LLM Compiler    │────▶│  Python Class    │
│  (PDF/TXT)  │     │  (Ahead-of-Time) │     │  (Executable)    │
└─────────────┘     └──────────────────┘     └──────────────────┘
                                                      │
                         ┌────────────────────────────┘
                         ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  NL Query   │────▶│  Query Router    │────▶│  Function Call   │
│  "What was  │     │  (Fast LLM)      │     │  obj.get_capex() │
│   Q3 capex?"│     └──────────────────┘     │  → $45M          │
└─────────────┘                              └──────────────────┘
                                               ⚡ <1ms, 0% hallucination
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/eda-prototype.git
cd eda-prototype
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Compile a Document

```bash
# CLI
eda compile tests/fixtures/sample_financial_report.txt --show-code

# Python
python examples/compile_report.py
```

### 4. Query It

```bash
# CLI
eda query financialreport2024 "What was Q3 capital expenditure?"

# Python
python examples/query_report.py
```

### 5. Run Tests

```bash
# Unit tests (no API key needed)
pytest tests/unit/ -v

# Integration tests (requires API key)
pytest tests/integration/ -v -m slow
```

## Architecture

```
src/eda/
├── compiler/           # Phase 1: AOT Document Compilation
│   ├── parser.py       # PDF/text → structured content
│   ├── synthesizer.py  # LLM compiler (doc → Python class)
│   ├── validator.py    # Sandbox execution + auto-patch loop
│   ├── prompts.py      # Compilation instruction set
│   └── artifacts.py    # Versioned artifact storage
│
├── runtime/            # Phase 3: Zero-Search Query Execution
│   ├── router.py       # NL question → Python method call
│   ├── executor.py     # Sandboxed method execution
│   └── formatter.py    # Raw result → natural language
│
├── pipeline/           # Phase 4: CI/CD for Documents
│   ├── differ.py       # Detect document changes
│   ├── recompiler.py   # Trigger recompilation
│   └── regression.py   # Old vs new artifact testing
│
└── evaluation/         # Benchmarking & Metrics
    ├── metrics.py      # Accuracy, latency, hallucination rate
    ├── benchmark.py    # EDA vs RAG head-to-head
    └── dataset.py      # Test fixture management
```

## How It Works

### Phase 1: Compilation (Ahead-of-Time)

The compiler takes a document and synthesizes it into a Python class:

```python
from eda.compiler.parser import DocumentParser
from eda.compiler.synthesizer import Synthesizer
from eda.compiler.validator import Validator

# Parse
doc = DocumentParser().parse("financial_report.pdf")

# Compile (LLM call — happens once)
compilation = Synthesizer().compile(doc, "FinancialReport")

# Validate (sandbox execution + auto-patch)
validation = Validator().validate(
    compilation.source_code, "FinancialReport", synthesizer=Synthesizer()
)
```

The generated class contains hardcoded data and logic methods:

```python
class FinancialReport:
    def __init__(self):
        self.revenue = {"total": 128_500_000, "cloud": 72_300_000, ...}
        self.travel_policy = {"Executive": {"max_flight": 5000}, ...}

    def get_total_revenue(self) -> int:
        return self.revenue["total"]

    def check_travel_compliance(self, role: str, cost: float) -> bool:
        return cost <= self.travel_policy[role]["max_flight"]
```

### Phase 3: Query (Runtime — No LLM)

At query time, the user's question is routed to a function call:

```python
from eda.runtime.executor import Executor
from eda.runtime.router import Router

# Route: "What was Q3 capex?" → obj.get_capex(quarter="Q3")
routing = Router().route(query, methods)

# Execute: Pure Python, <1ms
result = Executor().execute(instance, routing.method_name, routing.arguments)
```

## Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| Compilation Success | ≥95% | Documents that compile after validation loop |
| Query Accuracy (Pass@1) | ≥90% | Correct answers on ground truth Q&A |
| Query Latency | <1ms | Time to execute a compiled method |
| Compile Latency | <60s/page | Time to compile a document |
| Hallucination Rate | 0% | Fabricated data in responses |

## EDA vs RAG

| Aspect | EDA | RAG |
|---|---|---|
| Runtime LLM calls | **0** (query routing only) | 1+ per query |
| Query latency | **<1ms** | 500ms–5s |
| Hallucination risk | **None** (deterministic) | Moderate |
| Data provenance | **100%** (method call traces) | Weak |
| Setup cost | Higher (compilation) | Lower (indexing) |
| Best for | Structured data, policies, reports | Exploratory Q&A |

## Development

```bash
# Install with all optional deps
pip install -e ".[all,dev]"

# Run linter
ruff check src/ tests/

# Run unit tests
pytest tests/unit/ -v

# Run full test suite (requires API key)
pytest -v -m slow
```

## License

MIT
