"""
RAG Baseline — A standard Retrieval-Augmented Generation pipeline.

Used exclusively for head-to-head benchmarking against the EDA compiler.
Implements a naive chunk-embed-retrieve-generate approach using Langchain,
ChromaDB, and Google Gemini.
"""

from __future__ import annotations

import os
import time

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from eda.config import config
from eda.evaluation.benchmark import PipelineResult
from eda.evaluation.dataset import EvalFixture
from eda.evaluation.metrics import Metrics
from eda.tracker import get_tracker


class RAGBaseline:
    """Standard RAG pipeline for benchmark comparisons."""

    def __init__(self, model: str | None = None):
        self.model = model or config.compiler.model
        self.api_key = config.google_api_key

        # Standard RAG parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.k_retrieval = 3

    def run_benchmark(self, fixture: EvalFixture) -> PipelineResult:
        """Run the full RAG pipeline (index + query) on a fixture.

        Args:
            fixture: The evaluation fixture containing text and Q&A.

        Returns:
            PipelineResult containing accuracy and latency.
        """
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required for RAG baseline.")

        # Set API key for LangChain
        os.environ["GOOGLE_API_KEY"] = self.api_key

        # ---------------------------------------------------------
        # Phase 1: Indexing (equivalent to EDA Compilation)
        # ---------------------------------------------------------
        index_start = time.perf_counter()

        # Write text to temp file if passed directly
        import tempfile
        if fixture.document_text:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                f.write(fixture.document_text)
                temp_path = f.name
            doc_path = temp_path
        else:
            doc_path = fixture.document_path
            temp_path = None

        try:
            loader = TextLoader(doc_path, encoding="utf-8")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = text_splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Use an ephemeral in-memory Chroma database
            vectorstore = Chroma.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": self.k_retrieval})

            llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.0,
            )

            # Strict prompt to match EDA's factual boundaries
            prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer or the context does not provide the information, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible, and provide exactly what was asked.

Context:
{context}

Question: {question}
Answer:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
            )

            index_time_ms = (time.perf_counter() - index_start) * 1000

            # ---------------------------------------------------------
            # Phase 2: Querying
            # ---------------------------------------------------------
            query_results = []
            query_times = []
            tracker = get_tracker()

            for qa in fixture.qa_pairs:
                q_start = time.perf_counter()

                # Execute RAG query
                try:
                    # In Langchain v0.1+, invoke returns a dict with 'query' and 'result'
                    response = qa_chain.invoke({"query": qa.question})
                    actual_answer = response.get("result", "")
                    success = True
                except Exception as e:
                    actual_answer = str(e)
                    success = False

                q_time = (time.perf_counter() - q_start) * 1000
                query_times.append(q_time)

                # RAG always returns a string; let the metrics evaluator handle type coercion
                query_results.append({
                    "query": qa.question,
                    "expected": qa.expected_answer,
                    "actual": actual_answer,
                    "success": success,
                    "latency_ms": q_time,
                })

                # We can't automatically track tokens natively via LangChain's simple invoke
                # without callback handlers, so we'll just log the latency here.
                # (A production RAG setup would integrate token callbacks).
                tracker.record_call(
                    call_type="rag_query",
                    model=self.model,
                    response=None,  # Not tracked natively in this simple baseline
                    latency_ms=q_time,
                )

            # Compute metrics
            accuracy_result = Metrics.query_accuracy(query_results)
            latency_result = Metrics.latency(index_time_ms, query_times)

            return PipelineResult(
                pipeline_name="RAG",
                accuracy=accuracy_result.accuracy,
                avg_query_latency_ms=latency_result.avg_query_time_ms,
                compile_time_ms=index_time_ms,
                total_queries=accuracy_result.total_queries,
                correct_queries=accuracy_result.correct,
                query_results=query_results,
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
