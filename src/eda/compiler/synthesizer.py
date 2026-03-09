"""
LLM Synthesizer — The core compiler that transforms documents into executable Python code.

This module takes parsed DocumentContent and uses an LLM to generate a self-contained
Python class that encapsulates all facts, figures, and logic from the document.
"""

from __future__ import annotations

import re
import textwrap

from google import genai

from eda.compiler.parser import DocumentContent
from eda.compiler.prompts import (
    COMPILER_SYSTEM_PROMPT,
    MERGE_PROMPT,
    SECTION_COMPILER_PROMPT,
)
from eda.config import config

# Maximum characters before we switch to section-by-section compilation
MAX_SINGLE_PASS_CHARS = 15_000


class CompilationResult:
    """Result of a compilation attempt."""

    def __init__(
        self,
        source_code: str,
        class_name: str,
        success: bool,
        errors: list[str] | None = None,
        token_usage: dict | None = None,
        retries: int = 0,
    ):
        self.source_code = source_code
        self.class_name = class_name
        self.success = success
        self.errors = errors or []
        self.token_usage = token_usage or {}
        self.retries = retries

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"CompilationResult({status} {self.class_name}, retries={self.retries})"


class Synthesizer:
    """The LLM Compiler — synthesizes documents into executable Python classes.

    This is the core of the EDA system. It takes structured document content
    and produces a standalone Python class through LLM-powered code generation.
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.compiler.model
        self.client = genai.Client(api_key=config.google_api_key)

    def compile(
        self,
        document: DocumentContent,
        class_name: str | None = None,
    ) -> CompilationResult:
        """Compile a document into a Python class.

        Args:
            document: Parsed document content.
            class_name: Name for the generated class. Auto-generated if not provided.

        Returns:
            CompilationResult containing the generated source code.
        """
        if not class_name:
            class_name = self._generate_class_name(document.title)

        compilation_text = document.to_compilation_text()

        if len(compilation_text) > MAX_SINGLE_PASS_CHARS:
            return self._compile_sectioned(document, class_name)
        else:
            return self._compile_single_pass(compilation_text, class_name)

    def _compile_single_pass(
        self, text: str, class_name: str
    ) -> CompilationResult:
        """Compile the entire document in a single LLM call."""
        user_prompt = textwrap.dedent(f"""
            Compile the following document into a Python class named `{class_name}`.

            ## Document Content:

            {text}
        """).strip()

        source_code = self._call_llm(COMPILER_SYSTEM_PROMPT, user_prompt)
        source_code = self._clean_code(source_code)

        return CompilationResult(
            source_code=source_code,
            class_name=class_name,
            success=True,  # Validation happens in the validator module
        )

    def _compile_sectioned(
        self, document: DocumentContent, class_name: str
    ) -> CompilationResult:
        """Compile document section-by-section, then merge into one class."""
        section_codes = []

        for i, section in enumerate(document.sections):
            if not section.content.strip() and not section.heading:
                continue

            section_text = ""
            if section.heading:
                section_text += f"## {section.heading}\n\n"
            section_text += section.content
            for table in section.tables:
                section_text += "\n\n" + table.to_markdown()

            user_prompt = textwrap.dedent(f"""
                Compile this section (section {i + 1}) into Python methods.
                Section heading: {section.heading or 'Unnamed Section'}

                ## Section Content:

                {section_text}
            """).strip()

            code = self._call_llm(SECTION_COMPILER_PROMPT, user_prompt)
            code = self._clean_code(code)
            section_codes.append(code)

        # Merge all sections into one class
        all_sections = "\n\n# --- Section Boundary ---\n\n".join(section_codes)
        merge_prompt = textwrap.dedent(f"""
            Merge these compiled sections into a single Python class named `{class_name}`.

            ## Compiled Sections:

            {all_sections}
        """).strip()

        merged_code = self._call_llm(MERGE_PROMPT, merge_prompt)
        merged_code = self._clean_code(merged_code)

        return CompilationResult(
            source_code=merged_code,
            class_name=class_name,
            success=True,
        )

    def patch(self, source_code: str, error_traceback: str) -> str:
        """Attempt to fix broken code by feeding the error back to the LLM.

        Args:
            source_code: The broken Python code.
            error_traceback: The error/traceback from execution.

        Returns:
            Patched source code.
        """
        from eda.compiler.prompts import PATCH_PROMPT

        user_prompt = PATCH_PROMPT.format(
            error_traceback=error_traceback,
            code=source_code,
        )

        patched = self._call_llm(
            "You are a Python code debugger. Fix the code and output ONLY valid Python.",
            user_prompt,
        )
        return self._clean_code(patched)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make an LLM call using the configured provider."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            config={
                "system_instruction": system_prompt,
                "temperature": config.compiler.temperature,
            },
        )
        return response.text or ""

    def _clean_code(self, code: str) -> str:
        """Remove markdown fences and other non-code artifacts from LLM output."""
        # Remove ```python ... ``` fences
        code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
        # Remove leading/trailing whitespace
        code = code.strip()
        return code

    def _generate_class_name(self, title: str) -> str:
        """Generate a valid Python class name from a document title."""
        if not title:
            return "CompiledDocument"
        # Remove non-alphanumeric characters, convert to PascalCase
        words = re.sub(r"[^a-zA-Z0-9\s]", "", title).split()
        class_name = "".join(word.capitalize() for word in words)
        # Ensure it starts with a letter
        if class_name and not class_name[0].isalpha():
            class_name = "Doc" + class_name
        return class_name or "CompiledDocument"
