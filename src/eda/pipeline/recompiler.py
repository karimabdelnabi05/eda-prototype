"""
Recompiler — Orchestrates recompilation when documents change.

Integrates the differ, compiler, and validator to handle document updates.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console

from eda.compiler.artifacts import ArtifactManager
from eda.compiler.parser import DocumentParser
from eda.compiler.synthesizer import Synthesizer
from eda.compiler.validator import Validator
from eda.pipeline.differ import Differ, DocumentDiff

console = Console()


class RecompilationResult(BaseModel):
    """Result of a recompilation triggered by document changes."""

    success: bool = False
    diff: DocumentDiff | None = None
    new_version: int = 0
    errors: list[str] = Field(default_factory=list)
    recompiled: bool = False

    class Config:
        arbitrary_types_allowed = True


class Recompiler:
    """Orchestrates document recompilation on change.

    Watches for document changes, triggers recompilation, validates
    the new artifact, and saves it as a new version.
    """

    def __init__(self):
        self.parser = DocumentParser()
        self.differ = Differ()
        self.synthesizer = Synthesizer()
        self.validator = Validator()
        self.artifacts = ArtifactManager()

    def recompile_if_changed(
        self,
        new_document_path: str | Path,
        artifact_id: str,
        class_name: str,
    ) -> RecompilationResult:
        """Check if a document has changed and recompile if needed.

        Args:
            new_document_path: Path to the updated document.
            artifact_id: ID of the existing artifact.
            class_name: Class name for the compiled output.

        Returns:
            RecompilationResult with diff and new version info.
        """
        new_doc = self.parser.parse(new_document_path)

        # Load existing artifact metadata to compare hashes
        try:
            _, old_metadata = self.artifacts.load(artifact_id)
            old_hash = old_metadata.source_hash
        except FileNotFoundError:
            # No existing artifact — full compilation needed
            console.print("[yellow]No existing artifact found. Full compilation.[/yellow]")
            return self._full_recompile(new_doc, artifact_id, class_name)

        # Compare hashes
        if new_doc.source_hash == old_hash:
            console.print("[green]Document unchanged. No recompilation needed.[/green]")
            return RecompilationResult(
                success=True,
                recompiled=False,
            )

        # Document changed — recompile
        console.print("[yellow]Document changed. Recompiling...[/yellow]")
        return self._full_recompile(new_doc, artifact_id, class_name)

    def _full_recompile(
        self,
        document,
        artifact_id: str,
        class_name: str,
    ) -> RecompilationResult:
        """Full recompilation of a document."""
        # Compile
        compilation = self.synthesizer.compile(document, class_name)
        if not compilation.success:
            return RecompilationResult(
                success=False,
                errors=compilation.errors,
            )

        # Validate
        validation = self.validator.validate(
            compilation.source_code,
            class_name,
            synthesizer=self.synthesizer,
        )
        if not validation.success:
            return RecompilationResult(
                success=False,
                errors=validation.errors,
            )

        # Save new version
        metadata = self.artifacts.save(
            source_code=validation.source_code,
            class_name=class_name,
            source_path=document.source_path,
            source_hash=document.source_hash,
            methods=validation.methods_found,
            retries=validation.retries_used,
        )

        console.print(
            f"[green]✅ Recompiled {class_name} v{metadata.version} "
            f"({len(validation.methods_found)} methods)[/green]"
        )

        return RecompilationResult(
            success=True,
            new_version=metadata.version,
            recompiled=True,
        )
