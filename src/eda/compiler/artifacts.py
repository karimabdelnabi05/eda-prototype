"""
Artifact Manager — Save, load, and version compiled Python code artifacts.

Compiled documents are stored as versioned Python files in the artifacts directory.
Each artifact includes metadata about its source document for CI/CD tracking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from eda.config import config


class ArtifactMetadata(BaseModel):
    """Metadata stored alongside each compiled artifact."""

    artifact_id: str = ""
    class_name: str = ""
    source_path: str = ""
    source_hash: str = ""
    compiled_at: str = ""
    compiler_model: str = ""
    version: int = 1
    methods: list[str] = Field(default_factory=list)
    char_count: int = 0
    compilation_retries: int = 0


class ArtifactManager:
    """Manages compiled Python artifacts — save, load, version, diff."""

    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = artifacts_dir or config.artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        source_code: str,
        class_name: str,
        source_path: str = "",
        source_hash: str = "",
        methods: list[str] | None = None,
        retries: int = 0,
    ) -> ArtifactMetadata:
        """Save a compiled artifact to disk.

        Args:
            source_code: The compiled Python source code.
            class_name: Name of the compiled class.
            source_path: Path to the original document.
            source_hash: Hash of the original document content.
            methods: List of method names in the compiled class.
            retries: Number of compilation retries used.

        Returns:
            ArtifactMetadata for the saved artifact.
        """
        artifact_id = class_name.lower()
        artifact_dir = self.artifacts_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Determine version
        existing_versions = list(artifact_dir.glob("v*.py"))
        version = len(existing_versions) + 1

        # Save the Python file
        code_path = artifact_dir / f"v{version}.py"
        code_path.write_text(source_code, encoding="utf-8")

        # Also save as "latest.py" for easy access
        latest_path = artifact_dir / "latest.py"
        latest_path.write_text(source_code, encoding="utf-8")

        # Save metadata
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            class_name=class_name,
            source_path=source_path,
            source_hash=source_hash,
            compiled_at=datetime.now(timezone.utc).isoformat(),
            compiler_model=config.compiler.model,
            version=version,
            methods=methods or [],
            char_count=len(source_code),
            compilation_retries=retries,
        )

        meta_path = artifact_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata.model_dump(), indent=2),
            encoding="utf-8",
        )

        return metadata

    def load(self, artifact_id: str, version: str = "latest") -> tuple[str, ArtifactMetadata]:
        """Load a compiled artifact from disk.

        Args:
            artifact_id: The artifact identifier (lowercase class name).
            version: Version to load — "latest" or "v1", "v2", etc.

        Returns:
            Tuple of (source_code, metadata).

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        artifact_dir = self.artifacts_dir / artifact_id

        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")

        # Load code
        if version == "latest":
            code_path = artifact_dir / "latest.py"
        else:
            code_path = artifact_dir / f"{version}.py"

        if not code_path.exists():
            raise FileNotFoundError(f"Artifact version not found: {code_path}")

        source_code = code_path.read_text(encoding="utf-8")

        # Load metadata
        meta_path = artifact_dir / "metadata.json"
        if meta_path.exists():
            metadata = ArtifactMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
        else:
            metadata = ArtifactMetadata(artifact_id=artifact_id)

        return source_code, metadata

    def list_artifacts(self) -> list[ArtifactMetadata]:
        """List all saved artifacts with their metadata."""
        artifacts = []
        for artifact_dir in sorted(self.artifacts_dir.iterdir()):
            if artifact_dir.is_dir() and (artifact_dir / "metadata.json").exists():
                meta = json.loads(
                    (artifact_dir / "metadata.json").read_text(encoding="utf-8")
                )
                artifacts.append(ArtifactMetadata(**meta))
        return artifacts

    def get_version_count(self, artifact_id: str) -> int:
        """Get the number of versions for an artifact."""
        artifact_dir = self.artifacts_dir / artifact_id
        if not artifact_dir.exists():
            return 0
        return len(list(artifact_dir.glob("v*.py")))

    def diff_versions(self, artifact_id: str, v1: int, v2: int) -> str:
        """Get a unified diff between two versions of an artifact.

        Args:
            artifact_id: The artifact identifier.
            v1: First version number.
            v2: Second version number.

        Returns:
            Unified diff string.
        """
        import difflib

        code1, _ = self.load(artifact_id, f"v{v1}")
        code2, _ = self.load(artifact_id, f"v{v2}")

        diff = difflib.unified_diff(
            code1.splitlines(keepends=True),
            code2.splitlines(keepends=True),
            fromfile=f"v{v1}.py",
            tofile=f"v{v2}.py",
        )
        return "".join(diff)
