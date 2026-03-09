"""
EDA Configuration — loads settings from environment and provides defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class CompilerConfig(BaseModel):
    """Settings for the LLM compiler (Phase 1)."""

    model: str = Field(
        default_factory=lambda: os.getenv("EDA_COMPILER_MODEL", "gemini-2.0-flash")
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("EDA_MAX_RETRIES", "3"))
    )
    sandbox_timeout: int = Field(
        default_factory=lambda: int(os.getenv("EDA_SANDBOX_TIMEOUT", "30"))
    )
    temperature: float = 0.1  # Low temp for deterministic code generation


class RouterConfig(BaseModel):
    """Settings for the query router (Phase 3)."""

    model: str = Field(
        default_factory=lambda: os.getenv("EDA_ROUTER_MODEL", "gemini-2.0-flash-lite")
    )
    temperature: float = 0.0  # Zero temp for deterministic routing


class Config(BaseModel):
    """Top-level configuration."""

    # API Keys
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Paths
    artifacts_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("EDA_ARTIFACTS_DIR", "./artifacts"))
    )

    # Sub-configs
    compiler: CompilerConfig = Field(default_factory=CompilerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)

    def get_active_provider(self) -> str:
        """Determine which LLM provider is configured."""
        if self.google_api_key:
            return "google"
        elif self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        raise ValueError(
            "No LLM API key configured. Set GOOGLE_API_KEY, OPENAI_API_KEY, "
            "or ANTHROPIC_API_KEY in your .env file."
        )


# Singleton config instance
config = Config()
