"""
Central configuration for REQTRACE AI.

All configurable runtime values are loaded from the project-level .env file.
Other modules must import the validated settings from this module instead of
hard-coding model names, paths, thresholds, or feature switches.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Validated application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # Application
    app_name: str = Field(default="REQTRACE AI")
    app_environment: Literal["development", "test", "production"] = Field(
        default="development"
    )

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_llm_model: str = Field(default="llama3.2:1b")
    ollama_embed_model: str = Field(default="nomic-embed-text:latest")

    # Persistent storage
    chroma_db_path: Path = Field(default=Path("chroma_db"))

    # Collection names
    frd_collection_name: str = Field(default="frd_evidence")
    observation_collection_name: str = Field(
        default="observation_evidence"
    )
    code_collection_name: str = Field(default="code_evidence")
    test_collection_name: str = Field(default="test_evidence")
    reviews_collection_name: str = Field(default="review_evidence")

    # LLM
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(default=2048, ge=512, le=32768)
    llm_num_predict: int = Field(default=512, ge=32, le=4096)
    llm_timeout_seconds: int = Field(default=120, ge=10, le=600)

    # Embeddings
    embedding_batch_size: int = Field(default=16, ge=1, le=128)

    # Retrieval
    vector_top_k: int = Field(default=5, ge=1, le=100)
    bm25_top_k: int = Field(default=5, ge=1, le=100)
    fused_top_k: int = Field(default=6, ge=1, le=100)
    final_context_limit: int = Field(default=4, ge=1, le=50)
    min_relevance_score: float = Field(default=0.30, ge=0.0, le=1.0)

    # Feature switches
    enable_hybrid_retrieval: bool = Field(default=True)
    enable_reranking: bool = Field(default=False)
    enable_query_expansion: bool = Field(default=False)
    enable_response_cache: bool = Field(default=True)
    enable_debug_mode: bool = Field(default=False)

    # Logging
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(default="INFO")
    log_file_path: Path = Field(default=Path("logs/reqtrace_ai.log"))

    # Diagnostics
    ollama_connect_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
    )
    ollama_read_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
    )

    @field_validator(
        "app_name",
        "ollama_llm_model",
        "ollama_embed_model",
        "frd_collection_name",
        "observation_collection_name",
        "code_collection_name",
        "test_collection_name",
        "reviews_collection_name",
    )
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject blank configuration values."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Configuration value must not be empty.")

        return cleaned_value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        """Validate and normalize the configured Ollama base URL."""

        normalized_url = value.strip().rstrip("/")

        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError(
                "OLLAMA_BASE_URL must start with http:// or https://."
            )

        return normalized_url

    @field_validator("chroma_db_path", "log_file_path")
    @classmethod
    def resolve_project_paths(cls, value: Path) -> Path:
        """Resolve relative runtime paths against the project directory."""

        if value.is_absolute():
            return value.resolve()

        return (BASE_DIR / value).resolve()

    @field_validator(
        "frd_collection_name",
        "observation_collection_name",
        "code_collection_name",
        "test_collection_name",
        "reviews_collection_name",
    )
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        """Apply simple, predictable naming rules to collections."""

        normalized_name = value.strip().lower()

        allowed_characters = set(
            "abcdefghijklmnopqrstuvwxyz0123456789_-"
        )

        if any(
            character not in allowed_characters
            for character in normalized_name
        ):
            raise ValueError(
                "Collection names may contain only lowercase letters, "
                "numbers, underscores, and hyphens."
            )

        if len(normalized_name) < 3:
            raise ValueError(
                "Collection names must contain at least three characters."
            )

        return normalized_name

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "Settings":
        """Validate rules involving more than one configuration value."""

        collection_names = [
            self.frd_collection_name,
            self.observation_collection_name,
            self.code_collection_name,
            self.test_collection_name,
            self.reviews_collection_name,
        ]

        if len(collection_names) != len(set(collection_names)):
            raise ValueError(
                "Every ChromaDB collection name must be unique."
            )

        if self.final_context_limit > self.fused_top_k:
            raise ValueError(
                "FINAL_CONTEXT_LIMIT must not exceed FUSED_TOP_K."
            )

        return self

    @property
    def documents_path(self) -> Path:
        return BASE_DIR / "documents"

    @property
    def frd_documents_path(self) -> Path:
        return self.documents_path / "frd"

    @property
    def observation_documents_path(self) -> Path:
        return self.documents_path / "observations"

    @property
    def code_documents_path(self) -> Path:
        return self.documents_path / "code"

    @property
    def test_documents_path(self) -> Path:
        return self.documents_path / "tests"

    @property
    def review_documents_path(self) -> Path:
        return self.documents_path / "reviews"

    @property
    def data_path(self) -> Path:
        return BASE_DIR / "data"

    @property
    def logs_path(self) -> Path:
        return self.log_file_path.parent

    @property
    def evaluation_path(self) -> Path:
        return BASE_DIR / "evaluation"

    @property
    def tests_path(self) -> Path:
        return BASE_DIR / "tests"

    def required_directories(self) -> tuple[Path, ...]:

        """Return directories required by the planned project structure."""

        return (
            self.frd_documents_path,
            self.observation_documents_path,
            self.code_documents_path,
            self.test_documents_path,
            self.review_documents_path,
            self.chroma_db_path,
            self.data_path,
            self.logs_path,
            self.evaluation_path,
            self.tests_path,
        )

    def safe_summary(self) -> dict[str, object]:
        """
        Return non-secret settings suitable for diagnostics and logging.

        This method intentionally includes only settings that are safe to
        display. Future secrets must not be added to this dictionary.
        """

        return {
            "app_name": self.app_name,
            "app_environment": self.app_environment,
            "ollama_base_url": self.ollama_base_url,
            "ollama_llm_model": self.ollama_llm_model,
            "ollama_embed_model": self.ollama_embed_model,
            "chroma_db_path": str(self.chroma_db_path),
            "llm_temperature": self.llm_temperature,
            "llm_num_ctx": self.llm_num_ctx,
            "llm_num_predict": self.llm_num_predict,
            "embedding_batch_size": self.embedding_batch_size,
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "fused_top_k": self.fused_top_k,
            "final_context_limit": self.final_context_limit,
            "min_relevance_score": self.min_relevance_score,
            "enable_hybrid_retrieval": self.enable_hybrid_retrieval,
            "enable_reranking": self.enable_reranking,
            "enable_query_expansion": self.enable_query_expansion,
            "enable_response_cache": self.enable_response_cache,
            "enable_debug_mode": self.enable_debug_mode,
            "log_level": self.log_level,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache one validated Settings instance."""

    return Settings()