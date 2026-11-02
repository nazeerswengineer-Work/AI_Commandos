"""
Phase 2 tests for REQTRACE AI configuration.

These tests do not require Ollama, ChromaDB collections, or engineering
documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from config import Settings


def test_default_configuration_is_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "REQTRACE AI"
    assert settings.ollama_llm_model == "llama3.2:1b"
    assert settings.ollama_embed_model == "nomic-embed-text:latest"
    assert settings.llm_temperature == 0.0
    assert settings.enable_hybrid_retrieval is True
    assert settings.enable_reranking is False


def test_relative_paths_are_resolved() -> None:
    settings = Settings(
        _env_file=None,
        chroma_db_path=Path("chroma_db"),
        log_file_path=Path("logs/reqtrace_ai.log"),
    )

    assert settings.chroma_db_path.is_absolute()
    assert settings.log_file_path.is_absolute()


def test_collection_names_are_unique() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            frd_collection_name="duplicate_collection",
            code_collection_name="duplicate_collection",
        )


def test_invalid_ollama_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ollama_base_url="localhost:11434",
        )


def test_blank_model_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ollama_llm_model="   ",
        )


def test_context_limit_cannot_exceed_fused_top_k() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            fused_top_k=4,
            final_context_limit=5,
        )


def test_safe_summary_does_not_include_unknown_secrets() -> None:
    settings = Settings(_env_file=None)
    summary = settings.safe_summary()

    forbidden_names = {
        "password",
        "secret",
        "token",
        "api_key",
    }

    assert forbidden_names.isdisjoint(summary.keys())