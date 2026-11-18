"""
Inspect REQTRACE AI's existing embedding and ChromaDB integration.

This command is read-only.

It scans Python source files for:

- Ollama embedding configuration
- embedding model names
- ChromaDB client construction
- persistence directories
- collection names
- collection query/upsert/delete calls
- query and document embedding functions
- retrieval entry points

It does not display document content, environment values, embeddings,
or confidential engineering evidence.
"""

from __future__ import annotations

import ast
import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "chroma_db",
    "data",
    "documents",
}


SEARCH_TERMS = (
    "PersistentClient",
    "HttpClient",
    "EphemeralClient",
    "Client",
    "get_or_create_collection",
    "get_collection",
    "create_collection",
    "delete_collection",
    "collection.add",
    "collection.upsert",
    "collection.update",
    "collection.delete",
    "collection.query",
    ".add(",
    ".upsert(",
    ".delete(",
    ".query(",
    "embed_documents",
    "embed_document",
    "embed_query",
    "embed_text",
    "embeddings",
    "nomic-embed-text",
    "ollama",
    "chroma",
    "collection_name",
    "persist_directory",
    "embedding_model",
)


SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|token|password|secret)"
        r"\s*=\s*['\"][^'\"]+['\"]"
    ),
    re.compile(
        r"(?i)(authorization\s*[:=]\s*)"
        r"['\"][^'\"]+['\"]"
    ),
)


@dataclass(frozen=True)
class SourceMatch:
    """One relevant source-code line."""

    file: str
    line: int
    text: str
    matched_terms: tuple[str, ...]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the match into JSON-safe data."""

        return {
            "file": self.file,
            "line": self.line,
            "text": self.text,
            "matched_terms": list(
                self.matched_terms
            ),
        }


@dataclass(frozen=True)
class PythonModuleSummary:
    """AST-level module summary."""

    file: str
    imports: tuple[str, ...]
    classes: tuple[str, ...]
    functions: tuple[str, ...]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the summary into JSON-safe data."""

        return {
            "file": self.file,
            "imports": list(
                self.imports
            ),
            "classes": list(
                self.classes
            ),
            "functions": list(
                self.functions
            ),
        }


def relative_path(
    path: Path,
) -> str:
    """Return a portable project-relative path."""

    return (
        path
        .resolve()
        .relative_to(
            PROJECT_ROOT
        )
        .as_posix()
    )


def should_exclude(
    path: Path,
) -> bool:
    """Return whether a path belongs to an excluded directory."""

    try:
        parts = (
            path
            .resolve()
            .relative_to(
                PROJECT_ROOT
            )
            .parts
        )

    except ValueError:
        return True

    return any(
        part in EXCLUDED_DIRECTORIES
        for part in parts
    )


def python_files() -> tuple[Path, ...]:
    """Return project Python files in deterministic order."""

    files = [
        path
        for path in PROJECT_ROOT.rglob(
            "*.py"
        )
        if not should_exclude(
            path
        )
    ]

    return tuple(
        sorted(
            files,
            key=lambda value: (
                value.as_posix().lower()
            ),
        )
    )


def sanitize_line(
    line: str,
) -> str:
    """Remove possible credentials from diagnostic output."""

    sanitized = line.strip()

    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub(
            r"\1=<REDACTED>",
            sanitized,
        )

    return sanitized


def scan_source_matches(
    path: Path,
) -> tuple[SourceMatch, ...]:
    """Find relevant embedding/vector-store lines."""

    try:
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()

    except UnicodeDecodeError:
        return ()

    matches: list[
        SourceMatch
    ] = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        lowered = line.lower()

        matched_terms = tuple(
            term
            for term in SEARCH_TERMS
            if term.lower() in lowered
        )

        if not matched_terms:
            continue

        matches.append(
            SourceMatch(
                file=relative_path(
                    path
                ),
                line=line_number,
                text=sanitize_line(
                    line
                ),
                matched_terms=(
                    matched_terms
                ),
            )
        )

    return tuple(
        matches
    )


def ast_summary(
    path: Path,
) -> PythonModuleSummary | None:
    """Extract imports, classes, and functions without executing code."""

    try:
        source = path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            source,
            filename=str(
                path
            ),
        )

    except (
        UnicodeDecodeError,
        SyntaxError,
    ):
        return None

    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            module = (
                node.module
                or ""
            )

            imports.append(
                module
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            classes.append(
                node.name
            )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(
                node.name
            )

    return PythonModuleSummary(
        file=relative_path(
            path
        ),
        imports=tuple(
            sorted(
                set(
                    imports
                )
            )
        ),
        classes=tuple(
            sorted(
                set(
                    classes
                )
            )
        ),
        functions=tuple(
            sorted(
                set(
                    functions
                )
            )
        ),
    )


def important_module(
    summary: PythonModuleSummary,
    matches_by_file: dict[
        str,
        tuple[
            SourceMatch,
            ...
        ],
    ],
) -> bool:
    """Return whether a module is relevant to vector integration."""

    if summary.file in matches_by_file:
        return True

    keywords = (
        "embed",
        "retriev",
        "vector",
        "chroma",
        "index",
        "corpus",
    )

    combined = " ".join(
        (
            summary.file,
            *summary.imports,
            *summary.classes,
            *summary.functions,
        )
    ).lower()

    return any(
        keyword in combined
        for keyword in keywords
    )


def project_path_status() -> dict[str, Any]:
    """Report likely persistent storage locations without reading content."""

    candidates = (
        "chroma_db",
        "data/chroma_db",
        "data/chroma",
        "vector_db",
        "vectorstore",
        "documents",
        "data/ingestion_manifest.json",
    )

    output: dict[
        str,
        Any,
    ] = {}

    for candidate in candidates:
        path = (
            PROJECT_ROOT
            / candidate
        )

        output[
            candidate
        ] = {
            "exists": path.exists(),
            "is_directory": path.is_dir(),
            "is_file": path.is_file(),
        }

    return output


def main() -> int:
    """Generate a sanitized integration report."""

    files = python_files()

    all_matches: list[
        SourceMatch
    ] = []

    summaries: list[
        PythonModuleSummary
    ] = []

    for path in files:
        all_matches.extend(
            scan_source_matches(
                path
            )
        )

        summary = ast_summary(
            path
        )

        if summary is not None:
            summaries.append(
                summary
            )

    matches_by_file: dict[
        str,
        tuple[
            SourceMatch,
            ...
        ],
    ] = {}

    for file_name in sorted(
        {
            match.file
            for match in all_matches
        }
    ):
        matches_by_file[
            file_name
        ] = tuple(
            match
            for match in all_matches
            if match.file == file_name
        )

    relevant_summaries = tuple(
        summary
        for summary in summaries
        if important_module(
            summary,
            matches_by_file,
        )
    )

    payload = {
        "project_root": str(
            PROJECT_ROOT
        ),
        "python_files_scanned": len(
            files
        ),
        "relevant_module_count": len(
            relevant_summaries
        ),
        "storage_paths": (
            project_path_status()
        ),
        "modules": [
            summary.to_dict()
            for summary in relevant_summaries
        ],
        "source_matches": [
            match.to_dict()
            for match in all_matches
        ],
    }

    output_path = (
        PROJECT_ROOT
        / "vector_integration_report.json"
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - VECTOR INTEGRATION INSPECTION"
    )
    print("=" * 78)

    print(
        f"Python files scanned : {len(files)}"
    )

    print(
        "Relevant modules     : "
        f"{len(relevant_summaries)}"
    )

    print(
        "Source matches       : "
        f"{len(all_matches)}"
    )

    print(
        f"Report               : {output_path}"
    )

    print()
    print(
        "No project modules were imported or executed."
    )

    print(
        "No document content, embeddings, or environment "
        "secrets were included."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )