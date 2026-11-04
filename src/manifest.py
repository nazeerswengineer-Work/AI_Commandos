"""
REQTRACE AI Phase 8 ingestion manifest and traceability indexes.

The manifest describes validated evidence without embedding or indexing it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.corpus import EvidenceCorpus
from src.evidence import EvidenceRecord


MANIFEST_VERSION = "1.0"


def evidence_manifest_record(
    evidence: EvidenceRecord,
) -> dict[str, Any]:
    """Create one JSON-safe manifest record."""

    metadata = evidence.metadata

    return {
        "evidence_id": metadata.evidence_id,
        "document_type": (
            metadata.document_type.value
        ),
        "document_name": metadata.document_name,
        "document_revision": (
            metadata.document_revision
        ),
        "page_number": metadata.page_number,
        "requirement_id": metadata.requirement_id,
        "observation_id": metadata.observation_id,
        "test_case_id": metadata.test_case_id,
        "result_status": metadata.result_status,
        "file_name": metadata.file_name,
        "file_path": metadata.file_path,
        "function_name": metadata.function_name,
        "start_line": metadata.start_line,
        "end_line": metadata.end_line,
        "language": metadata.language,
        "chunk_id": metadata.chunk_id,
        "content_hash": metadata.content_hash,
        "parsing_method": (
            metadata.parsing_method.value
        ),
    }


def build_traceability_index(
    corpus: EvidenceCorpus,
) -> dict[str, Any]:
    """
    Build deterministic indexes from explicit parser metadata.

    These indexes do not create new traceability relationships.
    """

    requirements: dict[str, list[str]] = (
        defaultdict(list)
    )
    observations: dict[str, list[str]] = (
        defaultdict(list)
    )
    tests: dict[str, list[str]] = (
        defaultdict(list)
    )
    files: dict[str, list[str]] = (
        defaultdict(list)
    )
    functions: dict[str, list[str]] = (
        defaultdict(list)
    )

    for evidence in corpus.records:
        metadata = evidence.metadata
        evidence_id = metadata.evidence_id

        if metadata.requirement_id:
            requirements[
                metadata.requirement_id
            ].append(evidence_id)

        if metadata.observation_id:
            observations[
                metadata.observation_id
            ].append(evidence_id)

        if metadata.test_case_id:
            tests[
                metadata.test_case_id
            ].append(evidence_id)

        if metadata.file_name:
            files[
                metadata.file_name
            ].append(evidence_id)

        if metadata.function_name:
            functions[
                metadata.function_name
            ].append(evidence_id)

    def normalize_index(
        index: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        return {
            key: sorted(set(values))
            for key, values
            in sorted(index.items())
        }

    return {
        "requirements": normalize_index(
            requirements
        ),
        "observations": normalize_index(
            observations
        ),
        "tests": normalize_index(
            tests
        ),
        "files": normalize_index(
            files
        ),
        "functions": normalize_index(
            functions
        ),
    }


def build_ingestion_manifest(
    corpus: EvidenceCorpus,
) -> dict[str, Any]:
    """Build the complete ingestion manifest."""

    type_counts: dict[str, int] = (
        defaultdict(int)
    )

    for record in corpus.records:
        type_counts[
            record.metadata.document_type.value
        ] += 1

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "record_count": corpus.record_count,
        "document_count": len(
            corpus.document_names
        ),
        "record_counts_by_type": dict(
            sorted(type_counts.items())
        ),
        "duplicate_content_groups": [
            {
                "content_hash": group.content_hash,
                "evidence_ids": list(
                    group.evidence_ids
                ),
            }
            for group
            in corpus.duplicate_content_groups
        ],
        "records": [
            evidence_manifest_record(record)
            for record in corpus.records
        ],
    }


def write_json(
    path: str | Path,
    payload: dict[str, Any],
) -> Path:
    """Write deterministic UTF-8 JSON."""

    output_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def write_corpus_artifacts(
    corpus: EvidenceCorpus,
    data_directory: str | Path,
) -> tuple[Path, Path]:
    """Write manifest and traceability artifacts."""

    data_path = (
        Path(data_directory)
        .expanduser()
        .resolve()
    )

    manifest_path = write_json(
        data_path / "ingestion_manifest.json",
        build_ingestion_manifest(
            corpus
        ),
    )

    traceability_path = write_json(
        data_path / "traceability_index.json",
        build_traceability_index(
            corpus
        ),
    )

    return (
        manifest_path,
        traceability_path,
    )