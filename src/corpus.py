"""
REQTRACE AI Phase 8 evidence-corpus assembly.

Combines validated EvidenceRecord objects from all evidence parsers.

This module does not:
- generate embeddings
- access ChromaDB
- invoke an LLM
- make engineering assessments
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.code_parser import parse_code_directory
from src.document_parser import parse_frd_directory
from src.evidence import EvidenceRecord
from src.observation_parser import parse_observation_directory
from src.test_review_parser import (
    parse_review_directory,
    parse_test_directory,
)


class CorpusValidationError(RuntimeError):
    """Raised when corpus integrity cannot be guaranteed."""


@dataclass(frozen=True)
class DuplicateContentGroup:
    """Records containing identical normalized content."""

    content_hash: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCorpus:
    """Validated evidence corpus."""

    records: tuple[EvidenceRecord, ...]
    duplicate_content_groups: tuple[DuplicateContentGroup, ...]

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            record.metadata.evidence_id
            for record in self.records
        )

    @property
    def document_names(self) -> tuple[str, ...]:
        seen: set[str] = set()
        names: list[str] = []

        for record in self.records:
            name = record.metadata.document_name

            if name not in seen:
                seen.add(name)
                names.append(name)

        return tuple(names)

    def by_document_type(
        self,
        document_type: str,
    ) -> tuple[EvidenceRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.metadata.document_type.value
            == document_type
        )

    def by_evidence_id(
        self,
        evidence_id: str,
    ) -> EvidenceRecord | None:
        for record in self.records:
            if (
                record.metadata.evidence_id
                == evidence_id
            ):
                return record

        return None


def flatten_parser_results(
    results: tuple[object, ...],
) -> list[EvidenceRecord]:
    """Extract EvidenceRecord objects from parser results."""

    records: list[EvidenceRecord] = []

    for result in results:
        evidence_records = getattr(
            result,
            "evidence_records",
            None,
        )

        if evidence_records is None:
            raise CorpusValidationError(
                "Parser result does not provide evidence_records."
            )

        records.extend(evidence_records)

    return records


def validate_unique_evidence_ids(
    records: list[EvidenceRecord],
) -> None:
    """
    Ensure an evidence ID identifies exactly one evidence record.

    Equal IDs are never silently overwritten.
    """

    seen: dict[str, str] = {}

    for record in records:
        evidence_id = (
            record.metadata.evidence_id
        )
        content_hash = (
            record.metadata.content_hash
        )

        if evidence_id not in seen:
            seen[evidence_id] = content_hash
            continue

        previous_hash = seen[evidence_id]

        if previous_hash == content_hash:
            raise CorpusValidationError(
                "Duplicate evidence ID detected: "
                f"{evidence_id}"
            )

        raise CorpusValidationError(
            "Evidence ID collision detected with "
            "different content: "
            f"{evidence_id}"
        )


def detect_duplicate_content(
    records: list[EvidenceRecord],
) -> tuple[DuplicateContentGroup, ...]:
    """
    Detect identical normalized content across different evidence IDs.

    Duplicate content is reported rather than silently deleted because
    identical text may legitimately appear in distinct revisions or sources.
    """

    by_hash: dict[str, list[str]] = (
        defaultdict(list)
    )

    for record in records:
        by_hash[
            record.metadata.content_hash
        ].append(
            record.metadata.evidence_id
        )

    groups: list[DuplicateContentGroup] = []

    for content_hash, evidence_ids in sorted(
        by_hash.items()
    ):
        if len(evidence_ids) <= 1:
            continue

        groups.append(
            DuplicateContentGroup(
                content_hash=content_hash,
                evidence_ids=tuple(
                    sorted(evidence_ids)
                ),
            )
        )

    return tuple(groups)


def sort_evidence_records(
    records: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Create deterministic corpus order."""

    return sorted(
        records,
        key=lambda record: (
            record.metadata.document_type.value,
            record.metadata.document_name.lower(),
            record.metadata.page_number or 0,
            record.metadata.start_line or 0,
            record.metadata.evidence_id,
        ),
    )


def build_evidence_corpus(
    *,
    frd_directory: str | Path,
    observation_directory: str | Path,
    code_directory: str | Path,
    test_directory: str | Path,
    review_directory: str | Path,
) -> EvidenceCorpus:
    """Parse all evidence categories and assemble one corpus."""

    records: list[EvidenceRecord] = []

    records.extend(
        flatten_parser_results(
            parse_frd_directory(
                frd_directory
            )
        )
    )

    records.extend(
        flatten_parser_results(
            parse_observation_directory(
                observation_directory
            )
        )
    )

    records.extend(
        flatten_parser_results(
            parse_code_directory(
                code_directory
            )
        )
    )

    records.extend(
        flatten_parser_results(
            parse_test_directory(
                test_directory
            )
        )
    )

    records.extend(
        flatten_parser_results(
            parse_review_directory(
                review_directory
            )
        )
    )

    validate_unique_evidence_ids(
        records
    )

    sorted_records = sort_evidence_records(
        records
    )

    duplicates = detect_duplicate_content(
        sorted_records
    )

    return EvidenceCorpus(
        records=tuple(sorted_records),
        duplicate_content_groups=duplicates,
    )


def build_project_corpus(
    base_directory: str | Path,
) -> EvidenceCorpus:
    """Build the corpus using standard project directories."""

    base_path = (
        Path(base_directory)
        .expanduser()
        .resolve()
    )

    documents = base_path / "documents"

    return build_evidence_corpus(
        frd_directory=documents / "frd",
        observation_directory=(
            documents / "observations"
        ),
        code_directory=documents / "code",
        test_directory=documents / "tests",
        review_directory=documents / "reviews",
    )