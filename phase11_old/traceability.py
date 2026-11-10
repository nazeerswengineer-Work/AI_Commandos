"""
Explicit traceability expansion for REQTRACE AI.

Traceability expansion operates only on source-derived metadata.

Important rules:

- requirement IDs may create traceability relationships
- observation IDs may create traceability relationships
- test-case IDs may create traceability relationships
- function names may create traceability relationships

A shared file name alone does not establish an engineering relationship.

Example:

    alert_manager.c

contains both:

    process_alert()
    reset_alert()

Those functions belong to different engineering scenarios.
Sharing one source file must therefore not merge the two evidence families.

This module does not:

- infer new permanent engineering links
- correct observation requirement references
- use semantic similarity as traceability authority
- make engineering assessments
"""

from __future__ import annotations

from dataclasses import dataclass

from src.corpus import EvidenceCorpus
from src.evidence import EvidenceRecord
from src.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class TraceabilityKeys:
    """Explicit engineering keys collected from retrieved evidence."""

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    file_names: tuple[str, ...]
    function_names: tuple[str, ...]


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered duplicate-free non-empty strings."""

    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return tuple(output)


def collect_traceability_keys(
    evidence: list[RetrievedEvidence]
    | tuple[RetrievedEvidence, ...],
) -> TraceabilityKeys:
    """
    Collect explicit metadata keys from retrieved evidence.

    File names are retained for diagnostics but are deliberately not used
    by record_matches_keys() as standalone traceability evidence.
    """

    requirement_ids: list[str] = []
    observation_ids: list[str] = []
    test_case_ids: list[str] = []
    file_names: list[str] = []
    function_names: list[str] = []

    for item in evidence:
        metadata = item.metadata

        requirement_id = metadata.get(
            "requirement_id"
        )

        observation_id = metadata.get(
            "observation_id"
        )

        test_case_id = metadata.get(
            "test_case_id"
        )

        file_name = metadata.get(
            "file_name"
        )

        function_name = metadata.get(
            "function_name"
        )

        if requirement_id:
            requirement_ids.append(
                str(requirement_id)
            )

        if observation_id:
            observation_ids.append(
                str(observation_id)
            )

        if test_case_id:
            test_case_ids.append(
                str(test_case_id)
            )

        if file_name:
            file_names.append(
                str(file_name)
            )

        if function_name:
            function_names.append(
                str(function_name)
            )

    return TraceabilityKeys(
        requirement_ids=ordered_unique(
            requirement_ids
        ),
        observation_ids=ordered_unique(
            observation_ids
        ),
        test_case_ids=ordered_unique(
            test_case_ids
        ),
        file_names=ordered_unique(
            file_names
        ),
        function_names=ordered_unique(
            function_names
        ),
    )


def record_matches_keys(
    record: EvidenceRecord,
    keys: TraceabilityKeys,
) -> bool:
    """
    Check whether a corpus record shares an explicit engineering key.

    File-name equality is deliberately excluded.

    Sharing the same source file does not prove that two functions belong
    to the same requirement or observation family.
    """

    metadata = record.metadata

    requirement_match = bool(
        metadata.requirement_id
        and metadata.requirement_id
        in keys.requirement_ids
    )

    observation_match = bool(
        metadata.observation_id
        and metadata.observation_id
        in keys.observation_ids
    )

    test_match = bool(
        metadata.test_case_id
        and metadata.test_case_id
        in keys.test_case_ids
    )

    function_match = bool(
        metadata.function_name
        and metadata.function_name
        in keys.function_names
    )

    checks = (
        requirement_match,
        observation_match,
        test_match,
        function_match,
    )

    return any(checks)


def expand_traceability(
    corpus: EvidenceCorpus,
    seed_evidence: list[RetrievedEvidence]
    | tuple[RetrievedEvidence, ...],
    *,
    limit: int = 20,
) -> list[RetrievedEvidence]:
    """
    Expand seed evidence through explicit corpus metadata.

    Expansion preserves source metadata.

    Example:

        OBS-052 explicitly references REQ-041.

    Phase 11 does not rewrite that relationship to REQ-042 merely because
    REQ-042 is semantically similar to the observation description.
    """

    if limit < 1:
        raise ValueError(
            "Traceability expansion limit must be positive."
        )

    if not seed_evidence:
        return []

    keys = collect_traceability_keys(
        seed_evidence
    )

    seed_ids = {
        item.evidence_id
        for item in seed_evidence
    }

    expanded: list[
        RetrievedEvidence
    ] = []

    for record in corpus.records:
        evidence_id = (
            record.metadata.evidence_id
        )

        if evidence_id in seed_ids:
            continue

        if not record_matches_keys(
            record,
            keys,
        ):
            continue

        metadata = (
            record.metadata
            .to_chroma_metadata()
        )

        expanded.append(
            RetrievedEvidence(
                evidence_id=evidence_id,
                document_type=(
                    record.metadata
                    .document_type
                    .value
                ),
                content=record.content,
                metadata=metadata,
                retrieval_method=(
                    "traceability"
                ),
                score=1.0,
                rank=(
                    len(expanded) + 1
                ),
            )
        )

        if len(expanded) >= limit:
            break

    return expanded