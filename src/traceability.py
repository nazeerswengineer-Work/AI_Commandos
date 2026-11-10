"""
Controlled engineering-family expansion for REQTRACE AI.

Phase 11C expands retrieval anchors through explicit source-derived
engineering metadata:

- requirement_id
- observation_id
- test_case_id

A shared filename is not treated as traceability.

A shared function name is not independently treated as traceability.

Expansion does not:
- infer new engineering links
- rewrite observation references
- select an applicable requirement revision
- make an engineering assessment
- determine closure readiness
"""

from __future__ import annotations

from dataclasses import dataclass

from src.corpus import EvidenceCorpus
from src.evidence import EvidenceRecord
from src.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class TraceabilityKeys:
    """Explicit engineering keys used during family expansion."""

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """Return True when no explicit engineering key is available."""

        return not (
            self.requirement_ids
            or self.observation_ids
            or self.test_case_ids
        )


@dataclass(frozen=True)
class TraceabilityExpansionResult:
    """Result of controlled engineering-family expansion."""

    anchors: tuple[RetrievedEvidence, ...]
    expanded: tuple[RetrievedEvidence, ...]
    keys: TraceabilityKeys
    iterations: int

    @property
    def all_evidence(self) -> tuple[RetrievedEvidence, ...]:
        """Return anchors followed by expanded evidence."""

        combined: list[RetrievedEvidence] = []
        seen: set[str] = set()

        for item in self.anchors:
            if item.evidence_id in seen:
                continue

            seen.add(item.evidence_id)
            combined.append(item)

        for item in self.expanded:
            if item.evidence_id in seen:
                continue

            seen.add(item.evidence_id)
            combined.append(item)

        return tuple(combined)


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, duplicate-free, non-empty strings."""

    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        clean_value = str(value).strip()

        if not clean_value:
            continue

        if clean_value in seen:
            continue

        seen.add(clean_value)
        output.append(clean_value)

    return tuple(output)


def collect_traceability_keys(
    evidence: list[RetrievedEvidence]
    | tuple[RetrievedEvidence, ...],
) -> TraceabilityKeys:
    """Collect explicit REQ, OBS and TC identifiers."""

    requirement_ids: list[str] = []
    observation_ids: list[str] = []
    test_case_ids: list[str] = []

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
    )


def merge_traceability_keys(
    first: TraceabilityKeys,
    second: TraceabilityKeys,
) -> TraceabilityKeys:
    """Merge two key sets while preserving deterministic order."""

    requirement_values = list(
        first.requirement_ids
    )

    requirement_values.extend(
        second.requirement_ids
    )

    observation_values = list(
        first.observation_ids
    )

    observation_values.extend(
        second.observation_ids
    )

    test_values = list(
        first.test_case_ids
    )

    test_values.extend(
        second.test_case_ids
    )

    return TraceabilityKeys(
        requirement_ids=ordered_unique(
            requirement_values
        ),
        observation_ids=ordered_unique(
            observation_values
        ),
        test_case_ids=ordered_unique(
            test_values
        ),
    )


def record_matches_keys(
    record: EvidenceRecord,
    keys: TraceabilityKeys,
) -> bool:
    """
    Check whether evidence shares an explicit engineering identifier.

    Filename and function-name equality are deliberately excluded.
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

    checks = (
        requirement_match,
        observation_match,
        test_match,
    )

    return any(checks)


def evidence_record_to_retrieved(
    record: EvidenceRecord,
    rank: int,
) -> RetrievedEvidence:
    """Convert one corpus record into traceability evidence."""

    return RetrievedEvidence(
        evidence_id=(
            record.metadata.evidence_id
        ),
        document_type=(
            record.metadata
            .document_type
            .value
        ),
        content=record.content,
        metadata=(
            record.metadata
            .to_chroma_metadata()
        ),
        retrieval_method="traceability",
        score=1.0,
        rank=rank,
    )


def expand_engineering_family(
    corpus: EvidenceCorpus,
    anchors: list[RetrievedEvidence]
    | tuple[RetrievedEvidence, ...],
    *,
    maximum_iterations: int = 3,
    maximum_results: int = 20,
) -> TraceabilityExpansionResult:
    """
    Expand an evidence family through explicit engineering IDs.

    Expansion is iterative.

    Example:

        REQ-021 anchor
            |
            +--> OBS-031
            +--> TC-201
            +--> validate_command code
                        |
                        +--> Review OBS-031

    Expansion stops when:
    - no new evidence is found
    - no new keys are discovered
    - maximum_iterations is reached
    - maximum_results is reached
    """

    if maximum_iterations < 1:
        raise ValueError(
            "maximum_iterations must be positive."
        )

    if maximum_results < 1:
        raise ValueError(
            "maximum_results must be positive."
        )

    anchor_list: list[RetrievedEvidence] = []
    anchor_ids: set[str] = set()

    for item in anchors:
        if item.evidence_id in anchor_ids:
            continue

        anchor_ids.add(item.evidence_id)
        anchor_list.append(item)

    keys = collect_traceability_keys(
        anchor_list
    )

    if not anchor_list or keys.is_empty:
        return TraceabilityExpansionResult(
            anchors=tuple(anchor_list),
            expanded=(),
            keys=keys,
            iterations=0,
        )

    expanded: list[RetrievedEvidence] = []
    selected_ids: set[str] = set(
        anchor_ids
    )

    completed_iterations = 0

    for iteration in range(
        1,
        maximum_iterations + 1,
    ):
        completed_iterations = iteration

        iteration_results: list[
            RetrievedEvidence
        ] = []

        for record in corpus.records:
            evidence_id = (
                record.metadata.evidence_id
            )

            if evidence_id in selected_ids:
                continue

            if not record_matches_keys(
                record,
                keys,
            ):
                continue

            retrieved = (
                evidence_record_to_retrieved(
                    record,
                    rank=(
                        len(expanded)
                        + len(iteration_results)
                        + 1
                    ),
                )
            )

            iteration_results.append(
                retrieved
            )

            selected_ids.add(
                evidence_id
            )

            if (
                len(expanded)
                + len(iteration_results)
                >= maximum_results
            ):
                break

        if not iteration_results:
            break

        expanded.extend(
            iteration_results
        )

        new_keys = collect_traceability_keys(
            iteration_results
        )

        merged_keys = merge_traceability_keys(
            keys,
            new_keys,
        )

        if merged_keys == keys:
            keys = merged_keys
            break

        keys = merged_keys

        if len(expanded) >= maximum_results:
            break

    reranked_expanded: list[
        RetrievedEvidence
    ] = []

    for rank, item in enumerate(
        expanded,
        start=1,
    ):
        reranked_expanded.append(
            RetrievedEvidence(
                evidence_id=item.evidence_id,
                document_type=(
                    item.document_type
                ),
                content=item.content,
                metadata=item.metadata,
                retrieval_method=(
                    item.retrieval_method
                ),
                score=item.score,
                rank=rank,
            )
        )

    return TraceabilityExpansionResult(
        anchors=tuple(anchor_list),
        expanded=tuple(
            reranked_expanded
        ),
        keys=keys,
        iterations=completed_iterations,
    )