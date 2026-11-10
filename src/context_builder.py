"""
Final evidence-context assembly for REQTRACE AI.

Phase 11D / Phase 11E context assembly.

The context builder receives evidence that has already been retrieved
and validated by earlier retrieval stages.

Context priorities:

1. Preserve authoritative exact-ID evidence.
2. Preserve multiple exact FRD revisions for the same requirement.
3. Avoid allowing redundant exact records of the same evidence type
   to consume the complete final context.
4. Prefer the core engineering evidence chain:
       FRD
       Observation
       Code
       Test
5. Add Review evidence after core evidence types when context permits.
6. Prefer controlled traceability evidence before equivalent RRF evidence.
7. Deduplicate using stable evidence IDs.
8. Enforce the configured final context limit.

This module does not:
- perform retrieval
- generate embeddings
- query ChromaDB
- change traceability
- calculate engineering assessment
- calculate closure readiness
- invoke an LLM
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval import RetrievedEvidence


CORE_ENGINEERING_TYPES = (
    "frd",
    "observation",
    "code",
    "test",
)

SECONDARY_ENGINEERING_TYPES = (
    "review",
)


@dataclass(frozen=True)
class FinalEvidenceContext:
    """Final bounded engineering evidence context."""

    evidence: tuple[
        RetrievedEvidence,
        ...
    ]

    exact_count: int
    traceability_count: int
    rrf_count: int

    @property
    def evidence_ids(
        self,
    ) -> tuple[str, ...]:
        """Return final evidence IDs in context order."""

        return tuple(
            item.evidence_id
            for item in self.evidence
        )

    @property
    def document_types(
        self,
    ) -> tuple[str, ...]:
        """Return ordered unique evidence types."""

        return tuple(
            dict.fromkeys(
                item.document_type
                for item in self.evidence
            )
        )

    @property
    def record_count(
        self,
    ) -> int:
        """Return number of context records."""

        return len(
            self.evidence
        )


def deduplicate_evidence(
    items: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    """
    Remove duplicate evidence IDs while preserving input order.

    The first occurrence is retained because upstream retrieval
    has already established evidence priority.
    """

    output: list[
        RetrievedEvidence
    ] = []

    seen: set[str] = set()

    for item in items:

        if item.evidence_id in seen:
            continue

        seen.add(
            item.evidence_id
        )

        output.append(
            item
        )

    return output


def exact_frd_revision_key(
    item: RetrievedEvidence,
) -> tuple[str, str] | None:
    """
    Return the requirement/revision key for an exact FRD record.

    Example:

        REQ-050 / Rev01
        REQ-050 / Rev02

    Both must remain independently protectable.
    """

    if item.document_type != "frd":
        return None

    requirement_id = (
        item.metadata.get(
            "requirement_id"
        )
    )

    revision = (
        item.metadata.get(
            "document_revision"
        )
    )

    if not requirement_id:
        return None

    if not revision:
        return None

    return (
        str(requirement_id),
        str(revision),
    )


def select_protected_exact_evidence(
    exact_results: list[RetrievedEvidence],
    context_limit: int,
) -> list[RetrievedEvidence]:
    """
    Select exact-ID evidence that must be protected.

    Rules:

    1. Preserve one exact record per evidence type.
    2. Preserve additional FRD records when they represent distinct
       revisions of the same requirement.
    3. Do not let redundant exact code records consume the complete
       context.
    4. Never exceed context_limit.

    Example OBS-052:

        Observation
        process_alert code
        reset_alert code
        Review

    becomes initially protected as:

        Observation
        Code
        Review

    leaving room for REQ-041 FRD.

    Example REQ-050:

        FRD Rev01
        FRD Rev02
        Observation

    preserves both FRD revisions.
    """

    if context_limit < 1:
        return []

    exact = deduplicate_evidence(
        exact_results
    )

    if not exact:
        return []

    selected: list[
        RetrievedEvidence
    ] = []

    selected_ids: set[str] = set()
    represented_types: set[str] = set()

    # ---------------------------------------------------------
    # Pass 1
    #
    # Preserve one representative from each exact evidence type.
    # ---------------------------------------------------------

    for item in exact:

        if item.document_type in represented_types:
            continue

        selected.append(
            item
        )

        selected_ids.add(
            item.evidence_id
        )

        represented_types.add(
            item.document_type
        )

        if len(selected) >= context_limit:
            return selected

    # ---------------------------------------------------------
    # Pass 2
    #
    # Preserve additional FRD revisions.
    #
    # This is essential for scenarios such as REQ-050 where
    # Rev01 and Rev02 contain materially different requirements.
    # ---------------------------------------------------------

    selected_revision_keys = {
        key
        for item in selected
        for key in [
            exact_frd_revision_key(
                item
            )
        ]
        if key is not None
    }

    for item in exact:

        if item.evidence_id in selected_ids:
            continue

        revision_key = (
            exact_frd_revision_key(
                item
            )
        )

        if revision_key is None:
            continue

        if (
            revision_key
            in selected_revision_keys
        ):
            continue

        selected.append(
            item
        )

        selected_ids.add(
            item.evidence_id
        )

        selected_revision_keys.add(
            revision_key
        )

        if len(selected) >= context_limit:
            break

    return selected[
        :context_limit
    ]


def choose_diverse_additions(
    candidates: list[RetrievedEvidence],
    existing: list[RetrievedEvidence],
    limit: int,
) -> list[RetrievedEvidence]:
    """
    Select additional evidence using engineering-type priority.

    Preferred core evidence chain:

        FRD
        Observation
        Code
        Test

    Review is secondary after missing core evidence types.

    Existing protected evidence is never removed.
    """

    if limit < 1:
        return []

    existing_ids = {
        item.evidence_id
        for item in existing
    }

    represented_types = {
        item.document_type
        for item in existing
    }

    remaining: list[
        RetrievedEvidence
    ] = []

    for item in candidates:

        if item.evidence_id in existing_ids:
            continue

        remaining.append(
            item
        )

    remaining = deduplicate_evidence(
        remaining
    )

    selected: list[
        RetrievedEvidence
    ] = []

    selected_ids: set[str] = set()

    # ---------------------------------------------------------
    # Pass 1
    #
    # Complete missing core engineering evidence types.
    # ---------------------------------------------------------

    for required_type in CORE_ENGINEERING_TYPES:

        if required_type in represented_types:
            continue

        matching_item = next(
            (
                item
                for item in remaining
                if (
                    item.document_type
                    == required_type
                    and item.evidence_id
                    not in selected_ids
                )
            ),
            None,
        )

        if matching_item is None:
            continue

        selected.append(
            matching_item
        )

        selected_ids.add(
            matching_item.evidence_id
        )

        represented_types.add(
            matching_item.document_type
        )

        if len(selected) >= limit:
            return selected

    # ---------------------------------------------------------
    # Pass 2
    #
    # Add secondary evidence such as engineering review.
    # ---------------------------------------------------------

    for secondary_type in SECONDARY_ENGINEERING_TYPES:

        if secondary_type in represented_types:
            continue

        matching_item = next(
            (
                item
                for item in remaining
                if (
                    item.document_type
                    == secondary_type
                    and item.evidence_id
                    not in selected_ids
                )
            ),
            None,
        )

        if matching_item is None:
            continue

        selected.append(
            matching_item
        )

        selected_ids.add(
            matching_item.evidence_id
        )

        represented_types.add(
            matching_item.document_type
        )

        if len(selected) >= limit:
            return selected

    # ---------------------------------------------------------
    # Pass 3
    #
    # Fill remaining positions in original ranking order.
    # ---------------------------------------------------------

    for item in remaining:

        if item.evidence_id in selected_ids:
            continue

        selected.append(
            item
        )

        selected_ids.add(
            item.evidence_id
        )

        if len(selected) >= limit:
            break

    return selected


def assign_context_ranks(
    evidence: list[RetrievedEvidence],
) -> tuple[
    RetrievedEvidence,
    ...
]:
    """Assign final one-based context ranks."""

    ranked: list[
        RetrievedEvidence
    ] = []

    for rank, item in enumerate(
        evidence,
        start=1,
    ):

        ranked.append(
            RetrievedEvidence(
                evidence_id=(
                    item.evidence_id
                ),
                document_type=(
                    item.document_type
                ),
                content=(
                    item.content
                ),
                metadata=(
                    item.metadata
                ),
                retrieval_method=(
                    item.retrieval_method
                ),
                score=(
                    item.score
                ),
                rank=rank,
            )
        )

    return tuple(
        ranked
    )


def build_final_context(
    *,
    exact_results: list[RetrievedEvidence],
    traceability_results: list[RetrievedEvidence],
    rrf_results: list[RetrievedEvidence],
    context_limit: int = 4,
) -> FinalEvidenceContext:
    """
    Build the final bounded engineering evidence context.

    Strategy:

    1. Select exact evidence that truly requires protection.
    2. Preserve distinct FRD revision variants.
    3. Combine traceability and RRF candidates.
    4. Fill missing core engineering evidence types.
    5. Add review evidence when context permits.
    6. Deduplicate and enforce the final context limit.

    This prevents redundant exact records from suppressing important
    evidence such as an explicit FRD requirement.
    """

    if context_limit < 1:
        raise ValueError(
            "context_limit must be positive."
        )

    exact = deduplicate_evidence(
        exact_results
    )

    traceability = (
        deduplicate_evidence(
            traceability_results
        )
    )

    rrf = deduplicate_evidence(
        rrf_results
    )

    # ---------------------------------------------------------
    # Stage 1
    #
    # Protect authoritative exact evidence, while avoiding
    # redundant same-type exact records.
    # ---------------------------------------------------------

    protected_exact = (
        select_protected_exact_evidence(
            exact,
            context_limit,
        )
    )

    final: list[
        RetrievedEvidence
    ] = list(
        protected_exact
    )

    # ---------------------------------------------------------
    # Stage 2
    #
    # Build a single candidate pool.
    #
    # Traceability evidence comes before RRF evidence, while
    # choose_diverse_additions() can still select an RRF record
    # when that record supplies a missing core evidence type.
    # ---------------------------------------------------------

    combined_candidates: list[
        RetrievedEvidence
    ] = []

    combined_candidates.extend(
        traceability
    )

    combined_candidates.extend(
        rrf
    )

    # Include exact records that were not protected.
    #
    # They remain candidates rather than disappearing completely.
    # This can be useful when free context positions remain.

    protected_ids = {
        item.evidence_id
        for item in protected_exact
    }

    for item in exact:

        if item.evidence_id in protected_ids:
            continue

        combined_candidates.append(
            item
        )

    combined_candidates = (
        deduplicate_evidence(
            combined_candidates
        )
    )

    # ---------------------------------------------------------
    # Stage 3
    #
    # Fill the remaining context positions.
    # ---------------------------------------------------------

    available_slots = (
        context_limit
        - len(final)
    )

    additions: list[
        RetrievedEvidence
    ] = []

    if available_slots > 0:

        additions = (
            choose_diverse_additions(
                combined_candidates,
                final,
                available_slots,
            )
        )

        final.extend(
            additions
        )

    final = deduplicate_evidence(
        final
    )

    final = final[
        :context_limit
    ]

    # ---------------------------------------------------------
    # Stage 4
    #
    # Count final evidence origins.
    # ---------------------------------------------------------

    exact_ids = {
        item.evidence_id
        for item in exact
    }

    traceability_ids = {
        item.evidence_id
        for item in traceability
    }

    rrf_ids = {
        item.evidence_id
        for item in rrf
    }

    used_exact_count = 0
    used_traceability_count = 0
    used_rrf_count = 0

    for item in final:

        if item.evidence_id in exact_ids:

            used_exact_count += 1

        elif (
            item.evidence_id
            in traceability_ids
        ):

            used_traceability_count += 1

        elif item.evidence_id in rrf_ids:

            used_rrf_count += 1

    return FinalEvidenceContext(
        evidence=assign_context_ranks(
            final
        ),
        exact_count=(
            used_exact_count
        ),
        traceability_count=(
            used_traceability_count
        ),
        rrf_count=(
            used_rrf_count
        ),
    )