"""
Retrieval evaluation for REQTRACE AI.

Phase 11E evaluates final evidence contexts produced by the retrieval
pipeline.

Metrics include:

- expected engineering ID coverage
- required evidence-type coverage
- Precision@K
- Recall@K
- Reciprocal Rank
- scenario pass/fail

This module does not perform retrieval itself and does not invoke an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.context_builder import FinalEvidenceContext


@dataclass(frozen=True)
class RetrievalExpectation:
    """Expected evidence properties for one evaluation scenario."""

    case_id: str
    query: str

    requirement_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    test_case_ids: tuple[str, ...] = ()

    required_document_types: tuple[str, ...] = ()

    minimum_id_coverage: float = 1.0
    minimum_type_coverage: float = 1.0


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Evaluation metrics for one retrieval scenario."""

    case_id: str
    query: str

    requirement_coverage: float
    observation_coverage: float
    test_coverage: float
    type_coverage: float

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float

    passed: bool


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate evaluation result."""

    results: tuple[
        RetrievalEvaluationResult,
        ...
    ]

    passed_cases: int
    failed_cases: int

    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_type_coverage: float

    @property
    def all_passed(self) -> bool:
        return self.failed_cases == 0


def coverage(
    actual: set[str],
    expected: tuple[str, ...],
) -> float:
    """Calculate coverage of expected values."""

    if not expected:
        return 1.0

    expected_set = set(
        expected
    )

    return (
        len(
            actual
            & expected_set
        )
        / len(expected_set)
    )


def collect_context_ids(
    context: FinalEvidenceContext,
) -> tuple[
    set[str],
    set[str],
    set[str],
    set[str],
]:
    """Collect engineering identifiers and evidence types."""

    requirements: set[str] = set()
    observations: set[str] = set()
    tests: set[str] = set()
    document_types: set[str] = set()

    for item in context.evidence:
        metadata = item.metadata

        document_types.add(
            item.document_type
        )

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
            requirements.add(
                str(requirement_id)
            )

        if observation_id:
            observations.add(
                str(observation_id)
            )

        if test_case_id:
            tests.add(
                str(test_case_id)
            )

    return (
        requirements,
        observations,
        tests,
        document_types,
    )


def relevant_evidence_mask(
    context: FinalEvidenceContext,
    expectation: RetrievalExpectation,
) -> tuple[bool, ...]:
    """
    Determine whether each final-context record belongs to the labelled
    engineering family.
    """

    relevant: list[bool] = []

    expected_requirements = set(
        expectation.requirement_ids
    )

    expected_observations = set(
        expectation.observation_ids
    )

    expected_tests = set(
        expectation.test_case_ids
    )

    for item in context.evidence:
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

        matches = bool(
            (
                requirement_id
                and requirement_id
                in expected_requirements
            )
            or
            (
                observation_id
                and observation_id
                in expected_observations
            )
            or
            (
                test_case_id
                and test_case_id
                in expected_tests
            )
        )

        relevant.append(
            matches
        )

    return tuple(relevant)


def precision_at_k(
    relevance: tuple[bool, ...],
) -> float:
    """Calculate precision over the final context."""

    if not relevance:
        return 0.0

    return (
        sum(
            1
            for relevant in relevance
            if relevant
        )
        / len(relevance)
    )


def reciprocal_rank(
    relevance: tuple[bool, ...],
) -> float:
    """Calculate reciprocal rank of first relevant item."""

    for rank, relevant in enumerate(
        relevance,
        start=1,
    ):
        if relevant:
            return 1.0 / rank

    return 0.0


def expected_id_count(
    expectation: RetrievalExpectation,
) -> int:
    """Return number of unique expected engineering IDs."""

    values = set(
        expectation.requirement_ids
    )

    values.update(
        expectation.observation_ids
    )

    values.update(
        expectation.test_case_ids
    )

    return len(values)


def retrieved_expected_ids(
    context: FinalEvidenceContext,
    expectation: RetrievalExpectation,
) -> set[str]:
    """Return expected engineering IDs found in final context."""

    expected = set(
        expectation.requirement_ids
    )

    expected.update(
        expectation.observation_ids
    )

    expected.update(
        expectation.test_case_ids
    )

    actual: set[str] = set()

    for item in context.evidence:
        metadata = item.metadata

        for field_name in (
            "requirement_id",
            "observation_id",
            "test_case_id",
        ):
            value = metadata.get(
                field_name
            )

            if value and str(value) in expected:
                actual.add(
                    str(value)
                )

    return actual


def evaluate_context(
    context: FinalEvidenceContext,
    expectation: RetrievalExpectation,
) -> RetrievalEvaluationResult:
    """Evaluate one final context."""

    (
        requirements,
        observations,
        tests,
        document_types,
    ) = collect_context_ids(
        context
    )

    requirement_coverage = coverage(
        requirements,
        expectation.requirement_ids,
    )

    observation_coverage = coverage(
        observations,
        expectation.observation_ids,
    )

    test_coverage = coverage(
        tests,
        expectation.test_case_ids,
    )

    type_coverage = coverage(
        document_types,
        expectation.required_document_types,
    )

    expected_count = expected_id_count(
        expectation
    )

    actual_expected_ids = (
        retrieved_expected_ids(
            context,
            expectation,
        )
    )

    if expected_count:
        recall = (
            len(actual_expected_ids)
            / expected_count
        )
    else:
        recall = 1.0

    relevance = relevant_evidence_mask(
        context,
        expectation,
    )

    precision = precision_at_k(
        relevance
    )

    rr = reciprocal_rank(
        relevance
    )

    id_coverages = (
        requirement_coverage,
        observation_coverage,
        test_coverage,
    )

    applicable_id_coverages = []

    if expectation.requirement_ids:
        applicable_id_coverages.append(
            requirement_coverage
        )

    if expectation.observation_ids:
        applicable_id_coverages.append(
            observation_coverage
        )

    if expectation.test_case_ids:
        applicable_id_coverages.append(
            test_coverage
        )

    minimum_id_result = (
        min(applicable_id_coverages)
        if applicable_id_coverages
        else 1.0
    )

    passed = bool(
        minimum_id_result
        >= expectation.minimum_id_coverage
        and type_coverage
        >= expectation.minimum_type_coverage
    )

    return RetrievalEvaluationResult(
        case_id=expectation.case_id,
        query=expectation.query,
        requirement_coverage=(
            requirement_coverage
        ),
        observation_coverage=(
            observation_coverage
        ),
        test_coverage=(
            test_coverage
        ),
        type_coverage=type_coverage,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=rr,
        passed=passed,
    )


def build_evaluation_report(
    results: list[
        RetrievalEvaluationResult
    ],
) -> RetrievalEvaluationReport:
    """Create aggregate retrieval metrics."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    result_tuple = tuple(
        results
    )

    count = len(
        result_tuple
    )

    passed_cases = sum(
        1
        for item in result_tuple
        if item.passed
    )

    failed_cases = (
        count - passed_cases
    )

    return RetrievalEvaluationReport(
        results=result_tuple,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        mean_precision_at_k=(
            sum(
                item.precision_at_k
                for item in result_tuple
            )
            / count
        ),
        mean_recall_at_k=(
            sum(
                item.recall_at_k
                for item in result_tuple
            )
            / count
        ),
        mean_reciprocal_rank=(
 sum(
                item.reciprocal_rank
                for item in result_tuple
            )
            / count
        ),
        mean_type_coverage=(
            sum(
                item.type_coverage
                for item in result_tuple
            )
            / count
        ),
    )