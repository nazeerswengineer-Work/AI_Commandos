"""
Reusable grounded-answer pipeline for REQTRACE AI.

Pipeline:

Phase 11 retrieval
Phase 12 deterministic decision
Phase 13A citation construction
Phase 13B grounding validation
Phase 13C safe answer contract
Phase 13D deterministic answer package
Phase 13E adversarial package validation

No LLM is invoked.
"""

from __future__ import annotations

from pathlib import Path

from config import Settings

from src.adversarial_validation import (
    AdversarialValidationResult,
    validate_answer_package_adversarially,
)
from src.answer_builder import (
    EngineeringAnswerPackage,
    build_answer_package,
)
from src.answer_contract import (
    EngineeringAnswerContract,
    build_answer_contract,
)
from src.citations import (
    build_citation_collection,
)
from src.decision_engine import (
    evaluate_engineering_decision,
)
from src.grounding import (
    validate_grounding,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def build_grounded_answer_artifacts(
    query: str,
    settings: Settings,
    base_directory: str | Path,
    *,
    evaluate_closure: bool = False,
    maximum_excerpt_characters: int = 500,
) -> tuple[
    EngineeringAnswerPackage,
    EngineeringAnswerContract,
    AdversarialValidationResult,
]:
    """
    Build and strictly validate all Phase 13 answer artifacts.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Query must not be empty."
        )

    context = retrieve_final_context(
        clean_query,
        settings,
        base_directory,
    )

    decision = evaluate_engineering_decision(
        context,
        evaluate_closure=(
            evaluate_closure
        ),
    )

    citations = build_citation_collection(
        context
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=citations,
        strict=True,
    )

    contract = build_answer_contract(
        decision=decision,
        citations=citations,
        grounding=grounding,
    )

    package = build_answer_package(
        context=context,
        contract=contract,
        maximum_excerpt_characters=(
            maximum_excerpt_characters
        ),
    )

    adversarial = (
        validate_answer_package_adversarially(
            package=package,
            contract=contract,
            strict=True,
        )
    )

    return (
        package,
        contract,
        adversarial,
    )


def build_grounded_answer_package(
    query: str,
    settings: Settings,
    base_directory: str | Path,
    *,
    evaluate_closure: bool = False,
    maximum_excerpt_characters: int = 500,
) -> EngineeringAnswerPackage:
    """
    Build the complete strictly validated non-LLM answer package.
    """

    (
        package,
        _,
        _,
    ) = build_grounded_answer_artifacts(
        query,
        settings,
        base_directory,
        evaluate_closure=(
            evaluate_closure
        ),
        maximum_excerpt_characters=(
            maximum_excerpt_characters
        ),
    )

    return package