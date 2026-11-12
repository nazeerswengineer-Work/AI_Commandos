"""
Unified production pipeline for REQTRACE AI.

This module provides one application-facing orchestration layer over
the validated Phase 11 through Phase 14 architecture.

Production flow:

    User query
        |
        v
    Automatic closure-intent detection
        |
        v
    Phase 11 Hybrid Retrieval
        |
        v
    Phase 12 Deterministic Engineering Decision
        |
        v
    Phase 13 Citation, Grounding and Adversarial Validation
        |
        v
    Phase 14 Grounded Narration
        |
        +--> VALIDATED_GENERATION
        |
        +--> VALIDATED_REPAIR
        |
        +--> DETERMINISTIC_FALLBACK
        |
        v
    ProductionResponse

Important safety rules:

- Raw unvalidated LLM output is never returned as final_answer.
- Assessment, confidence and closure status remain deterministic.
- Narration failure degrades fluency, not engineering safety.
- Deterministic fallback must exactly match the validated Phase 13D text.
- Closure readiness remains separate from observation assessment.
"""

from __future__ import annotations

import re
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings

from src.answer_pipeline import (
    build_grounded_answer_package,
)
from src.grounded_narrator import (
    GroundedNarrator,
    GroundedNarratorConfig,
)
from src.narration_client import (
    NarrationClientConfig,
    OllamaNarrationClient,
)
from src.narration_models import (
    NarrationDelivery,
    NarrationStatus,
)
from src.narration_prompt import (
    build_narration_prompt,
)
from src.production_metrics import (
    ProductionTiming,
)


class ProductionPipelineError(
    RuntimeError
):
    """Raised for an unrecoverable production-pipeline failure."""


# =====================================================================
# Closure-intent detection
# =====================================================================

CLOSURE_INTENT_PATTERNS = (
    r"\bclosure\b",
    r"\bclosure status\b",
    r"\bclosure readiness\b",
    r"\bready for closure\b",
    r"\bclosed\b",
    r"\bclose\b",
    r"\bopen\b",
    r"\bresolved\b",
    r"\bresolve\b",
    r"\bresolution status\b",
    r"\bcan\s+.+\s+be\s+closed\b",
    r"\bis\s+.+\s+closed\b",
    r"\bis\s+.+\s+open\b",
    r"\bhas\s+.+\s+been\s+resolved\b",
)


def detect_closure_intent(
    query: str,
) -> bool:
    """
    Detect whether a query asks about observation closure readiness.

    Examples that return True:

        Is OBS-023 open?
        Is OBS-070 closed?
        What is the closure status of OBS-070?
        Can OBS-070 be closed?
        Is OBS-070 ready for closure?
        Has OBS-070 been resolved?

    Examples that return False:

        OBS-070
        Analyze OBS-023
        What does REQ-050 require?
    """

    normalized = " ".join(
        query
        .strip()
        .lower()
        .split()
    )

    if not normalized:
        return False

    return any(
        re.search(
            pattern,
            normalized,
        )
        is not None
        for pattern
        in CLOSURE_INTENT_PATTERNS
    )


# =====================================================================
# Production citation model
# =====================================================================

@dataclass(frozen=True)
class ProductionCitation:
    """
    Application-facing evidence citation.

    This record contains only validated Phase 13 provenance.
    """

    citation_id: str
    evidence_id: str

    document_type: str
    source_label: str
    provenance_location: str

    requirement_id: str | None = None
    observation_id: str | None = None
    test_case_id: str | None = None
    result_status: str | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate the application-facing citation."""

        if not self.citation_id.strip():
            raise ValueError(
                "citation_id must not be empty."
            )

        if not self.evidence_id.strip():
            raise ValueError(
                "evidence_id must not be empty."
            )

        if not self.document_type.strip():
            raise ValueError(
                "document_type must not be empty."
            )

        if not self.source_label.strip():
            raise ValueError(
                "source_label must not be empty."
            )

        if not self.provenance_location.strip():
            raise ValueError(
                "provenance_location must not be empty."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert citation into JSON-safe data."""

        return {
            "citation_id": (
                self.citation_id
            ),
            "evidence_id": (
                self.evidence_id
            ),
            "document_type": (
                self.document_type
            ),
            "source_label": (
                self.source_label
            ),
            "provenance_location": (
                self.provenance_location
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "observation_id": (
                self.observation_id
            ),
            "test_case_id": (
                self.test_case_id
            ),
            "result_status": (
                self.result_status
            ),
        }


# =====================================================================
# Stable application response
# =====================================================================

@dataclass(frozen=True)
class ProductionResponse:
    """Stable application-facing REQTRACE AI result."""

    query: str

    assessment: str
    assessment_rule: str
    assessment_reason: str

    confidence: str
    confidence_reason: str

    closure_status: str
    closure_reason: str

    delivery_status: NarrationStatus

    final_answer: str

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    citations: tuple[
        ProductionCitation,
        ...
    ]

    supporting_items: tuple[str, ...]
    blocking_items: tuple[str, ...]

    generation_attempts: int
    used_repair: bool
    used_fallback: bool
    fallback_reason: str | None

    model: str | None

    grounding_passed: bool

    timing: ProductionTiming

    def __post_init__(
        self,
    ) -> None:
        """Validate the application-facing response."""

        if not self.query.strip():
            raise ValueError(
                "ProductionResponse query must not be empty."
            )

        if not self.assessment.strip():
            raise ValueError(
                "ProductionResponse assessment must not be empty."
            )

        if not self.confidence.strip():
            raise ValueError(
                "ProductionResponse confidence must not be empty."
            )

        if not self.closure_status.strip():
            raise ValueError(
                "ProductionResponse closure_status "
                "must not be empty."
            )

        if not self.final_answer.strip():
            raise ValueError(
                "ProductionResponse final_answer "
                "must not be empty."
            )

        if self.generation_attempts < 0:
            raise ValueError(
                "generation_attempts must not be negative."
            )

        if not self.grounding_passed:
            raise ValueError(
                "ProductionResponse requires passed grounding."
            )

    @property
    def safe_delivery(
        self,
    ) -> bool:
        """Return True for an accepted production-delivery state."""

        return self.delivery_status in {
            NarrationStatus
            .VALIDATED_GENERATION,

            NarrationStatus
            .VALIDATED_REPAIR,

            NarrationStatus
            .DETERMINISTIC_FALLBACK,
        }

    @property
    def generated_by_llm(
        self,
    ) -> bool:
        """Return True only for validated LLM narration."""

        return self.delivery_status in {
            NarrationStatus
            .VALIDATED_GENERATION,

            NarrationStatus
            .VALIDATED_REPAIR,
        }

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the production response into JSON-safe data."""

        return {
            "query": (
                self.query
            ),

            "assessment": (
                self.assessment
            ),

            "assessment_rule": (
                self.assessment_rule
            ),

            "assessment_reason": (
                self.assessment_reason
            ),

            "confidence": (
                self.confidence
            ),

            "confidence_reason": (
                self.confidence_reason
            ),

            "closure_status": (
                self.closure_status
            ),

            "closure_reason": (
                self.closure_reason
            ),

            "delivery_status": (
                self.delivery_status.value
            ),

            "safe_delivery": (
                self.safe_delivery
            ),

            "generated_by_llm": (
                self.generated_by_llm
            ),

            "final_answer": (
                self.final_answer
            ),

            "requirement_ids": list(
                self.requirement_ids
            ),

            "observation_ids": list(
                self.observation_ids
            ),

            "test_case_ids": list(
                self.test_case_ids
            ),

            "citations": [
                citation.to_dict()
                for citation
                in self.citations
            ],

            "supporting_items": list(
                self.supporting_items
            ),

            "blocking_items": list(
                self.blocking_items
            ),

            "generation_attempts": (
                self.generation_attempts
            ),

            "used_repair": (
                self.used_repair
            ),

            "used_fallback": (
                self.used_fallback
            ),

            "fallback_reason": (
                self.fallback_reason
            ),

            "model": (
                self.model
            ),

            "grounding_passed": (
                self.grounding_passed
            ),

            "timing": (
                self.timing.to_dict()
            ),
        }


# =====================================================================
# Production configuration
# =====================================================================

@dataclass(frozen=True)
class ProductionPipelineConfig:
    """Production orchestration configuration."""

    narration_enabled: bool = True

    allow_repair_attempt: bool = True

    narration_model: str = (
        "llama3.2:1b"
    )

    maximum_output_tokens: int = 450

    context_window: int = 4096

    temperature: float = 0.1

    top_p: float = 0.8

    top_k: int = 20

    repeat_penalty: float = 1.1

    seed: int = 42

    maximum_excerpt_characters: int = 500

    def __post_init__(
        self,
    ) -> None:
        """Validate production configuration."""

        if not self.narration_model.strip():
            raise ValueError(
                "narration_model must not be empty."
            )

        if self.maximum_output_tokens < 1:
            raise ValueError(
                "maximum_output_tokens must be positive."
            )

        if self.context_window < 1024:
            raise ValueError(
                "context_window must be at least 1024."
            )

        if not (
            0.0
            <= self.temperature
            <= 2.0
        ):
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if not (
            0.0
            < self.top_p
            <= 1.0
        ):
            raise ValueError(
                "top_p must be greater than 0 "
                "and less than or equal to 1."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if self.repeat_penalty <= 0.0:
            raise ValueError(
                "repeat_penalty must be positive."
            )

        if self.maximum_excerpt_characters < 1:
            raise ValueError(
                "maximum_excerpt_characters "
                "must be positive."
            )


# =====================================================================
# Production conversion helpers
# =====================================================================

def build_production_citations(
    package,
) -> tuple[
    ProductionCitation,
    ...
]:
    """
    Convert validated Phase 13 evidence excerpts into application
    citation records.
    """

    citations: list[
        ProductionCitation
    ] = []

    seen_citation_ids: set[str] = set()

    for section in (
        package.evidence_sections
    ):

        for excerpt in (
            section.excerpts
        ):

            if (
                excerpt.citation_id
                in seen_citation_ids
            ):
                raise ProductionPipelineError(
                    "Duplicate production citation ID: "
                    f"{excerpt.citation_id}"
                )

            seen_citation_ids.add(
                excerpt.citation_id
            )

            citations.append(
                ProductionCitation(
                    citation_id=(
                        excerpt.citation_id
                    ),

                    evidence_id=(
                        excerpt.evidence_id
                    ),

                    document_type=(
                        excerpt.document_type
                    ),

                    source_label=(
                        excerpt.source_label
                    ),

                    provenance_location=(
                        excerpt
                        .provenance_location
                    ),

                    requirement_id=(
                        excerpt.requirement_id
                    ),

                    observation_id=(
                        excerpt.observation_id
                    ),

                    test_case_id=(
                        excerpt.test_case_id
                    ),

                    result_status=(
                        excerpt.result_status
                    ),
                )
            )

    return tuple(
        citations
    )


def deterministic_delivery(
    package,
    *,
    reason: str,
) -> NarrationDelivery:
    """
    Create a safe deterministic delivery without invoking the LLM.

    This is used when narration is explicitly disabled.
    """

    return NarrationDelivery(
        status=(
            NarrationStatus
            .DETERMINISTIC_FALLBACK
        ),

        narration=(
            package.deterministic_text
        ),

        model=None,

        generation_attempts=0,

        used_repair=False,

        used_fallback=True,

        fallback_reason=reason,

        initial_response=None,

        repair_response=None,

        initial_validation=None,

        repair_validation=None,

        deterministic_fallback_text=(
            package.deterministic_text
        ),
    )


# =====================================================================
# Unified production pipeline
# =====================================================================

class ReqTraceProductionPipeline:
    """Single production entry point for REQTRACE AI."""

    def __init__(
        self,
        *,
        settings: Settings,
        base_directory: str | Path,
        config: ProductionPipelineConfig
        | None = None,
    ) -> None:
        """Create the production pipeline."""

        self.settings = settings

        self.base_directory = (
            Path(
                base_directory
            )
            .expanduser()
            .resolve()
        )

        self.config = (
            config
            if config is not None
            else ProductionPipelineConfig()
        )

    def _run_narration(
        self,
        *,
        query: str,
        package,
    ) -> NarrationDelivery:
        """
        Run the validated Phase 14 narration layer.

        Narration errors and timeouts are handled by GroundedNarrator,
        which returns the deterministic fallback when necessary.
        """

        if not (
            self.config
            .narration_enabled
        ):
            return deterministic_delivery(
                package,
                reason=(
                    "LLM narration is disabled."
                ),
            )

        prompt = build_narration_prompt(
            query=query,
            package=package,
        )

        client_config = (
            NarrationClientConfig(
                base_url=(
                    self.settings
                    .ollama_base_url
                ),

                model=(
                    self.config
                    .narration_model
                ),

                connect_timeout_seconds=(
                    self.settings
                    .ollama_connect_timeout_seconds
                ),

                read_timeout_seconds=(
                    self.settings
                    .llm_timeout_seconds
                ),

                temperature=(
                    self.config.temperature
                ),

                top_p=(
                    self.config.top_p
                ),

                top_k=(
                    self.config.top_k
                ),

                repeat_penalty=(
                    self.config
                    .repeat_penalty
                ),

                seed=(
                    self.config.seed
                ),

                context_window=(
                    self.config
                    .context_window
                ),

                maximum_output_tokens=(
                    self.config
                    .maximum_output_tokens
                ),
            )
        )

        narrator_config = (
            GroundedNarratorConfig(
                allow_repair_attempt=(
                    self.config
                    .allow_repair_attempt
                ),

                maximum_generation_attempts=(
                    2
                    if (
                        self.config
                        .allow_repair_attempt
                    )
                    else 1
                ),
            )
        )

        with OllamaNarrationClient(
            client_config
        ) as client:

            narrator = GroundedNarrator(
                client=client,
                config=narrator_config,
            )

            return narrator.narrate(
                prompt
            )

    def run(
        self,
        query: str,
        *,
        evaluate_closure: bool | None = None,
    ) -> ProductionResponse:
        """
        Run the complete production pipeline.

        Closure evaluation behavior:

        - evaluate_closure=True:
            closure is always evaluated.

        - evaluate_closure=False:
            explicit closure intent in the query still activates
            closure evaluation.

        - evaluate_closure=None:
            closure evaluation depends entirely on automatic
            closure-intent detection.

        Optional LLM narration failure does not fail the engineering
        pipeline. It produces a deterministic grounded fallback.
        """

        request_started = (
            time.perf_counter()
        )

        clean_query = (
            query.strip()
        )

        if not clean_query:
            raise ValueError(
                "Query must not be empty."
            )

        # -------------------------------------------------------------
        # Detect explicit closure intent.
        # -------------------------------------------------------------

        closure_intent_detected = (
            detect_closure_intent(
                clean_query
            )
        )

        if evaluate_closure is None:

            should_evaluate_closure = (
                closure_intent_detected
            )

        else:

            should_evaluate_closure = bool(
                evaluate_closure
                or closure_intent_detected
            )

        # -------------------------------------------------------------
        # Phase 11 through Phase 13E.
        #
        # build_grounded_answer_package() includes:
        #
        # - retrieval
        # - deterministic assessment
        # - confidence
        # - optional closure evaluation
        # - citations
        # - grounding validation
        # - answer contract
        # - deterministic answer package
        # - adversarial validation
        # -------------------------------------------------------------

        deterministic_started = (
            time.perf_counter()
        )

        package = (
            build_grounded_answer_package(
                clean_query,

                self.settings,

                self.base_directory,

                evaluate_closure=(
                    should_evaluate_closure
                ),

                maximum_excerpt_characters=(
                    self.config
                    .maximum_excerpt_characters
                ),
            )
        )

        deterministic_seconds = (
            time.perf_counter()
            - deterministic_started
        )

        if not package.grounding_passed:
            raise ProductionPipelineError(
                "Grounding validation did not pass."
            )

        if (
            package.grounding_error_count
            != 0
        ):
            raise ProductionPipelineError(
                "Grounding errors remain in "
                "the production package."
            )

        if not package.deterministic_text.strip():
            raise ProductionPipelineError(
                "Deterministic answer package is empty."
            )

        # -------------------------------------------------------------
        # Phase 14 narration, validation, repair and fallback.
        # -------------------------------------------------------------

        narration_started = (
            time.perf_counter()
        )

        delivery = self._run_narration(
            query=clean_query,
            package=package,
        )

        narration_seconds = (
            time.perf_counter()
            - narration_started
        )

        safe_delivery_states = {
            NarrationStatus
            .VALIDATED_GENERATION,

            NarrationStatus
            .VALIDATED_REPAIR,

            NarrationStatus
            .DETERMINISTIC_FALLBACK,
        }

        if (
            delivery.status
            not in safe_delivery_states
        ):
            raise ProductionPipelineError(
                "Narration delivery state is unsafe."
            )

        if (
            delivery.used_fallback
            and delivery.narration
            != package.deterministic_text
        ):
            raise ProductionPipelineError(
                "Deterministic fallback text "
                "was modified."
            )

        if not delivery.narration.strip():
            raise ProductionPipelineError(
                "Final narration delivery is empty."
            )

        # -------------------------------------------------------------
        # Timing metrics.
        # -------------------------------------------------------------

        total_seconds = (
            time.perf_counter()
            - request_started
        )

        timing = ProductionTiming(
            deterministic_seconds=(
                deterministic_seconds
            ),

            narration_seconds=(
                narration_seconds
            ),

            total_seconds=(
                total_seconds
            ),
        )

        # -------------------------------------------------------------
        # Stable application-facing response.
        # -------------------------------------------------------------

        return ProductionResponse(
            query=clean_query,

            assessment=(
                package.decision
                .assessment
            ),

            assessment_rule=(
                package.decision
                .assessment_rule
            ),

            assessment_reason=(
                package.decision
                .assessment_reason
            ),

            confidence=(
                package.decision
                .confidence
            ),

            confidence_reason=(
                package.decision
                .confidence_reason
            ),

            closure_status=(
                package.decision
                .closure_status
            ),

            closure_reason=(
                package.decision
                .closure_reason
            ),

            delivery_status=(
                delivery.status
            ),

            final_answer=(
                delivery.narration
            ),

            requirement_ids=(
                package.requirement_ids
            ),

            observation_ids=(
                package.observation_ids
            ),

            test_case_ids=(
                package.test_case_ids
            ),

            citations=(
                build_production_citations(
                    package
                )
            ),

            supporting_items=(
                package.supporting_items
            ),

            blocking_items=(
                package.blocking_items
            ),

            generation_attempts=(
                delivery
                .generation_attempts
            ),

            used_repair=(
                delivery
                .used_repair
            ),

            used_fallback=(
                delivery
                .used_fallback
            ),

            fallback_reason=(
                delivery
                .fallback_reason
            ),

            model=(
                delivery.model
            ),

            grounding_passed=(
                package.grounding_passed
            ),

            timing=timing,
        )