"""
Narration data models for REQTRACE AI.

Phase 14A defines the controlled boundary between the validated
Phase 13 answer package and the future Ollama narration layer.

Important:

- NarrationRequest contains only Phase 13-validated information.
- Evidence blocks are explicitly marked as untrusted quoted evidence.
- NarrationResponse is untrusted until Phase 14C validation passes.
- The LLM controls narration wording only.
- Deterministic decision fields never come from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class NarrationModelError(
    ValueError
):
    """Raised when a narration model is invalid."""


class EvidenceTrustClass(
    str,
    Enum,
):
    """Trust classification for evidence supplied to the narrator."""

    UNTRUSTED_QUOTED_EVIDENCE = (
        "UNTRUSTED_QUOTED_EVIDENCE"
    )


EvidenceTrustClass.__test__ = False


class NarrationStatus(
    str,
    Enum,
):
    """Final narration-delivery status."""

    VALIDATED_GENERATION = (
        "VALIDATED_GENERATION"
    )

    VALIDATED_REPAIR = (
        "VALIDATED_REPAIR"
    )

    DETERMINISTIC_FALLBACK = (
        "DETERMINISTIC_FALLBACK"
    )


NarrationStatus.__test__ = False


@dataclass(frozen=True)
class NarrationEvidenceBlock:
    """
    One citation-bound evidence block supplied to the narrator.

    Evidence content remains untrusted quoted source material.
    """

    citation_id: str
    evidence_id: str

    document_type: str
    source_label: str
    provenance_location: str

    excerpt: str

    requirement_id: str | None = None
    observation_id: str | None = None
    test_case_id: str | None = None
    result_status: str | None = None

    trust_class: EvidenceTrustClass = (
        EvidenceTrustClass
        .UNTRUSTED_QUOTED_EVIDENCE
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate the evidence block."""

        if not self.citation_id.strip():
            raise NarrationModelError(
                "citation_id must not be empty."
            )

        if not self.evidence_id.strip():
            raise NarrationModelError(
                "evidence_id must not be empty."
            )

        if not self.document_type.strip():
            raise NarrationModelError(
                "document_type must not be empty."
            )

        if not self.source_label.strip():
            raise NarrationModelError(
                "source_label must not be empty."
            )

        if not self.provenance_location.strip():
            raise NarrationModelError(
                "provenance_location must not be empty."
            )

        if not self.excerpt.strip():
            raise NarrationModelError(
                "excerpt must not be empty."
            )

        if (
            self.trust_class
            != EvidenceTrustClass
            .UNTRUSTED_QUOTED_EVIDENCE
        ):
            raise NarrationModelError(
                "Narration evidence must remain "
                "UNTRUSTED_QUOTED_EVIDENCE."
            )

    @property
    def engineering_ids(
        self,
    ) -> tuple[str, ...]:
        """Return engineering IDs carried by the block."""

        values = (
            self.requirement_id,
            self.observation_id,
            self.test_case_id,
        )

        return tuple(
            str(value)
            for value in values
            if value
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the evidence block into JSON-safe data."""

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
            "excerpt": (
                self.excerpt
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
            "trust_class": (
                self.trust_class.value
            ),
        }


@dataclass(frozen=True)
class NarrationRequest:
    """
    Authoritative request supplied to the LLM narrator.

    All decision fields originate from the deterministic engine.
    """

    request_version: str
    query: str

    assessment: str
    assessment_rule: str
    assessment_reason: str

    confidence: str
    confidence_reason: str

    closure_status: str
    closure_reason: str
    closure_was_assessed: bool

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    supporting_items: tuple[str, ...]
    blocking_items: tuple[str, ...]

    allowed_claims: tuple[str, ...]
    prohibited_claims: tuple[str, ...]

    citation_allowlist: tuple[str, ...]
    engineering_id_allowlist: tuple[str, ...]

    evidence_blocks: tuple[
        NarrationEvidenceBlock,
        ...
    ]

    narrator_instructions: tuple[str, ...]

    deterministic_fallback_text: str

    def __post_init__(
        self,
    ) -> None:
        """Validate the complete narration request."""

        required_values = {
            "request_version": (
                self.request_version
            ),
            "query": self.query,
            "assessment": self.assessment,
            "assessment_rule": (
                self.assessment_rule
            ),
            "assessment_reason": (
                self.assessment_reason
            ),
            "confidence": self.confidence,
            "confidence_reason": (
                self.confidence_reason
            ),
            "closure_status": (
                self.closure_status
            ),
            "closure_reason": (
                self.closure_reason
            ),
            "deterministic_fallback_text": (
                self.deterministic_fallback_text
            ),
        }

        for field_name, value in (
            required_values.items()
        ):
            if not str(value).strip():
                raise NarrationModelError(
                    f"{field_name} must not be empty."
                )

        if not self.allowed_claims:
            raise NarrationModelError(
                "Narration request requires "
                "at least one allowed claim."
            )

        if not self.prohibited_claims:
            raise NarrationModelError(
                "Narration request requires "
                "at least one prohibited claim."
            )

        if not self.narrator_instructions:
            raise NarrationModelError(
                "Narration request requires "
                "narrator instructions."
            )

        citation_allowlist = set(
            self.citation_allowlist
        )

        engineering_allowlist = set(
            self.engineering_id_allowlist
        )

        for block in self.evidence_blocks:
            if (
                block.citation_id
                not in citation_allowlist
            ):
                raise NarrationModelError(
                    "Evidence block contains citation "
                    "outside the citation allowlist: "
                    f"{block.citation_id}"
                )

            unsupported_ids = (
                set(
                    block.engineering_ids
                )
                - engineering_allowlist
            )

            if unsupported_ids:
                raise NarrationModelError(
                    "Evidence block contains engineering "
                    "IDs outside the allowlist: "
                    + ", ".join(
                        sorted(
                            unsupported_ids
                        )
                    )
                )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert narration request into JSON-safe data."""

        return {
            "request_version": (
                self.request_version
            ),
            "query": self.query,
            "authoritative_decision": {
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
                "closure_was_assessed": (
                    self.closure_was_assessed
                ),
            },
            "engineering_ids": {
                "requirement_ids": list(
                    self.requirement_ids
                ),
                "observation_ids": list(
                    self.observation_ids
                ),
                "test_case_ids": list(
                    self.test_case_ids
                ),
                "allowlist": list(
                    self.engineering_id_allowlist
                ),
            },
            "supporting_items": list(
                self.supporting_items
            ),
            "blocking_items": list(
                self.blocking_items
            ),
            "allowed_claims": list(
                self.allowed_claims
            ),
            "prohibited_claims": list(
                self.prohibited_claims
            ),
            "citation_allowlist": list(
                self.citation_allowlist
            ),
            "evidence_blocks": [
                block.to_dict()
                for block
                in self.evidence_blocks
            ],
            "narrator_instructions": list(
                self.narrator_instructions
            ),
            "deterministic_fallback_text": (
                self.deterministic_fallback_text
            ),
        }


@dataclass(frozen=True)
class NarrationResponse:
    """
    Raw model output.

    NarrationResponse is untrusted until Phase 14C validation passes.
    """

    model: str
    text: str

    attempt_number: int
    is_repair_attempt: bool

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    load_duration_seconds: float | None = None
    prompt_eval_duration_seconds: float | None = None
    eval_duration_seconds: float | None = None
    generation_duration_seconds: (
        float | None
    ) = None

    def __post_init__(
        self,
    ) -> None:
        """Validate response metadata."""

        if not self.model.strip():
            raise NarrationModelError(
                "model must not be empty."
            )

        if not self.text.strip():
            raise NarrationModelError(
                "narration text must not be empty."
            )

        if self.attempt_number < 1:
            raise NarrationModelError(
                "attempt_number must be one-based."
            )

        if (
            self.prompt_tokens is not None
            and self.prompt_tokens < 0
        ):
            raise NarrationModelError(
                "prompt_tokens must not be negative."
            )

        if (
            self.completion_tokens is not None
            and self.completion_tokens < 0
        ):
            raise NarrationModelError(
                "completion_tokens must not be negative."
            )

        if (
            self.generation_duration_seconds
            is not None
            and self.generation_duration_seconds
            < 0.0
        ):
            raise NarrationModelError(
                "generation duration must not "
                "be negative."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert raw response into JSON-safe data."""

        return {
            "model": self.model,
            "text": self.text,
            "attempt_number": (
                self.attempt_number
            ),
            "is_repair_attempt": (
                self.is_repair_attempt
            ),
            "prompt_tokens": (
                self.prompt_tokens
            ),
            "completion_tokens": (
                self.completion_tokens
            ),
            "generation_duration_seconds": (
                self.generation_duration_seconds
            ),
            "load_duration_seconds": (
                self.load_duration_seconds
            ),
            "prompt_eval_duration_seconds": (
                self.prompt_eval_duration_seconds
            ),
            "eval_duration_seconds": (
                self.eval_duration_seconds
            ),
        }

@dataclass(frozen=True)
class NarrationDelivery:
    """
    Final safe narration delivery.

    Generated narration is deliverable only after Phase 14C validation.
    Otherwise the Phase 13D deterministic answer is returned.
    """

    status: NarrationStatus
    narration: str

    model: str | None

    generation_attempts: int
    used_repair: bool
    used_fallback: bool

    fallback_reason: str | None

    initial_response: NarrationResponse | None
    repair_response: NarrationResponse | None

    initial_validation: object | None
    repair_validation: object | None

    deterministic_fallback_text: str

    def __post_init__(
        self,
    ) -> None:
        """Validate final narration-delivery state."""

        if not self.narration.strip():
            raise NarrationModelError(
                "Final narration must not be empty."
            )

        if self.generation_attempts < 0:
            raise NarrationModelError(
                "generation_attempts must not be negative."
            )

        if (
            self.status
            == NarrationStatus.VALIDATED_GENERATION
        ):
            if self.used_repair:
                raise NarrationModelError(
                    "Validated initial generation "
                    "cannot use repair."
                )

            if self.used_fallback:
                raise NarrationModelError(
                    "Validated generation cannot use fallback."
                )

            if self.initial_response is None:
                raise NarrationModelError(
                    "Validated generation requires "
                    "initial_response."
                )

            if not bool(
                getattr(
                    self.initial_validation,
                    "passed",
                    False,
                )
            ):
                raise NarrationModelError(
                    "Validated generation requires "
                    "passed initial validation."
                )

        elif (
            self.status
            == NarrationStatus.VALIDATED_REPAIR
        ):
            if not self.used_repair:
                raise NarrationModelError(
                    "Validated repair requires "
                    "used_repair=True."
                )

            if self.used_fallback:
                raise NarrationModelError(
                    "Validated repair cannot use fallback."
                )

            if self.initial_response is None:
                raise NarrationModelError(
                    "Validated repair requires "
                    "initial_response."
                )

            if self.repair_response is None:
                raise NarrationModelError(
                    "Validated repair requires "
                    "repair_response."
                )

            if not bool(
                getattr(
                    self.repair_validation,
                    "passed",
                    False,
                )
            ):
                raise NarrationModelError(
                    "Validated repair requires "
                    "passed repair validation."
                )

        elif (
            self.status
            == NarrationStatus.DETERMINISTIC_FALLBACK
        ):
            if not self.used_fallback:
                raise NarrationModelError(
                    "Fallback status requires "
                    "used_fallback=True."
                )

            if (
                self.narration
                != self.deterministic_fallback_text
            ):
                raise NarrationModelError(
                    "Fallback narration must exactly equal "
                    "deterministic_fallback_text."
                )

            if not self.fallback_reason:
                raise NarrationModelError(
                    "Fallback delivery requires a reason."
                )

        else:
            raise NarrationModelError(
                "Unsupported narration-delivery status."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert delivery into JSON-safe data."""

        return {
            "status": self.status.value,
            "narration": self.narration,
            "model": self.model,
            "generation_attempts": (
                self.generation_attempts
            ),
            "used_repair": self.used_repair,
            "used_fallback": self.used_fallback,
            "fallback_reason": (
                self.fallback_reason
            ),
            "initial_response": (
                self.initial_response.to_dict()
                if self.initial_response
                else None
            ),
            "repair_response": (
                self.repair_response.to_dict()
                if self.repair_response
                else None
            ),
            "initial_validation_passed": (
                getattr(
                    self.initial_validation,
                    "passed",
                    None,
                )
            ),
            "repair_validation_passed": (
                getattr(
                    self.repair_validation,
                    "passed",
                    None,
                )
            ),
            "deterministic_fallback_text": (
                self.deterministic_fallback_text
            ),
        }