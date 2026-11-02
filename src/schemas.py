"""
Structured internal and final-response schemas for REQTRACE AI.

These schemas define interfaces used by later deterministic validation,
confidence calculation, grounding, LLM narration, and Streamlit rendering.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.normalization import (
    normalize_optional_observation_id,
    normalize_optional_requirement_id,
    normalize_whitespace,
)


class AssessmentValue(str, Enum):
    """Permitted deterministic assessment values."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ConfidenceLevel(str, Enum):
    """Permitted evidence-based confidence levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ClosureStatus(str, Enum):
    """Permitted closure-readiness values."""

    NOT_ASSESSED = "NOT ASSESSED"
    ESTABLISHED = "CLOSURE ESTABLISHED"
    NOT_ESTABLISHED = "CLOSURE NOT ESTABLISHED"


class GroundingStatus(str, Enum):
    """Permitted grounding-validation values."""

    PASSED = "PASSED"
    FAILED = "FAILED"


class EvidenceCompleteness(str, Enum):
    """Evidence completeness classifications."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class SourceReference(BaseModel):
    """Validated reference to retrieved source evidence."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=3)
    document_type: str = Field(min_length=2)

    document: str | None = None
    revision: str | None = None
    page: int | None = Field(default=None, ge=1)
    section: str | None = None

    id: str | None = None
    file: str | None = None
    function: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @field_validator(
        "evidence_id",
        "document_type",
        "document",
        "revision",
        "section",
        "id",
        "file",
        "function",
        mode="before",
    )
    @classmethod
    def clean_strings(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        cleaned = normalize_whitespace(str(value))
        return cleaned or None

    @model_validator(mode="after")
    def validate_line_range(
        self,
    ) -> "SourceReference":
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError(
                "start_line and end_line must both be supplied."
            )

        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError(
                "end_line must be greater than or equal to start_line."
            )

        return self


class DeterministicDecision(BaseModel):
    """Auditable output from the future deterministic rule engine."""

    model_config = ConfigDict(extra="forbid")

    assessment: AssessmentValue
    rule_id: str = Field(min_length=1)
    decision_explanation: str = Field(min_length=1)
    blocking_items: list[str] = Field(default_factory=list)
    evidence_completeness: EvidenceCompleteness

    @field_validator(
        "rule_id",
        "decision_explanation",
        mode="before",
    )
    @classmethod
    def clean_required_text(
        cls,
        value: Any,
    ) -> str:
        cleaned = normalize_whitespace(str(value))

        if not cleaned:
            raise ValueError("Value must not be empty.")

        return cleaned

    @field_validator("blocking_items", mode="before")
    @classmethod
    def clean_blocking_items(
        cls,
        value: Any,
    ) -> list[str]:
        if value is None:
            return []

        cleaned_items: list[str] = []
        seen: set[str] = set()

        for item in value:
            cleaned = normalize_whitespace(str(item))

            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                cleaned_items.append(cleaned)

        return cleaned_items

    @model_validator(mode="after")
    def enforce_inconclusive_relationships(
        self,
    ) -> "DeterministicDecision":
        if (
            self.assessment == AssessmentValue.INCONCLUSIVE
            and not self.blocking_items
        ):
            raise ValueError(
                "INCONCLUSIVE requires at least one blocking item."
            )

        return self


class ConfidenceResult(BaseModel):
    """Calculated confidence output."""

    model_config = ConfigDict(extra="forbid")

    level: ConfidenceLevel
    numeric_score: float = Field(ge=0.0, le=1.0)
    contributing_factors: dict[str, float] = Field(
        default_factory=dict
    )
    explanation: str = Field(min_length=1)

    @field_validator("explanation", mode="before")
    @classmethod
    def clean_explanation(
        cls,
        value: Any,
    ) -> str:
        cleaned = normalize_whitespace(str(value))

        if not cleaned:
            raise ValueError("Confidence explanation is required.")

        return cleaned

    @field_validator("contributing_factors")
    @classmethod
    def validate_factors(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        for factor_name, factor_value in value.items():
            if not factor_name.strip():
                raise ValueError(
                    "Confidence factor names must not be empty."
                )

            if not 0.0 <= factor_value <= 1.0:
                raise ValueError(
                    "Confidence factor values must be between 0 and 1."
                )

        return value


class ClosureDecision(BaseModel):
    """Separate closure-readiness result."""

    model_config = ConfigDict(extra="forbid")

    status: ClosureStatus
    explanation: str = Field(min_length=1)
    blocking_items: list[str] = Field(default_factory=list)

    @field_validator("explanation", mode="before")
    @classmethod
    def clean_explanation(
        cls,
        value: Any,
    ) -> str:
        cleaned = normalize_whitespace(str(value))

        if not cleaned:
            raise ValueError("Closure explanation is required.")

        return cleaned

    @model_validator(mode="after")
    def validate_closure_result(
        self,
    ) -> "ClosureDecision":
        if (
            self.status == ClosureStatus.NOT_ESTABLISHED
            and not self.blocking_items
        ):
            raise ValueError(
                "CLOSURE NOT ESTABLISHED requires blocking items."
            )

        if (
            self.status == ClosureStatus.ESTABLISHED
            and self.blocking_items
        ):
            raise ValueError(
                "CLOSURE ESTABLISHED cannot retain blocking items."
            )

        return self


class GroundingResult(BaseModel):
    """Grounding- and citation-validation output."""

    model_config = ConfigDict(extra="forbid")

    status: GroundingStatus
    validated_evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grounding_status(
        self,
    ) -> "GroundingResult":
        if self.status == GroundingStatus.PASSED:
            if self.invalid_evidence_ids:
                raise ValueError(
                    "PASSED grounding cannot contain invalid evidence IDs."
                )

            if self.unsupported_claims:
                raise ValueError(
                    "PASSED grounding cannot contain unsupported claims."
                )

        return self


class ObservationSummary(BaseModel):
    """Observation details for final structured output."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    summary: str = ""
    source: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return normalize_optional_observation_id(str(value))


class FRDRequirementSummary(BaseModel):
    """FRD requirement details for final structured output."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    summary: str = ""
    revision: str = ""
    source: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return normalize_optional_requirement_id(str(value))


class CodeAnalysis(BaseModel):
    """Code-evidence section of the engineering response."""

    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)


class TestEvidenceSummary(BaseModel):
    """One test-evidence item in the final response."""

    model_config = ConfigDict(extra="forbid")

    test_case_id: str | None = None
    result_status: str | None = None
    summary: str = ""
    source: str = ""


class EvidenceComparison(BaseModel):
    """Organized evidence comparison."""

    model_config = ConfigDict(extra="forbid")

    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class EngineeringAssessment(BaseModel):
    """
    Final validated internal assessment.

    The normal UI will render this as readable engineering cards rather than
    displaying the raw dictionary.
    """

    model_config = ConfigDict(extra="forbid")

    assessment: AssessmentValue
    confidence: ConfidenceLevel
    decision_rule_id: str = Field(min_length=1)

    observation: ObservationSummary
    frd_requirement: FRDRequirementSummary

    expected_behavior: str
    observed_behavior: str
    code_analysis: CodeAnalysis
    test_evidence: list[TestEvidenceSummary]

    evidence_comparison: EvidenceComparison
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    missing_information: list[str]
    blocking_items: list[str]

    closure_status: ClosureStatus
    recommended_reviewer_action: str
    justification: str

    sources: list[SourceReference]
    grounding_status: GroundingStatus

    @model_validator(mode="after")
    def validate_assessment_consistency(
        self,
    ) -> "EngineeringAssessment":
        if (
            self.assessment == AssessmentValue.INCONCLUSIVE
            and not self.blocking_items
        ):
            raise ValueError(
                "INCONCLUSIVE final assessments require blocking items."
            )

        if (
            self.closure_status == ClosureStatus.NOT_ESTABLISHED
            and not self.blocking_items
        ):
            raise ValueError(
                "CLOSURE NOT ESTABLISHED requires blocking items."
            )

        if (
            self.grounding_status == GroundingStatus.PASSED
            and not self.sources
        ):
            raise ValueError(
                "Grounding cannot pass without source references."
            )

        return self