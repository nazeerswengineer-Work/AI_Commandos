"""
Generalized deterministic QA response models for REQTRACE AI.

These models represent concise inventory and evidence-query answers.

They remain separate from ProductionResponse because COUNT and LIST
questions do not require:

- engineering assessment;
- confidence classification;
- closure assessment;
- deterministic assessment rules;
- grounded LLM narration.

The underlying answer is calculated deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.evidence_inventory import (
    InventoryAnswer,
)

from src.query_intent import (
    QueryIntent,
    QueryIntentResult,
)


class QAResponseMode(
    str,
    Enum,
):
    """REQTRACE AI response presentation modes."""

    CONCISE = "CONCISE"

    ENGINEERING_ASSESSMENT = (
        "ENGINEERING_ASSESSMENT"
    )

    CLOSURE_ASSESSMENT = (
        "CLOSURE_ASSESSMENT"
    )


QAResponseMode.__test__ = False


@dataclass(frozen=True)
class GeneralizedQAResponse:
    """
    Concise deterministic generalized-QA response.

    Counts, filters, engineering IDs, and entity membership have
    already been calculated by the deterministic inventory engine.
    """

    query: str

    intent: QueryIntent

    intent_rule: str

    response_mode: QAResponseMode

    answer: str

    record_count: int

    engineering_ids: tuple[
        str,
        ...
    ]

    source_statement: str

    deterministic: bool = True

    def __post_init__(
        self,
    ) -> None:
        """Validate generalized QA response."""

        if not self.query.strip():

            raise ValueError(
                "query must not be empty."
            )

        if not self.intent_rule.strip():

            raise ValueError(
                "intent_rule must not be empty."
            )

        if not self.answer.strip():

            raise ValueError(
                "answer must not be empty."
            )

        if self.record_count < 0:

            raise ValueError(
                "record_count must not be negative."
            )

        if not self.source_statement.strip():

            raise ValueError(
                "source_statement must not be empty."
            )

    @classmethod
    def from_inventory_answer(
        cls,
        *,
        original_query: str,
        intent_result: QueryIntentResult,
        inventory_answer: InventoryAnswer,
    ) -> "GeneralizedQAResponse":
        """
        Build a concise QA response from deterministic inventory output.
        """

        identifiers = tuple(
            record.entity_id
            for record
            in inventory_answer.records
        )

        return cls(
            query=original_query,

            intent=(
                intent_result.intent
            ),

            intent_rule=(
                intent_result.matched_rule
            ),

            response_mode=(
                QAResponseMode.CONCISE
            ),

            answer=(
                inventory_answer.answer_text
            ),

            record_count=(
                inventory_answer.count
            ),

            engineering_ids=(
                identifiers
            ),

            source_statement=(
                inventory_answer
                .source_statement
            ),

            deterministic=True,
        )

    @property
    def concise(
        self,
    ) -> bool:
        """Return whether this response uses concise QA presentation."""

        return (
            self.response_mode
            == QAResponseMode.CONCISE
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert response into JSON-safe data."""

        return {
            "query": self.query,

            "intent": (
                self.intent.value
            ),

            "intent_rule": (
                self.intent_rule
            ),

            "response_mode": (
                self.response_mode.value
            ),

            "answer": self.answer,

            "record_count": (
                self.record_count
            ),

            "engineering_ids": list(
                self.engineering_ids
            ),

            "source_statement": (
                self.source_statement
            ),

            "deterministic": (
                self.deterministic
            ),
        }