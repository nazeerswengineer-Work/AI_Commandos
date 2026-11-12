"""
Safe grounded narration orchestration for REQTRACE AI.

Phase 14D permits:

- one initial Ollama generation
- one constrained repair attempt
- deterministic fallback when validation or generation fails

Unvalidated model output is never returned as final narration.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.narration_client import (
    NarrationClientError,
    OllamaNarrationClient,
)
from src.narration_models import (
    NarrationDelivery,
    NarrationResponse,
    NarrationStatus,
)
from src.narration_prompt import (
    NarrationPrompt,
)
from src.narration_repair import (
    NarrationRepairRequest,
    build_repair_prompt,
)
from src.narration_validation import (
    NarrationValidationResult,
    validate_narration,
)


@dataclass(frozen=True)
class GroundedNarratorConfig:
    """Safe narration-generation policy."""

    allow_repair_attempt: bool = True
    maximum_generation_attempts: int = 2

    def __post_init__(
        self,
    ) -> None:
        """Validate generation policy."""

        if (
            self.maximum_generation_attempts
            not in {
                1,
                2,
            }
        ):
            raise ValueError(
                "maximum_generation_attempts must "
                "be either 1 or 2."
            )

        if (
            self.allow_repair_attempt
            and self.maximum_generation_attempts
            != 2
        ):
            raise ValueError(
                "Repair requires exactly two maximum attempts."
            )


class GroundedNarrator:
    """
    Generate, validate, optionally repair, and safely deliver narration.
    """

    def __init__(
        self,
        *,
        client: OllamaNarrationClient,
        config: GroundedNarratorConfig | None = None,
    ) -> None:
        """Create the grounded narrator."""

        self.client = client

        self.config = (
            config
            if config is not None
            else GroundedNarratorConfig()
        )

    def _fallback(
        self,
        *,
        prompt: NarrationPrompt,
        reason: str,
        attempts: int,
        initial_response: NarrationResponse | None,
        repair_response: NarrationResponse | None,
        initial_validation: NarrationValidationResult | None,
        repair_validation: NarrationValidationResult | None,
    ) -> NarrationDelivery:
        """Create deterministic fallback delivery."""

        model: str | None = None

        if repair_response is not None:
            model = repair_response.model

        elif initial_response is not None:
            model = initial_response.model

        return NarrationDelivery(
            status=(
                NarrationStatus
                .DETERMINISTIC_FALLBACK
            ),
            narration=(
                prompt.request
                .deterministic_fallback_text
            ),
            model=model,
            generation_attempts=attempts,
            used_repair=bool(
                repair_response is not None
                or attempts >= 2
            ),
            used_fallback=True,
            fallback_reason=reason,
            initial_response=(
                initial_response
            ),
            repair_response=(
                repair_response
            ),
            initial_validation=(
                initial_validation
            ),
            repair_validation=(
                repair_validation
            ),
            deterministic_fallback_text=(
                prompt.request
                .deterministic_fallback_text
            ),
        )

    def narrate(
        self,
        prompt: NarrationPrompt,
    ) -> NarrationDelivery:
        """
        Generate a safe final narration.

        Only narration with passed Phase 14C validation is delivered.
        """

        initial_response: (
            NarrationResponse
            | None
        ) = None

        repair_response: (
            NarrationResponse
            | None
        ) = None

        initial_validation: (
            NarrationValidationResult
            | None
        ) = None

        repair_validation: (
            NarrationValidationResult
            | None
        ) = None

        try:
            initial_response = (
                self.client.generate(
                    prompt,
                    attempt_number=1,
                    is_repair_attempt=False,
                )
            )

        except NarrationClientError as exc:
            return self._fallback(
                prompt=prompt,
                reason=(
                    "Initial narration generation failed: "
                    f"{exc}"
                ),
                attempts=1,
                initial_response=None,
                repair_response=None,
                initial_validation=None,
                repair_validation=None,
            )

        initial_validation = (
            validate_narration(
                request=prompt.request,
                response=initial_response,
            )
        )

        if initial_validation.passed:
            return NarrationDelivery(
                status=(
                    NarrationStatus
                    .VALIDATED_GENERATION
                ),
                narration=(
                    initial_response.text
                ),
                model=(
                    initial_response.model
                ),
                generation_attempts=1,
                used_repair=False,
                used_fallback=False,
                fallback_reason=None,
                initial_response=(
                    initial_response
                ),
                repair_response=None,
                initial_validation=(
                    initial_validation
                ),
                repair_validation=None,
                deterministic_fallback_text=(
                    prompt.request
                    .deterministic_fallback_text
                ),
            )

        repair_permitted = bool(
            self.config.allow_repair_attempt
            and (
                self.config
                .maximum_generation_attempts
                == 2
            )
        )

        if not repair_permitted:
            return self._fallback(
                prompt=prompt,
                reason=(
                    "Initial narration failed validation "
                    "and repair is disabled."
                ),
                attempts=1,
                initial_response=(
                    initial_response
                ),
                repair_response=None,
                initial_validation=(
                    initial_validation
                ),
                repair_validation=None,
            )

        repair_request = (
            NarrationRepairRequest(
                original_request=(
                    prompt.request
                ),
                failed_response=(
                    initial_response
                ),
                failed_validation=(
                    initial_validation
                ),
            )
        )

        repair_prompt = build_repair_prompt(
            repair_request
        )

        try:
            repair_response = (
                self.client.generate(
                    repair_prompt,
                    attempt_number=2,
                    is_repair_attempt=True,
                )
            )

        except NarrationClientError as exc:
            return self._fallback(
                prompt=prompt,
                reason=(
                    "Repair narration generation failed: "
                    f"{exc}"
                ),
                attempts=2,
                initial_response=(
                    initial_response
                ),
                repair_response=None,
                initial_validation=(
                    initial_validation
                ),
                repair_validation=None,
            )

        repair_validation = (
            validate_narration(
                request=prompt.request,
                response=repair_response,
            )
        )

        if repair_validation.passed:
            return NarrationDelivery(
                status=(
                    NarrationStatus
                    .VALIDATED_REPAIR
                ),
                narration=(
                    repair_response.text
                ),
                model=(
                    repair_response.model
                ),
                generation_attempts=2,
                used_repair=True,
                used_fallback=False,
                fallback_reason=None,
                initial_response=(
                    initial_response
                ),
                repair_response=(
                    repair_response
                ),
                initial_validation=(
                    initial_validation
                ),
                repair_validation=(
                    repair_validation
                ),
                deterministic_fallback_text=(
                    prompt.request
                    .deterministic_fallback_text
                ),
            )

        return self._fallback(
            prompt=prompt,
            reason=(
                "Initial narration and the single permitted "
                "repair attempt both failed validation."
            ),
            attempts=2,
            initial_response=(
                initial_response
            ),
            repair_response=(
                repair_response
            ),
            initial_validation=(
                initial_validation
            ),
            repair_validation=(
                repair_validation
            ),
        )