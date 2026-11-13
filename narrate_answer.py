"""
REQTRACE AI safe grounded narration command.

Production behavior:

1. Build the validated deterministic engineering package.
2. Enforce explicit engineering-identifier integrity.
3. Build the constrained narration prompt.
4. Generate narration through local Ollama.
5. Validate narration.
6. Permit at most one repair attempt.
7. Fall back to deterministic grounded output when narration fails.

Important:

- Unknown explicit IDs are never semantically substituted.
- Unvalidated LLM output is never delivered.
- Assessment, confidence and closure remain deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings

from src.answer_pipeline import (
    build_grounded_answer_package,
)

from src.grounded_narrator import (
    GroundedNarrator,
    GroundedNarratorConfig,
)

from src.identifier_integrity import (
    ExplicitIdentifierNotFound,
)

from src.narration_client import (
    NarrationClientConfig,
    OllamaNarrationClient,
)

from src.narration_prompt import (
    build_narration_prompt,
)


# =====================================================================
# Arguments
# =====================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI safe grounded engineering narration."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Engineering question, observation ID, "
            "requirement ID, or test-case ID."
        ),
    )

    parser.add_argument(
        "--closure",
        action="store_true",
        help=(
            "Also evaluate closure readiness."
        ),
    )

    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help=(
            "Local Ollama narration model."
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=350,
        help=(
            "Maximum narration output tokens."
        ),
    )

    parser.add_argument(
        "--no-repair",
        action="store_true",
        help=(
            "Disable the single permitted "
            "narration repair attempt."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print final safe delivery as JSON."
        ),
    )

    return parser.parse_args()


# =====================================================================
# Explicit-ID not-found output
# =====================================================================

def print_identifier_not_found(
    *,
    query: str,
    error: ExplicitIdentifierNotFound,
    as_json: bool,
) -> None:
    """
    Print a controlled response for an unknown explicit identifier.

    No LLM call occurs in this path.
    """

    if as_json:

        payload = {
            "query": query,
            "status": "NOT FOUND",
            "assessment": "NOT ASSESSED",
            "confidence": "NOT ASSESSED",
            "closure_status": "NOT ASSESSED",
            "requested_ids": list(
                error.requested_ids
            ),
            "missing_ids": list(
                error.missing_ids
            ),
            "message": str(
                error
            ),
            "substitution_performed": False,
            "llm_called": False,
        }

        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=True,
            )
        )

        return

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - IDENTIFIER NOT FOUND"
    )

    print("=" * 78)

    print(
        f"Query       : {query}"
    )

    print(
        "Status      : NOT FOUND"
    )

    print(
        "Assessment  : NOT ASSESSED"
    )

    print(
        "Confidence  : NOT ASSESSED"
    )

    print(
        "Closure     : NOT ASSESSED"
    )

    print()

    print(
        "Requested identifier(s)"
    )

    print("-" * 78)

    for identifier in (
        error.requested_ids
    ):

        print(
            f"  {identifier}"
        )

    print()

    print(
        "Missing identifier(s)"
    )

    print("-" * 78)

    for identifier in (
        error.missing_ids
    ):

        print(
            f"  {identifier}"
        )

    print()

    print(
        str(error)
    )

    print()

    print(
        "No semantically similar requirement, "
        "observation, or test case was substituted."
    )

    print()

    print(
        "LLM narration was not invoked."
    )


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    """Generate and safely deliver grounded narration."""

    args = parse_arguments()

    query = (
        args.query.strip()
    )

    if not query:

        print(
            "REQTRACE AI: query must not be empty."
        )

        return 2

    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    # -----------------------------------------------------------------
    # Phase 11 through Phase 13E.
    #
    # build_grounded_answer_package() now includes the explicit-ID
    # integrity gate.
    #
    # Therefore:
    #
    #     OBS-999
    #
    # cannot silently become:
    #
    #     OBS-052
    #
    # Processing stops before deterministic assessment and narration.
    # -----------------------------------------------------------------

    try:

        package = (
            build_grounded_answer_package(
                query,
                settings,
                base_directory,
                evaluate_closure=(
                    args.closure
                ),
            )
        )

    except ExplicitIdentifierNotFound as exc:

        print_identifier_not_found(
            query=query,
            error=exc,
            as_json=args.json,
        )

        return 2

    # -----------------------------------------------------------------
    # Build the constrained Phase 14 narration prompt.
    # -----------------------------------------------------------------

    prompt = build_narration_prompt(
        query=query,
        package=package,
    )

    # -----------------------------------------------------------------
    # Phase 16A optimized narration runtime.
    #
    # Current CPU-oriented configuration:
    #
    # context        = 4096
    # max output     = 350
    # keep alive     = 15 minutes
    #
    # Safety validation remains unchanged.
    # -----------------------------------------------------------------

    client_config = (
        NarrationClientConfig(
            base_url=(
                settings
                .ollama_base_url
            ),

            model=(
                args.model
            ),

            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),

            read_timeout_seconds=(
                settings
                .llm_timeout_seconds
            ),

            temperature=0.1,

            top_p=0.8,

            top_k=20,

            repeat_penalty=1.1,

            seed=42,

            context_window=4096,

            maximum_output_tokens=(
                args.max_tokens
            ),

            keep_alive="15m",
        )
    )

    narrator_config = (
        GroundedNarratorConfig(
            allow_repair_attempt=(
                not args.no_repair
            ),

            maximum_generation_attempts=(
                1
                if args.no_repair
                else 2
            ),
        )
    )

    # -----------------------------------------------------------------
    # Phase 14 safe generation.
    #
    # GroundedNarrator guarantees:
    #
    # - validated generation
    # - OR validated repair
    # - OR deterministic fallback
    #
    # Invalid raw narration is never returned as final narration.
    # -----------------------------------------------------------------

    with OllamaNarrationClient(
        client_config
    ) as client:

        narrator = GroundedNarrator(
            client=client,
            config=narrator_config,
        )

        delivery = narrator.narrate(
            prompt
        )

    # -----------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------

    if args.json:

        payload = (
            delivery.to_dict()
        )

        payload[
            "query"
        ] = query

        payload[
            "assessment"
        ] = (
            package.decision
            .assessment
        )

        payload[
            "confidence"
        ] = (
            package.decision
            .confidence
        )

        payload[
            "closure_status"
        ] = (
            package.decision
            .closure_status
        )

        payload[
            "requirement_ids"
        ] = list(
            package.requirement_ids
        )

        payload[
            "observation_ids"
        ] = list(
            package.observation_ids
        )

        payload[
            "test_case_ids"
        ] = list(
            package.test_case_ids
        )

        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    # -----------------------------------------------------------------
    # Safe console output
    # -----------------------------------------------------------------

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - SAFE GROUNDED NARRATION"
    )

    print("=" * 78)

    print(
        f"Query       : "
        f"{query}"
    )

    print(
        f"Assessment  : "
        f"{package.decision.assessment}"
    )

    print(
        f"Confidence  : "
        f"{package.decision.confidence}"
    )

    print(
        f"Closure     : "
        f"{package.decision.closure_status}"
    )

    print(
        f"Status      : "
        f"{delivery.status.value}"
    )

    print(
        f"Model       : "
        f"{delivery.model}"
    )

    print(
        f"Attempts    : "
        f"{delivery.generation_attempts}"
    )

    print(
        f"Used repair : "
        f"{delivery.used_repair}"
    )

    print(
        f"Fallback    : "
        f"{delivery.used_fallback}"
    )

    if delivery.fallback_reason:

        print(
            f"Reason      : "
            f"{delivery.fallback_reason}"
        )

    print()

    print("-" * 78)

    # -----------------------------------------------------------------
    # This is guaranteed to be either:
    #
    # - a Phase 14C validated narration
    # - a Phase 14C validated repaired narration
    # - the Phase 13D deterministic fallback
    #
    # Raw invalid model output is never printed here.
    # -----------------------------------------------------------------

    print(
        delivery.narration
    )

    print()

    print("-" * 78)

    print(
        "REQTRACE AI provides evidence-derived "
        "engineering decision support."
    )

    print(
        "Final engineering, quality, safety, "
        "certification, release, and closure "
        "authorization remain with the responsible "
        "human authority."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )