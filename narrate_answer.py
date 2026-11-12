"""
REQTRACE AI Phase 14D safe grounded narration.

Only validated model output or deterministic fallback is displayed.
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
from src.narration_client import (
    NarrationClientConfig,
    OllamaNarrationClient,
)
from src.narration_prompt import (
    build_narration_prompt,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 14D "
            "safe grounded narration."
        )
    )

    parser.add_argument(
        "query",
        help="Observation ID or engineering query.",
    )

    parser.add_argument(
        "--closure",
        action="store_true",
        help="Also evaluate closure readiness.",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:1b",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--no-repair",
        action="store_true",
        help=(
            "Disable the single narration repair attempt."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print delivery metadata as JSON.",
    )

    return parser.parse_args()


def main() -> int:
    """Generate and safely deliver narration."""

    args = parse_arguments()
    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    package = build_grounded_answer_package(
        args.query,
        settings,
        base_directory,
        evaluate_closure=(
            args.closure
        ),
    )

    prompt = build_narration_prompt(
        query=args.query,
        package=package,
    )

    client_config = (
        NarrationClientConfig(
            base_url=(
                settings.ollama_base_url
            ),
            model=args.model,
            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),
            read_timeout_seconds=(
                settings.llm_timeout_seconds
            ),
            maximum_output_tokens=(
                args.max_tokens
            ),
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

    if args.json:
        print(
            json.dumps(
                delivery.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - SAFE GROUNDED NARRATION"
    )
    print("=" * 78)

    print(
        f"Query       : {args.query}"
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

    print(
        delivery.narration
    )

    print()
    print("-" * 78)

    print(
        "Final engineering approval and closure "
        "authorization remain with the responsible "
        "human authority."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )