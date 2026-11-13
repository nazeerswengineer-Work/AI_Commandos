"""
REQTRACE AI Phase 14B raw narration diagnostic.

WARNING:

The generated narration is untrusted model output.

It has not passed Phase 14C narration validation and must not be used
as the final engineering answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.answer_pipeline import (
    build_grounded_answer_package,
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
            "REQTRACE AI Phase 14B "
            "raw Ollama narration diagnostic."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Observation ID or engineering query."
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

    return parser.parse_args()


def main() -> int:
    """Generate one raw diagnostic narration."""

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

    config = NarrationClientConfig(
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

    with OllamaNarrationClient(
        config
    ) as client:
        response = client.generate(
            prompt,
            attempt_number=1,
            is_repair_attempt=False,
        )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 14B "
        "RAW OLLAMA NARRATION"
    )
    print("=" * 78)

    print(
        f"Query             : "
        f"{args.query}"
    )

    print(
        f"Model             : "
        f"{response.model}"
    )

    print(
        f"Prompt tokens     : "
        f"{response.prompt_tokens}"
    )

    print(
        f"Completion tokens : "
        f"{response.completion_tokens}"
    )

    if (
        response
        .generation_duration_seconds
        is not None
    ):
        print(
            f"Duration seconds  : "
            f"{response.generation_duration_seconds:.3f}"
        )

    print()
    print(
        "WARNING: UNVALIDATED MODEL OUTPUT"
    )

    print(
        "Do not use this text as the final "
        "engineering answer."
    )

    print(
        "Phase 14C validation has not run."
    )

    print()
    print("-" * 78)
    print(
        response.text
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )