"""
REQTRACE AI production command-line entry point.

This command exposes only the unified Phase 15 production pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings

from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
)


def parse_arguments() -> argparse.Namespace:
    """Parse production CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI grounded engineering "
            "decision-support assistant."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Engineering question, requirement ID, "
            "or observation ID."
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
        "--deterministic-only",
        action="store_true",
        help=(
            "Disable LLM narration and return "
            "the deterministic grounded answer."
        ),
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
        help=(
            "Print the complete production response "
            "as JSON."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run REQTRACE AI."""

    args = parse_arguments()

    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    config = (
        ProductionPipelineConfig(
            narration_enabled=(
                not args
                .deterministic_only
            ),

            allow_repair_attempt=(
                not args.no_repair
            ),

            narration_model=(
                "llama3.2:1b"
            ),

            # Phase 15 optimization:
            # our narration prompt is currently
            # around 2K tokens, so 4096 is a more
            # appropriate initial context target
            # than the earlier 8192 diagnostic value.

            context_window=4096,

            # Existing live narrations were around
            # several hundred tokens. Keep generation
            # bounded for interactive use.

            maximum_output_tokens=450,
        )
    )

    pipeline = (
        ReqTraceProductionPipeline(
            settings=settings,
            base_directory=(
                base_directory
            ),
            config=config,
        )
    )

    result = pipeline.run(
        args.query,
        evaluate_closure=(
            args.closure
        ),
    )

    if args.json:

        print(
            json.dumps(
                result.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    print()

    print("=" * 78)

    print(
        "REQTRACE AI"
    )

    print("=" * 78)

    print(
        f"Query       : "
        f"{result.query}"
    )

    print(
        f"Assessment  : "
        f"{result.assessment}"
    )

    print(
        f"Confidence  : "
        f"{result.confidence}"
    )

    print(
        f"Closure     : "
        f"{result.closure_status}"
    )

    print(
        f"Delivery    : "
        f"{result.delivery_status.value}"
    )

    print(
        f"Grounding   : "
        f"{result.grounding_passed}"
    )

    print(
        f"LLM output  : "
        f"{result.generated_by_llm}"
    )

    print(
        f"Repair used : "
        f"{result.used_repair}"
    )

    print(
        f"Fallback    : "
        f"{result.used_fallback}"
    )

    if result.fallback_reason:

        print(
            f"Fallback reason: "
            f"{result.fallback_reason}"
        )

    print()

    print("-" * 78)

    print(
        result.final_answer
    )

    print()

    print("-" * 78)

    print(
        "Citations"
    )

    print("-" * 78)

    for citation in (
        result.citations
    ):

        print(
            f"[{citation.citation_id}] "
            f"{citation.source_label}"
        )

        print(
            f"    {citation.provenance_location}"
        )

    print()

    print(
        "REQTRACE AI provides evidence-derived "
        "engineering decision support. "
        "Final engineering and closure authority "
        "remain with the responsible human authority."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )