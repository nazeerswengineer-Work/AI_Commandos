"""
REQTRACE AI Phase 14A narration-prompt inspection.

Builds the fully validated Phase 13 package and the constrained
Phase 14 narrator prompt.

No Ollama call is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from src.answer_pipeline import (
    build_grounded_answer_package,
)
from src.narration_prompt import (
    build_narration_prompt,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 14A "
            "safe narration-prompt inspection."
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
        "--json",
        action="store_true",
        help=(
            "Print prompt and request as JSON."
        ),
    )

    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help=(
            "Print the complete system and user prompts."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Build and inspect the narration prompt."""

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

    if args.json:
        print(
            json.dumps(
                prompt.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 14A "
        "NARRATION PROMPT"
    )
    print("=" * 78)

    request = prompt.request

    print(
        f"Query             : "
        f"{request.query}"
    )

    print(
        f"Assessment        : "
        f"{request.assessment}"
    )

    print(
        f"Confidence        : "
        f"{request.confidence}"
    )

    print(
        f"Closure           : "
        f"{request.closure_status}"
    )

    print(
        f"Engineering IDs   : "
        f"{request.engineering_id_allowlist}"
    )

    print(
        f"Citations         : "
        f"{request.citation_allowlist}"
    )

    print(
        f"Evidence blocks   : "
        f"{len(request.evidence_blocks)}"
    )

    print(
        f"Allowed claims    : "
        f"{len(request.allowed_claims)}"
    )

    print(
        f"Prohibited claims : "
        f"{len(request.prohibited_claims)}"
    )

    if args.show_prompt:
        print()
        print(
            prompt.combined_prompt
        )

    else:
        print()
        print(
            "Use --show-prompt to inspect the complete prompt."
        )

    print()
    print(
        "Phase 14A generated a prompt only. "
        "No Ollama model was called."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )