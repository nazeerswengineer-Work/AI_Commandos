"""
REQTRACE AI production health-check command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings

from src.health import (
    HealthStatus,
    evaluate_system_health,
)


def parse_arguments() -> argparse.Namespace:
    """Parse health-check arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI production health check."
        )
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    """Run system-health evaluation."""

    args = parse_arguments()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    health = evaluate_system_health(
        settings=get_settings(),
        base_directory=(
            base_directory
        ),
        model_name="llama3.2:1b",
        include_deterministic_smoke_test=(
            not args.skip_smoke_test
        ),
    )

    if args.json:
        print(
            json.dumps(
                health.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

    else:
        print()

        print("=" * 78)

        print(
            "REQTRACE AI - PRODUCTION HEALTH"
        )

        print("=" * 78)

        print(
            f"Overall status : "
            f"{health.status.value}"
        )

        print(
            f"Core ready     : "
            f"{health.core_ready}"
        )

        print(
            f"Narration ready: "
            f"{health.narration_ready}"
        )

        print()

        for check in health.checks:

            print(
                f"[{check.status.value}] "
                f"{check.component}"
            )

            print(
                f"  {check.message}"
            )

    return (
        1
        if health.status
        == HealthStatus.UNAVAILABLE
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )