"""
Inspect the current REQTRACE AI deterministic evidence inventory.
"""

from __future__ import annotations

from pathlib import Path

from src.evidence_inventory import (
    InventoryEntityType,
    InventoryOperation,
    InventoryQuery,
    answer_inventory_query,
    load_inventory_summary,
)


def main() -> int:
    """Print current observation inventory."""

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    summary = load_inventory_summary(
        base_directory
        / "data"
        / "ingestion_manifest.json"
    )

    open_answer = (
        answer_inventory_query(
            summary,

            InventoryQuery(
                operation=(
                    InventoryOperation.COUNT
                ),

                entity_type=(
                    InventoryEntityType
                    .OBSERVATION
                ),

                status_filter="OPEN",
            ),
        )
    )

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - EVIDENCE INVENTORY"
    )

    print("=" * 78)

    print(
        f"Manifest version : "
        f"{summary.manifest_version}"
    )

    print(
        f"Requirements     : "
        f"{len(summary.requirements)}"
    )

    print(
        f"Observations     : "
        f"{len(summary.observations)}"
    )

    print(
        f"Tests            : "
        f"{len(summary.tests)}"
    )

    print(
        f"Code records     : "
        f"{len(summary.code_records)}"
    )

    print(
        f"Reviews          : "
        f"{len(summary.reviews)}"
    )

    print()

    print(
        open_answer.answer_text
    )

    print()

    print(
        open_answer.source_statement
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )