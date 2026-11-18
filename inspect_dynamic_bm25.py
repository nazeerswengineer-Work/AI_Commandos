from pathlib import Path

from src.dynamic_bm25 import (
    DynamicBM25Index,
)


base = (
    Path(__file__)
    .resolve()
    .parent
)

index = DynamicBM25Index.from_manifest(
    base
    / "data"
    / "ingestion_manifest.json"
)

results = index.retrieve(
    "REQ-900",
    top_k=5,
)
# results = index.retrieve(
#     "unsupported command value",
#     top_k=10,
# )
print(
    f"Dynamic BM25 records: "
    f"{index.record_count}"
)

for result in results:

    print()

    print(
        result.evidence_id
    )

    print(
        result.retrieval_method,
        result.score,
    )

    print(
        result.metadata.get(
            "document_name"
        )
    )

    print(
        result.metadata.get(
            "requirement_id"
        )
    )