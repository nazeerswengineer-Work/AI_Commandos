"""
Tests for Phase 16H dynamic vector integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.dynamic_vector_adapter import (
    DynamicVectorAdapter,
    DynamicVectorError,
    collection_name_for_type,
)

from src.ingestion_content import (
    IngestionChunk,
)

from src.dynamic_vector_adapter import (
    dynamic_chroma_metadata,
)


@dataclass(frozen=True)
class FakeEmbeddingResult:
    embeddings: tuple[
        tuple[float, ...],
        ...
    ]


class FakeEmbeddingClient:

    model = (
        "nomic-embed-text:latest"
    )

    def embed_texts(
        self,
        texts,
    ):
        return FakeEmbeddingResult(
            embeddings=tuple(
                (
                    float(index),
                    0.5,
                    1.0,
                )
                for index, _
                in enumerate(
                    texts,
                    start=1,
                )
            )
        )


class FakeCollection:

    def __init__(
        self,
    ) -> None:
        self.rows = {}

    def upsert(
        self,
        *,
        ids,
        documents,
        metadatas,
        embeddings,
    ) -> None:
        for (
            vector_id,
            document,
            metadata,
            embedding,
        ) in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            self.rows[
                vector_id
            ] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def delete(
        self,
        *,
        ids,
    ) -> None:
        for vector_id in ids:
            self.rows.pop(
                vector_id,
                None,
            )

    def get(
        self,
        *,
        ids,
        include,
    ):
        found = [
            vector_id
            for vector_id in ids
            if vector_id in self.rows
        ]

        response = {
            "ids": found,
        }

        if "documents" in include:
            response["documents"] = [
                self.rows[
                    vector_id
                ][
                    "document"
                ]
                for vector_id in found
            ]

        if "metadatas" in include:
            response["metadatas"] = [
                self.rows[
                    vector_id
                ][
                    "metadata"
                ]
                for vector_id in found
            ]

        if "embeddings" in include:
            response["embeddings"] = [
                self.rows[
                    vector_id
                ][
                    "embedding"
                ]
                for vector_id in found
            ]

        return response


class FailingCollection(
    FakeCollection
):

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.fail_next_upsert = False

    def upsert(
        self,
        **kwargs,
    ) -> None:
        if self.fail_next_upsert:
            self.fail_next_upsert = False

            raise RuntimeError(
                "simulated failure"
            )

        super().upsert(
            **kwargs
        )


class FakeClient:

    def __init__(
        self,
        collections,
    ) -> None:
        self.collections = (
            collections
        )

    def get_collection(
        self,
        *,
        name,
    ):
        if name not in self.collections:
            raise KeyError(
                name
            )

        return self.collections[
            name
        ]


@dataclass(frozen=True)
class FakeCollectionNames:

    frd: str = "frd_evidence"
    observation: str = (
        "observation_evidence"
    )
    code: str = "code_evidence"
    test: str = "test_evidence"
    review: str = "review_evidence"


def make_chunk(
    *,
    chunk_id: str = "dynamic-001",
    document_type: str = "frd",
) -> IngestionChunk:

    return IngestionChunk(
        chunk_id=chunk_id,
        document_id="a" * 64,
        document_name="FRD_REQ-900.md",
        relative_path=(
            "dynamic_test/FRD_REQ-900.md"
        ),
        document_type=document_type,
        document_category="FRD",
        file_hash="b" * 64,
        content=(
            "REQ-900 requires null input rejection."
        ),
        content_hash="c" * 64,
        chunk_index=0,
        requirement_ids=(
            "REQ-900",
        ),
        parsing_method="markdown_heading",
    )


def make_adapter(
    collection=None,
):
    active_collection = (
        collection
        or FakeCollection()
    )

    client = FakeClient(
        {
            "frd_evidence": (
                active_collection
            ),
            "observation_evidence": (
                FakeCollection()
            ),
            "code_evidence": (
                FakeCollection()
            ),
            "test_evidence": (
                FakeCollection()
            ),
            "review_evidence": (
                FakeCollection()
            ),
        }
    )

    adapter = DynamicVectorAdapter(
        chroma_client=client,
        collection_names=(
            FakeCollectionNames()
        ),
        embedding_client=(
            FakeEmbeddingClient()
        ),
    )

    return (
        adapter,
        active_collection,
    )


def test_collection_mapping() -> None:

    names = (
        FakeCollectionNames()
    )

    assert (
        collection_name_for_type(
            document_type="frd",
            collection_names=names,
        )
        == "frd_evidence"
    )

    assert (
        collection_name_for_type(
            document_type="observation",
            collection_names=names,
        )
        == "observation_evidence"
    )


def test_unknown_document_type_is_rejected() -> None:

    with pytest.raises(
        DynamicVectorError
    ):
        collection_name_for_type(
            document_type="other",
            collection_names=(
                FakeCollectionNames()
            ),
        )


def test_new_document_is_upserted_and_verified() -> None:

    adapter, collection = (
        make_adapter()
    )

    result = adapter.upsert_document(
        chunks=[
            make_chunk()
        ]
    )

    assert result.verified is True

    assert result.inserted_ids == (
        "dynamic-001",
    )

    assert (
        "dynamic-001"
        in collection.rows
    )

    metadata = collection.rows[
        "dynamic-001"
    ][
        "metadata"
    ]

    assert (
        metadata[
            "requirement_id"
        ]
        == "REQ-900"
    )

    assert (
        metadata[
            "embedding_model"
        ]
        == "nomic-embed-text:latest"
    )


def test_changed_document_replaces_old_vectors() -> None:

    adapter, collection = (
        make_adapter()
    )

    collection.upsert(
        ids=[
            "old-vector"
        ],
        documents=[
            "old content"
        ],
        metadatas=[
            {
                "evidence_id": (
                    "old-vector"
                )
            }
        ],
        embeddings=[
            [
                0.1,
                0.2,
                0.3,
            ]
        ],
    )

    result = adapter.upsert_document(
        chunks=[
            make_chunk()
        ],
        previous_chunk_ids=[
            "old-vector"
        ],
        previous_document_type=(
            "frd"
        ),
    )

    assert (
        "old-vector"
        not in collection.rows
    )

    assert (
        "dynamic-001"
        in collection.rows
    )

    assert result.removed_ids == (
        "old-vector",
    )


def test_failed_write_restores_old_vectors() -> None:

    collection = (
        FailingCollection()
    )

    adapter, _ = make_adapter(
        collection
    )

    collection.upsert(
        ids=[
            "old-vector"
        ],
        documents=[
            "old content"
        ],
        metadatas=[
            {
                "evidence_id": (
                    "old-vector"
                )
            }
        ],
        embeddings=[
            [
                0.1,
                0.2,
                0.3,
            ]
        ],
    )

    collection.fail_next_upsert = (
        True
    )

    with pytest.raises(
        DynamicVectorError
    ):
        adapter.upsert_document(
            chunks=[
                make_chunk()
            ],
            previous_chunk_ids=[
                "old-vector"
            ],
            previous_document_type=(
                "frd"
            ),
        )

    assert (
        "old-vector"
        in collection.rows
    )

    assert (
        "dynamic-001"
        not in collection.rows
    )


def test_deleted_document_vectors_are_removed() -> None:

    adapter, collection = (
        make_adapter()
    )

    collection.upsert(
        ids=[
            "old-vector"
        ],
        documents=[
            "old content"
        ],
        metadatas=[
            {
                "evidence_id": (
                    "old-vector"
                )
            }
        ],
        embeddings=[
            [
                0.1,
                0.2,
                0.3,
            ]
        ],
    )

    removed = adapter.delete_document(
        document_type="frd",
        chunk_ids=[
            "old-vector"
        ],
    )

    assert removed == (
        "old-vector",
    )

    assert (
        "old-vector"
        not in collection.rows
    )

def test_dynamic_code_metadata_preserves_legacy_provenance_fields() -> None:
    """Dynamic source code must satisfy existing citation provenance."""

    chunk = IngestionChunk(
        chunk_id="dynamic-code-001",

        document_id="a" * 64,

        document_name=(
            "req910_handler.c"
        ),

        relative_path=(
            "dynamic_acceptance/"
            "req910_handler.c"
        ),

        document_type="code",

        document_category=(
            "Source Code"
        ),

        file_hash="b" * 64,

        content=(
            "int validate_command(int command)\n"
            "{\n"
            "    return 1;\n"
            "}"
        ),

        content_hash="c" * 64,

        chunk_index=0,

        line_number_start=7,

        line_number_end=10,

        function_name=(
            "validate_command"
        ),

        requirement_ids=(
            "REQ-910",
        ),

        observation_ids=(
            "OBS-910",
        ),

        test_case_ids=(
            "TC-910",
        ),

        parsing_method=(
            "c_family_function"
        ),
    )

    metadata = (
        dynamic_chroma_metadata(
            chunk=chunk,

            embedding_model=(
                "nomic-embed-text:latest"
            ),
        )
    )

    assert (
        metadata[
            "file_name"
        ]
        == "req910_handler.c"
    )

    assert (
        metadata[
            "file_path"
        ]
        == (
            "dynamic_acceptance/"
            "req910_handler.c"
        )
    )

    assert (
        metadata[
            "function_name"
        ]
        == "validate_command"
    )

    assert (
        metadata[
            "start_line"
        ]
        == 7
    )

    assert (
        metadata[
            "end_line"
        ]
        == 10
    )

    # Referenced TC remains traceability metadata.
    assert (
        metadata[
            "test_case_ids"
        ]
        == "TC-910"
    )

    # Code is not actual Test evidence.
    assert (
        "test_case_id"
        not in metadata
    )