"""
Dynamic vector adapter for REQTRACE AI Phase 16H.

This module writes Phase 16H dynamic chunks into the existing
persistent ChromaDB evidence collections.

It reuses:

- src.embeddings.OllamaEmbeddingClient
- the existing Chroma PersistentClient
- the configured evidence collection names
- the configured nomic-embed-text model

It does not create a parallel vector database or new collection family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.ingestion_content import (
    IngestionChunk,
)
from pathlib import Path

class DynamicVectorError(
    RuntimeError
):
    """Base error for dynamic vector operations."""


class DynamicVectorVerificationError(
    DynamicVectorError
):
    """Raised when written vector IDs cannot be verified."""


class DynamicVectorRollbackError(
    DynamicVectorError
):
    """Raised when rollback cannot restore the previous vectors."""


class EmbeddingProvider(
    Protocol
):
    """Embedding interface already supplied by OllamaEmbeddingClient."""

    @property
    def model(self) -> str:
        """Return configured embedding model."""

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Any:
        """Generate a validated embedding batch."""


@dataclass(frozen=True)
class DynamicVectorWriteResult:
    """Result of one document-level vector write."""

    document_id: str
    document_type: str
    collection_name: str

    inserted_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]

    embedding_dimension: int

    verified: bool

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert result into JSON-safe data."""

        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "collection_name": (
                self.collection_name
            ),
            "inserted_ids": list(
                self.inserted_ids
            ),
            "removed_ids": list(
                self.removed_ids
            ),
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "verified": self.verified,
        }


@dataclass(frozen=True)
class StoredVectorSnapshot:
    """Restorable copy of existing Chroma records."""

    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[
        dict[str, Any],
        ...
    ]
    embeddings: tuple[
        tuple[float, ...],
        ...
    ]

    @property
    def empty(
        self,
    ) -> bool:
        """Return whether snapshot contains no vectors."""

        return not self.ids


def collection_name_for_type(
    *,
    document_type: str,
    collection_names: Any,
) -> str:
    """
    Map normalized document type to the existing configured collection.

    CollectionNames is the existing src.vector_store.CollectionNames
    instance.
    """

    normalized = (
        document_type
        .strip()
        .lower()
    )

    attribute_by_type = {
        "frd": "frd",
        "observation": "observation",
        "code": "code",
        "test": "test",
        "review": "review",
    }

    attribute = attribute_by_type.get(
        normalized
    )

    if attribute is None:
        raise DynamicVectorError(
            "Dynamic indexing does not support "
            f"document type {document_type!r}. "
            "Classify the source as frd, observation, "
            "code, test, or review before indexing."
        )

    value = getattr(
        collection_names,
        attribute,
        None,
    )

    if not isinstance(
        value,
        str,
    ) or not value.strip():
        raise DynamicVectorError(
            "No configured Chroma collection exists "
            f"for document type {document_type!r}."
        )

    return value.strip()


def flatten_ids(
    payload: Any,
) -> tuple[str, ...]:
    """Normalize Chroma ID response shapes."""

    if not isinstance(
        payload,
        dict,
    ):
        return ()

    raw_ids = payload.get(
        "ids",
        [],
    )

    if not isinstance(
        raw_ids,
        list,
    ):
        return ()

    if (
        raw_ids
        and isinstance(
            raw_ids[0],
            list,
        )
    ):
        raw_ids = [
            value
            for row in raw_ids
            for value in row
        ]

    return tuple(
        str(value)
        for value in raw_ids
    )


def verify_ids_present(
    *,
    collection: Any,
    expected_ids: Sequence[str],
) -> None:
    """Verify all vector IDs exist after upsert."""

    expected = tuple(
        dict.fromkeys(
            str(value)
            for value in expected_ids
        )
    )

    if not expected:
        raise DynamicVectorVerificationError(
            "No vector IDs were supplied for verification."
        )

    response = collection.get(
        ids=list(
            expected
        ),
        include=[],
    )

    actual = set(
        flatten_ids(
            response
        )
    )

    missing = [
        value
        for value in expected
        if value not in actual
    ]

    if missing:
        raise DynamicVectorVerificationError(
            "Chroma upsert verification failed. "
            "Missing IDs: "
            + ", ".join(
                missing
            )
        )


def verify_ids_absent(
    *,
    collection: Any,
    removed_ids: Sequence[str],
) -> None:
    """Verify deleted vector IDs no longer exist."""

    expected_absent = tuple(
        dict.fromkeys(
            str(value)
            for value in removed_ids
        )
    )

    if not expected_absent:
        return

    response = collection.get(
        ids=list(
            expected_absent
        ),
        include=[],
    )

    remaining = set(
        flatten_ids(
            response
        )
    )

    if remaining:
        raise DynamicVectorVerificationError(
            "Chroma deletion verification failed. "
            "IDs still present: "
            + ", ".join(
                sorted(
                    remaining
                )
            )
        )


def extract_embedding_batch(
    result: Any,
    *,
    expected_count: int,
) -> tuple[
    tuple[float, ...],
    ...
]:
    """
    Extract and validate embeddings from EmbeddingBatchResult.

    OllamaEmbeddingClient already validates count and dimensions. This
    adapter performs a defensive validation at the mutation boundary.
    """

    raw_embeddings = getattr(
        result,
        "embeddings",
        None,
    )

    if raw_embeddings is None:
        raise DynamicVectorError(
            "Embedding provider returned no embeddings."
        )

    embeddings = tuple(
        tuple(
            float(value)
            for value in embedding
        )
        for embedding in raw_embeddings
    )

    if len(embeddings) != expected_count:
        raise DynamicVectorError(
            "Embedding count does not match chunk count."
        )

    if not embeddings:
        raise DynamicVectorError(
            "Embedding provider returned an empty batch."
        )

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if (
        0 in dimensions
        or len(dimensions) != 1
    ):
        raise DynamicVectorError(
            "Embedding vectors have invalid or "
            "inconsistent dimensions."
        )

    return embeddings


def dynamic_chroma_metadata(
    *,
    chunk: IngestionChunk,
    embedding_model: str,
) -> dict[str, Any]:
    """
    Create retrieval-compatible metadata for dynamically indexed evidence.

    Phase 16H chunks use generic physical-location fields:

        document_name
        line_number_start
        line_number_end

    Existing REQTRACE AI citation and grounding layers historically use:

        file_name
        start_line
        end_line

    For dynamic source-code evidence we preserve both representations.
    """

    metadata = dict(
        chunk.to_metadata()
    )

    metadata[
        "evidence_id"
    ] = chunk.chunk_id

    metadata[
        "embedding_model"
    ] = embedding_model

    metadata[
        "dynamic_ingestion"
    ] = True

    source_extension = (
        Path(
            chunk.document_name
        )
        .suffix
        .lower()
    )

    if source_extension:

        metadata[
            "source_extension"
        ] = source_extension

    # ---------------------------------------------------------
    # Existing REQTRACE AI code-provenance compatibility
    # ---------------------------------------------------------

    if (
        chunk.document_type
        .strip()
        .lower()
        == "code"
    ):

        # The physical source file for dynamic code evidence.
        metadata[
            "file_name"
        ] = chunk.document_name

        # Preserve the portable source path as well.
        metadata[
            "file_path"
        ] = chunk.relative_path

        if (
            chunk.line_number_start
            is not None
        ):

            metadata[
                "start_line"
            ] = (
                chunk.line_number_start
            )

        if (
            chunk.line_number_end
            is not None
        ):

            metadata[
                "end_line"
            ] = (
                chunk.line_number_end
            )

        if chunk.function_name:

            metadata[
                "function_name"
            ] = (
                chunk.function_name
            )

        if chunk.class_name:

            metadata[
                "class_name"
            ] = (
                chunk.class_name
            )

    return {
        str(key): value

        for key, value
        in metadata.items()

        if value is not None
    }


def snapshot_vectors(
    *,
    collection: Any,
    vector_ids: Sequence[str],
) -> StoredVectorSnapshot:
    """Capture old vectors before a changed-document replacement."""

    unique_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in vector_ids
            if str(value).strip()
        )
    )

    if not unique_ids:
        return StoredVectorSnapshot(
            ids=(),
            documents=(),
            metadatas=(),
            embeddings=(),
        )

    response = collection.get(
        ids=list(
            unique_ids
        ),
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ],
    )

    ids = flatten_ids(
        response
    )

    raw_documents = response.get(
        "documents",
        [],
    )

    raw_metadatas = response.get(
        "metadatas",
        [],
    )

    raw_embeddings = response.get(
        "embeddings",
        [],
    )

    documents = tuple(
        str(value)
        for value in raw_documents
    )

    metadatas = tuple(
        dict(value)
        for value in raw_metadatas
    )

    embeddings = tuple(
        tuple(
            float(component)
            for component in vector
        )
        for vector in raw_embeddings
    )

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(embeddings)
    ):
        raise DynamicVectorError(
            "Existing Chroma vectors could not be "
            "snapshotted consistently."
        )

    return StoredVectorSnapshot(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def restore_snapshot(
    *,
    collection: Any,
    snapshot: StoredVectorSnapshot,
) -> None:
    """Restore a previously captured vector snapshot."""

    if snapshot.empty:
        return

    collection.upsert(
        ids=list(
            snapshot.ids
        ),
        documents=list(
            snapshot.documents
        ),
        metadatas=list(
            snapshot.metadatas
        ),
        embeddings=[
            list(vector)
            for vector in snapshot.embeddings
        ],
    )

    verify_ids_present(
        collection=collection,
        expected_ids=snapshot.ids,
    )


class DynamicVectorAdapter:
    """
    Document-level adapter over existing REQTRACE AI Chroma collections.

    New documents:
        embed -> upsert -> verify

    Changed documents:
        snapshot old -> embed new -> remove old -> upsert new -> verify
        -> restore old snapshot if mutation fails

    Deleted documents:
        delete old IDs -> verify absent
    """

    def __init__(
        self,
        *,
        chroma_client: Any,
        collection_names: Any,
        embedding_client: EmbeddingProvider,
    ) -> None:
        """Create adapter from existing production dependencies."""

        self.chroma_client = (
            chroma_client
        )

        self.collection_names = (
            collection_names
        )

        self.embedding_client = (
            embedding_client
        )

    def get_collection(
        self,
        document_type: str,
    ) -> tuple[str, Any]:
        """Open an existing configured evidence collection."""

        collection_name = (
            collection_name_for_type(
                document_type=document_type,
                collection_names=(
                    self.collection_names
                ),
            )
        )

        try:
            collection = (
                self.chroma_client
                .get_collection(
                    name=collection_name
                )
            )

        except Exception as exc:
            raise DynamicVectorError(
                "Unable to open existing Chroma "
                f"collection {collection_name!r}. "
                "Run the baseline corpus index first."
            ) from exc

        return (
            collection_name,
            collection,
        )

    def upsert_document(
        self,
        *,
        chunks: Sequence[
            IngestionChunk
        ],
        previous_chunk_ids: Sequence[
            str
        ] = (),
        previous_document_type: (
            str
            | None
        ) = None,
    ) -> DynamicVectorWriteResult:
        """
        Embed and upsert one NEW or CHANGED document.

        If document type changes, old IDs are removed from the previous
        type's collection and new IDs are written to the new collection.
        """

        prepared_chunks = tuple(
            chunks
        )

        if not prepared_chunks:
            raise DynamicVectorError(
                "Cannot index an empty chunk collection."
            )

        document_ids = {
            chunk.document_id
            for chunk in prepared_chunks
        }

        document_types = {
            chunk.document_type
            for chunk in prepared_chunks
        }

        if len(document_ids) != 1:
            raise DynamicVectorError(
                "All chunks must belong to one document."
            )

        if len(document_types) != 1:
            raise DynamicVectorError(
                "All chunks must have one document type."
            )

        document_id = next(
            iter(
                document_ids
            )
        )

        document_type = next(
            iter(
                document_types
            )
        )

        new_collection_name, new_collection = (
            self.get_collection(
                document_type
            )
        )

        old_document_type = (
            previous_document_type
            or document_type
        )

        old_collection_name, old_collection = (
            self.get_collection(
                old_document_type
            )
        )

        old_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in previous_chunk_ids
                if str(value).strip()
            )
        )

        new_ids = tuple(
            chunk.chunk_id
            for chunk in prepared_chunks
        )

        if len(new_ids) != len(
            set(
                new_ids
            )
        ):
            raise DynamicVectorError(
                "Dynamic chunks contain duplicate IDs."
            )

        # Embed before modifying any existing vectors.
        embedding_result = (
            self.embedding_client
            .embed_texts(
                [
                    chunk.content
                    for chunk in prepared_chunks
                ]
            )
        )

        embeddings = extract_embedding_batch(
            embedding_result,
            expected_count=len(
                prepared_chunks
            ),
        )

        embedding_model = str(
            getattr(
                self.embedding_client,
                "model",
                "",
            )
        ).strip()

        if not embedding_model:
            raise DynamicVectorError(
                "Embedding client does not expose "
                "its configured model."
            )

        old_snapshot = snapshot_vectors(
            collection=old_collection,
            vector_ids=old_ids,
        )

        try:
            if old_ids:
                old_collection.delete(
                    ids=list(
                        old_ids
                    )
                )

                verify_ids_absent(
                    collection=old_collection,
                    removed_ids=old_ids,
                )

            new_collection.upsert(
                ids=list(
                    new_ids
                ),
                documents=[
                    chunk.content
                    for chunk in prepared_chunks
                ],
                metadatas=[
                    dynamic_chroma_metadata(
                        chunk=chunk,
                        embedding_model=(
                            embedding_model
                        ),
                    )
                    for chunk in prepared_chunks
                ],
                embeddings=[
                    list(vector)
                    for vector in embeddings
                ],
            )

            verify_ids_present(
                collection=new_collection,
                expected_ids=new_ids,
            )

        except Exception as exc:
            # Remove partial new writes first.
            try:
                new_collection.delete(
                    ids=list(
                        new_ids
                    )
                )

            except Exception:
                pass

            try:
                restore_snapshot(
                    collection=old_collection,
                    snapshot=old_snapshot,
                )

            except Exception as rollback_exc:
                raise DynamicVectorRollbackError(
                    "Dynamic vector mutation failed and "
                    "the previous snapshot could not be restored."
                ) from rollback_exc

            if isinstance(
                exc,
                DynamicVectorError,
            ):
                raise

            raise DynamicVectorError(
                "Dynamic vector mutation failed. "
                "Previous vectors were restored."
            ) from exc

        return DynamicVectorWriteResult(
            document_id=document_id,
            document_type=document_type,
            collection_name=(
                new_collection_name
            ),
            inserted_ids=new_ids,
            removed_ids=old_ids,
            embedding_dimension=len(
                embeddings[0]
            ),
            verified=True,
        )

    def delete_document(
        self,
        *,
        document_type: str,
        chunk_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Delete and verify vectors for one removed document."""

        _, collection = (
            self.get_collection(
                document_type
            )
        )

        unique_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in chunk_ids
                if str(value).strip()
            )
        )

        if not unique_ids:
            return ()

        collection.delete(
            ids=list(
                unique_ids
            )
        )

        verify_ids_absent(
            collection=collection,
            removed_ids=unique_ids,
        )

        return unique_ids