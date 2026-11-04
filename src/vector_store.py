"""
Persistent ChromaDB indexing for REQTRACE AI.

This module indexes validated EvidenceRecord objects using embeddings supplied
by Ollama.

It supports:
- separate collections per evidence type
- idempotent indexing
- unchanged-record skipping
- changed-record upsert
- stale-record deletion
- collection verification

It does not implement semantic retrieval or LLM generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import chromadb

from src.corpus import EvidenceCorpus
from src.embeddings import OllamaEmbeddingClient
from src.evidence import EvidenceRecord
from src.metadata import DocumentType


class VectorStoreError(RuntimeError):
    """Raised when persistent indexing cannot be completed safely."""


class EmbeddingProvider(Protocol):
    """Interface required by the indexer."""

    def embed_texts(
        self,
        texts: list[str] | tuple[str, ...],
    ) -> Any:
        ...


@dataclass(frozen=True)
class CollectionNames:
    """Collection names by evidence type."""

    frd: str
    observation: str
    code: str
    test: str
    review: str

    def for_document_type(
        self,
        document_type: DocumentType,
    ) -> str:
        mapping = {
            DocumentType.FRD: self.frd,
            DocumentType.OBSERVATION: (
                self.observation
            ),
            DocumentType.CODE: self.code,
            DocumentType.TEST: self.test,
            DocumentType.REVIEW: self.review,
        }

        return mapping[document_type]

    def as_dict(self) -> dict[str, str]:
        return {
            "frd": self.frd,
            "observation": self.observation,
            "code": self.code,
            "test": self.test,
            "review": self.review,
        }


@dataclass
class CollectionIndexStats:
    """Indexing statistics for one collection."""

    collection_name: str
    corpus_records: int = 0
    existing_records: int = 0
    inserted_or_updated: int = 0
    unchanged: int = 0
    stale_deleted: int = 0
    batches_embedded: int = 0
    embedding_dimension: int | None = None
    final_collection_count: int = 0


@dataclass
class IndexingReport:
    """Combined indexing results."""

    collections: dict[str, CollectionIndexStats] = field(
        default_factory=dict
    )

    @property
    def total_corpus_records(self) -> int:
        return sum(
            stats.corpus_records
            for stats in self.collections.values()
        )

    @property
    def total_upserted(self) -> int:
        return sum(
            stats.inserted_or_updated
            for stats in self.collections.values()
        )

    @property
    def total_unchanged(self) -> int:
        return sum(
            stats.unchanged
            for stats in self.collections.values()
        )

    @property
    def total_stale_deleted(self) -> int:
        return sum(
            stats.stale_deleted
            for stats in self.collections.values()
        )


def chroma_metadata(
    record: EvidenceRecord,
    embedding_model: str,
) -> dict[str, Any]:
    """Create scalar-only metadata for ChromaDB."""

    metadata = (
        record.metadata.to_chroma_metadata()
    )

    metadata["embedding_model"] = (
        embedding_model
    )

    metadata["evidence_schema_version"] = (
        "1.0"
    )

    return metadata


def split_batches(
    records: list[EvidenceRecord],
    batch_size: int,
) -> tuple[tuple[EvidenceRecord, ...], ...]:
    """Create deterministic evidence batches."""

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    return tuple(
        tuple(
            records[index:index + batch_size]
        )
        for index in range(
            0,
            len(records),
            batch_size,
        )
    )


class ChromaEvidenceStore:
    """Persistent ChromaDB store for validated evidence."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        collection_names: CollectionNames,
        embedding_client: OllamaEmbeddingClient,
        batch_size: int = 16,
        client: Any | None = None,
    ) -> None:
        self.database_path = (
            Path(database_path)
            .expanduser()
            .resolve()
        )

        self.collection_names = (
            collection_names
        )

        self.embedding_client = (
            embedding_client
        )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        self.batch_size = batch_size

        self.database_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = (
            client
            if client is not None
            else chromadb.PersistentClient(
                path=str(self.database_path)
            )
        )

    def get_collection(
        self,
        document_type: DocumentType,
    ) -> Any:
        """Get or create one evidence collection."""

        collection_name = (
            self.collection_names
            .for_document_type(document_type)
        )

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "evidence_type": (
                    document_type.value
                ),
                "embedding_model": (
                    self.embedding_client
                    .config
                    .model
                ),
            },
        )

    def index_corpus(
        self,
        corpus: EvidenceCorpus,
        *,
        delete_stale: bool = True,
    ) -> IndexingReport:
        """Index every evidence type."""

        report = IndexingReport()

        for document_type in DocumentType:
            records = [
                record
                for record in corpus.records
                if (
                    record.metadata.document_type
                    == document_type
                )
            ]

            stats = self.index_collection(
                document_type=document_type,
                records=records,
                delete_stale=delete_stale,
            )

            report.collections[
                document_type.value
            ] = stats

        return report

    def index_collection(
        self,
        *,
        document_type: DocumentType,
        records: list[EvidenceRecord],
        delete_stale: bool = True,
    ) -> CollectionIndexStats:
        """Synchronize one collection with the validated corpus."""

        collection = self.get_collection(
            document_type
        )

        collection_name = (
            self.collection_names
            .for_document_type(document_type)
        )

        existing = collection.get(
            include=["metadatas"]
        )

        existing_ids = list(
            existing.get("ids") or []
        )

        existing_metadatas = list(
            existing.get("metadatas") or []
        )

        existing_hashes: dict[str, str | None] = {}

        for evidence_id, metadata in zip(
            existing_ids,
            existing_metadatas,
            strict=False,
        ):
            content_hash: str | None = None

            if isinstance(metadata, dict):
                raw_hash = metadata.get(
                    "content_hash"
                )

                if raw_hash is not None:
                    content_hash = str(
                        raw_hash
                    )

            existing_hashes[
                evidence_id
            ] = content_hash

        stats = CollectionIndexStats(
            collection_name=collection_name,
            corpus_records=len(records),
            existing_records=len(existing_ids),
        )

        changed_records: list[EvidenceRecord] = []

        for record in records:
            evidence_id = (
                record.metadata.evidence_id
            )
            new_hash = (
                record.metadata.content_hash
            )

            if (
                evidence_id in existing_hashes
                and existing_hashes[evidence_id]
                == new_hash
            ):
                stats.unchanged += 1
                continue

            changed_records.append(record)

        for batch in split_batches(
            changed_records,
            self.batch_size,
        ):
            texts = [
                record.content
                for record in batch
            ]

            embedding_result = (
                self.embedding_client
                .embed_texts(texts)
            )

            embeddings = [
                list(vector)
                for vector
                in embedding_result.embeddings
            ]

            if len(embeddings) != len(batch):
                raise VectorStoreError(
                    "Embedding count does not match "
                    "evidence batch size."
                )

            collection.upsert(
                ids=[
                    record.metadata.evidence_id
                    for record in batch
                ],
                documents=[
                    record.content
                    for record in batch
                ],
                metadatas=[
                    chroma_metadata(
                        record,
                        self.embedding_client
                        .config
                        .model,
                    )
                    for record in batch
                ],
                embeddings=embeddings,
            )

            stats.inserted_or_updated += (
                len(batch)
            )
            stats.batches_embedded += 1

            if stats.embedding_dimension is None:
                stats.embedding_dimension = (
                    embedding_result.dimension
                )
            elif (
                stats.embedding_dimension
                != embedding_result.dimension
            ):
                raise VectorStoreError(
                    "Embedding dimension changed "
                    "within one indexing operation."
                )

        corpus_ids = {
            record.metadata.evidence_id
            for record in records
        }

        stale_ids = sorted(
            set(existing_ids) - corpus_ids
        )

        if delete_stale and stale_ids:
            collection.delete(
                ids=stale_ids
            )

            stats.stale_deleted = len(
                stale_ids
            )

        stats.final_collection_count = (
            collection.count()
        )

        expected_count = len(records)

        if (
            delete_stale
            and stats.final_collection_count
            != expected_count
        ):
            raise VectorStoreError(
                f"Collection '{collection_name}' "
                "contains an unexpected record count. "
                f"Expected {expected_count}, found "
                f"{stats.final_collection_count}."
            )

        return stats

    def collection_counts(
        self,
    ) -> dict[str, int]:
        """Return record counts for all evidence collections."""

        counts: dict[str, int] = {}

        for document_type in DocumentType:
            collection = self.get_collection(
                document_type
            )

            counts[
                document_type.value
            ] = collection.count()

        return counts

    def verify_corpus(
        self,
        corpus: EvidenceCorpus,
    ) -> dict[str, bool]:
        """Verify persisted IDs and content hashes."""

        verification: dict[str, bool] = {}

        for document_type in DocumentType:
            collection = self.get_collection(
                document_type
            )

            expected_records = [
                record
                for record in corpus.records
                if (
                    record.metadata.document_type
                    == document_type
                )
            ]

            result = collection.get(
                include=["metadatas"]
            )

            actual_ids = list(
                result.get("ids") or []
            )
            actual_metadatas = list(
                result.get("metadatas") or []
            )

            actual_hashes = {
                evidence_id: (
                    metadata.get("content_hash")
                    if isinstance(metadata, dict)
                    else None
                )
                for evidence_id, metadata in zip(
                    actual_ids,
                    actual_metadatas,
                    strict=False,
                )
            }

            expected_hashes = {
                record.metadata.evidence_id:
                record.metadata.content_hash
                for record in expected_records
            }

            verification[
                document_type.value
            ] = (
                actual_hashes
                == expected_hashes
            )

        return verification