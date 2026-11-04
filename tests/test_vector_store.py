from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.corpus import build_project_corpus
from src.vector_store import (
    ChromaEvidenceStore,
    CollectionNames,
)


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FakeConfig:
    model: str = "fake-embedding-model"


@dataclass(frozen=True)
class FakeResult:
    embeddings: tuple[
        tuple[float, ...],
        ...
    ]
    dimension: int


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.config = FakeConfig()
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(
        self,
        texts: list[str] | tuple[str, ...],
    ) -> FakeResult:
        normalized = tuple(texts)
        self.calls.append(normalized)

        embeddings = tuple(
            (
                float(index + 1),
                float(len(text)),
                0.5,
            )
            for index, text
            in enumerate(normalized)
        )

        return FakeResult(
            embeddings=embeddings,
            dimension=3,
        )


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    def get(
        self,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        ids = sorted(
            self.records
        )

        return {
            "ids": ids,
            "metadatas": [
                self.records[
                    evidence_id
                ]["metadata"]
                for evidence_id in ids
            ],
            "documents": [
                self.records[
                    evidence_id
                ]["document"]
                for evidence_id in ids
            ],
        }

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for (
            evidence_id,
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
            self.records[evidence_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def delete(
        self,
        *,
        ids: list[str],
    ) -> None:
        for evidence_id in ids:
            self.records.pop(
                evidence_id,
                None,
            )

    def count(self) -> int:
        return len(self.records)


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[
            str,
            FakeCollection,
        ] = {}

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> FakeCollection:
        if name not in self.collections:
            self.collections[
                name
            ] = FakeCollection(name)

        return self.collections[name]


def collection_names() -> CollectionNames:
    return CollectionNames(
        frd="frd_evidence",
        observation="observation_evidence",
        code="code_evidence",
        test="test_evidence",
        review="review_evidence",
    )


def test_initial_index_embeds_all_records(
    tmp_path: Path,
) -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    embedding_client = (
        FakeEmbeddingClient()
    )

    store = ChromaEvidenceStore(
        database_path=tmp_path,
        collection_names=(
            collection_names()
        ),
        embedding_client=(
            embedding_client
        ),
        batch_size=4,
        client=FakeChromaClient(),
    )

    report = store.index_corpus(
        corpus
    )

    assert (
        report.total_upserted
        == corpus.record_count
    )

    assert report.total_unchanged == 0

    assert sum(
        len(call)
        for call in embedding_client.calls
    ) == corpus.record_count


def test_second_index_skips_unchanged_records(
    tmp_path: Path,
) -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    embedding_client = (
        FakeEmbeddingClient()
    )

    fake_client = FakeChromaClient()

    store = ChromaEvidenceStore(
        database_path=tmp_path,
        collection_names=(
            collection_names()
        ),
        embedding_client=(
            embedding_client
        ),
        batch_size=8,
        client=fake_client,
    )

    first = store.index_corpus(
        corpus
    )

    calls_after_first = len(
        embedding_client.calls
    )

    second = store.index_corpus(
        corpus
    )

    assert (
        first.total_upserted
        == corpus.record_count
    )

    assert second.total_upserted == 0

    assert (
        second.total_unchanged
        == corpus.record_count
    )

    assert (
        len(embedding_client.calls)
        == calls_after_first
    )


def test_verify_corpus_passes(
    tmp_path: Path,
) -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    store = ChromaEvidenceStore(
        database_path=tmp_path,
        collection_names=(
            collection_names()
        ),
        embedding_client=(
            FakeEmbeddingClient()
        ),
        batch_size=16,
        client=FakeChromaClient(),
    )

    store.index_corpus(
        corpus
    )

    verification = store.verify_corpus(
        corpus
    )

    assert all(
        verification.values()
    )


def test_collection_counts_match_corpus(
    tmp_path: Path,
) -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    store = ChromaEvidenceStore(
        database_path=tmp_path,
        collection_names=(
            collection_names()
        ),
        embedding_client=(
            FakeEmbeddingClient()
        ),
        batch_size=16,
        client=FakeChromaClient(),
    )

    store.index_corpus(
        corpus
    )

    counts = store.collection_counts()

    for evidence_type, count in counts.items():
        expected = sum(
            1
            for record in corpus.records
            if (
                record.metadata
                .document_type
                .value
                == evidence_type
            )
        )

        assert count == expected