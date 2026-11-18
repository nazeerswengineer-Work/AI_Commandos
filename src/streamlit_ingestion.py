"""
Streamlit-facing incremental ingestion integration.

This module reuses the validated Phase 16H ingestion pipeline.
It contains no alternate parsing, embedding, or Chroma logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from config import get_settings

from src.document_discovery import (
    detect_document_changes,
)

from src.dynamic_vector_adapter import (
    DynamicVectorAdapter,
)

from src.embeddings import (
    EmbeddingClientConfig,
    OllamaEmbeddingClient,
)

from src.incremental_ingestion import (
    IncrementalIngestionCoordinator,
    IncrementalSyncReport,
)

from src.ingestion_lock import (
    IngestionFileLock,
)

from src.ingestion_manifest import (
    IngestionManifest,
    load_ingestion_manifest,
)

from src.intelligent_chunking import (
    ChunkingConfig,
)

from src.vector_store import (
    CollectionNames,
)


@dataclass(frozen=True)
class DocumentIndexStatus:
    """Read-only document synchronization status."""

    discovered: int
    indexed: int

    new: int
    changed: int
    unchanged: int
    deleted: int
    failed: int

    manifest_version: str

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "discovered": self.discovered,
            "indexed": self.indexed,

            "new": self.new,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "deleted": self.deleted,
            "failed": self.failed,

            "manifest_version": (
                self.manifest_version
            ),
        }


def project_paths(
    base_directory: str | Path,
) -> tuple[
    Path,
    Path,
    Path,
]:
    """Return documents, manifest and lock paths."""

    base = (
        Path(base_directory)
        .expanduser()
        .resolve()
    )

    settings = get_settings()

    documents_root = (
        Path(
            settings.documents_path
        )
        .expanduser()
        .resolve()
    )

    manifest_path = (
        base
        / "data"
        / "ingestion_manifest.json"
    ).resolve()

    lock_path = (
        manifest_path.parent
        / "incremental_ingestion.lock"
    )

    return (
        documents_root,
        manifest_path,
        lock_path,
    )


def inspect_document_index(
    base_directory: str | Path,
) -> DocumentIndexStatus:
    """
    Inspect filesystem-to-index state.

    This operation does not call Ollama or mutate ChromaDB.
    """

    (
        documents_root,
        manifest_path,
        _,
    ) = project_paths(
        base_directory
    )

    manifest = (
        load_ingestion_manifest(
            manifest_path
        )
    )

    changes = (
        detect_document_changes(
            documents_root=(
                documents_root
            ),
            manifest=manifest,
        )
    )

    return DocumentIndexStatus(
        discovered=(
            changes.discovered_count
        ),

        indexed=len(
            manifest.documents
        ),

        new=len(
            changes.new_documents
        ),

        changed=len(
            changes.changed_documents
        ),

        unchanged=len(
            changes.unchanged_documents
        ),

        deleted=len(
            changes.deleted_documents
        ),

        failed=len(
            changes.failed_documents
        ),

        manifest_version=(
            manifest.manifest_version
        ),
    )


def synchronize_document_index(
    base_directory: str | Path,
    *,
    continue_on_error: bool = True,
) -> IncrementalSyncReport:
    """
    Apply incremental synchronization.

    NEW:
        parse -> chunk -> embed -> upsert -> verify

    CHANGED:
        parse new -> embed -> replace vectors -> verify

    DELETED:
        remove vectors -> remove manifest records

    UNCHANGED:
        skip
    """

    base = (
        Path(base_directory)
        .expanduser()
        .resolve()
    )

    settings = get_settings()

    (
        documents_root,
        manifest_path,
        lock_path,
    ) = project_paths(
        base
    )

    embedding_config = (
        EmbeddingClientConfig(
            base_url=(
                settings
                .ollama_base_url
            ),

            model=(
                settings
                .ollama_embed_model
            ),

            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),

            read_timeout_seconds=(
                settings
                .llm_timeout_seconds
            ),
        )
    )

    collection_names = (
        CollectionNames(
            frd=(
                settings
                .frd_collection_name
            ),

            observation=(
                settings
                .observation_collection_name
            ),

            code=(
                settings
                .code_collection_name
            ),

            test=(
                settings
                .test_collection_name
            ),

            review=(
                settings
                .reviews_collection_name
            ),
        )
    )

    with IngestionFileLock(
        lock_path,
        timeout_seconds=2.0,
        stale_after_seconds=3600.0,
    ):

        manifest: IngestionManifest = (
            load_ingestion_manifest(
                manifest_path
            )
        )

        chroma_client = (
            chromadb.PersistentClient(
                path=str(
                    settings
                    .chroma_db_path
                )
            )
        )

        with OllamaEmbeddingClient(
            embedding_config
        ) as embedding_client:

            vector_adapter = (
                DynamicVectorAdapter(
                    chroma_client=(
                        chroma_client
                    ),

                    collection_names=(
                        collection_names
                    ),

                    embedding_client=(
                        embedding_client
                    ),
                )
            )

            coordinator = (
                IncrementalIngestionCoordinator(
                    vector_adapter=(
                        vector_adapter
                    ),

                    chunking_config=(
                        ChunkingConfig(
                            maximum_characters=4000,
                            overlap_characters=400,
                        )
                    ),

                    continue_on_error=(
                        continue_on_error
                    ),
                )
            )

            return (
                coordinator.synchronize(
                    documents_root=(
                        documents_root
                    ),

                    manifest=manifest,

                    manifest_path=(
                        manifest_path
                    ),

                    apply=True,
                )
            )