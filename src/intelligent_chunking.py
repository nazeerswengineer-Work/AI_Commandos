"""
Intelligent chunking for dynamically ingested REQTRACE AI documents.

The chunker preserves:

- source section or page
- REQ / OBS / TC identifiers
- test result status
- source-code function/class names
- physical line ranges

Source-code sections are kept intact whenever they fit the configured
size. Long sections are divided using physical source lines.

Document sections are divided by paragraphs before falling back to
character windows.
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass

from src.ingestion_content import (
    IngestionChunk,
    ParsedDocument,
    ParsedSection,
)

from src.ingestion_metadata import (
    extract_engineering_ids,
    extract_result_status,
    ordered_unique,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """Dynamic ingestion chunking configuration."""

    maximum_characters: int = 4000
    overlap_characters: int = 400

    minimum_characters: int = 80

    def __post_init__(
        self,
    ) -> None:
        """Validate chunking limits."""

        if self.maximum_characters < 500:
            raise ValueError(
                "maximum_characters must be at least 500."
            )

        if self.overlap_characters < 0:
            raise ValueError(
                "overlap_characters must not be negative."
            )

        if (
            self.overlap_characters
            >= self.maximum_characters
        ):
            raise ValueError(
                "overlap_characters must be smaller "
                "than maximum_characters."
            )

        if self.minimum_characters < 1:
            raise ValueError(
                "minimum_characters must be positive."
            )


def content_hash(
    content: str,
) -> str:
    """Calculate normalized chunk SHA-256."""

    normalized = (
        content
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def build_chunk_id(
    *,
    document_id: str,
    file_hash: str,
    chunk_index: int,
    content_digest: str,
) -> str:
    """Build deterministic chunk identity."""

    payload = (
        f"{document_id}:"
        f"{file_hash}:"
        f"{chunk_index}:"
        f"{content_digest}"
    )

    return (
        "DYN::"
        + hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()
    )


def split_paragraphs(
    content: str,
) -> list:
    """Split document content at paragraph boundaries."""

    return [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            content,
        )
        if paragraph.strip()
    ]


def split_character_windows(
    content: str,
    *,
    maximum_characters: int,
    overlap_characters: int,
) -> list:
    """Fallback character-based splitting."""

    output: list[str] = []

    clean_content = (
        content.strip()
    )

    if not clean_content:
        return output

    start = 0

    while start < len(
        clean_content
    ):
        desired_end = min(
            len(clean_content),
            start
            + maximum_characters,
        )

        end = desired_end

        if desired_end < len(
            clean_content
        ):
            boundary = max(
                clean_content.rfind(
                    "\n",
                    start,
                    desired_end,
                ),
                clean_content.rfind(
                    ". ",
                    start,
                    desired_end,
                ),
                clean_content.rfind(
                    " ",
                    start,
                    desired_end,
                ),
            )

            if boundary > start:
                end = (
                    boundary + 1
                )

        chunk = clean_content[
            start:end
        ].strip()

        if chunk:
            output.append(
                chunk
            )

        if end >= len(
            clean_content
        ):
            break

        next_start = max(
            0,
            end
            - overlap_characters,
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return output


def split_document_section(
    section: ParsedSection,
    config: ChunkingConfig,
) -> list:
    """Split a non-code document section."""

    if (
        len(
            section.content
        )
        <= config.maximum_characters
    ):
        return [
            section.content.strip()
        ]

    paragraphs = split_paragraphs(
        section.content
    )

    if len(paragraphs) <= 1:
        return split_character_windows(
            section.content,
            maximum_characters=(
                config.maximum_characters
            ),
            overlap_characters=(
                config.overlap_characters
            ),
        )

    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n\n".join(
            current
            + [
                paragraph
            ]
        )

        if (
            current
            and len(candidate)
            > config.maximum_characters
        ):
            chunks.append(
                "\n\n".join(
                    current
                )
            )

            current = [
                paragraph
            ]

        else:
            current.append(
                paragraph
            )

    if current:
        chunks.append(
            "\n\n".join(
                current
            )
        )

    output: list[str] = []

    for chunk in chunks:
        if (
            len(chunk)
            <= config.maximum_characters
        ):
            output.append(
                chunk
            )

        else:
            output.extend(
                split_character_windows(
                    chunk,
                    maximum_characters=(
                        config.maximum_characters
                    ),
                    overlap_characters=(
                        config.overlap_characters
                    ),
                )
            )

    return output


def split_code_section(
    section: ParsedSection,
    config: ChunkingConfig,
) -> list[
    tuple[
        str,
        int | None,
        int | None,
    ]
]:
    """
    Split source code while preserving calculated physical lines.

    Returns:

        content
        start line
        end line
    """

    if (
        len(
            section.content
        )
        <= config.maximum_characters
    ):
        return [
            (
                section.content.strip(),
                section.line_number_start,
                section.line_number_end,
            )
        ]

    source_lines = (
        section.content
        .splitlines()
    )

    if not source_lines:
        return []

    output: list[
        tuple[
            str,
            int | None,
            int | None,
        ]
    ] = []

    current_lines: list[str] = []
    current_length = 0
    chunk_start_offset = 0

    for line_offset, line in enumerate(
        source_lines
    ):
        added_length = (
            len(line)
            + 1
        )

        if (
            current_lines
            and current_length
            + added_length
            > config.maximum_characters
        ):
            chunk_content = "\n".join(
                current_lines
            ).strip()

            start_line = (
                section.line_number_start
                + chunk_start_offset
                if (
                    section.line_number_start
                    is not None
                )
                else None
            )

            end_line = (
                start_line
                + len(
                    current_lines
                )
                - 1
                if start_line is not None
                else None
            )

            output.append(
                (
                    chunk_content,
                    start_line,
                    end_line,
                )
            )

            current_lines = []
            current_length = 0
            chunk_start_offset = (
                line_offset
            )

        current_lines.append(
            line
        )

        current_length += (
            added_length
        )

    if current_lines:
        chunk_content = "\n".join(
            current_lines
        ).strip()

        start_line = (
            section.line_number_start
            + chunk_start_offset
            if (
                section.line_number_start
                is not None
            )
            else None
        )

        end_line = (
            start_line
            + len(
                current_lines
            )
            - 1
            if start_line is not None
            else None
        )

        output.append(
            (
                chunk_content,
                start_line,
                end_line,
            )
        )

    return output


def merge_identifiers(
    original: tuple[str, ...],
    extracted: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge structural and local IDs without duplicates."""

    return ordered_unique(
        list(original)
        + list(extracted)
    )


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    config: ChunkingConfig
    | None = None,
) -> tuple[
    IngestionChunk,
    ...
]:
    """Convert one parsed document into embedding-ready chunks."""

    active_config = (
        config
        if config is not None
        else ChunkingConfig()
    )

    is_code = bool(
        document.document_type
        == "code"
    )

    pending_chunks: list[
        tuple[
            ParsedSection,
            str,
            int | None,
            int | None,
        ]
    ] = []

    for section in document.sections:
        if is_code:
            for (
                split_content,
                start_line,
                end_line,
            ) in split_code_section(
                section,
                active_config,
            ):
                pending_chunks.append(
                    (
                        section,
                        split_content,
                        start_line,
                        end_line,
                    )
                )

        else:
            for split_content in (
                split_document_section(
                    section,
                    active_config,
                )
            ):
                pending_chunks.append(
                    (
                        section,
                        split_content,
                        (
                            section
                            .line_number_start
                        ),
                        (
                            section
                            .line_number_end
                        ),
                    )
                )

    chunks: list[
        IngestionChunk
    ] = []

    for (
        section,
        chunk_content,
        start_line,
        end_line,
    ) in pending_chunks:
        if (
            len(
                chunk_content.strip()
            )
            < active_config.minimum_characters
            and len(pending_chunks) > 1
        ):
            # Do not discard explicit engineering evidence solely due
            # to length.
            (
                local_requirements,
                local_observations,
                local_tests,
            ) = extract_engineering_ids(
                chunk_content
            )

            if not any(
                (
                    local_requirements,
                    local_observations,
                    local_tests,
                )
            ):
                continue

        (
            local_requirements,
            local_observations,
            local_tests,
        ) = extract_engineering_ids(
            chunk_content
        )

        result_status = (
            extract_result_status(
                chunk_content
            )
            or section.result_status
        )

        digest = content_hash(
            chunk_content
        )

        chunk_index = len(
            chunks
        )

        chunks.append(
            IngestionChunk(
                chunk_id=build_chunk_id(
                    document_id=(
                        document.document_id
                    ),
                    file_hash=(
                        document.file_hash
                    ),
                    chunk_index=(
                        chunk_index
                    ),
                    content_digest=(
                        digest
                    ),
                ),
                document_id=(
                    document.document_id
                ),
                document_name=(
                    document.document_name
                ),
                relative_path=(
                    document.relative_path
                ),
                document_type=(
                    document.document_type
                ),
                document_category=(
                    document.document_category
                ),
                file_hash=(
                    document.file_hash
                ),
                content=(
                    chunk_content.strip()
                ),
                content_hash=(
                    digest
                ),
                chunk_index=(
                    chunk_index
                ),
                section_name=(
                    section.section_name
                ),
                page_number=(
                    section.page_number
                ),
                line_number_start=(
                    start_line
                ),
                line_number_end=(
                    end_line
                ),
                function_name=(
                    section.function_name
                ),
                class_name=(
                    section.class_name
                ),
                requirement_ids=(
                    merge_identifiers(
                        section.requirement_ids,
                        local_requirements,
                    )
                ),
                observation_ids=(
                    merge_identifiers(
                        section.observation_ids,
                        local_observations,
                    )
                ),
                test_case_ids=(
                    merge_identifiers(
                        section.test_case_ids,
                        local_tests,
                    )
                ),
                result_status=(
                    result_status
                ),
                parsing_method=(
                    document.parsing_method
                ),
            )
        )

    if not chunks:
        raise ValueError(
            "No embedding-ready chunks were produced for "
            f"{document.document_name}."
        )

    return tuple(
        chunks
    )