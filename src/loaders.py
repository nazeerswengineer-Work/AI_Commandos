"""
Low-level PDF loading for REQTRACE AI.

This module opens PDFs safely and extracts page text while preserving:

- original document name
- source path
- one-based page number
- PDF metadata
- empty-page information

This module does not interpret requirements or create evidence records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pymupdf

from src.normalization import normalize_multiline_content


class PDFLoadError(RuntimeError):
    """Raised when a PDF cannot be loaded reliably."""


class UnsupportedPDFError(PDFLoadError):
    """Raised when the supplied path is not a supported PDF file."""


class EncryptedPDFError(PDFLoadError):
    """Raised when an encrypted PDF cannot be read."""


@dataclass(frozen=True)
class LoadedPDFPage:
    """Extracted content and source information for one PDF page."""

    document_name: str
    file_path: Path
    page_number: int
    text: str
    is_empty: bool
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("PDF page numbers must be one-based.")

        if not self.document_name.strip():
            raise ValueError("document_name must not be empty.")

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "PDF page width and height must be positive."
            )


@dataclass(frozen=True)
class LoadedPDF:
    """One safely loaded PDF and its extracted pages."""

    document_name: str
    file_path: Path
    page_count: int
    pages: tuple[LoadedPDFPage, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.page_count < 1:
            raise ValueError("A loaded PDF must contain at least one page.")

        if self.page_count != len(self.pages):
            raise ValueError(
                "page_count must match the number of extracted pages."
            )

        expected_page_numbers = tuple(
            range(1, self.page_count + 1)
        )
        actual_page_numbers = tuple(
            page.page_number for page in self.pages
        )

        if actual_page_numbers != expected_page_numbers:
            raise ValueError(
                "Loaded PDF pages must be ordered and one-based."
            )

    @property
    def empty_page_numbers(self) -> tuple[int, ...]:
        """Return one-based numbers of pages without extractable text."""

        return tuple(
            page.page_number
            for page in self.pages
            if page.is_empty
        )

    @property
    def non_empty_pages(self) -> tuple[LoadedPDFPage, ...]:
        """Return pages containing extractable text."""

        return tuple(
            page for page in self.pages if not page.is_empty
        )


def validate_pdf_path(path: Path) -> Path:
    """Validate and resolve a PDF input path."""

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise UnsupportedPDFError(
            f"PDF path is not a file: {resolved_path}"
        )

    if resolved_path.suffix.lower() != ".pdf":
        raise UnsupportedPDFError(
            f"Unsupported file type '{resolved_path.suffix}'. "
            "Expected a .pdf file."
        )

    return resolved_path


def clean_pdf_metadata(
    raw_metadata: Mapping[str, object] | None,
) -> dict[str, str]:
    """Keep only non-empty string PDF metadata values."""

    if not raw_metadata:
        return {}

    cleaned: dict[str, str] = {}

    for key, value in raw_metadata.items():
        if value is None:
            continue

        key_text = str(key).strip()
        value_text = str(value).strip()

        if key_text and value_text:
            cleaned[key_text] = value_text

    return cleaned


def extract_page_text(
    page: pymupdf.Page,
    *,
    sort_text: bool = True,
) -> str:
    """
    Extract normalized searchable text from one PDF page.

    sort=True requests top-left to bottom-right ordering from PyMuPDF.
    Exact whitespace and visual line wrapping are not treated as evidence.
    """

    raw_text = page.get_text(
        "text",
        sort=sort_text,
    )

    return normalize_multiline_content(raw_text)


def load_pdf(
    path: str | Path,
    *,
    sort_text: bool = True,
) -> LoadedPDF:
    """
    Load one PDF and extract all pages.

    Empty pages are preserved as page records instead of being silently
    removed. This ensures page-number traceability remains correct.
    """

    pdf_path = validate_pdf_path(Path(path))

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise PDFLoadError(
            f"Unable to open PDF '{pdf_path.name}': {exc}"
        ) from exc

    try:
        if document.needs_pass:
            raise EncryptedPDFError(
                f"PDF '{pdf_path.name}' requires a password."
            )

        if len(document) < 1:
            raise PDFLoadError(
                f"PDF '{pdf_path.name}' contains no pages."
            )

        pages: list[LoadedPDFPage] = []

        for page_index in range(len(document)):
            page = document.load_page(page_index)
            page_text = extract_page_text(
                page,
                sort_text=sort_text,
            )

            page_rectangle = page.rect

            pages.append(
                LoadedPDFPage(
                    document_name=pdf_path.name,
                    file_path=pdf_path,
                    page_number=page_index + 1,
                    text=page_text,
                    is_empty=not bool(page_text.strip()),
                    width=float(page_rectangle.width),
                    height=float(page_rectangle.height),
                )
            )

        return LoadedPDF(
            document_name=pdf_path.name,
            file_path=pdf_path,
            page_count=len(document),
            pages=tuple(pages),
            metadata=clean_pdf_metadata(document.metadata),
        )
    except PDFLoadError:
        raise
    except Exception as exc:
        raise PDFLoadError(
            f"Failed while reading PDF '{pdf_path.name}': {exc}"
        ) from exc
    finally:
        document.close()


def load_pdf_directory(
    directory: str | Path,
    *,
    sort_text: bool = True,
) -> tuple[LoadedPDF, ...]:
    """
    Load all PDFs from one directory in deterministic filename order.

    This function raises an error for a failed PDF rather than silently
    omitting engineering evidence.
    """

    directory_path = Path(directory).expanduser().resolve()

    if not directory_path.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"PDF directory is not a directory: {directory_path}"
        )

    pdf_paths = sorted(
        (
            path
            for path in directory_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda item: item.name.lower(),
    )

    return tuple(
        load_pdf(
            path,
            sort_text=sort_text,
        )
        for path in pdf_paths
    )