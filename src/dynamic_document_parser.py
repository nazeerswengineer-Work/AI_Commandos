"""
Dynamic document parser for REQTRACE AI.

Supported formats:

- PDF
- DOCX
- TXT
- MD
- C
- C++
- Header files
- Python

This phase extracts normalized, physically traceable sections.

It does not generate embeddings or update ChromaDB.
"""

from __future__ import annotations

import re

from pathlib import Path

from src.ingestion_content import (
    ParsedDocument,
    ParsedSection,
)

from src.ingestion_metadata import (
    detect_section_heading,
    extract_engineering_ids,
    extract_result_status,
    infer_document_type,
)

from src.ingestion_models import (
    DiscoveredDocument,
)


class DynamicDocumentParserError(
    RuntimeError
):
    """Raised when a source document cannot be parsed."""


CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
}


C_FUNCTION_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?!(?:if|for|while|switch|return|sizeof)\b)"
    r"(?:[A-Za-z_][\w\s\*:&<>,\[\]]*?)"
    r"\b(?P<name>[A-Za-z_]\w*)"
    r"\s*\([^;{}]*\)"
    r"\s*(?:const\s*)?"
    r"\{",
    re.MULTILINE,
)


PYTHON_DEFINITION_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<kind>async\s+def|def|class)"
    r"\s+"
    r"(?P<name>[A-Za-z_]\w*)"
    r"\b",
    re.MULTILINE,
)


def build_section(
    *,
    content: str,
    section_index: int,
    section_name: str | None = None,
    page_number: int | None = None,
    line_number_start: int | None = None,
    line_number_end: int | None = None,
    function_name: str | None = None,
    class_name: str | None = None,
) -> ParsedSection:
    """Build one section with explicit extracted metadata."""

    (
        requirement_ids,
        observation_ids,
        test_case_ids,
    ) = extract_engineering_ids(
        content
    )

    return ParsedSection(
        content=content.strip(),
        section_index=section_index,
        section_name=section_name,
        page_number=page_number,
        line_number_start=(
            line_number_start
        ),
        line_number_end=(
            line_number_end
        ),
        function_name=function_name,
        class_name=class_name,
        requirement_ids=(
            requirement_ids
        ),
        observation_ids=(
            observation_ids
        ),
        test_case_ids=(
            test_case_ids
        ),
        result_status=(
            extract_result_status(
                content
            )
        ),
    )


def parse_pdf(
    document: DiscoveredDocument,
) -> tuple[ParsedSection, ...]:
    """Extract one section per non-empty PDF page."""

    try:
        from pypdf import PdfReader

    except ImportError as exc:
        raise DynamicDocumentParserError(
            "PDF parsing requires the installed "
            "'pypdf' package."
        ) from exc

    try:
        reader = PdfReader(
            str(
                document.absolute_path
            )
        )

    except Exception as exc:
        raise DynamicDocumentParserError(
            "Unable to open PDF document: "
            f"{document.document_name}"
        ) from exc

    sections: list[
        ParsedSection
    ] = []

    for page_index, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            text = (
                page.extract_text()
                or ""
            )

        except Exception as exc:
            raise DynamicDocumentParserError(
                "Unable to extract PDF page "
                f"{page_index} from "
                f"{document.document_name}."
            ) from exc

        if not text.strip():
            continue

        sections.append(
            build_section(
                content=text,
                section_index=len(
                    sections
                ),
                section_name=(
                    f"Page {page_index}"
                ),
                page_number=(
                    page_index
                ),
            )
        )

    if not sections:
        raise DynamicDocumentParserError(
            "No extractable text was found in PDF: "
            f"{document.document_name}. "
            "OCR is not automatically applied."
        )

    return tuple(
        sections
    )


def parse_docx(
    document: DiscoveredDocument,
) -> tuple[ParsedSection, ...]:
    """
    Extract DOCX paragraphs grouped by heading boundaries.

    Word page numbers are not fabricated because DOCX does not retain
    stable page pagination independently of rendering.
    """

    try:
        from docx import Document

    except ImportError as exc:
        raise DynamicDocumentParserError(
            "DOCX parsing requires the installed "
            "'python-docx' package."
        ) from exc

    try:
        source = Document(
            str(
                document.absolute_path
            )
        )

    except Exception as exc:
        raise DynamicDocumentParserError(
            "Unable to open DOCX document: "
            f"{document.document_name}"
        ) from exc

    sections: list[
        ParsedSection
    ] = []

    current_heading: (
        str
        | None
    ) = None

    current_paragraphs: list[
        str
    ] = []

    def flush() -> None:
        nonlocal current_paragraphs

        content = "\n\n".join(
            current_paragraphs
        ).strip()

        if content:
            sections.append(
                build_section(
                    content=content,
                    section_index=len(
                        sections
                    ),
                    section_name=(
                        current_heading
                    ),
                )
            )

        current_paragraphs = []

    for paragraph in source.paragraphs:
        text = paragraph.text.strip()

        if not text:
            continue

        style_name = (
            paragraph.style.name
            if paragraph.style
            else ""
        )

        is_heading = (
            style_name
            .lower()
            .startswith(
                "heading"
            )
        )

        if is_heading:
            flush()

            current_heading = text
            continue

        current_paragraphs.append(
            text
        )

    flush()

    if not sections:
        raise DynamicDocumentParserError(
            "No extractable paragraphs were "
            f"found in DOCX: {document.document_name}"
        )

    return tuple(
        sections
    )


def parse_text_document(
    document: DiscoveredDocument,
) -> tuple[ParsedSection, ...]:
    """Parse TXT or Markdown using explicit heading boundaries."""

    try:
        content = (
            document.absolute_path
            .read_text(
                encoding="utf-8"
            )
        )

    except UnicodeDecodeError:
        try:
            content = (
                document.absolute_path
                .read_text(
                    encoding="utf-8-sig"
                )
            )

        except UnicodeDecodeError as exc:
            raise DynamicDocumentParserError(
                "Text document is not valid UTF-8: "
                f"{document.document_name}"
            ) from exc

    if not content.strip():
        raise DynamicDocumentParserError(
            "Text document is empty: "
            f"{document.document_name}"
        )

    markdown = bool(
        document.extension == ".md"
    )

    sections: list[
        ParsedSection
    ] = []

    current_heading: (
        str
        | None
    ) = None

    current_lines: list[str] = []

    section_start_line = 1

    def flush(
        end_line: int,
    ) -> None:
        nonlocal current_lines
        nonlocal section_start_line

        section_content = "\n".join(
            current_lines
        ).strip()

        if section_content:
            sections.append(
                build_section(
                    content=(
                        section_content
                    ),
                    section_index=len(
                        sections
                    ),
                    section_name=(
                        current_heading
                    ),
                    line_number_start=(
                        section_start_line
                    ),
                    line_number_end=(
                        end_line
                    ),
                )
            )

        current_lines = []

    lines = content.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        heading = detect_section_heading(
            line,
            markdown=markdown,
        )

        if heading is not None:
            flush(
                line_number - 1
            )

            current_heading = heading
            section_start_line = (
                line_number
            )

            current_lines.append(
                line
            )

            continue

        if not current_lines:
            section_start_line = (
                line_number
            )

        current_lines.append(
            line
        )

    flush(
        len(lines)
    )

    if not sections:
        raise DynamicDocumentParserError(
            "No text sections were extracted from: "
            f"{document.document_name}"
        )

    return tuple(
        sections
    )


def find_matching_brace_line(
    lines: list[str],
    start_index: int,
) -> int:
    """
    Locate the physical end line of a C/C++ declaration block.

    This is a lightweight structural parser, not a compiler frontend.
    """

    depth = 0
    opened = False

    in_block_comment = False

    for index in range(
        start_index,
        len(lines),
    ):
        line = lines[index]

        position = 0

        while position < len(line):
            character = line[
                position
            ]

            next_character = (
                line[position + 1]
                if (
                    position + 1
                    < len(line)
                )
                else ""
            )

            if in_block_comment:
                if (
                    character == "*"
                    and next_character == "/"
                ):
                    in_block_comment = False
                    position += 2
                    continue

                position += 1
                continue

            if (
                character == "/"
                and next_character == "*"
            ):
                in_block_comment = True
                position += 2
                continue

            if (
                character == "/"
                and next_character == "/"
            ):
                break

            if character == "{":
                opened = True
                depth += 1

            elif character == "}":
                depth -= 1

                if (
                    opened
                    and depth == 0
                ):
                    return index

            position += 1

    return len(
        lines
    ) - 1


def parse_c_family_code(
    document: DiscoveredDocument,
) -> tuple[ParsedSection, ...]:
    """Parse C/C++ functions using physical source lines."""

    try:
        content = (
            document.absolute_path
            .read_text(
                encoding="utf-8"
            )
        )

    except UnicodeDecodeError as exc:
        raise DynamicDocumentParserError(
            "Source code is not valid UTF-8: "
            f"{document.document_name}"
        ) from exc

    lines = content.splitlines()

    matches = list(
        C_FUNCTION_PATTERN.finditer(
            content
        )
    )

    sections: list[
        ParsedSection
    ] = []

    occupied_lines: set[int] = set()

    for match in matches:
        start_line = (
            content.count(
                "\n",
                0,
                match.start(),
            )
            + 1
        )

        start_index = (
            start_line - 1
        )

        end_index = find_matching_brace_line(
            lines,
            start_index,
        )

        end_line = (
            end_index + 1
        )

        function_content = "\n".join(
            lines[
                start_index:
                end_index + 1
            ]
        )

        if not function_content.strip():
            continue

        occupied_lines.update(
            range(
                start_line,
                end_line + 1,
            )
        )

        sections.append(
            build_section(
                content=(
                    function_content
                ),
                section_index=len(
                    sections
                ),
                section_name=(
                    match.group(
                        "name"
                    )
                ),
                line_number_start=(
                    start_line
                ),
                line_number_end=(
                    end_line
                ),
                function_name=(
                    match.group(
                        "name"
                    )
                ),
            )
        )

    remaining_lines = [
        (
            line_number,
            line,
        )
        for line_number, line
        in enumerate(
            lines,
            start=1,
        )
        if (
            line_number
            not in occupied_lines
            and line.strip()
        )
    ]

    if remaining_lines:
        remaining_content = "\n".join(
            line
            for _, line
            in remaining_lines
        )

        sections.insert(
            0,
            build_section(
                content=(
                    remaining_content
                ),
                section_index=0,
                section_name=(
                    "File declarations"
                ),
                line_number_start=(
                    remaining_lines[0][0]
                ),
                line_number_end=(
                    remaining_lines[-1][0]
                ),
            ),
        )

        sections = [
            ParsedSection(
                content=section.content,
                section_index=index,
                section_name=(
                    section.section_name
                ),
                page_number=(
                    section.page_number
                ),
                line_number_start=(
                    section.line_number_start
                ),
                line_number_end=(
                    section.line_number_end
                ),
                function_name=(
                    section.function_name
                ),
                class_name=(
                    section.class_name
                ),
                requirement_ids=(
                    section.requirement_ids
                ),
                observation_ids=(
                    section.observation_ids
                ),
                test_case_ids=(
                    section.test_case_ids
                ),
                result_status=(
                    section.result_status
                ),
            )
            for index, section
            in enumerate(
                sections
            )
        ]

    if not sections:
        raise DynamicDocumentParserError(
            "No source-code content was extracted from: "
            f"{document.document_name}"
        )

    return tuple(
        sections
    )


def parse_python_code(
    document: DiscoveredDocument,
) -> tuple[ParsedSection, ...]:
    """Parse Python classes and functions with physical line ranges."""

    try:
        content = (
            document.absolute_path
            .read_text(
                encoding="utf-8"
            )
        )

    except UnicodeDecodeError as exc:
        raise DynamicDocumentParserError(
            "Python source is not valid UTF-8: "
            f"{document.document_name}"
        ) from exc

    lines = content.splitlines()

    matches = list(
        PYTHON_DEFINITION_PATTERN.finditer(
            content
        )
    )

    sections: list[
        ParsedSection
    ] = []

    definitions: list[
        tuple[
            int,
            str,
            str,
            int,
        ]
    ] = []

    for match in matches:
        line_number = (
            content.count(
                "\n",
                0,
                match.start(),
            )
            + 1
        )

        indent_size = len(
            match.group(
                "indent"
            ).replace(
                "\t",
                "    ",
            )
        )

        definitions.append(
            (
                line_number,
                match.group(
                    "kind"
                ),
                match.group(
                    "name"
                ),
                indent_size,
            )
        )

    if definitions:
        first_definition_line = (
            definitions[0][0]
        )

        module_content = "\n".join(
            lines[
                0:
                first_definition_line - 1
            ]
        ).strip()

        if module_content:
            sections.append(
                build_section(
                    content=module_content,
                    section_index=0,
                    section_name=(
                        "Module declarations"
                    ),
                    line_number_start=1,
                    line_number_end=(
                        first_definition_line - 1
                    ),
                )
            )

    for definition_index, (
        start_line,
        kind,
        name,
        indent_size,
    ) in enumerate(
        definitions
    ):
        end_line = len(
            lines
        )

        for next_definition in definitions[
            definition_index + 1:
        ]:
            next_line = (
                next_definition[0]
            )

            next_indent = (
                next_definition[3]
            )

            if next_indent <= indent_size:
                end_line = (
                    next_line - 1
                )
                break

        section_content = "\n".join(
            lines[
                start_line - 1:
                end_line
            ]
        ).rstrip()

        if not section_content:
            continue

        is_class = bool(
            kind == "class"
        )

        sections.append(
            build_section(
                content=(
                    section_content
                ),
                section_index=len(
                    sections
                ),
                section_name=name,
                line_number_start=(
                    start_line
                ),
                line_number_end=(
                    end_line
                ),
                function_name=(
                    None
                    if is_class
                    else name
                ),
                class_name=(
                    name
                    if is_class
                    else None
                ),
            )
        )

    if not sections:
        if not content.strip():
            raise DynamicDocumentParserError(
                "Python source file is empty: "
                f"{document.document_name}"
            )

        sections.append(
            build_section(
                content=content,
                section_index=0,
                section_name=(
                    "Module"
                ),
                line_number_start=1,
                line_number_end=max(
                    1,
                    len(lines),
                ),
            )
        )

    return tuple(
        sections
    )


def parse_discovered_document(
    document: DiscoveredDocument,
) -> ParsedDocument:
    """Parse one discovered document using its file type."""

    extension = (
        document.extension.lower()
    )

    if extension == ".pdf":
        sections = parse_pdf(
            document
        )

        parsing_method = (
            "pdf_page_text"
        )

    elif extension == ".docx":
        sections = parse_docx(
            document
        )

        parsing_method = (
            "docx_heading_paragraph"
        )

    elif extension in {
        ".txt",
        ".md",
    }:
        sections = (
            parse_text_document(
                document
            )
        )

        parsing_method = (
            "markdown_heading"
            if extension == ".md"
            else "text_heading"
        )

    elif extension == ".py":
        sections = parse_python_code(
            document
        )

        parsing_method = (
            "python_definition"
        )

    elif extension in {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
    }:
        sections = (
            parse_c_family_code(
                document
            )
        )

        parsing_method = (
            "c_family_function"
        )

    else:
        raise DynamicDocumentParserError(
            "Unsupported source-file extension: "
            f"{extension}"
        )

    sample = "\n".join(
        section.content
        for section
        in sections[:3]
    )

    (
        document_type,
        document_category,
    ) = infer_document_type(
        relative_path=(
            document.relative_path
        ),
        extension=extension,
        content_sample=sample,
    )

    return ParsedDocument(
        document_id=(
            document.document_id
        ),
        document_name=(
            document.document_name
        ),
        relative_path=(
            document.relative_path
        ),
        extension=extension,
        file_hash=(
            document.file_hash
        ),
        document_type=(
            document_type
        ),
        document_category=(
            document_category
        ),
        parsing_method=(
            parsing_method
        ),
        sections=sections,
    )