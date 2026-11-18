"""
Tests for dynamic document parsing.
"""

from __future__ import annotations

from src.document_discovery import (
    discover_document,
)

from src.dynamic_document_parser import (
    parse_discovered_document,
)


def discovered(
    path,
    root,
):
    """Build discovered-document fixture."""

    return discover_document(
        path=path,
        documents_root=root,
    )


def test_markdown_sections_and_ids(
    tmp_path,
) -> None:
    """Markdown headings and IDs should survive parsing."""

    root = tmp_path / "documents"
    root.mkdir()

    source = root / "FRD.md"

    source.write_text(
        (
            "# Input Validation\n"
            "REQ-075 requires null-input handling.\n\n"
            "# Verification\n"
            "Verification uses TC-501.\n"
        ),
        encoding="utf-8",
    )

    parsed = parse_discovered_document(
        discovered(
            source,
            root,
        )
    )

    assert (
        parsed.document_type
        == "frd"
    )

    assert len(
        parsed.sections
    ) == 2

    assert (
        parsed.sections[0]
        .requirement_ids
        == (
            "REQ-075",
        )
    )

    assert (
        parsed.sections[1]
        .test_case_ids
        == (
            "TC-501",
        )
    )


def test_c_function_provenance(
    tmp_path,
) -> None:
    """C functions should preserve names and physical lines."""

    root = tmp_path / "documents"
    root.mkdir()

    source = root / "processor.c"

    source.write_text(
        (
            "#include <stddef.h>\n"
            "\n"
            "int process_data(int *value)\n"
            "{\n"
            "    if (value == NULL)\n"
            "    {\n"
            "        return -1;\n"
            "    }\n"
            "\n"
            "    return *value;\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    parsed = parse_discovered_document(
        discovered(
            source,
            root,
        )
    )

    functions = [
        section
        for section in parsed.sections
        if section.function_name
    ]

    assert len(
        functions
    ) == 1

    function = functions[0]

    assert (
        function.function_name
        == "process_data"
    )

    assert (
        function.line_number_start
        == 3
    )

    assert (
        function.line_number_end
        == 11
    )


def test_python_definition_provenance(
    tmp_path,
) -> None:
    """Python definitions should preserve source lines."""

    root = tmp_path / "documents"
    root.mkdir()

    source = root / "validator.py"

    source.write_text(
        (
            "VALUE = 1\n"
            "\n"
            "def validate_command(command):\n"
            "    if command < 0:\n"
            "        return False\n"
            "    return True\n"
        ),
        encoding="utf-8",
    )

    parsed = parse_discovered_document(
        discovered(
            source,
            root,
        )
    )

    functions = [
        section
        for section in parsed.sections
        if section.function_name
    ]

    assert (
        functions[0].function_name
        == "validate_command"
    )

    assert (
        functions[0].line_number_start
        == 3
    )