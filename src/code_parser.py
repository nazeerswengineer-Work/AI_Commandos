"""
Source-code parsing for REQTRACE AI.

Phase 7 converts supported source-code files into validated EvidenceRecord
objects with one-based physical line numbers.

The parser:

- preserves the original source file
- identifies C/C++ functions
- preserves one-based physical line numbers
- extracts explicit engineering IDs from each function and nearby comments
- creates stable evidence IDs
- creates normalized content hashes
- preserves partial-evidence limitations
- does not assess compliance
- does not invoke an LLM
- does not create embeddings or ChromaDB records
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.evidence import (
    EvidenceRecord,
    build_code_evidence_id,
    calculate_content_hash,
)
from src.metadata import (
    DocumentType,
    EvidenceMetadata,
    ParsingMethod,
)
from src.normalization import (
    extract_observation_ids,
    extract_requirement_ids,
    extract_test_case_ids,
    normalize_multiline_content,
)


class CodeParsingError(RuntimeError):
    """Raised when source code cannot be parsed reliably."""


SUPPORTED_CODE_EXTENSIONS = {
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".py": "python",
}


C_FUNCTION_START_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<return_type>
        [A-Za-z_][A-Za-z0-9_]*
        (?:\s*\*)?
    )
    \s+
    (?P<name>
        [A-Za-z_][A-Za-z0-9_]*
    )
    \s*
    \(
    """,
    re.VERBOSE,
)


PYTHON_FUNCTION_PATTERN = re.compile(
    r"^(?P<indent>\s*)def\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\("
)


@dataclass(frozen=True)
class SourceFunction:
    """One source-code function with physical source provenance."""

    file_name: str
    file_path: Path
    language: str
    function_name: str
    start_line: int
    end_line: int
    content: str

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    is_partial_evidence: bool
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError(
                "Source-code start lines must be one-based."
            )

        if self.end_line < self.start_line:
            raise ValueError(
                "end_line must be greater than or equal to start_line."
            )

        if not self.function_name.strip():
            raise ValueError(
                "function_name must not be empty."
            )

        if not self.content.strip():
            raise ValueError(
                "Function content must not be empty."
            )


@dataclass(frozen=True)
class CodeParseWarning:
    """Non-fatal source parsing issue."""

    code: str
    message: str
    file_name: str
    line_number: int | None = None


@dataclass(frozen=True)
class CodeParseResult:
    """Complete parsing result for one source-code file."""

    file_name: str
    file_path: Path
    language: str
    physical_line_count: int

    functions: tuple[SourceFunction, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    warnings: tuple[CodeParseWarning, ...]

    @property
    def function_names(self) -> tuple[str, ...]:
        return tuple(
            function.function_name
            for function in self.functions
        )

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_records)


def validate_source_path(
    path: str | Path,
) -> tuple[Path, str]:
    """Validate a source-code path and return its language."""

    source_path = Path(path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source file does not exist: {source_path}"
        )

    if not source_path.is_file():
        raise CodeParsingError(
            f"Source path is not a file: {source_path}"
        )

    language = SUPPORTED_CODE_EXTENSIONS.get(
        source_path.suffix.lower()
    )

    if language is None:
        raise CodeParsingError(
            f"Unsupported source extension "
            f"'{source_path.suffix}'."
        )

    return source_path, language


def load_source_lines(
    path: Path,
) -> tuple[str, ...]:
    """Load source lines while preserving physical line positions."""

    try:
        content = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        try:
            content = path.read_text(
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError as exc:
            raise CodeParsingError(
                f"Unable to decode source file '{path.name}'."
            ) from exc
    except OSError as exc:
        raise CodeParsingError(
            f"Unable to read source file '{path.name}': {exc}"
        ) from exc

    normalized = (
        content
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    return tuple(normalized.splitlines())


def count_braces_outside_strings(
    line: str,
) -> tuple[int, int]:
    """
    Count opening and closing braces while ignoring quoted strings.

    This is a lightweight structural scanner, not a complete C parser.
    """

    opening = 0
    closing = 0

    in_single_quote = False
    in_double_quote = False
    escaped = False

    for character in line:
        if escaped:
            escaped = False
            continue

        if character == "\\":
            escaped = True
            continue

        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if in_single_quote or in_double_quote:
            continue

        if character == "{":
            opening += 1
        elif character == "}":
            closing += 1

    return opening, closing


def collect_preceding_comment_start(
    lines: tuple[str, ...],
    function_start_index: int,
) -> int:
    """
    Include a contiguous comment block immediately preceding a function.

    The returned value is a zero-based line index.
    """

    index = function_start_index - 1

    while index >= 0 and not lines[index].strip():
        index -= 1

    if index < 0:
        return function_start_index

    stripped = lines[index].strip()

    if stripped.startswith("//"):
        while (
            index >= 0
            and lines[index].strip().startswith("//")
        ):
            index -= 1

        return index + 1

    if stripped.endswith("*/"):
        while index >= 0:
            if "/*" in lines[index]:
                return index

            index -= 1

    return function_start_index


def extract_code_limitations(
    content: str,
) -> tuple[str, ...]:
    """Extract explicit evidence limitations from source comments."""

    limitations: list[str] = []

    normalized = " ".join(
        content.split()
    ).lower()

    limitation_phrases = {
        "partial source code": (
            "The supplied source code is explicitly marked as partial."
        ),
        "intentionally omits": (
            "The source evidence explicitly omits implementation content."
        ),
        "not included": (
            "Referenced implementation content is not included."
        ),
        "not supplied": (
            "Referenced implementation evidence is not supplied."
        ),
        "no corrected implementation": (
            "No corrected implementation is supplied."
        ),
    }

    for phrase, message in limitation_phrases.items():
        if phrase in normalized:
            limitations.append(message)

    return tuple(dict.fromkeys(limitations))

def extract_file_header_text(
    lines: tuple[str, ...],
) -> str:
    """
    Extract the leading source-file comment area.

    The scan stops at the first substantive source declaration.
    Include directives may appear between header comments and declarations.
    """

    header_lines: list[str] = []
    in_block_comment = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if header_lines:
                header_lines.append(line)

            continue

        if stripped.startswith("/*"):
            in_block_comment = True
            header_lines.append(line)

            if "*/" in stripped:
                in_block_comment = False

            continue

        if in_block_comment:
            header_lines.append(line)

            if "*/" in stripped:
                in_block_comment = False

            continue

        if stripped.startswith("//"):
            header_lines.append(line)
            continue

        if stripped.startswith("#"):
            continue

        break

    return normalize_multiline_content(
        "\n".join(header_lines)
    )

def extract_file_level_traceability(
    lines: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Extract explicit engineering IDs from the source-file header."""

    header_text = extract_file_header_text(
        lines
    )

    return (
        extract_requirement_ids(header_text),
        extract_observation_ids(header_text),
        extract_test_case_ids(header_text),
    )

def merge_engineering_ids(
    file_level_ids: tuple[str, ...],
    function_level_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Merge IDs while preserving source order and removing duplicates.
    """

    merged: list[str] = []
    seen: set[str] = set()

    for engineering_id in (
        *file_level_ids,
        *function_level_ids,
    ):
        if engineering_id not in seen:
            seen.add(engineering_id)
            merged.append(engineering_id)

    return tuple(merged)

def create_source_function(
    *,
    path: Path,
    language: str,
    function_name: str,
    start_index: int,
    end_index: int,
    lines: tuple[str, ...],
    file_requirement_ids: tuple[str, ...] = (),
    file_observation_ids: tuple[str, ...] = (),
    file_test_case_ids: tuple[str, ...] = (),
    file_limitations: tuple[str, ...] = (),
) -> SourceFunction:
    """Create one validated source-function model."""

    content = normalize_multiline_content(
        "\n".join(
            lines[start_index:end_index + 1]
        )
    )

    function_limitations = extract_code_limitations(
        content
    )

    limitations = tuple(
        dict.fromkeys(
            (
                *file_limitations,
                *function_limitations,
            )
        )
    )

    function_requirement_ids = extract_requirement_ids(
        content
    )

    function_observation_ids = extract_observation_ids(
        content
    )

    function_test_case_ids = extract_test_case_ids(
        content
    )

    return SourceFunction(
        file_name=path.name,
        file_path=path,
        language=language,
        function_name=function_name,
        start_line=start_index + 1,
        end_line=end_index + 1,
        content=content,
        requirement_ids=merge_engineering_ids(
            file_requirement_ids,
            function_requirement_ids,
        ),
        observation_ids=merge_engineering_ids(
            file_observation_ids,
            function_observation_ids,
        ),
        test_case_ids=merge_engineering_ids(
            file_test_case_ids,
            function_test_case_ids,
        ),
        is_partial_evidence=bool(limitations),
        limitations=limitations,
    )

def parse_c_functions(
    path: Path,
    language: str,
    lines: tuple[str, ...],
) -> tuple[SourceFunction, ...]:
    """
    Parse C/C++ functions using signature and brace-aware scanning.

    Supports styles such as:

        int function(int value)
        {
            ...
        }

    and:

        int function(
            int value,
            bool enabled)
        {
            ...
        }
    """

    functions: list[SourceFunction] = []

    (
        file_requirement_ids,
        file_observation_ids,
        file_test_case_ids,
    ) = extract_file_level_traceability(
        lines
    )

    file_header_text = extract_file_header_text(
        lines
    )

    file_limitations = extract_code_limitations(
        file_header_text
    )

    line_index = 0

    while line_index < len(lines):
        start_line_text = lines[line_index]

        match = C_FUNCTION_START_PATTERN.match(
            start_line_text
        )

        if match is None:
            line_index += 1
            continue

        function_name = match.group("name")

        # --------------------------------------------------
        # Locate the opening brace of the function body.
        # --------------------------------------------------

        signature_end_index = line_index
        opening_brace_found = False
        parenthesis_depth = 0
        signature_started = False

        for candidate_index in range(
            line_index,
            len(lines),
        ):
            candidate_line = lines[candidate_index]

            for character in candidate_line:
                if character == "(":
                    parenthesis_depth += 1
                    signature_started = True

                elif character == ")":
                    if parenthesis_depth > 0:
                        parenthesis_depth -= 1

                elif (
                    character == "{"
                    and signature_started
                    and parenthesis_depth == 0
                ):
                    signature_end_index = candidate_index
                    opening_brace_found = True
                    break

                elif (
                    character == ";"
                    and signature_started
                    and parenthesis_depth == 0
                ):
                    # Function prototype, not a definition.
                    break

            if opening_brace_found:
                break

            stripped_candidate = candidate_line.strip()

            if (
                ";" in stripped_candidate
                and parenthesis_depth == 0
            ):
                break

            # Avoid accidentally consuming unrelated code
            # when the candidate was not actually a
            # function definition.
            if candidate_index - line_index > 20:
                break

        if not opening_brace_found:
            line_index += 1
            continue

        # --------------------------------------------------
        # Include the comment block immediately before the
        # function. This preserves traceability comments.
        # --------------------------------------------------

        comment_start = collect_preceding_comment_start(
            lines,
            line_index,
        )

        # --------------------------------------------------
        # Count braces from the line containing the opening
        # function brace until its matching closing brace.
        # --------------------------------------------------

        brace_depth = 0
        body_started = False
        end_index = signature_end_index

        for candidate_index in range(
            signature_end_index,
            len(lines),
        ):
            opening_count, closing_count = (
                count_braces_outside_strings(
                    lines[candidate_index]
                )
            )

            if opening_count > 0:
                body_started = True

            if body_started:
                brace_depth += opening_count
                brace_depth -= closing_count

            if body_started and brace_depth == 0:
                end_index = candidate_index
                break
        else:
            raise CodeParsingError(
                f"Function '{function_name}' in "
                f"'{path.name}' has unbalanced braces."
            )

        functions.append(
            create_source_function(
                path=path,
                language=language,
                function_name=function_name,
                start_index=comment_start,
                end_index=end_index,
                lines=lines,
                file_requirement_ids=file_requirement_ids,
                file_observation_ids=file_observation_ids,
                file_test_case_ids=file_test_case_ids,
                file_limitations=file_limitations,
            )
        )

        line_index = end_index + 1

    return tuple(functions)

        
def parse_python_functions(
    path: Path,
    language: str,
    lines: tuple[str, ...],
) -> tuple[SourceFunction, ...]:
    """Parse top-level and nested Python function blocks."""

    functions: list[SourceFunction] = []

    starts: list[tuple[int, str, int]] = []

    for index, line in enumerate(lines):
        match = PYTHON_FUNCTION_PATTERN.match(
            line
        )

        if match is None:
            continue

        starts.append(
            (
                index,
                match.group("name"),
                len(match.group("indent")),
            )
        )

    for position, (
        start_index,
        function_name,
        indentation,
    ) in enumerate(starts):
        end_index = len(lines) - 1

        for candidate_index in range(
            start_index + 1,
            len(lines),
        ):
            candidate = lines[candidate_index]

            if not candidate.strip():
                continue

            candidate_indent = (
                len(candidate)
                - len(candidate.lstrip())
            )

            if (
                candidate_indent <= indentation
                and not candidate.lstrip().startswith(
                    ("#", "@")
                )
            ):
                end_index = candidate_index - 1
                break

        comment_start = collect_preceding_comment_start(
            lines,
            start_index,
        )

        functions.append(
            create_source_function(
                path=path,
                language=language,
                function_name=function_name,
                start_index=comment_start,
                end_index=end_index,
                lines=lines,
            )
        )

    return tuple(functions)


def make_code_chunk_id(
    function: SourceFunction,
) -> str:
    """Create a deterministic source-code chunk identifier."""

    safe_file_stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        Path(function.file_name).stem,
    ).strip("-")

    safe_function_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        function.function_name,
    ).strip("-")

    return (
        f"CODE-{safe_file_stem}"
        f"-{safe_function_name}"
        f"-{function.start_line}"
        f"-{function.end_line}"
    )


def create_code_evidence(
    function: SourceFunction,
) -> EvidenceRecord:
    """Convert one parsed source function into evidence."""

    evidence_id = build_code_evidence_id(
        file_name=function.file_name,
        function_name=function.function_name,
        start_line=function.start_line,
        end_line=function.end_line,
    )

    content_hash = calculate_content_hash(
        function.content
    )

    metadata = EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.CODE,
        document_name=function.file_name,
        requirement_id=(
            function.requirement_ids[0]
            if function.requirement_ids
            else None
        ),
        observation_id=(
            function.observation_ids[0]
            if function.observation_ids
            else None
        ),
        test_case_id=(
            function.test_case_ids[0]
            if function.test_case_ids
            else None
        ),
        file_name=function.file_name,
        file_path=str(function.file_path),
        function_name=function.function_name,
        start_line=function.start_line,
        end_line=function.end_line,
        language=function.language,
        chunk_id=make_code_chunk_id(
            function
        ),
        content_hash=content_hash,
        parsing_method=ParsingMethod.FUNCTION,
    )

    return EvidenceRecord(
        content=function.content,
        metadata=metadata,
    )


def parse_code_file(
    path: str | Path,
) -> CodeParseResult:
    """Parse one supported source-code file."""

    source_path, language = validate_source_path(
        path
    )

    lines = load_source_lines(
        source_path
    )

    if language in {"c", "cpp"}:
        functions = parse_c_functions(
            source_path,
            language,
            lines,
        )
    elif language == "python":
        functions = parse_python_functions(
            source_path,
            language,
            lines,
        )
    else:
        functions = ()

    warnings: list[CodeParseWarning] = []

    if not functions:
        warnings.append(
            CodeParseWarning(
                code="NO_FUNCTIONS_PARSED",
                message=(
                    "No supported function definition was identified."
                ),
                file_name=source_path.name,
            )
        )

    evidence_records = tuple(
        create_code_evidence(function)
        for function in functions
    )

    return CodeParseResult(
        file_name=source_path.name,
        file_path=source_path,
        language=language,
        physical_line_count=len(lines),
        functions=functions,
        evidence_records=evidence_records,
        warnings=tuple(warnings),
    )


def parse_code_directory(
    directory: str | Path,
) -> tuple[CodeParseResult, ...]:
    """Parse supported source files in deterministic order."""

    directory_path = (
        Path(directory)
        .expanduser()
        .resolve()
    )

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Code directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Code path is not a directory: {directory_path}"
        )

    source_paths = sorted(
        (
            path
            for path in directory_path.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_CODE_EXTENSIONS
        ),
        key=lambda item: item.name.lower(),
    )

    return tuple(
        parse_code_file(path)
        for path in source_paths
    )