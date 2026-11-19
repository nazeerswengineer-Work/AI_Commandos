"""
REQTRACE AI Streamlit production application.

REQTRACE AI:
Evidence-Grounded RAG Assistant for Engineering Requirement
Traceability and Observation Analysis.

Production capabilities:

- Unified deterministic engineering pipeline
- Explicit engineering-identifier integrity
- Automatic closure-intent detection
- Optional validated Ollama narration
- Deterministic safety fallback
- Evidence Matrix
- Decision Trace
- Provenance Inspector
- Safety Inspector
- System Health
- Request timing
- Phase 16H dynamic document synchronization

Dynamic document flow:

    documents/
        |
        v
    Detect NEW / CHANGED / DELETED
        |
        v
    Parse and extract metadata
        |
        v
    Intelligent chunking
        |
        v
    Existing nomic-embed-text embedding model
        |
        v
    Existing persistent ChromaDB collections
        |
        v
    Vector verification
        |
        v
    Manifest version 1.1 update
        |
        v
    Immediately available for retrieval

Important safety rules:

- The UI never exposes raw unvalidated LLM output.
- LLM narration does not calculate assessment.
- LLM narration does not calculate confidence.
- LLM narration does not determine closure readiness.
- Explicit IDs cannot be replaced by semantically similar entities.
- Manifest state is updated only after vector verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from config import get_settings

from src.generalized_qa_models import (
    GeneralizedQAResponse,
)
from src.health import (
    HealthStatus,
    evaluate_system_health,
)

from src.identifier_integrity import (
    ExplicitIdentifierNotFound,
)

from src.ingestion_lock import (
    IngestionLockError,
)

from src.narration_models import (
    NarrationStatus,
)

from src.production_pipeline import (
    ProductionPipelineConfig,
    ProductionResponse,
    ReqTraceProductionPipeline,
    detect_closure_intent,
)

from src.streamlit_ingestion import (
    DocumentIndexStatus,
    inspect_document_index,
    synchronize_document_index,
)


# =====================================================================
# Project configuration
# =====================================================================

BASE_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parent
)


# =====================================================================
# Page configuration
# =====================================================================

st.set_page_config(
    page_title="REQTRACE AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Styling
# =====================================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .reqtrace-title {
        font-size: 2.35rem;
        font-weight: 760;
        margin-bottom: 0.15rem;
        color: #123b66;
    }

    .reqtrace-subtitle {
        color: #607181;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }

    .status-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 1rem;
        min-height: 115px;
        background: rgba(128,128,128,0.025);
    }

    .status-label {
        color: #6b7280;
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .status-value {
        font-size: 1.3rem;
        font-weight: 750;
        margin-top: 0.4rem;
    }

    .index-card {
        border: 1px solid rgba(14,138,146,0.27);
        border-radius: 11px;
        padding: 0.8rem;
        background: rgba(14,138,146,0.035);
        margin-bottom: 0.8rem;
    }

    .index-title {
        color: #0b5a78;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .index-detail {
        color: #536878;
        font-size: 0.80rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Production pipeline
# =====================================================================

@st.cache_resource
def create_pipeline(
    narration_enabled: bool,
    allow_repair: bool,
) -> ReqTraceProductionPipeline:
    """
    Create the unified production pipeline.

    Streamlit caches a separate resource for each combination of:

    - narration enabled/disabled;
    - narration repair enabled/disabled.
    """

    settings = get_settings()

    config = ProductionPipelineConfig(
        narration_enabled=(
            narration_enabled
        ),

        allow_repair_attempt=(
            allow_repair
        ),

        narration_model=(
            "llama3.2:1b"
        ),

        context_window=4096,

        maximum_output_tokens=350,

        keep_alive="15m",

        temperature=0.1,

        top_p=0.8,

        top_k=20,

        repeat_penalty=1.1,

        seed=42,
    )

    return ReqTraceProductionPipeline(
        settings=settings,
        base_directory=BASE_DIRECTORY,
        config=config,
    )


# =====================================================================
# Operational health
# =====================================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def get_system_health():
    """Return cached operational system health."""

    return evaluate_system_health(
        settings=get_settings(),
        base_directory=BASE_DIRECTORY,
        model_name="llama3.2:1b",

        # Avoid expensive deterministic smoke tests
        # during ordinary Streamlit reruns.
        include_deterministic_smoke_test=False,
    )


# =====================================================================
# Document-index status
# =====================================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_document_index_status() -> DocumentIndexStatus:
    """
    Return current filesystem-to-index synchronization status.

    This operation:

    - scans the documents directory;
    - calculates complete-file hashes;
    - compares hashes with manifest version 1.1;
    - does not invoke Ollama;
    - does not modify ChromaDB;
    - does not modify the manifest.
    """

    return inspect_document_index(
        BASE_DIRECTORY
    )


def pending_document_changes(
    status: DocumentIndexStatus,
) -> int:
    """Return number of pending index changes."""

    return int(
        status.new
        + status.changed
        + status.deleted
    )


def clear_runtime_caches() -> None:
    """
    Clear application caches after document synchronization.

    Production resources are recreated on the next request so the
    application does not retain stale retrieval state.
    """

    get_document_index_status.clear()
    get_system_health.clear()
    create_pipeline.clear()


# =====================================================================
# UI helpers
# =====================================================================

def status_color(
    value: str,
) -> str:
    """Return UI color for engineering status."""

    normalized = (
        value
        .strip()
        .upper()
    )

    if normalized in {
        "SUPPORTED",
        "HIGH",
        "CLOSURE ESTABLISHED",
        "READY",
        "INDEXED",
        "SYNCHRONIZED",
    }:
        return "#107C10"

    if normalized in {
        "NOT SUPPORTED",
        "FAILED",
        "NOT FOUND",
    }:
        return "#D13438"

    if normalized in {
        "INCONCLUSIVE",
        "MEDIUM",
        "CLOSURE NOT ESTABLISHED",
        "DEGRADED",
        "CHANGED",
        "NEW",
    }:
        return "#CA5010"

    if normalized == "LOW":
        return "#D13438"

    return "#0078D4"


def render_status_card(
    title: str,
    value: str,
) -> None:
    """Render one engineering status card."""

    color = status_color(
        value
    )

    html = (
        '<div class="status-card">'
        f'<div class="status-label">{title}</div>'
        f'<div class="status-value" '
        f'style="color:{color};">'
        f'{value}'
        '</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_identifier_group(
    title: str,
    values: tuple[str, ...],
) -> None:
    """Render one engineering-ID group."""

    st.markdown(
        f"**{title}**"
    )

    if not values:

        st.caption(
            "None"
        )

        return

    st.write(
        " · ".join(
            values
        )
    )


def evidence_type_label(
    document_type: str,
) -> str:
    """Return user-facing evidence type."""

    labels = {
        "frd": "Requirement",
        "observation": "Observation",
        "code": "Source Code",
        "test": "Test",
        "review": "Review",
        "other": "Other",
    }

    return labels.get(
        document_type.lower(),
        document_type.upper(),
    )


def evidence_type_icon(
    document_type: str,
) -> str:
    """Return icon for an evidence type."""

    icons = {
        "frd": "📘",
        "observation": "📋",
        "code": "💻",
        "test": "🧪",
        "review": "🔍",
        "other": "📄",
    }

    return icons.get(
        document_type.lower(),
        "📄",
    )


# =====================================================================
# Document-index UI
# =====================================================================

def render_index_status(
    status: DocumentIndexStatus,
) -> None:
    """Render current document-index state."""

    pending = pending_document_changes(
        status
    )

    st.markdown(
        (
            '<div class="index-card">'
            '<div class="index-title">'
            'Phase 16H Dynamic Index'
            '</div>'
            '<div class="index-detail">'
            f'Manifest {status.manifest_version} · '
            f'{status.indexed} indexed document(s)'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    metric_column_1, metric_column_2 = (
        st.columns(2)
    )

    with metric_column_1:

        st.metric(
            "Discovered",
            status.discovered,
        )

        st.metric(
            "New",
            status.new,
        )

        st.metric(
            "Changed",
            status.changed,
        )

    with metric_column_2:

        st.metric(
            "Indexed",
            status.indexed,
        )

        st.metric(
            "Deleted",
            status.deleted,
        )

        st.metric(
            "Failed",
            status.failed,
        )

    if (
        pending == 0
        and status.failed == 0
    ):

        st.success(
            "Document index is synchronized.",
            icon="✅",
        )

    elif status.failed:

        st.error(
            (
                f"{status.failed} document inspection "
                "failure(s) require attention."
            ),
            icon="❌",
        )

    else:

        st.warning(
            (
                f"{pending} document change(s) "
                "are waiting for synchronization."
            ),
            icon="⚠️",
        )


def render_sync_report(
    report,
) -> None:
    """Render the latest document synchronization report."""

    st.caption(
        (
            f"Discovered: {report.discovered} · "
            f"New: {report.new} · "
            f"Changed: {report.changed} · "
            f"Deleted: {report.deleted}"
        )
    )

    st.caption(
        (
            f"Indexed: {report.indexed} · "
            f"Removed: {report.removed} · "
            f"Skipped: {report.skipped} · "
            f"Failed: {report.failed}"
        )
    )

    for item in report.results:

        label = (
            f"{item.relative_path} "
            f"• {item.change_type} "
            f"• {item.status}"
        )

        if item.status in {
            "INDEXED",
            "REMOVED",
        }:

            st.success(
                label,
                icon="✅",
            )

        elif item.status == "FAILED":

            st.error(
                label,
                icon="❌",
            )

            if item.message:

                st.caption(
                    item.message
                )

        else:

            st.caption(
                label
            )


def render_document_management_sidebar() -> None:
    """Render Phase 16H document-index controls."""

    st.markdown(
        "### Document index"
    )

    index_status: (
        DocumentIndexStatus
        | None
    ) = None

    try:

        index_status = (
            get_document_index_status()
        )

        render_index_status(
            index_status
        )

    except Exception as exc:

        st.error(
            "Document-index status is unavailable.",
            icon="❌",
        )

        with st.expander(
            "Index diagnostics"
        ):

            st.code(
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                language=None,
            )

    (
        detect_column,
        synchronize_column,
    ) = st.columns(2)

    with detect_column:

        detect_requested = st.button(
            "Detect changes",
            use_container_width=True,
            key="reqtrace_detect_changes",
        )

    if detect_requested:

        get_document_index_status.clear()

        st.rerun()

    synchronization_disabled = bool(
        index_status is None
        or (
            index_status.new == 0
            and index_status.changed == 0
            and index_status.deleted == 0
        )
    )

    with synchronize_column:

        synchronization_requested = (
            st.button(
                "Synchronize",
                type="primary",
                use_container_width=True,
                disabled=(
                    synchronization_disabled
                ),
                key=(
                    "reqtrace_synchronize_documents"
                ),
            )
        )

    if synchronization_requested:

        try:

            with st.spinner(
                (
                    "Synchronizing engineering documents, "
                    "embeddings, and persistent vector index..."
                )
            ):

                sync_report = (
                    synchronize_document_index(
                        BASE_DIRECTORY,
                        continue_on_error=True,
                    )
                )

            st.session_state[
                "reqtrace_last_sync_report"
            ] = sync_report

            clear_runtime_caches()

            if sync_report.failed == 0:

                st.success(
                    (
                        "Document synchronization completed. "
                        f"Indexed: {sync_report.indexed}, "
                        f"Removed: {sync_report.removed}."
                    ),
                    icon="✅",
                )

            else:

                st.warning(
                    (
                        "Synchronization completed with "
                        f"{sync_report.failed} failed "
                        "document(s). Successful documents "
                        "remain indexed."
                    ),
                    icon="⚠️",
                )

            st.rerun()

        except IngestionLockError as exc:

            st.warning(
                (
                    "Document synchronization is already "
                    "running in another process."
                ),
                icon="⚠️",
            )

            with st.expander(
                "Synchronization diagnostics"
            ):

                st.code(
                    str(exc),
                    language=None,
                )

        except Exception as exc:

            st.error(
                (
                    "REQTRACE AI could not synchronize "
                    "the document index."
                ),
                icon="❌",
            )

            with st.expander(
                "Synchronization diagnostics"
            ):

                st.code(
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    language=None,
                )

    last_sync_report = (
        st.session_state.get(
            "reqtrace_last_sync_report"
        )
    )

    if last_sync_report is not None:

        with st.expander(
            "Last synchronization"
        ):

            render_sync_report(
                last_sync_report
            )

    st.caption(
        (
            "Synchronization reuses the existing "
            "nomic-embed-text model and persistent "
            "REQTRACE AI Chroma collections."
        )
    )


# =====================================================================
# Generalized deterministic QA
# =====================================================================


def render_generalized_qa_response(
    result: GeneralizedQAResponse,
) -> None:
    """Render concise deterministic inventory/evidence QA."""

    st.divider()

    st.markdown(
        "### Generalized engineering QA"
    )

    st.success(
        "Deterministic evidence result",
        icon="✅",
    )

    metric_column_1, metric_column_2 = st.columns(2)

    with metric_column_1:
        st.metric(
            "Query intent",
            result.intent.value.replace("_", " "),
        )

    with metric_column_2:
        st.metric(
            "Matching records",
            result.record_count,
        )

    st.markdown(
        result.answer
    )

    if result.engineering_ids:
        st.markdown(
            "#### Matching engineering IDs"
        )

        identifier_text = " · ".join(
            result.engineering_ids
        )

        st.code(
            identifier_text,
            language=None,
        )

    st.info(
        result.source_statement,
        icon="ℹ️",
    )

    with st.expander(
        "Routing details"
    ):
        st.markdown(
            f"**Intent rule:** `{result.intent_rule}`"
        )

        st.markdown(
            f"**Response mode:** `{result.response_mode.value}`"
        )

        st.markdown(
            "**LLM used:** No"
        )

        st.caption(
            "Counts, filters, evidence availability, and entity IDs "
            "are calculated deterministically from the synchronized "
            "engineering evidence manifest."
        )


# =====================================================================
# Narration-delivery status
# =====================================================================

def render_delivery_status(
    result: ProductionResponse,
) -> None:
    """Render the safe narration delivery path."""

    if (
        result.delivery_status
        == NarrationStatus
        .VALIDATED_GENERATION
    ):

        st.success(
            (
                "Validated grounded LLM narration "
                "delivered."
            ),
            icon="✅",
        )

        return

    if (
        result.delivery_status
        == NarrationStatus
        .VALIDATED_REPAIR
    ):

        st.success(
            (
                "Validated repaired LLM narration "
                "delivered."
            ),
            icon="✅",
        )

        return

    st.info(
        (
            "Safe deterministic engineering "
            "answer delivered."
        ),
        icon="ℹ️",
    )

    if result.fallback_reason:

        st.caption(
            result.fallback_reason
        )


# =====================================================================
# Engineering Answer tab
# =====================================================================

def render_engineering_answer_tab(
    result: ProductionResponse,
) -> None:
    """Render the final safe answer."""

    st.markdown(
        "### Grounded engineering answer"
    )

    render_delivery_status(
        result
    )

    st.markdown(
        result.final_answer
    )

    st.divider()

    support_column, blocker_column = (
        st.columns(2)
    )

    with support_column:

        st.markdown(
            "#### Supporting evidence"
        )

        if result.supporting_items:

            for item in result.supporting_items:

                st.success( 
                    item,
                    icon="✅",
                )

        else:

            st.caption(
                (
                    "No decisive supporting item "
                    "was recorded."
                )
            )

    with blocker_column:

        st.markdown(
            "#### Blocking or missing evidence"
        )

        if result.blocking_items:

            for item in result.blocking_items:

                st.warning(
                    item,
                    icon="⚠️",
                )

        else:

            st.caption(
                "No deterministic blocker recorded."
            )


# =====================================================================
# Evidence Matrix tab
# =====================================================================

def render_evidence_matrix_tab(
    result: ProductionResponse,
) -> None:
    """Render validated evidence citations."""

    st.markdown(
        "### Evidence matrix"
    )

    st.caption(
        (
            "Validated evidence contributing "
            "to the engineering decision."
        )
    )

    if not result.citations:

        st.info(
            "No evidence citations are available."
        )

        return

    rows: list[
        dict[str, Any]
    ] = []

    for citation in result.citations:

        rows.append(
            {
                "Citation": (
                    citation.citation_id
                ),

                "Type": (
                    evidence_type_label(
                        citation.document_type
                    )
                ),

                "Source": (
                    citation.source_label
                ),

                "Requirement": (
                    citation.requirement_id
                    or ""
                ),

                "Observation": (
                    citation.observation_id
                    or ""
                ),

                "Test": (
                    citation.test_case_id
                    or ""
                ),

                "Result": (
                    citation.result_status
                    or ""
                ),

                "Location": (
                    citation.provenance_location
                ),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### Evidence coverage"
    )

    represented_types = {
        citation.document_type.lower()
        for citation in result.citations
    }

    coverage_columns = st.columns(5)

    core_types = (
        (
            "frd",
            "Requirement",
        ),

        (
            "observation",
            "Observation",
        ),

        (
            "code",
            "Code",
        ),

        (
            "test",
            "Test",
        ),

        (
            "review",
            "Review",
        ),
    )

    for (
        column,
        (
            document_type,
            label,
        ),
    ) in zip(
        coverage_columns,
        core_types,
        strict=True,
    ):

        with column:

            if (
                document_type
                in represented_types
            ):

                st.success(
                    f"{label}\n\nAvailable",
                    icon="✅",
                )

            else:

                st.info(
                    (
                        f"{label}\n\n"
                        "Not in final context"
                    ),
                    icon="ℹ️",
                )


# =====================================================================
# Decision Trace tab
# =====================================================================

def render_decision_trace_tab(
    result: ProductionResponse,
) -> None:
    """Render deterministic decision progression."""

    st.markdown(
        "### Decision trace"
    )

    st.caption(
        (
            "The engineering result is determined "
            "before LLM narration."
        )
    )

    st.markdown(
        "#### 1. Grounded evidence"
    )

    st.success(
        (
            f"{len(result.citations)} validated "
            "evidence record(s) support this "
            "production response."
        ),
        icon="✅",
    )

    trace_columns = st.columns(3)

    with trace_columns[0]:
        render_identifier_group(
            "Requirements",
            result.requirement_ids,
        )

    with trace_columns[1]:
        render_identifier_group(
            "Observations",
            result.observation_ids,
        )

    with trace_columns[2]:
        render_identifier_group(
            "Tests",
            result.test_case_ids,
        )

    st.markdown(
        "#### 2. Evidence-condition analysis"
    )

    if result.blocking_items:

        for item in result.blocking_items:

            st.warning(
                item,
                icon="⚠️",
            )

    else:

        st.success(
            (
                "No deterministic assessment "
                "blocker was identified."
            ),
            icon="✅",
        )

    st.markdown(
        "#### 3. Applied deterministic rule"
    )

    st.code(
        result.assessment_rule,
        language=None,
    )

    st.write(
        result.assessment_reason
    )

    st.markdown(
        "#### 4. Assessment"
    )

    render_status_card(
        "Assessment",
        result.assessment,
    )

    st.markdown(
        "#### 5. Confidence"
    )

    render_status_card(
        "Confidence",
        result.confidence,
    )

    st.write(
        result.confidence_reason
    )

    st.markdown(
        "#### 6. Closure readiness"
    )

    render_status_card(
        "Closure Status",
        result.closure_status,
    )

    st.write(
        result.closure_reason
    )

    st.info(
        (
            "Observation assessment and closure "
            "readiness are evaluated separately."
        ),
        icon="ℹ️",
    )


# =====================================================================
# Provenance tab
# =====================================================================

def render_provenance_tab(
    result: ProductionResponse,
) -> None:
    """Render physical source provenance."""

    st.markdown(
        "### Source provenance inspector"
    )

    st.caption(
        (
            "Pages, filenames, functions, sections, "
            "and physical line ranges originate from "
            "validated engineering metadata."
        )
    )

    if not result.citations:

        st.info(
            "No provenance records are available."
        )

        return

    for citation in result.citations:

        icon = evidence_type_icon(
            citation.document_type
        )

        type_label = (
            evidence_type_label(
                citation.document_type
            )
        )

        with st.expander(
            (
                f"{icon} "
                f"[{citation.citation_id}] "
                f"{type_label}"
            )
        ):

            left_column, right_column = (
                st.columns(
                    [
                        2,
                        1,
                    ]
                )
            )

            with left_column:

                st.markdown(
                    "**Source**"
                )

                st.write(
                    citation.source_label
                )

                st.markdown(
                    "**Physical provenance**"
                )

                st.write(
                    citation.provenance_location
                )

                st.markdown(
                    "**Evidence ID**"
                )

                st.code(
                    citation.evidence_id,
                    language=None,
                )

            with right_column:

                if citation.requirement_id:

                    st.markdown(
                        "**REQ**"
                    )

                    st.code(
                        citation.requirement_id,
                        language=None,
                    )

                if citation.observation_id:

                    st.markdown(
                        "**OBS**"
                    )

                    st.code(
                        citation.observation_id,
                        language=None,
                    )

                if citation.test_case_id:

                    st.markdown(
                        "**TEST**"
                    )

                    st.code(
                        citation.test_case_id,
                        language=None,
                    )

                if citation.result_status:

                    st.markdown(
                        "**Test result**"
                    )

                    status = (
                        citation
                        .result_status
                        .upper()
                    )

                    if status == "PASS":

                        st.success(
                            "PASS",
                            icon="✅",
                        )

                    elif status == "FAIL":

                        st.error(
                            "FAIL",
                            icon="❌",
                        )

                    elif status == (
                        "NOT EXECUTED"
                    ):

                        st.warning(
                            "NOT EXECUTED",
                            icon="⚠️",
                        )

                    else:

                        st.warning(
                            citation.result_status,
                            icon="⚠️",
                        )


# =====================================================================
# Safety tab
# =====================================================================

def render_safety_tab(
    result: ProductionResponse,
) -> None:
    """Render grounding and delivery safety information."""

    st.markdown(
        "### Safety and delivery inspector"
    )

    st.markdown(
        "#### Grounding"
    )

    if result.grounding_passed:

        st.success(
            "Grounding validation passed.",
            icon="✅",
        )

    else:

        st.error(
            "Grounding validation failed.",
            icon="❌",
        )

    st.markdown(
        "#### Delivery path"
    )

    delivery_map = {
        (
            NarrationStatus
            .VALIDATED_GENERATION
        ): (
            "Validated LLM generation",
            (
                "The initial narration passed "
                "the Phase 14 validation gate."
            ),
        ),

        (
            NarrationStatus
            .VALIDATED_REPAIR
        ): (
            "Validated LLM repair",
            (
                "The initial narration was rejected. "
                "The one permitted repair passed validation."
            ),
        ),

        (
            NarrationStatus
            .DETERMINISTIC_FALLBACK
        ): (
            "Deterministic fallback",
            (
                "No unvalidated LLM output was delivered. "
                "REQTRACE AI returned the grounded "
                "deterministic engineering answer."
            ),
        ),
    }

    (
        delivery_title,
        delivery_explanation,
    ) = delivery_map[
        result.delivery_status
    ]

    st.info(
        (
            f"**{delivery_title}**\n\n"
            f"{delivery_explanation}"
        ),
        icon="ℹ️",
    )

    metric_1, metric_2 = st.columns(2)

    with metric_1:

        st.metric(
            "Generation attempts",
            result.generation_attempts,
        )

        st.metric(
            "Repair used",
            (
                "Yes"
                if result.used_repair
                else "No"
            ),
        )

    with metric_2:

        st.metric(
            "Fallback used",
            (
                "Yes"
                if result.used_fallback
                else "No"
            ),
        )

        st.metric(
            "Validated LLM answer",
            (
                "Yes"
                if result.generated_by_llm
                else "No"
            ),
        )

    if result.model:

        st.markdown(
            "**Narration model**"
        )

        st.code( 
            result.model,
            language=None,
        )

    if result.fallback_reason:

        st.markdown(
            "#### Fallback reason"
        )

        st.warning(
            result.fallback_reason,
            icon="⚠️",
        )

    st.markdown(
        "#### Request timing"
    )

    (
        timing_column_1,
        timing_column_2,
        timing_column_3,
    ) = st.columns(3)

    with timing_column_1:

        st.metric(
            "Deterministic pipeline",
            (
                f"{result.timing.deterministic_seconds:.2f} s"
            ),
        )

    with timing_column_2:

        st.metric(
            "Narration",
            (
                f"{result.timing.narration_seconds:.2f} s"
            ),
        )

    with timing_column_3:

        st.metric(
            "Total request",
            (
                f"{result.timing.total_seconds:.2f} s"
            ),
        )

    st.markdown(
        "#### Safety boundaries"
    )

    safety_boundaries = (
        "LLM does not calculate assessment.",
        "LLM does not calculate confidence.",
        "LLM does not determine closure readiness.",
        "Unsupported engineering IDs are blocked.",
        "Explicit IDs cannot be semantically substituted.",
        "Fabricated citations are blocked.",
        "Revision conflicts cannot be resolved by narration.",
        "Explicit traceability cannot be rewritten by narration.",
        "Referenced Test IDs are not treated as executed Test evidence.",
        "Invalid narration falls back to deterministic output.",
    )

    for boundary in safety_boundaries:

        st.markdown(
            f"- ✅ {boundary}"
        )


# =====================================================================
# Complete production result
# =====================================================================

def render_result(
    result: ProductionResponse,
) -> None:
    """Render all production result views."""

    st.divider()

    (
        assessment_column,
        confidence_column,
        closure_column,
    ) = st.columns(3)

    with assessment_column:

        render_status_card(
            "Assessment",
            result.assessment,
        )

    with confidence_column:

        render_status_card(
            "Confidence",
            result.confidence,
        )

    with closure_column:

        render_status_card(
            "Closure Status",
            result.closure_status,
        )

    st.markdown("")

    (
        answer_tab,
        evidence_tab,
        decision_tab,
        provenance_tab,
        safety_tab,
    ) = st.tabs(
        [
            "Engineering Answer",
            "Evidence Matrix",
            "Decision Trace",
            "Provenance",
            "Safety",
        ]
    )

    with answer_tab:

        render_engineering_answer_tab(
            result
        )

    with evidence_tab:

        render_evidence_matrix_tab(
            result
        )

    with decision_tab:

        render_decision_trace_tab(
            result
        )

    with provenance_tab:

        render_provenance_tab(
            result
        )

    with safety_tab:

        render_safety_tab(
            result
        )


# =====================================================================
# Header
# =====================================================================

st.markdown(
    """
    <div class="reqtrace-title">
        REQTRACE AI
    </div>

    <div class="reqtrace-subtitle">
        Evidence-Grounded RAG Assistant for Engineering Requirement
        Traceability and Observation Analysis
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Sidebar
# =====================================================================

with st.sidebar:

    st.markdown(
        "## Analysis settings"
    )

    evaluate_closure = st.toggle(
        "Evaluate closure readiness",
        value=False,
        help=(
            "Closure evaluation is also activated "
            "automatically for closure-related queries."
        ),
    )

    narration_enabled = st.toggle(
        "Enable grounded LLM narration",
        value=True,
        help=(
            "Only validated narration is delivered. "
            "Failures fall back to deterministic output."
        ),
    )

    allow_repair = st.toggle(
        "Allow one narration repair",
        value=True,
        disabled=(
            not narration_enabled
        ),
    )

    st.divider()

    # --------------------------------------------------------------
    # Phase 16H.5 dynamic document synchronization
    # --------------------------------------------------------------

    render_document_management_sidebar()

    # --------------------------------------------------------------
    # Phase 15D operational system health
    # --------------------------------------------------------------

    st.divider()

    st.markdown(
        "### System health"
    )

    if st.button(
        "Refresh health",
        use_container_width=True,
        key="reqtrace_refresh_health",
    ):

        get_system_health.clear()

        st.rerun()

    try:

        health = get_system_health()

        if (
            health.status
            == HealthStatus.READY
        ):

            st.success(
                "System ready",
                icon="✅",
            )

        elif (
            health.status
            == HealthStatus.DEGRADED
        ):

            st.warning(
                "System operating in degraded mode",
                icon="⚠️",
            )

        else:

            st.error(
                "Core system unavailable",
                icon="❌",
            )

        with st.expander(
            "Health details"
        ):

            for check in health.checks:

                message = (
                    f"{check.component}: "
                    f"{check.message}"
                )

                if (
                    check.status
                    == HealthStatus.READY
                ):

                    st.success(
                        message,
                        icon="✅",
                    )

                elif (
                    check.status
                    == HealthStatus.DEGRADED
                ):

                    st.warning(
                        message,
                        icon="⚠️",
                    )

                else:

                    st.error(
                        message,
                        icon="❌",
                    )

    except Exception as exc:

        st.error(
            "System health is unavailable.",
            icon="❌",
        )

        with st.expander(
            "Health diagnostics"
        ):

            st.code(
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                language=None,
            )

    st.divider()

    st.markdown(
        "### Benchmark examples"
    )

    st.code(
        "\n".join(
            [
                "OBS-023",
                "OBS-031",
                "OBS-044",
                "OBS-052",
                "REQ-050",
                "OBS-070",
                "REQ-900",
            ]
        ),
        language=None,
    )

    st.markdown(
        "### Natural-language examples"
    )

    st.code(
        "\n".join(
            [
                "Is OBS-070 open or closed?",
                "What is the closure status of OBS-070?",
                "Can OBS-070 be closed?",
                "What does REQ-900 specify?",
                "How many observations are open?",
                "List all open observations",
                "Which requirements have missing test evidence?",
            ]
        ),
        language=None,
    )

    st.divider()

    st.caption(
        (
            "REQTRACE AI provides evidence-derived "
            "engineering decision support. Final "
            "engineering and closure authority remain "
            "with responsible human personnel."
        )
    )


# =====================================================================
# Query form
# =====================================================================

with st.form(
    "reqtrace_query_form",
    clear_on_submit=False,
):

    query = st.text_input(
        "Engineering query",
        placeholder=(
            "Example: OBS-023, REQ-900, "
            "How many observations are open?"
        ),
    )

    submitted = st.form_submit_button(
        "Analyze evidence",
        type="primary",
        use_container_width=True,
    )


# =====================================================================
# Execute production query
# =====================================================================

if submitted:

    clean_query = query.strip()

    if not clean_query:

        st.warning(
            (
                "Enter an engineering query "
                "before running analysis."
            ),
            icon="⚠️",
        )

        st.stop()

    automatic_closure_intent = (
        detect_closure_intent(
            clean_query
        )
    )

    effective_closure = bool(
        evaluate_closure
        or automatic_closure_intent
    )

    automatic_only = bool(
        automatic_closure_intent
        and not evaluate_closure
    )

    try:

        pipeline = create_pipeline(
            narration_enabled=(
                narration_enabled
            ),

            allow_repair=(
                allow_repair
                if narration_enabled
                else False
            ),
        )

        with st.spinner(
            (
                "Running deterministic engineering query routing and analysis..."
            )
        ):

            result = pipeline.run_routed(
                clean_query,
                evaluate_closure=(
                    effective_closure
                ),
            )

        st.session_state[
            "reqtrace_last_result"
        ] = result

        st.session_state[
            "reqtrace_last_query"
        ] = clean_query

        st.session_state[
            "reqtrace_closure_auto_detected"
        ] = bool(
            automatic_only
            and isinstance(
                result,
                ProductionResponse,
            )
        )

    except ExplicitIdentifierNotFound as exc:

        st.warning(
            (
                "The requested engineering identifier "
                "was not found in the authoritative "
                "evidence corpus."
            ),
            icon="⚠️",
        )

        st.markdown(
            "**Missing identifier(s)**"
        )

        for identifier in exc.missing_ids:

            st.code(
                identifier,
                language=None,
            )

        st.info(
            (
                "REQTRACE AI did not substitute another "
                "requirement, observation, or test case. "
                "LLM narration was not invoked."
            ),
            icon="ℹ️",
        )

        st.stop()

    except ValueError as exc:

        st.warning(
            str(exc),
            icon="⚠️",
        )

        st.stop()

    except Exception as exc:

        st.error(
            (
                "REQTRACE AI could not complete the "
                "engineering analysis. The deterministic "
                "pipeline or evidence store may currently "
                "be unavailable."
            ),
            icon="❌",
        )

        with st.expander(
            "Technical diagnostics"
        ):

            st.code(
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                language=None,
            )

        st.stop()


# =====================================================================
# Display latest result
# =====================================================================

result = st.session_state.get(
    "reqtrace_last_result"
)

if result is not None:

    last_query = (
        st.session_state.get(
            "reqtrace_last_query",
            "",
        )
    )

    st.markdown(
        f"**Last query:** `{last_query}`"
    )

    if (
        st.session_state.get(
            "reqtrace_closure_auto_detected",
            False,
        )
    ):

        st.info(
            (
                "Closure intent was detected "
                "automatically. Closure-readiness "
                "evaluation was enabled for this query."
            ),
            icon="ℹ️",
        )

    if isinstance(
        result,
        GeneralizedQAResponse,
    ):
        render_generalized_qa_response(
            result
        )

    elif isinstance(
        result,
        ProductionResponse,
    ):
        render_result(
            result
        )

    else:
        st.error(
            "REQTRACE AI returned an unsupported response type.",
            icon="❌",
        )
