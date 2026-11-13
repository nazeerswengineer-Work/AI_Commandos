"""
REQTRACE AI Streamlit application.

Production UI for:

REQTRACE AI:
Evidence-Grounded RAG Assistant for Engineering Requirement
Traceability and Observation Analysis

The application consumes only the unified production pipeline.

The UI never exposes raw unvalidated LLM output.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import get_settings

from src.health import (
    HealthStatus,
    evaluate_system_health,
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
from src.identifier_integrity import (
    ExplicitIdentifierNotFound,
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
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .reqtrace-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .reqtrace-subtitle {
        color: #6b7280;
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

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Pipeline
# =====================================================================

@st.cache_resource
def create_pipeline(
    narration_enabled: bool,
    allow_repair: bool,
) -> ReqTraceProductionPipeline:
    """Create the unified production pipeline."""

    settings = get_settings()

    config = ProductionPipelineConfig(
        narration_enabled=(
            narration_enabled
        ),
        allow_repair_attempt=(
            allow_repair
        ),
        narration_model="llama3.2:1b",
        context_window=4096,
        maximum_output_tokens=350,
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
# Health
# =====================================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def get_system_health():
    """Return cached operational health."""

    return evaluate_system_health(
        settings=get_settings(),
        base_directory=BASE_DIRECTORY,
        model_name="llama3.2:1b",

        # Avoid expensive deterministic smoke tests
        # during normal Streamlit reruns.
        include_deterministic_smoke_test=False,
    )


# =====================================================================
# UI helpers
# =====================================================================

def status_color(
    value: str,
) -> str:

    normalized = (
        value
        .strip()
        .upper()
    )

    if normalized in {
        "SUPPORTED",
        "HIGH",
        "CLOSURE ESTABLISHED",
    }:
        return "#107C10"

    if normalized == "NOT SUPPORTED":
        return "#D13438"

    if normalized in {
        "INCONCLUSIVE",
        "MEDIUM",
        "CLOSURE NOT ESTABLISHED",
    }:
        return "#CA5010"

    if normalized == "LOW":
        return "#D13438"

    return "#0078D4"


def render_status_card(
    title: str,
    value: str,
) -> None:
    """Render a compact engineering decision card."""

    color = status_color(
        value
    )

    # Keep HTML on one logical line to prevent
    # Streamlit treating indented HTML as code.

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

    labels = {
        "frd": "Requirement",
        "observation": "Observation",
        "code": "Source Code",
        "test": "Test",
        "review": "Review",
    }

    return labels.get(
        document_type.lower(),
        document_type.upper(),
    )


def evidence_type_icon(
    document_type: str,
) -> str:

    icons = {
        "frd": "📘",
        "observation": "📋",
        "code": "💻",
        "test": "🧪",
        "review": "🔍",
    }

    return icons.get(
        document_type.lower(),
        "📄",
    )


# =====================================================================
# Delivery
# =====================================================================

def render_delivery_status(
    result: ProductionResponse,
) -> None:

    if (
        result.delivery_status
        == NarrationStatus
        .VALIDATED_GENERATION
    ):

        st.success(
            "Validated grounded LLM narration delivered.",
            icon="✅",
        )

        return

    if (
        result.delivery_status
        == NarrationStatus
        .VALIDATED_REPAIR
    ):

        st.success(
            "Validated repaired LLM narration delivered.",
            icon="✅",
        )

        return

    st.info(
        "Safe deterministic engineering answer delivered.",
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

            for item in (
                result.supporting_items
            ):

                st.success(
                    item,
                    icon="✅",
                )

        else:

            st.caption(
                "No decisive supporting item recorded."
            )

    with blocker_column:

        st.markdown(
            "#### Blocking / missing evidence"
        )

        if result.blocking_items:

            for item in (
                result.blocking_items
            ):

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

    st.markdown(
        "### Evidence matrix"
    )

    st.caption(
        "Validated evidence contributing to the "
        "engineering decision."
    )

    if not result.citations:

        st.info(
            "No evidence citations are available."
        )

        return

    rows: list[
        dict[str, str]
    ] = []

    for citation in (
        result.citations
    ):

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
        citation
        .document_type
        .lower()

        for citation
        in result.citations
    }

    coverage_columns = ( 
        st.columns(5)
    )

    core_types = (
        ("frd", "Requirement"),
        ("observation", "Observation"),
        ("code", "Code"),
        ("test", "Test"),
        ("review", "Review"),
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
                    f"{label}\n\nNot in final context",
                    icon="ℹ️",
                )


# =====================================================================
# Decision Trace tab
# =====================================================================

def render_decision_trace_tab(
    result: ProductionResponse,
) -> None:

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
            f"{len(result.citations)} "
            "validated evidence record(s) "
            "support this production response."
        ),
        icon="✅",
    )

    trace_columns = ( 
        st.columns(3)
    )

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

        for item in (
            result.blocking_items
        ):

            st.warning(
                item,
                icon="⚠️",
            )

    else:

        st.success(
            "No deterministic assessment blocker identified.",
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
            "Observation assessment and "
            "closure readiness are evaluated separately."
        ),
        icon="ℹ️",
    )


# =====================================================================
# Provenance tab
# =====================================================================

def render_provenance_tab(
    result: ProductionResponse,
) -> None:

    st.markdown(
        "### Source provenance inspector"
    )

    st.caption(
        (
            "Page numbers, filenames, functions "
            "and physical source-line ranges originate "
            "from validated engineering metadata."
        )
    )

    if not result.citations:

        st.info(
            "No provenance records available."
        )

        return

    for citation in ( 
        result.citations
    ):

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

    st.markdown(
        "### Safety & delivery inspector"
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

    metric_1, metric_2 = (
        st.columns(2)
    )

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

    # --------------------------------------------------------------
    # Phase 15D request timing
    # --------------------------------------------------------------

    st.markdown(
        "#### Request timing"
    )

    (
        timing_1,
        timing_2,
        timing_3,
    ) = st.columns(3)

    with timing_1:

        st.metric(
            "Deterministic pipeline",
            (
                f"{result.timing.deterministic_seconds:.2f} s"
            ),
        )

    with timing_2:

        st.metric(
            "Narration",
            (
                f"{result.timing.narration_seconds:.2f} s"
            ),
        )

    with timing_3:

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
        "Fabricated citations are blocked.",
        "Revision conflicts cannot be resolved by narration.",
        "Explicit traceability cannot be rewritten by narration.",
        "Invalid narration falls back to deterministic output.",
    )

    for boundary in (
        safety_boundaries
    ):

        st.markdown(
            f"- ✅ {boundary}"
        )


# =====================================================================
# Complete result
# =====================================================================

def render_result(
    result: ProductionResponse,
) -> None:

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

    evaluate_closure = (
        st.toggle(
            "Evaluate closure readiness",
            value=False,
            help=(
                "Closure evaluation is also activated "
                "automatically for closure-related queries."
            ),
        )
    )

    narration_enabled = (
        st.toggle(
            "Enable grounded LLM narration",
            value=True,
            help=(
                "Only validated narration is delivered. "
                "Failures fall back to deterministic output."
            ),
        )
    )

    allow_repair = (
        st.toggle(
            "Allow one narration repair",
            value=True,
            disabled=(
                not narration_enabled
            ),
        )
    )

    # --------------------------------------------------------------
    # Phase 15D health
    # --------------------------------------------------------------

    st.divider()

    st.markdown(
        "### System health"
    )

    if st.button(
        "Refresh health",
        use_container_width=True,
    ):

        get_system_health.clear()

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

        for check in (
            health.checks
        ):

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
                "Is OBS-070 ready for closure?",
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
            "with the responsible human authority."
        )
    )


# =====================================================================
# Query
# =====================================================================

with st.form(
    "reqtrace_query_form",
    clear_on_submit=False,
):

    query = st.text_input(
        "Engineering query",
        placeholder=(
            "Example: OBS-023, REQ-050, "
            "Is OBS-070 open or closed?"
        ),
    )

    submitted = (
        st.form_submit_button(
            "Analyze evidence",
            type="primary",
            use_container_width=True,
        )
    )


# =====================================================================
# Execute
# =====================================================================

if submitted:

    clean_query = (
        query.strip()
    )

    if not clean_query:

        st.warning(
            "Enter an engineering query before running analysis.",
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
            "Running grounded engineering analysis..."
        ):

            result = pipeline.run(
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
        ] = automatic_only

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

        for identifier in (
            exc.missing_ids
        ):

            st.code(
                identifier,
                language=None,
            )

        st.info(
            (
                "REQTRACE AI did not substitute another "
                "requirement, observation, or test case."
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
# Display result
# =====================================================================

result = (
    st.session_state.get(
        "reqtrace_last_result"
    )
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
                "Closure intent was detected automatically. "
                "Closure-readiness evaluation was enabled "
                "for this query."
            ),
            icon="ℹ️",
        )

    render_result(
        result
    )
