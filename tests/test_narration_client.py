"""
Tests for the Phase 14B bounded Ollama narration client.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.narration_client import (
    NarrationAPIError,
    NarrationClientConfig,
    NarrationResponseError,
    OllamaNarrationClient,
    extract_response_text,
    nanoseconds_to_seconds,
)
from src.narration_models import (
    NarrationEvidenceBlock,
    NarrationRequest,
)
from src.narration_prompt import (
    NarrationPrompt,
)


def make_prompt() -> NarrationPrompt:
    """Create one valid constrained narration prompt."""

    block = NarrationEvidenceBlock(
        citation_id="CIT-001",
        evidence_id="OBS-023",
        document_type="observation",
        source_label=(
            "OBSERVATION | report.pdf | page 1"
        ),
        provenance_location=(
            "report.pdf, page 1"
        ),
        excerpt=(
            "Null input caused the observed failure."
        ),
        requirement_id="REQ-017",
        observation_id="OBS-023",
        test_case_id="TC-102",
    )

    request = NarrationRequest(
        request_version="1.0",
        query="OBS-023",
        assessment="SUPPORTED",
        assessment_rule=(
            "AR-008_EXECUTED_FAIL_SUPPORT"
        ),
        assessment_reason=(
            "Complete failing evidence supports "
            "the observation."
        ),
        confidence="HIGH",
        confidence_reason=(
            "The evidence chain is complete."
        ),
        closure_status="NOT ASSESSED",
        closure_reason=(
            "Closure was not requested."
        ),
        closure_was_assessed=False,
        requirement_ids=(
            "REQ-017",
        ),
        observation_ids=(
            "OBS-023",
        ),
        test_case_ids=(
            "TC-102",
        ),
        supporting_items=(
            "Executed FAIL evidence is available.",
        ),
        blocking_items=(),
        allowed_claims=(
            "OBS-023 is SUPPORTED.",
        ),
        prohibited_claims=(
            "Do not state that OBS-023 is closed.",
        ),
        citation_allowlist=(
            "CIT-001",
        ),
        engineering_id_allowlist=(
            "REQ-017",
            "OBS-023",
            "TC-102",
        ),
        evidence_blocks=(
            block,
        ),
        narrator_instructions=(
            "Preserve the assessment exactly.",
        ),
        deterministic_fallback_text=(
            "Assessment: SUPPORTED"
        ),
    )

    return NarrationPrompt(
        system_prompt=(
            "Preserve all deterministic fields."
        ),
        user_prompt=(
            "Explain OBS-023 using CIT-001."
        ),
        request=request,
    )


def successful_response() -> dict[str, object]:
    """Return one realistic Ollama response."""

    return {
        "model": "llama3.2:1b",
        "created_at": (
            "2026-11-11T12:00:00Z"
        ),
        "message": {
            "role": "assistant",
            "content": (
                "Assessment\n\n"
                "OBS-023 is SUPPORTED [CIT-001]."
            ),
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": (
            2_500_000_000
        ),
        "prompt_eval_count": 250,
        "eval_count": 50,
    }


def test_request_payload_is_bounded() -> None:
    """Payload must use constrained generation settings."""

    config = NarrationClientConfig()

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=successful_response(),
        )
    )

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        payload = (
            client.build_request_payload(
                make_prompt()
            )
        )

    assert payload["model"] == (
        "llama3.2:1b"
    )

    assert payload["stream"] is False
    assert payload["think"] is False

    options = payload["options"]

    assert options["temperature"] == (
        0.1
    )

    assert options["num_predict"] == (
        350
    )

    assert options["seed"] == 42


def test_messages_remain_separate() -> None:
    """System and user messages must remain separate."""

    config = NarrationClientConfig()

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=successful_response(),
        )
    )

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        payload = (
            client.build_request_payload(
                make_prompt()
            )
        )

    messages = payload["messages"]

    assert messages[0]["role"] == (
        "system"
    )

    assert messages[1]["role"] == (
        "user"
    )


def test_successful_generation() -> None:
    """Successful API response should create NarrationResponse."""

    captured_payload: dict[
        str,
        object,
    ] = {}

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_payload.update(
            json.loads(
                request.content
            )
        )

        return httpx.Response(
            200,
            json=successful_response(),
        )

    transport = httpx.MockTransport(
        handler
    )

    config = NarrationClientConfig()

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        response = client.generate(
            make_prompt()
        )

    assert response.model == (
        "llama3.2:1b"
    )

    assert response.prompt_tokens == (
        250
    )

    assert (
        response.completion_tokens
        == 50
    )

    assert (
        response
        .generation_duration_seconds
        == pytest.approx(2.5)
    )

    assert (
        captured_payload["stream"]
        is False
    )


def test_repair_metadata_is_preserved() -> None:
    """Repair-attempt metadata must survive generation."""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=successful_response(),
        )
    )

    config = NarrationClientConfig()

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        response = client.generate(
            make_prompt(),
            attempt_number=2,
            is_repair_attempt=True,
        )

    assert response.attempt_number == 2

    assert (
        response.is_repair_attempt
        is True
    )


def test_http_error_is_rejected() -> None:
    """Non-success status must not produce narration."""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            500,
            text="model failure",
        )
    )

    config = NarrationClientConfig()

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        with pytest.raises(
            NarrationAPIError
        ):
            client.generate(
                make_prompt()
            )


def test_invalid_json_is_rejected() -> None:
    """Invalid JSON response must fail."""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"not-json",
            headers={
                "Content-Type": (
                    "application/json"
                ),
            },
        )
    )

    config = NarrationClientConfig()

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        with pytest.raises(
            NarrationResponseError
        ):
            client.generate(
                make_prompt()
            )


def test_empty_model_content_is_rejected() -> None:
    """Empty assistant text must never enter narration."""

    payload = successful_response()

    payload["message"] = {
        "role": "assistant",
        "content": "   ",
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=payload,
        )
    )

    config = NarrationClientConfig()

    with httpx.Client(
        transport=transport,
        base_url=config.base_url,
    ) as http_client:

        client = OllamaNarrationClient(
            config,
            http_client=http_client,
        )

        with pytest.raises(
            NarrationResponseError
        ):
            client.generate(
                make_prompt()
            )


def test_duration_conversion() -> None:
    """Ollama nanosecond durations should convert to seconds."""

    assert nanoseconds_to_seconds(
        1_500_000_000
    ) == pytest.approx(
        1.5
    )


def test_extract_response_text() -> None:
    """Chat content should be extracted exactly."""

    text = extract_response_text(
        successful_response()
    )

    assert "SUPPORTED" in text