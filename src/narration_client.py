"""
Bounded Ollama narration client for REQTRACE AI.

Phase 14B sends a Phase 14A constrained narration prompt to the local
Ollama chat API.

The returned NarrationResponse is untrusted model output.

It must not be shown to an end user until Phase 14C narration validation
has passed.

This module does not:

- validate generated narration
- modify deterministic decisions
- expose raw model output to the UI
- calculate assessment or confidence
- determine closure readiness
- create citations
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.narration_models import (
    NarrationResponse,
)
from src.narration_prompt import (
    NarrationPrompt,
)


class NarrationClientError(
    RuntimeError
):
    """Base exception for narration-client failures."""


class NarrationConnectionError(
    NarrationClientError
):
    """Raised when the local Ollama service is unavailable."""


class NarrationTimeoutError(
    NarrationClientError
):
    """Raised when narration generation times out."""


class NarrationAPIError(
    NarrationClientError
):
    """Raised when Ollama returns an unsuccessful response."""


class NarrationResponseError(
    NarrationClientError
):
    """Raised when Ollama returns malformed or empty data."""


@dataclass(frozen=True)
class NarrationClientConfig:
    """Configuration for bounded local narration generation."""

    base_url: str = (
        "http://localhost:11434"
    )

    model: str = (
        "llama3.2:1b"
    )

    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 120.0
    write_timeout_seconds: float = 30.0

    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 20
    repeat_penalty: float = 1.1

    seed: int = 42

    context_window: int = 4096
    maximum_output_tokens: int = 350

    keep_alive: str = "15m"

    def __post_init__(
        self,
    ) -> None:
        """Validate generation configuration."""

        if not self.base_url.strip():
            raise ValueError(
                "base_url must not be empty."
            )

        if not self.model.strip():
            raise ValueError(
                "model must not be empty."
            )

        if (
            self.connect_timeout_seconds
            <= 0.0
        ):
            raise ValueError(
                "connect timeout must be positive."
            )

        if (
            self.read_timeout_seconds
            <= 0.0
        ):
            raise ValueError(
                "read timeout must be positive."
            )

        if (
            self.write_timeout_seconds
            <= 0.0
        ):
            raise ValueError(
                "write timeout must be positive."
            )

        if not (
            0.0
            <= self.temperature
            <= 2.0
        ):
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if not (
            0.0
            < self.top_p
            <= 1.0
        ):
            raise ValueError(
                "top_p must be greater than 0 "
                "and less than or equal to 1."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if self.repeat_penalty <= 0.0:
            raise ValueError(
                "repeat_penalty must be positive."
            )

        if self.context_window < 1024:
            raise ValueError(
                "context_window must be at least 1024."
            )

        if self.maximum_output_tokens < 1:
            raise ValueError(
                "maximum_output_tokens must be positive."
            )


def nanoseconds_to_seconds(
    value: object,
) -> float | None:
    """Convert an Ollama nanosecond duration to seconds."""

    if value is None:
        return None

    try:
        nanoseconds = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if nanoseconds < 0:
        return None

    return (
        nanoseconds
        / 1_000_000_000.0
    )


def optional_non_negative_integer(
    value: object,
) -> int | None:
    """Return a validated optional non-negative integer."""

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        converted = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if converted < 0:
        return None

    return converted


def extract_response_text(
    payload: dict[str, Any],
) -> str:
    """Extract assistant text from an Ollama chat response."""

    message = payload.get(
        "message"
    )

    if not isinstance(
        message,
        dict,
    ):
        raise NarrationResponseError(
            "Ollama response does not contain "
            "a valid message object."
        )

    content = message.get(
        "content"
    )

    if not isinstance(
        content,
        str,
    ):
        raise NarrationResponseError(
            "Ollama response message does not "
            "contain text content."
        )

    normalized = content.strip()

    if not normalized:
        raise NarrationResponseError(
            "Ollama returned an empty narration."
        )

    return normalized


class OllamaNarrationClient:
    """
    Local Ollama chat client for constrained narration.

    Raw output remains untrusted until Phase 14C validation.
    """

    def __init__(
        self,
        config: NarrationClientConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Create the narration client."""

        self.config = config

        self._owns_client = bool(
            http_client is None
        )

        if http_client is not None:
            self._client = http_client

        else:
            timeout = httpx.Timeout(
                connect=(
                    config
                    .connect_timeout_seconds
                ),
                read=(
                    config
                    .read_timeout_seconds
                ),
                write=(
                    config
                    .write_timeout_seconds
                ),
                pool=(
                    config
                    .connect_timeout_seconds
                ),
            )

            self._client = httpx.Client(
                base_url=(
                    config
                    .base_url
                    .rstrip("/")
                ),
                timeout=timeout,
                headers={
                    "Content-Type": (
                        "application/json"
                    ),
                },
            )

    def __enter__(
        self,
    ) -> "OllamaNarrationClient":
        """Enter context manager."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close owned HTTP resources."""

        self.close()

    def close(
        self,
    ) -> None:
        """Close the HTTP client when owned locally."""

        if self._owns_client:
            self._client.close()

    def build_request_payload(
        self,
        prompt: NarrationPrompt,
    ) -> dict[str, Any]:
        """
        Build the bounded Ollama chat request.

        The model receives separate system and user messages.
        """

        return {
            "model": (
                self.config.model
            ),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        prompt.system_prompt
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt.user_prompt
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "keep_alive": (
                self.config.keep_alive
            ),
            "options": {
                "temperature": (
                    self.config.temperature
                ),
                "top_p": (
                    self.config.top_p
                ),
                "top_k": (
                    self.config.top_k
                ),
                "repeat_penalty": (
                    self.config
                    .repeat_penalty
                ),
                "seed": (
                    self.config.seed
                ),
                "num_ctx": (
                    self.config
                    .context_window
                ),
                "num_predict": (
                    self.config
                    .maximum_output_tokens
                ),
            },
        }

    def generate(
        self,
        prompt: NarrationPrompt,
        *,
        attempt_number: int = 1,
        is_repair_attempt: bool = False,
    ) -> NarrationResponse:
        """
        Generate one raw narration response.

        The response is not safe for user display until Phase 14C
        validation passes.
        """

        if attempt_number < 1:
            raise ValueError(
                "attempt_number must be one-based."
            )

        payload = self.build_request_payload(
            prompt
        )

        """ print(
        "Narration runtime:",
        {
            "model": payload["model"],
            "num_ctx": payload["options"]["num_ctx"],
            "num_predict": payload["options"]["num_predict"],
            "keep_alive": payload["keep_alive"],
        },
) """

        request_started = (
            time.perf_counter()
        )

        try:
            response = self._client.post(
                "/api/chat",
                json=payload,
            )

        except httpx.ConnectError as exc:
            raise NarrationConnectionError(
                "Unable to connect to local Ollama at "
                f"{self.config.base_url}. "
                "Ensure Ollama is running."
            ) from exc

        except httpx.TimeoutException as exc:
            raise NarrationTimeoutError(
                "Ollama narration generation "
                "exceeded the configured timeout."
            ) from exc

        except httpx.HTTPError as exc:
            raise NarrationClientError(
                "Unexpected Ollama HTTP failure."
            ) from exc

        measured_duration = (
            time.perf_counter()
            - request_started
        )

        if not response.is_success:
            error_detail = (
                response.text.strip()
            )

            if not error_detail:
                error_detail = (
                    "No error body returned."
                )

            raise NarrationAPIError(
                "Ollama chat request failed "
                f"with HTTP {response.status_code}: "
                f"{error_detail}"
            )

        try:
            response_payload = (
                response.json()
            )

        except ValueError as exc:
            raise NarrationResponseError(
                "Ollama returned invalid JSON."
            ) from exc

        if not isinstance(
            response_payload,
            dict,
        ):
            raise NarrationResponseError(
                "Ollama response JSON must be an object."
            )

        narration_text = (
            extract_response_text(
                response_payload
            )
        )

        model_name = response_payload.get(
            "model"
        )

        if not isinstance(
            model_name,
            str,
        ):
            model_name = (
                self.config.model
            )

        model_name = model_name.strip()

        if not model_name:
            model_name = (
                self.config.model
            )

        reported_duration = (
            nanoseconds_to_seconds(
                response_payload.get(
                    "total_duration"
                )
            )
        )

        generation_duration = (
            reported_duration
            if reported_duration
            is not None
            else measured_duration
        )

        load_duration_seconds=(
            nanoseconds_to_seconds(
                response_payload.get(
                    "load_duration"
                )
            )
        ),

        prompt_eval_duration_seconds=(
            nanoseconds_to_seconds(
                response_payload.get(
                    "prompt_eval_duration"
                )
            )
        ),

        eval_duration_seconds=(
            nanoseconds_to_seconds(
                response_payload.get(
                    "eval_duration"
                )
            )
        ),

        return NarrationResponse(
            model=model_name,
            text=narration_text,
            attempt_number=(
                attempt_number
            ),
            is_repair_attempt=(
                is_repair_attempt
            ),
            prompt_tokens=(
                optional_non_negative_integer(
                    response_payload.get(
                        "prompt_eval_count"
                    )
                )
            ),
            completion_tokens=(
                optional_non_negative_integer(
                    response_payload.get(
                        "eval_count"
                    )
                )
            ),
            generation_duration_seconds=(
                generation_duration
            ),
        )

@property
def prompt_tokens_per_second(
    self,
) -> float | None:

    if (
        self.prompt_tokens is None
        or self.prompt_eval_duration_seconds
        is None
        or self.prompt_eval_duration_seconds
        <= 0.0
    ):
        return None

    return (
        self.prompt_tokens
        / self.prompt_eval_duration_seconds
    )


@property
def generation_tokens_per_second(
    self,
) -> float | None:

    if (
        self.completion_tokens is None
        or self.eval_duration_seconds
        is None
        or self.eval_duration_seconds
        <= 0.0
    ):
        return None

    return (
        self.completion_tokens
        / self.eval_duration_seconds
    )