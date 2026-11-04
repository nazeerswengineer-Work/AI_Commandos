"""
Ollama embedding client for REQTRACE AI.

Uses the current Ollama /api/embed endpoint with batch input.

This module:
- generates embeddings only
- validates response shape and dimensions
- implements bounded retry behavior
- does not access ChromaDB
- does not perform retrieval
- does not invoke a generation model
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated reliably."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when Ollama returns an invalid embedding response."""


@dataclass(frozen=True)
class EmbeddingBatchResult:
    """Validated result from one Ollama embedding request."""

    model: str
    embeddings: tuple[tuple[float, ...], ...]
    dimension: int
    input_count: int
    prompt_eval_count: int | None = None
    total_duration_ns: int | None = None

    def __post_init__(self) -> None:
        if self.input_count < 1:
            raise ValueError(
                "Embedding batch must contain at least one input."
            )

        if self.dimension < 1:
            raise ValueError(
                "Embedding dimension must be positive."
            )

        if len(self.embeddings) != self.input_count:
            raise ValueError(
                "Embedding count does not match input count."
            )

        for embedding in self.embeddings:
            if len(embedding) != self.dimension:
                raise ValueError(
                    "Embedding dimensions are inconsistent."
                )


@dataclass(frozen=True)
class EmbeddingClientConfig:
    """Configuration for the local Ollama embedding client."""

    base_url: str
    model: str
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 120
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    truncate: bool = True
    keep_alive: str = "5m"

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError(
                "Ollama base URL must not be empty."
            )

        if not self.model.strip():
            raise ValueError(
                "Embedding model must not be empty."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries must not be negative."
            )


class OllamaEmbeddingClient:
    """Small synchronous client for Ollama embeddings."""

    def __init__(
        self,
        config: EmbeddingClientConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._owns_client = http_client is None

        if http_client is not None:
            self._client = http_client
        else:
            timeout = httpx.Timeout(
                connect=config.connect_timeout_seconds,
                read=config.read_timeout_seconds,
                write=config.read_timeout_seconds,
                pool=config.connect_timeout_seconds,
            )

            self._client = httpx.Client(
                timeout=timeout
            )

    @property
    def endpoint(self) -> str:
        """Return the modern Ollama embedding endpoint."""

        return (
            f"{self.config.base_url.rstrip('/')}"
            "/api/embed"
        )

    def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaEmbeddingClient":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def embed_texts(
        self,
        texts: list[str] | tuple[str, ...],
    ) -> EmbeddingBatchResult:
        """Generate one validated batch of embeddings."""

        normalized_texts = [
            str(text).strip()
            for text in texts
        ]

        if not normalized_texts:
            raise ValueError(
                "At least one text is required."
            )

        if any(
            not text
            for text in normalized_texts
        ):
            raise ValueError(
                "Embedding inputs must not be empty."
            )

        payload = {
            "model": self.config.model,
            "input": normalized_texts,
            "truncate": self.config.truncate,
            "keep_alive": self.config.keep_alive,
        }

        response: httpx.Response | None = None
        final_error: Exception | None = None

        for attempt in range(
            self.config.max_retries + 1
        ):
            try:
                response = self._client.post(
                    self.endpoint,
                    json=payload,
                )

                response.raise_for_status()
                break

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
            ) as exc:
                final_error = exc

                if attempt >= self.config.max_retries:
                    raise EmbeddingError(
                        "Ollama embedding request failed "
                        "after retries."
                    ) from exc

                time.sleep(
                    self.config.retry_delay_seconds
                )

            except httpx.HTTPStatusError as exc:
                status_code = (
                    exc.response.status_code
                )

                if (
                    status_code >= 500
                    and attempt
                    < self.config.max_retries
                ):
                    final_error = exc
                    time.sleep(
                        self.config.retry_delay_seconds
                    )
                    continue

                raise EmbeddingError(
                    "Ollama embedding request returned "
                    f"HTTP {status_code}: "
                    f"{exc.response.text[:500]}"
                ) from exc

            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"Ollama embedding request failed: {exc}"
                ) from exc

        if response is None:
            raise EmbeddingError(
                "No Ollama response was received."
            ) from final_error

        try:
            payload_data: dict[str, Any] = (
                response.json()
            )
        except ValueError as exc:
            raise EmbeddingResponseError(
                "Ollama returned invalid JSON."
            ) from exc

        return self._validate_response(
            payload_data,
            expected_count=len(normalized_texts),
        )

    def embed_text(
        self,
        text: str,
    ) -> tuple[float, ...]:
        """Embed one text and return one vector."""

        result = self.embed_texts([text])
        return result.embeddings[0]

    def _validate_response(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
    ) -> EmbeddingBatchResult:
        """Validate Ollama's response shape."""

        raw_embeddings = payload.get(
            "embeddings"
        )

        if not isinstance(
            raw_embeddings,
            list,
        ):
            raise EmbeddingResponseError(
                "Ollama response does not contain "
                "an embeddings array."
            )

        if len(raw_embeddings) != expected_count:
            raise EmbeddingResponseError(
                "Ollama embedding count does not match "
                "the number of requested inputs."
            )

        converted: list[tuple[float, ...]] = []

        for index, raw_embedding in enumerate(
            raw_embeddings
        ):
            if (
                not isinstance(raw_embedding, list)
                or not raw_embedding
            ):
                raise EmbeddingResponseError(
                    f"Embedding {index} is empty or invalid."
                )

            try:
                embedding = tuple(
                    float(value)
                    for value in raw_embedding
                )
            except (TypeError, ValueError) as exc:
                raise EmbeddingResponseError(
                    f"Embedding {index} contains "
                    "a non-numeric value."
                ) from exc

            converted.append(embedding)

        dimension = len(converted[0])

        if any(
            len(embedding) != dimension
            for embedding in converted
        ):
            raise EmbeddingResponseError(
                "Ollama returned inconsistent "
                "embedding dimensions."
            )

        response_model = payload.get(
            "model",
            self.config.model,
        )

        return EmbeddingBatchResult(
            model=str(response_model),
            embeddings=tuple(converted),
            dimension=dimension,
            input_count=expected_count,
            prompt_eval_count=(
                payload.get("prompt_eval_count")
                if isinstance(
                    payload.get("prompt_eval_count"),
                    int,
                )
                else None
            ),
            total_duration_ns=(
                payload.get("total_duration")
                if isinstance(
                    payload.get("total_duration"),
                    int,
                )
                else None
            ),
        )


def batched(
    values: list[str],
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Split strings into deterministic batches."""

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    return tuple(
        tuple(
            values[index:index + batch_size]
        )
        for index in range(
            0,
            len(values),
            batch_size,
        )
    )