from __future__ import annotations

import json

import httpx
import pytest

from src.embeddings import (
    EmbeddingClientConfig,
    EmbeddingResponseError,
    OllamaEmbeddingClient,
    batched,
)


def make_mock_client(
    handler: object,
) -> httpx.Client:
    transport = httpx.MockTransport(
        handler
    )

    return httpx.Client(
        transport=transport
    )


def test_batch_helper() -> None:
    batches = batched(
        ["a", "b", "c", "d", "e"],
        batch_size=2,
    )

    assert batches == (
        ("a", "b"),
        ("c", "d"),
        ("e",),
    )


def test_embedding_batch_response() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(
            request.content
        )

        assert payload["model"] == (
            "nomic-embed-text:latest"
        )

        assert payload["input"] == [
            "first",
            "second",
        ]

        return httpx.Response(
            status_code=200,
            json={
                "model": (
                    "nomic-embed-text:latest"
                ),
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                "prompt_eval_count": 4,
            },
        )

    client = make_mock_client(
        handler
    )

    embedding_client = OllamaEmbeddingClient(
        EmbeddingClientConfig(
            base_url=(
                "http://localhost:11434"
            ),
            model=(
                "nomic-embed-text:latest"
            ),
        ),
        http_client=client,
    )

    result = embedding_client.embed_texts(
        ["first", "second"]
    )

    assert result.input_count == 2
    assert result.dimension == 3
    assert len(result.embeddings) == 2


def test_empty_embedding_input_is_rejected() -> None:
    embedding_client = OllamaEmbeddingClient(
        EmbeddingClientConfig(
            base_url=(
                "http://localhost:11434"
            ),
            model=(
                "nomic-embed-text:latest"
            ),
        ),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    500
                )
            )
        ),
    )

    with pytest.raises(ValueError):
        embedding_client.embed_texts([])


def test_invalid_embedding_count_is_rejected() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2]
                ]
            },
        )

    embedding_client = OllamaEmbeddingClient(
        EmbeddingClientConfig(
            base_url=(
                "http://localhost:11434"
            ),
            model=(
                "nomic-embed-text:latest"
            ),
            max_retries=0,
        ),
        http_client=make_mock_client(
            handler
        ),
    )

    with pytest.raises(
        EmbeddingResponseError
    ):
        embedding_client.embed_texts(
            ["first", "second"]
        )


def test_inconsistent_dimensions_are_rejected() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4, 0.5],
                ]
            },
        )

    embedding_client = OllamaEmbeddingClient(
        EmbeddingClientConfig(
            base_url=(
                "http://localhost:11434"
            ),
            model=(
                "nomic-embed-text:latest"
            ),
            max_retries=0,
        ),
        http_client=make_mock_client(
            handler
        ),
    )

    with pytest.raises(
        EmbeddingResponseError
    ):
        embedding_client.embed_texts(
            ["first", "second"]
        )