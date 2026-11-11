"""Minimal safe OpenAI embedding and Pinecone Documents API smoke workflow."""

from __future__ import annotations

import math
import os
from typing import Any, Callable

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PINECONE_INDEX = "career-market-knowledge"
VECTOR_FIELD = "chunk_vector"
SMOKE_NAMESPACE = "career-rag-smoke-test"
KNOWLEDGE_NAMESPACE = "career-market-knowledge-v1"
QUERY_TEXT = "Which skills and tasks are relevant to a junior data analyst?"

# Original fictional examples written for this smoke test; never source PDF content.
SYNTHETIC_RECORDS: tuple[dict[str, str], ...] = (
    {
        "_id": "smoke-career-analyst-001",
        "chunk_text": "A junior data analyst often works with SQL, Python and data visualization tools.",
        "document_title": "Synthetic career example: junior data analyst",
        "source_type": "synthetic_test_fixture",
        "rights_status": "original_synthetic_text",
    },
    {
        "_id": "smoke-career-ai-consultant-002",
        "chunk_text": "An AI transformation consultant helps organizations identify and implement AI use cases.",
        "document_title": "Synthetic career example: AI transformation consultant",
        "source_type": "synthetic_test_fixture",
        "rights_status": "original_synthetic_text",
    },
    {
        "_id": "smoke-career-project-manager-003",
        "chunk_text": "A project manager coordinates project timelines, stakeholders and delivery risks.",
        "document_title": "Synthetic career example: project manager",
        "source_type": "synthetic_test_fixture",
        "rights_status": "original_synthetic_text",
    },
)


class SmokeTestError(RuntimeError):
    """Configuration, validation, or external API error without credentials."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _validate_embedding(embedding: Any, *, expected_count: int = 1) -> list[list[float]]:
    if not isinstance(embedding, list) or len(embedding) != expected_count:
        raise SmokeTestError("Embedding response did not contain the expected number of vectors.")
    vectors: list[list[float]] = []
    for vector in embedding:
        if not isinstance(vector, (list, tuple)) or len(vector) != EMBEDDING_DIMENSION:
            raise SmokeTestError(f"Embedding dimension must be exactly {EMBEDDING_DIMENSION}.")
        try:
            converted = [float(value) for value in vector]
        except (TypeError, ValueError):
            raise SmokeTestError("Embedding contains a non-numeric value.") from None
        if any(not math.isfinite(value) for value in converted):
            raise SmokeTestError("Embedding contains a non-finite value.")
        vectors.append(converted)
    return vectors


def _response_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    data = _get(response, "data")
    if not isinstance(data, (list, tuple)) or len(data) != expected_count:
        raise SmokeTestError("OpenAI returned an invalid embedding response.")
    # Preserve input order even if an API response arrives with explicit indices.
    ordered = sorted(data, key=lambda item: _get(item, "index", 0))
    return _validate_embedding([_get(item, "embedding") for item in ordered], expected_count=expected_count)


def _make_clients(openai_api_key: str, pinecone_api_key: str) -> tuple[Any, Any, Callable[..., Any]]:
    try:
        from openai import OpenAI
        from pinecone import DenseVectorQuery, Pinecone
    except ImportError as exc:
        raise SmokeTestError("OpenAI or Pinecone SDK is missing. Install project requirements first.") from exc
    return OpenAI(api_key=openai_api_key), Pinecone(api_key=pinecone_api_key), DenseVectorQuery


def create_openai_client(api_key: str) -> Any:
    """Construct the existing OpenAI SDK client without exposing its key."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SmokeTestError("OpenAI SDK is missing. Install project requirements first.") from exc
    return OpenAI(api_key=api_key)


def create_pinecone_client(api_key: str) -> Any:
    """Construct the Pinecone SDK v10 client without touching index settings."""
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise SmokeTestError("Pinecone SDK is missing. Install project requirements first.") from exc
    return Pinecone(api_key=api_key)


def embed_texts(client: Any, texts: list[str]) -> list[list[float]]:
    """Create and dimension-check embeddings with the shared model settings."""
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise SmokeTestError("Embedding input must contain non-empty strings.")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSION,
    )
    return _response_embeddings(response, len(texts))


def run_smoke_test(
    *,
    live: bool = False,
    openai_api_key: str | None = None,
    pinecone_api_key: str | None = None,
    openai_client: Any = None,
    pinecone_client: Any = None,
    dense_vector_query: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Preview offline or run a bounded synthetic-text live smoke test."""
    if not live:
        return {
            "mode": "dry_run",
            "namespace": SMOKE_NAMESPACE,
            "record_count": len(SYNTHETIC_RECORDS),
            "records": [dict(record) for record in SYNTHETIC_RECORDS],
            "query": QUERY_TEXT,
            "message": "Dry run only; no credentials read and no network calls made.",
        }

    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if not openai_api_key or not openai_api_key.strip():
        raise SmokeTestError("OPENAI_API_KEY is missing; live smoke test was not started.")
    if not pinecone_api_key or not pinecone_api_key.strip():
        raise SmokeTestError("PINECONE_API_KEY is missing; live smoke test was not started.")

    if openai_client is None or pinecone_client is None or dense_vector_query is None:
        created_openai, created_pinecone, created_query = _make_clients(openai_api_key, pinecone_api_key)
        openai_client = openai_client or created_openai
        pinecone_client = pinecone_client or created_pinecone
        dense_vector_query = dense_vector_query or created_query

    try:
        source_texts = [record["chunk_text"] for record in SYNTHETIC_RECORDS]
        document_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=source_texts,
            dimensions=EMBEDDING_DIMENSION,
        )
        document_vectors = _response_embeddings(document_response, len(SYNTHETIC_RECORDS))
        query_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=QUERY_TEXT,
            dimensions=EMBEDDING_DIMENSION,
        )
        query_vector = _response_embeddings(query_response, 1)[0]

        index = pinecone_client.index(PINECONE_INDEX)
        documents = []
        for record, vector in zip(SYNTHETIC_RECORDS, document_vectors, strict=True):
            documents.append({**record, VECTOR_FIELD: vector})
        upsert_result = index.documents.upsert(namespace=SMOKE_NAMESPACE, documents=documents)
        search_result = index.documents.search(
            namespace=SMOKE_NAMESPACE,
            top_k=len(SYNTHETIC_RECORDS),
            score_by=[dense_vector_query(field=VECTOR_FIELD, values=query_vector)],
            include_fields=["chunk_text", "document_title", "source_type", "rights_status"],
        )
    except SmokeTestError:
        raise
    except Exception as exc:
        raise SmokeTestError(f"Live smoke test API operation failed ({type(exc).__name__}).") from None

    matches = []
    for match in (_get(search_result, "matches", []) or []):
        matches.append({
            "id": _get(match, "id"),
            "score": _get(match, "score"),
            "chunk_text": _get(match, "chunk_text", ""),
            "document_title": _get(match, "document_title", ""),
        })
    return {
        "mode": "live",
        "namespace": SMOKE_NAMESPACE,
        "upserted_count": _get(upsert_result, "upserted_count", len(documents)),
        "query": QUERY_TEXT,
        "matches": matches,
    }
