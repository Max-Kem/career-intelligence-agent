"""Shared OpenAI embedding, Pinecone Documents API smoke, and retrieval helpers."""

from __future__ import annotations

import math
import json
import os
import hashlib
from pathlib import Path
from typing import Any, Callable

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PINECONE_INDEX = "career-market-knowledge"
VECTOR_FIELD = "chunk_vector"
SMOKE_NAMESPACE = "career-rag-smoke-test"
KNOWLEDGE_NAMESPACE = "career-market-knowledge-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
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


def run_knowledge_retrieval(
    query: str,
    *,
    live: bool = False,
    openai_api_key: str | None = None,
    pinecone_api_key: str | None = None,
    openai_client: Any = None,
    pinecone_client: Any = None,
    dense_vector_query: Callable[..., Any] | None = None,
    metadata_fallback: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Preview or perform read-only top-three semantic retrieval across imported sources."""
    if not isinstance(query, str) or not query.strip():
        raise SmokeTestError("Query must be a non-empty string.")
    query = query.strip()
    if not live:
        return {
            "mode": "dry_run",
            "index": PINECONE_INDEX,
            "namespace": KNOWLEDGE_NAMESPACE,
            "query": query,
            "top_k": 3,
            "matches": [],
            "message": "Dry run only; no credentials read and no network calls made.",
        }

    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if not openai_api_key or not openai_api_key.strip():
        raise SmokeTestError("OPENAI_API_KEY is missing; live retrieval was not started.")
    if not pinecone_api_key or not pinecone_api_key.strip():
        raise SmokeTestError("PINECONE_API_KEY is missing; live retrieval was not started.")

    if openai_client is None or pinecone_client is None or dense_vector_query is None:
        try:
            from pinecone import DenseVectorQuery
            openai_client = openai_client or create_openai_client(openai_api_key)
            pinecone_client = pinecone_client or create_pinecone_client(pinecone_api_key)
            dense_vector_query = dense_vector_query or DenseVectorQuery
        except SmokeTestError:
            raise
        except ImportError as exc:
            raise SmokeTestError("Pinecone SDK is missing. Install project requirements first.") from exc
        except Exception as exc:
            raise SmokeTestError(f"Retrieval client initialization failed ({type(exc).__name__}).") from None

    try:
        query_vector = embed_texts(openai_client, [query])[0]
        index = pinecone_client.index(PINECONE_INDEX)
        response = index.documents.search(
            namespace=KNOWLEDGE_NAMESPACE,
            top_k=3,
            score_by=[dense_vector_query(field=VECTOR_FIELD, values=query_vector)],
            include_fields=[
                "chunk_id", "chunk_text", "document_title", "publisher",
                "publication_year", "page_number", "source_url",
                "page_start", "page_end", "geographic_scope", "document_id",
            ],
        )
    except SmokeTestError:
        raise
    except Exception as exc:
        raise SmokeTestError(f"Live knowledge retrieval failed ({type(exc).__name__}); no credentials or response details shown.") from None

    fallback_lookup = metadata_fallback
    matches = []
    for match in (_get(response, "matches", []) or [])[:3]:
        fields = _get(match, "fields", None) or _get(match, "metadata", None) or {}

        def field(name: str, default: Any = None) -> Any:
            value = _get(match, name, None)
            return _get(fields, name, default) if value is None else value

        chunk_id = field("chunk_id") or _get(match, "id") or "Not provided"
        if fallback_lookup is None:
            fallback_lookup = _local_citation_metadata
        try:
            local = fallback_lookup(str(chunk_id)) if chunk_id != "Not provided" else {}
        except Exception:
            local = {}
        used_local_metadata = any(
            field(name) is None and local.get(name) is not None
            for name in ("page_number", "page_start", "publication_year", "document_title", "publisher", "source_url", "geographic_scope")
        )
        page_start, page_end = field("page_start"), field("page_end")
        page_number = field("page_number")
        if page_number is None and page_start is None:
            page_number, page_start, page_end = local.get("page_number"), local.get("page_start"), local.get("page_end")
        elif page_number is None:
            page_number = page_start if page_end in (None, page_start) else None
        page_number = _normalize_page(page_number)
        page_start, page_end = _normalize_page(page_start), _normalize_page(page_end)
        year = _normalize_publication_year(field("publication_year"))
        if year is None:
            year = _normalize_publication_year(local.get("publication_year"))
        text = field("chunk_text")
        matches.append({
            "score": _get(match, "score"),
            "chunk_id": chunk_id,
            "document_title": field("document_title") or local.get("document_title") or "Not provided",
            "publisher": field("publisher") or local.get("publisher") or "Not provided",
            "publication_year": year if year is not None else "Not provided",
            "page_number": page_number if page_number is not None else "Not provided",
            "page_start": page_start,
            "page_end": page_end,
            "source_url": field("source_url") or local.get("source_url") or "Not provided",
            "geographic_scope": field("geographic_scope") or local.get("geographic_scope") or "Not provided",
            "citation_metadata_source": "pinecone_with_local_manifest_fallback" if used_local_metadata else "pinecone",
            "excerpt": (text[:360].rstrip() + "…") if isinstance(text, str) and len(text) > 360 else text if isinstance(text, str) and text else "Text unavailable",
        })
    return {
        "mode": "live",
        "index": PINECONE_INDEX,
        "namespace": KNOWLEDGE_NAMESPACE,
        "query": query,
        "top_k": 3,
        "matches": matches,
    }


def _normalize_publication_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2200 else None
    if isinstance(value, float) and value.is_integer():
        year = int(value)
        return year if 1900 <= year <= 2200 else None
    if isinstance(value, str) and value.strip().isdigit():
        year = int(value.strip())
        return year if 1900 <= year <= 2200 else None
    return None


def _normalize_page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
        return int(value.strip())
    return None


def _local_citation_metadata(chunk_id: str) -> dict[str, Any]:
    """Resolve citation metadata only through a verified document/chunk manifest entry."""
    root = REPO_ROOT / "knowledge_base"
    try:
        manifest = json.loads((root / "processed" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    documents = {row.get("document_id"): row for row in manifest.get("documents", []) if isinstance(row, dict)}
    batch_records: dict[str, dict[str, Any]] = {}
    batch_path = root / "review" / "rag_batch_manifest.json"
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        canonical = json.dumps({key: value for key, value in batch.items() if key != "manifest_sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() == batch.get("manifest_sha256"):
            batch_records = {row.get("record_id"): row for row in batch.get("records", []) if isinstance(row, dict)}
    except (OSError, json.JSONDecodeError):
        pass
    record = batch_records.get(chunk_id)
    document_id = record.get("document_id") if record else None
    if record and document_id in documents:
        try:
            source_path = root / "processed" / f"{document_id}.chunks.jsonl"
            with source_path.open(encoding="utf-8") as stream:
                chunks = {row.get("chunk_id"): row for row in (json.loads(line) for line in stream)}
            fingerprints = {row.get("chunk_id"): row.get("text_sha256") for row in record.get("source_text_fingerprints", []) if isinstance(row, dict)}
            source_ids = record.get("source_chunk_ids", [])
            source_rows = [chunks.get(source_id) for source_id in source_ids]
            valid_sources = bool(source_ids) and all(
                row is not None
                and row.get("document_id") == document_id
                and isinstance(row.get("page_number"), int)
                and hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest() == fingerprints.get(row.get("chunk_id"))
                for row in source_rows
            )
            pages = [row["page_number"] for row in source_rows if row is not None]
            valid_page_range = (
                valid_sources and pages
                and min(pages) == record.get("page_start")
                and max(pages) == record.get("page_end")
            )
            if not valid_page_range:
                record, document_id = None, None
        except (OSError, json.JSONDecodeError):
            record, document_id = None, None
    page_start = record.get("page_start") if record else None
    page_end = record.get("page_end") if record else None
    page_number = page_start if record and page_start == page_end else None
    document = documents.get(document_id) if document_id else None
    if document is None:
        for candidate_id, candidate_document in documents.items():
            path = root / "processed" / f"{candidate_id}.chunks.jsonl"
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        chunk = json.loads(line)
                        if chunk.get("chunk_id") == chunk_id:
                            page = chunk.get("page_number")
                            if isinstance(page, int) and page > 0:
                                document_id, document, page_number = candidate_id, candidate_document, page
                            break
            except (OSError, json.JSONDecodeError):
                continue
            if document is not None:
                break
    if not document or not isinstance(document.get("publication_year"), int):
        return {}
    geographic_scope = record.get("geographic_scope") if record else None
    if geographic_scope not in {"Germany_focused", "international_global"}:
        try:
            refined = json.loads((root / "review" / "refined_rag_selection.json").read_text(encoding="utf-8"))
            geographic_scope = refined.get("document_results", {}).get(document_id, {}).get("geographic_scope")
        except (OSError, json.JSONDecodeError):
            geographic_scope = None
    if geographic_scope not in {"Germany_focused", "international_global"}:
        geographic_scope = None
    return {
        "document_id": document_id,
        "document_title": document.get("title"),
        "publisher": document.get("publisher"),
        "publication_year": document.get("publication_year"),
        "source_url": document.get("source_url") if document.get("source_url_status") == "verified" else None,
        "page_number": page_number,
        "page_start": page_start,
        "page_end": page_end,
        "geographic_scope": geographic_scope,
    }
