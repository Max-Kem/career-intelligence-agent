from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from career_agent import vector_store as store


class FakeEmbeddings:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeDocuments:
    def __init__(self):
        self.upsert_call = None
        self.search_call = None

    def upsert(self, **kwargs):
        self.upsert_call = kwargs
        return SimpleNamespace(upserted_count=len(kwargs["documents"]))

    def search(self, **kwargs):
        self.search_call = kwargs
        return SimpleNamespace(matches=[SimpleNamespace(
            id="smoke-career-analyst-001",
            score=0.91,
            chunk_text=store.SYNTHETIC_RECORDS[0]["chunk_text"],
            document_title=store.SYNTHETIC_RECORDS[0]["document_title"],
        )])


class FakePinecone:
    def __init__(self):
        self.requested_index = None
        self.documents = FakeDocuments()

    def index(self, name):
        self.requested_index = name
        return SimpleNamespace(documents=self.documents)


def embedding_response(vectors):
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)])


def test_default_dry_run_uses_only_fictional_records_and_makes_no_calls():
    class ForbiddenClient:
        def __getattr__(self, _name):
            raise AssertionError("dry run must not call any external client")

    result = store.run_smoke_test(openai_client=ForbiddenClient(), pinecone_client=ForbiddenClient())
    assert result["mode"] == "dry_run"
    assert result["namespace"] == "career-rag-smoke-test"
    assert result["record_count"] == 3
    assert all(record["rights_status"] == "original_synthetic_text" for record in result["records"])
    assert "no network calls" in result["message"]


def test_live_mode_requires_both_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(store.SmokeTestError, match="OPENAI_API_KEY"):
        store.run_smoke_test(live=True)
    with pytest.raises(store.SmokeTestError, match="PINECONE_API_KEY"):
        store.run_smoke_test(live=True, openai_api_key="test-key")


def test_live_workflow_uses_openai_dimensions_documents_api_namespace_and_vector_field():
    embeddings = FakeEmbeddings([
        embedding_response([[0.1] * store.EMBEDDING_DIMENSION for _ in store.SYNTHETIC_RECORDS]),
        embedding_response([[0.2] * store.EMBEDDING_DIMENSION]),
    ])
    openai_client = SimpleNamespace(embeddings=embeddings)
    pinecone_client = FakePinecone()

    class DenseVectorQuery:
        def __init__(self, *, field, values):
            self.field = field
            self.values = values

    result = store.run_smoke_test(
        live=True,
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        dense_vector_query=DenseVectorQuery,
    )

    assert [call["model"] for call in embeddings.calls] == [store.EMBEDDING_MODEL] * 2
    assert all(call["dimensions"] == 1536 for call in embeddings.calls)
    assert pinecone_client.requested_index == "career-market-knowledge"
    upsert = pinecone_client.documents.upsert_call
    assert upsert["namespace"] == "career-rag-smoke-test"
    assert len(upsert["documents"]) == 3
    assert upsert["documents"][0]["chunk_vector"] == [0.1] * 1536
    search = pinecone_client.documents.search_call
    assert search["namespace"] == "career-rag-smoke-test"
    assert search["score_by"][0].field == "chunk_vector"
    assert search["score_by"][0].values == [0.2] * 1536
    assert result["matches"][0]["chunk_text"] == store.SYNTHETIC_RECORDS[0]["chunk_text"]


def test_invalid_embedding_dimension_fails_before_pinecone_upsert():
    embeddings = FakeEmbeddings([
        embedding_response([[0.1] * 10 for _ in store.SYNTHETIC_RECORDS]),
    ])
    pinecone_client = FakePinecone()
    with pytest.raises(store.SmokeTestError, match="dimension must be exactly 1536"):
        store.run_smoke_test(
            live=True,
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            openai_client=SimpleNamespace(embeddings=embeddings),
            pinecone_client=pinecone_client,
            dense_vector_query=lambda **kwargs: kwargs,
        )
    assert pinecone_client.documents.upsert_call is None


@pytest.mark.parametrize("failing_service", ["openai", "pinecone"])
def test_api_failures_are_reported_without_sensitive_exception_text(failing_service):
    valid_documents = embedding_response([[0.1] * 1536 for _ in store.SYNTHETIC_RECORDS])
    valid_query = embedding_response([[0.2] * 1536])
    embeddings = FakeEmbeddings([
        RuntimeError("secret-looking detail") if failing_service == "openai" else valid_documents,
        valid_query,
    ])
    pinecone_client = FakePinecone()
    if failing_service == "pinecone":
        class BrokenPinecone:
            def index(self, _name):
                raise RuntimeError("secret-looking detail")
        pinecone_client = BrokenPinecone()
    with pytest.raises(store.SmokeTestError, match="API operation failed") as error:
        store.run_smoke_test(
            live=True,
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            openai_client=SimpleNamespace(embeddings=embeddings),
            pinecone_client=pinecone_client,
            dense_vector_query=lambda **kwargs: kwargs,
        )
    assert "secret-looking detail" not in str(error.value)
