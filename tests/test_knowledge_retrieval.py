from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json

import pytest

from career_agent import vector_store


class FakeEmbeddings:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeDocuments:
    def __init__(self, matches=None, error=None):
        self.matches = matches if matches is not None else []
        self.error = error
        self.search_call = None

    def search(self, **kwargs):
        self.search_call = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(matches=self.matches)


class FakePinecone:
    def __init__(self, documents):
        self.documents = documents
        self.index_name = None

    def index(self, name):
        self.index_name = name
        return SimpleNamespace(documents=self.documents)


def embedding_response(vector):
    return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=vector)])


def test_dry_run_has_no_credentials_or_client_calls(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"dry-run attempted external client access: {name}")

    result = vector_store.run_knowledge_retrieval(
        "How can AI change occupations?", openai_client=Forbidden(), pinecone_client=Forbidden(),
    )
    assert result["mode"] == "dry_run"
    assert result["query"] == "How can AI change occupations?"
    assert result["top_k"] == 3
    assert result["namespace"] == vector_store.KNOWLEDGE_NAMESPACE
    assert result["matches"] == []


def test_live_retrieval_uses_shared_embedding_and_documents_api_read_only():
    embedding = [0.25] * vector_store.EMBEDDING_DIMENSION
    embeddings = FakeEmbeddings(response=embedding_response(embedding))
    documents = FakeDocuments(matches=[SimpleNamespace(
        id=f"ilo-chunk-{number}", score=0.84 - number / 100,
        chunk_id=f"ilo-chunk-{number}", document_title="Generative AI and Jobs",
        publisher="International Labour Organization", publication_year=2025,
        page_number=50, source_url="https://example.org/ilo.pdf",
        chunk_text="This is a supported excerpt.",
    ) for number in range(1, 5)])
    pinecone = FakePinecone(documents)

    class DenseVectorQuery:
        def __init__(self, *, field, values):
            self.field, self.values = field, values

    result = vector_store.run_knowledge_retrieval(
        "Why does exposure not equal replacement?",
        live=True,
        openai_api_key="mock-openai-key",
        pinecone_api_key="mock-pinecone-key",
        openai_client=SimpleNamespace(embeddings=embeddings),
        pinecone_client=pinecone,
        dense_vector_query=DenseVectorQuery,
    )

    assert embeddings.calls == [{
        "model": "text-embedding-3-small",
        "input": ["Why does exposure not equal replacement?"],
        "dimensions": 1536,
    }]
    assert pinecone.index_name == "career-market-knowledge"
    assert documents.search_call["namespace"] == "career-market-knowledge-v1"
    assert documents.search_call["top_k"] == 3
    assert documents.search_call["score_by"][0].field == "chunk_vector"
    assert documents.search_call["score_by"][0].values == embedding
    assert "upsert" not in documents.search_call
    assert result["matches"][0]["page_number"] == 50
    assert result["matches"][0]["excerpt"] == "This is a supported excerpt."
    assert result["matches"][0]["source_passage"] == "This is a supported excerpt."
    assert result["matches"][0]["source_passage_complete"] is True
    assert result["matches"][0]["source_passage_truncated"] is False
    assert len(result["matches"]) == 3


def test_retrieval_handles_empty_results_and_missing_metadata():
    embedding = [0.1] * 1536
    no_results = vector_store.run_knowledge_retrieval(
        "question", live=True, openai_api_key="key", pinecone_api_key="key",
        openai_client=SimpleNamespace(embeddings=FakeEmbeddings(response=embedding_response(embedding))),
        pinecone_client=FakePinecone(FakeDocuments(matches=[])),
        dense_vector_query=lambda **kwargs: kwargs,
    )
    assert no_results["matches"] == []

    missing = vector_store.run_knowledge_retrieval(
        "question", live=True, openai_api_key="key", pinecone_api_key="key",
        openai_client=SimpleNamespace(embeddings=FakeEmbeddings(response=embedding_response(embedding))),
        pinecone_client=FakePinecone(FakeDocuments(matches=[SimpleNamespace(score=0.2)])),
        dense_vector_query=lambda **kwargs: kwargs,
    )
    assert missing["matches"][0]["chunk_id"] == "Not provided"
    assert missing["matches"][0]["source_url"] == "Not provided"
    assert missing["matches"][0]["excerpt"] == "Text unavailable"


def test_retrieval_normalizes_stored_page_range_and_float_publication_year():
    embedding = [0.1] * 1536
    match = SimpleNamespace(id="chunk-2025", score=0.77, fields={
        "chunk_id": "chunk-2025", "chunk_text": "Supported source text.",
        "document_title": "Skills forecast", "publisher": "Cedefop",
        "publication_year": 2025.0, "page_start": 9.0, "page_end": 9.0,
        "source_url": "https://example.org/source.pdf", "geographic_scope": "Germany_focused",
    })
    result = vector_store.run_knowledge_retrieval(
        "question", live=True, openai_api_key="key", pinecone_api_key="key",
        openai_client=SimpleNamespace(embeddings=FakeEmbeddings(response=embedding_response(embedding))),
        pinecone_client=FakePinecone(FakeDocuments(matches=[match])),
        dense_vector_query=lambda **kwargs: kwargs,
    )
    item = result["matches"][0]
    assert item["publication_year"] == 2025 and isinstance(item["publication_year"], int)
    assert item["page_number"] == 9
    assert isinstance(item["page_number"], int)
    assert item["geographic_scope"] == "Germany_focused"


def test_retrieval_marks_overlong_source_passage_incomplete_instead_of_presenting_it_as_full():
    embedding = [0.1] * 1536
    text = "x" * 3100
    result = vector_store.run_knowledge_retrieval(
        "question", live=True, openai_api_key="key", pinecone_api_key="key",
        openai_client=SimpleNamespace(embeddings=FakeEmbeddings(response=embedding_response(embedding))),
        pinecone_client=FakePinecone(FakeDocuments(matches=[SimpleNamespace(
            id="long", chunk_id="long", chunk_text=text,
            document_title="Skills", source_url="https://example.org/skills",
        )])), dense_vector_query=lambda **kwargs: kwargs,
    )
    item = result["matches"][0]
    assert len(item["source_passage"]) == 3000
    assert item["source_passage_complete"] is False
    assert item["source_passage_truncated"] is True


def test_retrieval_uses_local_metadata_fallback_for_missing_page_and_year():
    embedding = [0.1] * 1536
    result = vector_store.run_knowledge_retrieval(
        "question", live=True, openai_api_key="key", pinecone_api_key="key",
        openai_client=SimpleNamespace(embeddings=FakeEmbeddings(response=embedding_response(embedding))),
        pinecone_client=FakePinecone(FakeDocuments(matches=[SimpleNamespace(id="known-chunk", chunk_id="known-chunk")])),
        dense_vector_query=lambda **kwargs: kwargs,
        metadata_fallback=lambda chunk_id: {
            "publication_year": 2026, "page_number": 21,
            "document_title": "Germany digital report", "publisher": "European Commission",
            "source_url": "https://example.org/verified", "geographic_scope": "Germany_focused",
        } if chunk_id == "known-chunk" else {},
    )
    item = result["matches"][0]
    assert item["page_number"] == 21
    assert item["publication_year"] == 2026
    assert item["source_url"] == "https://example.org/verified"
    assert item["citation_metadata_source"] == "pinecone_with_local_manifest_fallback"


def test_local_manifest_metadata_fallback_validates_record_and_source_chunk(tmp_path, monkeypatch):
    root = tmp_path / "knowledge_base"
    processed, review = root / "processed", root / "review"
    processed.mkdir(parents=True)
    review.mkdir()
    document_id = "eu_digital_decade_germany_2026"
    text = "Local page text used only to validate source provenance."
    chunk_id = f"{document_id}-p0021-c0001-aaaaaaaaaaaa"
    chunk = {"chunk_id": chunk_id, "document_id": document_id, "page_number": 21, "text": text}
    (processed / f"{document_id}.chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    (processed / "manifest.json").write_text(json.dumps({"documents": [{
        "document_id": document_id, "title": "Germany Digital Decade", "publisher": "European Commission",
        "publication_year": 2026, "source_url": "https://example.org/report", "source_url_status": "verified",
    }]}), encoding="utf-8")
    batch = {
        "records": [{
            "record_id": chunk_id, "document_id": document_id, "source_chunk_ids": [chunk_id],
            "source_text_fingerprints": [{"chunk_id": chunk_id, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}],
            "page_start": 21, "page_end": 21, "geographic_scope": "Germany_focused",
        }],
    }
    canonical = json.dumps(batch, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    batch["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    (review / "rag_batch_manifest.json").write_text(json.dumps(batch), encoding="utf-8")
    monkeypatch.setattr(vector_store, "REPO_ROOT", tmp_path)

    metadata = vector_store._local_citation_metadata(chunk_id)
    assert metadata["page_number"] == 21
    assert metadata["publication_year"] == 2026
    assert metadata["source_url"] == "https://example.org/report"

    chunk["text"] = "Changed text must invalidate provenance."
    (processed / f"{document_id}.chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    changed = vector_store._local_citation_metadata(chunk_id)
    assert changed["page_number"] == 21  # Direct source chunk still verifies the page.
    assert changed["geographic_scope"] is None  # Batch-only scope is discarded when its fingerprint is stale.


def test_live_retrieval_requires_credentials_and_nonempty_question(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(vector_store.SmokeTestError, match="OPENAI_API_KEY"):
        vector_store.run_knowledge_retrieval("question", live=True)
    with pytest.raises(vector_store.SmokeTestError, match="non-empty"):
        vector_store.run_knowledge_retrieval("  ")


@pytest.mark.parametrize("failed_service", ["openai", "pinecone"])
def test_live_api_failures_are_sanitized(failed_service):
    embedding = [0.1] * 1536
    embeddings = FakeEmbeddings(
        response=embedding_response(embedding),
        error=RuntimeError("private response content") if failed_service == "openai" else None,
    )
    docs = FakeDocuments(error=RuntimeError("private response content") if failed_service == "pinecone" else None)
    with pytest.raises(vector_store.SmokeTestError, match="Live knowledge retrieval failed") as error:
        vector_store.run_knowledge_retrieval(
            "question", live=True, openai_api_key="mock-key", pinecone_api_key="mock-key",
            openai_client=SimpleNamespace(embeddings=embeddings),
            pinecone_client=FakePinecone(docs),
            dense_vector_query=lambda **kwargs: kwargs,
        )
    assert "private response content" not in str(error.value)
