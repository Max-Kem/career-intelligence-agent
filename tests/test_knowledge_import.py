from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from career_agent import knowledge_import as importer


@pytest.fixture
def local_source(tmp_path, monkeypatch):
    ids = ("ilo-p0004-c0001-test", "ilo-p0012-c0001-test")
    monkeypatch.setattr(importer, "PILOT_CHUNK_IDS", ids)
    processed = tmp_path / "processed"
    processed.mkdir()
    document = {
        "document_id": importer.ILO_DOCUMENT_ID,
        "title": "Generative AI and Jobs: A Refined Global Index of Occupational Exposure",
        "publisher": "International Labour Organization",
        "publication_year": 2025,
        "source_url": "https://example.org/ilo-report.pdf",
        "filename": "ilo_report.pdf",
        "page_count": 76,
        "licensing_status": "verified_in_pdf",
        "license": "CC BY 4.0",
        "rights_preflight": {"document_notice_status": "confirmed_in_pdf", "document_notice_page": 2},
    }
    (processed / "manifest.json").write_text(json.dumps({"documents": [document]}), encoding="utf-8")
    source_chunks = [
        {"chunk_id": ids[0], "document_id": importer.ILO_DOCUMENT_ID, "page_number": 4, "text": "ILO copyright covered analytical summary about occupational exposure."},
        {"chunk_id": ids[1], "document_id": importer.ILO_DOCUMENT_ID, "page_number": 12, "text": "ILO copyright covered introduction on labor-market implications."},
    ]
    (processed / importer.ILO_CHUNK_FILE).write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in source_chunks), encoding="utf-8",
    )
    review_dir = tmp_path / "review"
    approval_path, summary_path, count = importer.prepare_ilo_review(processed, review_dir)
    assert count == len(ids)
    return processed, approval_path, summary_path


def _approve(approval_path: Path, chunk_id: str | None = None) -> None:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    for entry in approval["candidates"]:
        if chunk_id is None or entry["chunk_id"] == chunk_id:
            entry.update({
                "approval_status": "approved",
                "approved": True,
                "reviewer": "human reviewer",
                "approved_at": "2026-11-10",
                "third_party_review": "cleared",
            })
    approval_path.write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")


def test_review_preparation_marks_all_candidates_pending_and_writes_summary(local_source):
    _processed, approval_path, summary_path = local_source
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    assert len(approval["candidates"]) == 2
    assert all(entry["approved"] is False and entry["approval_status"] == "pending_review" for entry in approval["candidates"])
    summary = summary_path.read_text(encoding="utf-8")
    assert "All entries are pending review" in summary
    assert "third-party content is excluded" in summary.casefold()
    assert "occupational exposure" in summary


def test_dry_run_reports_pending_as_rejected_and_estimates_tokens_without_credentials(local_source, monkeypatch):
    processed, approval_path, _ = local_source
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    plan = importer.run_import(processed, approval_path, live=False)
    assert (plan["candidate_count"], plan["approved_count"], plan["rejected_count"]) == (2, 0, 2)
    assert plan["estimated_full_pilot_tokens"] > 0
    assert plan["estimated_full_pilot_cost_usd"] > 0
    assert "not_explicitly_approved" in plan["warnings"][0]


@pytest.mark.parametrize("contents, expected", [("{bad", "invalid JSON"), ("[]", "malformed structure")])
def test_invalid_approval_files_fail_clearly(tmp_path, contents, expected):
    approval = tmp_path / "approval.json"
    approval.write_text(contents, encoding="utf-8")
    with pytest.raises(importer.ImportPipelineError, match=expected):
        importer._load_approval(approval)


def test_missing_approval_file_fails_safely(local_source, tmp_path):
    processed, _approval_path, _summary_path = local_source
    with pytest.raises(importer.ImportPipelineError, match="Approval file not found"):
        importer.build_import_plan(processed, tmp_path / "missing.json")


def test_duplicate_ids_and_missing_metadata_are_rejected(local_source):
    processed, approval_path, _ = local_source
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["candidates"].append(dict(approval["candidates"][0]))
    approval["candidates"][1]["publisher"] = ""
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    plan = importer.build_import_plan(processed, approval_path)
    assert plan["approved_count"] == 0
    assert any("duplicate_chunk_id_in_approval_file" in item["reasons"] for item in plan["rejected"])
    assert any("metadata_mismatch_or_missing:publisher" in item["reasons"] for item in plan["rejected"])


def test_source_fingerprint_mismatch_is_rejected(local_source):
    processed, approval_path, _ = local_source
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["candidates"][0]["text_sha256"] = "0" * 64
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    plan = importer.build_import_plan(processed, approval_path)
    assert "source_text_fingerprint_mismatch" in plan["rejected"][0]["reasons"]


def test_live_mode_with_no_approved_chunks_fails_before_api_calls(local_source, monkeypatch):
    processed, approval_path, _ = local_source
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(importer.ImportPipelineError, match="No explicitly approved"):
        importer.run_import(processed, approval_path, live=True)


class FakeOpenAI:
    def __init__(self, dimension=1536, error=None):
        self.dimension = dimension
        self.error = error
        self.calls = []

    class Embeddings:
        pass

    @property
    def embeddings(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.01] * self.dimension) for _ in kwargs["input"]])


class FakeDocuments:
    def __init__(self, error=None):
        self.error = error
        self.call = None

    def upsert(self, **kwargs):
        self.call = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(upserted_count=len(kwargs["documents"]))


class FakePinecone:
    def __init__(self, error=None):
        self.error = error
        self.documents = FakeDocuments(error=error if error and error.args == ("upsert",) else None)
        self.index_name = None

    def index(self, name):
        self.index_name = name
        if self.error and self.error.args != ("upsert",):
            raise self.error
        return SimpleNamespace(documents=self.documents)


def test_approved_chunks_embed_and_upsert_only_to_knowledge_namespace(local_source):
    processed, approval_path, _ = local_source
    _approve(approval_path)
    openai = FakeOpenAI()
    pinecone = FakePinecone()
    result = importer.run_import(
        processed, approval_path, live=True,
        openai_api_key="test-openai", pinecone_api_key="test-pinecone",
        openai_client=openai, pinecone_client=pinecone,
    )
    assert result["upserted_count"] == 2
    assert len(openai.calls) == 1
    assert openai.calls[0]["model"] == "text-embedding-3-small"
    assert openai.calls[0]["dimensions"] == 1536
    assert pinecone.index_name == "career-market-knowledge"
    assert pinecone.documents.call["namespace"] == "career-market-knowledge-v1"
    record = pinecone.documents.call["documents"][0]
    assert record["_id"] == record["chunk_id"]
    assert record["chunk_vector"] == [0.01] * 1536
    assert record["page_number"] == 4
    assert record["attribution"]
    assert "CC BY 4.0" in record["rights_status"]


def test_invalid_embedding_dimension_stops_before_pinecone(local_source):
    processed, approval_path, _ = local_source
    _approve(approval_path)
    pinecone = FakePinecone()
    with pytest.raises(importer.ImportPipelineError, match="dimension must be exactly 1536"):
        importer.run_import(
            processed, approval_path, live=True,
            openai_api_key="test-openai", pinecone_api_key="test-pinecone",
            openai_client=FakeOpenAI(dimension=3), pinecone_client=pinecone,
        )
    assert pinecone.documents.call is None


@pytest.mark.parametrize("failed", ["embedding", "pinecone"])
def test_live_api_errors_are_sanitized(local_source, failed):
    processed, approval_path, _ = local_source
    _approve(approval_path)
    openai = FakeOpenAI(error=RuntimeError("private api detail")) if failed == "embedding" else FakeOpenAI()
    pinecone = FakePinecone(error=RuntimeError("private api detail")) if failed == "pinecone" else FakePinecone()
    with pytest.raises(importer.ImportPipelineError, match="Live embedding or Pinecone import failed") as error:
        importer.run_import(
            processed, approval_path, live=True,
            openai_api_key="test-openai", pinecone_api_key="test-pinecone",
            openai_client=openai, pinecone_client=pinecone,
        )
    assert "private api detail" not in str(error.value)
