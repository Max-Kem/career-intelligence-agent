from __future__ import annotations

import hashlib
import json

import pytest

from career_agent.knowledge_batch import (
    BatchImportError,
    INDEXED_ILO_IDS,
    approve_batch,
    build_batch_manifest,
    run_batch_import,
    write_batch_artifacts,
    _overlap_join,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_adjacent_chunk_overlap_is_removed_exactly():
    phrase = "Shared words remain in the first source chunk exactly."
    first = f"The workforce needs digital skills for the changing labour market. {phrase}"
    second = f"{phrase} New source text begins here."
    repaired, removed = _overlap_join([first, second])
    assert removed == 9
    assert repaired.count(phrase) == 1
    assert repaired.endswith("New source text begins here.")


def test_manifest_repairs_joined_source_chunks_without_repeated_passage(tmp_path):
    processed, review, _ = _fixture(tmp_path)
    doc_id = "eu_digital_decade_germany_2026"
    first_id = f"{doc_id}-p0030-c0001-aaaaaaaaaaaa"
    second_id = f"{doc_id}-p0030-c0002-bbbbbbbbbbbb"
    shared = "This exact shared passage contains eight repeated words."
    first = f"Germany needs digital skills and workforce development. {shared}"
    second = f"{shared} Additional exact source content follows here."
    raw_join = f"{first} {second}"
    sources = [
        {"chunk_id": first_id, "document_id": doc_id, "page_number": 30, "text": first},
        {"chunk_id": second_id, "document_id": doc_id, "page_number": 30, "text": second},
    ]
    path = processed / f"{doc_id}.chunks.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        for source in sources:
            stream.write(json.dumps(source) + "\n")
    selection_path = review / "refined_rag_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    document = next(row for row in json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["documents"] if row["document_id"] == doc_id)
    selection["selected_candidates"].append({
        "chunk_id": first_id, "document_id": doc_id, "candidate_text": raw_join,
        "source_url": document["source_url"],
        "derived_text_sha256": _sha(raw_join),
        "derived_from": [{"chunk_id": row["chunk_id"], "text_sha256": _sha(row["text"])} for row in sources],
        "repair_kind": "joined_exact_adjacent_same_page_text", "page_number": 30,
        "page_start": 30, "page_end": 30, "document_scope": "Germany_focused",
        "quality_issue_codes": [], "unresolved_boundary_codes": [], "approved": False,
    })
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    manifest = build_batch_manifest(processed, review)
    record = next(row for row in manifest["records"] if row["record_id"] == first_id)
    expected, word_count = _overlap_join([first, second])
    assert record["candidate_text"] == expected
    assert record["overlap_removed_word_count"] == word_count >= 8
    assert record["candidate_text"].count(shared) == 1
    assert record["repair_kind"] == "joined_exact_adjacent_same_page_text_overlap_removed"


def _fixture(tmp_path):
    processed, review = tmp_path / "processed", tmp_path / "review"
    processed.mkdir()
    review.mkdir()
    definitions = [
        ("ilo_generative_ai_jobs_2025", "ILO Global Jobs", "CC BY 4.0", "verified_in_pdf", "conditional_on_third_party_review"),
        ("cedefop_skills_forecast_germany_2025", "Germany Skills Forecast", None, "publisher_policy_verified_document_license_unverified", "additional_review_required"),
        ("eu_digital_decade_germany_2026", "Germany Digital Decade", None, "publisher_policy_verified_document_license_unverified", "additional_review_required"),
        ("ba_ikt_arbeitsmarkt_2026", "BA IKT Report", None, "document_reuse_terms_confirmed_no_standard_license", "additional_review_required"),
    ]
    docs, chunks, candidates = [], [], []
    for doc_id, title, license_name, license_status, embed_status in definitions:
        docs.append({
            "document_id": doc_id,
            "filename": f"{doc_id}.pdf",
            "title": title,
            "publisher": doc_id,
            "publication_year": 2025,
            "source_url": f"https://example.org/{doc_id}.pdf",
            "license": license_name,
            "licensing_status": license_status,
            "license_source_url": "https://example.org/reuse",
            "rights_preflight": {"external_embedding_status": embed_status, "attribution_requirements": f"Credit {doc_id}.", "license_or_reuse_conditions": "Use with attribution.", "third_party_rights": "Check separately."},
        })

    rows = [
        (definitions[0][0], "ilo_generative_ai_jobs_2025-p0014-c0001-38187f3a7d1b", "Existing indexed ILO record about automation, employment and skills.", True, "cleared", []),
        (definitions[0][0], "ilo_generative_ai_jobs_2025-p0012-c0001-a1b316cc53e3", "Generative AI and automation shape occupational tasks and workforce development through training and reskilling.", False, "pending", []),
        (definitions[1][0], "cedefop_skills_forecast_germany_2025-p0009-c0001-4b6effd86ac8", "Germany job openings are forecast to require qualifications and skills through 2035.", True, "cleared", ["chart_or_table_residue"]),
        (definitions[2][0], "eu_digital_decade_germany_2026-p0005-c0001-aaaaaaaaaaaa", "German enterprises face digital skills gaps in artificial intelligence, employment and ICT occupations.", False, "pending", []),
        (definitions[2][0], "eu_digital_decade_germany_2026-p0006-c0001-bbbbbbbbbbbb", "Digital skills demand in Germany has area s split due to a broken word.", False, "pending", ["probable_broken_words"]),
        (definitions[3][0], "ba_ikt_arbeitsmarkt_2026-p0006-c0001-cccccccccccc", "German ICT employment and skills show changing demand.", False, "pending", []),
    ]
    for doc_id, chunk_id, text, approved, third_party, quality_codes in rows:
        page = int(chunk_id.split("-p", 1)[1][:4])
        document = next(d for d in docs if d["document_id"] == doc_id)
        chunk = {"chunk_id": chunk_id, "document_id": doc_id, "document_title": document["title"], "publisher": document["publisher"], "publication_year": document["publication_year"], "source_url": document["source_url"], "source_filename": document["filename"], "page_number": page, "text": text}
        chunks.append(chunk)
        candidates.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_title": document["title"],
            "extracted_chunk_title": document["title"],
            "publisher": document["publisher"],
            "publication_year": document["publication_year"],
            "source_url": document["source_url"],
            "source_filename": document["filename"],
            "page_number": page,
            "page_start": page,
            "page_end": page,
            "document_scope": "international_global" if doc_id.startswith("ilo_") else "Germany_focused",
            "candidate_text": text,
            "derived_text_sha256": _sha(text),
            "derived_from": [{"chunk_id": chunk_id, "text_sha256": _sha(text)}],
            "repair_kind": None,
            "quality_issue_codes": quality_codes,
            "unresolved_boundary_codes": [],
            "approved": approved,
            "approval_status": "approved" if approved else "pending_review",
            "approval_fingerprint_match": True,
            "approval_reviewer": "project_owner" if approved else None,
            "approved_at": "2026-11-11T00:00:00Z" if approved else None,
            "approval_note": "Individual review" if approved else None,
            "third_party_review": third_party,
        })
    for doc in docs:
        source_rows = [chunk for chunk in chunks if chunk["document_id"] == doc["document_id"]]
        (processed / f"{doc['document_id']}.chunks.jsonl").write_text("\n".join(json.dumps(row) for row in source_rows) + "\n", encoding="utf-8")
    (processed / "manifest.json").write_text(json.dumps({"documents": docs}), encoding="utf-8")
    (review / "refined_rag_selection.json").write_text(json.dumps({"selected_candidates": candidates}), encoding="utf-8")

    def approval(chunk_id, third_party):
        source = next(row for row in chunks if row["chunk_id"] == chunk_id)
        doc = next(d for d in docs if d["document_id"] == source["document_id"])
        return {"chunk_id": chunk_id, "document_id": source["document_id"], "text_sha256": _sha(source["text"]), "approval_status": "approved", "approved": True, "reviewer": "project_owner", "approved_at": "2026-11-11T00:00:00Z", "third_party_review": third_party, "document_title": doc["title"], "publisher": doc["publisher"], "publication_year": doc["publication_year"], "source_url": doc["source_url"], "source_filename": doc["filename"], "attribution": "Credit source."}

    (review / "ilo_pilot_approvals.json").write_text(json.dumps({"schema_version": 1, "candidates": [approval(rows[0][1], "cleared")] }), encoding="utf-8")
    (review / "multi_source_pilot_approvals.json").write_text(json.dumps({"schema_version": 1, "candidates": [approval(rows[2][1], "cleared")] }), encoding="utf-8")
    return processed, review, rows


def test_batch_manifest_excludes_ba_indexed_ilo_and_risky_records(tmp_path, monkeypatch):
    processed, review, rows = _fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    manifest = build_batch_manifest(processed, review)
    assert manifest["proposed_counts_by_source"] == {
        "cedefop_skills_forecast_germany_2025": 1,
        "eu_digital_decade_germany_2026": 1,
        "ilo_generative_ai_jobs_2025": 1,
    }
    assert len(manifest["previously_indexed_excluded"]) == 1
    assert manifest["previously_indexed_excluded"][0] in INDEXED_ILO_IDS
    assert len(manifest["rights_blocked_records"]) == 1
    assert {row["reason"] for row in manifest["rights_blocked_records"]} == {"reuse_policy_unresolved; no external processing"}
    assert any("probable_broken_words" in row["reasons"] for row in manifest["other_exclusions"])
    ced = next(row for row in manifest["records"] if row["document_id"].startswith("cedefop_"))
    assert ced["prior_individual_approval"]["approved"] is True
    assert ced["batch_approval_status"] == "pending_owner_approval"


def test_exact_batch_approval_required_and_manifest_mutation_invalidates(tmp_path):
    processed, review, _ = _fixture(tmp_path)
    manifest = write_batch_artifacts(processed, review)
    dry = run_batch_import(processed, review, live=False)
    assert dry["included_count"] == manifest["proposed_record_count"]
    assert dry["batch_approval_valid"] is False
    with pytest.raises(BatchImportError, match="No valid batch approval"):
        run_batch_import(processed, review, live=True, openai_api_key="mock", pinecone_api_key="mock")
    with pytest.raises(BatchImportError, match="Confirmation does not match"):
        approve_batch(review, reviewer="project_owner", confirmation_digest="wrong", processed_dir=processed)
    approve_batch(review, reviewer="project_owner", confirmation_digest=manifest["manifest_sha256"], approved_at="2026-11-11T10:00:00Z", processed_dir=processed)
    assert run_batch_import(processed, review, live=False)["batch_approval_valid"] is True

    path = review / "rag_batch_manifest.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["records"][0]["candidate_text"] += " changed"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(BatchImportError, match="manifest fingerprint mismatch"):
        run_batch_import(processed, review, live=True, openai_api_key="mock", pinecone_api_key="mock")


def test_source_mutation_after_approval_stops_before_clients(tmp_path):
    processed, review, _ = _fixture(tmp_path)
    manifest = write_batch_artifacts(processed, review)
    approve_batch(review, reviewer="project_owner", confirmation_digest=manifest["manifest_sha256"], approved_at="2026-11-11T10:00:00Z", processed_dir=processed)
    chunk_path = processed / "eu_digital_decade_germany_2026.chunks.jsonl"
    records = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]
    records[0]["text"] += " changed"
    chunk_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    with pytest.raises(BatchImportError, match="Source chunk changed after review"):
        run_batch_import(processed, review, live=True, openai_api_key="mock", pinecone_api_key="mock", openai_client=object(), pinecone_client=object())


def test_mocked_batch_upload_uses_approved_text_and_knowledge_namespace(tmp_path, monkeypatch):
    import career_agent.knowledge_batch as batch

    processed, review, _ = _fixture(tmp_path)
    manifest = write_batch_artifacts(processed, review)
    approve_batch(review, reviewer="project_owner", confirmation_digest=manifest["manifest_sha256"], approved_at="2026-11-11T10:00:00Z", processed_dir=processed)
    monkeypatch.setattr(batch, "embed_texts", lambda _client, texts: [[0.1] * 1536 for _ in texts])

    class Documents:
        call = None

        def upsert(self, **kwargs):
            self.call = kwargs
            return {"upserted_count": len(kwargs["documents"])}

    class Index:
        def __init__(self):
            self.documents = Documents()

    class Pinecone:
        def __init__(self):
            self.target = Index()

        def index(self, name):
            assert name == "career-market-knowledge"
            return self.target

    pinecone = Pinecone()
    result = run_batch_import(processed, review, live=True, openai_api_key="mock", pinecone_api_key="mock", openai_client=object(), pinecone_client=pinecone)
    assert result["upserted_count"] == manifest["proposed_record_count"]
    assert pinecone.target.documents.call["namespace"] == "career-market-knowledge-v1"
    record = pinecone.target.documents.call["documents"][0]
    assert record["chunk_id"] == record["_id"]
    assert record["chunk_text"]
    assert record["source_chunk_ids"]
    assert record["source_text_fingerprints"]
    assert record["attribution"]
