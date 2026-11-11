from __future__ import annotations

import hashlib
import json
from pathlib import Path

from career_agent.knowledge_audit import _select_candidates, audit_knowledge_base


def _document(document_id: str, *, rights_status: str = "conditional_on_third_party_review") -> dict:
    return {
        "document_id": document_id,
        "filename": f"{document_id}.pdf",
        "title": f"{document_id} report",
        "publisher": "Example public institution",
        "publication_year": 2025,
        "source_url": f"https://example.org/{document_id}.pdf",
        "source_url_status": "verified",
        "page_count": 5,
        "licensing_status": "publisher_policy_verified_document_license_unverified",
        "license": None,
        "license_source_url": "https://example.org/reuse-policy",
        "rights_preflight": {
            "external_embedding_status": rights_status,
            "attribution_requirements": "Cite publisher, report title, and URL.",
        },
    }


def _chunk(document: dict, text: str, *, page: int = 1, bad_fingerprint: bool = False) -> dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    suffix = "0" * 12 if bad_fingerprint else digest[:12]
    return {
        "chunk_id": f"{document['document_id']}-p{page:04d}-c0001-{suffix}",
        "document_id": document["document_id"],
        "document_title": document["title"],
        "publisher": document["publisher"],
        "publication_year": document["publication_year"],
        "source_url": document["source_url"],
        "source_filename": document["filename"],
        "page_number": page,
        "text": text,
        "character_count": len(text),
        "licensing_status": document["licensing_status"],
        "license": document["license"],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    processed = tmp_path / "processed"
    review = tmp_path / "review"
    processed.mkdir()
    review.mkdir()
    docs = {
        "ilo_generative_ai_jobs_2025": _document("ilo_generative_ai_jobs_2025", rights_status="conditional_on_third_party_review"),
        "ba_ikt_arbeitsmarkt_2026": _document("ba_ikt_arbeitsmarkt_2026", rights_status="additional_review_required"),
        "cedefop_skills_forecast_germany_2025": _document("cedefop_skills_forecast_germany_2025", rights_status="additional_review_required"),
        "eu_digital_decade_germany_2026": _document("eu_digital_decade_germany_2026", rights_status="additional_review_required"),
    }
    text = (
        "Generative AI and automation affect the German labour market and employment. "
        "ICT occupations require digital skills and qualifications; training, upskilling and reskilling "
        "support long-term career development through 2035."
    )
    all_chunks = {}
    for doc_id, document in docs.items():
        source_specific = {
            "ilo_generative_ai_jobs_2025": " International Labour Office exposure analysis explains occupational task transformation.",
            "ba_ikt_arbeitsmarkt_2026": " Bundesagentur für Arbeit reports German ICT employment indicators and workforce changes.",
            "cedefop_skills_forecast_germany_2025": " Cedefop forecasts qualification demand and supply in its national skills model.",
            "eu_digital_decade_germany_2026": " European Commission country monitoring reports ICT recruitment and digital transformation policy.",
        }[doc_id]
        chunk = _chunk(document, text + source_specific)
        all_chunks[doc_id] = chunk
        (processed / f"{doc_id}.chunks.jsonl").write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
    (processed / "manifest.json").write_text(json.dumps({"documents": list(docs.values())}, ensure_ascii=False), encoding="utf-8")

    def approval(chunk: dict, *, approved: bool, third_party: str) -> dict:
        return {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page_number": chunk["page_number"],
            "document_title": chunk["document_title"],
            "publisher": chunk["publisher"],
            "publication_year": chunk["publication_year"],
            "source_url": chunk["source_url"],
            "source_filename": chunk["source_filename"],
            "license": "CC BY 4.0" if chunk["document_id"].startswith("ilo_") else None,
            "attribution": "Attribution text retained.",
            "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
            "approval_status": "approved" if approved else "pending_review",
            "approved": approved,
            "reviewer": "project_owner" if approved else None,
            "approved_at": "2026-11-11T00:00:00Z" if approved else None,
            "third_party_review": third_party,
            "approval_note": "Explicit individual approval." if approved else None,
        }

    (review / "ilo_pilot_approvals.json").write_text(json.dumps({"schema_version": 1, "candidates": [approval(all_chunks["ilo_generative_ai_jobs_2025"], approved=True, third_party="cleared")]}, ensure_ascii=False), encoding="utf-8")
    multi = [
        approval(all_chunks["ba_ikt_arbeitsmarkt_2026"], approved=True, third_party="cleared"),
        approval(all_chunks["cedefop_skills_forecast_germany_2025"], approved=True, third_party="pending"),
        approval(all_chunks["eu_digital_decade_germany_2026"], approved=False, third_party="pending"),
    ]
    (review / "multi_source_pilot_approvals.json").write_text(json.dumps({"schema_version": 1, "candidates": multi}, ensure_ascii=False), encoding="utf-8")
    return processed, review, all_chunks


def test_offline_audit_is_deterministic_and_does_not_modify_source_or_approvals(tmp_path, monkeypatch):
    processed, review, chunks = _fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    source_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in processed.glob("*.chunks.jsonl")}
    approval_files = ("ilo_pilot_approvals.json", "multi_source_pilot_approvals.json")
    approval_before = {name: hashlib.sha256((review / name).read_bytes()).hexdigest() for name in approval_files}

    first = audit_knowledge_base(processed, review)
    report_first = (review / "knowledge_base_audit.md").read_bytes()
    shortlist_first = (review / "knowledge_base_shortlist.json").read_bytes()
    second = audit_knowledge_base(processed, review)

    assert first["total_chunk_count"] == 4
    assert first["selected_candidate_count"] == second["selected_candidate_count"]
    assert report_first == (review / "knowledge_base_audit.md").read_bytes()
    assert shortlist_first == (review / "knowledge_base_shortlist.json").read_bytes()
    assert source_before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in processed.glob("*.chunks.jsonl")}
    assert approval_before == {name: hashlib.sha256((review / name).read_bytes()).hexdigest() for name in approval_files}
    assert all(c["text"] in (processed / f"{doc_id}.chunks.jsonl").read_text(encoding="utf-8") for doc_id, c in chunks.items())


def test_approved_candidate_is_recorded_and_ba_rights_gate_cannot_be_bypassed(tmp_path):
    processed, review, chunks = _fixture(tmp_path)
    result = audit_knowledge_base(processed, review)
    by_id = {c["chunk_id"]: c for c in result["selected_candidates"]}
    ced = chunks["cedefop_skills_forecast_germany_2025"]
    ba = chunks["ba_ikt_arbeitsmarkt_2026"]
    assert by_id[ced["chunk_id"]]["approval_status"] == "approved"
    assert by_id[ced["chunk_id"]]["rights_gate"] == "approved_by_project_owner_third_party_review_not_separately_cleared"
    assert by_id[ba["chunk_id"]]["approved"] is True
    assert by_id[ba["chunk_id"]]["rights_gate"] == "blocked_pending_reuse_policy_clarification"
    assert by_id[ba["chunk_id"]]["selection_status"] == "selected_but_rights_blocked"
    assert result["selected_rights_blocked_count"] == 1


def test_quality_blockers_and_warning_types_are_reported(tmp_path):
    processed, review, _ = _fixture(tmp_path)
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    ba = manifest["documents"][1]
    malformed = _chunk(ba, "References https://example.org/ref DOI 10.1/x", page=99, bad_fingerprint=True)
    (processed / f"{ba['document_id']}.chunks.jsonl").write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    result = audit_knowledge_base(processed, review)
    bad = next(c for c in result["_all_records"] if c["chunk_id"] == malformed["chunk_id"])
    codes = {issue["code"] for issue in bad["quality_issues"]}
    assert {"invalid_page_reference", "chunk_id_fingerprint_mismatch", "very_short_text"} <= codes
    assert bad["blocking_issues"]
    assert result["document_results"][ba["document_id"]]["blocking_chunk_count"] == 1


def test_explicit_approval_is_preserved_even_without_keyword_relevance():
    approved = {
        "chunk_id": "owner-approved",
        "approval_status": "approved",
        "blocking_issues": [],
        "relevance_score": 0,
        "topic_matches": {},
        "document_id": "example",
        "duplicate_matches": [],
    }
    assert _select_candidates([approved]) == [approved]
