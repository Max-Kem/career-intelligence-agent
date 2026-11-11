from __future__ import annotations

import hashlib
import json

from career_agent.refined_rag_selection import refine_knowledge_base


def _chunk(document_id: str, title: str, page: int, number: int, text: str, *, explicit_id: str | None = None) -> dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "chunk_id": explicit_id or f"{document_id}-p{page:04d}-c{number:04d}-{digest[:12]}",
        "document_id": document_id,
        "document_title": title,
        "publisher": "Example publisher",
        "publication_year": 2025,
        "source_url": f"https://example.org/{document_id}.pdf",
        "source_filename": f"{document_id}.pdf",
        "page_number": page,
        "text": text,
        "character_count": len(text),
        "licensing_status": "verified_in_pdf",
        "license": "CC BY 4.0",
    }


def _write_fixture(tmp_path):
    processed, review = tmp_path / "processed", tmp_path / "review"
    processed.mkdir()
    review.mkdir()
    docs = [
        {"document_id": "ilo_generative_ai_jobs_2025", "filename": "ilo_generative_ai_jobs_2025.pdf", "title": "Global AI Jobs", "publisher": "ILO", "publication_year": 2025, "source_url": "https://example.org/ilo.pdf", "page_count": 10, "licensing_status": "verified_in_pdf", "license_source_url": "https://example.org/ilo-rights", "rights_preflight": {"external_embedding_status": "conditional_on_third_party_review", "attribution_requirements": "Cite ILO."}},
        {"document_id": "cedefop_skills_forecast_germany_2025", "filename": "cedefop_skills_forecast_germany_2025.pdf", "title": "Germany Skills Forecast", "publisher": "Cedefop", "publication_year": 2025, "source_url": "https://example.org/cedefop.pdf", "page_count": 10, "licensing_status": "publisher_policy_verified_document_license_unverified", "license_source_url": "https://example.org/cedefop-rights", "rights_preflight": {"external_embedding_status": "additional_review_required", "attribution_requirements": "Cite Cedefop."}},
    ]
    ilo_id = docs[0]["document_id"]
    ilo_text1 = "Generative AI and automation are changing occupational tasks across the global labour market, with important differences among job groups and exposure levels"
    ilo_text2 = "while workforce retraining, digital skills and augmentation may change work rather than remove every job."
    ilo_approved = _chunk(ilo_id, docs[0]["title"], 2, 1, "Automation exposure differs by occupation and task; the ILO describes augmentation and potential job transformation.")
    ilo_chunks = [_chunk(ilo_id, docs[0]["title"], 2, 2, ilo_text1), _chunk(ilo_id, docs[0]["title"], 2, 3, ilo_text2), ilo_approved]
    ced_id = docs[1]["document_id"]
    ced_approved = _chunk(ced_id, docs[1]["title"], 9, 1, "Germany's workforce qualifications and skills demand are forecast to change through 2035, supporting vocational training and reskilling.")
    ced_approved["chunk_id"] = "cedefop_skills_forecast_germany_2025-p0009-c0001-4b6effd86ac8"
    ced_short = _chunk(ced_id, docs[1]["title"], 1, 1, "ENEN\n2025 skills forecast\nGermany")
    rejected_text = "Increasing specialisation in Germany changes occupational structure and employment."
    ced_rejected = _chunk(ced_id, docs[1]["title"], 7, 1, rejected_text, explicit_id="cedefop_skills_forecast_germany_2025-p0007-c0001-b18b78fd75cb")
    for document, chunks in ((docs[0], ilo_chunks), (docs[1], [ced_approved, ced_short, ced_rejected])):
        (processed / f"{document['document_id']}.chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")
    (processed / "manifest.json").write_text(json.dumps({"documents": docs}, ensure_ascii=False), encoding="utf-8")

    def approval(chunk, third_party):
        return {"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"], "page_number": chunk["page_number"], "document_title": chunk["document_title"], "publisher": chunk["publisher"], "publication_year": chunk["publication_year"], "source_url": chunk["source_url"], "source_filename": chunk["source_filename"], "text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(), "approval_status": "approved", "approved": True, "reviewer": "project_owner", "approved_at": "2026-11-11T09:39:54Z", "third_party_review": third_party, "attribution": "Cite source."}
    ilo_approvals = {"schema_version": 1, "candidates": [approval(ilo_approved, "cleared")]}
    cedefop_approvals = {"schema_version": 1, "candidates": [approval(ced_approved, "pending")]}
    (review / "ilo_pilot_approvals.json").write_text(json.dumps(ilo_approvals), encoding="utf-8")
    (review / "multi_source_pilot_approvals.json").write_text(json.dumps(cedefop_approvals), encoding="utf-8")
    return processed, review, ilo_chunks, ced_approved


def test_refined_selection_preserves_provenance_and_rights_without_credentials(tmp_path, monkeypatch):
    processed, review, ilo_chunks, ced_approved = _write_fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in processed.glob("*.chunks.jsonl")}
    approval_paths = [review / "ilo_pilot_approvals.json", review / "multi_source_pilot_approvals.json"]
    approval_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in approval_paths}
    result = refine_knowledge_base(processed, review)
    md_first = (review / "refined_rag_selection.md").read_bytes()
    json_first = (review / "refined_rag_selection.json").read_bytes()
    refine_knowledge_base(processed, review)
    assert md_first == (review / "refined_rag_selection.md").read_bytes()
    assert json_first == (review / "refined_rag_selection.json").read_bytes()
    by_id = {row["chunk_id"]: row for row in result["selected_candidates"]}

    repaired = by_id[ilo_chunks[0]["chunk_id"]]
    assert len(repaired["derived_from"]) == 2
    assert "while workforce retraining" in repaired["candidate_text"]
    assert repaired["derived_text_sha256"] != repaired["source_text_sha256"]
    assert repaired["document_scope"] == "international_global"
    assert "german_labour_market" not in repaired["topic_matches"]

    assert by_id[ilo_chunks[2]["chunk_id"]]["import_eligibility"] == "already_indexed_do_not_reupload"
    assert by_id[ced_approved["chunk_id"]]["approved"] is True
    assert by_id[ced_approved["chunk_id"]]["import_eligibility"] == "blocked_third_party_review_unresolved"
    assert all(row["chunk_id"] != "cedefop_skills_forecast_germany_2025-p0007-c0001-b18b78fd75cb" for row in result["selected_candidates"])
    assert any("previously_rejected_candidate" in row["reasons"] for row in result["excluded_examples"])
    assert source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in processed.glob("*.chunks.jsonl")}
    assert approval_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in approval_paths}
    assert result["import_eligible_count"] == 0


def test_extremely_short_and_title_only_chunks_are_excluded(tmp_path):
    processed, review, _, _ = _write_fixture(tmp_path)
    result = refine_knowledge_base(processed, review)
    excluded = {item["chunk_id"]: item for item in result["excluded_examples"]}
    assert result["excluded_chunk_count"] > 0
    assert result["exclusion_reason_counts"]["extremely_short"] >= 1
    title_rows = [row for row in excluded.values() if "title_only_or_fragment" in row["reasons"]]
    assert title_rows
