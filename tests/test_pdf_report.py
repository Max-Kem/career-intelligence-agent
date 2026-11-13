from __future__ import annotations

from pypdf import PdfReader
from io import BytesIO

from career_agent.pdf_report import generate_career_report_pdf


def report_fixture(*, recommendations: bool = True) -> dict:
    job = {
        "job_id": "fixture-job-1", "title": "AI Transformation Consultant",
        "company": "Synthetic Example GmbH", "location": "Berlin", "work_model": "hybrid",
        "source": "fixture", "source_url": "https://example.test/jobs/1?role=ai&country=de",
        "description": "MUST_NOT_APPEAR_FULL_DESCRIPTION_" * 100,
    }
    recommendation = {
        "job": job, "score": 82, "evidence_coverage": 75, "support_status": "sufficient",
        "career_direction": {"status": "aligned", "evidence": ["Job title matches target role."]},
        "strengths": ["Documented project-management experience."], "gaps": [],
        "geographic_status": "eligible", "geographic_evidence": ["Provider confirms eligibility from Germany."],
        "seniority_status": 80, "seniority_evidence": ["Profile preference aligns with the stated level."],
        "requirements_to_verify": [{
            "requirement": "Fluent German", "requirement_type": "required",
            "candidate_evidence_status": "not_documented",
            "candidate_evidence": "Language proficiency is not documented; this does not mean the candidate lacks it.",
            "posting_evidence": "Fluent German required.",
        }],
    }
    review = {
        "job": {**job, "job_id": "fixture-job-2", "title": "Senior AI Program Lead"},
        "score": 68, "evidence_coverage": 90, "support_status": "provisional",
        "career_direction": {"status": "uncertain", "evidence": ["Related work needs review."]},
        "reasons": ["A material requirement is not documented."], "requirements_to_verify": [{
            "requirement": "Six years of leadership experience", "requirement_type": "required",
            "candidate_evidence_status": "does_not_meet_requirement",
            "candidate_evidence": "The supplied profile reports fewer years than this minimum.",
            "posting_evidence": "At least six years of leadership experience.",
        }],
    }
    return {
        "candidate": {
            "name": "SYNTHETIC Profile C", "summary": "A synthetic candidate with documented transformation experience.",
            "desired_job_titles": ["AI Transformation Consultant"], "skills": ["Project management", "Python"],
            "years_experience": 7, "education": [{"qualification": "Master's degree", "field": "Business Systems"}],
            "seniority_preferences": ["mid-level"],
        },
        "search_scope": {"offline": False, "providers": ["arbeitnow", "himalayas"], "queries": ["AI Transformation Consultant"],
            "provider_diagnostics": {"provider_counts": {"arbeitnow": 10, "himalayas": 8}, "duplicates_removed": 2}},
        "evaluated_jobs": 18,
        "recommendations": [recommendation] if recommendations else [],
        "potential_matches_requiring_review": [review],
        "workflow": {"status": "completed" if recommendations else "completed_no_recommendations",
            "completed_at": "2026-11-12T10:30:00+00:00", "nodes_executed": ["load_candidate", "generate_report", "finalize"],
            "warnings": [] if recommendations else ["No discovered jobs met recommendation criteria."], "errors": []},
        "market_intelligence": {"status": "succeeded", "evidence": [{
            "document_title": "Germany Skills Forecast", "publisher": "Cedefop", "publication_year": 2025,
            "page_number": 9, "geographic_scope": "Germany_focused", "source_url": "https://example.test/source/report.pdf",
            "excerpt": "The report describes qualification and workforce trends.",
        }]},
        "intelligence_synthesis": {"status": "succeeded", "generation_mode": "llm_generated", "request_count": 1,
            "request_limit": 1, "market_intelligence": [{"text": "The source reports workforce-development trends in Germany.",
                "citations": [{"kind": "market_source", "document_title": "Germany Skills Forecast", "publisher": "Cedefop",
                    "publication_year": 2025, "page_number": 9, "source_url": "https://example.test/source/report.pdf"}]}],
            "personalized_career_intelligence": []},
        "personalized_career_intelligence": {"status": "available", "insights": [{
            "topic": "Digital skills", "source_finding": "The source discusses qualifications.",
            "candidate_implication": "This is relevant context, not a fit decision.", "suggested_next_step": "Review target postings.",
            "citation": {"document_title": "Germany Skills Forecast", "publisher": "Cedefop", "publication_year": 2025,
                "page_number": 9, "source_url": "https://example.test/source/report.pdf"}}]},
        "company_research": {"status": "skipped_no_recommendations" if not recommendations else "succeeded",
            "results": [] if not recommendations else [{"company": "Synthetic Example GmbH", "identity_status": "unverified",
                "snapshot": "Must not be attributed as company facts.", "sources": [{"title": "Possible unrelated result",
                    "url": "https://example.test/search", "identity_status": "unverified"}]}]},
        "esco_enrichment": {"status": "skipped_no_recommendations" if not recommendations else "succeeded",
            "lookup_count": 0 if not recommendations else 1, "lookups": [] if not recommendations else [{"term": "Python",
                "concept_type": "Skill", "esco_result": {"status": "exact", "source_reference": "https://example.test/esco",
                    "concepts": [{"label": "programming", "uri": "https://example.test/esco/python", "match_type": "exact"}]}}]},
    }


def test_pdf_is_deterministic_multi_page_source_linked_and_preserves_evidence_statuses():
    report = report_fixture()
    first = generate_career_report_pdf(report)
    second = generate_career_report_pdf(report)
    assert first.startswith(b"%PDF-")
    assert first == second
    reader = PdfReader(BytesIO(first))
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in (
        "Executive overview", "Candidate summary", "Recommended roles", "Potential matches requiring review",
        "Career intelligence", "Company research", "ESCO skill and occupation references", "not documented", "Explicit mismatch in supplied evidence",
        "not hiring probability", "PDF page 9", "SYNTHETIC Profile C",
    ):
        assert phrase.casefold() in text.casefold(), phrase
    assert "2026-11-12T10:30:00" not in text
    assert "nodes executed" not in text.casefold()
    assert "MUST_NOT_APPEAR_FULL_DESCRIPTION" not in text
    uris = [annotation.get_object().get("/A", {}).get("/URI") for page in reader.pages for annotation in page.get("/Annots", [])]
    assert "https://example.test/jobs/1?role=ai&country=de" in uris
    assert "https://example.test/source/report.pdf" in uris
    assert "https://example.test/esco/python" in uris
    assert "Must not be attributed as company facts." not in text


def test_zero_recommendations_is_reported_as_a_valid_completion_and_sections_explain_skips():
    report = report_fixture(recommendations=False)
    text = " ".join((page.extract_text() or "").replace("\n", " ") for page in PdfReader(BytesIO(generate_career_report_pdf(report))).pages)
    assert "no roles met the current recommendation criteria" in text.casefold()
    assert "completed result, not a failed run" in text.casefold()
    assert "was not run because no roles qualified" in text.casefold()
    assert "potential matches requiring review" in text.casefold()


def test_pdf_handles_missing_enrichment_and_empty_report_sections():
    report = report_fixture(recommendations=False)
    report["market_intelligence"] = {"status": "unavailable", "evidence": []}
    report["intelligence_synthesis"] = {"status": "unavailable_no_evidence", "request_count": 0, "request_limit": 1}
    report["company_research"] = {"status": "skipped_disabled", "results": []}
    report["esco_enrichment"] = {"status": "skipped_offline", "lookups": []}
    text = " ".join((page.extract_text() or "").replace("\n", " ") for page in PdfReader(BytesIO(generate_career_report_pdf(report))).pages)
    assert "no retrieved market evidence is available" in text.casefold()
    assert "no sufficiently complete source passages" in text.casefold()
    assert "was turned off for this run" in text.casefold()
    assert "was not run in offline mode" in text.casefold()


def test_review_details_distinguish_unknown_evidence_from_explicit_mismatch():
    report = report_fixture()
    report["potential_matches_requiring_review"][0]["requirements_to_verify"].append({
        "requirement": "German language", "requirement_type": "preferred",
        "candidate_evidence_status": "not_documented",
        "candidate_evidence": "Not documented in profile.",
        "posting_evidence": "German is preferred.",
    })
    text = " ".join((page.extract_text() or "").replace("\n", " ") for page in PdfReader(BytesIO(generate_career_report_pdf(report))).pages)
    assert "Not documented — verify" in text or "Not documented - verify" in text
    assert "Explicit mismatch in supplied evidence" in text
    assert "preferred" in text.casefold()
    assert "this is not a confirmed mismatch" in text.casefold()


def test_personalized_summary_does_not_turn_undocumented_evidence_into_a_failure():
    report = report_fixture()
    report["intelligence_synthesis"]["personalized_career_intelligence"] = [{
        "text": "The candidate does not currently meet every listed qualification.",
        "citations": [],
    }]
    text = " ".join((page.extract_text() or "").replace("\n", " ") for page in PdfReader(BytesIO(generate_career_report_pdf(report))).pages)
    assert "the candidate does not currently meet every listed qualification" not in text.casefold()
    assert "broad qualification conclusion" in text.casefold()
    assert "not documented" in text.casefold()
