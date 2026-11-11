from __future__ import annotations

import json
from pathlib import Path

from career_agent.candidate import CandidateProfile
from career_agent.discovery import DiscoveryResult
from career_agent.integration import CandidateDiscoveryRun
from career_agent.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]


def _profile(name="a"):
    return CandidateProfile.model_validate_json(
        (ROOT / "fixtures" / f"candidate_profile_{name}.json").read_text(encoding="utf-8")
    )


def _evidence(chunk, title, publisher, year, page, scope, url):
    return {
        "chunk_id": chunk, "document_title": title, "publisher": publisher,
        "publication_year": year, "page_number": page, "page_start": page,
        "page_end": page, "geographic_scope": scope, "source_url": url,
        "score": 0.81, "excerpt": "Source excerpt on skills, occupational change and workforce training.",
    }


def _invoke(tmp_path, runner, profile="a", **state):
    return build_workflow(market_intelligence_runner=runner).invoke({
        "profile_name": profile, "offline": True, "allow_live_market_intelligence": True,
        "market_intelligence_enabled": True, "esco_enabled": False,
        "company_research_enabled": False, "output_dir": str(tmp_path), **state,
    })


def test_rag_node_retrieves_cited_multisource_evidence_and_reports_candidate_context(tmp_path):
    calls = []

    def retrieve(question):
        calls.append(question)
        if "AI changing" in question:
            return {"matches": [_evidence(
                "ilo-1", "Generative AI and Jobs", "ILO", 2025, 41,
                "international_global", "https://example.org/ilo",
            )]}
        return {"matches": [_evidence(
            "cedefop-1", "Germany Skills Forecast", "Cedefop", 2025.0, 9,
            "Germany_focused", "https://example.org/cedefop",
        )]}

    result = _invoke(tmp_path, retrieve)
    nodes = result["completed_nodes"]
    assert nodes.index("research_companies") < nodes.index("retrieve_market_intelligence") < nodes.index("generate_report")
    assert len(calls) == 3
    assert result["market_intelligence"]["status"] == "succeeded"
    assert len(result["market_intelligence"]["evidence"]) == 2
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["market_intelligence"]["evidence"][0]["source_url"].startswith("https://")
    assert report["market_intelligence"]["candidate_context"]["target_roles"]
    personalized = report["personalized_career_intelligence"]
    assert personalized["status"] == "available"
    assert any(item["citation"]["geographic_scope"] == "international_global" for item in personalized["insights"])
    assert all(isinstance(item["citation"]["page_number"], int) for item in personalized["insights"])
    assert all(item["citation"]["source_url"].startswith("https://") for item in personalized["insights"])
    assert all("does not create a new gap" in item["candidate_implication"] for item in personalized["insights"])
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "## Labour-market intelligence" in markdown
    assert "### Personalized Career Intelligence" in markdown
    assert "PDF page 9" in markdown
    assert "PDF page 9.0" not in markdown
    assert "Cedefop (2025)" in markdown
    assert "Germany-focused" in markdown
    assert "Global/international" in markdown
    assert "Similarity scores indicate semantic closeness" in markdown
    assert "Suggested next step:" in markdown
    assert "PDF page 41 — international/global" in markdown


def test_rag_node_reports_no_results_without_fabricating_findings(tmp_path):
    result = _invoke(tmp_path, lambda question: {"matches": []})
    assert result["market_intelligence"]["status"] == "no_results"
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "No market evidence was returned" in markdown
    assert "No replacement findings or citations were generated" in markdown
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["personalized_career_intelligence"]["status"] == "unavailable"


def test_personalized_insights_skip_incomplete_citations_and_do_not_invent_gaps(tmp_path):
    incomplete = {
        "chunk_id": "uncited", "excerpt": "A source mentions digital skills and workforce development.",
        "document_title": "Report", "geographic_scope": "Germany_focused",
    }
    result = _invoke(tmp_path, lambda question: {"matches": [incomplete]})
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    intelligence = report["personalized_career_intelligence"]
    assert intelligence["status"] == "unavailable"
    assert intelligence["insights"] == []
    assert "incomplete citation metadata" in intelligence["message"]
    assert report["recommendations"][0]["gaps"] == []


def test_rag_provider_failure_is_nonfatal_and_reported(tmp_path):
    calls = 0

    def retrieve(question):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"matches": [_evidence("ec-1", "Digital Decade Germany", "European Commission", 2026, 22,
                                            "Germany_focused", "https://example.org/ec")]}
        raise RuntimeError("private provider body")

    result = _invoke(tmp_path, retrieve)
    assert result["market_intelligence"]["status"] == "partial"
    assert result["report_paths"]["markdown"]
    assert "private provider body" not in str(result["market_intelligence"])
    assert result["nonfatal_errors"]
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "retrieval was partial" in markdown
    assert "PDF page 22" in markdown


def test_offline_rag_skips_without_explicit_allow_flag(tmp_path):
    def must_not_retrieve(question):
        raise AssertionError("ordinary offline execution must not call the retrieval client")

    result = _invoke(tmp_path, must_not_retrieve, allow_live_market_intelligence=False)
    assert result["market_intelligence"]["status"] == "skipped_offline"
    assert "retrieve_market_intelligence" in result["completed_nodes"]


def test_live_discovery_mode_can_explicitly_enable_rag_without_external_providers(tmp_path):
    calls = []

    def mock_discovery(candidate, **kwargs):
        return CandidateDiscoveryRun(candidate, [], DiscoveryResult(jobs=[]))

    def mock_retrieval(question):
        calls.append(question)
        return {"matches": []}

    result = build_workflow(
        discovery_runner=mock_discovery,
        market_intelligence_runner=mock_retrieval,
    ).invoke({
        "profile_name": "a", "offline": False,
        "allow_live_market_intelligence": True, "market_intelligence_enabled": True,
        "esco_enabled": False, "company_research_enabled": False,
        "output_dir": str(tmp_path),
    })
    assert result["market_intelligence"]["status"] == "no_results"
    assert len(calls) == 3
    assert result["report_paths"]["json"]
