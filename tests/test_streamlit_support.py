from __future__ import annotations

import json

import pytest

from career_agent.candidate import CandidateProfile
from career_agent.streamlit_support import (
    SESSION_DEFAULTS,
    build_candidate_profile,
    build_workflow_input,
    candidate_review_summary,
    confirmation_is_current,
    dashboard_view,
    enrichment_message,
    input_fingerprint,
    invalidate_stale_run,
    invalidate_review_and_run,
    invoke_workflow,
    load_example_profile,
    parse_lines,
    profile_completeness,
    recommendation_content,
    readable_requirement,
    recommendation_link,
    report_download_payloads,
    reset_session_state,
)


def valid_fields(**overrides):
    values = {
        "candidate_id": "local-test-001",
        "name": "Candidate",
        "desired_job_titles": ["AI Consultant"],
        "skills": ["AI implementation"],
        "professional_experience": [{"title": "AI project", "responsibilities": ["Implemented an AI use case with stakeholders."]}],
    }
    values.update(overrides)
    return values


def test_custom_profile_construction_uses_candidate_model_and_germany_hard_requirement():
    candidate = build_candidate_profile(valid_fields())
    assert isinstance(candidate, CandidateProfile)
    assert candidate.required_preferences.countries == ["DE"]
    assert candidate.country_preferences == ["DE"]
    assert candidate.years_experience is None
    assert candidate.education == []
    assert candidate.professional_experience[0].responsibilities


@pytest.mark.parametrize("override", [
    {"desired_job_titles": []}, {"skills": []}, {"professional_experience": []},
    {"professional_experience": [{"title": "Project", "responsibilities": []}]},
    {"professional_experience": [{"title": " ", "responsibilities": ["Work"]}]},
    {"professional_experience": [{"title": "123", "responsibilities": ["123"]}]},
])
def test_required_profile_fields_are_validated(override):
    with pytest.raises(ValueError):
        build_candidate_profile(valid_fields(**override))


def test_whitespace_only_titles_and_skills_do_not_satisfy_required_fields():
    with pytest.raises(ValueError, match="desired job title"):
        build_candidate_profile(valid_fields(desired_job_titles=[" "]))
    with pytest.raises(ValueError, match="relevant skill"):
        build_candidate_profile(valid_fields(skills=["\n "]))


def test_optional_profile_sections_remain_unknown_and_incomplete_optional_rows_are_rejected():
    candidate = build_candidate_profile(valid_fields(education=[], years_experience=None, summary=""))
    assert candidate.education == [] and candidate.years_experience is None and candidate.summary is None
    with pytest.raises(ValueError, match="Complete or remove"):
        build_candidate_profile(valid_fields(professional_experience=[
            {"title": "AI project", "responsibilities": ["Implemented use case"]},
            {"title": "Second project", "responsibilities": []},
        ]))
    with pytest.raises(ValueError, match="qualification"):
        build_candidate_profile(valid_fields(education=[{"qualification": "", "field": "Economics"}]))


@pytest.mark.parametrize("name", ["a", "b"])
def test_example_profiles_load_and_can_be_edited_as_custom_profiles(name):
    example = load_example_profile(name)
    example["name"] = "Edited Candidate"
    example["desired_job_titles"] = ["Edited role"]
    candidate = build_candidate_profile(example)
    assert candidate.name == "Edited Candidate"
    assert candidate.desired_job_titles == ["Edited role"]
    assert candidate.required_preferences.countries == ["DE"]


def test_profile_completeness_is_section_count_not_match_score():
    candidate = build_candidate_profile(valid_fields(summary="Interested in responsible AI."))
    populated, total, names = profile_completeness(candidate)
    assert 0 < populated < total
    assert "Desired roles" in names and "Skills" in names and "Experience/projects" in names


def test_demo_workflow_input_forces_all_external_integrations_off():
    candidate = build_candidate_profile(valid_fields())
    state = build_workflow_input(candidate, {
        "mode": "demo", "allow_live_esco": True, "allow_live_company_research": True,
        "allow_live_rag": True, "queries": 3, "pages": 1, "top_n": 5,
    })
    assert state["offline"] is True
    assert state["esco_enabled"] is False and state["allow_live_esco"] is False
    assert state["company_research_enabled"] is False and state["allow_live_company_research"] is False
    assert state["market_intelligence_enabled"] is False and state["allow_live_market_intelligence"] is False


def test_live_workflow_input_requires_explicit_switches_and_bounds_search():
    candidate = build_candidate_profile(valid_fields())
    state = build_workflow_input(candidate, {
        "mode": "live", "allow_live_esco": True, "allow_live_company_research": False,
        "allow_live_rag": True, "queries": 2, "pages": 2, "top_n": 4,
    })
    assert state["offline"] is False
    assert state["esco_enabled"] is True and state["allow_live_esco"] is True
    assert state["company_research_enabled"] is False
    assert state["market_intelligence_enabled"] is True
    with pytest.raises(ValueError, match="limits"):
        build_workflow_input(candidate, {"mode": "live", "queries": 4})


def test_existing_workflow_invokes_custom_candidate_in_demo_mode_without_live_enrichment(tmp_path):
    example = load_example_profile("a")
    candidate = build_candidate_profile(example)
    result = invoke_workflow(candidate, {
        "mode": "demo", "queries": 3, "pages": 1, "top_n": 5, "output_dir": str(tmp_path),
    })
    assert result["candidate"].candidate_id == candidate.candidate_id
    assert result["offline"] is True
    assert result["report_data"]["search_scope"]["offline"] is True
    assert result["report_data"]["esco_enrichment"]["status"] == "skipped_offline"
    assert result["report_data"]["company_research"]["status"] == "skipped_offline"
    assert result["report_data"]["market_intelligence"]["status"] == "skipped_offline"


def test_zero_recommendation_and_review_match_dashboard_presentation():
    result = {"workflow_status": "completed_no_recommendations", "report_data": {
        "discovered_jobs": 4, "evaluated_jobs": 4, "recommendations": [],
        "potential_matches_requiring_review": [{"job": {"title": "Consultant KI", "company": "ProcuVia"},
            "score": 83, "support_status": "provisional", "reasons": ["Evidence status: provisional."]}],
        "personalized_career_intelligence": {"status": "unavailable", "message": "No evidence."},
        "market_intelligence": {"status": "skipped_offline"},
        "esco_enrichment": {"status": "skipped_offline"},
        "company_research": {"status": "skipped_offline"},
    }}
    view = dashboard_view(result)
    assert view["recommendations"] == []
    assert view["potential_matches"][0]["support_status"] == "provisional"
    assert view["discovered_jobs"] == 4 and view["evaluated_jobs"] == 4
    assert view["market_intelligence"]["status"] == "skipped_offline"


def test_reset_clears_all_profile_dynamic_review_result_and_download_state():
    state = {
        "candidate_id": "old", "display_name": "Entered name", "summary": "Private text",
        "desired_titles": "Role", "skills": "Skill", "experience_count": 4, "education_count": 2,
        "experience_title_3": "Old project", "experience_responsibilities_3": "Old work",
        "education_qualification_1": "Old degree", "confirm_old": True,
        "reviewed_signature": "old", "run_signature": "old", "run_result": {"report_data": {}},
        "mode": "Live mode", "queries": 1, "allow_esco": True, "allow_company": True, "allow_rag": True,
        "example_choice": "Profile A",
    }
    reset_session_state(state)
    assert state["display_name"] == "Candidate" and state["summary"] == ""
    assert state["candidate_id"].startswith("local-") and state["candidate_id"] != "old"
    assert state["desired_titles"] == "" and state["skills"] == ""
    assert state["experience_count"] == 1 and state["education_count"] == 0
    assert not any(key.startswith(("experience_title_", "experience_responsibilities_", "education_qualification_", "education_field_", "education_institution_", "confirm_")) for key in state)
    assert not any(key in state for key in ("reviewed_signature", "run_signature", "run_result"))
    assert state["example_choice"] == "None"
    assert state["mode"] == "Demo mode" and state["queries"] == 3
    assert state["allow_esco"] is state["allow_company"] is state["allow_rag"] is False
    assert state["location"] == SESSION_DEFAULTS["location"]


def test_empty_profile_completeness_after_reset_is_zero():
    fields = {
        "desired_job_titles": [], "skills": [], "professional_experience": [{"title": "", "responsibilities": []}],
        "summary": "", "years_experience": None, "education": [], "career_interests": [],
        "preferred_industries": [], "seniority_preferences": [], "location_preferences": [],
    }
    populated, total, labels = profile_completeness(fields)
    assert (populated, total, labels) == (0, 8, [])


def test_candidate_and_execution_changes_invalidate_review_fingerprint():
    candidate = build_candidate_profile(valid_fields())
    settings = {"mode": "demo", "queries": 3}
    original = input_fingerprint(candidate, settings)
    assert confirmation_is_current(original, input_fingerprint(candidate, settings))
    edited_candidate = candidate.model_copy(update={"skills": ["Python"]})
    assert not confirmation_is_current(original, input_fingerprint(edited_candidate, settings))
    edited_settings = {**settings, "queries": 2}
    assert not confirmation_is_current(original, input_fingerprint(candidate, edited_settings))


def test_widget_edit_callback_clears_confirmation_review_and_results_for_both_edit_types():
    for edited_value in ("candidate field changed", "live setting changed"):
        state = {"confirm_snapshot": True, "reviewed_signature": "snapshot", "run_signature": "snapshot", "run_result": edited_value}
        invalidate_review_and_run(state)
        assert not any(key.startswith("confirm_") for key in state)
        assert not any(key in state for key in ("reviewed_signature", "run_signature", "run_result"))


def test_stale_results_and_report_downloads_are_removed_after_input_change():
    candidate = build_candidate_profile(valid_fields())
    old_fingerprint = input_fingerprint(candidate, {"mode": "demo"})
    state = {"run_signature": old_fingerprint, "run_result": {"report_markdown": "old report"}}
    changed = input_fingerprint(candidate, {"mode": "live"})
    assert invalidate_stale_run(state, changed)
    assert "run_result" not in state and "run_signature" not in state
    assert not invalidate_stale_run(state, changed)


def test_requirement_presentation_is_compact_and_preserves_evidence_fields():
    item = {
        "requirement": "Core-banking domain expertise", "requirement_type": "preferred",
        "candidate_evidence_status": "not_documented",
        "candidate_evidence": "Not documented; this does not mean the candidate lacks it.",
        "posting_evidence": "Core-banking experience is preferred.",
    }
    formatted = readable_requirement(item)
    assert "preferred" in formatted and "not documented" in formatted
    assert "Posting: Core-banking experience is preferred." in formatted
    assert "does not mean" in formatted


def test_recommendation_lists_use_plain_text_and_accurate_empty_messages():
    content = recommendation_content({"strengths": ["Matched project management"], "gaps": [], "requirements_to_verify": []})
    assert content["strengths"] == ["Matched project management"]
    assert "does not verify" in content["gaps_empty"]
    assert "does not guarantee full qualification" in content["requirements_empty"]
    formatted = recommendation_content({"requirements_to_verify": [{
        "requirement": "Energy-sector experience", "requirement_type": "required",
        "candidate_evidence_status": "not_documented", "candidate_evidence": "Not documented.",
        "posting_evidence": "Experience in energy is required.",
    }]})
    assert "required" in formatted["requirements"][0]
    assert "Posting: Experience in energy is required." in formatted["requirements"][0]


def test_synthetic_and_invalid_job_urls_are_hidden_but_real_live_links_are_kept():
    synthetic = {"source": "synthetic", "source_url": "https://example.invalid/jobs/role"}
    live = {"source": "arbeitnow", "source_url": "https://www.arbeitnow.com/jobs/role"}
    assert recommendation_link(synthetic, offline=False) == {
        "url": None, "message": "Synthetic example – no external job posting.",
    }
    assert recommendation_link(live, offline=True)["url"] is None
    assert recommendation_link(live, offline=False) == {"url": live["source_url"], "message": None}
    assert recommendation_link({"source_url": "file:///fake"}, offline=False)["url"] is None


def test_demo_status_is_friendly_but_raw_status_and_warnings_remain_in_diagnostics():
    warning = "Synthetic fixture jobs were used for discovery and matching; job provider APIs were not called."
    result = {"workflow_status": "completed_with_warnings", "warnings": [warning], "report_data": {
        "search_scope": {"offline": True}, "workflow": {"status": "completed_with_warnings", "warnings": [warning]},
    }}
    view = dashboard_view(result)
    assert view["status_label"] == "Completed successfully using synthetic demo data."
    assert view["warnings"] == []
    assert view["diagnostics"]["raw_status"] == "completed_with_warnings"
    assert view["diagnostics"]["warnings"] == [warning]

    problematic = {**result, "errors": ["Provider failed."], "warnings": [warning, "Tavily timed out."]}
    visible = dashboard_view(problematic)
    assert visible["status_label"] == "Completed with warnings"
    assert visible["warnings"] == [warning, "Tavily timed out."]
    assert visible["errors"] == ["Provider failed."]


@pytest.mark.parametrize("service,expected", [
    ("market_intelligence", "RAG enabled"),
    ("esco", "ESCO enabled"),
    ("company_research", "Tavily enabled"),
])
def test_demo_enrichment_explanations_are_plain_language(service, expected):
    assert expected in enrichment_message(service, "skipped_offline", True)


def test_dashboard_uses_plain_offline_enrichment_messages_and_preserves_raw_statuses():
    result = {"report_data": {
        "search_scope": {"offline": True},
        "esco_enrichment": {"status": "skipped_offline"},
        "company_research": {"status": "skipped_offline"},
        "market_intelligence": {"status": "skipped_offline"},
    }}
    view = dashboard_view(result)
    assert "Demo Mode" in view["enrichment_messages"]["market_intelligence"]
    assert "ESCO enrichment is disabled" in view["enrichment_messages"]["esco"]
    assert "Company research is disabled" in view["enrichment_messages"]["company_research"]
    assert set(view["diagnostics"]["enrichment_statuses"].values()) == {"skipped_offline"}


def test_human_review_summary_displays_missing_optional_fields_as_not_provided():
    candidate = build_candidate_profile(valid_fields())
    summary = candidate_review_summary(candidate)
    assert summary["display_name"] == "Candidate"
    assert summary["years_experience"] == "Not provided"
    assert summary["education"] == []
    assert summary["geographic_assumption"].startswith("Candidate resides in Germany")


def test_markdown_and_json_download_payloads_preserve_report_contents():
    markdown = "# Career report\n"
    report = {"recommendations": [{"score": 83}], "warnings": ["fixture data"]}
    md_payload, json_payload = report_download_payloads(markdown, report)
    assert md_payload == markdown
    assert json.loads(json_payload) == report


def test_line_input_parsing_drops_blank_lines_only():
    assert parse_lines("  AI Consultant\n\nTransformation Lead  ") == ["AI Consultant", "Transformation Lead"]
