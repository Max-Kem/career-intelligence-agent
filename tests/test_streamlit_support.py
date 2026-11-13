from __future__ import annotations

import json
from pathlib import Path

import pytest

from career_agent.candidate import CandidateProfile
from career_agent.streamlit_support import (
    SESSION_DEFAULTS,
    build_candidate_profile,
    build_workflow_input,
    candidate_review_summary,
    citation_presentation,
    confirmation_is_current,
    compose_experience_title,
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
    primary_reasons,
    job_context_label,
    synthesis_presentation,
    readable_requirement,
    recommendation_link,
    job_posting_highlights,
    report_download_payloads,
    reset_session_state,
    split_experience_title,
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


def test_multiple_experience_entries_keep_titles_and_add_their_own_dates():
    candidate = build_candidate_profile(valid_fields(professional_experience=[
        {"title": "Data Analyst", "start_date": "2021-01", "end_date": "2022-06", "responsibilities": ["Built reports."]},
        {"title": "AI Project Lead", "start_date": "2022-07", "ongoing": True, "responsibilities": ["Delivered a pilot."]},
    ]))

    assert [item.title for item in candidate.professional_experience] == [
        "Data Analyst (2021-01 - 2022-06)",
        "AI Project Lead (2022-07 - Present)",
    ]


def test_experience_dates_are_optional_and_ongoing_roles_show_present():
    candidate = build_candidate_profile(valid_fields(professional_experience=[
        {"title": "Independent analysis project", "responsibilities": ["Reviewed survey data."]},
        {"title": "Current analyst", "start_date": "Jan 2024", "ongoing": True, "responsibilities": ["Prepare monthly reports."]},
    ]))

    assert candidate.professional_experience[0].title == "Independent analysis project"
    assert candidate.professional_experience[1].title == "Current analyst (Jan 2024 - Present)"


def test_existing_dated_titles_split_and_recompose_without_duplicate_dates():
    title = "Digital Transformation Consultant (Jul 2019-Oct 2023)"
    clean_title, start_date, end_date, ongoing = split_experience_title(title)

    assert (clean_title, start_date, end_date, ongoing) == (
        "Digital Transformation Consultant", "Jul 2019", "Oct 2023", False,
    )
    assert compose_experience_title(clean_title, start_date, end_date) == (
        "Digital Transformation Consultant (Jul 2019 - Oct 2023)"
    )
    assert compose_experience_title(title, "2019-07", "2023-10") == title
    assert split_experience_title("AI Consultant (Nov 2023-present)") == (
        "AI Consultant", "Nov 2023", "", True,
    )


def test_experience_date_fields_are_cleared_when_session_is_reset():
    state = {
        "experience_start_date_0": "2023-01",
        "experience_end_date_0": "2024-01",
        "experience_ongoing_0": True,
    }
    reset_session_state(state)
    assert not any(key.startswith(("experience_start_date_", "experience_end_date_", "experience_ongoing_")) for key in state)


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
        "allow_live_rag": True, "allow_llm_synthesis": True, "queries": 3, "pages": 1, "top_n": 5,
    })
    assert state["offline"] is True
    assert state["esco_enabled"] is False and state["allow_live_esco"] is False
    assert state["company_research_enabled"] is False and state["allow_live_company_research"] is False
    assert state["market_intelligence_enabled"] is False and state["allow_live_market_intelligence"] is False
    assert state["llm_synthesis_enabled"] is False and state["allow_live_llm_synthesis"] is False


def test_live_workflow_input_requires_explicit_switches_and_bounds_search():
    candidate = build_candidate_profile(valid_fields())
    state = build_workflow_input(candidate, {
        "mode": "live", "allow_live_esco": True, "allow_live_company_research": False,
        "allow_live_rag": True, "allow_llm_synthesis": True, "queries": 2, "pages": 2, "top_n": 4,
    })
    assert state["offline"] is False
    assert state["esco_enabled"] is True and state["allow_live_esco"] is True
    assert state["company_research_enabled"] is False
    assert state["market_intelligence_enabled"] is True
    assert state["llm_synthesis_enabled"] is True and state["allow_live_llm_synthesis"] is True
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
    assert view["status_label"] == "Completed; no roles met the recommendation criteria."
    assert view["data_source_label"] == "Live job providers"


def test_saved_live_zero_recommendation_case_has_truthful_dashboard_metrics():
    report_path = Path(__file__).resolve().parents[1] / "reports/streamlit/career_report_custom_20261112T092129298274Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    view = dashboard_view({"workflow_status": report["workflow"]["status"], "report_data": report})
    assert view["discovered_jobs"] == 37
    assert view["evaluated_jobs"] == 37
    assert view["recommendations"] == []
    assert len(view["potential_matches"]) == 8
    assert "no roles met" in view["status_label"].casefold()
    assert view["data_source_label"] == "Live job providers"


def test_diagnostic_summary_is_plain_language_and_keeps_demo_skips_nonfatal():
    view = dashboard_view({"report_data": {
        "search_scope": {"offline": True, "provider_diagnostics": {"provider_counts": {"synthetic": 5}, "errors": {}}},
        "discovered_jobs": 5, "evaluated_jobs": 5,
        "recommendations": [{"score": 90}], "potential_matches_requiring_review": [],
        "market_intelligence": {"status": "skipped_offline"},
        "intelligence_synthesis": {"status": "skipped_offline"},
        "esco_enrichment": {"status": "skipped_offline"},
        "company_research": {"status": "skipped_offline"},
        "workflow": {"status": "completed_with_warnings", "warnings": ["Synthetic fixture jobs were used."], "errors": []},
    }})
    summary = view["diagnostics"]["summary"]
    assert summary["job_discovery"] == "Completed successfully."
    assert summary["matching"] == "Evaluated 5 jobs."
    assert "1 role qualified" in summary["recommendations"]
    assert all(summary[key] == "Not run because Demo Mode is offline."
               for key in ("market_intelligence", "llm_synthesis", "esco", "company_research"))
    assert summary["warnings"] == ["Synthetic fixture jobs were used."]


def test_partial_esco_status_is_human_readable_and_timeout_diagnostics_remain_visible():
    view = dashboard_view({"workflow_status": "completed_with_warnings", "report_data": {
        "search_scope": {"offline": False, "provider_diagnostics": {}},
        "workflow": {"status": "completed_with_warnings", "nonfatal_errors": ["ESCO Python: ReadTimeout"]},
        "recommendations": [], "potential_matches_requiring_review": [], "esco_enrichment": {"status": "partial", "lookup_count": 6},
    }})
    assert view["enrichment_messages"]["esco"].startswith("ESCO enrichment: Partial")
    assert "Partially completed" in view["diagnostics"]["summary"]["esco"]
    assert "ReadTimeout" in " ".join(view["diagnostics"]["summary"]["nonfatal_errors"])


def test_primary_reasons_are_limited_and_unmodified_full_evidence_remains_available():
    item = {"strengths": ["reason one", "reason two", "reason three", "reason four"]}
    assert primary_reasons(item) == ["reason one", "reason two", "reason three"]
    assert item["strengths"] == ["reason one", "reason two", "reason three", "reason four"]
    assert primary_reasons({"strengths": [], "reasons": ["review one", "review two", "review three", "review four"]}) == [
        "review one", "review two", "review three",
    ]


def test_job_highlights_separate_provider_remote_from_posted_location_and_attendance():
    result = job_posting_highlights({
        "work_model": "remote",
        "description": (
            "<h3>Location</h3><p>This is a full-time position based in Munich or Cologne. "
            "Following onboarding at our Zurich office for a minimum of one month, you will work from home. "
            "You will make regular customer visits across the DACH region.</p>"
            "<h3>Key Responsibilities</h3><ul><li>Run implementations end to end, from planning to go-live.</li>"
            "<li>Lead delivery teams across customer engagements.</li></ul>"
            "<h3>What You Bring</h3><p>Four years of enterprise implementation experience required.</p>"
            "<h3>Bonus Points</h3><p>ERP implementation experience is a bonus.</p>"
        ),
    })
    assert result["provider_work_model"] == "remote"
    items = result["items"]
    assert len(items) <= 6
    text = " ".join(item["text"] for item in items)
    assert "Munich or Cologne" in text
    assert "Zurich" in text
    assert "customer visits" in text
    assert any(item["category"] == "Preferred" and "bonus" in item["text"].casefold() for item in items)
    assert any(item["category"] in {"Responsibility", "Qualification", "Required"} for item in items)


def test_job_highlights_do_not_infer_facts_from_missing_or_marketing_only_descriptions():
    missing = job_posting_highlights({"work_model": "remote", "description": None})
    assert missing["provider_work_model"] == "remote"
    assert missing["description_available"] is False
    assert missing["items"] == []

    marketing = job_posting_highlights({
        "description": "<p>We are a fast-growing company with an amazing culture and great benefits.</p>"
    })
    assert marketing["items"] == []


def test_job_highlights_exclude_benefits_marketing_and_compensation_but_keep_explicit_sections():
    result = job_posting_highlights({
        "work_model": "hybrid",
        "description": (
            "<p>We are a fast-growing leader with a friendly culture.</p>"
            "<h3>What You'll Do</h3><ul><li>Rebuild executive reporting and document metric definitions.</li></ul>"
            "<h3>What You Need to Succeed</h3><ul><li>Expert-level Tableau dashboard development skills.</li></ul>"
            "<p>Generalist range and real autonomy in a lean remote team.</p>"
            "<h3>Where You'll Work</h3><p>Based in Berlin with two days onsite each week.</p>"
            "<h3>Why Shiftmove</h3><p>Dog-friendly office and collaborative culture.</p>"
            "<h3>Compensation</h3><p>Salary range of €70,000 to €80,000.</p>"
            "<h3>Benefits</h3><p>Wellbeing budget and free lunch.</p>"
        ),
    })
    assert result["provider_work_model"] == "hybrid"
    texts = [item["text"] for item in result["items"]]
    all_text = " ".join(texts).casefold()
    assert any(item["category"] == "Responsibility" and "executive reporting" in item["text"] for item in result["items"])
    assert any(item["category"] == "Qualification" and "tableau" in item["text"].casefold() for item in result["items"])
    assert any(item["category"] == "Location & attendance" and "onsite" in item["text"].casefold() for item in result["items"])
    assert "dog-friendly" not in all_text and "fast-growing" not in all_text and "salary" not in all_text
    assert not any("remote team" in item["text"].casefold() and item["category"] == "Location & attendance" for item in result["items"])


@pytest.mark.parametrize(("job", "expected"), [
    ({"location": "Remote", "work_model": "remote"}, "Remote"),
    ({"location": "Remote - Germany", "work_model": "remote"}, "Remote - Germany"),
    ({"location": "Berlin", "work_model": "hybrid"}, "Berlin | Hybrid"),
    ({"location": None, "work_model": None}, "Location not provided | Unknown"),
])
def test_job_context_label_avoids_repeated_location_and_work_model(job, expected):
    assert job_context_label(job) == expected


@pytest.mark.parametrize(
    ("synthesis", "demo", "kind", "message_fragment"),
    [
        ({"status": "succeeded", "generation_mode": "llm_generated"}, False, "success", "generated from retrieved evidence"),
        ({"status": "failed", "generation_mode": "deterministic_fallback"}, False, "info", "Deterministic guidance"),
        ({"status": "skipped_offline", "generation_mode": "deterministic_fallback"}, True, "hidden", "Deterministic guidance"),
        ({"status": "skipped_disabled"}, False, "caption", "was not requested"),
    ],
)
def test_synthesis_presentation_distinguishes_live_success_fallback_and_demo(
    synthesis, demo, kind, message_fragment,
):
    presentation = synthesis_presentation(synthesis, demo)
    assert presentation["kind"] == kind
    assert message_fragment in presentation["message"]


def test_dashboard_diagnostics_summarize_saved_live_statuses():
    report_path = Path(__file__).resolve().parents[1] / "reports/streamlit/career_report_custom_20261112T092129298274Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    view = dashboard_view({"workflow_status": report["workflow"]["status"], "report_data": report})
    summary = view["diagnostics"]["summary"]
    assert summary["job_discovery"] == "Completed successfully."
    assert summary["matching"] == "Evaluated 37 jobs."
    assert "No roles qualified" in summary["recommendations"]
    assert summary["market_intelligence"].startswith("Completed successfully.")
    assert summary["llm_synthesis"].startswith("Could not complete;")


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
    assert visible["status_label"] == "Completed with warnings."
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
    assert view["data_source_label"] == "Synthetic demo jobs"


@pytest.mark.parametrize(("citation", "kind", "label", "ref"), [
    ({"kind": "market_source", "reference_id": "S2", "document_title": "Skills Forecast",
      "publisher": "Cedefop", "publication_year": 2025, "page_number": 9,
      "source_url": "https://example.org/report"}, "market_source", "Market source", "S2"),
    ({"kind": "candidate_profile", "reference_id": "C2", "label": "documented skills"},
     "candidate_profile", "Candidate profile evidence", "C2"),
    ({"kind": "job_posting", "reference_id": "J1", "document_title": "AI Consultant",
      "company": "Example Co", "source_url": "https://example.org/job"}, "job_posting", "Job posting", "J1"),
])
def test_typed_citation_presentation_preserves_evidence_categories(citation, kind, label, ref):
    presentation = citation_presentation(citation)
    assert presentation["kind"] == kind
    assert presentation["label"] == label
    assert presentation["reference_id"] == ref
    if kind == "market_source":
        assert presentation["publisher"] == "Cedefop"
        assert presentation["page"] == 9
        assert presentation["url"] == citation["source_url"]
    elif kind == "job_posting":
        assert presentation["title"] == "AI Consultant - Example Co"
        assert presentation["url"] == citation["source_url"]
    else:
        assert presentation["url"] is None


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
