"""Local Streamlit interface for the Career Intelligence Agent."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from career_agent.candidate import CandidateProfile  # noqa: E402
from career_agent.streamlit_support import (  # noqa: E402
    SESSION_DEFAULTS,
    build_candidate_profile,
    candidate_review_summary,
    confirmation_is_current,
    dashboard_view,
    invalidate_review_and_run,
    input_fingerprint,
    invalidate_stale_run,
    load_example_profile,
    parse_lines,
    profile_completeness,
    recommendation_content,
    recommendation_link,
    report_download_payloads,
    reset_session_state,
    invoke_workflow,
)


def _init_state() -> None:
    st.session_state.setdefault("candidate_id", f"local-{uuid4().hex[:12]}")
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, list(value) if isinstance(value, list) else value)


def _reset_all() -> None:
    reset_session_state(st.session_state)


def _apply_example(name: str) -> None:
    example = load_example_profile(name)
    for key in list(st.session_state.keys()):
        if key.startswith(("experience_title_", "experience_responsibilities_", "education_")):
            st.session_state.pop(key, None)
    st.session_state["display_name"] = example.get("name") or "Candidate"
    st.session_state["summary"] = example.get("summary") or ""
    st.session_state["desired_titles"] = "\n".join(example.get("desired_job_titles", []))
    st.session_state["skills"] = "\n".join(example.get("skills", []))
    st.session_state["career_interests"] = "\n".join(example.get("career_interests", []))
    st.session_state["industries"] = "\n".join(example.get("preferred_industries", []))
    st.session_state["seniority"] = example.get("seniority_preferences", [])
    st.session_state["work_models"] = example.get("work_model_preferences", [])
    st.session_state["employment_types"] = example.get("employment_type_preferences", [])
    locations = example.get("location_preferences", [])
    st.session_state["location"] = locations[0] if locations and locations[0] in _LOCATION_OPTIONS else "No preference"
    years = example.get("years_experience")
    st.session_state["years_enabled"] = years is not None
    st.session_state["years_value"] = float(years or 0)
    experiences = example.get("professional_experience", [])
    st.session_state["experience_count"] = max(1, len(experiences))
    for index, item in enumerate(experiences):
        st.session_state[f"experience_title_{index}"] = item.get("title", "")
        st.session_state[f"experience_responsibilities_{index}"] = "\n".join(item.get("responsibilities", []))
    education = example.get("education", [])
    st.session_state["education_count"] = len(education)
    for index, item in enumerate(education):
        st.session_state[f"education_qualification_{index}"] = item.get("qualification", "")
        st.session_state[f"education_field_{index}"] = item.get("field") or ""
        st.session_state[f"education_institution_{index}"] = item.get("institution") or ""
    for key in list(st.session_state.keys()):
        if key.startswith("confirm_"):
            st.session_state.pop(key, None)
    for key in ("reviewed_signature", "run_result", "run_signature"):
        st.session_state.pop(key, None)


_LOCATION_OPTIONS = ["Germany", "Berlin", "Munich", "Hamburg", "Frankfurt", "Stuttgart", "Düsseldorf", "Cologne", "No preference"]


def _render_profile() -> tuple[CandidateProfile | None, str | None, dict]:
    st.subheader("Candidate profile")
    st.button("Clear All / Start New Profile", on_click=_reset_all, help="Clear profile inputs, review confirmation, run results, and live settings.")
    example = st.selectbox("Example profile", ["None", "Profile A", "Profile B"], key="example_choice", on_change=_invalidate_after_edit)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Load example", disabled=example == "None"):
            _apply_example(example[-1].lower())
            st.rerun()
    with col_b:
        st.caption("Examples populate these same editable fields. You can change every value before review.")

    st.text_input("Display name (non-identifying)", key="display_name", on_change=_invalidate_after_edit)
    st.text_area("Professional summary (optional)", key="summary", height=80, on_change=_invalidate_after_edit)
    col_a, col_b = st.columns(2)
    with col_a:
        st.text_area("Desired job titles — one per line *", key="desired_titles", height=100, on_change=_invalidate_after_edit)
    with col_b:
        st.text_area("Relevant skills — one per line *", key="skills", height=100, on_change=_invalidate_after_edit)
    col_a, col_b = st.columns(2)
    with col_a:
        st.text_area("Career interests (optional)", key="career_interests", height=80, on_change=_invalidate_after_edit)
    with col_b:
        st.text_area("Preferred industries (optional)", key="industries", height=80, on_change=_invalidate_after_edit)

    st.markdown("**Experience or project entries**")
    st.caption("At least one title and meaningful responsibility or project description are required.")
    experience_count = st.number_input("Number of entries", min_value=1, max_value=5, step=1, key="experience_count", on_change=_invalidate_after_edit)
    experiences = []
    for index in range(int(experience_count)):
        with st.expander("Experience / project" if index == 0 else f"Additional experience / project {index + 1}", expanded=index == 0):
            title = st.text_input("Title", key=f"experience_title_{index}", on_change=_invalidate_after_edit)
            responsibilities = st.text_area("Responsibilities or project description — one item per line", key=f"experience_responsibilities_{index}", height=90, on_change=_invalidate_after_edit)
            experiences.append({"title": title, "responsibilities": parse_lines(responsibilities)})

    col_a, col_b = st.columns(2)
    with col_a:
        provide_years = st.checkbox("Provide years of experience", key="years_enabled", on_change=_invalidate_after_edit)
        years = st.number_input("Years of experience", min_value=0.0, max_value=80.0, step=0.5, key="years_value", disabled=not provide_years, on_change=_invalidate_after_edit) if provide_years else None
    with col_b:
        education_count = st.number_input("Education entries (optional)", min_value=0, max_value=4, step=1, key="education_count", on_change=_invalidate_after_edit)
    education = []
    for index in range(int(education_count)):
        cols = st.columns(3)
        qualification = cols[0].text_input("Qualification", key=f"education_qualification_{index}", on_change=_invalidate_after_edit)
        field = cols[1].text_input("Field", key=f"education_field_{index}", on_change=_invalidate_after_edit)
        institution = cols[2].text_input("Institution", key=f"education_institution_{index}", on_change=_invalidate_after_edit)
        education.append({"qualification": qualification, "field": field or None, "institution": institution or None})

    st.markdown("**Preferences (optional)**")
    col_a, col_b = st.columns(2)
    with col_a:
        seniority = st.multiselect("Seniority preferences", ["entry-level", "junior", "mid-level", "senior", "lead", "director"], key="seniority", on_change=_invalidate_after_edit)
        work_models = st.multiselect("Work-model preferences", ["remote", "hybrid", "onsite"], key="work_models", on_change=_invalidate_after_edit)
    with col_b:
        employment_types = st.multiselect("Employment-type preferences", ["full-time", "part-time", "contract", "freelance"], key="employment_types", on_change=_invalidate_after_edit)
        location = st.selectbox("Preferred location within Germany", _LOCATION_OPTIONS, key="location", on_change=_invalidate_after_edit)

    fields = {
        "candidate_id": st.session_state["candidate_id"],
        "name": st.session_state["display_name"],
        "summary": st.session_state["summary"],
        "desired_job_titles": parse_lines(st.session_state["desired_titles"]),
        "skills": parse_lines(st.session_state["skills"]),
        "professional_experience": experiences,
        "years_experience": years,
        "education": [item for item in education if item["qualification"].strip() or item["field"] or item["institution"]],
        "career_interests": parse_lines(st.session_state["career_interests"]),
        "preferred_industries": parse_lines(st.session_state["industries"]),
        "seniority_preferences": seniority,
        "work_model_preferences": work_models,
        "employment_type_preferences": employment_types,
        "location_preferences": [] if location == "No preference" else [location],
    }
    try:
        return build_candidate_profile(fields), None, fields
    except Exception as exc:
        return None, str(exc), fields


def _render_settings() -> dict:
    st.sidebar.header("Execution")
    mode = st.sidebar.radio("Mode", ["Demo mode", "Live mode"], key="mode", on_change=_invalidate_after_edit)
    demo = mode == "Demo mode"
    if demo:
        st.sidebar.info("Demo mode uses synthetic fixture jobs and makes no external requests.")
        allow_esco = allow_company = allow_rag = False
    else:
        st.sidebar.warning("Live mode contacts external services. Provider limits apply; ESCO, Tavily, OpenAI and Pinecone can incur API charges.")
        allow_esco = st.sidebar.checkbox("Enable live ESCO enrichment", key="allow_esco", on_change=_invalidate_after_edit)
        allow_company = st.sidebar.checkbox("Enable live Tavily company research", key="allow_company", on_change=_invalidate_after_edit)
        allow_rag = st.sidebar.checkbox("Enable live OpenAI/Pinecone RAG retrieval", key="allow_rag", on_change=_invalidate_after_edit)
    with st.sidebar.expander("Bounded search settings"):
        queries = st.slider("Role queries", 1, 3, key="queries", on_change=_invalidate_after_edit)
        pages = st.slider("Pages per provider/query", 1, 5, key="pages", on_change=_invalidate_after_edit)
        top_n = st.slider("Maximum recommendations", 1, 5, key="top_n", on_change=_invalidate_after_edit)
        query_override = st.text_input("Optional single query override", key="query_override", on_change=_invalidate_after_edit)
    return {
        "mode": "demo" if demo else "live", "allow_live_esco": allow_esco,
        "allow_live_company_research": allow_company, "allow_live_rag": allow_rag,
        "queries": queries, "pages": pages, "top_n": top_n,
        "query_override": query_override, "output_dir": "reports/streamlit",
    }


def _invalidate_after_edit() -> None:
    invalidate_review_and_run(st.session_state)


def _signature(candidate: CandidateProfile, settings: dict) -> str:
    return input_fingerprint(candidate, settings)


def _render_bullets(values: list[str], empty_message: str) -> None:
    if values:
        st.markdown("\n".join(f"- {value}" for value in values))
    else:
        st.caption(empty_message)


def _render_results(result: dict) -> None:
    view = dashboard_view(result)
    st.header("Results")
    st.caption(view["score_interpretation"])
    st.write("**Workflow status:**", view["status_label"])
    cols = st.columns(3)
    cols[0].metric("Jobs discovered", view["discovered_jobs"])
    cols[1].metric("Jobs evaluated", view["evaluated_jobs"])
    cols[2].metric("Qualified recommendations", len(view["recommendations"]))
    st.caption("Data source: synthetic fixtures" if result.get("report_data", {}).get("search_scope", {}).get("offline") else "Data source: live job providers")
    if view["warnings"]:
        for warning in view["warnings"]:
            st.warning(warning)
    if view["errors"]:
        for error in view["errors"]:
            st.error(error)
    for error in view["nonfatal_errors"]:
        st.warning(f"Nonfatal provider issue: {error}")
    with st.expander("Run diagnostics"):
        st.write("Raw workflow status:", view["diagnostics"]["raw_status"])
        st.write("Original warnings:", view["diagnostics"]["warnings"] or ["None"])
        st.write("Errors:", view["diagnostics"]["errors"] or ["None"])
        st.write("Nonfatal errors:", view["diagnostics"]["nonfatal_errors"] or ["None"])
        st.write("Raw enrichment statuses:", view["diagnostics"]["enrichment_statuses"])

    st.subheader("Qualified recommendations")
    if not view["recommendations"]:
        st.info("No jobs met the existing recommendation criteria in this run.")
    for item in view["recommendations"]:
        job = item["job"]
        with st.expander(f"{job['title']} — {job.get('company', 'Unknown company')} · {item.get('score', 'N/A')}/100"):
            content = recommendation_content(item)
            st.write(f"**Location:** {job.get('location', 'Unknown')} · **Work model:** {job.get('work_model', 'unknown')}")
            st.write(f"**Evidence coverage:** {item.get('evidence_coverage', 0)}% · **Support:** {item.get('support_status', 'unknown')}")
            link = recommendation_link(job, offline=result.get("report_data", {}).get("search_scope", {}).get("offline", False))
            if link["url"]:
                st.link_button("Open original posting", link["url"])
            else:
                st.caption(link["message"])
            st.write("**Strengths**")
            _render_bullets(content["strengths"], content["strengths_empty"])
            st.write("**Potential gaps**")
            _render_bullets(content["gaps"], content["gaps_empty"])
            st.write("**Requirements to verify**")
            _render_bullets(content["requirements"], content["requirements_empty"])

    st.subheader("Potential matches requiring review")
    if not view["potential_matches"]:
        st.caption("No high-scoring provisional matches were reported.")
    for item in view["potential_matches"]:
        job = item["job"]
        with st.expander(f"{job['title']} — {job.get('company', 'Unknown company')} · {item.get('score')}/100 · {item.get('support_status')}"):
            content = recommendation_content(item)
            link = recommendation_link(job, offline=result.get("report_data", {}).get("search_scope", {}).get("offline", False))
            if link["url"]:
                st.link_button("Open original posting", link["url"])
            else:
                st.caption(link["message"])
            st.write("**Why it needs review**")
            _render_bullets(item.get("reasons", []), "No additional review reasons were recorded.")
            st.write("**Requirements to verify**")
            _render_bullets(content["requirements"], content["requirements_empty"])

    intelligence = view["career_intelligence"]
    st.subheader("Career intelligence")
    st.write(f"Personalized insight status: **{intelligence.get('status', 'unavailable')}**")
    if intelligence.get("message"):
        st.info(intelligence["message"])
    for insight in intelligence.get("insights", []):
        citation = insight.get("citation", {})
        with st.expander(insight.get("topic", "Career insight")):
            st.write("**Source finding:**", insight.get("source_finding"))
            st.write("**Candidate implication:**", insight.get("candidate_implication"))
            st.write("**Suggested next step:**", insight.get("suggested_next_step"))
            st.caption(f"{citation.get('document_title', 'Source unavailable')} · {citation.get('publisher', 'Publisher unavailable')} ({citation.get('publication_year', 'Year unavailable')}) · PDF page {citation.get('page_number', 'Not provided')}")
            if citation.get("source_url"):
                st.markdown(f"[Open source]({citation['source_url']})")
    market = view["market_intelligence"]
    st.write(f"**{view['enrichment_messages']['market_intelligence']}**")
    if market.get("status") in {"skipped_offline", "skipped_disabled", "skipped_no_candidate"}:
        st.caption("RAG retrieval was skipped; no market findings were inferred.")
    for evidence in market.get("evidence", []):
        with st.expander(evidence.get("document_title", "Retrieved source")):
            st.write(evidence.get("excerpt", "Text unavailable"))
            st.caption(f"{evidence.get('publisher', 'Publisher unavailable')} · {evidence.get('publication_year', 'Year unavailable')} · PDF page {evidence.get('page_number', 'Not provided')} · similarity {evidence.get('score', 'not provided')}")
            if evidence.get("source_url"):
                st.markdown(f"[Open source]({evidence['source_url']})")
    st.caption("RAG similarity scores are not factual confidence.")

    esco = view["esco"]
    company = view["company_research"]
    with st.expander("External enrichment details"):
        st.write(f"**{view['enrichment_messages']['esco']}**")
        st.caption(f"ESCO lookups: {esco.get('lookup_count', 0)}")
        for lookup in esco.get("lookups", []):
            result_data = lookup.get("esco_result", {})
            matches = [concept for concept in result_data.get("concepts", []) if concept.get("match_type") in {"exact", "alternative"}]
            if matches:
                st.write(lookup.get("term"), matches)
        st.write(f"**{view['enrichment_messages']['company_research']}**")
        for entry in company.get("results", []):
            st.write(f"{entry.get('company')}: {entry.get('status')} · identity {entry.get('identity_status', 'unverified')}")
            if entry.get("identity_status") != "verified":
                st.caption("Search results are not attributed as company facts because identity was not confirmed.")

    report = result.get("report_data", {})
    markdown, json_text = report_download_payloads(result.get("report_markdown", ""), report)
    st.subheader("Download reports")
    col_a, col_b = st.columns(2)
    col_a.download_button("Download Markdown report", markdown, file_name="career_intelligence_report.md", mime="text/markdown")
    col_b.download_button("Download JSON report", json_text, file_name="career_intelligence_report.json", mime="application/json")


def _render_candidate_review(candidate: CandidateProfile) -> None:
    summary = candidate_review_summary(candidate)
    st.subheader("Review and confirm your profile")
    st.markdown(f"**Candidate display name:** {summary['display_name']}")
    st.write("**Professional summary:**", summary["summary"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Desired job titles**")
        _render_bullets(summary["desired_job_titles"], "Not provided")
        st.markdown("**Skills**")
        _render_bullets(summary["skills"], "Not provided")
        st.markdown("**Career interests**")
        _render_bullets(summary["career_interests"], "Not provided")
        st.markdown("**Preferred industries**")
        _render_bullets(summary["preferred_industries"], "Not provided")
    with right:
        st.markdown("**Experience and projects**")
        for entry in summary["experience"]:
            st.markdown(f"- **{entry['title']}**")
            for responsibility in entry["responsibilities"]:
                st.markdown(f"  - {responsibility}")
        st.markdown(f"**Years of experience:** {summary['years_experience']}")
        st.markdown("**Education**")
        if summary["education"]:
            for entry in summary["education"]:
                st.markdown(f"- {entry['qualification']} · {entry['field']} · {entry['institution']}")
        else:
            st.caption("Not provided")
        st.markdown("**Preferences**")
        st.markdown(f"- Seniority: {', '.join(summary['seniority_preferences']) or 'Not specified'}")
        st.markdown(f"- Employment type: {', '.join(summary['employment_type_preferences']) or 'Not specified'}")
        st.markdown(f"- Work model: {', '.join(summary['work_model_preferences']) or 'Not specified'}")
        st.markdown(f"- Preferred location: {', '.join(summary['location_preferences']) or 'Not specified'}")
    st.info(summary["geographic_assumption"])


def main() -> None:
    st.set_page_config(page_title="Career Intelligence Agent", page_icon="🧭", layout="wide")
    _init_state()
    st.title("Career Intelligence Agent")
    st.caption("Evidence-based job discovery and career insights. Runs locally on this computer.")
    settings = _render_settings()
    if settings["mode"] == "demo":
        st.info("Demo mode uses synthetic fixture jobs, makes no external API requests, and labels results as synthetic.")
    else:
        st.warning("Live mode contacts Arbeitnow and Himalayas when you run the workflow. Optional enrichment requires separate opt-ins and may incur charges.")

    candidate, validation_error, fields = _render_profile()
    count, total, _ = profile_completeness(fields)
    st.markdown("**Profile completeness**")
    st.progress(count / total)
    st.caption(f"{count} of {total} profile sections populated. This is not a match score or evidence coverage.")
    st.info("More complete profile information can improve the specificity of recommendations. Missing qualifications are treated as unknown, not automatically as unmet.")
    st.caption("MVP geographic assumption: candidates reside in Germany and are authorized to work in Germany. Jobs are still checked for whether they can be performed from Germany; this does not infer work authorization from job location.")

    if validation_error:
        st.warning(validation_error)
    signature = _signature(candidate, settings) if candidate else None
    if signature is not None:
        invalidate_stale_run(st.session_state, signature)
    elif st.session_state.get("run_result") is not None:
        invalidate_stale_run(st.session_state, None)
    review_col, _ = st.columns([1, 3])
    with review_col:
        if st.button("Review candidate profile", disabled=candidate is None):
            st.session_state["reviewed_signature"] = signature
            st.session_state.pop("run_result", None)
            st.rerun()
    reviewed = candidate is not None and confirmation_is_current(st.session_state.get("reviewed_signature"), signature)
    confirmed = False
    if reviewed:
        _render_candidate_review(candidate)
        if settings["mode"] == "live":
            confirmed = st.checkbox("I reviewed this profile and authorize the selected live workflow requests.", key=f"confirm_{signature}")
        else:
            confirmed = st.checkbox("I reviewed this profile and want to run the local demo.", key=f"confirm_{signature}")
    run_disabled = not reviewed or not confirmed or candidate is None
    if st.button("Run Career Intelligence Agent", type="primary", disabled=run_disabled):
        try:
            if settings["mode"] == "live":
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env", override=False)
            with st.spinner("Running the existing Career Intelligence workflow…"):
                result = invoke_workflow(candidate, settings)
            st.session_state["run_result"] = result
            st.session_state["run_signature"] = signature
        except Exception as exc:
            st.error(f"Workflow execution failed: {type(exc).__name__}: {exc}")

    if st.session_state.get("run_result") and st.session_state.get("run_signature") == signature:
        _render_results(st.session_state["run_result"])


if __name__ == "__main__":
    main()
