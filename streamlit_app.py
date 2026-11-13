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
    citation_presentation,
    confirmation_is_current,
    dashboard_view,
    invalidate_review_and_run,
    input_fingerprint,
    invalidate_stale_run,
    load_example_profile,
    parse_lines,
    profile_completeness,
    recommendation_content,
    primary_reasons,
    job_context_label,
    synthesis_presentation,
    recommendation_link,
    report_download_payloads,
    reset_session_state,
    split_experience_title,
    invoke_workflow,
)
from career_agent.pdf_report import generate_career_report_pdf  # noqa: E402


def _init_state() -> None:
    st.session_state.setdefault("candidate_id", f"local-{uuid4().hex[:12]}")
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, list(value) if isinstance(value, list) else value)


def _reset_all() -> None:
    reset_session_state(st.session_state)


def _apply_example(name: str) -> None:
    example = load_example_profile(name)
    for key in list(st.session_state.keys()):
        if key.startswith((
            "experience_title_", "experience_responsibilities_", "experience_start_date_",
            "experience_end_date_", "experience_ongoing_", "education_",
        )):
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
        title, start_date, end_date, ongoing = split_experience_title(item.get("title", ""))
        st.session_state[f"experience_title_{index}"] = title
        st.session_state[f"experience_start_date_{index}"] = start_date
        st.session_state[f"experience_end_date_{index}"] = end_date
        st.session_state[f"experience_ongoing_{index}"] = ongoing
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
    st.subheader("Profile details")
    st.button("Clear All / Start New Profile", on_click=_reset_all, help="Clear profile inputs, review confirmation, run results, and live settings.")

    with st.container(border=False, key="cia_profile_direction"):
        st.markdown("#### Career direction")
        st.caption("Start with the roles and professional direction you want to explore.")
        example = st.selectbox("Example profile", ["None", "Profile A", "Profile B"], key="example_choice", on_change=_invalidate_after_edit)
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("Load example", disabled=example == "None"):
                _apply_example(example[-1].lower())
                st.rerun()
        with col_b:
            st.caption("Examples fill these same editable fields. Review and change them before continuing.")
        st.text_input("Display name (non-identifying)", key="display_name", on_change=_invalidate_after_edit)
        st.text_area("Professional summary (optional)", key="summary", height=80, on_change=_invalidate_after_edit)
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_area("Desired job titles (one per line) *", key="desired_titles", height=100, on_change=_invalidate_after_edit)
        with col_b:
            st.text_area("Relevant skills (one per line) *", key="skills", height=100, on_change=_invalidate_after_edit)
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_area("Career interests (optional)", key="career_interests", height=80, on_change=_invalidate_after_edit)
        with col_b:
            st.text_area("Preferred industries (optional)", key="industries", height=80, on_change=_invalidate_after_edit)

    with st.container(border=False, key="cia_profile_experience"):
        st.markdown("#### Experience and background")
        st.caption("Add professional or project evidence. Missing optional qualifications remain unknown.")
        st.caption("At least one title and meaningful responsibility or project description are required.")
        st.caption("Dates are optional, especially for standalone projects. Use YYYY, YYYY-MM, YYYY-MM-DD, or Month YYYY.")
        experience_count = st.number_input("Number of entries", min_value=1, max_value=5, step=1, key="experience_count", on_change=_invalidate_after_edit)
        experiences = []
        for index in range(int(experience_count)):
            with st.expander("Experience / project" if index == 0 else f"Additional experience / project {index + 1}", expanded=index == 0):
                title = st.text_input("Job or project title", key=f"experience_title_{index}", on_change=_invalidate_after_edit)
                date_col, ongoing_col = st.columns([1, 1])
                with date_col:
                    start_date = st.text_input(
                        "Start date (optional)", key=f"experience_start_date_{index}",
                        placeholder="YYYY-MM", on_change=_invalidate_after_edit,
                    )
                    end_date = st.text_input(
                        "End date (optional)", key=f"experience_end_date_{index}",
                        placeholder="YYYY-MM", disabled=st.session_state.get(f"experience_ongoing_{index}", False),
                        on_change=_invalidate_after_edit,
                    )
                with ongoing_col:
                    st.markdown(" ")
                    ongoing = st.checkbox(
                        "Ongoing role/project (Present)", key=f"experience_ongoing_{index}",
                        on_change=_invalidate_after_edit,
                    )
                responsibilities = st.text_area("Responsibilities or project description (one item per line)", key=f"experience_responsibilities_{index}", height=90, on_change=_invalidate_after_edit)
                experiences.append({
                    "title": title,
                    "start_date": start_date,
                    "end_date": end_date,
                    "ongoing": ongoing,
                    "responsibilities": parse_lines(responsibilities),
                })

        col_a, col_b = st.columns(2)
        with col_a:
            provide_years = st.checkbox("Provide years of experience", key="years_enabled", on_change=_invalidate_after_edit)
            years = st.number_input("Total professional experience (years)", min_value=0.0, max_value=80.0, step=0.5, key="years_value", disabled=not provide_years, help="Approximate total professional experience across all roles, not years per role.", on_change=_invalidate_after_edit) if provide_years else None
        with col_b:
            education_count = st.number_input("Education entries (optional)", min_value=0, max_value=4, step=1, key="education_count", on_change=_invalidate_after_edit)
        education = []
        for index in range(int(education_count)):
            cols = st.columns(3)
            qualification = cols[0].text_input("Qualification", key=f"education_qualification_{index}", on_change=_invalidate_after_edit)
            field = cols[1].text_input("Field", key=f"education_field_{index}", on_change=_invalidate_after_edit)
            institution = cols[2].text_input("Institution", key=f"education_institution_{index}", on_change=_invalidate_after_edit)
            education.append({"qualification": qualification, "field": field or None, "institution": institution or None})

    with st.container(border=False, key="cia_profile_preferences"):
        st.markdown("#### Work preferences")
        st.caption("These preferences guide discovery and eligibility checks.")
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
    st.sidebar.header("Run settings")
    mode = st.sidebar.radio("Choose a mode", ["Demo mode", "Live mode"], key="mode", on_change=_invalidate_after_edit)
    demo = mode == "Demo mode"
    if demo:
        with st.sidebar.container(key="cia_sidebar_mode_notice"):
            st.info("Demo mode uses synthetic fixture jobs and makes no external requests.")
        allow_esco = allow_company = allow_rag = allow_synthesis = False
    else:
        with st.sidebar.container(key="cia_sidebar_mode_notice"):
            st.info("Live mode contacts external services. Provider limits apply; ESCO, Tavily, OpenAI and Pinecone can incur API charges.")
        with st.sidebar.expander("Optional research and AI features", expanded=False):
            st.caption("Each option is off until selected. These services can incur usage charges.")
            allow_esco = st.checkbox("ESCO skill enrichment", key="allow_esco", help="Look up skills and occupations in the European classification.", on_change=_invalidate_after_edit)
            allow_company = st.checkbox("Company research", key="allow_company", help="Search for source material about recommended employers.", on_change=_invalidate_after_edit)
            allow_rag = st.checkbox("Labour-market research", key="allow_rag", help="Retrieve evidence from the career knowledge base.", on_change=_invalidate_after_edit)
            allow_synthesis = st.checkbox(
                "Write grounded career insights", key="allow_synthesis",
                help="One bounded OpenAI request using retrieved evidence. Requires labour-market research.",
                disabled=not allow_rag, on_change=_invalidate_after_edit,
            )
    with st.sidebar.expander("Bounded search settings"):
        queries = st.slider("Role queries", 1, 3, key="queries", on_change=_invalidate_after_edit)
        pages = st.slider("Pages per provider/query", 1, 5, key="pages", on_change=_invalidate_after_edit)
        top_n = st.slider("Maximum recommendations", 1, 5, key="top_n", on_change=_invalidate_after_edit)
        query_override = st.text_input("Optional single query override", key="query_override", on_change=_invalidate_after_edit)
    return {
        "mode": "demo" if demo else "live", "allow_live_esco": allow_esco,
        "allow_live_company_research": allow_company, "allow_live_rag": allow_rag,
        "allow_llm_synthesis": allow_synthesis,
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


def _render_job_transparency(content: dict) -> None:
    career_direction = content.get("career_direction") or {}
    if career_direction.get("status") not in {None, "unassessed"}:
        evidence = "; ".join(career_direction.get("evidence") or []) or "No specific alignment evidence was recorded."
        st.markdown("**Career-direction alignment**")
        st.caption(f"{career_direction['status'].title()}: {evidence}")

    details = content.get("job_highlights") or {}
    st.markdown("**Key job requirements & working conditions**")
    provider_model = details.get("provider_work_model")
    if provider_model:
        st.caption(f"Provider-listed work model: {provider_model.title()}. Description-based conditions below are shown separately.")
    items = details.get("items") or []
    if items:
        _render_bullets(
            [f"**{item['category']}:** {item['text']}" for item in items],
            "No additional job details were extracted.",
        )
    elif not details.get("description_available"):
        st.caption("Job description not provided; no requirements or working conditions were inferred.")
    else:
        st.caption("No concise requirements or working conditions could be reliably extracted from the description.")


def _render_citations(paragraph: dict) -> None:
    citations = paragraph.get("citations")
    if not isinstance(citations, list):
        citations = [
            {"kind": "market_source", "reference_id": None, **source}
            for source in paragraph.get("sources", [])
        ]
    for raw in citations:
        citation = citation_presentation(raw)
        label = citation["label"]
        if citation["kind"] == "market_source":
            page = citation.get("page")
            if isinstance(page, float) and page.is_integer():
                page = int(page)
            page_text = f" | PDF page {page}" if page not in (None, "", "Not provided") else ""
            st.caption(f"{label}: {citation['title']} | {citation['publisher']} ({citation['year']}){page_text}")
        else:
            st.caption(f"{label}: {citation['title']}")
        url = citation.get("url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            st.markdown(f"[Open cited source]({url})")


def _render_results(result: dict) -> None:
    view = dashboard_view(result)
    report = result.get("report_data", {})
    st.header("Run results")
    if view["errors"]:
        st.error(view["status_label"])
    elif view["warnings"] or view["nonfatal_errors"]:
        st.warning(view["status_label"])
    elif view["status"] == "completed_no_recommendations":
        st.info(view["status_label"])
    else:
        st.success(view["status_label"])

    metrics = st.columns(4)
    for index, label, value in (
        (1, "Jobs found", view["discovered_jobs"]),
        (2, "Jobs evaluated", view["evaluated_jobs"]),
        (3, "Recommended", len(view["recommendations"])),
        (4, "Need review", len(view["potential_matches"])),
    ):
        with metrics[index - 1]:
            with st.container(border=False, key=f"cia_metric_{index}"):
                st.metric(label, value)
    st.caption(f"{view['data_source_label']} | Scores show indicative evidence alignment, not hiring probability.")

    for warning in view["warnings"]:
        st.warning(warning)
    for issue in view["nonfatal_errors"]:
        st.warning(issue)
    for error in view["errors"]:
        st.error(error)

    st.subheader("Recommended roles")
    st.caption("Matching evidence coverage measures evaluated signals only; it does not indicate full qualification or hiring probability.")
    if not view["recommendations"]:
        count = len(view["potential_matches"])
        follow_up = (
            f" {count} potential match{'es' if count != 1 else ''} still need review below."
            if count else " Try adjusting the profile or search settings before running again."
        )
        st.info("No roles met the current recommendation criteria. This is a valid result, not an application error." + follow_up)

    for index, item in enumerate(view["recommendations"], start=1):
        job = item["job"]
        content = recommendation_content(item)
        with st.container(border=False, key=f"cia_recommendation_card_{index}"):
            title_col, score_col = st.columns([4, 1])
            title_col.markdown(f"### {job.get('title', 'Untitled role')}")
            title_col.write(job.get("company", "Company not provided"))
            score_col.metric("Indicative match", f"{item.get('score', 'N/A')}/100")
            st.caption(f"{job_context_label(job)} | Profile evidence alignment")
            _render_job_transparency(content)
            link = recommendation_link(job, offline=report.get("search_scope", {}).get("offline", False))
            if link["url"]:
                st.link_button("View original job posting", link["url"])
            else:
                st.caption(link["message"])
            strength_col, verify_col = st.columns(2)
            with strength_col:
                st.markdown("**Why it may fit**")
                _render_bullets(primary_reasons(item), content["strengths_empty"])
            with verify_col:
                st.markdown("**Qualifications to verify**")
                _render_bullets(content["requirements"], content["requirements_empty"])
            if content["requirements"]:
                st.caption("“Not documented” means the profile does not provide evidence; it does not establish that the candidate lacks the qualification.")
            if any(item.get(key) for key in (
                "gaps", "strengths", "reasons", "score_breakdown",
                "geographic_evidence", "seniority_evidence", "uncertainties", "warnings",
            )):
                with st.expander("Full matching evidence and score details"):
                    st.markdown("**All recorded strengths**")
                    _render_bullets(content["strengths"], content["strengths_empty"])
                    st.markdown("**Additional matching reasons**")
                    _render_bullets(item.get("reasons", []), "No additional reasons were recorded.")
                    st.markdown("**Potential gaps**")
                    _render_bullets(content["gaps"], content["gaps_empty"])
                    if item.get("evidence_coverage") is not None:
                        st.write(f"Matching evidence coverage: {item['evidence_coverage']}% of evaluated signals")
                    if item.get("score_breakdown"):
                        st.write("Score details:", item["score_breakdown"])
                    if item.get("geographic_evidence"):
                        st.markdown("**Geographic evidence**")
                        _render_bullets(item["geographic_evidence"], "No geographic evidence details.")
                    if item.get("seniority_evidence"):
                        st.markdown("**Seniority evidence**")
                        _render_bullets(item["seniority_evidence"], "No seniority evidence details.")
                    if item.get("uncertainties"):
                        st.markdown("**Uncertainties**")
                        _render_bullets(item["uncertainties"], "None recorded.")
                    if item.get("warnings"):
                        st.markdown("**Warnings**")
                        _render_bullets(item["warnings"], "None recorded.")
            st.caption("A recommendation is an evidence-based lead, not confirmation that every employer requirement is met.")

    st.subheader("Potential matches to review")
    st.caption("These roles did not qualify as recommendations. Review the evidence and requirements before treating them as options.")
    st.caption("Matching evidence coverage measures evaluated signals only; it does not indicate full qualification or hiring probability.")
    if not view["potential_matches"]:
        st.caption("No additional high-scoring matches were flagged for review.")
    for index, item in enumerate(view["potential_matches"], start=1):
        job = item["job"]
        content = recommendation_content(item)
        with st.container(border=False, key=f"cia_review_card_{index}"):
            title_col, score_col = st.columns([4, 1])
            title_col.markdown(f"### {job.get('title', 'Untitled role')}")
            title_col.write(job.get("company", "Company not provided"))
            score_col.metric("Indicative match", f"{item.get('score', 'N/A')}/100")
            st.caption(f"{job_context_label(job)} | Needs evidence review")
            _render_job_transparency(content)
            link = recommendation_link(job, offline=report.get("search_scope", {}).get("offline", False))
            if link["url"]:
                st.link_button("View original job posting", link["url"])
            else:
                st.caption(link["message"])
            st.markdown("**Why it needs review**")
            _render_bullets(primary_reasons(item), "No additional review reasons were recorded.")
            st.markdown("**Qualifications to verify**")
            _render_bullets(content["requirements"], content["requirements_empty"])
            if content["requirements"]:
                st.caption("“Not documented” means the profile does not provide evidence; it does not establish that the candidate lacks the qualification.")
            with st.expander("Full review evidence"):
                st.markdown("**All recorded reasons**")
                _render_bullets(item.get("reasons", []), "No additional review reasons were recorded.")
                st.markdown("**Potential gaps**")
                _render_bullets(content["gaps"], content["gaps_empty"])
                if item.get("evidence_coverage") is not None:
                    st.write(f"Matching evidence coverage: {item['evidence_coverage']}% of evaluated signals")
                for label, key in (("Geographic evidence", "geographic_evidence"), ("Seniority evidence", "seniority_evidence"),
                                   ("Uncertainties", "uncertainties"), ("Warnings", "warnings")):
                    if item.get(key):
                        st.markdown(f"**{label}**")
                        _render_bullets(item[key], "None recorded.")

    st.header("Career and market intelligence")
    synthesis = view["intelligence_synthesis"]
    demo_mode = report.get("search_scope", {}).get("offline", False)
    if demo_mode:
        st.info("Demo Mode is fully offline. Market Intelligence and grounded AI synthesis are available in Live Mode.")
    synthesis_view = synthesis_presentation(synthesis, demo_mode)
    if synthesis_view["kind"] == "success":
        st.success(synthesis_view["message"])
    elif synthesis_view["kind"] == "info":
        st.info(synthesis_view["message"])
    elif synthesis_view["kind"] == "caption":
        st.caption(synthesis_view["message"])
    if synthesis.get("error") and not demo_mode:
        st.caption(synthesis["error"])

    with st.container(border=False, key="cia_intelligence"):
        for section_key, heading, intro in (
            ("market_intelligence", "Market Intelligence", "What the retrieved labour-market sources report."),
            ("personalized_career_intelligence", "Personalized Career Intelligence", "How that evidence relates to the candidate's documented profile and career direction."),
        ):
            st.subheader(heading)
            st.caption(intro)
            paragraphs = synthesis.get(section_key, [])
            if paragraphs:
                for index, paragraph in enumerate(paragraphs, start=1):
                    with st.container(border=False, key=f"cia_intelligence_card_{section_key}_{index}"):
                        st.write(paragraph.get("text", ""))
                        _render_citations(paragraph)
            elif section_key == "personalized_career_intelligence":
                intelligence = view["career_intelligence"]
                if intelligence.get("message"):
                    st.info(intelligence["message"])
                for insight in intelligence.get("insights", []):
                    citation = insight.get("citation", {})
                    with st.container(border=True):
                        st.markdown(f"**{insight.get('topic', 'Career insight')}**")
                        st.write("**Source finding:**", insight.get("source_finding"))
                        st.write("**Candidate implication:**", insight.get("candidate_implication"))
                        st.write("**Suggested next step:**", insight.get("suggested_next_step"))
                        if citation:
                            _render_citations({"citations": [{"kind": "market_source", **citation}]})
            elif view["market_intelligence"].get("evidence"):
                st.info("Research sources were retrieved, but no generated market summary is available. Review the passages in Diagnostics and retrieved evidence.")
            elif view["market_intelligence"].get("status") in {"skipped_offline", "skipped_disabled", "skipped_no_candidate"}:
                if not demo_mode:
                    st.info(view["enrichment_messages"]["market_intelligence"])
            else:
                st.caption("No market evidence was retrieved. No market findings are inferred.")

    market = view["market_intelligence"]
    with st.expander("Run status and diagnostics"):
        summary = view["diagnostics"]["summary"]
        left, right = st.columns(2)
        for column, entries in (
            (left, (
                ("Job discovery", summary["job_discovery"]),
                ("Matching", summary["matching"]),
                ("Recommendations", summary["recommendations"]),
                ("Market Intelligence", summary["market_intelligence"]),
            )),
            (right, (
                ("Grounded synthesis", summary["llm_synthesis"]),
                ("ESCO enrichment", summary["esco"]),
                ("Company research", summary["company_research"]),
                ("Overall workflow", view["status_label"]),
            )),
        ):
            for label, value in entries:
                column.markdown(f"**{label}:** {value}")
        if summary["warnings"]:
            st.warning("Warnings: " + " ".join(summary["warnings"]))
        if summary["errors"]:
            st.error("Errors: " + " ".join(summary["errors"]))
        if summary["nonfatal_errors"]:
            st.warning("Other issues: " + " ".join(summary["nonfatal_errors"]))

        with st.expander("Provider counts and filter details"):
            search = report.get("search_scope", {})
            provider = search.get("provider_diagnostics", {})
            st.write(f"Mode: {'Demo' if search.get('offline') else 'Live'}")
            st.write("Search queries:", search.get("queries", []))
            st.write("Providers:", search.get("providers", []))
            st.write("Provider counts:", provider.get("provider_counts", {}))
            st.write("Final provider counts:", provider.get("final_provider_counts", {}))
            st.write("Duplicates removed:", provider.get("duplicates_removed", 0))
            st.write("Filter exclusions:", provider.get("filter_exclusions", {}))
            st.write("Provider errors:", provider.get("errors", {}))
            st.write("Raw workflow status:", view["diagnostics"]["raw_status"])
            st.write("Enrichment statuses:", view["diagnostics"]["enrichment_statuses"])

        with st.expander("Retrieved evidence and source metadata"):
            st.caption("Similarity ranks retrieval relevance; it is not factual confidence.")
            for evidence in market.get("evidence", []):
                with st.container(border=True):
                    st.markdown(f"**{evidence.get('document_title', 'Retrieved source')}**")
                    st.write(evidence.get("excerpt", "Text unavailable"))
                    st.caption(f"{evidence.get('publisher', 'Publisher unavailable')} | {evidence.get('publication_year', 'Year unavailable')} | PDF page {evidence.get('page_number', 'Not provided')} | Similarity {evidence.get('score', 'not provided')}")
                    if evidence.get("source_url"):
                        st.markdown(f"[Open source]({evidence['source_url']})")
                    if evidence.get("chunk_id"):
                        st.caption(f"Internal source ID: {evidence['chunk_id']}")

        with st.expander("External service details"):
            esco = view["esco"]
            company = view["company_research"]
            st.write(view["enrichment_messages"]["esco"])
            st.write(f"ESCO lookups: {esco.get('lookup_count', 0)} / {esco.get('lookup_limit', 'not specified')}")
            for lookup in esco.get("lookups", []):
                result_data = lookup.get("esco_result", {})
                matches = [concept for concept in result_data.get("concepts", [])
                           if concept.get("match_type") in {"exact", "alternative"}]
                if matches:
                    st.markdown(f"**{lookup.get('term', 'Skill lookup')}**")
                    st.write(matches)
            st.write("Company research status:", view["enrichment_messages"]["company_research"])
            st.write("Company research results:")
            for entry in company.get("results", []):
                st.write(f"{entry.get('company')}: {entry.get('status')} | identity {entry.get('identity_status', 'unverified')}")
                if entry.get("identity_status") != "verified":
                    st.caption("Search results are not attributed as company facts because identity was not confirmed.")
                for source in entry.get("sources", []):
                    title = source.get("title") or "Search result"
                    url = source.get("url")
                    if isinstance(url, str) and url.startswith(("https://", "http://")):
                        st.markdown(f"[{title}]({url})")
                    elif title:
                        st.caption(title)
            st.write("Synthesis diagnostics:", {key: synthesis.get(key) for key in ("status", "model", "request_count", "request_limit", "failure_stage", "error")})

    report_markdown, json_text = report_download_payloads(result.get("report_markdown", ""), report)
    pdf_bytes = result.get("_pdf_report_bytes")
    if not isinstance(pdf_bytes, bytes):
        pdf_bytes = generate_career_report_pdf(report)
        # Keep the rendered export with this session's completed result so UI
        # reruns do not rebuild a multi-page PDF. It is not added to report JSON.
        result["_pdf_report_bytes"] = pdf_bytes
    with st.container(border=False, key="cia_exports"):
        st.subheader("Take your report with you")
        st.caption("Download the report as PDF, Markdown, or structured JSON.")
        col_a, col_b, col_c = st.columns(3)
        col_a.download_button("Download Markdown report", report_markdown, file_name="career_intelligence_report.md", mime="text/markdown")
        col_b.download_button("Download JSON report", json_text, file_name="career_intelligence_report.json", mime="application/json")
        col_c.download_button("Download PDF Report", pdf_bytes, file_name="career_intelligence_report.pdf", mime="application/pdf")


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


def _render_visual_theme() -> None:
    """Apply the shared dark emerald design to native Streamlit controls and panels."""
    st.html(
        """<style>
        :root {
            --cia-bg: #07151C; --cia-surface: #0D2430; --cia-raised: #173B42;
            --cia-emerald: #1EEA91; --cia-turquoise: #48E0D0;
            --cia-text: #F2FAF8; --cia-muted: #AEC6C5;
            --cia-border: #28545A; --cia-amber: #E9B765;
        }
        .block-container {max-width: 1360px; padding-top: 3.5rem; padding-bottom: 3rem;}
        [data-testid="stAppViewContainer"] {
            background-color: var(--cia-bg);
            background-image: radial-gradient(ellipse at 12% 2%, rgba(30, 234, 145, .12), transparent 34%),
                              radial-gradient(ellipse at 90% 9%, rgba(72, 224, 208, .11), transparent 31%);
            color: var(--cia-text);
        }
        [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
        [data-testid="stMain"] h4, [data-testid="stMain"] p, [data-testid="stMain"] label {color: var(--cia-text);}
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {color: var(--cia-muted);}
        [data-testid="stHeader"] {background: rgba(7, 21, 28, .78);}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #0C2029, #091820); border-right: 1px solid var(--cia-border);}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: var(--cia-muted);}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {color: var(--cia-text);}
        [data-testid="stSidebar"] [data-testid="stRadio"] input,
        [data-testid="stCheckbox"] input {accent-color: var(--cia-emerald);}
        a {color: var(--cia-turquoise);}
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {
            background: var(--cia-surface) !important; color: var(--cia-text) !important;
            border: 1px solid var(--cia-border) !important; border-radius: 10px !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background: var(--cia-surface) !important; color: var(--cia-text) !important;
            border: 1px solid var(--cia-border) !important; border-radius: 10px !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] input,
        [data-testid="stMultiSelect"] [data-baseweb="select"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] div {color: var(--cia-text) !important;}
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: #0F766E !important; color: #F2FAF8 !important;
            border: 1px solid rgba(72, 224, 208, .35) !important; border-radius: 7px !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {color: #F2FAF8 !important; fill: #F2FAF8 !important;}
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] {
            background: #173B42 !important; color: var(--cia-text) !important; border-color: var(--cia-border) !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="menu"],
        [data-baseweb="menu"] {
            background: #0D2430 !important; color: var(--cia-text) !important;
            border: 1px solid #28545A !important; box-shadow: 0 14px 32px rgba(0, 0, 0, .42) !important;
        }
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="menu"] [role="menuitem"] {background: #0D2430 !important; color: var(--cia-text) !important;}
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"][aria-selected="true"],
        [data-baseweb="menu"] [role="menuitem"]:hover,
        [data-baseweb="menu"] [role="menuitem"][aria-selected="true"] {
            background: #173B42 !important; color: #F2FAF8 !important;
        }
        [data-baseweb="popover"] [role="option"][aria-selected="true"] {box-shadow: inset 3px 0 #1EEA91;}
        [data-testid="stCheckbox"] input[type="checkbox"],
        [data-testid="stRadio"] input[type="radio"] {accent-color: #1EEA91;}
        [data-testid="stCheckbox"] [role="checkbox"],
        [data-testid="stRadio"] [role="radio"] {border-color: #48E0D0 !important;}
        [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"],
        [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
            background-color: #0F766E !important; border-color: #1EEA91 !important;
        }
        [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder {color: #91A9A8;}
        [data-testid="stTextInput"] input:focus-visible, [data-testid="stTextArea"] textarea:focus-visible,
        [data-testid="stNumberInput"] input:focus-visible, [data-testid="stBaseButton"]:focus-visible,
        [data-testid="stCheckbox"] input:focus-visible, [data-testid="stRadio"] input:focus-visible {
            outline: 2px solid var(--cia-turquoise) !important; outline-offset: 2px;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
        [data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {
            outline: 2px solid var(--cia-turquoise) !important; outline-offset: 2px;
        }
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(100deg, #1EEA91, #48E0D0) !important; color: #06221D !important;
            border: 0 !important; border-radius: 10px !important; font-weight: 750 !important;
            box-shadow: 0 5px 18px rgba(30, 234, 145, .19);
        }
        [data-testid="stBaseButton-primary"]:hover {filter: brightness(1.08); box-shadow: 0 7px 22px rgba(30, 234, 145, .28);}
        [data-testid="stBaseButton-secondary"] {
            background: #0D2430 !important; color: #F2FAF8 !important;
            border: 1px solid #28545A !important; border-radius: 10px !important;
        }
        [data-testid="stBaseButton-secondary"]:hover {background: #173B42 !important; border-color: #48E0D0 !important;}
        [data-testid="stExpander"] {border: 1px solid rgba(112, 207, 190, .18); border-radius: 12px; background: rgba(16, 34, 45, .48);}
        [data-testid="stAlert"] {
            background: #112D3A !important; border: 1px solid #286073 !important;
            border-radius: 12px !important; color: #F2FAF8 !important;
        }
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {color: #F2FAF8 !important;}
        .st-key-cia_sidebar_mode_notice [data-testid="stIconMaterial"],
        .st-key-cia_main_mode_notice [data-testid="stIconMaterial"] {color: #48E0D0 !important;}
        .st-key-cia_validation_notice [data-testid="stAlert"] {
            border-left: 3px solid var(--cia-amber) !important;
        }
        [class*="st-key-cia_profile_"] {
            border: 1px solid var(--cia-border); border-radius: 16px; padding: .8rem 1rem;
            margin: .7rem 0; background: linear-gradient(145deg, rgba(20, 49, 58, .86), rgba(13, 31, 41, .88));
        }
        [class*="st-key-cia_profile_"] h4 {color: var(--cia-turquoise); margin-bottom: .1rem;}
        .st-key-cia_review_workspace {border: 1px solid rgba(72, 224, 208, .28); border-radius: 16px; padding: 1rem; background: #10222D;}
        .st-key-cia_exports {border: 1px solid var(--cia-border); border-radius: 16px; padding: .7rem 1rem; background: linear-gradient(135deg, #12313A, #10222D);}
        .st-key-cia_hero {
            border: 1px solid rgba(72, 224, 208, .32); border-radius: 22px; padding: 1.35rem 1.45rem;
            margin: .3rem 0 1rem; background: linear-gradient(125deg, #103F3C 0%, #12313D 53%, #10222D 100%);
            box-shadow: 0 18px 48px rgba(0, 0, 0, .26), inset 0 1px rgba(255,255,255,.06);
        }
        .st-key-cia_hero [data-testid="stCaptionContainer"] {color: var(--cia-emerald); letter-spacing: .16em; font-weight: 700;}
        .st-key-cia_hero h2 {font-size: clamp(2rem, 3.2vw, 3.1rem); line-height: 1.06; margin: .1rem 0 .06rem; letter-spacing: -.035em;}
        .st-key-cia_hero h2 + h2 {margin-top: 0;}
        .st-key-cia_hero p {max-width: 620px; color: #D2E4E0; font-size: 1.03rem; line-height: 1.6;}
        .st-key-cia_hero [data-testid="stMarkdownContainer"] p strong {color: var(--cia-emerald);}
        .st-key-cia_step_1, .st-key-cia_step_2, .st-key-cia_step_3, .st-key-cia_step_4 {
            height: 100%; border: 1px solid var(--cia-border); border-radius: 14px;
            padding: .35rem .55rem; background: linear-gradient(145deg, rgba(20, 49, 58, .82), rgba(12, 29, 39, .88));
        }
        .st-key-cia_step_1 [data-testid="stCaptionContainer"], .st-key-cia_step_2 [data-testid="stCaptionContainer"],
        .st-key-cia_step_3 [data-testid="stCaptionContainer"], .st-key-cia_step_4 [data-testid="stCaptionContainer"] {color: var(--cia-muted);}
        .st-key-cia_metric_1, .st-key-cia_metric_2, .st-key-cia_metric_3, .st-key-cia_metric_4 {
            min-height: 102px; border: 1px solid var(--cia-border); border-radius: 15px;
            padding: .55rem .8rem; background: linear-gradient(145deg, #14313A, #10222D);
            box-shadow: inset 0 1px rgba(255,255,255,.04);
        }
        [data-testid="stMetricValue"] {font-size: 2.15rem; color: var(--cia-emerald); font-weight: 750;}
        [data-testid="stMetricLabel"] {color: var(--cia-muted); font-size: .88rem;}
        [class*="st-key-cia_recommendation_card_"] {
            border: 1px solid rgba(30, 234, 145, .3); border-radius: 17px; padding: .45rem .7rem;
            background: linear-gradient(145deg, rgba(16, 50, 48, .90), rgba(14, 31, 41, .96));
            box-shadow: 0 10px 28px rgba(0,0,0,.16);
        }
        [class*="st-key-cia_review_card_"] {
            border: 1px solid rgba(233, 183, 101, .35); border-radius: 17px; padding: .45rem .7rem;
            background: linear-gradient(145deg, rgba(57, 46, 29, .56), rgba(18, 31, 39, .96));
        }
        .st-key-cia_intelligence {
            border: 1px solid rgba(72, 224, 208, .27); border-radius: 19px; padding: .7rem 1rem;
            background: linear-gradient(145deg, rgba(17, 42, 53, .92), rgba(12, 27, 37, .96));
            box-shadow: 0 14px 34px rgba(0,0,0,.18);
        }
        .st-key-cia_intelligence h3 {color: var(--cia-turquoise);}
        .st-key-cia_intelligence a {color: var(--cia-turquoise);}
        [class*="st-key-cia_intelligence_card_"] {border-left: 3px solid rgba(30, 234, 145, .68); padding: .25rem .8rem; margin: .35rem 0;}
        [class*="st-key-cia_intelligence_card_"] [data-testid="stCaptionContainer"] {color: #A9D8CF;}
        @media (max-width: 800px) {
            .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 2.8rem;}
            .st-key-cia_hero {padding: 1rem; border-radius: 17px;}
            [data-testid="stMetricValue"] {font-size: 1.65rem;}
        }
        </style>"""
    )


def _hero_dashboard_preview() -> str:
    """Fixed, decorative product illustration; it contains no candidate or report data."""
    return """<style>
    .cia-preview {position:relative; min-height:340px; padding:18px; overflow:hidden; border-radius:18px;
      border:1px solid rgba(111,231,208,.28); color:#F2FAF8;
      background:radial-gradient(circle at 78% 18%,rgba(30,234,145,.19),transparent 30%),
                 linear-gradient(145deg,#0A1A24,#102C37 62%,#123B3C);
      box-shadow:inset 0 1px rgba(255,255,255,.07),0 16px 38px rgba(0,0,0,.27);
      font-family:inherit}
    .cia-preview * {box-sizing:border-box}
    .cia-preview-head {display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:16px}
    .cia-preview-brand {display:flex;align-items:center;gap:9px;color:#EAF8F4;font-size:12px;font-weight:700;letter-spacing:.04em}
    .cia-preview-mark {width:27px;height:27px;border-radius:9px;display:grid;place-items:center;color:#06221D;
      background:linear-gradient(135deg,#1EEA91,#48E0D0);font-size:15px}
    .cia-preview-badge {border:1px solid rgba(72,224,208,.3);border-radius:99px;padding:5px 8px;color:#8BE9D3;
      font-size:9px;font-weight:700;letter-spacing:.1em}
    .cia-preview-layout {display:grid;grid-template-columns:1fr 28px 1fr;align-items:center;gap:8px}
    .cia-preview-stack {display:grid;gap:9px}
    .cia-preview-node {position:relative;border:1px solid rgba(148,196,196,.18);border-radius:12px;padding:11px 12px;
      background:linear-gradient(145deg,rgba(20,49,58,.96),rgba(12,29,39,.96));}
    .cia-preview-node small {display:block;color:#9CBDB8;font-size:9px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px}
    .cia-preview-node strong {display:block;color:#F2FAF8;font-size:12px;line-height:1.3}
    .cia-preview-node span {display:block;color:#AEC6C5;font-size:10px;margin-top:4px}
    .cia-preview-node.accent {border-color:rgba(30,234,145,.35);background:linear-gradient(145deg,rgba(14,66,55,.95),rgba(14,39,45,.96))}
    .cia-preview-link {height:2px;background:linear-gradient(90deg,#1EEA91,#48E0D0);position:relative;opacity:.9}
    .cia-preview-link:after {content:"";position:absolute;right:-1px;top:-4px;width:9px;height:9px;border-radius:50%;background:#48E0D0;box-shadow:0 0 13px #48E0D0}
    .cia-preview-pulse {height:2px;background:linear-gradient(90deg,transparent,rgba(72,224,208,.75),transparent);margin:14px 0 11px}
    .cia-preview-bottom {display:grid;grid-template-columns:1fr 1fr;gap:9px}
    .cia-preview-chip {display:flex;align-items:center;gap:8px;border:1px solid rgba(148,196,196,.15);border-radius:10px;
      padding:9px 10px;background:rgba(8,23,31,.7);color:#B8CFCC;font-size:10px}
    .cia-preview-dot {width:7px;height:7px;flex:0 0 7px;border-radius:50%;background:#1EEA91;box-shadow:0 0 10px rgba(30,234,145,.6)}
    @media(max-width:850px){.cia-preview{min-height:300px;padding:14px}.cia-preview-node{padding:9px}.cia-preview-node strong{font-size:11px}}
    </style>
    <div class="cia-preview" role="img" aria-label="Illustrative preview of profile, role matching, evidence review, and market intelligence">
      <div class="cia-preview-head">
        <div class="cia-preview-brand"><span class="cia-preview-mark">✦</span><span>Career intelligence</span></div>
        <span class="cia-preview-badge">ILLUSTRATIVE PREVIEW</span>
      </div>
      <div class="cia-preview-layout">
        <div class="cia-preview-stack">
          <div class="cia-preview-node"><small>01 · Candidate context</small><strong>Skills and experience</strong><span>Profile evidence</span></div>
          <div class="cia-preview-node"><small>02 · Opportunity discovery</small><strong>Relevant role signals</strong><span>Source-linked listings</span></div>
        </div>
        <div class="cia-preview-link" aria-hidden="true"></div>
        <div class="cia-preview-node accent"><small>03 · Transparent evaluation</small><strong>Evidence-led fit</strong><span>Requirements to verify</span></div>
      </div>
      <div class="cia-preview-pulse" aria-hidden="true"></div>
      <div class="cia-preview-bottom">
        <div class="cia-preview-chip"><i class="cia-preview-dot"></i>Traceable job evidence</div>
        <div class="cia-preview-chip"><i class="cia-preview-dot"></i>Cited market insights</div>
      </div>
    </div>"""


def main() -> None:
    st.set_page_config(page_title="Career Intelligence Agent", page_icon="🧭", layout="wide")
    _init_state()
    _render_visual_theme()
    with st.container(border=False, key="cia_hero"):
        hero_copy, hero_visual = st.columns([1.12, .88], gap="large", vertical_alignment="center")
        with hero_copy:
            st.caption("CAREER INTELLIGENCE AGENT")
            st.markdown("## Discover :green[career opportunities].\n## Understand your potential.")
            st.write(
                "AI-assisted career intelligence connecting your experience to real opportunities, "
                "transparent matching evidence, and cited labour-market insights."
            )
            benefit_cols = st.columns(3)
            for column, benefit in zip(
                benefit_cols,
                ("Evidence-led matching", "Qualifications to verify", "Cited market context"),
            ):
                with column:
                    st.caption(f"✦ {benefit}")
        with hero_visual:
            st.html(_hero_dashboard_preview())
    step_cols = st.columns(4)
    for column, number, title, description in zip(
        step_cols,
        ("01", "02", "03", "04"),
        ("Build a profile", "Explore roles", "Review evidence", "Plan next steps"),
        ("Add skills and experience.", "Compare discovered jobs.", "Check fit and requirements.", "Use cited market context."),
    ):
        with column:
            with st.container(border=False, key=f"cia_step_{int(number)}"):
                st.caption(number)
                st.markdown(f"**{title}**")
                st.caption(description)

    settings = _render_settings()
    if settings["mode"] == "demo":
        with st.container(key="cia_main_mode_notice"):
            st.info("Demo Mode uses clearly labeled synthetic job examples and makes no external requests.")
    else:
        with st.container(key="cia_main_mode_notice"):
            st.info("Live Mode searches real job providers when you run. Optional research and AI features stay off until enabled and may incur charges.")

    st.header("Your candidate profile")
    st.caption("Start with an example or enter your own information. Review and edit everything before running the analysis.")
    candidate, validation_error, fields = _render_profile()
    count, total, _ = profile_completeness(fields)
    st.markdown("**Profile completeness**")
    st.progress(count / total)
    st.caption(f"{count} of {total} profile sections populated. This is not a match score or evidence coverage.")
    st.caption("A fuller profile can improve specificity. Qualifications not listed in the profile are treated as unknown, not automatically unmet.")
    with st.expander("About location and work authorization"):
        st.write("For this MVP, the candidate is assumed to live in Germany and be authorized to work there. Job eligibility from Germany is checked separately. A job location does not prove work authorization.")

    if validation_error:
        with st.container(key="cia_validation_notice"):
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
        with st.container(border=False, key="cia_review_workspace"):
            _render_candidate_review(candidate)
            if settings["mode"] == "live":
                confirmed = st.checkbox("I reviewed this profile and authorize the selected live requests.", key=f"confirm_{signature}")
            else:
                confirmed = st.checkbox("I reviewed this profile and want to run the local demo.", key=f"confirm_{signature}")
    run_disabled = not reviewed or not confirmed or candidate is None
    if st.button("Run Career Intelligence Agent", type="primary", disabled=run_disabled):
        try:
            if settings["mode"] == "live":
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env", override=False)
            with st.spinner("Analysing the profile and available job evidence..."):
                result = invoke_workflow(candidate, settings)
            st.session_state["run_result"] = result
            st.session_state["run_signature"] = signature
        except Exception as exc:
            st.error(f"Workflow execution failed: {type(exc).__name__}: {exc}")

    if st.session_state.get("run_result") and st.session_state.get("run_signature") == signature:
        _render_results(st.session_state["run_result"])


if __name__ == "__main__":
    main()
