"""Presentation-facing helpers for the local Streamlit MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from uuid import uuid4
from urllib.parse import urlparse

from .candidate import CandidateProfile
from .workflow import build_workflow

ROOT = Path(__file__).resolve().parents[2]

SESSION_DEFAULTS = {
    "display_name": "Candidate", "summary": "", "desired_titles": "", "skills": "",
    "career_interests": "", "industries": "", "seniority": [], "work_models": [],
    "employment_types": [], "location": "No preference", "experience_count": 1,
    "education_count": 0, "years_enabled": False, "years_value": 0.0,
    "mode": "Demo mode", "queries": 3, "pages": 1, "top_n": 5,
    "query_override": "", "allow_esco": False, "allow_company": False, "allow_rag": False,
    "example_choice": "None",
}


def parse_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def reset_session_state(state) -> None:
    """Reset keyed candidate/settings widgets and clear confirmation/results."""
    for key in list(state.keys()):
        if key.startswith(("experience_title_", "experience_responsibilities_", "education_", "confirm_")):
            state.pop(key, None)
    for key, value in SESSION_DEFAULTS.items():
        state[key] = list(value) if isinstance(value, list) else value
    state["candidate_id"] = f"local-{uuid4().hex[:12]}"
    for key in ("reviewed_signature", "run_result", "run_signature"):
        state.pop(key, None)


def invalidate_review_and_run(state) -> None:
    """Callback target used by candidate and execution widgets on every edit."""
    for key in list(state.keys()):
        if key.startswith("confirm_"):
            state.pop(key, None)
    for key in ("reviewed_signature", "run_result", "run_signature"):
        state.pop(key, None)


def input_fingerprint(candidate: CandidateProfile, settings: dict) -> str:
    import hashlib

    value = {"candidate": candidate.model_dump(mode="json"), "settings": settings}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def invalidate_stale_run(state, current_fingerprint: str | None) -> bool:
    """Drop stale reports/downloads after a meaningful candidate or setting edit."""
    if state.get("run_result") is not None and state.get("run_signature") != current_fingerprint:
        state.pop("run_result", None)
        state.pop("run_signature", None)
        return True
    return False


def confirmation_is_current(reviewed_fingerprint: str | None, current_fingerprint: str) -> bool:
    return bool(reviewed_fingerprint and reviewed_fingerprint == current_fingerprint)


def load_example_profile(profile_name: str, fixtures_dir: Path | None = None) -> dict:
    if profile_name not in {"a", "b"}:
        raise ValueError("Example profile must be 'a' or 'b'.")
    path = (fixtures_dir or ROOT / "fixtures") / f"candidate_profile_{profile_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_candidate_profile(fields: dict) -> CandidateProfile:
    """Validate UI values with the application's existing Pydantic schema."""
    desired_titles = [item.strip() for item in fields.get("desired_job_titles", []) if isinstance(item, str) and item.strip()]
    skills = [item.strip() for item in fields.get("skills", []) if isinstance(item, str) and item.strip()]
    experiences = []
    for item in fields.get("professional_experience", []):
        title = (item.get("title") or "").strip()
        responsibilities = [text.strip() for text in item.get("responsibilities", []) if isinstance(text, str) and text.strip()]
        if not title and not responsibilities:
            continue
        if not title or not responsibilities:
            raise ValueError("Complete or remove each additional experience/project entry.")
        experiences.append({"title": title, "responsibilities": responsibilities})
    if not desired_titles:
        raise ValueError("Add at least one desired job title.")
    if not skills:
        raise ValueError("Add at least one relevant skill.")
    if not experiences:
        raise ValueError("Add at least one experience or project entry.")
    first = experiences[0]
    if not (first.get("title") or "").strip() or not any(char.isalpha() for char in (first.get("title") or "")):
        raise ValueError("The first experience or project needs a title.")
    if not any(char.isalpha() for char in " ".join(first.get("responsibilities", []))):
        raise ValueError("The first experience or project needs a meaningful responsibility or description.")
    for item in experiences[1:]:
        if bool((item.get("title") or "").strip()) != bool(item.get("responsibilities")):
            raise ValueError("Complete or remove each additional experience/project entry.")
    for item in fields.get("education", []):
        if (item.get("field") or item.get("institution")) and not (item.get("qualification") or "").strip():
            raise ValueError("Add a qualification or clear the optional education fields.")
    return CandidateProfile.model_validate({
        "candidate_id": fields.get("candidate_id") or f"local-{uuid4().hex[:12]}",
        "name": fields.get("name") or "Candidate",
        "summary": fields.get("summary") or None,
        "skills": skills,
        "professional_experience": experiences,
        "years_experience": fields.get("years_experience"),
        "education": fields.get("education", []),
        "desired_job_titles": desired_titles,
        "career_interests": fields.get("career_interests", []),
        "preferred_industries": fields.get("preferred_industries", []),
        "seniority_preferences": fields.get("seniority_preferences", []),
        "country_preferences": ["DE"],
        "location_preferences": fields.get("location_preferences", []),
        "work_model_preferences": fields.get("work_model_preferences", []),
        "employment_type_preferences": fields.get("employment_type_preferences", []),
        "required_preferences": {"countries": ["DE"]},
    })


def profile_completeness(candidate: CandidateProfile | dict) -> tuple[int, int, list[str]]:
    if isinstance(candidate, CandidateProfile):
        roles = candidate.desired_job_titles
        skills = candidate.skills
        experiences = candidate.professional_experience
        summary = candidate.summary
        years = candidate.years_experience
        education = candidate.education
        interests = candidate.career_interests
        preferences = candidate.preferred_industries or candidate.seniority_preferences or candidate.location_preferences
        experience_present = bool(experiences)
        education_present = bool(education)
    else:
        roles = candidate.get("desired_job_titles", [])
        skills = candidate.get("skills", [])
        experiences = candidate.get("professional_experience", [])
        summary = candidate.get("summary")
        years = candidate.get("years_experience")
        education = candidate.get("education", [])
        interests = candidate.get("career_interests", [])
        preferences = candidate.get("preferred_industries") or candidate.get("seniority_preferences") or candidate.get("location_preferences")
        experience_present = any(
            (item.get("title") or "").strip() or any(item.get("responsibilities", []))
            for item in experiences
        )
        education_present = any(
            (item.get("qualification") or "").strip() or item.get("field") or item.get("institution")
            for item in education
        )
    sections = {
        "Desired roles": bool(roles),
        "Skills": bool(skills),
        "Experience/projects": experience_present,
        "Professional summary": bool(summary),
        "Years of experience": years is not None,
        "Education": education_present,
        "Career interests": bool(interests),
        "Preferences": bool(preferences),
    }
    populated = [name for name, is_populated in sections.items() if is_populated]
    return len(populated), len(sections), populated


def build_workflow_input(candidate: CandidateProfile, settings: dict) -> dict:
    demo = settings.get("mode", "demo") == "demo"
    switches = {} if demo else settings
    queries = int(settings.get("queries", 3))
    pages = int(settings.get("pages", 1))
    top_n = int(settings.get("top_n", 5))
    if not 1 <= queries <= 3 or not 1 <= pages <= 5 or not 1 <= top_n <= 5:
        raise ValueError("Workflow limits are queries 1-3, pages 1-5, and recommendations 1-5.")
    allow_esco = bool(switches.get("allow_live_esco", False))
    allow_company = bool(switches.get("allow_live_company_research", False))
    allow_rag = bool(switches.get("allow_live_rag", False))
    return {
        "profile_data": candidate.model_dump(mode="json"),
        "profile_name": "custom",
        "offline": demo,
        "query_override": (settings.get("query_override") or "").strip() or None,
        "query_limit": queries,
        "max_pages_per_provider": pages,
        "top_n": top_n,
        "output_dir": settings.get("output_dir", "reports/streamlit"),
        "allow_live_esco": allow_esco,
        "esco_enabled": allow_esco,
        "company_research_enabled": allow_company,
        "allow_live_company_research": allow_company,
        "market_intelligence_enabled": allow_rag,
        "allow_live_market_intelligence": allow_rag,
    }


def invoke_workflow(candidate: CandidateProfile, settings: dict, workflow_factory: Callable = build_workflow) -> dict:
    return workflow_factory().invoke(build_workflow_input(candidate, settings))


def dashboard_view(result: dict) -> dict:
    report = result.get("report_data") or {}
    scope = report.get("search_scope", {})
    workflow = report.get("workflow", {})
    raw_status = result.get("workflow_status", workflow.get("status", "unknown"))
    warnings = list(dict.fromkeys([*workflow.get("warnings", []), *result.get("warnings", [])]))
    errors = list(dict.fromkeys([*workflow.get("errors", []), *result.get("errors", [])]))
    nonfatal = list(dict.fromkeys([*workflow.get("nonfatal_errors", []), *result.get("nonfatal_errors", [])]))
    synthetic_warning_only = (
        scope.get("offline") is True
        and raw_status == "completed_with_warnings"
        and bool(warnings)
        and all("synthetic fixture jobs were used" in item.casefold() for item in warnings)
        and not errors and not nonfatal
    )
    status_label = (
        "Completed successfully using synthetic demo data."
        if synthetic_warning_only else raw_status.replace("_", " ").capitalize()
    )
    esco = report.get("esco_enrichment", {})
    company = report.get("company_research", {})
    market = report.get("market_intelligence", {})
    return {
        "status": raw_status,
        "status_label": status_label,
        "synthetic_warning_only": synthetic_warning_only,
        "discovered_jobs": report.get("discovered_jobs", len(result.get("jobs", []))),
        "evaluated_jobs": report.get("evaluated_jobs", 0),
        "recommendations": report.get("recommendations", []),
        "potential_matches": report.get("potential_matches_requiring_review", []),
        "career_intelligence": report.get("personalized_career_intelligence", {}),
        "market_intelligence": market,
        "esco": esco,
        "company_research": company,
        "warnings": [] if synthetic_warning_only else warnings,
        "errors": errors,
        "nonfatal_errors": nonfatal,
        "enrichment_messages": {
            "market_intelligence": enrichment_message("market_intelligence", market.get("status", "unknown"), scope.get("offline", False)),
            "esco": enrichment_message("esco", esco.get("status", "unknown"), scope.get("offline", False)),
            "company_research": enrichment_message("company_research", company.get("status", "unknown"), scope.get("offline", False)),
        },
        "diagnostics": {
            "raw_status": raw_status, "warnings": warnings, "errors": errors, "nonfatal_errors": nonfatal,
            "enrichment_statuses": {
                "market_intelligence": market.get("status", "unknown"),
                "esco": esco.get("status", "unknown"),
                "company_research": company.get("status", "unknown"),
            },
        },
        "score_interpretation": report.get("score_interpretation", "Indicative alignment based on available evidence, not hiring probability."),
    }


def enrichment_message(service: str, status: str, offline: bool) -> str:
    names = {
        "market_intelligence": "Market intelligence",
        "esco": "ESCO enrichment",
        "company_research": "Company research",
    }
    name = names.get(service, service.replace("_", " ").capitalize())
    if status in {"succeeded", "partially_succeeded", "available"}:
        return f"{name} completed with status: {status.replace('_', ' ')}."
    if offline or status == "skipped_offline":
        if service == "market_intelligence":
            return "Market intelligence is disabled in Demo Mode. Run in Live Mode with RAG enabled to retrieve source-grounded career insights."
        if service == "esco":
            return "ESCO enrichment is disabled in Demo Mode. Run in Live Mode with ESCO enabled to look up occupational concepts."
        if service == "company_research":
            return "Company research is disabled in Demo Mode. Run in Live Mode with Tavily enabled to research recommended employers."
    if status == "skipped_no_recommendations":
        return f"{name} did not run because no recommendations qualified."
    if status in {"skipped_disabled", "skipped"}:
        return f"{name} was disabled or skipped for this run."
    if status in {"unavailable", "failed"}:
        return f"{name} was unavailable; other workflow results were retained."
    return f"{name} status: {status.replace('_', ' ')}."


def candidate_review_summary(candidate: CandidateProfile) -> dict:
    return {
        "display_name": candidate.name,
        "summary": candidate.summary or "Not provided",
        "desired_job_titles": candidate.desired_job_titles or [],
        "skills": candidate.skills or [],
        "career_interests": candidate.career_interests or [],
        "preferred_industries": candidate.preferred_industries or [],
        "experience": [
            {"title": item.title, "responsibilities": item.responsibilities or ["Not provided"]}
            for item in candidate.professional_experience
        ] or [],
        "years_experience": f"{candidate.years_experience:g} years" if candidate.years_experience is not None else "Not provided",
        "education": [
            {"qualification": item.qualification, "field": item.field or "Not provided", "institution": item.institution or "Not provided"}
            for item in candidate.education
        ] or [],
        "seniority_preferences": candidate.seniority_preferences or [],
        "employment_type_preferences": [item.value for item in candidate.employment_type_preferences] or [],
        "work_model_preferences": [item.value for item in candidate.work_model_preferences] or [],
        "location_preferences": candidate.location_preferences or [],
        "geographic_assumption": "Candidate resides in Germany and is authorized to work there. Job eligibility from Germany is checked separately.",
    }


def readable_requirement(item: dict) -> str:
    requirement = item.get("requirement", "Requirement unspecified")
    requirement_type = item.get("requirement_type", "not_specified").replace("_", " ")
    evidence_status = item.get("candidate_evidence_status", "not_documented").replace("_", " ")
    parts = [f"{requirement} — {requirement_type}; candidate evidence: {evidence_status}."]
    evidence = item.get("candidate_evidence")
    posting = item.get("posting_evidence")
    if evidence:
        parts.append(evidence)
    if posting:
        parts.append(f"Posting: {posting}")
    return " ".join(parts)


def recommendation_content(item: dict) -> dict:
    requirements = [readable_requirement(requirement) for requirement in item.get("requirements_to_verify", [])]
    return {
        "strengths": item.get("strengths") or [],
        "strengths_empty": "No specific strengths were recorded by the matching engine.",
        "gaps": item.get("gaps") or [],
        "gaps_empty": "No specific gaps were recorded; this does not verify that every requirement is met.",
        "requirements": requirements,
        "requirements_empty": "No additional requirements were identified by the deterministic checks; this does not guarantee full qualification.",
    }


def recommendation_link(job: dict, *, offline: bool) -> dict:
    url = job.get("source_url")
    source = str(job.get("source", "")).casefold()
    if offline or source in {"synthetic", "fixture"}:
        return {"url": None, "message": "Synthetic example – no external job posting."}
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.endswith(".invalid") or parsed.hostname == "example.invalid":
        return {"url": None, "message": "Synthetic example – no external job posting."}
    return {"url": url, "message": None}


def report_download_payloads(markdown: str, report_data: dict) -> tuple[str, str]:
    return markdown, json.dumps(report_data, ensure_ascii=False, indent=2)
