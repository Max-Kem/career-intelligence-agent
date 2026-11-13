"""Presentation-facing helpers for the local Streamlit MVP."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4
from urllib.parse import urlparse

from .candidate import CandidateProfile
from .requirements_review import clean_job_description
from .workflow import build_workflow

ROOT = Path(__file__).resolve().parents[2]

SESSION_DEFAULTS = {
    "display_name": "Candidate", "summary": "", "desired_titles": "", "skills": "",
    "career_interests": "", "industries": "", "seniority": [], "work_models": [],
    "employment_types": [], "location": "No preference", "experience_count": 1,
    "education_count": 0, "years_enabled": False, "years_value": 0.0,
    "mode": "Demo mode", "queries": 3, "pages": 1, "top_n": 5,
    "query_override": "", "allow_esco": False, "allow_company": False, "allow_rag": False,
    "allow_synthesis": False,
    "example_choice": "None",
}


def parse_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


_DATE_LABEL = r"(?:\d{4}(?:-\d{2}(?:-\d{2})?)?|[A-Za-z]{3,9}\s+\d{4})"
_DATED_EXPERIENCE_SUFFIX = re.compile(
    rf"\s*\((?P<start>{_DATE_LABEL})\s*(?:-|\u2013)\s*(?P<end>{_DATE_LABEL}|present|current)\)\s*$",
    re.IGNORECASE,
)


def split_experience_title(title: str) -> tuple[str, str, str, bool]:
    """Separate a recognized trailing date range from a legacy experience title."""
    cleaned = (title or "").strip()
    match = _DATED_EXPERIENCE_SUFFIX.search(cleaned)
    if not match:
        return cleaned, "", "", False
    end_value = match.group("end")
    ongoing = end_value.casefold() in {"present", "current"}
    return (
        cleaned[:match.start()].rstrip(),
        match.group("start"),
        "" if ongoing else end_value,
        ongoing,
    )


def _date_label(value) -> str:
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}", text):
        return text
    if re.fullmatch(r"\d{4}-\d{2}", text):
        try:
            datetime.strptime(text, "%Y-%m")
        except ValueError as exc:
            raise ValueError("Enter dates as YYYY, YYYY-MM, YYYY-MM-DD, or Month YYYY.") from exc
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("Enter dates as YYYY, YYYY-MM, YYYY-MM-DD, or Month YYYY.") from exc
        return text
    if re.fullmatch(r"[A-Za-z]{3,9}\s+\d{4}", text):
        for pattern in ("%b %Y", "%B %Y"):
            try:
                datetime.strptime(text, pattern)
                return text
            except ValueError:
                continue
    raise ValueError("Enter dates as YYYY, YYYY-MM, YYYY-MM-DD, or Month YYYY.")


def compose_experience_title(title: str, start_date=None, end_date=None, ongoing: bool = False) -> str:
    """Append optional dates to the existing title field without duplicating legacy ranges."""
    cleaned = (title or "").strip()
    if not cleaned:
        return ""
    # Preserve titles from older fixtures/profiles that already include their date range.
    if _DATED_EXPERIENCE_SUFFIX.search(cleaned):
        return cleaned
    start = _date_label(start_date)
    end = "Present" if ongoing else _date_label(end_date)
    if start and end:
        return f"{cleaned} ({start} - {end})"
    if start:
        return f"{cleaned} (from {start})"
    if end:
        suffix = end if ongoing else f"until {end}"
        return f"{cleaned} ({suffix})"
    return cleaned


def reset_session_state(state) -> None:
    """Reset keyed candidate/settings widgets and clear confirmation/results."""
    for key in list(state.keys()):
        if key.startswith((
            "experience_title_", "experience_responsibilities_", "experience_start_date_",
            "experience_end_date_", "experience_ongoing_", "education_", "confirm_",
        )):
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
        title = compose_experience_title(
            item.get("title") or "",
            item.get("start_date"),
            item.get("end_date"),
            bool(item.get("ongoing", False)),
        )
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
    allow_synthesis = bool(switches.get("allow_llm_synthesis", False))
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
        "llm_synthesis_enabled": allow_synthesis,
        "allow_live_llm_synthesis": allow_synthesis,
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
    status_labels = {
        "completed": "Completed successfully.",
        "completed_with_warnings": "Completed with warnings.",
        "completed_no_recommendations": "Completed; no roles met the recommendation criteria.",
        "failed": "Run could not be completed.",
    }
    status_label = (
        "Completed successfully using synthetic demo data."
        if synthetic_warning_only else status_labels.get(raw_status, raw_status.replace("_", " ").capitalize())
    )
    esco = report.get("esco_enrichment", {})
    company = report.get("company_research", {})
    market = report.get("market_intelligence", {})
    search = report.get("search_scope", {})
    provider_diagnostics = search.get("provider_diagnostics", {})
    provider_errors = provider_diagnostics.get("errors", {})
    recommendations = report.get("recommendations", [])
    potential_matches = report.get("potential_matches_requiring_review", [])
    evaluated_jobs = report.get("evaluated_jobs", 0)
    diagnostic_summary = {
        "job_discovery": (
            "Completed with provider issues; available results were retained."
            if provider_errors and provider_diagnostics.get("provider_counts")
            else "Unavailable; no provider results were returned."
            if provider_errors
            else "Completed successfully."
        ),
        "matching": f"Evaluated {evaluated_jobs} job{'s' if evaluated_jobs != 1 else ''}.",
        "recommendations": (
            f"{len(recommendations)} role{'s' if len(recommendations) != 1 else ''} qualified under the existing evidence and requirement checks."
            if recommendations
            else f"No roles qualified; {len(potential_matches)} potential match{'es' if len(potential_matches) != 1 else ''} remain for review."
        ),
        "market_intelligence": _plain_status(
            market.get("status", "unknown"), search.get("offline", False),
            detail=f"Retrieved {len(market.get('evidence', []))} source passages." if market.get("evidence") else None,
        ),
        "llm_synthesis": _plain_status(
            report.get("intelligence_synthesis", {}).get("status", "not_requested"),
            search.get("offline", False),
        ),
        "esco": _plain_status(esco.get("status", "unknown"), search.get("offline", False)),
        "company_research": _plain_status(company.get("status", "unknown"), search.get("offline", False)),
        "warnings": warnings,
        "errors": errors,
        "nonfatal_errors": nonfatal,
    }
    return {
        "status": raw_status,
        "status_label": status_label,
        "synthetic_warning_only": synthetic_warning_only,
        "discovered_jobs": report.get("discovered_jobs", len(result.get("jobs", []))),
        "evaluated_jobs": evaluated_jobs,
        "recommendations": recommendations,
        "potential_matches": potential_matches,
        "data_source_label": "Synthetic demo jobs" if scope.get("offline") is True else "Live job providers",
        "career_intelligence": report.get("personalized_career_intelligence", {}),
        "intelligence_synthesis": report.get("intelligence_synthesis", {}),
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
            "summary": diagnostic_summary,
            "enrichment_statuses": {
                "market_intelligence": market.get("status", "unknown"),
                "llm_synthesis": report.get("intelligence_synthesis", {}).get(
                    "status", "skipped_offline" if scope.get("offline") else "not_requested"
                ),
                "esco": esco.get("status", "unknown"),
                "company_research": company.get("status", "unknown"),
            },
        },
        "score_interpretation": report.get("score_interpretation", "Indicative alignment based on available evidence, not hiring probability."),
    }


def _plain_status(status: str, offline: bool, detail: str | None = None) -> str:
    """Translate integration statuses for the human-readable diagnostics panel."""
    if status in {"succeeded", "available"}:
        text = "Completed successfully."
    elif status == "partial":
        text = "Partially completed; timed-out lookups were retained as diagnostics and completed results remain available."
    elif status == "partially_succeeded":
        text = "Partly completed; some results may be unavailable."
    elif status == "skipped_offline":
        text = "Not run because Demo Mode is offline."
    elif status in {"skipped_disabled", "skipped", "not_requested"}:
        text = "Not run in Demo Mode (offline)." if offline else "Not enabled for this run."
    elif status == "skipped_no_recommendations":
        text = "Not run because no recommendations qualified."
    elif status in {"unavailable", "failed"}:
        text = "Could not complete; other workflow results were retained."
    else:
        text = "Status is not available."
    return f"{text} {detail}" if detail else text


def synthesis_presentation(synthesis: dict, demo_mode: bool) -> dict:
    """Describe synthesis provenance without exposing raw diagnostic labels."""
    if synthesis.get("generation_mode") == "llm_generated" and synthesis.get("status") == "succeeded":
        return {"kind": "success", "message": "Personalized analysis generated from retrieved evidence. Citations are labeled by evidence type."}
    if synthesis.get("generation_mode") == "deterministic_fallback":
        return {
            "kind": "info" if not demo_mode else "hidden",
            "message": "AI-written synthesis was unavailable or not requested. Deterministic guidance and retrieved sources are shown where available.",
        }
    return {
        "kind": "hidden" if demo_mode else "caption",
        "message": "AI-written synthesis was not requested for this run.",
    }


def enrichment_message(service: str, status: str, offline: bool) -> str:
    names = {
        "market_intelligence": "Market intelligence",
        "esco": "ESCO enrichment",
        "company_research": "Company research",
    }
    name = names.get(service, service.replace("_", " ").capitalize())
    if service == "esco" and status == "partial":
        return "ESCO enrichment: Partial. Some lookups timed out; completed results and timeout diagnostics were retained."
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
        "career_direction": item.get("career_direction") or {"status": "unassessed", "evidence": []},
        "job_highlights": job_posting_highlights(item.get("job") or {}),
    }


_JOB_CONDITION_PATTERN = re.compile(
    r"\b(?:based in|located in|office[- ]based|office in|on[- ]site|in[- ]person|hybrid|fully remote|remote role|remote work|work remotely|working remotely|remote from|remote within|work from home|work from anywhere|travel|customer visits?|visits? across|visits? to|onboarding|relocation|team days?|all[- ]hands|offsites?|must be based|must be located|full[- ]time|part[- ]time|contract role|\d+ hours? per week)\b",
    re.I,
)
_JOB_PREFERRED_PATTERN = re.compile(
    r"\b(?:preferred|ideally|desirable|nice to have|bonus|a plus|additional advantage|advantageous)\b", re.I,
)
_JOB_REQUIRED_PATTERN = re.compile(
    r"\b(?:required|must|minimum|essential|mandatory|at least|requirement)\b", re.I,
)
_JOB_ACTION_PATTERN = re.compile(
    r"\b(?:lead|manage|deliver|implement|coordinate|own|develop|build|run|conduct|support|design|drive|facilitate|analy[sz]e|oversee|plan|configure|test|roll out)\b",
    re.I,
)


def job_posting_highlights(job: dict, limit: int = 6) -> dict:
    """Extract short source-grounded posting facts without paraphrasing job text."""
    description = job.get("description")
    model = str(job.get("work_model") or "unknown").casefold()
    provider_model = model if model in {"remote", "hybrid", "onsite"} else None
    result = {"provider_work_model": provider_model, "items": [], "description_available": bool(description and str(description).strip())}
    if not result["description_available"]:
        return result

    text = clean_job_description(str(description))
    section = "other"
    career_path = False
    groups: dict[str, list[str]] = {"Responsibility": [], "Qualification": [], "Required": [], "Location & attendance": [], "Preferred": []}
    heading_pattern = re.compile(r"^\W*(.+?)\W*$")
    for raw_line in text.splitlines():
        line = re.sub(r"^[\s*\-\u2022]+", "", raw_line).strip()
        if not line:
            continue
        heading_match = heading_pattern.match(line)
        heading = heading_match.group(1).strip(" :\t") if heading_match else line
        heading_folded = re.sub(r"[^a-z ]", " ", heading.casefold()).strip()
        if "career path" in heading_folded or "career progression" in heading_folded:
            career_path = True
            section = "career_path"
            continue
        if career_path:
            level_line = heading_folded
            if re.match(r"(?:key responsibilities|responsibilities|what you will do|what we ll do|your mission|what we re really looking for|what you bring|qualifications|requirements|what skills)", level_line):
                career_path = False
            else:
                continue

        if re.match(r"(?:key responsibilities|responsibilities|what you will do|what you ll do|your mission|mission|your tasks|duties)", heading_folded):
            section = "responsibility"
            continue
        if re.match(r"(?:location|work location|work arrangement|working location|place of work|where you ll work|where youll work|offer essentials)", heading_folded):
            section = "conditions"
            continue
        if re.match(r"(?:bonus points|nice to have|preferred qualifications|additional advantages)", heading_folded):
            section = "preferred"
            continue
        if re.match(r"(?:benefits|what we offer|perks|why\b|about (?:us|the company)|company overview|our story|our culture|compensation|salary|remuneration)", heading_folded):
            section = "excluded"
            continue
        if re.match(r"(?:what you bring|what you need to succeed|qualifications|requirements|what we.re really looking for|what skills|your profile|experience)", heading_folded):
            section = "qualification"
            continue
        # Avoid treating headings and company overview/benefit copy as job facts.
        if len(line) < 4 or (len(line) < 90 and not re.search(r"[.!?]$", line) and line == heading):
            continue

        sentences = [part.strip(" \t-\u2022") for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line) if part.strip()]
        for sentence in sentences:
            if len(sentence) < 25:
                continue
            if re.search(r"\b(?:salary|compensation|remuneration|pay range|annual base|hourly rate|\d+\s*%\s*bonus)\b|[€$£]", sentence, re.I):
                continue
            if section == "responsibility" and len(sentence) < 45 and not _JOB_ACTION_PATTERN.search(sentence):
                continue
            if "following key attributes ensure success" in sentence.casefold():
                continue
            if _JOB_PREFERRED_PATTERN.search(sentence) or section == "preferred":
                category = "Preferred"
            elif _JOB_CONDITION_PATTERN.search(sentence):
                category = "Location & attendance"
            elif _JOB_REQUIRED_PATTERN.search(sentence):
                category = "Required"
            elif section == "qualification" and re.search(
                r"\b(?:experience|qualification|degree|fluent|skills?|knowledge|proficiency|years?|ability|proven|track record|technical|certification|certificate|must|required)\b",
                sentence, re.I,
            ):
                category = "Qualification"
            elif section == "responsibility":
                category = "Responsibility"
            else:
                continue
            clean_sentence = re.sub(r"\s+", " ", sentence).strip()
            if clean_sentence not in groups[category]:
                groups[category].append(clean_sentence)

    # Interleave fact types so a long responsibilities section cannot hide a
    # posting's mandatory qualifications or attendance conditions. Keep source
    # language verbatim and preserve Preferred as its own category.
    cap = max(0, limit)
    ordered: list[dict] = []
    offsets = {category: 0 for category in groups}

    def take(category: str, maximum: int) -> None:
        while offsets[category] < len(groups[category]) and maximum > 0 and len(ordered) < cap:
            ordered.append({"category": category, "text": groups[category][offsets[category]]})
            offsets[category] += 1
            maximum -= 1

    # Preserve a compact balance: a core duty, up to two qualification facts,
    # then explicit place/attendance facts (including onboarding and travel).
    take("Responsibility", 1)
    take("Required", 2)
    take("Qualification", 2 - min(2, offsets["Required"]))
    take("Location & attendance", 3)
    # Preferred qualifications are clearly marked and retained when there is
    # room; they never displace the principal duties or explicit conditions.
    take("Preferred", cap - len(ordered))
    for category in ("Responsibility", "Required", "Qualification", "Location & attendance"):
        take(category, cap - len(ordered))
    result["items"] = ordered
    return result


def primary_reasons(item: dict, limit: int = 3) -> list[str]:
    """Select existing matching evidence for a concise card summary, without paraphrasing."""
    strengths = item.get("strengths") or []
    reasons = strengths or item.get("reasons") or []
    return list(dict.fromkeys(reason for reason in reasons if isinstance(reason, str) and reason.strip()))[:limit]


def job_context_label(job: dict) -> str:
    """Format location and work model without repeating equivalent text."""
    location = str(job.get("location") or "Location not provided").strip()
    work_model = str(job.get("work_model") or "unknown").replace("_", " ").strip().title()
    if work_model.casefold() in location.casefold():
        return location
    return f"{location} | {work_model}"


def recommendation_link(job: dict, *, offline: bool) -> dict:
    url = job.get("source_url")
    source = str(job.get("source", "")).casefold()
    if offline or source in {"synthetic", "fixture"}:
        return {"url": None, "message": "Synthetic example – no external job posting."}
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.endswith(".invalid") or parsed.hostname == "example.invalid":
        return {"url": None, "message": "Synthetic example – no external job posting."}
    return {"url": url, "message": None}


def citation_presentation(citation: dict) -> dict:
    """Return plain-language attribution fields while keeping typed references distinct."""
    kind = citation.get("kind")
    reference_id = citation.get("reference_id")
    if kind == "market_source":
        return {
            "kind": kind, "label": "Market source", "reference_id": reference_id,
            "title": citation.get("document_title") or "Source title unavailable",
            "publisher": citation.get("publisher") or "Publisher unavailable",
            "year": citation.get("publication_year") or "Year unavailable",
            "page": citation.get("page_number"), "url": citation.get("source_url"),
        }
    if kind == "candidate_profile":
        return {
            "kind": kind, "label": "Candidate profile evidence", "reference_id": reference_id,
            "title": citation.get("label") or "Profile information", "url": None,
        }
    if kind == "job_posting":
        company = citation.get("company")
        title = citation.get("document_title") or "Job posting"
        return {
            "kind": kind, "label": "Job posting", "reference_id": reference_id,
            "title": f"{title} - {company}" if company else title,
            "url": citation.get("source_url"),
        }
    return {
        "kind": "unknown", "label": "Unclassified reference", "reference_id": reference_id,
        "title": citation.get("document_title") or "Reference unavailable", "url": None,
    }


def report_download_payloads(markdown: str, report_data: dict) -> tuple[str, str]:
    return markdown, json.dumps(report_data, ensure_ascii=False, indent=2)
