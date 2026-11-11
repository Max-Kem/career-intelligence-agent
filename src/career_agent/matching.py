from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidate import CandidateProfile
from .filtering import country_eligibility
from .geography import country_from_location, normalize_country_code
from .models import Eligibility, EmploymentType, Job, WorkModel


class ScoringWeights(BaseModel):
    skills: float = Field(default=35, ge=0)
    experience: float = Field(default=25, ge=0)
    role_interests: float = Field(default=20, ge=0)
    seniority_qualifications: float = Field(default=15, ge=0)
    additional_preferences: float = Field(default=5, ge=0)

    @model_validator(mode="after")
    def weights_have_positive_total(self):
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one scoring weight must be positive")
        return self


DEFAULT_WEIGHTS = ScoringWeights()
MINIMUM_MATCH_SCORE = 50
MIN_EXPERIENCE_SCORE_FOR_SENIORITY_MISMATCH = 70


class SupportStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PROVISIONAL = "provisional"
    INSUFFICIENT = "insufficiently_supported"


@dataclass
class DimensionScore:
    name: str
    weight: float
    score: int | None
    evidence: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None


@dataclass
class JobMatch:
    job: Job
    score: int | None
    dimensions: list[DimensionScore]
    evidence_coverage: int
    evaluated_dimensions: int
    support_status: SupportStatus
    hard_filter_status: str
    geographic_status: Eligibility | None = None
    geographic_evidence: list[str] = field(default_factory=list)
    unresolved_requirements: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_or_unverified_skills: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendation_warnings: list[str] = field(default_factory=list)
    recommendation_blockers: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class HardFilterExclusion:
    job: Job
    violations: list[str]
    score: int | None = None
    explanation: str = ""
    geographic_status: Eligibility | None = None
    geographic_evidence: list[str] = field(default_factory=list)


@dataclass
class MatchingResult:
    ranked_matches: list[JobMatch]
    top_matches: list[JobMatch]
    excluded_jobs: list[HardFilterExclusion]


# Explicit aliases only: canonical term -> variants seen in profiles or job text.
SKILL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "project management": ("project management", "project manager", "project coordination", "project delivery", "deliver projects", "projektmanagement", "projektkoordination", "projektsteuerung"),
    "stakeholder management": ("stakeholder management", "stakeholder engagement", "stakeholder communication", "kundenkommunikation", "kommunikation mit kunden", "client communication", "customer communication"),
    "process optimization": ("process optimization", "process optimisation", "process improvement", "process redesign", "prozessoptimierung", "prozessverbesserung", "ablaufoptimierung"),
    "ai implementation": ("ai project delivery", "artificial intelligence project delivery", "ai implementation", "ai transformation project", "ki praktisch einsetzen", "ki praktisch einsetzt", "setzt ki praktisch ein", "ki sinnvoll einsetzen", "ki implementieren", "implementierung von ki", "einfuhrung von ki", "ai adoption", "artificial intelligence adoption", "kunstliche intelligenz praktisch einsetzen", "kunstliche intelligenz implementieren"),
    "sql": ("sql", "structured query language"),
    "python": ("python",),
    "excel": ("excel", "microsoft excel"),
    "data visualization": ("data visualization", "data visualisation", "data viz"),
    "business intelligence": ("business intelligence", "bi reporting"),
    "tableau": ("tableau",),
    "power bi": ("power bi",),
}

_TASK_SYNONYMS: dict[str, tuple[str, ...]] = {
    "project delivery": ("project delivery", "deliver projects", "manage projects", "project management", "projektmanagement", "projektkoordination"),
    "stakeholder management": ("stakeholder management", "stakeholder engagement", "stakeholder communication", "kundenkommunikation", "kommunikation mit kunden", "client communication", "customer communication"),
    "process improvement": ("process optimization", "process optimisation", "process improvement", "process redesign", "prozessoptimierung", "prozessverbesserung"),
    "data analysis": ("data analysis", "data analytics", "analyze data", "analyse data"),
    "reporting and dashboards": ("reporting", "dashboard", "dashboards", "data visualization", "data visualisation"),
    "change management": ("change management", "organizational change", "transformation adoption"),
}


def match_candidate_to_jobs(
    candidate: CandidateProfile,
    jobs: list[Job],
    *,
    weights: ScoringWeights | None = None,
    top_n: int = 5,
) -> MatchingResult:
    """Rank jobs, excluding hard violations and only top-listing supported fits >= 50/100."""
    if top_n < 0:
        raise ValueError("top_n cannot be negative")
    scoring_weights = weights or DEFAULT_WEIGHTS
    ranked: list[JobMatch] = []
    excluded: list[HardFilterExclusion] = []
    for job in jobs:
        violations, unresolved, geographic_status, geographic_evidence = _check_hard_requirements(candidate, job)
        scored_match = score_job(
            candidate, job, weights=scoring_weights, unresolved_requirements=unresolved,
            geographic_status=geographic_status, geographic_evidence=geographic_evidence,
        )
        if violations:
            excluded.append(HardFilterExclusion(
                job=job, violations=violations, score=scored_match.score,
                explanation=scored_match.explanation, geographic_status=geographic_status,
                geographic_evidence=geographic_evidence,
            ))
            continue
        _apply_seniority_preference_gate(candidate, job, scored_match)
        ranked.append(scored_match)
    ranked.sort(key=lambda match: (match.score is not None, match.score or -1, match.evidence_coverage), reverse=True)
    supported = [
        match for match in ranked
        if match.support_status == SupportStatus.SUFFICIENT
        and match.hard_filter_status == "passed"
        and match.score is not None
        and match.score >= MINIMUM_MATCH_SCORE
        and not match.recommendation_blockers
    ]
    return MatchingResult(ranked_matches=ranked, top_matches=supported[:top_n], excluded_jobs=excluded)


def score_job(
    candidate: CandidateProfile,
    job: Job,
    *,
    weights: ScoringWeights | None = None,
    unresolved_requirements: list[str] | None = None,
    geographic_status: Eligibility | None = None,
    geographic_evidence: list[str] | None = None,
) -> JobMatch:
    w = weights or DEFAULT_WEIGHTS
    required_skills = extract_job_skills(job.description or "", candidate.skills)
    candidate_skills = {canonical_skill(skill) for skill in candidate.skills if canonical_skill(skill)}
    matched = sorted(required_skills & candidate_skills)
    missing_skills = sorted(required_skills - candidate_skills)
    plain_description = _plain_text(job.description or "")

    dimensions = [
        _skills_dimension(required_skills, matched, candidate_skills, w.skills, plain_description),
        _experience_dimension(candidate, plain_description, w.experience),
        _role_dimension(candidate, job, w.role_interests),
        _seniority_dimension(candidate, job, plain_description, w.seniority_qualifications),
        _preference_dimension(candidate, job, w.additional_preferences),
    ]
    total_weight = sum(w.model_dump().values())
    evaluable = [dimension for dimension in dimensions if dimension.score is not None]
    coverage = _rounded(100 * sum(d.weight for d in evaluable) / total_weight) if total_weight else 0
    dimension_count = len(evaluable)
    sufficient = dimension_count >= 2 and coverage >= 50
    score = _rounded(sum(d.score * d.weight for d in evaluable) / sum(d.weight for d in evaluable)) if sufficient else None
    unresolved = list(unresolved_requirements or [])
    if not sufficient:
        status = SupportStatus.INSUFFICIENT
    elif coverage < 70 or unresolved:
        status = SupportStatus.PROVISIONAL
    else:
        status = SupportStatus.SUFFICIENT
    hard_status = "unknown" if unresolved else "passed"

    strengths = [
        f"Matched listed skill: {skill}." for skill in matched
    ]
    for dimension in evaluable:
        if dimension.score is not None and dimension.score >= 70:
            strengths.extend(dimension.evidence[:1])
    gaps = [f"Job mentions {skill}, which is not listed in the candidate profile." for skill in missing_skills]
    for dimension in evaluable:
        if dimension.score is not None and dimension.score < 50:
            gaps.extend(dimension.evidence[:1])
    if unresolved:
        gaps.extend(f"Unverified required preference: {item}." for item in unresolved)
    if not required_skills:
        gaps.append("The job description contains no recognized skills for comparison.")
    explanation = _explanation(score, status, matched, unresolved)
    return JobMatch(
        job=job, score=score, dimensions=dimensions, evidence_coverage=coverage,
        evaluated_dimensions=dimension_count, support_status=status,
        hard_filter_status=hard_status, geographic_status=geographic_status,
        geographic_evidence=list(geographic_evidence or []), unresolved_requirements=unresolved,
        matched_skills=matched, missing_or_unverified_skills=missing_skills,
        strengths=_dedupe(strengths), gaps=_dedupe(gaps), explanation=explanation,
    )


def extract_job_skills(description: str, candidate_skills: list[str] | None = None) -> set[str]:
    text = _plain_text(description)
    found: set[str] = set()
    for canonical, aliases in SKILL_SYNONYMS.items():
        if _has_any_phrase(text, aliases):
            found.add(canonical)
    # Profile-listed skills extend extraction without fuzzy or substring matching.
    for skill in candidate_skills or []:
        canonical = canonical_skill(skill)
        if canonical and _has_any_phrase(text, (skill,)):
            found.add(canonical)
    return found


def canonical_skill(value: str) -> str:
    normalized = _normalize(value)
    if not normalized:
        return ""
    for canonical, aliases in SKILL_SYNONYMS.items():
        if normalized == _normalize(canonical) or normalized in {_normalize(alias) for alias in aliases}:
            return canonical
    return normalized


def _skills_dimension(
    required: set[str], matched: list[str], candidate_skills: set[str], weight: float, description: str = "",
) -> DimensionScore:
    if not candidate_skills:
        return DimensionScore("skills", weight, None, unavailable_reason="Candidate skills are not provided.")
    if not required:
        return DimensionScore("skills", weight, None, unavailable_reason="No recognized job skills were found.")
    score = _rounded(100 * len(matched) / len(required))
    evidence = []
    for skill in matched:
        aliases = SKILL_SYNONYMS.get(skill, (skill,))
        observed = next((alias for alias in aliases if _has_any_phrase(description, (alias,))), skill)
        if observed == skill:
            evidence.append(f"Candidate lists {skill}, and the job description mentions {skill}.")
        else:
            evidence.append(f"Candidate lists {skill}; job wording '{observed}' is an explicit supported equivalent.")
    evidence += [f"Job description mentions {skill}; candidate profile does not list it." for skill in sorted(required - set(matched))]
    return DimensionScore("skills", weight, score, evidence)


def _experience_dimension(candidate: CandidateProfile, description: str, weight: float) -> DimensionScore:
    required_years = _required_years(description)
    if required_years is not None and candidate.years_experience is not None:
        score = _rounded(min(1.0, candidate.years_experience / required_years) * 100) if required_years else 100
        evidence = [f"Candidate reports {candidate.years_experience:g} years; job description asks for {required_years:g} years."]
        return DimensionScore("experience", weight, score, evidence)
    job_tasks = _extract_concepts(description, _TASK_SYNONYMS)
    candidate_text = " ".join(
        " ".join(entry.responsibilities) for entry in candidate.professional_experience
    )
    candidate_tasks = _extract_concepts(candidate_text, _TASK_SYNONYMS)
    if not job_tasks or not candidate_tasks:
        return DimensionScore("experience", weight, None, unavailable_reason="Job and candidate responsibility evidence do not overlap in the supported task vocabulary.")
    overlap = job_tasks & candidate_tasks
    score = _rounded(100 * len(overlap) / len(job_tasks))
    evidence = [f"Both the candidate's experience and the job mention {task}." for task in sorted(overlap)]
    evidence += [f"Job responsibility {task} is not evidenced in the candidate's experience." for task in sorted(job_tasks - overlap)]
    return DimensionScore("experience", weight, score, evidence)


def _role_dimension(candidate: CandidateProfile, job: Job, weight: float) -> DimensionScore:
    desired = candidate.desired_job_titles + candidate.career_interests
    if not desired or not job.title.strip():
        return DimensionScore("role_interests", weight, None, unavailable_reason="Candidate role interests or job title are missing.")
    best = 0.0
    best_title = ""
    job_tokens = _role_tokens(job.title)
    for title in desired:
        tokens = _role_tokens(title)
        if not tokens:
            continue
        overlap = tokens & job_tokens
        union = tokens | job_tokens
        similarity = len(overlap) / len(union) if union else 0
        if len(tokens) > 1 and len(job_tokens) > 1 and len(overlap) < 2:
            similarity = 0
        if similarity > best:
            best, best_title = similarity, title
    score = _rounded(100 * best)
    evidence = [f"Job title '{job.title}' shares role terms with candidate interest '{best_title}'."] if best else [f"Job title '{job.title}' does not match the listed role interests closely."]
    return DimensionScore("role_interests", weight, score, evidence)


def _seniority_dimension(candidate: CandidateProfile, job: Job, description: str, weight: float) -> DimensionScore:
    candidate_levels = {_seniority_label(item) for item in (candidate.seniority_preferences + candidate.required_preferences.seniority)}
    candidate_levels.discard("")
    job_levels = _job_seniority(job, description)
    pieces: list[tuple[int, str]] = []
    if candidate_levels and job_levels:
        matched = candidate_levels & job_levels
        pieces.append((100 if matched else 0, f"Candidate seniority preferences: {', '.join(sorted(candidate_levels))}; job indicates: {', '.join(sorted(job_levels))}."))
    required_degrees = _degree_terms(description)
    candidate_degrees = _degree_terms(" ".join(
        f"{education.qualification} {education.field or ''}" for education in candidate.education
    ))
    if required_degrees and candidate_degrees:
        overlap = required_degrees & candidate_degrees
        pieces.append((100 if overlap else 0, f"Candidate qualifications {', '.join(sorted(candidate_degrees))}; job description mentions {', '.join(sorted(required_degrees))}."))
    if not pieces:
        return DimensionScore("seniority_qualifications", weight, None, unavailable_reason="Seniority or qualification evidence is missing on one side.")
    return DimensionScore("seniority_qualifications", weight, _rounded(sum(score for score, _ in pieces) / len(pieces)), [evidence for _, evidence in pieces])


def _preference_dimension(candidate: CandidateProfile, job: Job, weight: float) -> DimensionScore:
    preferences = candidate
    pieces: list[tuple[int, str]] = []
    preferred_countries = [country.upper() for country in preferences.country_preferences]
    if preferred_countries:
        statuses = [country_eligibility(job, country) for country in preferred_countries]
        if Eligibility.ELIGIBLE in statuses:
            pieces.append((100, f"Job is explicitly eligible for at least one preferred country ({', '.join(preferred_countries)})."))
        elif all(status == Eligibility.INELIGIBLE for status in statuses):
            pieces.append((0, f"Job restrictions do not include preferred countries ({', '.join(preferred_countries)})."))
    if preferences.work_model_preferences and job.work_model != WorkModel.UNKNOWN:
        wanted = set(preferences.work_model_preferences)
        pieces.append((100 if job.work_model in wanted else 0, f"Job work model is {job.work_model.value}; preferred: {', '.join(item.value for item in preferences.work_model_preferences)}."))
    if preferences.employment_type_preferences and job.employment_type != EmploymentType.UNKNOWN:
        wanted_types = set(preferences.employment_type_preferences)
        pieces.append((100 if job.employment_type in wanted_types else 0, f"Job employment type is {job.employment_type.value}; preferred: {', '.join(item.value for item in preferences.employment_type_preferences)}."))
    if preferences.location_preferences and job.location not in {"Unknown", "Remote"}:
        haystack = _normalize(f"{job.location} {job.country or ''}")
        job_country = _country_code(job.country) or country_from_location(job.location)
        hits = [location for location in preferences.location_preferences
                if _contains_phrase(haystack, _normalize(location))
                or (_country_code(location) is not None and _country_code(location) == job_country)]
        pieces.append((100 if hits else 0, f"Job location {'matches' if hits else 'does not match'} preferred location(s) {', '.join(preferences.location_preferences)}."))
    if preferences.preferred_industries:
        tags = job.raw_metadata.get("tags", []) + job.raw_metadata.get("categories", [])
        text = _normalize(" ".join([job.title, job.description or "", *(str(tag) for tag in tags)]))
        detectable = [industry for industry in preferences.preferred_industries if _contains_phrase(text, _normalize(industry))]
        if detectable:
            pieces.append((100, f"Job text explicitly mentions preferred industry: {', '.join(detectable)}."))
    if not pieces:
        return DimensionScore("additional_preferences", weight, None, unavailable_reason="No optional preference could be checked from available job data.")
    return DimensionScore("additional_preferences", weight, _rounded(sum(score for score, _ in pieces) / len(pieces)), [evidence for _, evidence in pieces])


def _check_hard_requirements(
    candidate: CandidateProfile, job: Job,
) -> tuple[list[str], list[str], Eligibility | None, list[str]]:
    req = candidate.required_preferences
    violations: list[str] = []
    unresolved: list[str] = []
    geographic_status: Eligibility | None = None
    geographic_evidence: list[str] = []
    if req.countries:
        decisions = [_hard_country_assessment(job, country) for country in req.countries]
        states = [state for state, _ in decisions]
        geographic_evidence = [item for _, evidence in decisions for item in evidence]
        geographic_status = _combined_eligibility(states)
        if Eligibility.ELIGIBLE not in states and all(state == Eligibility.INELIGIBLE for state in states):
            violations.append(f"Job is not eligible in any required country ({', '.join(req.countries)}).")
        elif Eligibility.ELIGIBLE not in states:
            unresolved.append(f"Eligibility in required country/countries ({', '.join(req.countries)}) is unknown.")
    if req.work_models:
        if job.work_model == WorkModel.UNKNOWN:
            unresolved.append("Required work model cannot be verified.")
        elif job.work_model not in req.work_models:
            violations.append(f"Job work model {job.work_model.value} is outside required models.")
    if req.employment_types:
        if job.employment_type == EmploymentType.UNKNOWN:
            unresolved.append("Required employment type cannot be verified.")
        elif job.employment_type not in req.employment_types:
            violations.append(f"Job employment type {job.employment_type.value} is outside required types.")
    if req.seniority:
        levels = _job_seniority(job, _plain_text(job.description or ""))
        wanted = {_seniority_label(level) for level in req.seniority}
        if not levels:
            unresolved.append("Required seniority cannot be verified.")
        elif not levels & wanted:
            violations.append(f"Job seniority {', '.join(sorted(levels))} does not meet required level(s).")
    return violations, unresolved, geographic_status, geographic_evidence


def _apply_seniority_preference_gate(candidate: CandidateProfile, job: Job, match: JobMatch) -> None:
    preferred_levels = {_seniority_label(level) for level in candidate.seniority_preferences}
    actual_levels = _job_seniority(job, _plain_text(job.description or ""))
    if not preferred_levels or not actual_levels or preferred_levels & actual_levels:
        return
    warning = (
        f"Seniority mismatch: candidate prefers {', '.join(sorted(preferred_levels))}; "
        f"job indicates {', '.join(sorted(actual_levels))}."
    )
    match.recommendation_warnings.append(warning)
    experience_dimension = next((item for item in match.dimensions if item.name == "experience"), None)
    if experience_dimension is None or experience_dimension.score is None or experience_dimension.score < MIN_EXPERIENCE_SCORE_FOR_SENIORITY_MISMATCH:
        observed = "unavailable" if experience_dimension is None or experience_dimension.score is None else f"{experience_dimension.score}/100"
        match.recommendation_blockers.append(
            f"Senior-level exception requires relevant-experience evidence of at least "
            f"{MIN_EXPERIENCE_SCORE_FOR_SENIORITY_MISMATCH}/100; current evidence is {observed}."
        )
    match.explanation = " ".join([match.explanation, warning, *match.recommendation_blockers]).strip()


def _required_years(text: str) -> float | None:
    patterns = (
        r"(?:at least\s+|minimum(?: of)?\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
        r"(\d+)\s*[-–]\s*\d+\s+years?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_seniority(text: str) -> set[str]:
    """Recognize explicit title/structured seniority terms, not generic role words."""
    lowered = _normalize(text)
    levels: set[str] = set()
    patterns = {
        "junior": r"\b(junior|entry level|graduate|intern)\b",
        "mid-level": r"\b(mid level|mid career|intermediate)\b",
        "senior": r"\b(senior|sr|lead|principal)\b",
        "director": r"\b(director|vp|vice president|chief|c suite)\b",
    }
    for level, pattern in patterns.items():
        if re.search(pattern, lowered):
            levels.add(level)
    return levels


def _job_seniority(job: Job, description: str = "") -> set[str]:
    """Title is the strongest signal; description fallback requires role context."""
    title_levels = _extract_seniority(job.title)
    if title_levels:
        return title_levels
    structured_levels = _extract_seniority(job.seniority or "")
    if structured_levels:
        return structured_levels
    if job.seniority:
        # Himalayas supplies a constrained seniority enum such as "Manager".
        structured = {_seniority_label(value.strip()) for value in job.seniority.split(",") if value.strip()}
        known = {"junior", "mid-level", "senior", "manager", "director", "executive"}
        return structured & known
    return _extract_description_seniority(description)


def _extract_description_seniority(description: str) -> set[str]:
    text = _normalize(description)
    contextual = {
        "junior": r"\b(?:junior|entry level)\s+(?:level\s+)?(?:role|position|opportunity|candidate|professional|analyst|engineer|consultant)\b|\b(?:seeking|looking for|hiring)\s+(?:a|an)?\s*(?:junior|entry level)\b",
        "mid-level": r"\bmid level\s+(?:role|position|opportunity|candidate|professional)\b|\b(?:seeking|looking for|hiring)\s+(?:a|an)?\s*mid level\b",
        "senior": r"\bsenior level\s+(?:role|position|opportunity|candidate|professional)\b|\b(?:seeking|looking for|hiring)\s+(?:a|an)?\s*senior\b",
        "director": r"\b(?:director level|vice president role|chief level)\b",
    }
    return {level for level, pattern in contextual.items() if re.search(pattern, text)}


def _seniority_label(value: str) -> str:
    text = _normalize(value)
    aliases = {"entry level": "junior", "entry-level": "junior", "mid level": "mid-level", "mid-level": "mid-level", "lead": "manager"}
    return aliases.get(text, text)


def _degree_terms(text: str) -> set[str]:
    lowered = _normalize(text)
    result = set()
    for degree, patterns in {
        "bachelor": ("bachelor", "bachelors", "bachelor's", "bsc", "ba degree"),
        "master": ("master", "masters", "master's", "msc", "mba"),
        "doctorate": ("phd", "doctorate", "doctoral"),
    }.items():
        if any(_contains_phrase(lowered, _normalize(pattern)) for pattern in patterns):
            result.add(degree)
    return result


def _extract_concepts(text: str, dictionary: dict[str, tuple[str, ...]]) -> set[str]:
    lowered = _plain_text(text)
    return {concept for concept, aliases in dictionary.items() if _has_any_phrase(lowered, aliases)}


def _role_tokens(value: str) -> set[str]:
    text = _normalize(value)
    text = re.sub(r"\bbi\b", "business intelligence", text)
    text = re.sub(r"\bki\b|\bkunstliche intelligenz\b", "ai", text)
    text = re.sub(r"\bberatung\b|\bberater(in)?\b|\bconsulting\b", "consultant", text)
    text = re.sub(r"\bm w d\b", " ", text)
    # Seniority is scored separately, so it should not obscure otherwise identical role names.
    text = re.sub(r"\b(junior|entry level|mid level|senior|sr|lead|principal)\b", " ", text)
    generic_roles = {"analyst", "consultant", "manager", "executive"}
    return {
        token for token in re.findall(r"[a-z0-9]+", text)
        if token not in {"the", "and", "of", "for", "in"} | generic_roles
    }


_COUNTRY_ALIASES = {
    "de": {"de", "germany", "deutschland"},
    "ch": {"ch", "switzerland", "schweiz"},
    "gb": {"gb", "uk", "united kingdom", "great britain", "england"},
}
_CITY_COUNTRIES = {
    "zurich": "ch", "zuerich": "ch", "zürich": "ch", "london": "gb",
    "berlin": "de", "munich": "de", "munchen": "de", "muenchen": "de", "münchen": "de",
    "hamburg": "de", "frankfurt": "de", "stuttgart": "de", "cologne": "de", "koln": "de", "köln": "de",
}


def _country_code(value: str | None) -> str | None:
    return normalize_country_code(value)


def _country_from_job_location(job: Job) -> str | None:
    return _country_from_location_text(job.location)


def _country_from_location_text(value: str | None) -> str | None:
    return country_from_location(value)


def _hard_country_assessment(job: Job, required_country: str) -> tuple[Eligibility, list[str]]:
    target = _country_code(required_country) or required_country.strip().upper()
    structured_country = _country_code(job.country)
    location_country = _country_from_job_location(job)
    actual_location_country = structured_country or location_country
    accepted_candidate_country = country_eligibility(job, target)
    conflicts = job.raw_metadata.get("eligibility_conflicts", [])
    evidence: list[str] = [f"Required geographic region: {target}."]
    if job.country:
        evidence.append(f"Normalized job country field reports {job.country}.")
    elif actual_location_country:
        if structured_country:
            evidence.append(f"Structured job country reports {structured_country}.")
        else:
            evidence.append(f"Recognized job location '{job.location}' as {actual_location_country} using the explicit location/city map.")
    elif job.location and job.location.casefold() == "remote":
        evidence.append("Job location is marked remote; no physical job country is provided.")
    else:
        evidence.append(f"Job location '{job.location}' does not establish a country.")

    if isinstance(conflicts, list) and target in {str(code).upper() for code in conflicts}:
        evidence.append(f"Job text conflicts with provider country restrictions for {target}; eligibility remains unverified.")
        return Eligibility.UNKNOWN, evidence

    if structured_country and location_country and structured_country != location_country:
        evidence.append(
            f"Structured country {structured_country} conflicts with location '{job.location}' ({location_country}); eligibility remains unverified."
        )
        return Eligibility.UNKNOWN, evidence

    if actual_location_country == target:
        if accepted_candidate_country == Eligibility.INELIGIBLE:
            evidence.append(f"Provider applicant-country restrictions exclude {target}.")
            return Eligibility.INELIGIBLE, evidence
        if job.work_model == WorkModel.REMOTE:
            evidence.append("Remote role requires confirmed applicant-country eligibility.")
            if accepted_candidate_country == Eligibility.ELIGIBLE:
                evidence.append(f"Provider confirms remote eligibility in {target}.")
            else:
                evidence.append(f"Provider does not confirm remote eligibility in {target}.")
            return accepted_candidate_country, evidence
        if job.work_model in {WorkModel.ONSITE, WorkModel.HYBRID, WorkModel.UNKNOWN}:
            evidence.append(
                f"Verified physical job location is in required region {target}; "
                f"work model remains {job.work_model.value}."
            )
            return Eligibility.ELIGIBLE, evidence
    if actual_location_country and actual_location_country != target:
        if job.work_model == WorkModel.REMOTE:
            evidence.append(f"Physical job location is outside {target}; remote eligibility determines whether it can qualify.")
            evidence.append(
                f"Provider {'confirms' if accepted_candidate_country == Eligibility.ELIGIBLE else 'does not confirm'} applicant eligibility in {target}."
            )
            return accepted_candidate_country, evidence
        if job.work_model in {WorkModel.ONSITE, WorkModel.HYBRID}:
            evidence.append(f"Physical {job.work_model.value} job location is outside required region {target}.")
            return Eligibility.INELIGIBLE, evidence
        evidence.append("Physical job location is outside the required region and remote eligibility is not verified.")
        return Eligibility.INELIGIBLE, evidence

    # Without a physical job country, only a remote role's verified eligibility
    # can satisfy a hard country requirement. Missing metadata stays unresolved.
    if job.work_model == WorkModel.REMOTE:
        if accepted_candidate_country == Eligibility.ELIGIBLE:
            evidence.append(f"Provider confirms remote eligibility in {target}.")
        elif accepted_candidate_country == Eligibility.INELIGIBLE:
            evidence.append(f"Provider restrictions exclude remote applicants in {target}.")
        else:
            evidence.append(f"Provider eligibility for remote applicants in {target} is unknown.")
        return accepted_candidate_country, evidence
    evidence.append("Neither physical job country nor verified remote eligibility satisfies the hard geographic requirement.")
    return Eligibility.UNKNOWN, evidence


def _combined_eligibility(states: list[Eligibility]) -> Eligibility:
    if Eligibility.ELIGIBLE in states:
        return Eligibility.ELIGIBLE
    if states and all(state == Eligibility.INELIGIBLE for state in states):
        return Eligibility.INELIGIBLE
    return Eligibility.UNKNOWN


def _has_any_phrase(text: str, phrases) -> bool:
    return any(_contains_phrase(text, _normalize(phrase)) for phrase in phrases if phrase)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _plain_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value))


def _rounded(value: float) -> int:
    return int(value + 0.5)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _explanation(score: int | None, status: SupportStatus, matched: list[str], unresolved: list[str]) -> str:
    if score is None:
        return "Insufficient profile-job evidence to calculate a supported overall fit score."
    summary = f"Estimated profile-job fit: {score}/100 ({status.value}); this is not a hiring probability."
    if matched:
        summary += f" Matched listed skills: {', '.join(matched[:5])}."
    if unresolved:
        summary += " One or more hard requirements remain unverified."
    return summary
