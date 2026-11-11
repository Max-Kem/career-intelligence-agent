from __future__ import annotations

from dataclasses import dataclass, field

from .models import Eligibility, EmploymentType, Job, WorkModel


@dataclass(frozen=True)
class FilterDecision:
    job: Job
    country_eligibility: Eligibility
    reasons: tuple[str, ...] = ()


@dataclass
class FilterDiagnostics:
    """Counts and a few identifiers explaining sequential filter exclusions."""

    excluded_counts: dict[str, int] = field(default_factory=lambda: {
        "keyword": 0, "country": 0, "work_model": 0, "employment_type": 0,
    })
    exclusion_reasons: dict[str, dict[str, int]] = field(default_factory=dict)
    excluded_job_ids: dict[str, list[str]] = field(default_factory=dict)
    missing_country_count: int = 0
    missing_location_count: int = 0
    missing_country_exclusions: int = 0
    missing_location_exclusions: int = 0


@dataclass
class FilterResult:
    jobs: list[Job]
    diagnostics: FilterDiagnostics


def country_eligibility(job: Job, country: str | None) -> Eligibility:
    if not country:
        return job.country_eligibility
    wanted = country.strip().upper()
    conflicts = job.raw_metadata.get("eligibility_conflicts", [])
    if isinstance(conflicts, list) and wanted in {str(code).upper() for code in conflicts}:
        return Eligibility.UNKNOWN
    if job.eligible_countries:
        return Eligibility.ELIGIBLE if wanted in {c.upper() for c in job.eligible_countries} else Eligibility.INELIGIBLE
    if job.country_eligibility != Eligibility.UNKNOWN:
        return job.country_eligibility
    # `country` describes the job's location; it does not prove where an
    # employer permits a remote candidate to reside.
    return Eligibility.UNKNOWN


def filter_jobs(
    jobs: list[Job], *, keywords: str = "", country: str | None = None,
    work_model: WorkModel | str | None = None,
    employment_type: EmploymentType | str | None = None,
    strict_unknown: bool = False,
) -> list[Job]:
    """Hard-filter verified mismatches; unknown metadata is retained by default."""
    return filter_jobs_with_diagnostics(
        jobs, keywords=keywords, country=country, work_model=work_model,
        employment_type=employment_type, strict_unknown=strict_unknown,
    ).jobs


def filter_jobs_with_diagnostics(
    jobs: list[Job], *, keywords: str = "", country: str | None = None,
    work_model: WorkModel | str | None = None,
    employment_type: EmploymentType | str | None = None,
    strict_unknown: bool = False,
) -> FilterResult:
    """Apply existing filter rules in order and report each job's first exclusion."""
    result: list[Job] = []
    diagnostics = FilterDiagnostics()
    terms = [term.casefold() for term in keywords.split()]
    requested_model = _enum_or_none(WorkModel, work_model)
    requested_employment = _enum_or_none(EmploymentType, employment_type)
    for job in jobs:
        if not job.country:
            diagnostics.missing_country_count += 1
        if not job.location or job.location.strip().casefold() in {"unknown", "not specified", "n/a"}:
            diagnostics.missing_location_count += 1

        haystack = " ".join((job.title, job.company, job.location, job.description or "", " ".join(str(t) for t in job.raw_metadata.get("tags", [])))).casefold()
        if terms and not all(term in haystack for term in terms):
            _record_exclusion(diagnostics, "keyword", "keyword_terms_not_found", job)
            continue
        eligibility = country_eligibility(job, country)
        if eligibility == Eligibility.INELIGIBLE:
            _record_exclusion(diagnostics, "country", "country_ineligible", job)
            continue
        if strict_unknown and country and eligibility == Eligibility.UNKNOWN:
            _record_exclusion(diagnostics, "country", "country_eligibility_unknown_strict", job)
            if not job.country:
                diagnostics.missing_country_exclusions += 1
            continue
        if requested_model and requested_model != WorkModel.UNKNOWN:
            if job.work_model not in {requested_model, WorkModel.UNKNOWN}:
                _record_exclusion(diagnostics, "work_model", "work_model_mismatch", job)
                continue
            if strict_unknown and job.work_model == WorkModel.UNKNOWN:
                _record_exclusion(diagnostics, "work_model", "work_model_unknown_strict", job)
                continue
        if requested_employment and requested_employment != EmploymentType.UNKNOWN:
            if job.employment_type not in {requested_employment, EmploymentType.UNKNOWN}:
                _record_exclusion(diagnostics, "employment_type", "employment_type_mismatch", job)
                continue
            if strict_unknown and job.employment_type == EmploymentType.UNKNOWN:
                _record_exclusion(diagnostics, "employment_type", "employment_type_unknown_strict", job)
                continue
        result.append(job)
    return FilterResult(jobs=result, diagnostics=diagnostics)


def _record_exclusion(diagnostics: FilterDiagnostics, filter_name: str, reason: str, job: Job) -> None:
    diagnostics.excluded_counts[filter_name] += 1
    reasons = diagnostics.exclusion_reasons.setdefault(filter_name, {})
    reasons[reason] = reasons.get(reason, 0) + 1
    ids = diagnostics.excluded_job_ids.setdefault(filter_name, [])
    if len(ids) < 5:
        ids.append(job.job_id)


def _enum_or_none(enum_type: type, value):
    if value is None:
        return None
    try:
        return enum_type(value)
    except ValueError:
        return enum_type(str(value).lower())
