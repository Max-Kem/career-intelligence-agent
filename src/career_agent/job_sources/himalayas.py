from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..models import Eligibility, Job, normalize_employment_type, normalize_work_model
from .common import BaseAdapter, ProviderError


class HimalayasAdapter(BaseAdapter):
    source = "himalayas"
    endpoint = "https://himalayas.app/jobs/api/search"

    def search(self, query: str = "", *, country: str | None = None, max_pages: int = 2) -> list[Job]:
        if max_pages < 1 or max_pages > 20:
            raise ValueError("max_pages must be between 1 and 20")
        jobs: list[Job] = []
        for page in range(1, max_pages + 1):
            params = {"q": query, "sort": "recent", "page": page}
            if country:
                params["country"] = country
            payload = self._get_json(self.endpoint, params=params)
            if not isinstance(payload, dict) or not isinstance(payload.get("jobs", []), list):
                raise ProviderError("Himalayas returned an unexpected response shape")
            batch = [normalize_himalayas(item) for item in payload.get("jobs", []) if isinstance(item, dict)]
            jobs.extend(batch)
            # The published API notes pages can be short before the end of a walk.
            if not batch:
                break
        return jobs


def normalize_himalayas(data: dict[str, Any]) -> Job:
    restrictions = data.get("locationRestrictions")
    eligible_countries: list[str] = []
    if isinstance(restrictions, list):
        for item in restrictions:
            if isinstance(item, dict):
                code = item.get("alpha2")
                if isinstance(code, str) and code.strip():
                    eligible_countries.append(code.strip().upper())
            elif isinstance(item, str) and item.strip():
                eligible_countries.append(item.strip().upper())
    eligibility = Eligibility.ELIGIBLE if isinstance(restrictions, list) and not restrictions else Eligibility.UNKNOWN
    # A title can state a work location while locationRestrictions describe
    # where the applicant must live. If those facts conflict, do not treat the
    # restriction list as a confirmed rejection for that explicitly named place.
    mentioned_countries = _mentioned_country_codes(data.get("title"))
    restriction_conflicts = sorted(mentioned_countries - set(eligible_countries)) if eligible_countries else []
    raw_seniority = data.get("seniority")
    seniority = ", ".join(str(v) for v in raw_seniority) if isinstance(raw_seniority, list) else raw_seniority
    published = data.get("pubDate")
    try:
        published_at = datetime.fromtimestamp(float(published), tz=timezone.utc) if published is not None else None
    except (TypeError, ValueError, OverflowError, OSError):
        published_at = None
    guid = data.get("guid") or data.get("id")
    return Job(
        job_id=f"himalayas:{guid or _fallback_id(data)}",
        provider_id=str(guid) if guid is not None else None,
        title=data.get("title") or "Untitled job",
        company=data.get("companyName") or "Unknown company",
        # Himalayas is a remote-jobs feed. `locationRestrictions` is where the
        # applicant must reside, not the physical location of the job.
        location="Remote",
        country=None,
        work_model=normalize_work_model(data.get("workModel") or "remote"),
        employment_type=normalize_employment_type(data.get("employmentType")),
        seniority=seniority if isinstance(seniority, str) else None,
        description=data.get("description") if isinstance(data.get("description"), str) else data.get("excerpt"),
        salary_min=_number(data.get("minSalary")),
        salary_max=_number(data.get("maxSalary")),
        salary_currency=data.get("currency") if isinstance(data.get("currency"), str) else None,
        salary_period=data.get("salaryPeriod") if isinstance(data.get("salaryPeriod"), str) else None,
        source="himalayas",
        source_url=data.get("applicationLink") if isinstance(data.get("applicationLink"), str) else None,
        published_at=published_at,
        country_eligibility=eligibility,
        eligible_countries=eligible_countries,
        timezone_restrictions=[x for x in data.get("timezoneRestrictions", []) if isinstance(x, str)] if isinstance(data.get("timezoneRestrictions"), list) else [],
        raw_metadata={
            "company_slug": data.get("companySlug"),
            "categories": data.get("categories", []),
            "excerpt": data.get("excerpt"),
            "eligibility_conflicts": restriction_conflicts,
        },
    )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def _mentioned_country_codes(title: Any) -> set[str]:
    """Return explicitly named country codes from titles, without inferring from descriptions."""
    if not isinstance(title, str):
        return set()
    normalized = f" {title.casefold()} "
    if " germany " in normalized or " deutschland " in normalized:
        return {"DE"}
    return set()


def _fallback_id(data: dict[str, Any]) -> str:
    return "-".join(str(data.get(key, "")) for key in ("companyName", "title")) or "unknown"
