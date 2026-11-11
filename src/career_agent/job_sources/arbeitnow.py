from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

from ..models import EmploymentType, Job, WorkModel, normalize_employment_type, normalize_work_model
from ..geography import country_from_location
from .common import BaseAdapter, ProviderError


class ArbeitnowAdapter(BaseAdapter):
    source = "arbeitnow"
    endpoint = "https://www.arbeitnow.com/api/job-board-api"

    def search(self, query: str = "", *, max_pages: int = 2) -> list[Job]:
        if max_pages < 1 or max_pages > 20:
            raise ValueError("max_pages must be between 1 and 20")
        url = self.endpoint
        jobs: list[Job] = []
        seen_pages: set[str] = set()
        for _ in range(max_pages):
            if url in seen_pages:
                break
            seen_pages.add(url)
            payload = self._get_json(url)
            if not isinstance(payload, dict) or not isinstance(payload.get("data", []), list):
                raise ProviderError("Arbeitnow returned an unexpected response shape")
            for item in payload.get("data", []):
                if isinstance(item, dict):
                    job = normalize_arbeitnow(item)
                    if not query or _matches(job, query):
                        jobs.append(job)
            links = payload.get("links")
            next_url = links.get("next") if isinstance(links, dict) else None
            if not isinstance(next_url, str) or not next_url:
                break
            url = urljoin(self.endpoint, next_url)
        return jobs


def _matches(job: Job, query: str) -> bool:
    haystack = " ".join((job.title, job.company, job.location, job.description or "", " ".join(job.raw_metadata.get("tags", [])))).casefold()
    return all(term.casefold() in haystack for term in query.split())


def normalize_arbeitnow(data: dict[str, Any]) -> Job:
    raw_types = data.get("job_types")
    employment = normalize_employment_type(raw_types)
    published = data.get("created_at")
    try:
        published_at = datetime.fromtimestamp(float(published), tz=timezone.utc) if published is not None else None
    except (TypeError, ValueError, OverflowError, OSError):
        published_at = None
    tags = data.get("tags") if isinstance(data.get("tags"), list) else []
    url = data.get("url") if isinstance(data.get("url"), str) else None
    source_id = data.get("slug") or data.get("id") or url
    location = data.get("location") or "Unknown"
    work_model = normalize_work_model(data.get("remote"))
    return Job(
        job_id=f"arbeitnow:{source_id or url or _fallback_id(data)}",
        provider_id=str(source_id) if source_id is not None else None,
        title=data.get("title") or "Untitled job",
        company=data.get("company_name") or "Unknown company",
        location=location,
        # A named physical city can establish a country. A remote listing's
        # city-like text is not applicant eligibility, so leave it unassigned.
        country=country_from_location(location) if work_model != WorkModel.REMOTE else None,
        work_model=work_model,
        employment_type=employment,
        description=data.get("description") if isinstance(data.get("description"), str) else None,
        source="arbeitnow",
        source_url=url,
        published_at=published_at,
        raw_metadata={"tags": [str(tag) for tag in tags], "job_types": raw_types, "remote": data.get("remote")},
    )


def _fallback_id(data: dict[str, Any]) -> str:
    return "-".join(str(data.get(key, "")) for key in ("company_name", "title", "location")) or "unknown"
