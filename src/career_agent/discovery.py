from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator

from .filtering import FilterDiagnostics, filter_jobs_with_diagnostics
from .job_sources import ArbeitnowAdapter, HimalayasAdapter
from .job_sources.common import ProviderError
from .models import EmploymentType, Job, WorkModel


class SearchPreferences(BaseModel):
    query: str = ""
    country: str | None = None
    work_model: WorkModel | None = None
    employment_type: EmploymentType | None = None
    strict_unknown: bool = False
    max_pages_per_provider: int = Field(default=2, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_is_string(cls, value: str) -> str:
        return value.strip()

    @field_validator("country")
    @classmethod
    def normalize_country(cls, value: str | None) -> str | None:
        return value.strip().upper() if value and value.strip() else None


@dataclass
class DiscoveryResult:
    jobs: list[Job]
    provider_counts: dict[str, int] = field(default_factory=dict)
    deduplicated_provider_counts: dict[str, int] = field(default_factory=dict)
    final_provider_counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    duplicates_removed: int = 0
    filter_diagnostics: FilterDiagnostics = field(default_factory=FilterDiagnostics)


class JobDiscoveryService:
    def __init__(self, arbeitnow: Any | None = None, himalayas: Any | None = None):
        self.providers = {
            "arbeitnow": arbeitnow if arbeitnow is not None else ArbeitnowAdapter(),
            "himalayas": himalayas if himalayas is not None else HimalayasAdapter(),
        }

    def search(self, preferences: SearchPreferences | None = None, **kwargs: Any) -> DiscoveryResult:
        prefs = preferences or SearchPreferences(**kwargs)
        by_provider: dict[str, list[Job]] = {}
        errors: dict[str, str] = {}
        for name, adapter in self.providers.items():
            try:
                if name == "arbeitnow":
                    found = adapter.search(prefs.query, max_pages=prefs.max_pages_per_provider)
                else:
                    found = adapter.search(prefs.query, country=prefs.country, max_pages=prefs.max_pages_per_provider)
                by_provider[name] = found
            except (ProviderError, ValueError) as exc:
                errors[name] = str(exc)
                by_provider[name] = []
            except Exception as exc:  # isolate unexpected adapter failures
                errors[name] = f"Unexpected provider error: {type(exc).__name__}: {exc}"
                by_provider[name] = []
        combined = [job for batch in by_provider.values() for job in batch]
        unique, removed = deduplicate_jobs(combined)
        unique_by_provider = {name: 0 for name in by_provider}
        for job in unique:
            unique_by_provider[job.source] = unique_by_provider.get(job.source, 0) + 1
        filter_result = filter_jobs_with_diagnostics(
            unique, keywords=prefs.query, country=prefs.country,
            work_model=prefs.work_model, employment_type=prefs.employment_type,
            strict_unknown=prefs.strict_unknown,
        )
        final_by_provider = {name: 0 for name in by_provider}
        for job in filter_result.jobs:
            final_by_provider[job.source] = final_by_provider.get(job.source, 0) + 1
        return DiscoveryResult(
            jobs=filter_result.jobs,
            provider_counts={name: len(batch) for name, batch in by_provider.items()},
            deduplicated_provider_counts=unique_by_provider,
            final_provider_counts=final_by_provider,
            errors=errors,
            duplicates_removed=removed,
            filter_diagnostics=filter_result.diagnostics,
        )


def deduplicate_jobs(jobs: list[Job]) -> tuple[list[Job], int]:
    unique: list[Job] = []
    seen: set[str] = set()
    for job in jobs:
        keys = _dedupe_keys(job)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        unique.append(job)
    return unique, len(jobs) - len(unique)


def _dedupe_keys(job: Job) -> set[str]:
    keys: set[str] = set()
    if job.source_url:
        parts = urlsplit(job.source_url.strip())
        url = urlunsplit((parts.scheme.lower(), parts.netloc.lower().removeprefix("www."), parts.path.rstrip("/"), parts.query, ""))
        keys.add(f"url:{url.casefold()}")
    if job.provider_id:
        # Provider IDs are namespaced to avoid accidental cross-source collisions.
        keys.add(f"id:{job.source.casefold()}:{job.provider_id.casefold()}")
    canonical = "|".join((job.company, job.title, job.location))
    keys.add("fallback:" + " ".join(canonical.casefold().split()))
    return keys
