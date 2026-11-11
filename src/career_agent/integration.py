from __future__ import annotations

from dataclasses import dataclass

from .candidate import CandidateProfile
from .discovery import DiscoveryResult, JobDiscoveryService, SearchPreferences
from .filtering import FilterDiagnostics
from .matching import MatchingResult, match_candidate_to_jobs
from .discovery import deduplicate_jobs


@dataclass
class CandidateJobRun:
    candidate: CandidateProfile
    queries: list[str]
    discovery: DiscoveryResult
    matching: MatchingResult

    @property
    def query(self) -> str:
        """Joined query string retained for convenient display/compatibility."""
        return "; ".join(self.queries)


@dataclass
class CandidateDiscoveryRun:
    candidate: CandidateProfile
    queries: list[str]
    discovery: DiscoveryResult


def derive_search_query(candidate: CandidateProfile) -> str:
    return derive_search_queries(candidate, limit=1)[0]


def derive_search_queries(candidate: CandidateProfile, *, limit: int = 3) -> list[str]:
    """Choose up to three distinct role queries, preferring desired titles."""
    if not 1 <= limit <= 3:
        raise ValueError("search query limit must be between 1 and 3")
    queries: list[str] = []
    seen: set[str] = set()
    role_sources = candidate.desired_job_titles or candidate.career_interests
    for value in role_sources:
        query = " ".join(value.split())
        key = query.casefold()
        if query and key not in seen:
            queries.append(query)
            seen.add(key)
        if len(queries) >= limit:
            break
    if not queries:
        raise ValueError("Candidate profile needs at least one desired job title or career interest.")
    return queries


def discover_and_match(
    candidate: CandidateProfile,
    *,
    query_override: str | None = None,
    max_pages_per_provider: int = 1,
    search_query_limit: int = 3,
    top_n: int = 5,
    discovery_service: JobDiscoveryService | None = None,
) -> CandidateJobRun:
    discovery_run = discover_candidate_jobs(
        candidate, query_override=query_override,
        max_pages_per_provider=max_pages_per_provider,
        search_query_limit=search_query_limit, discovery_service=discovery_service,
    )
    matching = match_candidate_to_jobs(candidate, discovery_run.discovery.jobs, top_n=top_n)
    return CandidateJobRun(
        candidate=candidate, queries=discovery_run.queries,
        discovery=discovery_run.discovery, matching=matching,
    )


def discover_candidate_jobs(
    candidate: CandidateProfile,
    *,
    query_override: str | None = None,
    max_pages_per_provider: int = 1,
    search_query_limit: int = 3,
    discovery_service: JobDiscoveryService | None = None,
) -> CandidateDiscoveryRun:
    """Discover and combine jobs for a profile without running candidate matching."""
    if not 1 <= max_pages_per_provider <= 5:
        raise ValueError("max_pages_per_provider must be between 1 and 5 for bounded multi-query searches")
    if not 1 <= search_query_limit <= 3:
        raise ValueError("search_query_limit must be between 1 and 3")
    queries = [query_override.strip()] if query_override and query_override.strip() else derive_search_queries(candidate, limit=search_query_limit)
    country = candidate.country_preferences[0] if candidate.country_preferences else None
    service = discovery_service or JobDiscoveryService()
    query_results = [service.search(SearchPreferences(
        query=query, country=country, max_pages_per_provider=max_pages_per_provider,
    )) for query in queries]
    jobs, cross_query_duplicates = deduplicate_jobs([job for item in query_results for job in item.jobs])
    provider_names = sorted(set().union(*(item.provider_counts.keys() for item in query_results)))
    unique_counts = {provider: sum(job.source == provider for job in jobs) for provider in provider_names}
    errors: dict[str, str] = {}
    for provider in provider_names:
        failures = [f"{query!r}: {item.errors[provider]}" for query, item in zip(queries, query_results) if provider in item.errors]
        if failures:
            errors[provider] = "; ".join(failures)
    discovery = DiscoveryResult(
        jobs=jobs,
        provider_counts={provider: sum(item.provider_counts.get(provider, 0) for item in query_results) for provider in provider_names},
        deduplicated_provider_counts=unique_counts,
        final_provider_counts=unique_counts.copy(),
        errors=errors,
        duplicates_removed=sum(item.duplicates_removed for item in query_results) + cross_query_duplicates,
        filter_diagnostics=_combine_diagnostics(query_results),
    )
    return CandidateDiscoveryRun(candidate=candidate, queries=queries, discovery=discovery)


def _combine_diagnostics(results: list[DiscoveryResult]) -> FilterDiagnostics:
    combined = FilterDiagnostics()
    for result in results:
        current = result.filter_diagnostics
        for name, count in current.excluded_counts.items():
            combined.excluded_counts[name] = combined.excluded_counts.get(name, 0) + count
        for name, reasons in current.exclusion_reasons.items():
            bucket = combined.exclusion_reasons.setdefault(name, {})
            for reason, count in reasons.items():
                bucket[reason] = bucket.get(reason, 0) + count
        for name, ids in current.excluded_job_ids.items():
            bucket = combined.excluded_job_ids.setdefault(name, [])
            bucket.extend(job_id for job_id in ids if job_id not in bucket)
            del bucket[5:]
        for field_name in (
            "missing_country_count", "missing_location_count",
            "missing_country_exclusions", "missing_location_exclusions",
        ):
            setattr(combined, field_name, getattr(combined, field_name) + getattr(current, field_name))
    return combined
