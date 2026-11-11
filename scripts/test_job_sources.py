"""Small live smoke test for the public job APIs (no credentials required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_agent.discovery import JobDiscoveryService, SearchPreferences  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="AI Consultant")
    parser.add_argument("--country", default="DE")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.pages < 1 or args.limit < 1:
        parser.error("--pages and --limit must be positive")

    result = JobDiscoveryService().search(SearchPreferences(
        query=args.query, country=args.country, max_pages_per_provider=args.pages,
    ))
    print(f"Query: {args.query!r} | country: {args.country} | pages/provider: {args.pages}")
    for provider, count in result.provider_counts.items():
        status = f"FAILED: {result.errors[provider]}" if provider in result.errors else "ok"
        print(f"{provider}: {count} normalized results; {result.final_provider_counts.get(provider, 0)} final ({status})")
    print(f"Combined after deduplication and filtering: {len(result.jobs)} ({result.duplicates_removed} duplicates removed)")
    print(f"After deduplication by provider: {result.deduplicated_provider_counts}")
    diagnostics = result.filter_diagnostics
    print("Excluded by filter:")
    for filter_name, count in diagnostics.excluded_counts.items():
        details = diagnostics.exclusion_reasons.get(filter_name, {})
        print(f"  {filter_name}: {count} {details}")
    print(
        "Missing information among deduplicated jobs: "
        f"country={diagnostics.missing_country_count}, "
        f"location={diagnostics.missing_location_count}; "
        f"excluded due to missing country={diagnostics.missing_country_exclusions}, "
        f"excluded due to missing location={diagnostics.missing_location_exclusions}"
    )
    if diagnostics.excluded_job_ids:
        print(f"Sample excluded job IDs (up to 5 per filter): {diagnostics.excluded_job_ids}")
    for job in result.jobs[:args.limit]:
        print(f"- {job.title} — {job.company} | {job.location} | {job.employment_type.value} | {job.source}")
        print(f"  {job.source_url or 'No source URL provided'}")
    return 1 if result.errors and not result.jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
