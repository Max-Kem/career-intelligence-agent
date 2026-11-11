"""Discover live jobs and rank them for one synthetic candidate profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.candidate import CandidateProfile  # noqa: E402
from career_agent.integration import discover_and_match  # noqa: E402
from career_agent.matching import MINIMUM_MATCH_SCORE, SupportStatus  # noqa: E402


def _rejection_reasons(match) -> list[str]:
    reasons: list[str] = []
    if match.score is None:
        reasons.append("insufficient evidence for an overall score")
    if match.support_status.value == "provisional":
        reasons.append(f"provisional evidence coverage ({match.evidence_coverage}%)")
    if match.support_status.value == "insufficiently_supported":
        reasons.append(f"insufficient evidence coverage ({match.evidence_coverage}%)")
    if match.hard_filter_status == "unknown":
        reasons.extend(match.unresolved_requirements)
    if match.score is not None and match.score < 50:
        reasons.append(f"fit score below existing 50/100 recommendation threshold ({match.score})")
    reasons.extend(match.recommendation_warnings)
    reasons.extend(match.recommendation_blockers)
    return reasons


def _print_job(job) -> None:
    print(f"   Source URL: {job.source_url or 'not supplied by provider'}")


def _print_geographic_evidence(status, evidence) -> None:
    if status is not None:
        print(f"   Geographic status: {status.value}")
    for item in evidence:
        print(f"   Geographic evidence: {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("a", "b"), default="a")
    parser.add_argument("--query", help="Optional single search query override; otherwise uses up to three desired roles.")
    parser.add_argument("--queries", type=int, default=3, help="Maximum distinct profile role queries (1-3).")
    parser.add_argument("--pages", type=int, default=1, help="Maximum pages per provider and query (1-5).")
    args = parser.parse_args()
    if not 1 <= args.pages <= 5:
        parser.error("--pages must be between 1 and 5")
    if not 1 <= args.queries <= 3:
        parser.error("--queries must be between 1 and 3")

    profile_path = ROOT / "fixtures" / f"candidate_profile_{args.profile}.json"
    candidate = CandidateProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    run = discover_and_match(
        candidate, query_override=args.query, max_pages_per_provider=args.pages,
        search_query_limit=args.queries,
    )
    discovery, matching = run.discovery, run.matching

    print(f"Candidate: {candidate.name} ({candidate.candidate_id})")
    print(f"Search queries: {', '.join(repr(query) for query in run.queries)} | pages/provider/query: {args.pages}")
    for provider, count in discovery.provider_counts.items():
        error = f" | ERROR: {discovery.errors[provider]}" if provider in discovery.errors else ""
        print(f"{provider}: {count} normalized jobs; {discovery.final_provider_counts.get(provider, 0)} after discovery filters{error}")
    print(f"Discovery duplicates removed: {discovery.duplicates_removed}")
    print(f"Jobs entering matching: {len(discovery.jobs)}")

    scored = sum(match.score is not None for match in matching.ranked_matches)
    qualified = [
        match for match in matching.ranked_matches
        if match.support_status == SupportStatus.SUFFICIENT
        and match.hard_filter_status == "passed"
        and match.score is not None
        and match.score >= MINIMUM_MATCH_SCORE
        and not match.recommendation_blockers
    ]
    qualified_ids = {match.job.job_id for match in qualified}
    unresolved = [match for match in matching.ranked_matches if match.hard_filter_status == "unknown"]
    insufficient = [match for match in matching.ranked_matches if match.support_status != SupportStatus.SUFFICIENT]
    seniority_blocked = [match for match in matching.ranked_matches if match.recommendation_blockers]
    nonqualified = [match for match in matching.ranked_matches if match.job.job_id not in qualified_ids]
    nonrecommendation_union = [
        match for match in nonqualified
        if match.support_status != SupportStatus.SUFFICIENT or match.hard_filter_status == "unknown"
    ]
    low_fit = [
        match for match in nonqualified
        if match.score is not None and match.score < MINIMUM_MATCH_SCORE
        and match.hard_filter_status == "passed" and match.support_status == SupportStatus.SUFFICIENT
    ]
    print(f"Jobs receiving an overall score: {scored}")
    print(f"Jobs qualifying as recommendations: {len(qualified)} (displaying top {len(matching.top_matches)})")
    print(f"Not recommended due to insufficient/provisional evidence or unresolved hard requirements: {len(nonrecommendation_union)}")
    print(f"  Insufficient or provisional evidence: {len(insufficient)}")
    print(f"  Unresolved hard requirements (may overlap): {len(unresolved)}")
    print(f"  Seniority preference blockers (may overlap): {len(seniority_blocked)}")
    print(f"  Sufficient evidence but below fit threshold: {len(low_fit)}")
    print(f"Confirmed hard-filter exclusions: {len(matching.excluded_jobs)}")

    if matching.top_matches:
        print("\nRecommendations:")
        for index, match in enumerate(matching.top_matches, start=1):
            print(f"\n{index}. {match.job.title} - {match.job.company}")
            print(f"   Score: {match.score}/100 | evidence coverage: {match.evidence_coverage}%")
            print(f"   Main strengths: {'; '.join(match.strengths[:4]) or 'none recorded'}")
            print(f"   Potential gaps: {'; '.join(match.gaps[:4]) or 'none recorded'}")
            _print_geographic_evidence(match.geographic_status, match.geographic_evidence)
            _print_job(match.job)
    else:
        print("\nNo jobs met the existing recommendation criteria.")

    not_recommended = nonqualified
    if not_recommended:
        print("\nJobs not qualifying as recommendations:")
        for match in not_recommended:
            print(f"- {match.job.title} - {match.job.company} | score: {match.score if match.score is not None else 'unscored'} | coverage: {match.evidence_coverage}%")
            reasons = _rejection_reasons(match)
            print(f"  Reason: {'; '.join(reasons) if reasons else 'below recommendation qualification'}")
            _print_geographic_evidence(match.geographic_status, match.geographic_evidence)
            _print_job(match.job)
    for exclusion in matching.excluded_jobs:
        print(f"- Hard-filtered: {exclusion.job.title} - {exclusion.job.company} | score: {exclusion.score if exclusion.score is not None else 'unscored'} | {'; '.join(exclusion.violations)}")
        print(f"  Explanation: {exclusion.explanation}")
        _print_geographic_evidence(exclusion.geographic_status, exclusion.geographic_evidence)
        _print_job(exclusion.job)

    if discovery.errors and not discovery.jobs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
