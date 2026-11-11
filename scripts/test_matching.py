"""Run explainable matching on the two synthetic candidate profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.candidate import CandidateProfile  # noqa: E402
from career_agent.matching import match_candidate_to_jobs  # noqa: E402
from career_agent.models import Job  # noqa: E402


def main() -> int:
    profiles = [
        CandidateProfile.model_validate_json((ROOT / "fixtures" / f"candidate_profile_{letter}.json").read_text(encoding="utf-8"))
        for letter in ("a", "b")
    ]
    jobs_data = json.loads((ROOT / "fixtures" / "matching_jobs.json").read_text(encoding="utf-8"))
    jobs = [Job.model_validate(item) for item in jobs_data]

    for profile in profiles:
        print(f"\n=== {profile.name} ({profile.candidate_id}) ===")
        print(profile.summary or "No summary provided.")
        result = match_candidate_to_jobs(profile, jobs)
        if not result.top_matches:
            print("No sufficiently supported matches for the top five.")
        for index, match in enumerate(result.top_matches, start=1):
            print(f"\n{index}. {match.job.title} - {match.job.company}")
            print(f"   Score: {match.score}/100 | Evidence coverage: {match.evidence_coverage}% | {match.support_status.value}")
            print(f"   Source: {match.job.source} | {match.job.source_url or 'URL unavailable'}")
            print(f"   Strengths: {'; '.join(match.strengths[:4]) or 'No strong evidence recorded.'}")
            print(f"   Potential gaps: {'; '.join(match.gaps[:4]) or 'No explicit gaps recorded.'}")
            print(f"   Explanation: {match.explanation}")
        provisional = [match for match in result.ranked_matches if match not in result.top_matches]
        if provisional:
            print(f"\nNot in top five (low score, provisional, or insufficient evidence): {len(provisional)}")
            for match in provisional:
                print(f"- {match.job.title}: {match.score if match.score is not None else 'no score'}; {match.support_status.value}; coverage {match.evidence_coverage}%")
        if result.excluded_jobs:
            print(f"Confirmed hard-filter exclusions: {len(result.excluded_jobs)}")
            for exclusion in result.excluded_jobs:
                print(f"- {exclusion.job.title}: {'; '.join(exclusion.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
