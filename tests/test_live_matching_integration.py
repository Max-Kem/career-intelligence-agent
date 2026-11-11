from __future__ import annotations

from pathlib import Path

from career_agent.candidate import CandidateProfile
from career_agent.discovery import JobDiscoveryService
from career_agent.integration import discover_and_match
from career_agent.models import Job, WorkModel


ROOT = Path(__file__).resolve().parents[1]


class MockAdapter:
    def __init__(self, jobs):
        self.jobs = jobs
        self.queries = []

    def search(self, query="", **kwargs):
        self.queries.append(query)
        return self.jobs


def test_discovery_to_matching_integration_uses_profile_query_and_preserves_url():
    profile = CandidateProfile.model_validate_json(
        (ROOT / "fixtures" / "candidate_profile_b.json").read_text(encoding="utf-8")
    )
    job = Job(
        job_id="mock-junior-data-analyst", title="Junior Data Analyst", company="Mock Analytics",
        location="Berlin", work_model=WorkModel.HYBRID, employment_type="full-time",
        seniority="Entry-level",
        description="Use SQL and Python for data analysis. Prepare Excel reports and data visualization dashboards.",
        source="arbeitnow", source_url="https://jobs.example.test/mock-analyst",
        eligible_countries=["DE"], raw_metadata={"categories": ["Technology", "Analytics"]},
    )
    arbeitnow = MockAdapter([job])
    himalayas = MockAdapter([])
    run = discover_and_match(
        profile,
        search_query_limit=2,
        discovery_service=JobDiscoveryService(arbeitnow=arbeitnow, himalayas=himalayas),
    )

    assert run.queries == ["Junior Data Analyst", "Business Intelligence Analyst"]
    assert arbeitnow.queries == himalayas.queries == run.queries
    assert run.discovery.jobs == [job]
    assert len(run.matching.top_matches) == 1
    match = run.matching.top_matches[0]
    assert match.job.source_url == "https://jobs.example.test/mock-analyst"
    assert match.score is not None


def test_query_override_is_used_instead_of_derived_title():
    profile = CandidateProfile(candidate_id="override", name="Override", desired_job_titles=["Original title"])
    arbeitnow = MockAdapter([])
    himalayas = MockAdapter([])
    run = discover_and_match(
        profile, query_override="  Data Analyst  ",
        discovery_service=JobDiscoveryService(arbeitnow=arbeitnow, himalayas=himalayas),
    )
    assert run.queries == ["Data Analyst"]
    assert arbeitnow.queries == himalayas.queries == ["Data Analyst"]


def test_multiple_role_queries_are_bounded_and_deduplicated_across_queries_and_sources():
    candidate = CandidateProfile(
        candidate_id="multi", name="Multi", desired_job_titles=["Data Analyst", "Business Intelligence Analyst"],
    )
    same_listing = Job(
        job_id="shared-role-listing", title="Data Analyst - Business Intelligence Analyst",
        company="Shared Analytics", source="arbeitnow", source_url="https://jobs.example.test/shared-role",
        description="Analyze data and build business intelligence reports.",
    )
    duplicate_from_second_source = same_listing.model_copy(update={"source": "himalayas"})
    arbeitnow = MockAdapter([same_listing])
    himalayas = MockAdapter([duplicate_from_second_source])
    run = discover_and_match(
        candidate, search_query_limit=2,
        discovery_service=JobDiscoveryService(arbeitnow=arbeitnow, himalayas=himalayas),
    )
    assert run.queries == ["Data Analyst", "Business Intelligence Analyst"]
    assert arbeitnow.queries == himalayas.queries == run.queries
    assert len(run.discovery.jobs) == 1
    assert run.discovery.duplicates_removed == 3
    assert run.discovery.provider_counts == {"arbeitnow": 2, "himalayas": 2}
