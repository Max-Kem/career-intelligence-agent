from __future__ import annotations

import httpx
import pytest

from career_agent.discovery import JobDiscoveryService, deduplicate_jobs
from career_agent.filtering import country_eligibility, filter_jobs
from career_agent.job_sources.arbeitnow import ArbeitnowAdapter, normalize_arbeitnow
from career_agent.job_sources.common import ProviderError
from career_agent.job_sources.himalayas import HimalayasAdapter, normalize_himalayas
from career_agent.models import Eligibility, EmploymentType, Job, WorkModel


def test_arbeitnow_normalization_and_unknown_false_remote():
    job = normalize_arbeitnow({"title": "Consultant", "company_name": "Acme", "remote": False,
                               "job_types": ["Full time"], "url": "https://example.test/job",
                               "created_at": "not a date", "tags": ["AI"]})
    assert job.work_model == WorkModel.UNKNOWN
    assert job.employment_type == EmploymentType.FULL_TIME
    assert job.published_at is None
    assert job.country is None
    assert job.raw_metadata["tags"] == ["AI"]


def test_arbeitnow_empty_and_malformed_entries_are_safe():
    job = normalize_arbeitnow({"title": None, "job_types": "odd", "tags": "bad"})
    assert job.title == "Untitled job"
    assert job.company == "Unknown company"
    assert job.employment_type == EmploymentType.UNKNOWN
    assert ArbeitnowAdapter(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [], "links": {}})))).search() == []


def test_himalayas_normalization_extracts_confirmed_schema_fields():
    job = normalize_himalayas({"guid": "acme-role-1", "title": "AI Consultant", "companyName": "Acme",
                               "employmentType": "Contractor", "locationRestrictions": [{"alpha2": "DE", "name": "Germany"}],
                               "timezoneRestrictions": ["UTC+1"], "description": "Build AI systems", "minSalary": 10,
                               "currency": "EUR", "pubDate": 1760000000, "applicationLink": "https://example.test"})
    assert job.job_id == "himalayas:acme-role-1"
    assert job.employment_type == EmploymentType.CONTRACT
    assert job.eligible_countries == ["DE"]
    assert job.location == "Remote"
    assert job.salary_min == 10
    assert job.description == "Build AI systems"
    assert job.country_eligibility == Eligibility.UNKNOWN


def test_himalayas_missing_fields_and_global_listing():
    job = normalize_himalayas({"title": None, "locationRestrictions": [], "minSalary": "bad"})
    assert job.title == "Untitled job"
    assert job.country_eligibility == Eligibility.ELIGIBLE
    assert job.salary_min is None
    assert job.work_model == WorkModel.REMOTE


def test_himalayas_germany_restriction_is_eligible_for_germany():
    job = normalize_himalayas({
        "guid": "germany-role", "title": "Pre-Sales Engineer - Remote in Germany",
        "locationRestrictions": [{"alpha2": "DE", "name": "Germany", "slug": "germany"}],
    })
    assert job.country is None  # Restrictions describe candidate residence, not the job's physical country.
    assert job.eligible_countries == ["DE"]
    assert country_eligibility(job, "DE") == Eligibility.ELIGIBLE
    assert filter_jobs([job], country="DE") == [job]


def test_himalayas_other_country_restriction_is_ineligible_unless_title_conflicts():
    restricted = normalize_himalayas({
        "guid": "us-only", "title": "AI Consultant", "locationRestrictions": [
            {"alpha2": "US", "name": "United States", "slug": "united-states"},
        ],
    })
    assert country_eligibility(restricted, "DE") == Eligibility.INELIGIBLE
    assert filter_jobs([restricted], country="DE") == []

    conflicting = normalize_himalayas({
        "guid": "germany-title-us-restriction", "title": "Debt Collections Advisor - Germany",
        "locationRestrictions": [{"alpha2": "US", "name": "United States", "slug": "united-states"}],
    })
    assert country_eligibility(conflicting, "DE") == Eligibility.UNKNOWN
    assert filter_jobs([conflicting], country="DE") == [conflicting]


def test_deduplicate_by_url_and_fallback():
    a = Job(job_id="a", title="AI Consultant", company="Acme", location="Berlin", source="a", source_url="https://example.test/x/")
    b = Job(job_id="b", title="AI Consultant", company="Other", location="Elsewhere", source="b", source_url="https://www.example.test/x")
    c = Job(job_id="c", title="Analyst", company="Acme", location="Berlin", source="a")
    d = Job(job_id="d", title="Analyst", company="Acme", location="Berlin", source="b")
    unique, removed = deduplicate_jobs([a, b, c, d])
    assert len(unique) == 2
    assert removed == 2


def test_himalayas_paging_and_query_params_mocked():
    requests = []
    def handler(request):
        requests.append(request)
        jobs = [{"guid": "x", "title": "AI Consultant", "locationRestrictions": []}] if len(requests) == 1 else []
        return httpx.Response(200, json={"jobs": jobs})
    adapter = HimalayasAdapter(client=httpx.Client(transport=httpx.MockTransport(handler)))
    jobs = adapter.search("AI Consultant", country="DE", max_pages=2)
    assert len(jobs) == 1
    assert len(requests) == 2  # A short non-empty page does not necessarily end the result set.
    assert requests[0].url.params["country"] == "DE"
    assert requests[0].url.params["q"] == "AI Consultant"


def test_provider_failure_does_not_hide_other_results():
    class Failing:
        def search(self, *args, **kwargs):
            raise ProviderError("temporary outage")
    class Working:
        def search(self, *args, **kwargs):
            return [Job(job_id="ok", title="AI Consultant", company="A", source="himalayas")]
    result = JobDiscoveryService(arbeitnow=Failing(), himalayas=Working()).search(query="AI Consultant")
    assert len(result.jobs) == 1
    assert result.errors == {"arbeitnow": "temporary outage"}
    assert result.provider_counts == {"arbeitnow": 0, "himalayas": 1}


def test_http_error_has_provider_context_and_retries_transient():
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json={"data": [], "links": {}})
    adapter = ArbeitnowAdapter(retries=1, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert adapter.search() == []
    assert calls == 2


def test_unexpected_payload_shape_raises_useful_error():
    adapter = HimalayasAdapter(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        adapter.search()
