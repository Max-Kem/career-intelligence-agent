from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from career_agent.company_research import TavilyCompanyResearchClient, unique_recommended_companies


def client_for(handler, *, api_key="test-secret", max_retries=0, sleeps=None):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return TavilyCompanyResearchClient(
        api_key=api_key,
        client=http_client,
        max_retries=max_retries,
        sleep=sleeps.append if sleeps is not None else lambda _delay: None,
    )


def test_search_uses_basic_depth_and_preserves_sources_with_company_site_preferred():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "results": [
                {"title": "Company profile", "url": "https://news.example/company", "content": "Third-party profile excerpt."},
                {"title": "About Celonis", "url": "https://www.celonis.com/about-us", "content": "Celonis process intelligence platform.", "published_date": "2026-05-10"},
            ]
        })

    result = client_for(handler).research_company("Celonis")

    assert result.status == "succeeded"
    assert result.snapshot == "Celonis process intelligence platform."
    assert len(result.sources) == 2
    assert result.sources[0].title == "About Celonis"
    assert result.sources[0].url == "https://www.celonis.com/about-us"
    assert result.sources[0].published_date == "2026-05-10"
    assert result.sources[0].source_type == "possible_company_website"
    assert result.sources[0].identity_status == "verified"
    assert result.sources[1].source_type == "third_party"
    assert result.identity_status == "verified"
    assert result.raw_results[0]["title"] == "Company profile"
    assert result.raw_results[1]["url"] == "https://www.celonis.com/about-us"
    assert requests[0].url == "https://api.tavily.com/search"
    assert requests[0].headers["Authorization"] == "Bearer test-secret"
    request_body = json.loads(requests[0].content)
    assert request_body["search_depth"] == "basic"
    assert request_body["max_results"] == 3
    assert "Celonis" in request_body["query"]
    assert requests[0].headers["Content-Type"] == "application/json"


def test_search_caps_results_at_three():
    def handler(_request):
        return httpx.Response(200, json={"results": [
            {"title": f"Result {n}", "url": f"https://source{n}.example", "content": f"Snippet {n}"}
            for n in range(5)
        ]})

    result = client_for(handler).research_company("Example Corp")
    assert len(result.sources) == 3


def test_missing_api_key_skips_request_and_does_not_load_real_dotenv(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    calls = []
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request)))
    client = TavilyCompanyResearchClient(
        client=http_client, dotenv_loader=lambda **_kwargs: None,
    )
    result = client.research_company("Example")
    assert result.status == "missing_api_key"
    assert calls == []


def test_authentication_failure_is_not_retried():
    calls = []

    def handler(_request):
        calls.append(1)
        return httpx.Response(401, json={"detail": "Unauthorized"})

    result = client_for(handler, max_retries=1).research_company("Example")
    assert result.status == "unavailable"
    assert "HTTP 401" in result.error
    assert len(calls) == 1
    assert "test-secret" not in str(result.model_dump())


def test_rate_limit_is_retried_once():
    calls = []
    sleeps = []

    def handler(_request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"results": []})

    result = client_for(handler, max_retries=1, sleeps=sleeps).research_company("Example")
    assert result.status == "no_results"
    assert len(calls) == 2
    assert sleeps == [0.2]


def test_recommended_companies_are_deduplicated_case_insensitively():
    recommendations = [
        SimpleNamespace(job=SimpleNamespace(company="Celonis")),
        SimpleNamespace(job=SimpleNamespace(company=" celonis ")),
        SimpleNamespace(job=SimpleNamespace(company="Other Company")),
    ]
    assert unique_recommended_companies(recommendations) == ["Celonis", "Other Company"]


def test_unrelated_company_results_are_not_attributed_as_a_snapshot():
    def handler(_request):
        return httpx.Response(200, json={"results": [
            {"title": "Advisory Board", "url": "https://www.advisory.com/home", "content": "A research membership organization."},
            {"title": "Synthetic advisory boards", "url": "https://example.org/article", "content": "An article about simulated advisory panels."},
        ]})

    result = client_for(handler).research_company("Synthetic Advisory Group")
    assert result.status == "succeeded"  # API succeeded, identity did not.
    assert result.identity_status == "unverified"
    assert result.snapshot is None
    assert all(source.identity_status == "unverified" for source in result.sources)
    assert len(result.raw_results) == 2
    assert result.raw_results[0]["url"] == "https://www.advisory.com/home"


def test_plausible_identity_requires_both_domain_and_explicit_company_name():
    def handler(_request):
        return httpx.Response(200, json={"results": [
            {"title": "About Acme Robotics", "url": "https://acmerobotics.com/about", "content": "Acme Robotics builds industrial inspection systems."},
            {"title": "Acme Robotics profile", "url": "https://unrelated.example/acme", "content": "A third-party profile of Acme Robotics."},
        ]})

    result = client_for(handler).research_company("Acme Robotics LLC")
    assert result.status == "succeeded"
    assert result.identity_status == "verified"
    assert result.snapshot == "Acme Robotics builds industrial inspection systems."
    assert result.sources[0].identity_status == "verified"
    assert result.sources[1].identity_status == "unverified"


def test_similar_domain_without_explicit_name_evidence_is_unverified():
    def handler(_request):
        return httpx.Response(200, json={"results": [
            {"title": "Industrial automation solutions", "url": "https://acmerobotics.com/solutions", "content": "Robotics solutions for manufacturers."},
        ]})

    result = client_for(handler).research_company("Acme Robotics")
    assert result.identity_status == "unverified"
    assert result.snapshot is None
    assert "explicitly identify" in result.sources[0].identity_reason


def test_empty_search_results_have_unknown_identity():
    result = client_for(lambda _request: httpx.Response(200, json={"results": []})).research_company("Unknown Company")
    assert result.status == "no_results"
    assert result.identity_status == "unverified"
    assert result.snapshot is None
