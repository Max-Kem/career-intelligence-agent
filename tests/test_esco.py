from __future__ import annotations

import httpx
import pytest

from career_agent.esco import ESCOClient, ESCOClientError


def client_for(payload_or_handler, *, max_retries=0, sleeps=None):
    if callable(payload_or_handler):
        handler = payload_or_handler
    else:
        handler = lambda request: httpx.Response(200, json=payload_or_handler)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    sleeper = sleeps.append if sleeps is not None else lambda _delay: None
    return ESCOClient(client=http_client, max_retries=max_retries, sleep=sleeper)


def concept(uri: str, label: str, alt_labels: list[str] | None = None) -> dict:
    return {
        "uri": uri,
        "title": label,
        "className": "Skill",
        "preferredLabel": {"en": label},
        "altLabels": {"en": alt_labels or []},
    }


def test_skill_lookup_uses_official_search_endpoint_and_returns_traceable_exact_result():
    payload = {"_embedded": {"results": [concept("https://data.europa.eu/esco/skill/1", "Python")]}}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    client = client_for(handler)
    result = client.lookup_skill("Python")
    assert result.status == "exact"
    assert result.concepts[0].label == "Python"
    assert result.concepts[0].uri == "https://data.europa.eu/esco/skill/1"
    assert result.concepts[0].concept_type == "Skill"
    assert result.concepts[0].language == "en"
    assert "ec.europa.eu/esco/api/search" in result.source_reference
    request = requests[0]
    assert request.url.path == "/esco/api/search"
    assert dict(request.url.params) == {
        "text": "Python", "language": "en", "type": "skill", "full": "true", "limit": "8",
    }


def test_alternative_label_is_reported_separately_from_preferred_label():
    payload = {"_embedded": {"results": [concept("urn:skill:1", "Software programming", ["Python programming"])]}}
    result = client_for(payload).lookup_skill("Python programming")
    assert result.status == "alternative"
    assert result.concepts[0].label == "Software programming"
    assert result.concepts[0].matched_label == "Python programming"


def test_ambiguous_and_unmatched_search_terms_are_distinguished():
    ambiguous = {"_embedded": {"results": [
        concept("urn:skill:1", "Analysis"), concept("urn:skill:2", "Analysis"),
    ]}}
    assert client_for(ambiguous).lookup_skill("Analysis").status == "ambiguous"
    assert client_for({"_embedded": {"results": []}}).lookup_skill("No such skill").status == "no_match"
    punctuation_distinction = {"_embedded": {"results": [concept("urn:skill:c-plus", "C++")]}}
    punctuation_result = client_for(punctuation_distinction).lookup_skill("C")
    assert punctuation_result.status == "non_exact"
    assert punctuation_result.concepts[0].label == "C++"


def test_python_query_preserves_non_exact_esco_search_results():
    payload = {"_embedded": {"results": [
        {
            "title": "Python (computer programming)",
            "className": "Skill",
            "uri": "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d",
            "preferredLabel": {
                "en": "Python (computer programming)",
                "de": "Python (Computerprogrammierung)",
                "fr": "Python (programmation informatique)",
            },
        },
        {
            "title": "Python (computer programming)",
            "className": "Skill",
            "uri": "http://data.europa.eu/esco/skill/another-python-result",
            "preferredLabel": {"en": "Python (computer programming)"},
        },
    ]}}

    result = client_for(payload).lookup_skill("Python")

    assert result.status == "non_exact"
    assert len(result.concepts) == 2
    assert result.concepts[0].label == "Python (computer programming)"
    assert result.concepts[0].uri == (
        "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"
    )
    assert result.concepts[0].concept_type == "Skill"
    assert result.concepts[0].language == "en"
    assert result.concepts[0].match_type == "non_exact"


def test_occupation_lookup_uses_occupation_type():
    request_params = {}

    def handler(request):
        request_params.update(dict(request.url.params))
        return httpx.Response(200, json={"_embedded": {"results": [
            {"uri": "urn:occupation:1", "title": "Data analyst", "className": "Occupation",
             "preferredLabel": {"en": "Data analyst"}},
        ]}})

    result = client_for(handler).lookup_occupation("Data analyst")
    assert result.status == "exact"
    assert request_params["type"] == "occupation"
    assert request_params["full"] == "true"
    assert request_params["limit"] == "8"


def test_hal_result_parsing_uses_title_and_self_link():
    payload = {"_embedded": {"results": [{
        "title": "Python",
        "className": "Skill",
        "_links": {"self": {"href": "https://data.europa.eu/esco/skill/hal-python"}},
    }]}}
    result = client_for(payload).lookup_skill("Python")
    assert result.status == "exact"
    assert result.concepts[0].label == "Python"
    assert result.concepts[0].uri == "https://data.europa.eu/esco/skill/hal-python"


def test_bad_request_includes_validation_body_and_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"message": "Invalid type value: Skill"})

    client = client_for(handler, max_retries=2)
    with pytest.raises(ESCOClientError, match="HTTP 400.*Invalid type value: Skill"):
        client.lookup_skill("Python")
    assert len(calls) == 1


def test_transient_api_failure_is_retried_once_then_reported():
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": "temporary"})

    client = client_for(handler, max_retries=1, sleeps=sleeps)
    with pytest.raises(ESCOClientError, match="HTTP 503"):
        client.lookup_skill("Python")
    assert len(calls) == 2
    assert sleeps == [0.2]
