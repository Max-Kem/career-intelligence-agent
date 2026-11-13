from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from career_agent.candidate import CandidateProfile
from career_agent import intelligence_synthesis
from career_agent.intelligence_synthesis import prepare_synthesis_input, run_grounded_synthesis


def _candidate():
    return CandidateProfile.model_validate({
        "candidate_id": "test", "name": "Private Name", "summary": "AI transformation professional",
        "skills": ["Project Management"], "desired_job_titles": ["AI Consultant"],
        "professional_experience": [{"title": "Consultant", "responsibilities": ["Led delivery."]}],
    })


def _evidence(**updates):
    return {
        "chunk_id": "chunk-1", "document_title": "Germany Skills Forecast", "publisher": "Cedefop",
        "publication_year": 2025, "page_number": 9, "geographic_scope": "Germany_focused",
        "source_url": "https://example.test/report.pdf", "source_passage": "Complete source passage about digital skills and workforce development.",
        "source_passage_complete": True, "source_passage_truncated": False,
        "excerpt": "Short excerpt only.", **updates,
    }


def _response(payload, *, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=json.dumps(payload) if payload is not None else None, refusal=refusal),
    )])


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _valid_payload(source_id="S1"):
    return {
        "market_intelligence": [{
            "focus": "market_finding",
            "text": "The retrieved Germany-focused source discusses digital skills development and workforce needs.",
            "references": [{"kind": "market_source", "reference_id": source_id}],
        }],
        "personalized_career_intelligence": [{
            "focus": "career_implication",
            "text": "The candidate lists project management; this source offers context, not confirmation of job suitability.",
            "references": [
                {"kind": "candidate_profile", "reference_id": "C2"},
                {"kind": "market_source", "reference_id": source_id},
            ],
        }],
    }


def test_success_uses_complete_passage_and_returns_only_valid_traceable_source_references():
    market = {"evidence": [_evidence()]}
    context, sources = prepare_synthesis_input(market, _candidate(), [], [])
    assert sources[0]["text"] == market["evidence"][0]["source_passage"]
    assert "Private Name" not in str(context)

    completions = FakeCompletions(_response(_valid_payload()))
    result = run_grounded_synthesis(
        market, _candidate(), [], api_key="mock-key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    assert result["status"] == "succeeded"
    assert result["generation_mode"] == "llm_generated"
    assert result["request_count"] == 1 and len(completions.calls) == 1
    assert result["market_intelligence"][0]["sources"][0]["source_url"] == "https://example.test/report.pdf"
    assert result["market_intelligence"][0]["evidence_ids"] == ["chunk-1"]
    assert result["market_intelligence"][0]["model_source_ids"] == ["S1"]
    assert result["market_intelligence"][0]["citations"][0]["kind"] == "market_source"
    assert result["personalized_career_intelligence"][0]["citations"][0]["kind"] == "candidate_profile"
    assert "text" not in result["market_intelligence"][0]["sources"][0]
    assert completions.calls[0]["max_completion_tokens"] == intelligence_synthesis.MAX_OUTPUT_TOKENS
    assert completions.calls[0]["max_completion_tokens"] <= 2000
    assert completions.calls[0]["timeout"] <= 25
    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert "untrusted data" in call["messages"][0]["content"]
    assert "Complete source passage" in call["messages"][1]["content"]
    request_payload = json.loads(call["messages"][1]["content"])
    supplied_ids = request_payload["allowed_references"]["market_source"]
    assert supplied_ids == ["S1"]
    assert request_payload["retrieved_sources"][0]["source_id"] == "S1"
    assert "evidence_id" not in request_payload["retrieved_sources"][0]
    assert request_payload["allowed_references"]["candidate_profile"] == ["C1", "C2", "C3", "C4"]
    assert "market_source IDs" in call["messages"][0]["content"]
    assert "job_posting ID" in call["messages"][0]["content"]
    assert result["market_intelligence"][0]["model_source_ids"] == supplied_ids
    assert result["market_intelligence"][0]["evidence_ids"] == ["chunk-1"]


def test_saved_report_source_map_is_deterministic_and_restores_original_citations():
    from career_agent.candidate import CandidateProfile

    report_path = Path(__file__).resolve().parents[1] / "reports/streamlit/career_report_custom_20261112T092129298274Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["recommendations"] == []
    assert len(report["potential_matches_requiring_review"]) == 8
    market = report["market_intelligence"]
    candidate = CandidateProfile.model_validate(report["candidate"])
    context, sources = prepare_synthesis_input(
        market, candidate, [], report["potential_matches_requiring_review"],
    )
    assert [source["source_id"] for source in sources] == [f"S{i}" for i in range(1, 7)]
    originals = [source["evidence_id"] for source in sources]
    assert originals == [item["chunk_id"] for item in market["evidence"]]
    assert len(set(originals)) == 6
    # Same publication is represented by different per-request IDs and chunk IDs.
    same_publication = [source for source in sources if source["source_url"] == sources[3]["source_url"]]
    assert len(same_publication) >= 2
    assert len({source["source_id"] for source in same_publication}) == len(same_publication)

    def paragraph(focus, text, *references):
        return {"focus": focus, "text": text, "references": list(references)}

    def market_ref(ref_id):
        return {"kind": "market_source", "reference_id": ref_id}

    def candidate_ref(ref_id):
        return {"kind": "candidate_profile", "reference_id": ref_id}

    def job_ref(ref_id):
        return {"kind": "job_posting", "reference_id": ref_id}

    payload = {
        "market_intelligence": [
            paragraph("market_finding", "The first retrieved global source discusses occupational exposure to generative AI.", market_ref("S4")),
            paragraph("market_finding", "A separate passage from the same publication provides additional context on the topic.", market_ref("S6")),
        ],
        "personalized_career_intelligence": [
                paragraph("career_implication", "The candidate documents project management; evidence offers context and does not prove qualifications.", candidate_ref("C2"), market_ref("S4")),
            paragraph("job_requirement", "The reviewed position specifies a requirement that must be confirmed from the original posting.", job_ref("J1")),
        ],
    }
    completions = FakeCompletions(_response(payload))
    result = run_grounded_synthesis(
        market, candidate, [], report["potential_matches_requiring_review"],
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    assert result["status"] == "succeeded"
    request = json.loads(completions.calls[0]["messages"][1]["content"])
    visible_ids = request["allowed_references"]["market_source"]
    assert visible_ids == [f"S{i}" for i in range(1, 7)]
    assert [entry["source_id"] for entry in request["retrieved_sources"]] == visible_ids
    assert all("chunk_id" not in entry and "evidence_id" not in entry for entry in request["retrieved_sources"])
    assert all(source_id not in json.dumps(request) for source_id in originals)
    cited_sources = result["market_intelligence"][1]["sources"]
    assert cited_sources[0]["source_id"] == "S6"
    assert cited_sources[0]["evidence_id"] == originals[5]
    assert cited_sources[0]["source_url"] == sources[5]["source_url"]
    assert cited_sources[0]["document_title"] == sources[5]["document_title"]
    assert cited_sources[0]["publisher"] == sources[5]["publisher"]
    assert cited_sources[0]["publication_year"] == sources[5]["publication_year"]
    assert cited_sources[0]["page_number"] == sources[5]["page_number"]
    # Multiple claims can cite the same allowed source and still resolve to the original chunk.
    assert result["personalized_career_intelligence"][0]["sources"][0]["source_id"] == "S4"
    assert result["personalized_career_intelligence"][0]["sources"][0]["evidence_id"] == originals[3]
    job_citation = result["personalized_career_intelligence"][1]["citations"][0]
    job_detail = request["candidate_and_job_context"]["job_reference_details"][0]
    assert job_citation["kind"] == "job_posting"
    assert job_citation["source_url"] == job_detail["source_url"]
    assert job_citation["document_title"] == job_detail["title"]


def test_job_requirement_cannot_use_market_citation_and_invented_refs_fallback():
    market = {"evidence": [_evidence()]}
    payload = _valid_payload()
    payload["personalized_career_intelligence"] = [{
        "focus": "job_requirement",
        "text": "A particular job posting names a specialist qualification that needs verification.",
        "references": [{"kind": "market_source", "reference_id": "S1"}],
    }]
    result = run_grounded_synthesis(
        market, _candidate(), [], client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_response(payload)))),
    )
    assert result["status"] == "failed"
    assert result["failure_stage"] == "citation_validation"
    assert result["generation_mode"] == "deterministic_fallback"


def test_guardrails_cover_demand_claims_relevance_and_unknown_qualifications():
    market = {"evidence": [_evidence()]}
    completions = FakeCompletions(_response(_valid_payload()))
    run_grounded_synthesis(
        market, _candidate(), [], client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    prompt = completions.calls[0]["messages"][0]["content"].casefold()
    assert "do not infer ai-consulting hiring demand" in prompt
    assert "market growth" in prompt and "cannot be established" in prompt
    assert "candidate's stated target roles and interests" in prompt
    assert "unrelated specialist requirements" in prompt
    assert "never imply an undocumented qualification is missing" in prompt
    assert "not_documented means the profile is silent" in prompt
    assert "partially_documented means related evidence exists" in prompt
    assert "does_not_meet_requirement may be stated only when explicit profile evidence" in prompt
    assert "prefer documenting existing experience" in prompt
    assert "normal spacing" in prompt


def test_undocumented_years_and_education_are_cited_as_unknown_not_absent():
    candidate = _candidate()
    assert candidate.years_experience is None
    assert candidate.education == []
    payload = {
        "market_intelligence": [{
            "focus": "market_finding",
            "text": "The retrieved source discusses workforce development in the context it documents.",
            "references": [{"kind": "market_source", "reference_id": "S1"}],
        }],
        "personalized_career_intelligence": [{
            "focus": "candidate_evidence",
            "text": "The supplied profile does not document years of experience or education; this does not establish that either is absent.",
            "references": [
                {"kind": "candidate_profile", "reference_id": "C3"},
                {"kind": "candidate_profile", "reference_id": "C4"},
            ],
        }],
    }
    result = run_grounded_synthesis(
        {"evidence": [_evidence()]}, candidate, [],
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_response(payload)))),
    )
    assert result["status"] == "succeeded"
    citations = result["personalized_career_intelligence"][0]["citations"]
    assert [citation["label"] for citation in citations] == [
        "documented experience and years of experience", "documented education",
    ]


def test_synthesis_can_state_qualification_is_not_documented_without_claiming_absence():
    payload = _valid_payload()
    payload["personalized_career_intelligence"][0]["text"] = (
        "The profile documents transferable skills, while the role's years requirement is not documented "
        "in the supplied profile and should be verified before drawing a conclusion."
    )
    result = run_grounded_synthesis(
        {"evidence": [_evidence()]}, _candidate(), [],
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_response(payload)))),
    )
    assert result["status"] == "succeeded"


def test_obvious_missing_space_triggers_deterministic_fallback():
    payload = _valid_payload()
    payload["market_intelligence"][0]["text"] = "The report states this.Germany has a separate context."
    result = run_grounded_synthesis(
        {"evidence": [_evidence()]}, _candidate(), [],
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_response(payload)))),
    )
    assert result["status"] == "failed"
    assert result["failure_stage"] == "format_validation"
    assert result["generation_mode"] == "deterministic_fallback"


def test_no_evidence_or_incomplete_excerpt_never_calls_openai():
    called = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: called.append(kw))))
    for evidence in (
        {"chunk_id": "excerpt-only", "excerpt": "This short snippet is not the complete passage.", "source_url": "https://example.test"},
        _evidence(chunk_id="Not provided"),
        _evidence(source_passage_truncated=True, source_passage_complete=False),
        _evidence(source_passage="x" * 4000),
    ):
        result = run_grounded_synthesis({"evidence": [evidence]}, _candidate(), [], api_key="mock", client=client)
        assert result["status"] == "unavailable"
        assert result["request_count"] == 0
    assert called == []


def test_unknown_citation_and_invalid_structured_output_fall_back_safely():
    market = {"evidence": [_evidence()]}
    for response in (_response(_valid_payload("S99")), _response({"market_intelligence": []})):
        result = run_grounded_synthesis(
            market, _candidate(), [], api_key="mock",
            client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(response))),
        )
        assert result["status"] == "failed"
        assert result["generation_mode"] == "deterministic_fallback"
        assert result["market_intelligence"] == []
    assert run_grounded_synthesis(
        market, _candidate(), [], api_key="mock",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_response(_valid_payload("S99"))))),
    )["failure_stage"] == "citation_validation"
    assert run_grounded_synthesis(
        market, _candidate(), [], api_key="mock",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{bad json", refusal=None))]
        )))),
    )["failure_stage"] == "invalid_json"


def test_output_limit_and_refusal_have_distinct_nonfatal_diagnostics():
    market = {"evidence": [_evidence()]}
    for response, expected in (
        (_response(_valid_payload(), finish_reason="length"), "output_limit"),
        (_response(None, refusal="refused"), "refusal"),
    ):
        result = run_grounded_synthesis(
            market, _candidate(), [], api_key="mock",
            client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(response))),
        )
        assert result["status"] == "failed"
        assert result["failure_stage"] == expected
        assert result["generation_mode"] == "deterministic_fallback"


def test_strict_schema_has_required_properties_and_supported_reference_shape():
    wrapped = intelligence_synthesis._json_schema()
    schema = wrapped["schema"]
    assert wrapped["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])
    for section in schema["properties"].values():
        assert section["items"]["$ref"] in {"#/$defs/GroundedParagraph"}
    serialized = json.dumps(schema)
    assert "minLength" not in serialized
    assert "maxLength" not in serialized


def test_string_length_limits_remain_enforced_locally_after_api_schema_omits_them():
    payload = _valid_payload()
    payload["market_intelligence"][0]["text"] = "Too short"
    result = run_grounded_synthesis(
        {"evidence": [_evidence()]}, _candidate(), [], api_key="mock",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_response(payload)))),
    )
    assert result["status"] == "failed"
    assert result["failure_stage"] == "schema_validation"


def test_openai_failure_is_nonfatal_and_does_not_expose_exception_details():
    result = run_grounded_synthesis(
        {"evidence": [_evidence()]}, _candidate(), [], api_key="mock",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(error=RuntimeError("secret body")))),
    )
    assert result["status"] == "failed"
    assert "RuntimeError" in result["error"]
    assert "secret body" not in result["error"]


def test_api_error_reports_safe_http_diagnostics_without_raw_error_text():
    class BadRequestError(RuntimeError):
        status_code = 400
        code = "invalid_json_schema"

    result = run_grounded_synthesis(
        {"evidence": [_evidence()]}, _candidate(), [], api_key="mock",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(
            error=BadRequestError("private request details"),
        ))),
    )
    assert result["failure_stage"] == "api_request"
    assert "HTTP 400" in result["error"]
    assert "invalid_json_schema" in result["error"]
    assert "private request details" not in result["error"]


def test_missing_api_key_returns_deterministic_fallback_without_client_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_grounded_synthesis({"evidence": [_evidence()]}, _candidate(), [])
    assert result["status"] == "unavailable"
    assert result["generation_mode"] == "deterministic_fallback"
    assert result["request_count"] == 0


def test_client_configuration_bounds_timeout_and_disables_retries(monkeypatch):
    seen = {}
    completions = FakeCompletions(_response(_valid_payload()))
    def factory(key, **kwargs):
        seen.update(key=key, **kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(intelligence_synthesis, "create_openai_client", factory)
    result = run_grounded_synthesis({"evidence": [_evidence()]}, _candidate(), [], api_key="mock-key")
    assert result["status"] == "succeeded"
    assert seen == {"key": "mock-key", "timeout": 25, "max_retries": 0}
