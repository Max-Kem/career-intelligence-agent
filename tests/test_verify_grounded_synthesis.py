from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_grounded_synthesis.py"
SPEC = importlib.util.spec_from_file_location("verify_grounded_synthesis", SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify)


def _saved_report() -> dict:
    return {
        "candidate": {
            "candidate_id": "synthetic-a", "name": "Synthetic Profile A",
            "summary": "AI transformation professional.",
            "skills": ["Project Management"],
            "desired_job_titles": ["AI Consultant"],
            "professional_experience": [],
        },
        "recommendations": [],
        "potential_matches_requiring_review": [{
            "job": {"title": "AI Consultant", "company": "Example", "description": "AI adoption role."},
            "score": 70, "reasons": ["Eligibility remains unresolved."],
            "requirements_to_verify": [],
        }],
        "market_intelligence": {"evidence": [{
            "chunk_id": "saved-chunk-1", "document_title": "Labour Market Report",
            "publisher": "Example Publisher", "publication_year": 2025,
            "page_number": 4, "geographic_scope": "Germany_focused",
            "source_url": "https://example.test/report.pdf",
            "source_passage": "Complete retrieved passage about digital skills and workforce development.",
            "source_passage_complete": True, "source_passage_truncated": False,
        }]},
    }


def _model_response(source_id="S1"):
    payload = {
        "market_intelligence": [{
            "focus": "market_finding",
            "text": "The retrieved report discusses digital skills and workforce development in its stated scope.",
            "references": [{"kind": "market_source", "reference_id": source_id}],
        }],
        "personalized_career_intelligence": [{
            "focus": "career_implication",
            "text": "The candidate documents project management; the evidence provides context, not proof of qualifications.",
            "references": [
                {"kind": "candidate_profile", "reference_id": "C2"},
                {"kind": "market_source", "reference_id": source_id},
            ],
        }],
    }
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=json.dumps(payload), refusal=None),
    )], usage=SimpleNamespace(prompt_tokens=300, completion_tokens=100, total_tokens=400))


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_live_flag_is_required_before_dotenv_or_any_request(monkeypatch):
    loaded = []
    monkeypatch.setattr(verify, "run_saved_report", lambda *args, **kwargs: loaded.append(True))
    with pytest.raises(SystemExit):
        verify.main(["--report", "unused.json"])
    assert loaded == []


def test_saved_report_uses_production_schema_and_citation_validation_once(tmp_path, monkeypatch):
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(_saved_report()), encoding="utf-8")
    completions = FakeCompletions(_model_response())
    factory_calls = []

    def factory(key, **kwargs):
        factory_calls.append((key, kwargs))
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    dotenv_calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "mock-secret")
    monkeypatch.delenv("CAREER_INTELLIGENCE_MODEL", raising=False)
    code, result = verify.run_saved_report(
        path,
        dotenv_loader=lambda **kwargs: dotenv_calls.append(kwargs),
        client_factory=factory,
    )

    assert code == 0
    assert result["status"] == "succeeded"
    assert result["model"] == verify.MODEL
    assert result["request_count"] == 1
    assert result["request_completed"] is True
    assert result["pydantic_validation"] == "passed"
    assert result["citation_validation"] == "passed"
    assert result["all_cited_ids_resolve"] is True
    assert result["token_usage"]["total_tokens"] == 400
    assert len(result["market_intelligence"]) == 1
    assert len(result["personalized_career_intelligence"]) == 1
    assert len(completions.calls) == 1
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert factory_calls == [("mock-secret", {"timeout": verify.REQUEST_TIMEOUT_SECONDS, "max_retries": 0})]
    assert dotenv_calls == [{"dotenv_path": verify.ROOT / ".env", "override": False}]


def test_missing_credentials_and_invalid_citation_never_pass_validation(tmp_path, monkeypatch):
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(_saved_report()), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    code, result = verify.run_saved_report(
        path,
        dotenv_loader=lambda **kwargs: None,
        client_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert code == 2
    assert result["request_count"] == 0
    assert calls == []

    monkeypatch.setenv("OPENAI_API_KEY", "mock-secret")
    bad_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(_model_response("S99"))))
    code, result = verify.run_saved_report(
        path,
        dotenv_loader=lambda **kwargs: None,
        client_factory=lambda *args, **kwargs: bad_client,
    )
    assert code == 1
    assert result["request_count"] == 1
    assert result["failure_stage"] == "citation_validation"
    assert result["pydantic_validation"] == "passed"
    assert result["citation_validation"] == "failed"
    assert result["all_cited_ids_resolve"] == "not applicable"


def test_one_request_guard_rejects_second_call():
    delegate = FakeCompletions(_model_response())
    guarded = verify.OneRequestCompletions(delegate)
    guarded.create()
    with pytest.raises(RuntimeError, match="request limit"):
        guarded.create()
    assert guarded.request_count == 1
    assert len(delegate.calls) == 1
