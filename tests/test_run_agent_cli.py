from __future__ import annotations

from scripts import run_agent


class CapturingWorkflow:
    def __init__(self, captured):
        self.captured = captured

    def invoke(self, state):
        self.captured.update(state)
        return {
            "workflow_status": "completed", "search_queries": [], "jobs": [],
            "recommendations": [], "report_paths": {}, "warnings": [],
            "nonfatal_errors": [], "errors": [],
        }


def _fake_workflow(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_agent, "build_workflow", lambda: CapturingWorkflow(captured))
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    return captured


def test_live_cli_allows_explicit_rag_esco_and_company_research(monkeypatch, capsys):
    captured = _fake_workflow(monkeypatch)
    assert run_agent.main([
        "--profile", "a", "--queries", "3", "--pages", "1",
        "--allow-live-rag", "--allow-live-esco", "--allow-live-company-research",
        "--output-dir", "reports/test-live",
    ]) == 0
    assert captured["offline"] is False
    assert captured["allow_live_market_intelligence"] is True
    assert captured["market_intelligence_enabled"] is True
    assert captured["allow_live_esco"] is True
    assert captured["esco_enabled"] is True
    assert captured["allow_live_company_research"] is True
    assert captured["company_research_enabled"] is True


def test_live_rag_remains_opt_in_and_offline_defaults_are_network_free(monkeypatch, capsys):
    captured = _fake_workflow(monkeypatch)
    assert run_agent.main(["--profile", "a"]) == 0
    assert captured["offline"] is False
    assert captured["allow_live_market_intelligence"] is False
    assert captured["market_intelligence_enabled"] is False

    assert run_agent.main(["--profile", "b", "--offline"]) == 0
    assert captured["offline"] is True
    assert captured["allow_live_market_intelligence"] is False
    assert captured["market_intelligence_enabled"] is False


def test_offline_can_explicitly_enable_live_rag_without_enabling_esco_or_tavily(monkeypatch, capsys):
    captured = _fake_workflow(monkeypatch)
    assert run_agent.main(["--profile", "a", "--offline", "--allow-live-rag"]) == 0
    assert captured["offline"] is True
    assert captured["allow_live_market_intelligence"] is True
    assert captured["market_intelligence_enabled"] is True
    assert captured["allow_live_esco"] is False
    assert captured["allow_live_company_research"] is False
