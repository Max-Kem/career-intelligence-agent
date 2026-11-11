from __future__ import annotations

import json
from pathlib import Path

from career_agent.candidate import CandidateProfile
from career_agent.discovery import DiscoveryResult
from career_agent.integration import CandidateDiscoveryRun
from career_agent.models import Job
from career_agent.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]


def candidate_a() -> CandidateProfile:
    return CandidateProfile.model_validate_json(
        (ROOT / "fixtures" / "candidate_profile_a.json").read_text(encoding="utf-8")
    )


def fixture_jobs() -> list[Job]:
    raw = json.loads((ROOT / "fixtures" / "matching_jobs.json").read_text(encoding="utf-8"))
    return [Job.model_validate(item) for item in raw]


def discovery_run(jobs: list[Job], *, errors: dict[str, str] | None = None) -> CandidateDiscoveryRun:
    provider_counts: dict[str, int] = {}
    for job in jobs:
        provider_counts[job.source] = provider_counts.get(job.source, 0) + 1
    result = DiscoveryResult(
        jobs=jobs, provider_counts=provider_counts,
        deduplicated_provider_counts=provider_counts.copy(),
        final_provider_counts=provider_counts.copy(), errors=errors or {},
    )
    return CandidateDiscoveryRun(candidate_a(), ["AI Consultant"], result)


def invoke(tmp_path: Path, runner, *, offline: bool = False) -> dict:
    return build_workflow(discovery_runner=runner).invoke({
        "profile_data": candidate_a().model_dump(mode="json"),
        "profile_name": "a", "output_dir": str(tmp_path),
        "offline": offline, "esco_enabled": False,
    })


def test_full_offline_graph_generates_traceable_reports_without_provider_calls(tmp_path, monkeypatch):
    import career_agent.workflow as workflow

    def no_external_calls(*args, **kwargs):
        raise AssertionError("offline mode must not call live discovery")

    monkeypatch.setattr(workflow, "discover_candidate_jobs", no_external_calls)
    result = build_workflow().invoke({
        "profile_name": "a", "offline": True, "output_dir": str(tmp_path),
    })
    assert result["workflow_status"] == "completed_with_warnings"
    assert "load_candidate" in result["completed_nodes"]
    assert "discover_jobs" in result["completed_nodes"]
    assert "match_jobs" in result["completed_nodes"]
    assert "select_recommendations" in result["completed_nodes"]
    assert "generate_report" in result["completed_nodes"]
    assert len(result["jobs"]) == 5
    assert result["report_paths"].keys() == {"markdown", "json"}
    report_path = Path(result["report_paths"]["json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["search_scope"]["offline"] is True
    assert report["discovered_jobs"] == 5
    assert report["evaluated_jobs"] == 5
    assert report["recommendations"]
    for recommendation in report["recommendations"]:
        assert recommendation["job"]["source_url"].startswith("https://")
        assert recommendation["score"] is not None
        assert recommendation["geographic_evidence"]
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "Why fewer than five?" in markdown
    assert "Source:" in markdown
    assert "Run completion" in markdown


def test_empty_discovery_takes_empty_route_and_reports_useful_result(tmp_path):
    result = invoke(tmp_path, lambda candidate, **kwargs: discovery_run([]))
    assert result["jobs"] == []
    assert "note_no_jobs" in result["completed_nodes"]
    assert "match_jobs" not in result["completed_nodes"]
    assert "No jobs were discovered" in " ".join(result["warnings"])
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["recommendations"] == []
    assert "No jobs were available" in Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")


def test_no_qualifying_recommendation_takes_diagnostic_route(tmp_path):
    weak = Job(job_id="weak", title="Unrelated Role", company="Other", source="mock")
    result = invoke(tmp_path, lambda candidate, **kwargs: discovery_run([weak]))
    assert "match_jobs" in result["completed_nodes"]
    assert "note_no_recommendations" in result["completed_nodes"]
    assert result["recommendations"] == []
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["evaluated_jobs"] == 1
    assert report["non_recommendations"][0]["job"]["job_id"] == "weak"
    assert report["non_recommendations"][0]["reasons"]


def test_partial_provider_failure_keeps_available_jobs_and_reports_error(tmp_path):
    jobs = fixture_jobs()[:1]
    runner = lambda candidate, **kwargs: discovery_run(jobs, errors={"himalayas": "temporary outage"})
    result = invoke(tmp_path, runner)
    assert "note_partial_failure" in result["completed_nodes"]
    assert "match_jobs" in result["completed_nodes"]
    assert result["jobs"] == jobs
    assert result["nonfatal_errors"] == ["himalayas: temporary outage"]
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["search_scope"]["provider_diagnostics"]["errors"]["himalayas"] == "temporary outage"
    assert report["workflow"]["nonfatal_errors"] == ["himalayas: temporary outage"]
    assert any("providers failed" in warning.lower() for warning in result["warnings"])


def test_profile_validation_failure_still_writes_a_report(tmp_path):
    result = build_workflow().invoke({
        "profile_name": "not-a-profile", "output_dir": str(tmp_path),
    })
    assert result["workflow_status"] == "failed"
    assert "discover_jobs" not in result["completed_nodes"]
    assert Path(result["report_paths"]["markdown"]).exists()
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["candidate"] is None
    assert report["workflow"]["errors"]
