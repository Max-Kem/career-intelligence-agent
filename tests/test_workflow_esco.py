from __future__ import annotations

import json
from pathlib import Path

from career_agent.candidate import CandidateProfile
from career_agent.company_research import CompanyResearchResult, CompanySource
from career_agent.discovery import DiscoveryResult
from career_agent.esco import ESCOConcept, ESCOQueryResult
from career_agent.integration import CandidateDiscoveryRun, derive_search_queries
from career_agent.models import Job
from career_agent.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]


def profile() -> CandidateProfile:
    return CandidateProfile.model_validate_json(
        (ROOT / "fixtures" / "candidate_profile_a.json").read_text(encoding="utf-8")
    )


def jobs() -> list[Job]:
    raw = json.loads((ROOT / "fixtures" / "matching_jobs.json").read_text(encoding="utf-8"))
    return [Job.model_validate(item) for item in raw]


def runner(candidate, **kwargs):
    job_list = jobs()
    counts = {source: sum(job.source == source for job in job_list) for source in {job.source for job in job_list}}
    return CandidateDiscoveryRun(candidate, derive_search_queries(candidate), DiscoveryResult(
        jobs=job_list, provider_counts=counts, deduplicated_provider_counts=counts,
        final_provider_counts=counts,
    ))


class SuccessfulESCOClient:
    def __init__(self):
        self.calls = []
        self.closed = False

    def lookup(self, term, *, concept_type, language):
        self.calls.append((term, concept_type, language))
        concept = ESCOConcept(
            label=term, uri=f"https://data.europa.eu/esco/{concept_type.lower()}/sample",
            concept_type=concept_type, language=language, matched_label=term,
            match_type="exact", source_reference=f"https://data.europa.eu/esco/{concept_type.lower()}/sample",
        )
        return ESCOQueryResult(
            term=term, concept_type=concept_type, language=language,
            status="exact", concepts=[concept], source_reference="https://ec.europa.eu/esco/api/search",
        )

    def close(self):
        self.closed = True


def test_langgraph_esco_node_enriches_only_after_selection_and_preserves_match_scores(tmp_path):
    esco = SuccessfulESCOClient()
    result = build_workflow(discovery_runner=runner, esco_client_factory=lambda: esco).invoke({
        "profile_data": profile().model_dump(mode="json"), "profile_name": "a",
        "output_dir": str(tmp_path), "esco_enabled": True,
    })
    nodes = result["completed_nodes"]
    assert nodes.index("select_recommendations") < nodes.index("enrich_esco") < nodes.index("generate_report")
    assert esco.closed
    assert result["esco_enrichment"]["status"] == "succeeded"
    assert result["esco_enrichment"]["lookup_count"] == len(esco.calls)
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["esco_enrichment"]["source_reference"] == "https://ec.europa.eu/esco/api/search"
    assert any(
        concept["uri"].startswith("https://data.europa.eu/esco/")
        for lookup in report["esco_enrichment"]["lookups"]
        for concept in lookup["esco_result"]["concepts"]
    )
    assert [item["score"] for item in report["recommendations"]] == [
        item.score for item in result["matching"].top_matches
    ]
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "ESCO skill and occupation enrichment" in markdown
    assert "not by itself a verified skill equivalence" in markdown
    assert "https://data.europa.eu/esco/" in markdown


def test_markdown_hides_non_exact_concepts_while_json_preserves_details(tmp_path):
    class MixedLabelClient:
        def __init__(self):
            self.calls = 0

        def lookup(self, term, *, concept_type, language):
            self.calls += 1
            source = "https://ec.europa.eu/esco/api/search"
            if self.calls == 1:
                status = "non_exact"
                concepts = [
                    ESCOConcept(
                        label="unrelated skill alpha", uri="https://data.europa.eu/esco/skill/alpha",
                        concept_type="Skill", language=language, match_type="non_exact",
                        source_reference="https://data.europa.eu/esco/skill/alpha",
                    ),
                    ESCOConcept(
                        label="unrelated skill beta", uri="https://data.europa.eu/esco/skill/beta",
                        concept_type="Skill", language=language, match_type="non_exact",
                        source_reference="https://data.europa.eu/esco/skill/beta",
                    ),
                ]
            elif self.calls == 2:
                status = "exact"
                concepts = [ESCOConcept(
                    label=term, uri="https://data.europa.eu/esco/skill/exact",
                    concept_type="Skill", language=language, matched_label=term,
                    match_type="exact", source_reference="https://data.europa.eu/esco/skill/exact",
                )]
            elif self.calls == 3:
                status = "alternative"
                concepts = [ESCOConcept(
                    label="engage with stakeholders", uri="https://data.europa.eu/esco/skill/alternative",
                    concept_type="Skill", language=language, matched_label=term,
                    match_type="alternative", source_reference="https://data.europa.eu/esco/skill/alternative",
                )]
            else:
                status, concepts = "no_match", []
            return ESCOQueryResult(
                term=term, concept_type=concept_type, language=language,
                status=status, concepts=concepts, source_reference=source,
            )

        def close(self):
            pass

    result = build_workflow(discovery_runner=runner, esco_client_factory=MixedLabelClient).invoke({
        "profile_data": profile().model_dump(mode="json"), "profile_name": "a",
        "output_dir": str(tmp_path), "esco_enabled": True,
    })
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    lookups = report["esco_enrichment"]["lookups"]

    assert "Enrichment status: succeeded" in markdown
    assert f"Lookups: {report['esco_enrichment']['lookup_count']}" in markdown
    assert "Verified label match (exact)" in markdown
    assert "Verified label match (alternative)" in markdown
    assert "ESCO returned candidate concepts, but no verified label match was found." in markdown
    assert "unrelated skill alpha" not in markdown
    assert "unrelated skill beta" not in markdown
    assert "https://data.europa.eu/esco/skill/alpha" not in markdown
    assert "https://data.europa.eu/esco/skill/beta" not in markdown
    assert "A returned search result is not by itself a verified skill equivalence." in markdown
    first_concepts = lookups[0]["esco_result"]["concepts"]
    assert [concept["label"] for concept in first_concepts] == ["unrelated skill alpha", "unrelated skill beta"]
    assert [concept["uri"] for concept in first_concepts] == [
        "https://data.europa.eu/esco/skill/alpha", "https://data.europa.eu/esco/skill/beta",
    ]


def test_report_continues_with_unavailable_esco_client(tmp_path):
    class FailedClient:
        def lookup(self, term, *, concept_type, language):
            raise TimeoutError("mocked timeout")

        def close(self):
            pass

    result = build_workflow(discovery_runner=runner, esco_client_factory=FailedClient).invoke({
        "profile_data": profile().model_dump(mode="json"), "profile_name": "a",
        "output_dir": str(tmp_path), "esco_enabled": True,
    })
    assert result["esco_enrichment"]["status"] == "unavailable"
    assert result["recommendations"]
    assert result["workflow_status"] == "completed_with_warnings"
    assert result["nonfatal_errors"]
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["esco_enrichment"]["failed_lookup_count"] > 0
    assert report["recommendations"][0]["score"] == result["matching"].top_matches[0].score
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "Enrichment status: unavailable" in markdown
    assert "mocked timeout" in markdown


def test_offline_graph_skips_esco_client_and_keeps_report_available(tmp_path):
    def must_not_create_client():
        raise AssertionError("offline workflow must not access ESCO")

    result = build_workflow(esco_client_factory=must_not_create_client).invoke({
        "profile_name": "a", "offline": True, "output_dir": str(tmp_path),
    })
    assert result["esco_enrichment"]["status"] == "skipped_offline"
    assert "enrich_esco" in result["completed_nodes"]
    assert Path(result["report_paths"]["markdown"]).exists()


def test_offline_with_explicit_live_esco_enriches_fixture_recommendations(tmp_path):
    esco = SuccessfulESCOClient()
    result = build_workflow(esco_client_factory=lambda: esco).invoke({
        "profile_name": "a", "offline": True, "allow_live_esco": True,
        "output_dir": str(tmp_path),
    })

    assert result["recommendations"]
    assert result["esco_enrichment"]["status"] == "succeeded"
    assert 0 < len(esco.calls) <= 32
    assert result["completed_nodes"].index("enrich_esco") < result["completed_nodes"].index("generate_report")
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["search_scope"]["job_data_source"] == "synthetic_fixtures"
    assert report["search_scope"]["allow_live_esco"] is True
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "synthetic fixture jobs + live ESCO enrichment" in markdown
    assert "synthetic fixture jobs, not real vacancies" in markdown


def test_offline_live_esco_failure_still_generates_reports(tmp_path):
    class FailedClient:
        def lookup(self, term, *, concept_type, language):
            raise TimeoutError("mocked ESCO outage")

        def close(self):
            pass

    result = build_workflow(esco_client_factory=FailedClient).invoke({
        "profile_name": "a", "offline": True, "allow_live_esco": True,
        "output_dir": str(tmp_path),
    })

    assert result["esco_enrichment"]["status"] == "unavailable"
    assert result["report_paths"]["markdown"]
    assert result["report_paths"]["json"]
    assert result["workflow_status"] == "completed_with_warnings"
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["esco_enrichment"]["status"] == "unavailable"
    assert report["recommendations"]


def test_company_research_runs_after_recommendations_and_cites_sources(tmp_path):
    class MockCompanyResearcher:
        def __init__(self):
            self.calls = []

        def research_company(self, company):
            self.calls.append(company)
            source = CompanySource(
                title="Official company overview", url="https://example.com/about",
                snippet="A source-provided overview snippet.", source_type="possible_company_website",
            )
            return CompanyResearchResult(
                company=company, status="succeeded", snapshot=source.snippet,
                sources=[source], uncertainties=["Snippet not independently verified."],
            )

        def close(self):
            pass

    researcher = MockCompanyResearcher()

    def duplicate_company_runner(candidate, **kwargs):
        run = runner(candidate, **kwargs)
        duplicate_jobs = [job.model_copy(update={"company": "Same Company"}) for job in run.discovery.jobs]
        counts = {"synthetic": len(duplicate_jobs)}
        return CandidateDiscoveryRun(candidate, run.queries, DiscoveryResult(
            jobs=duplicate_jobs, provider_counts=counts, deduplicated_provider_counts=counts,
            final_provider_counts=counts,
        ))

    result = build_workflow(
        discovery_runner=duplicate_company_runner,
        company_researcher_factory=lambda: researcher,
    ).invoke({
        "profile_data": profile().model_dump(mode="json"), "profile_name": "a",
        "output_dir": str(tmp_path), "esco_enabled": False, "company_research_enabled": True,
    })

    nodes = result["completed_nodes"]
    assert nodes.index("select_recommendations") < nodes.index("research_companies") < nodes.index("generate_report")
    assert result["company_research"]["status"] == "succeeded"
    assert result["company_research"]["results"][0]["identity_status"] == "unverified"
    assert researcher.calls == ["Same Company"]
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["company_research"]["results"][0]["sources"][0]["url"] == "https://example.com/about"
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "## Company research" in markdown
    assert "[Official company overview](https://example.com/about)" in markdown
    assert "Reliable company information could not be confirmed" in markdown
    assert "A source-provided overview snippet." not in markdown


def test_company_research_skips_when_no_recommendations(tmp_path):
    def no_recommendation_runner(candidate, **kwargs):
        weak = Job(job_id="weak", title="Unrelated job", company="No Research Inc", source="mock")
        return CandidateDiscoveryRun(candidate, [], DiscoveryResult(jobs=[weak]))

    def must_not_create_researcher():
        raise AssertionError("company research must not run without recommendations")

    result = build_workflow(
        discovery_runner=no_recommendation_runner,
        company_researcher_factory=must_not_create_researcher,
    ).invoke({
        "profile_data": profile().model_dump(mode="json"), "profile_name": "a",
        "output_dir": str(tmp_path), "esco_enabled": False, "company_research_enabled": True,
    })
    assert not result["recommendations"]
    assert result["company_research"]["status"] == "skipped_no_recommendations"
    assert Path(result["report_paths"]["markdown"]).exists()


def test_offline_graph_skips_company_research_even_when_enabled(tmp_path):
    def must_not_create_researcher():
        raise AssertionError("offline mode must not access Tavily")

    result = build_workflow(company_researcher_factory=must_not_create_researcher).invoke({
        "profile_name": "a", "offline": True, "company_research_enabled": True,
        "output_dir": str(tmp_path), "esco_enabled": False,
    })
    assert result["company_research"]["status"] == "skipped_offline"
    assert Path(result["report_paths"]["markdown"]).exists()


def test_explicit_live_company_research_uses_fixture_jobs_and_keeps_esco_offline(tmp_path):
    class MockCompanyResearcher:
        def __init__(self):
            self.calls = []

        def research_company(self, company):
            self.calls.append(company)
            source = CompanySource(
                title="Search result", url="https://example.test/company",
                snippet="Tavily returned this source excerpt.",
            )
            return CompanyResearchResult(
                company=company, status="succeeded", snapshot=source.snippet,
                sources=[source], uncertainties=["Search result is not independently verified."],
            )

        def close(self):
            pass

    researcher = MockCompanyResearcher()

    def esco_must_remain_offline():
        raise AssertionError("ESCO should stay offline without --allow-live-esco")

    result = build_workflow(
        esco_client_factory=esco_must_remain_offline,
        company_researcher_factory=lambda: researcher,
    ).invoke({
        "profile_name": "a", "offline": True, "company_research_enabled": True,
        "allow_live_company_research": True, "output_dir": str(tmp_path),
    })

    assert result["company_research"]["status"] == "succeeded"
    assert result["esco_enrichment"]["status"] == "skipped_offline"
    assert 1 <= len(researcher.calls) <= 5
    assert len(researcher.calls) == len(set(company.casefold() for company in researcher.calls))
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["search_scope"]["job_data_source"] == "synthetic_fixtures"
    assert report["search_scope"]["allow_live_company_research"] is True
    assert report["company_research"]["identity_warning"]
    assert all(item["identity_uncertain"] for item in report["company_research"]["results"])
    assert all("may be fictional" in " ".join(item["uncertainties"]) for item in report["company_research"]["results"])
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "synthetic fixture jobs + live Tavily company research" in markdown
    assert "These are synthetic fixture jobs, not real vacancies." in markdown
    assert "Company research used live Tavily search results" in markdown
    assert "fixture company identities are uncertain" in markdown
    assert "Tavily returned this source excerpt." not in markdown
    assert "Reliable company information could not be confirmed" in markdown


def test_live_tavily_failure_with_fixture_jobs_still_reports_identity_uncertainty(tmp_path):
    class FailedCompanyResearcher:
        def research_company(self, company):
            return CompanyResearchResult(
                company=company, status="unavailable", error="Tavily search failed (HTTP 503).",
                uncertainties=["Company information could not be verified."],
            )

        def close(self):
            pass

    result = build_workflow(company_researcher_factory=FailedCompanyResearcher).invoke({
        "profile_name": "a", "offline": True, "company_research_enabled": True,
        "allow_live_company_research": True, "output_dir": str(tmp_path),
    })

    assert result["company_research"]["status"] == "unavailable"
    assert result["report_paths"]["json"] and result["report_paths"]["markdown"]
    report = json.loads(Path(result["report_paths"]["json"]).read_text(encoding="utf-8"))
    assert report["company_research"]["identity_warning"]
    assert report["recommendations"]
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "Research status: unavailable" in markdown
    assert "may be fictional" in markdown
