from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .candidate import CandidateProfile
from .company_research import CompanyResearchResult, TavilyCompanyResearchClient, unique_recommended_companies
from .discovery import DiscoveryResult
from .esco import ESCOClient, ESCO_LANGUAGE, ESCO_SOURCE_URL, enrich_candidate_and_matches
from .filtering import FilterDiagnostics
from .integration import CandidateDiscoveryRun, discover_candidate_jobs, derive_search_queries
from .matching import MINIMUM_MATCH_SCORE, JobMatch, MatchingResult, match_candidate_to_jobs
from .models import Job


class WorkflowState(TypedDict, total=False):
    profile_name: str
    profile_data: dict
    candidate: CandidateProfile | None
    offline: bool
    allow_live_esco: bool
    esco_enabled: bool
    company_research_enabled: bool
    allow_live_company_research: bool
    esco_language: str
    query_override: str | None
    query_limit: int
    max_pages_per_provider: int
    top_n: int
    output_dir: str
    search_queries: list[str]
    jobs: list[Job]
    discovery: DiscoveryResult | None
    matching: MatchingResult | None
    recommendations: list[JobMatch]
    provider_diagnostics: dict
    workflow_status: str
    errors: list[str]
    nonfatal_errors: list[str]
    warnings: list[str]
    completed_nodes: list[str]
    report_markdown: str
    report_data: dict
    esco_enrichment: dict
    company_research: dict
    report_paths: dict[str, str]
    started_at: str
    completed_at: str


DiscoveryRunner = Callable[..., CandidateDiscoveryRun]
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_workflow(
    *, discovery_runner: DiscoveryRunner | None = None,
    esco_client_factory: Callable[[], ESCOClient] | None = None,
    company_researcher_factory: Callable[[], TavilyCompanyResearchClient] | None = None,
):
    """Create the research-to-report graph; discovery is injectable for tests."""
    run_discovery = discovery_runner or discover_candidate_jobs
    make_esco_client = esco_client_factory or ESCOClient
    make_company_researcher = company_researcher_factory or TavilyCompanyResearchClient
    builder = StateGraph(WorkflowState)
    builder.add_node("load_candidate", load_candidate_node)
    builder.add_node("discover_jobs", _discovery_node(run_discovery))
    builder.add_node("note_no_jobs", note_no_jobs_node)
    builder.add_node("note_partial_failure", note_partial_failure_node)
    builder.add_node("match_jobs", match_jobs_node)
    builder.add_node("select_recommendations", select_recommendations_node)
    builder.add_node("note_no_recommendations", note_no_recommendations_node)
    builder.add_node("enrich_esco", _esco_node(make_esco_client))
    builder.add_node("research_companies", _company_research_node(make_company_researcher))
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "load_candidate")
    builder.add_conditional_edges("load_candidate", _route_after_load, {
        "discover": "discover_jobs", "report": "generate_report",
    })
    builder.add_conditional_edges("discover_jobs", _route_after_discovery, {
        "empty": "note_no_jobs", "partial": "note_partial_failure", "continue": "match_jobs",
    })
    builder.add_edge("note_no_jobs", "select_recommendations")
    builder.add_edge("note_partial_failure", "match_jobs")
    builder.add_edge("match_jobs", "select_recommendations")
    builder.add_conditional_edges("select_recommendations", _route_after_selection, {
        "none": "note_no_recommendations", "selected": "enrich_esco",
    })
    builder.add_edge("note_no_recommendations", "enrich_esco")
    builder.add_edge("enrich_esco", "research_companies")
    builder.add_edge("research_companies", "generate_report")
    builder.add_edge("generate_report", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _mark(state: WorkflowState, node: str, **updates) -> dict:
    return {"completed_nodes": [*state.get("completed_nodes", []), node], **updates}


def load_candidate_node(state: WorkflowState) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    try:
        candidate = state.get("candidate")
        if candidate is None:
            data = state.get("profile_data")
            if data is None:
                profile_name = state.get("profile_name", "a")
                if profile_name not in {"a", "b"}:
                    raise ValueError("profile must be 'a' or 'b'")
                path = REPO_ROOT / "fixtures" / f"candidate_profile_{profile_name}.json"
                data = json.loads(path.read_text(encoding="utf-8"))
            candidate = CandidateProfile.model_validate(data)
        # Validate options early, before any provider calls.
        query_limit = state.get("query_limit", 3)
        pages = state.get("max_pages_per_provider", 1)
        top_n = state.get("top_n", 5)
        if not 1 <= query_limit <= 3:
            raise ValueError("query_limit must be between 1 and 3")
        if not 1 <= pages <= 5:
            raise ValueError("max_pages_per_provider must be between 1 and 5")
        if not 1 <= top_n <= 5:
            raise ValueError("top_n must be between 1 and 5")
        return _mark(state, "load_candidate", candidate=candidate, started_at=now,
                     errors=list(state.get("errors", [])), warnings=list(state.get("warnings", [])),
                     workflow_status="running")
    except Exception as exc:
        return _mark(state, "load_candidate", started_at=now, candidate=None,
                     errors=[*state.get("errors", []), f"Candidate/profile validation failed: {exc}"],
                     warnings=list(state.get("warnings", [])), workflow_status="failed")


def _discovery_node(discovery_runner: DiscoveryRunner):
    def discover_node(state: WorkflowState) -> dict:
        try:
            candidate = state["candidate"]
            assert candidate is not None
            if state.get("offline", False):
                result = _load_fixture_discovery(candidate)
                queries = _queries(candidate, state)
                warnings = [*state.get("warnings", []), "Synthetic fixture jobs were used for discovery and matching; job provider APIs were not called."]
            else:
                run = discovery_runner(
                    candidate,
                    query_override=state.get("query_override"),
                    max_pages_per_provider=state.get("max_pages_per_provider", 1),
                    search_query_limit=state.get("query_limit", 3),
                )
                result, queries = run.discovery, run.queries
                warnings = list(state.get("warnings", []))
            nonfatal_errors = list(state.get("nonfatal_errors", []))
            diagnostics = {
                "provider_counts": result.provider_counts,
                "deduplicated_provider_counts": result.deduplicated_provider_counts,
                "final_provider_counts": result.final_provider_counts,
                "duplicates_removed": result.duplicates_removed,
                "filter_exclusions": result.filter_diagnostics.excluded_counts,
                "filter_reasons": result.filter_diagnostics.exclusion_reasons,
                "errors": result.errors,
            }
            if result.errors:
                warnings.append("One or more job providers failed; available provider results were retained.")
                nonfatal_errors.extend(f"{provider}: {message}" for provider, message in result.errors.items())
            return _mark(state, "discover_jobs", discovery=result, jobs=result.jobs,
                         search_queries=queries, provider_diagnostics=diagnostics,
                         warnings=warnings, nonfatal_errors=nonfatal_errors)
        except Exception as exc:
            warning = f"Job discovery failed: {type(exc).__name__}: {exc}"
            diagnostics = {"provider_counts": {}, "deduplicated_provider_counts": {},
                           "final_provider_counts": {}, "duplicates_removed": 0,
                           "filter_exclusions": {}, "filter_reasons": {}, "errors": {"discovery": warning}}
            return _mark(state, "discover_jobs", discovery=DiscoveryResult(jobs=[], errors={"discovery": warning}),
                         jobs=[], search_queries=state.get("search_queries", []), provider_diagnostics=diagnostics,
                         errors=[*state.get("errors", []), warning])
    return discover_node


def _queries(candidate: CandidateProfile, state: WorkflowState) -> list[str]:
    override = state.get("query_override")
    if override and override.strip():
        return [override.strip()]
    return derive_search_queries(candidate, limit=state.get("query_limit", 3))


def _load_fixture_discovery(candidate: CandidateProfile) -> DiscoveryResult:
    fixture = REPO_ROOT / "fixtures" / "matching_jobs.json"
    jobs = [Job.model_validate(item) for item in json.loads(fixture.read_text(encoding="utf-8"))]
    providers = sorted({job.source for job in jobs})
    counts = {provider: sum(job.source == provider for job in jobs) for provider in providers}
    return DiscoveryResult(
        jobs=jobs, provider_counts=counts, deduplicated_provider_counts=counts.copy(),
        final_provider_counts=counts.copy(), filter_diagnostics=FilterDiagnostics(),
    )


def note_no_jobs_node(state: WorkflowState) -> dict:
    return _mark(state, "note_no_jobs", warnings=[*state.get("warnings", []), "No jobs were discovered after provider retrieval and filtering."])


def note_partial_failure_node(state: WorkflowState) -> dict:
    return _mark(state, "note_partial_failure")


def match_jobs_node(state: WorkflowState) -> dict:
    try:
        candidate = state["candidate"]
        assert candidate is not None
        matching = match_candidate_to_jobs(candidate, state.get("jobs", []), top_n=state.get("top_n", 5))
        return _mark(state, "match_jobs", matching=matching)
    except Exception as exc:
        message = f"Matching failed: {type(exc).__name__}: {exc}"
        return _mark(state, "match_jobs", matching=None, errors=[*state.get("errors", []), message])


def select_recommendations_node(state: WorkflowState) -> dict:
    matching = state.get("matching")
    recommendations = matching.top_matches if matching else []
    return _mark(state, "select_recommendations", recommendations=recommendations)


def note_no_recommendations_node(state: WorkflowState) -> dict:
    return _mark(state, "note_no_recommendations", warnings=[
        *state.get("warnings", []), "No discovered jobs met the existing evidence, hard-requirement, and match qualification criteria.",
    ])


def _esco_node(client_factory: Callable[[], ESCOClient]):
    def enrich_node(state: WorkflowState) -> dict:
        if state.get("offline", False) and not state.get("allow_live_esco", False):
            return _mark(state, "enrich_esco", esco_enrichment=_empty_esco_result("skipped_offline"))
        if not state.get("esco_enabled", True):
            return _mark(state, "enrich_esco", esco_enrichment=_empty_esco_result("skipped_disabled"))
        candidate = state.get("candidate")
        recommendations = state.get("recommendations", [])
        if candidate is None or not recommendations:
            return _mark(state, "enrich_esco", esco_enrichment=_empty_esco_result("skipped_no_recommendations"))

        client = None
        close_error = None
        try:
            client = client_factory()
            outcome = enrich_candidate_and_matches(
                candidate, recommendations[:5], client,
                language=state.get("esco_language", ESCO_LANGUAGE), max_lookups=32,
            )
        except Exception as exc:
            outcome = _empty_esco_result("unavailable")
            outcome.update({"failed_lookup_count": 1, "errors": [f"{type(exc).__name__}: {exc}"]})
        finally:
            close = getattr(client, "close", None) if client is not None else None
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    close_error = f"ESCO client cleanup failed: {type(exc).__name__}: {exc}"

        warnings = list(state.get("warnings", []))
        nonfatal_errors = list(state.get("nonfatal_errors", []))
        if close_error:
            warnings.append(close_error)
            nonfatal_errors.append(close_error)
        if outcome["status"] in {"partial", "unavailable"}:
            warnings.append(f"ESCO enrichment was {outcome['status']}; matching results and reports were retained.")
            for lookup in outcome.get("lookups", []):
                error = lookup.get("esco_result", {}).get("error")
                if error:
                    nonfatal_errors.append(f"ESCO {lookup.get('term')}: {error}")
            nonfatal_errors.extend(f"ESCO: {error}" for error in outcome.get("errors", []))
        return _mark(state, "enrich_esco", esco_enrichment=outcome,
                     warnings=warnings, nonfatal_errors=nonfatal_errors)
    return enrich_node


def _company_research_node(client_factory: Callable[[], TavilyCompanyResearchClient]):
    def research_node(state: WorkflowState) -> dict:
        if state.get("offline", False) and not state.get("allow_live_company_research", False):
            return _mark(state, "research_companies", company_research=_empty_company_research("skipped_offline"))
        if not state.get("company_research_enabled", False):
            return _mark(state, "research_companies", company_research=_empty_company_research("skipped_disabled"))
        companies = unique_recommended_companies(state.get("recommendations", []))
        if not companies:
            return _mark(state, "research_companies", company_research=_empty_company_research("skipped_no_recommendations"))

        researcher = None
        results: list[dict] = []
        warnings = list(state.get("warnings", []))
        nonfatal_errors = list(state.get("nonfatal_errors", []))
        identity_uncertainty = (
            "The company name comes from synthetic fixture data and may be fictional; "
            "search results are unverified identity matches."
        )
        try:
            researcher = client_factory()
            for company in companies:
                try:
                    result = researcher.research_company(company)
                except Exception as exc:
                    result = CompanyResearchResult(
                        company=company,
                        status="unavailable",
                        error=f"{type(exc).__name__}: company research failed.",
                        uncertainties=["Company information could not be verified because the search request failed."],
                    )
                result_data = result.model_dump(mode="json")
                if state.get("offline", False) and state.get("allow_live_company_research", False):
                    result_data["identity_uncertain"] = True
                    result_data["identity_status"] = "unverified"
                    result_data["identity_reason"] = identity_uncertainty
                    result_data["snapshot"] = None
                    for source in result_data.get("sources", []):
                        source["identity_status"] = "unverified"
                        source["identity_reason"] = identity_uncertainty
                    result_data.setdefault("uncertainties", []).append(identity_uncertainty)
                results.append(result_data)
        except Exception as exc:
            safe_error = f"{type(exc).__name__}: company research client could not be initialized."
            results = [CompanyResearchResult(
                company=company, status="unavailable", error=safe_error,
                uncertainties=["Company information could not be verified because the search client was unavailable."],
            ).model_dump(mode="json") for company in companies]
            if state.get("offline", False) and state.get("allow_live_company_research", False):
                for result_data in results:
                    result_data["identity_uncertain"] = True
                    result_data["identity_status"] = "unverified"
                    result_data["identity_reason"] = identity_uncertainty
                    result_data["snapshot"] = None
                    result_data["uncertainties"].append(
                        identity_uncertainty
                    )
                    for source in result_data.get("sources", []):
                        source["identity_status"] = "unverified"
                        source["identity_reason"] = identity_uncertainty
        finally:
            close = getattr(researcher, "close", None) if researcher is not None else None
            if callable(close):
                try:
                    close()
                except Exception:
                    warnings.append("Company research client cleanup failed.")

        statuses = [result["status"] for result in results]
        if all(status == "missing_api_key" for status in statuses):
            status = "missing_api_key"
        elif all(status == "unavailable" for status in statuses):
            status = "unavailable"
        elif any(result_status in {"unavailable", "missing_api_key"} for result_status in statuses):
            status = "partial"
        elif all(result_status == "no_results" for result_status in statuses):
            status = "no_results"
        else:
            status = "succeeded"
        if status in {"missing_api_key", "unavailable", "partial"}:
            warnings.append(f"Company research was {status}; report generation continued.")
            nonfatal_errors.extend(
                f"Company research for {result['company']}: {result['error']}"
                for result in results if result.get("error")
            )
        outcome = {
            "status": status,
            "company_count": len(companies),
            "request_limit": len(companies),
            "source_reference": "https://api.tavily.com/search",
            "identity_warning": (
                "Fixture company names may be fictional; Tavily results are unverified identity matches."
                if state.get("offline", False) and state.get("allow_live_company_research", False)
                else None
            ),
            "results": results,
        }
        return _mark(state, "research_companies", company_research=outcome,
                     warnings=warnings, nonfatal_errors=nonfatal_errors)
    return research_node


def _empty_company_research(status: str) -> dict:
    return {
        "status": status, "company_count": 0, "request_limit": 5,
        "source_reference": "https://api.tavily.com/search", "results": [],
    }


def _empty_esco_result(status: str) -> dict:
    return {
        "status": status, "language": ESCO_LANGUAGE, "lookup_count": 0,
        "failed_lookup_count": 0, "lookup_limit": 32, "lookups_truncated": False,
        "source_reference": ESCO_SOURCE_URL,
        "official_api_documentation": "https://ec.europa.eu/esco/api/doc/esco_api_doc.html",
        "lookups": [],
    }


def generate_report_node(state: WorkflowState) -> dict:
    markdown, report_data = create_report(state)
    return _mark(state, "generate_report", report_markdown=markdown, report_data=report_data)


def create_report(state: WorkflowState) -> tuple[str, dict]:
    candidate = state.get("candidate")
    discovery = state.get("discovery")
    matching = state.get("matching")
    recommendations = state.get("recommendations", [])
    provider_diagnostics = state.get("provider_diagnostics", {})
    started_at = state.get("started_at") or datetime.now(timezone.utc).isoformat()
    scope = {
        "queries": state.get("search_queries", []),
        "providers": sorted(provider_diagnostics.get("provider_counts", {}).keys()),
        "offline": state.get("offline", False),
        "allow_live_esco": state.get("allow_live_esco", False),
        "allow_live_company_research": state.get("allow_live_company_research", False),
        "job_data_source": "synthetic_fixtures" if state.get("offline", False) else "live_providers",
        "provider_diagnostics": provider_diagnostics,
    }
    report_data = {
        "candidate": candidate.model_dump(mode="json") if candidate else None,
        "search_scope": scope,
        "discovered_jobs": len(discovery.jobs) if discovery else 0,
        "evaluated_jobs": len(matching.ranked_matches) + len(matching.excluded_jobs) if matching else 0,
        "recommendations": [_serialize_match(match) for match in recommendations],
        "non_recommendations": _serialize_non_recommendations(matching),
        "esco_enrichment": state.get("esco_enrichment", _empty_esco_result("skipped")),
        "company_research": state.get("company_research", _empty_company_research("skipped")),
        "workflow": {
            "status": state.get("workflow_status", "running"),
            "started_at": started_at,
            "errors": state.get("errors", []),
            "nonfatal_errors": state.get("nonfatal_errors", []),
            "warnings": state.get("warnings", []),
            "nodes_executed": state.get("completed_nodes", []),
        },
    }

    lines = [
        "# Career Intelligence Report",
        "",
        f"**Generated:** {started_at}",
        f"**Candidate:** {candidate.name if candidate else 'Profile unavailable'}",
        "",
        "## Candidate summary",
        "",
        candidate.summary or "No summary provided." if candidate else "Candidate profile could not be loaded.",
        "",
        "## Search scope and diagnostics",
        "",
        f"- Queries: {', '.join(scope['queries']) or 'none'}",
        f"- Providers: {', '.join(scope['providers']) or 'none'}",
        f"- Mode: {_report_mode(scope)}",
        f"- Jobs discovered after filtering: {report_data['discovered_jobs']}",
        f"- Jobs evaluated: {report_data['evaluated_jobs']}",
        f"- Recommendations: {len(recommendations)} (maximum 5)",
        f"- Provider counts: {_format_counts(provider_diagnostics.get('provider_counts', {}))}",
        f"- Duplicates removed: {provider_diagnostics.get('duplicates_removed', 0)}",
        f"- Filter exclusions: {_format_counts(provider_diagnostics.get('filter_exclusions', {}))}",
    ]
    if scope["offline"]:
        lines.extend(["", "**Job data notice:** These are synthetic fixture jobs, not real vacancies."])
        if scope["allow_live_esco"]:
            lines.append("Job discovery and matching used fixtures; ESCO enrichment used live API requests.")
        if scope["allow_live_company_research"]:
            lines.append("Company research used live Tavily search results; fixture company identities are uncertain.")
    if provider_diagnostics.get("errors"):
        lines.extend(["", "Provider errors:"])
        lines.extend(f"- {name}: {error}" for name, error in provider_diagnostics["errors"].items())
    esco = report_data["esco_enrichment"]
    lines.extend([
        "", "## ESCO skill and occupation enrichment", "",
        f"- Enrichment status: {esco.get('status', 'skipped')}",
        f"- Lookups: {esco.get('lookup_count', 0)} (bounded to {esco.get('lookup_limit', 32)})",
        f"- Official source: {esco.get('source_reference', ESCO_SOURCE_URL)}",
        "- ESCO labels and URIs are official classification data. A returned search result is not by itself a verified skill equivalence. Exact/alternative statuses report label comparisons only and do not alter matching scores.",
        "",
    ])
    for lookup in esco.get("lookups", []):
        esco_result = lookup.get("esco_result", {})
        lines.append(f"### {lookup.get('subject', 'unknown')}: {lookup.get('term', '')} ({lookup.get('concept_type', '')})")
        lines.append(f"- Evidence basis: {lookup.get('evidence_source', 'unknown')}")
        if esco_result.get("source_reference"):
            lines.append(f"- ESCO search reference: {esco_result['source_reference']}")
        if esco_result.get("error"):
            lines.append(f"- ESCO error: {esco_result['error']}")
        status = esco_result.get("status")
        if status == "non_exact":
            lines.append("- ESCO returned candidate concepts, but no verified label match was found.")
        elif status == "ambiguous":
            lines.append("- ESCO returned multiple matching labels; no single concept was selected.")
        elif status == "no_match":
            lines.append("- ESCO returned no concepts for this term; no verified label match was found.")
        else:
            matched_concepts = [
                concept for concept in esco_result.get("concepts", [])
                if concept.get("match_type") in {"exact", "alternative"}
            ]
            for concept in matched_concepts:
                label, uri = concept.get("label") or "Label not returned", concept.get("uri")
                lines.append(f"- Verified label match ({concept.get('match_type')}): [{label}]({uri})" if uri else f"- Verified label match ({concept.get('match_type')}): {label}")
                lines.append(f"  - Type: {concept.get('concept_type')}; language: {concept.get('language')}; label relation: {concept.get('match_type')}")
                lines.append(f"  - Reference: {concept.get('source_reference')}")
            if not matched_concepts and not esco_result.get("error"):
                lines.append("- No verified label match was found.")
        if not esco_result.get("concepts") and status is None and not esco_result.get("error"):
            lines.append("- ESCO concepts: none returned for this term.")
        lines.append("")
    company_research = report_data["company_research"]
    lines.extend([
        "", "## Company research", "",
        f"- Research status: {company_research.get('status', 'skipped')}",
        f"- Companies researched: {company_research.get('company_count', 0)} (maximum {company_research.get('request_limit', 5)} requests)",
        f"- Search source: {company_research.get('source_reference', 'Tavily Search API')}",
        "- Tavily API success and company identity verification are reported separately. A similar domain alone does not establish identity.",
        "",
    ])
    if company_research.get("identity_warning"):
        lines.extend([f"**Identity uncertainty:** {company_research['identity_warning']}", ""])
    for company_result in company_research.get("results", []):
        lines.append(f"### {company_result.get('company', 'Company')}")
        lines.append(f"- Tavily research status: {company_result.get('status', 'unknown')}")
        identity_status = company_result.get("identity_status", "unverified")
        lines.append(f"- Company identity: {identity_status}")
        if company_result.get("error"):
            lines.append(f"- Research issue: {company_result['error']}")
        sources = company_result.get("sources", [])
        if identity_status == "verified" and company_result.get("snapshot"):
            lines.append(f"- Identity evidence: {company_result.get('identity_reason', 'source evidence matched the company name and domain.')}")
            verified_sources = [source for source in sources if source.get("identity_status") == "verified"]
            for index, source in enumerate(verified_sources[:3]):
                published = f" (published {source['published_date']})" if source.get("published_date") else " (publication date not provided)"
                label = "Company snapshot" if index == 0 else "Additional supported finding"
                lines.append(f"- {label} from [{source['title']}]({source['url']}){published}: {source['snippet'][:300]}")
        else:
            lines.append("- Reliable company information could not be confirmed; these search results are not attributed to the target company.")
            if sources:
                lines.append("- Unverified search results (source references only):")
                for source in sources[:3]:
                    lines.append(f"  - [{source['title']}]({source['url']})")
        for uncertainty in company_result.get("uncertainties", []):
            lines.append(f"- Uncertainty: {uncertainty}")
        lines.append("")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        for index, match in enumerate(recommendations, start=1):
            lines.extend(_match_markdown(index, match))
    else:
        lines.append("No positions qualified as recommendations.")
    if len(recommendations) < 5:
        lines.extend(["", "### Why fewer than five?", ""])
        if not discovery or not discovery.jobs:
            lines.append("No jobs were available after discovery and filtering.")
        elif not matching or not matching.top_matches:
            lines.append("No jobs satisfied the existing recommendation evidence, score, and hard-requirement criteria.")
        else:
            lines.append(f"Only {len(recommendations)} jobs met the existing qualification criteria; the workflow does not add unsupported matches.")
        lines.append("See the non-recommendation diagnostics in the JSON report for individual scores and exclusion reasons.")
    if state.get("warnings"):
        lines.extend(["", "## Warnings", "", *[f"- {item}" for item in state["warnings"]]])
    if state.get("errors"):
        lines.extend(["", "## Workflow errors", "", *[f"- {item}" for item in state["errors"]]])
    markdown = "\n".join(lines) + "\n"
    return markdown, report_data


def _serialize_match(match: JobMatch) -> dict:
    seniority = next((dim for dim in match.dimensions if dim.name == "seniority_qualifications"), None)
    return {
        "job": match.job.model_dump(mode="json"),
        "score": match.score,
        "evidence_coverage": match.evidence_coverage,
        "support_status": match.support_status.value,
        "strengths": match.strengths,
        "gaps": match.gaps,
        "geographic_status": match.geographic_status.value if match.geographic_status else None,
        "geographic_evidence": match.geographic_evidence,
        "seniority_evidence": seniority.evidence if seniority else [],
        "seniority_status": "unknown" if seniority is None or seniority.score is None else seniority.score,
        "uncertainties": match.unresolved_requirements,
        "warnings": match.recommendation_warnings,
    }


def _serialize_non_recommendations(matching: MatchingResult | None) -> list[dict]:
    if matching is None:
        return []
    selected = {match.job.job_id for match in matching.top_matches}
    items = []
    for match in matching.ranked_matches:
        if match.job.job_id in selected:
            continue
        reasons = [*match.unresolved_requirements, *match.recommendation_blockers]
        if match.score is None or match.support_status.value != "sufficient":
            reasons.append(f"Evidence status: {match.support_status.value}.")
        if match.score is not None and match.score < MINIMUM_MATCH_SCORE:
            reasons.append(f"Score {match.score} is below the existing {MINIMUM_MATCH_SCORE}/100 recommendation threshold.")
        if not reasons:
            reasons.append("Qualified, but outside the selected recommendation limit or ranking cutoff.")
        items.append({
            "job": match.job.model_dump(mode="json"), "score": match.score,
            "evidence_coverage": match.evidence_coverage, "support_status": match.support_status.value,
            "reasons": reasons, "gaps": match.gaps, "warnings": match.recommendation_warnings,
        })
    items.extend({
        "job": item.job.model_dump(mode="json"), "score": item.score,
        "reasons": item.violations, "geographic_status": item.geographic_status.value if item.geographic_status else None,
        "geographic_evidence": item.geographic_evidence,
    } for item in matching.excluded_jobs)
    return items


def _match_markdown(index: int, match: JobMatch) -> list[str]:
    job = match.job
    seniority = next((dim for dim in match.dimensions if dim.name == "seniority_qualifications"), None)
    lines = [
        f"### {index}. {job.title} — {job.company}", "",
        f"- Location: {job.location}{f', {job.country}' if job.country else ''}",
        f"- Source: [{job.source}]({job.source_url})" if job.source_url else f"- Source: {job.source} (URL not provided)",
        f"- Score: {match.score}/100" if match.score is not None else "- Score: unavailable",
        f"- Evidence coverage: {match.evidence_coverage}%",
        f"- Work model: {job.work_model.value}",
        f"- Geographic status: {match.geographic_status.value if match.geographic_status else 'not required'}",
        f"- Strengths: {'; '.join(match.strengths[:4]) or 'none recorded'}",
        f"- Potential gaps: {'; '.join(match.gaps[:4]) or 'none recorded'}",
        f"- Seniority: {'; '.join(seniority.evidence) if seniority and seniority.evidence else 'unknown or unsupported'}",
        f"- Uncertainties: {'; '.join(match.unresolved_requirements) or 'none recorded'}",
    ]
    if match.recommendation_warnings:
        lines.append(f"- Considerations: {'; '.join(match.recommendation_warnings)}")
    lines.append("")
    return lines


def _format_counts(values: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items())) or "none"


def _report_mode(scope: dict) -> str:
    if not scope["offline"]:
        return "live"
    modes = ["synthetic fixture jobs"]
    if scope["allow_live_esco"]:
        modes.append("live ESCO enrichment")
    if scope["allow_live_company_research"]:
        modes.append("live Tavily company research")
    return " + ".join(modes) if len(modes) > 1 else "offline fixture (synthetic job data)"


def finalize_node(state: WorkflowState) -> dict:
    now = datetime.now(timezone.utc)
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    if state.get("workflow_status") != "failed":
        if errors:
            status = "completed_with_errors"
        elif not state.get("jobs"):
            status = "completed_no_jobs"
        elif not state.get("recommendations"):
            status = "completed_no_recommendations"
        elif warnings:
            status = "completed_with_warnings"
        else:
            status = "completed"
    else:
        status = "failed"
    completed = now.isoformat()
    report_data = dict(state.get("report_data", {}))
    report_data["workflow"] = {
        **report_data.get("workflow", {}), "status": status,
        "completed_at": completed, "errors": errors,
        "nonfatal_errors": state.get("nonfatal_errors", []), "warnings": warnings,
        "nodes_executed": [*state.get("completed_nodes", []), "finalize"],
    }
    directory = Path(state.get("output_dir", "reports"))
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    profile_name = state.get("profile_name", "custom")
    markdown_path = directory / f"career_report_{profile_name}_{stamp}.md"
    json_path = directory / f"career_report_{profile_name}_{stamp}.json"
    paths: dict[str, str] = {}
    markdown = state.get("report_markdown", "").rstrip()
    markdown += (
        "\n\n## Run completion\n\n"
        f"- Status: {status}\n"
        f"- Completed: {completed}\n"
        f"- Nodes executed: {', '.join([*state.get('completed_nodes', []), 'finalize'])}\n"
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths = {"markdown": str(markdown_path), "json": str(json_path)}
    except OSError as exc:
        errors.append(f"Could not save report files: {exc}")
        status = "completed_with_errors" if status != "failed" else status
        report_data["workflow"]["errors"] = errors
        report_data["workflow"]["status"] = status
    return _mark(state, "finalize", completed_at=completed, workflow_status=status,
                 report_markdown=markdown, report_data=report_data, report_paths=paths,
                 errors=errors, warnings=warnings)


def _route_after_load(state: WorkflowState) -> str:
    return "discover" if state.get("candidate") is not None else "report"


def _route_after_discovery(state: WorkflowState) -> str:
    if not state.get("jobs"):
        return "empty"
    discovery = state.get("discovery")
    return "partial" if discovery and discovery.errors else "continue"


def _route_after_selection(state: WorkflowState) -> str:
    return "selected" if state.get("recommendations") else "none"
