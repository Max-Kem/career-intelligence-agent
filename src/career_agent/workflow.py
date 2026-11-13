from __future__ import annotations

import html
import json
import re
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
from .intelligence_synthesis import run_grounded_synthesis
from .matching import MINIMUM_MATCH_SCORE, JobMatch, MatchingResult, match_candidate_to_jobs
from .models import Job
from .requirements_review import requirements_to_verify
from .vector_store import run_knowledge_retrieval


class WorkflowState(TypedDict, total=False):
    profile_name: str
    profile_data: dict
    candidate: CandidateProfile | None
    offline: bool
    allow_live_esco: bool
    esco_enabled: bool
    company_research_enabled: bool
    allow_live_company_research: bool
    market_intelligence_enabled: bool
    allow_live_market_intelligence: bool
    llm_synthesis_enabled: bool
    allow_live_llm_synthesis: bool
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
    market_intelligence: dict
    intelligence_synthesis: dict
    report_paths: dict[str, str]
    started_at: str
    completed_at: str


DiscoveryRunner = Callable[..., CandidateDiscoveryRun]
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_workflow(
    *, discovery_runner: DiscoveryRunner | None = None,
    esco_client_factory: Callable[[], ESCOClient] | None = None,
    company_researcher_factory: Callable[[], TavilyCompanyResearchClient] | None = None,
    market_intelligence_runner: Callable[[str], dict] | None = None,
    synthesis_runner: Callable[..., dict] | None = None,
):
    """Create the research-to-report graph; discovery is injectable for tests."""
    run_discovery = discovery_runner or discover_candidate_jobs
    make_esco_client = esco_client_factory or ESCOClient
    make_company_researcher = company_researcher_factory or TavilyCompanyResearchClient
    retrieve_market = market_intelligence_runner or (lambda question: run_knowledge_retrieval(question, live=True))
    synthesize = synthesis_runner or run_grounded_synthesis
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
    builder.add_node("retrieve_market_intelligence", _market_intelligence_node(retrieve_market))
    builder.add_node("synthesize_career_intelligence", _synthesis_node(synthesize))
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
    builder.add_edge("research_companies", "retrieve_market_intelligence")
    builder.add_edge("retrieve_market_intelligence", "synthesize_career_intelligence")
    builder.add_edge("synthesize_career_intelligence", "generate_report")
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


def _market_questions(candidate: CandidateProfile, recommendations: list[JobMatch]) -> list[str]:
    roles: list[str] = []
    for role in [*candidate.desired_job_titles, *(match.job.title for match in recommendations[:2])]:
        if role and role.casefold() not in {item.casefold() for item in roles}:
            roles.append(role)
        if len(roles) >= 3:
            break
    role_text = ", ".join(roles) or "the candidate's target roles"
    skills = list(dict.fromkeys(candidate.skills))[:4]
    gaps: list[str] = []
    for match in recommendations[:3]:
        for gap in match.gaps:
            if gap not in gaps:
                gaps.append(gap)
    skills_text = ", ".join(skills) if skills else "digital and technical skills"
    gap_text = "; ".join(gaps[:2])
    return [
        f"In Germany, what labour-market and qualification trends affect {role_text} and skills such as {skills_text}?",
        f"How is AI changing occupational tasks and employment relevant to {role_text}?",
        f"What digital skills, ICT specialist demand, and upskilling pathways are relevant in Germany for {role_text}?"
        + (f" Consider these currently identified candidate-job gaps: {gap_text}." if gap_text else ""),
    ]


def _market_intelligence_node(retrieve: Callable[[str], dict]):
    def retrieve_node(state: WorkflowState) -> dict:
        if state.get("offline", False) and not state.get("allow_live_market_intelligence", False):
            return _mark(state, "retrieve_market_intelligence", market_intelligence=_empty_market_intelligence("skipped_offline"))
        if not state.get("market_intelligence_enabled", False):
            return _mark(state, "retrieve_market_intelligence", market_intelligence=_empty_market_intelligence("skipped_disabled"))
        candidate = state.get("candidate")
        if candidate is None:
            return _mark(state, "retrieve_market_intelligence", market_intelligence=_empty_market_intelligence("skipped_no_candidate"))

        questions = _market_questions(candidate, state.get("recommendations", []))
        per_query: list[dict] = []
        errors: list[str] = []
        evidence_by_id: dict[str, dict] = {}
        for question in questions:
            try:
                outcome = retrieve(question)
                matches = outcome.get("matches", []) if isinstance(outcome, dict) else []
                per_query.append({"question": question, "status": "succeeded" if matches else "no_results", "result_count": len(matches)})
                for item in matches[:3]:
                    key = item.get("chunk_id")
                    if not key or key == "Not provided":
                        key = f"{item.get('document_title')}|{item.get('page_number')}|{item.get('excerpt')}"
                    if key in evidence_by_id:
                        evidence_by_id[key]["matched_queries"].append(question)
                    else:
                        evidence_by_id[key] = {**item, "matched_queries": [question]}
            except Exception as exc:
                error = f"{type(exc).__name__}: retrieval request failed."
                errors.append(error)
                per_query.append({"question": question, "status": "unavailable", "result_count": 0, "error": error})
        evidence = list(evidence_by_id.values())[:9]
        if errors and evidence:
            status = "partial"
        elif errors:
            status = "unavailable"
        elif evidence:
            status = "succeeded"
        else:
            status = "no_results"
        outcome = {
            "status": status,
            "source": "Pinecone career-market-knowledge / career-market-knowledge-v1",
            "query_count": len(questions),
            "query_limit": 3,
            "questions": questions,
            "query_results": per_query,
            "evidence": evidence,
            "similarity_note": "Similarity scores indicate semantic closeness to a query, not factual confidence.",
            "candidate_context": {
                "target_roles": candidate.desired_job_titles,
                "listed_skills": candidate.skills,
                "matching_gaps": list(dict.fromkeys(gap for match in state.get("recommendations", []) for gap in match.gaps))[:8],
            },
            "errors": errors,
        }
        warnings = list(state.get("warnings", []))
        nonfatal_errors = list(state.get("nonfatal_errors", []))
        if status in {"partial", "unavailable"}:
            warnings.append(f"Labour-market intelligence retrieval was {status}; report generation continued without fabricated replacement evidence.")
            nonfatal_errors.extend(f"Market intelligence: {message}" for message in errors)
        return _mark(state, "retrieve_market_intelligence", market_intelligence=outcome,
                     warnings=warnings, nonfatal_errors=nonfatal_errors)
    return retrieve_node


def _empty_market_intelligence(status: str) -> dict:
    return {
        "status": status, "source": "Pinecone career-market-knowledge / career-market-knowledge-v1",
        "query_count": 0, "query_limit": 3, "questions": [], "query_results": [],
        "evidence": [], "similarity_note": "Similarity scores indicate semantic closeness to a query, not factual confidence.",
        "candidate_context": {}, "errors": [],
    }


def _synthesis_node(synthesize: Callable[..., dict]):
    def synthesize_node(state: WorkflowState) -> dict:
        if (state.get("offline", False) or not state.get("llm_synthesis_enabled", False)
                or not state.get("allow_live_llm_synthesis", False)):
            return _mark(state, "synthesize_career_intelligence",
                         intelligence_synthesis=_empty_synthesis("skipped_opt_in_required"))
        market = state.get("market_intelligence", _empty_market_intelligence("skipped"))
        candidate = state.get("candidate")
        if not candidate or not market.get("evidence"):
            return _mark(state, "synthesize_career_intelligence",
                         intelligence_synthesis=_empty_synthesis("unavailable_no_evidence"))
        recommendations = state.get("recommendations", [])
        review = _potential_matches_requiring_review(state.get("matching"), candidate)
        try:
            outcome = synthesize(market, candidate, recommendations, review)
            if not isinstance(outcome, dict):
                outcome = _empty_synthesis("failed_invalid_response")
        except Exception as exc:
            outcome = _empty_synthesis("failed")
            outcome["error"] = f"Synthesis request failed ({type(exc).__name__}); deterministic intelligence remains available."
        warnings = list(state.get("warnings", []))
        nonfatal_errors = list(state.get("nonfatal_errors", []))
        if outcome.get("status") in {"failed", "unavailable"}:
            reason = outcome.get("error") or "No sufficiently complete, traceable market passages were available."
            warnings.append("LLM career-intelligence synthesis was unavailable; deterministic report sections were retained.")
            nonfatal_errors.append(f"LLM synthesis: {reason}")
        return _mark(state, "synthesize_career_intelligence", intelligence_synthesis=outcome,
                     warnings=warnings, nonfatal_errors=nonfatal_errors)
    return synthesize_node


def _empty_synthesis(status: str) -> dict:
    return {
        "status": status, "generation_mode": "deterministic_fallback", "model": None,
        "request_count": 0, "request_limit": 1, "market_intelligence": [],
        "personalized_career_intelligence": [], "error": None, "failure_stage": None,
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


def _normalize_citation_integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
        return int(value.strip())
    return None


def _personalized_intelligence(market: dict, candidate: CandidateProfile | None, recommendations: list[JobMatch]) -> dict:
    """Create conservative evidence-linked implications; never add skills or market claims."""
    if not market.get("evidence"):
        return {
            "status": "unavailable",
            "message": "Personalized career intelligence is unavailable because no retrieved market evidence was available.",
            "insights": [],
            "limitations": ["No retrieved evidence was available; no market conclusions or new skill gaps were generated."],
        }
    roles = candidate.desired_job_titles if candidate else []
    listed_skills = candidate.skills if candidate else []
    existing_gaps = list(dict.fromkeys(gap for match in recommendations for gap in match.gaps))
    role_text = ", ".join(roles[:3]) or "the candidate's target roles"
    insights: list[dict] = []
    seen_topics: set[tuple[str, str]] = set()
    incomplete: list[str] = []
    for evidence in market.get("evidence", []):
        excerpt = evidence.get("excerpt")
        title = evidence.get("document_title")
        publisher = evidence.get("publisher")
        year = _normalize_citation_integer(evidence.get("publication_year"))
        page = _normalize_citation_integer(evidence.get("page_number"))
        url = evidence.get("source_url")
        scope = evidence.get("geographic_scope")
        if not (isinstance(excerpt, str) and excerpt.strip() and isinstance(title, str) and title not in {"", "Not provided"}
                and isinstance(publisher, str) and publisher not in {"", "Not provided"} and year and page
                and isinstance(url, str) and url.startswith(("https://", "http://"))):
            incomplete.append(str(evidence.get("chunk_id", "retrieved record")))
            continue
        text = excerpt.casefold()
        if any(term in text for term in ("generative ai", "genai", "occupation", "automation", "occupational task")):
            topic = "AI and occupational change"
        elif any(term in text for term in ("job openings", "qualification", "labour force", "employment forecast")):
            topic = "Qualifications and labour-market trends"
        elif any(term in text for term in ("ict specialist", "digital skills", "digital skill", "upskilling", "training", "skills development")):
            topic = "Digital skills and workforce development"
        else:
            topic = "Digital transformation context"
        scope_key = scope if scope in {"Germany_focused", "international_global"} else "unknown_scope"
        topic_key = (topic, scope_key)
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        citation = {
            "document_title": title, "publisher": publisher, "publication_year": year,
            "page_number": page, "source_url": url,
            "geographic_scope": scope if scope in {"Germany_focused", "international_global"} else "Not provided",
        }
        if citation["geographic_scope"] == "international_global":
            implication = f"For {role_text}, this is international context on occupational change; it does not establish Germany-specific demand or individual suitability."
            next_step = "Use this as global context, then check Germany-focused sources and current target-role postings before making a local career decision."
        elif citation["geographic_scope"] == "Germany_focused":
            implication = f"This Germany-focused finding is relevant context for {role_text}, but it does not establish that a specific vacancy exists or confirm candidate fit."
            next_step = f"Compare the cited finding with current postings for {role_text}; record only requirements they state explicitly."
        else:
            implication = f"This source provides career context for {role_text}, but its geographic scope is not verified. Do not use it alone for a Germany-specific conclusion."
            next_step = "Confirm geographic relevance in a source with a verified country scope before applying this finding locally."
        matched_skills = [skill for skill in listed_skills if skill.casefold() in text]
        gap_terms = []
        for gap in existing_gaps:
            found = re.search(r"Job mentions (.+?), which is not listed in the candidate profile", gap, flags=re.IGNORECASE)
            term = found.group(1).strip(" .") if found else None
            if term and term.casefold() in text and term not in gap_terms:
                gap_terms.append(term)
        if matched_skills:
            implication += f" The excerpt explicitly mentions candidate-listed skill(s) {', '.join(matched_skills[:3])}; this is context, not proof of proficiency or employer demand."
        if gap_terms:
            implication += f" It also explicitly mentions currently reported match gap(s) {', '.join(gap_terms[:3])}; validate these against target postings before prioritizing development."
            next_step = f"Check whether {', '.join(gap_terms[:3])} recur in current postings for {role_text}, then decide whether development is useful."
        elif existing_gaps:
            implication += " The excerpt does not explicitly mention the existing matching gaps, so it does not validate them."
        elif recommendations:
            implication += " The matching results recorded no skill gaps; this finding does not create a new gap."
        insights.append({
            "topic": topic, "source_finding": excerpt,
            "candidate_implication": implication, "suggested_next_step": next_step,
            "citation": citation,
        })
        if len(insights) >= 4:
            break
    if not insights:
        return {
            "status": "unavailable",
            "message": "Personalized career intelligence is unavailable because retrieved evidence had incomplete citation metadata that could not be traced safely.",
            "insights": [],
            "limitations": [f"Incomplete citation metadata in {len(incomplete)} retrieved record(s); those records were not used to generate personalized claims."],
        }
    limitations = ["Implications and next steps are deterministic suggestions, not additional source findings or a suitability score."]
    if incomplete:
        limitations.append(f"{len(incomplete)} retrieved record(s) with incomplete citation metadata were omitted from personalized insights.")
    return {"status": "available", "message": None, "insights": insights, "limitations": limitations}


def create_report(state: WorkflowState) -> tuple[str, dict]:
    candidate = state.get("candidate")
    discovery = state.get("discovery")
    matching = state.get("matching")
    recommendations = state.get("recommendations", [])
    provider_diagnostics = state.get("provider_diagnostics", {})
    market_intelligence = state.get("market_intelligence", _empty_market_intelligence("skipped"))
    personalized_intelligence = _personalized_intelligence(market_intelligence, candidate, recommendations)
    started_at = state.get("started_at") or datetime.now(timezone.utc).isoformat()
    scope = {
        "queries": state.get("search_queries", []),
        "providers": sorted(provider_diagnostics.get("provider_counts", {}).keys()),
        "offline": state.get("offline", False),
        "allow_live_esco": state.get("allow_live_esco", False),
        "allow_live_company_research": state.get("allow_live_company_research", False),
        "market_intelligence_enabled": state.get("market_intelligence_enabled", False),
        "allow_live_market_intelligence": state.get("allow_live_market_intelligence", False),
        "llm_synthesis_enabled": state.get("llm_synthesis_enabled", False),
        "allow_live_llm_synthesis": state.get("allow_live_llm_synthesis", False),
        "job_data_source": "synthetic_fixtures" if state.get("offline", False) else "live_providers",
        "provider_diagnostics": provider_diagnostics,
    }
    report_data = {
        "candidate": candidate.model_dump(mode="json") if candidate else None,
        "search_scope": scope,
        "discovered_jobs": len(discovery.jobs) if discovery else 0,
        "evaluated_jobs": len(matching.ranked_matches) + len(matching.excluded_jobs) if matching else 0,
        "score_interpretation": "Indicative alignment based on available profile and job evidence; not hiring probability or verified qualification.",
        "evidence_coverage_interpretation": "Percentage of weighted matching signals with evidence available for evaluation; it is not a measure of full employer requirement fulfillment, qualification, or hiring probability.",
        "recommendations": [_serialize_match(match, candidate) for match in recommendations],
        "potential_matches_requiring_review": _potential_matches_requiring_review(matching, candidate),
        "non_recommendations": _serialize_non_recommendations(matching, candidate),
        "esco_enrichment": state.get("esco_enrichment", _empty_esco_result("skipped")),
        "company_research": state.get("company_research", _empty_company_research("skipped")),
        "market_intelligence": market_intelligence,
        "personalized_career_intelligence": personalized_intelligence,
        "intelligence_synthesis": state.get("intelligence_synthesis", _empty_synthesis("skipped_opt_in_required")),
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
    market = report_data["market_intelligence"]
    lines.extend([
        "## Labour-market intelligence", "",
        f"- Retrieval status: {market.get('status', 'skipped')}",
        f"- Queries: {market.get('query_count', 0)} (maximum {market.get('query_limit', 3)} per run)",
        f"- Source: {market.get('source', 'Pinecone knowledge base')}",
        f"- Similarity: {market.get('similarity_note', 'Scores are not factual confidence.')}",
    ])
    context = market.get("candidate_context", {})
    roles = context.get("target_roles", [])
    skills = context.get("listed_skills", [])
    gaps = context.get("matching_gaps", [])
    if roles or skills or gaps:
        lines.append(f"- Candidate connection: queries were scoped to target roles ({', '.join(roles) or 'not specified'}), listed skills ({', '.join(skills[:6]) or 'not specified'}), and matching gaps where available.")
        if gaps:
            lines.append(f"- Matching engine gaps for comparison: {'; '.join(gaps[:4])}")
    lines.append("")
    lines.extend(["### Personalized Career Intelligence", "", f"- Status: {personalized_intelligence['status']}"])
    if personalized_intelligence.get("message"):
        lines.extend([personalized_intelligence["message"], ""])
    for insight in personalized_intelligence.get("insights", []):
        citation = insight["citation"]
        source_title = citation["document_title"]
        source_link = f"[{source_title}]({citation['source_url']})"
        page = _normalize_citation_integer(citation.get("page_number"))
        scope = {"international_global": "international/global", "Germany_focused": "Germany-focused"}.get(citation.get("geographic_scope"), "geographic scope unverified")
        reference = f"{source_link}, {citation['publisher']} ({citation['publication_year']}), PDF page {page} — {scope}"
        lines.extend([
            f"**{insight['topic']}**",
            f"- Source finding: “{insight['source_finding']}” ({reference}).",
            f"- Candidate implication (reasoned): {insight['candidate_implication']} ({reference}).",
            f"- Suggested next step: {insight['suggested_next_step']}",
            "",
        ])
    for limitation in personalized_intelligence.get("limitations", []):
        lines.append(f"- Limitation: {limitation}")
    if personalized_intelligence.get("limitations"):
        lines.append("")
    synthesis = report_data["intelligence_synthesis"]
    lines.extend(["### Grounded LLM Synthesis", "", f"- Status: {synthesis.get('status', 'skipped')}",
                  f"- Provenance: {synthesis.get('generation_mode', 'deterministic_fallback')}",
                  f"- Model requests: {synthesis.get('request_count', 0)} (maximum {synthesis.get('request_limit', 1)})"])
    if synthesis.get("error"):
        lines.append(f"- Note: {synthesis['error']}")
    if synthesis.get("failure_stage"):
        lines.append(f"- Diagnostic stage: {synthesis['failure_stage']}")
    for section_key, section_title in (("market_intelligence", "Market Intelligence"),
                                       ("personalized_career_intelligence", "Personalized Career Intelligence")):
        paragraphs = synthesis.get(section_key, [])
        if paragraphs:
            lines.extend(["", f"**{section_title}**", ""])
        for paragraph in paragraphs:
            lines.append(paragraph["text"])
            citations = paragraph.get("citations")
            if isinstance(citations, list):
                for citation in citations:
                    kind = citation.get("kind")
                    reference_id = citation.get("reference_id", "")
                    if kind == "market_source":
                        source = citation
                        page = source.get("page_number")
                        page_num = _normalize_citation_integer(page)
                        page_label = f", PDF page {page_num}" if page_num else ""
                        publisher_year = source.get("publisher", "Not provided")
                        year = source.get("publication_year", "Not provided")
                        geographic_scope = {
                            "Germany_focused": "Germany-focused", "international_global": "international/global",
                        }.get(source.get("geographic_scope"), "scope not verified")
                        url = source.get("source_url")
                        title = source.get("document_title", source.get("evidence_id", "Source"))
                        if isinstance(url, str) and url.startswith(("https://", "http://")):
                            lines.append(f"Market source: [{title}]({url}), {publisher_year} ({year}){page_label}, {geographic_scope} [ID: {source.get('evidence_id')}].")
                        else:
                            lines.append(f"Market source: {title}{page_label}, {geographic_scope} [ID: {source.get('evidence_id')}].")
                    elif kind == "job_posting":
                        title = citation.get("document_title", "Job posting")
                        company = citation.get("company")
                        label = f"{title} — {company}" if company else title
                        url = citation.get("source_url")
                        if isinstance(url, str) and url.startswith(("https://", "http://")):
                            lines.append(f"Job posting: [{label}]({url}) [Reference: {reference_id}].")
                        else:
                            lines.append(f"Job posting: {label} [Reference: {reference_id}; URL unavailable].")
                    elif kind == "candidate_profile":
                        lines.append(f"Candidate profile: {citation.get('label', 'profile evidence')} [Reference: {reference_id}].")
            else:
                # Compatibility for reports generated before typed citations.
                for source in paragraph.get("sources", []):
                    page = source.get("page_number")
                    page_num = _normalize_citation_integer(page)
                    page_label = f", PDF page {page_num}" if page_num else ""
                    publisher_year = source.get("publisher", "Not provided")
                    year = source.get("publication_year", "Not provided")
                    geographic_scope = {
                        "Germany_focused": "Germany-focused", "international_global": "international/global",
                    }.get(source.get("geographic_scope"), "scope not verified")
                    url = source.get("source_url")
                    title = source.get("document_title", source.get("evidence_id", "Source"))
                    if isinstance(url, str) and url.startswith(("https://", "http://")):
                        lines.append(f"Source: [{title}]({url}), {publisher_year} ({year}){page_label}, {geographic_scope} [ID: {source.get('evidence_id')}].")
                    else:
                        lines.append(f"Source: {title}{page_label}, {geographic_scope} [ID: {source.get('evidence_id')}].")
            lines.append("")
    if market.get("status") in {"skipped_offline", "skipped_disabled", "skipped_no_candidate"}:
        lines.append("Market evidence retrieval was not run; no findings are inferred.")
    elif not market.get("evidence"):
        lines.append("No market evidence was returned. No replacement findings or citations were generated.")
    for index, evidence in enumerate(market.get("evidence", []), start=1):
        title = evidence.get("document_title", "Source title unavailable")
        publisher = evidence.get("publisher", "Publisher unavailable")
        year = evidence.get("publication_year", "Year unavailable")
        if isinstance(year, float) and year.is_integer():
            year = int(year)
        page = _normalize_citation_integer(evidence.get("page_number")) or "Not provided"
        if page == "Not provided" and evidence.get("page_start") is not None:
            page = str(evidence["page_start"]) if evidence.get("page_end") in (None, evidence["page_start"]) else f"{evidence['page_start']}-{evidence['page_end']}"
        url = evidence.get("source_url")
        source = f"[{title}]({url})" if isinstance(url, str) and url.startswith(("http://", "https://")) else title
        scope_label = {"international_global": "Global/international", "Germany_focused": "Germany-focused"}.get(evidence.get("geographic_scope"), evidence.get("geographic_scope", "Scope not provided"))
        lines.extend([
            f"### {index}. {source}",
            f"- Publisher/year: {publisher} ({year}); PDF page {page}",
            f"- Evidence scope: {scope_label}",
            f"- Retrieved for: {'; '.join(evidence.get('matched_queries', []))}",
            f"- Similarity score: {evidence.get('score', 'Not provided')}",
            f"- Excerpt: {evidence.get('excerpt', 'Text unavailable')}",
            "",
        ])
    for error in market.get("errors", []):
        lines.append(f"- Retrieval issue: {error}")
    if market.get("errors"):
        lines.append("")
    lines.extend([
        "", "## Recommendations", "",
        f"**Score interpretation:** {report_data['score_interpretation']}",
        f"**Matching evidence coverage:** {report_data['evidence_coverage_interpretation']}", "",
    ])
    if recommendations:
        for index, match in enumerate(recommendations, start=1):
            lines.extend(_match_markdown(index, match, candidate))
    else:
        lines.append("No positions qualified as recommendations.")
    potential_matches = report_data["potential_matches_requiring_review"]
    if potential_matches:
        lines.extend(["", "## Potential matches requiring review", "", "These jobs did not qualify as recommendations. Scores indicate evidence alignment only; review the unresolved support before treating them as suitable.", ""])
        for item in potential_matches:
            job = item["job"]
            url = job.get("source_url")
            title = f"[{job['title']}]({url})" if url else job["title"]
            lines.append(f"- **{title} — {job.get('company', 'Unknown company')}**: {item['score']}/100; {item['support_status']}. {'; '.join(item['reasons'][:3])}")
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


def _serialize_match(match: JobMatch, candidate: CandidateProfile | None = None) -> dict:
    seniority = next((dim for dim in match.dimensions if dim.name == "seniority_qualifications"), None)
    return {
        "job": match.job.model_dump(mode="json"),
        "score": match.score,
        "evidence_coverage": match.evidence_coverage,
        "career_direction": {
            "status": match.career_direction_status,
            "evidence": match.career_direction_evidence,
        },
        "support_status": match.support_status.value,
        "strengths": match.strengths,
        "gaps": match.gaps,
        "geographic_status": match.geographic_status.value if match.geographic_status else None,
        "geographic_evidence": match.geographic_evidence,
        "seniority_evidence": seniority.evidence if seniority else [],
        "seniority_status": "unknown" if seniority is None or seniority.score is None else seniority.score,
        "uncertainties": match.unresolved_requirements,
        "warnings": match.recommendation_warnings,
        "requirements_to_verify": requirements_to_verify(match.job.description or "", candidate, match.job.title),
    }


def _serialize_non_recommendations(matching: MatchingResult | None, candidate: CandidateProfile | None = None) -> list[dict]:
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
        if match.support_status.value == "provisional" and match.evidence_coverage < 70:
            reasons.append(f"Evidence coverage is {match.evidence_coverage}%; current support remains provisional.")
        if match.score is not None and match.score < MINIMUM_MATCH_SCORE:
            reasons.append(f"Score {match.score} is below the existing {MINIMUM_MATCH_SCORE}/100 recommendation threshold.")
        if not reasons:
            reasons.append("Qualified, but outside the selected recommendation limit or ranking cutoff.")
        items.append({
            "job": match.job.model_dump(mode="json"), "score": match.score,
            "evidence_coverage": match.evidence_coverage, "support_status": match.support_status.value,
            "career_direction": {"status": match.career_direction_status, "evidence": match.career_direction_evidence},
            "reasons": reasons, "gaps": match.gaps, "warnings": match.recommendation_warnings,
            "requirements_to_verify": requirements_to_verify(match.job.description or "", candidate, match.job.title),
        })
    items.extend({
        "job": item.job.model_dump(mode="json"), "score": item.score,
        "reasons": item.violations, "geographic_status": item.geographic_status.value if item.geographic_status else None,
        "geographic_evidence": item.geographic_evidence,
    } for item in matching.excluded_jobs)
    return items


def _potential_matches_requiring_review(matching: MatchingResult | None, candidate: CandidateProfile | None) -> list[dict]:
    if matching is None:
        return []
    selected = {match.job.job_id for match in matching.top_matches}
    result = []
    for match in matching.ranked_matches:
        if (match.job.job_id in selected or match.score is None or match.score < MINIMUM_MATCH_SCORE
                or (match.support_status.value == "sufficient" and not match.recommendation_blockers)):
            continue
        reasons = [*match.unresolved_requirements, *match.recommendation_blockers]
        if match.support_status.value != "sufficient":
            reasons.append(f"Evidence status: {match.support_status.value}.")
        if match.support_status.value == "provisional" and match.evidence_coverage < 70:
            reasons.append(f"Evidence coverage is {match.evidence_coverage}%; current support remains provisional.")
        result.append({
            "job": match.job.model_dump(mode="json"), "score": match.score,
            "evidence_coverage": match.evidence_coverage, "support_status": match.support_status.value,
            "career_direction": {"status": match.career_direction_status, "evidence": match.career_direction_evidence},
            "reasons": list(dict.fromkeys(reasons)),
            "requirements_to_verify": requirements_to_verify(match.job.description or "", candidate, match.job.title),
        })
    return result


def _match_markdown(index: int, match: JobMatch, candidate: CandidateProfile | None = None) -> list[str]:
    job = match.job
    seniority = next((dim for dim in match.dimensions if dim.name == "seniority_qualifications"), None)
    lines = [
        f"### {index}. {job.title} — {job.company}", "",
        f"- Location: {job.location}{f', {job.country}' if job.country else ''}",
        f"- Source: [{job.source}]({job.source_url})" if job.source_url else f"- Source: {job.source} (URL not provided)",
        f"- Score: {match.score}/100" if match.score is not None else "- Score: unavailable",
        f"- Matching evidence coverage: {match.evidence_coverage}% of evaluated signals",
        f"- Career-direction alignment: {match.career_direction_status} ({'; '.join(match.career_direction_evidence) or 'not assessed'})",
        f"- Work model: {job.work_model.value}",
        f"- Geographic status: {match.geographic_status.value if match.geographic_status else 'not required'}",
        f"- Strengths: {'; '.join(match.strengths[:4]) or 'none recorded'}",
        f"- Potential gaps: {'; '.join(match.gaps[:4]) or 'none recorded'}",
        f"- Seniority: {'; '.join(seniority.evidence) if seniority and seniority.evidence else 'unknown or unsupported'}",
        f"- Uncertainties: {'; '.join(match.unresolved_requirements) or 'none recorded'}",
    ]
    requirements = requirements_to_verify(job.description or "", candidate, job.title)
    if requirements:
        lines.append("- Requirements to verify:")
        for requirement in requirements:
            lines.append(
                f"  - {requirement['requirement']} ({requirement['requirement_type']}; "
                f"candidate evidence: {requirement['candidate_evidence_status']}). "
                f"{requirement['candidate_evidence']}"
            )
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
    if scope.get("allow_live_market_intelligence"):
        modes.append("live Pinecone market retrieval")
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
