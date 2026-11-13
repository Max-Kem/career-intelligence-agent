"""Run the production grounded synthesis against one saved report.

This script makes no external requests unless --allow-live-openai is present.
It never runs discovery, retrieval, or any other workflow node.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.candidate import CandidateProfile  # noqa: E402
from career_agent.intelligence_synthesis import (  # noqa: E402
    MODEL,
    REQUEST_TIMEOUT_SECONDS,
    prepare_synthesis_input,
    run_grounded_synthesis,
)
from career_agent.vector_store import create_openai_client  # noqa: E402


class OneRequestCompletions:
    """Guard the live client so this command cannot make more than one request."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.request_count = 0
        self.response: Any = None

    def create(self, **kwargs: Any) -> Any:
        if self.request_count >= 1:
            raise RuntimeError("OpenAI request limit reached")
        self.request_count += 1
        self.response = self._delegate.create(**kwargs)
        return self.response


def _as_match(record: dict[str, Any]) -> Any:
    """Adapt an already serialized report match to the synthesis input shape."""
    job = record.get("job") or {}
    return SimpleNamespace(
        job=SimpleNamespace(
            title=str(job.get("title") or ""),
            company=str(job.get("company") or ""),
            description=job.get("description") if isinstance(job.get("description"), str) else "",
        ),
        score=record.get("score"),
        strengths=record.get("strengths", []),
        gaps=record.get("gaps", []),
    )


def _response_usage(response: Any) -> dict[str, int | None] | None:
    usage = getattr(response, "usage", None) if response is not None else None
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _validation_status(result: dict[str, Any]) -> tuple[str, str]:
    stage = result.get("failure_stage")
    if result.get("status") == "succeeded":
        return "passed", "passed"
    if stage == "citation_validation":
        return "passed", "failed"
    if stage == "schema_validation":
        return "failed", "not reached"
    return "not reached", "not reached"


def run_saved_report(
    report_path: Path,
    *,
    dotenv_loader: Callable[..., Any],
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Load one report, perform at most one synthesis request, and return safe results."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = CandidateProfile.model_validate(report["candidate"])
        market = report.get("market_intelligence") or {}
        recommendations = [
            _as_match(item) for item in report.get("recommendations", []) if isinstance(item, dict)
        ]
        review_matches = [
            item for item in report.get("potential_matches_requiring_review", []) if isinstance(item, dict)
        ]
        context, sources = prepare_synthesis_input(market, candidate, recommendations, review_matches)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return 2, {"error": f"Could not prepare saved report ({type(exc).__name__}).", "request_count": 0}

    if not sources:
        return 2, {"error": "Saved report has no complete passages with valid source references.", "request_count": 0}

    # Credentials are loaded only after CLI opt-in. The explicit path avoids
    # accidentally loading a .env from the caller's current directory.
    dotenv_loader(dotenv_path=ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return 2, {"error": "OPENAI_API_KEY is missing; no request was made.", "request_count": 0}

    model = os.getenv("CAREER_INTELLIGENCE_MODEL") or MODEL
    client_factory = client_factory or create_openai_client
    try:
        openai_client = client_factory(
            api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0,
        )
    except Exception as exc:
        return 2, {
            "error": f"OpenAI client initialization failed ({type(exc).__name__}); no request was made.",
            "model": model, "request_count": 0,
        }

    guarded = OneRequestCompletions(openai_client.chat.completions)
    client = SimpleNamespace(chat=SimpleNamespace(completions=guarded))
    result = run_grounded_synthesis(
        market, candidate, recommendations, review_matches, model=model, client=client,
    )
    pydantic_status, citation_status = _validation_status(result)
    allowed_references = context["allowed_references"]
    citation_maps = {
        "market_source": set(allowed_references["market_source"]),
        "candidate_profile": set(allowed_references["candidate_profile"]),
        "job_posting": set(allowed_references["job_posting"]),
    }
    citations = [
        citation
        for section in ("market_intelligence", "personalized_career_intelligence")
        for paragraph in result.get(section, [])
        for citation in paragraph.get("citations", [])
    ]
    citation_resolves = bool(citations) and all(
        citation.get("reference_id") in citation_maps.get(citation.get("kind"), set())
        for citation in citations
    )
    cited_refs = sorted({f"{citation.get('kind')}:{citation.get('reference_id')}" for citation in citations})
    cited_chunk_ids = sorted({
        evidence_id
        for section in ("market_intelligence", "personalized_career_intelligence")
        for paragraph in result.get(section, [])
        for evidence_id in paragraph.get("evidence_ids", [])
    })
    outcome = {
        **result,
        "request_count": guarded.request_count,
        "request_limit": 1,
        "request_completed": guarded.response is not None,
        "finish_reason": (
            getattr(guarded.response.choices[0], "finish_reason", None)
            if guarded.response is not None and getattr(guarded.response, "choices", None) else None
        ),
        "token_usage": _response_usage(guarded.response),
        "pydantic_validation": pydantic_status,
        "citation_validation": citation_status,
    "all_cited_ids_resolve": citation_resolves if citations else "not applicable",
    "cited_reference_ids": cited_refs,
    "cited_market_chunk_ids": cited_chunk_ids,
        "input_counts": {
            "recommendations": len(recommendations),
            "potential_matches_requiring_review": len(review_matches),
            "usable_source_passages": len(sources),
        },
    }
    return (0 if result.get("status") == "succeeded" else 1), outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to a saved workflow JSON report.")
    parser.add_argument(
        "--allow-live-openai", action="store_true",
        help="Explicitly permit one OpenAI synthesis request; no other services are used.",
    )
    args = parser.parse_args(argv)
    if not args.allow_live_openai:
        parser.error("--allow-live-openai is required; no credentials or network will be used without it.")

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Status: failed\nSafe error: python-dotenv is not installed; no request was made.")
        return 2

    exit_code, result = run_saved_report(report_path, dotenv_loader=load_dotenv)
    print(f"Synthesis status: {result.get('status', 'failed')}")
    print(f"Model: {result.get('model', os.getenv('CAREER_INTELLIGENCE_MODEL') or MODEL)}")
    print(f"OpenAI requests: {result.get('request_count', 0)} (maximum 1)")
    print(f"Request completed: {result.get('request_completed', False)}")
    print(f"Finish reason: {result.get('finish_reason') or 'not available'}")
    print(f"Pydantic validation: {result.get('pydantic_validation', 'not reached')}")
    print(f"Citation validation: {result.get('citation_validation', 'not reached')}")
    print(f"All cited IDs resolve to retrieved passages: {result.get('all_cited_ids_resolve', 'not applicable')}")
    print(f"Token usage: {result.get('token_usage') or 'not available'}")
    print(f"Failure stage: {result.get('failure_stage') or 'none'}")
    print(f"Safe error: {result.get('error') or 'none'}")
    if result.get("input_counts"):
        counts = result["input_counts"]
        print(
            "Input: "
            f"{counts['recommendations']} recommendations, "
            f"{counts['potential_matches_requiring_review']} potential matches, "
            f"{counts['usable_source_passages']} source passages"
        )

    for key, label in (
        ("market_intelligence", "Market Intelligence"),
        ("personalized_career_intelligence", "Personalized Career Intelligence"),
    ):
        paragraphs = result.get(key, [])
        if paragraphs:
            print(f"\n{label}")
        for paragraph in paragraphs:
            print(paragraph["text"])
            for citation in paragraph.get("citations", []):
                if citation.get("kind") == "market_source":
                    print(
                        "  Market source: "
                        f"{citation.get('document_title', 'Title not provided')} — "
                        f"{citation.get('publisher', 'Publisher not provided')} "
                        f"({citation.get('publication_year', 'Year not provided')}), "
                        f"PDF page {citation.get('page_number', 'not provided')}, "
                        f"{citation.get('geographic_scope', 'scope not verified')} — "
                        f"{citation.get('source_url', 'URL not provided')} "
                        f"[Chunk: {citation.get('evidence_id', 'ID not provided')}; ref {citation.get('reference_id')}]"
                    )
                elif citation.get("kind") == "job_posting":
                    print(
                        "  Job posting: "
                        f"{citation.get('document_title', 'Title not provided')} — "
                        f"{citation.get('company') or 'Company not provided'} — "
                        f"{citation.get('source_url') or 'URL unavailable'} "
                        f"[Ref {citation.get('reference_id')}]"
                    )
                elif citation.get("kind") == "candidate_profile":
                    print(f"  Candidate profile: {citation.get('label')} [Ref {citation.get('reference_id')}]")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
