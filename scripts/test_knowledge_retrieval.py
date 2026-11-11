"""Dry-run by default; search imported multi-source labour-market knowledge read-only with --live."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.vector_store import SmokeTestError, run_knowledge_retrieval  # noqa: E402

DEFAULT_QUERY = "Why does AI exposure not necessarily mean job replacement?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call OpenAI and Pinecone for read-only retrieval.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question to retrieve relevant imported knowledge-base evidence for.")
    args = parser.parse_args(argv)
    if args.live:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    try:
        result = run_knowledge_retrieval(args.query, live=args.live)
    except SmokeTestError as exc:
        print(f"Knowledge retrieval failed: {exc}", file=sys.stderr)
        return 1

    print(f"Mode: {result['mode']}")
    print(f"Index: {result['index']}")
    print(f"Namespace: {result['namespace']}")
    print(f"Question: {result['query']}")
    if result["mode"] == "dry_run":
        print(result["message"])
        print("Planned search: OpenAI text-embedding-3-small to Pinecone Documents API, top 3.")
        return 0
    if not result["matches"]:
        print("No matching documents were returned.")
        return 0

    print("Similarity scores indicate relative semantic closeness, not factual confidence or correctness.")
    for number, match in enumerate(result["matches"], start=1):
        page = match["page_number"]
        if page == "Not provided" and match.get("page_start") is not None:
            page = str(match["page_start"]) if match.get("page_end") in (None, match["page_start"]) else f"{match['page_start']}-{match['page_end']}"
        print(f"\n{number}. Similarity score: {match['score'] if match['score'] is not None else 'Not provided'}")
        print(f"   Chunk: {match['chunk_id']}")
        print(f"   Source: {match['document_title']} — {match['publisher']} ({match['publication_year']}), PDF page {page}")
        print(f"   URL: {match['source_url']}")
        print(f"   Excerpt: {match['excerpt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
