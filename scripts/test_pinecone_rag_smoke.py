"""Dry-run by default; use --live to embed and search synthetic text only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.vector_store import SmokeTestError, run_smoke_test  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call OpenAI and Pinecone using synthetic fixtures only.")
    args = parser.parse_args(argv)
    if args.live:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    try:
        result = run_smoke_test(live=args.live)
    except SmokeTestError as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(f"Mode: {result['mode']}")
    print(f"Namespace: {result['namespace']}")
    if result["mode"] == "dry_run":
        print(result["message"])
        print(f"Synthetic records: {result['record_count']}")
        for record in result["records"]:
            print(f"- {record['document_title']}: {record['chunk_text']}")
        print(f"Query preview: {result['query']}")
        return 0

    print(f"Upserted records: {result['upserted_count']}")
    print(f"Retrieved matches: {len(result['matches'])}")
    for match in result["matches"]:
        print(f"- {match['document_title']} (score={match['score']}, id={match['id']})")
        print(f"  {match['chunk_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
