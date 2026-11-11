"""Create the offline refined RAG candidate review artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.refined_rag_selection import refine_knowledge_base  # noqa: E402
from career_agent.knowledge_audit import KnowledgeAuditError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "knowledge_base" / "processed")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "knowledge_base" / "review")
    args = parser.parse_args(argv)
    try:
        result = refine_knowledge_base(args.processed_dir, args.review_dir)
    except (KnowledgeAuditError, OSError) as exc:
        print(f"Refined RAG selection failed: {exc}", file=sys.stderr)
        return 1
    print("Mode: offline deterministic (no credentials or network requests)")
    print(f"Chunks evaluated: {result['total_chunks_analyzed']}")
    print(f"Review candidates: {result['selected_candidate_count']}")
    print(f"Excluded: {result['excluded_chunk_count']}")
    print(f"Candidates repaired by adjacent-text combination: {result['repaired_candidate_count']}")
    print(f"Candidates with whitespace-only normalization: {result['whitespace_normalized_candidate_count']}")
    print(f"Import eligible now: {result['import_eligible_count']}")
    print("Candidates by source:")
    for document_id, count in result["selected_counts_by_document"].items():
        print(f"  {document_id}: {count}")
    print(f"Markdown review: {args.review_dir / 'refined_rag_selection.md'}")
    print(f"JSON selection: {args.review_dir / 'refined_rag_selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
