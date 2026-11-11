"""Run the deterministic, offline local knowledge-base audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.knowledge_audit import KnowledgeAuditError, audit_knowledge_base  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "knowledge_base" / "processed")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "knowledge_base" / "review")
    args = parser.parse_args(argv)
    try:
        result = audit_knowledge_base(args.processed_dir, args.review_dir)
    except KnowledgeAuditError as exc:
        print(f"Knowledge-base audit failed: {exc}", file=sys.stderr)
        return 1

    print("Mode: offline deterministic (no credentials or network requests)")
    print(f"Chunks analyzed: {result['total_chunk_count']}")
    print(f"Chunks with blocking quality issues: {result['total_blocking_chunk_count']}")
    print(f"Quality warnings: {result['total_warning_count']}")
    print(f"Proposed shortlist: {result['selected_candidate_count']}")
    print(f"Approved entries in shortlist: {result['selected_approved_count']}")
    print(f"Pending entries in shortlist: {result['selected_pending_count']}")
    print(f"Rights-blocked entries in shortlist: {result['selected_rights_blocked_count']}")
    print("Candidates by source:")
    for document_id, count in result["selected_counts_by_document"].items():
        print(f"  {document_id}: {count}")
    print(f"Report: {args.review_dir / 'knowledge_base_audit.md'}")
    print(f"Machine shortlist: {args.review_dir / 'knowledge_base_shortlist.json'}")
    for warning in result["approval_warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
