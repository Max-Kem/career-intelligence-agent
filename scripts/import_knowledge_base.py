"""Review or import the explicitly approved ILO pilot chunks. Dry-run is default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.knowledge_import import (  # noqa: E402
    ImportPipelineError,
    prepare_ilo_review,
    run_import,
)

PROCESSED_DIR = ROOT / "knowledge_base" / "processed"
REVIEW_DIR = ROOT / "knowledge_base" / "review"
APPROVAL_FILE = REVIEW_DIR / "ilo_pilot_approvals.json"


def _print_plan(plan: dict) -> None:
    print(f"Mode: {plan['mode']}")
    print(f"Candidates: {plan['candidate_count']}")
    print(f"Approved and valid: {plan['approved_count']}")
    print(f"Rejected or pending: {plan['rejected_count']}")
    print(f"Estimated tokens for approved chunks: {plan['estimated_approved_tokens']:,}")
    print(f"Estimated embedding cost for approved chunks: ${plan['estimated_approved_cost_usd']:.8f}")
    print(f"Estimated tokens for full pilot if approved: {plan['estimated_full_pilot_tokens']:,}")
    print(f"Estimated embedding cost for full pilot: ${plan['estimated_full_pilot_cost_usd']:.8f}")
    print(f"Estimate uses approximately 1 token per 4 characters and $0.02 per 1M input tokens; actual token use and pricing may vary.")
    print(f"Target: index={plan['index']}; namespace={plan['namespace']}; model={plan['embedding_model']} ({plan['embedding_dimension']} dimensions)")
    if plan.get("mode") == "dry_run" and not plan["approved_count"]:
        print("No records would be uploaded. Review the pending allowlist and approve chunks individually first.")
    if plan.get("mode") == "live":
        print(f"Upserted records: {plan['upserted_count']}")
    for warning in plan["warnings"]:
        print(f"Warning: {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Embed and upload approved chunks only.")
    mode.add_argument("--prepare-review", action="store_true", help="Create pending-only review files; no network access.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--approval-file", type=Path, default=APPROVAL_FILE)
    args = parser.parse_args(argv)

    try:
        if args.prepare_review:
            approval, summary, count = prepare_ilo_review(args.processed_dir, args.review_dir)
            print(f"Prepared {count} pending review candidates.")
            print(f"Approval file: {approval}")
            print(f"Human-readable summary: {summary}")
            print("No candidates were approved and no network requests were made.")
            return 0
        if args.live:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
        plan = run_import(args.processed_dir, args.approval_file, live=args.live)
        _print_plan(plan)
    except ImportPipelineError as exc:
        print(f"Import pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
