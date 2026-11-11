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
from career_agent.knowledge_batch import (  # noqa: E402
    BatchImportError,
    approve_batch,
    run_batch_import,
    validate_batch_for_import,
    validate_batch_sources,
    write_batch_artifacts,
)
from career_agent.knowledge_audit import KnowledgeAuditError  # noqa: E402

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
    mode.add_argument("--prepare-batch", action="store_true", help="Build the proposed multi-source batch; no network access.")
    mode.add_argument("--approve-batch", action="store_true", help="Approve the exact batch manifest after checksum confirmation.")
    mode.add_argument("--batch-dry-run", action="store_true", help="Validate the batch without credentials or network calls.")
    mode.add_argument("--live-batch", action="store_true", help="Embed and upload the explicitly approved multi-source batch.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--approval-file", type=Path, default=APPROVAL_FILE)
    parser.add_argument("--reviewer", help="Required identity for explicit batch approval.")
    args = parser.parse_args(argv)

    try:
        if args.prepare_review:
            approval, summary, count = prepare_ilo_review(args.processed_dir, args.review_dir)
            print(f"Prepared {count} pending review candidates.")
            print(f"Approval file: {approval}")
            print(f"Human-readable summary: {summary}")
            print("No candidates were approved and no network requests were made.")
            return 0
        if args.prepare_batch:
            manifest = write_batch_artifacts(args.processed_dir, args.review_dir)
            print(f"Prepared {manifest['proposed_record_count']} proposed records; batch status remains pending approval.")
            print(f"Manifest SHA-256: {manifest['manifest_sha256']}")
            print(f"Summary: {args.review_dir / 'rag_batch_summary.md'}")
            print(f"Manifest: {args.review_dir / 'rag_batch_manifest.json'}")
            print("No approvals were created and no network requests were made.")
            return 0
        if args.approve_batch:
            if not args.reviewer:
                raise BatchImportError("--reviewer is required with --approve-batch.")
            validate_batch_sources(args.processed_dir, args.review_dir)
            import json

            try:
                manifest = json.loads((args.review_dir / "rag_batch_manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise BatchImportError("Batch manifest is missing or invalid; run --prepare-batch first.") from None
            print(f"Manifest SHA-256 to approve: {manifest.get('manifest_sha256', 'missing')}")
            print("Approve only after reviewing rag_batch_summary.md and every record in rag_batch_manifest.json.")
            try:
                typed_digest = input("Type the full manifest SHA-256 to record approval: ").strip()
            except EOFError:
                raise BatchImportError("Interactive checksum confirmation is required.") from None
            approval = approve_batch(args.review_dir, reviewer=args.reviewer, confirmation_digest=typed_digest, processed_dir=args.processed_dir)
            print(f"Recorded approval by {approval['reviewer']} at {approval['approved_at']}.")
            print("No external API calls were made.")
            return 0
        if args.batch_dry_run:
            plan = run_batch_import(args.processed_dir, args.review_dir, live=False)
            print(f"Mode: {plan['mode']} (no credentials or network requests)")
            print(f"Included: {plan['included_count']}; previously indexed excluded: {plan['previously_indexed_count']}; rights-blocked: {plan['rights_blocked_count']}; other excluded: {plan['excluded_count']}")
            print(f"Batch approval valid: {plan['batch_approval_valid']}")
            print(f"Estimated tokens: {plan['estimated_tokens']:,}; estimated embedding cost: ${plan['estimated_embedding_cost_usd']:.8f}")
            print(f"Target: index={plan['index']}; namespace={plan['namespace']}")
            return 0
        if args.live_batch:
            validate_batch_for_import(args.processed_dir, args.review_dir)
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
            plan = run_batch_import(args.processed_dir, args.review_dir, live=True)
            print(f"Mode: {plan['mode']}; uploaded {plan['upserted_count']} records to {plan['index']} / {plan['namespace']}.")
            print(f"Batch ID: {plan['batch_id']}")
            return 0
        if args.live:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
        plan = run_import(args.processed_dir, args.approval_file, live=args.live)
        _print_plan(plan)
    except (ImportPipelineError, BatchImportError, KnowledgeAuditError, OSError) as exc:
        print(f"Import pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
