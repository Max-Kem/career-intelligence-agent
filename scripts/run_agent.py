"""Run the Career Intelligence research-to-report workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.workflow import build_workflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("a", "b"), default="a")
    parser.add_argument("--query", help="Optional single search query override.")
    parser.add_argument("--queries", type=int, default=3, help="Maximum distinct role queries (1-3).")
    parser.add_argument("--pages", type=int, default=1, help="Maximum pages/provider/query (1-5).")
    parser.add_argument("--top-n", type=int, default=5, help="Maximum recommendations (1-5).")
    parser.add_argument("--output-dir", default="reports", help="Directory for Markdown and JSON reports.")
    parser.add_argument("--offline", action="store_true", help="Use normalized fixture jobs; do not call APIs.")
    parser.add_argument(
        "--allow-live-esco", action="store_true",
        help="With --offline, allow bounded live ESCO lookups for selected fixture-job recommendations.",
    )
    parser.add_argument(
        "--allow-live-company-research", action="store_true",
        help="With --offline, allow bounded live Tavily research for selected fixture-job recommendations.",
    )
    parser.add_argument("--no-esco", action="store_true", help="Skip ESCO lookups.")
    parser.add_argument("--no-company-research", action="store_true", help="Disable Tavily company research.")
    parser.add_argument("--esco-language", default="en", help="Language for ESCO concept labels (default: en).")
    args = parser.parse_args()
    if args.allow_live_esco and not args.offline:
        parser.error("--allow-live-esco requires --offline")
    if args.allow_live_company_research and not args.offline:
        parser.error("--allow-live-company-research requires --offline")
    if args.allow_live_company_research and args.no_company_research:
        parser.error("--allow-live-company-research cannot be combined with --no-company-research")
    if args.allow_live_esco and args.no_esco:
        parser.error("--allow-live-esco cannot be combined with --no-esco")
    if not 1 <= args.queries <= 3:
        parser.error("--queries must be between 1 and 3")
    if not 1 <= args.pages <= 5:
        parser.error("--pages must be between 1 and 5")
    if not 1 <= args.top_n <= 5:
        parser.error("--top-n must be between 1 and 5")

    result = build_workflow().invoke({
        "profile_name": args.profile,
        "query_override": args.query,
        "query_limit": args.queries,
        "max_pages_per_provider": args.pages,
        "top_n": args.top_n,
        "output_dir": args.output_dir,
        "offline": args.offline,
        "allow_live_esco": args.allow_live_esco,
        "esco_enabled": not args.no_esco,
        "company_research_enabled": not args.no_company_research,
        "allow_live_company_research": args.allow_live_company_research,
        "esco_language": args.esco_language,
    })
    print(f"Workflow status: {result.get('workflow_status', 'unknown')}")
    print(f"Queries: {', '.join(result.get('search_queries', [])) or 'none'}")
    print(f"Jobs discovered: {len(result.get('jobs', []))}")
    print(f"Recommendations: {len(result.get('recommendations', []))}")
    for report_type, path in result.get("report_paths", {}).items():
        print(f"{report_type.title()} report: {path}")
    for warning in result.get("warnings", []):
        print(f"Warning: {warning}")
    for error in result.get("nonfatal_errors", []):
        print(f"Provider error: {error}")
    for error in result.get("errors", []):
        print(f"Error: {error}")
    return 1 if result.get("workflow_status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
