"""Extract local PDF pages to ignored JSONL files and inspect the manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.document_extraction import extract_directory, inspect_processed  # noqa: E402
from career_agent.document_chunking import chunk_processed_directory  # noqa: E402
from career_agent.document_rights import update_manifest_rights_preflight  # noqa: E402


def _print_manifest(manifest: dict) -> None:
    print(f"Documents: {manifest.get('document_count', 0)}")
    for document in manifest.get("documents", []):
        print(
            f"- {document['filename']}: {document['extraction_status']}; "
            f"pages={document.get('page_count')}; "
            f"characters={document.get('extracted_character_count', 0)}; "
            f"empty/unreadable={len(document.get('empty_or_unreadable_pages', []))}; "
            f"sparse-under-100={len(document.get('sparse_text_pages_under_100_chars', []))}"
        )
        if document.get("page_issues"):
            pages = [str(issue.get("page_number", "document")) for issue in document["page_issues"]]
            print(f"  Page issues: {', '.join(pages)}")
    print("Page text is stored in the processed JSONL files and is not printed by this command.")


def _print_chunking(summary: dict) -> None:
    print(f"Documents processed: {summary['documents_processed']}")
    for document in summary["chunks_per_document"]:
        print(f"- {document['document_id']}: {document['chunk_count']} chunks ({document['chunk_file'] or 'no output'})")
    print(
        f"Total chunks: {summary['total_chunks']}; average size: {summary['average_chunk_size']} chars; "
        f"maximum size: {summary['maximum_chunk_size']} chars"
    )
    print(f"Skipped empty sections: {summary['skipped_empty_sections']}")
    print(f"Quality warnings: {len(summary['warnings'])}")
    for warning in summary["warnings"][:10]:
        print(f"Warning: {warning}")
    if len(summary["warnings"]) > 10:
        print(f"... {len(summary['warnings']) - 10} additional warnings are in chunking_summary.json")
    print("Chunk text is stored in ignored JSONL files and is not printed by this command.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser("extract", help="Extract all PDFs page by page.")
    extract_parser.add_argument("--raw-dir", default=str(ROOT / "knowledge_base" / "raw"))
    extract_parser.add_argument("--output-dir", default=str(ROOT / "knowledge_base" / "processed"))
    inspect_parser = commands.add_parser("inspect", help="Print manifest and extraction diagnostics only.")
    inspect_parser.add_argument("--output-dir", default=str(ROOT / "knowledge_base" / "processed"))
    chunk_parser = commands.add_parser("chunk", help="Create deterministic page-level chunks from extracted text.")
    chunk_parser.add_argument("--processed-dir", default=str(ROOT / "knowledge_base" / "processed"))
    chunk_parser.add_argument("--target-chars", type=int, default=1300)
    chunk_parser.add_argument("--overlap-chars", type=int, default=180)
    chunk_parser.add_argument("--short-chunk-chars", type=int, default=200)
    rights_parser = commands.add_parser("preflight", help="Update manifest rights and bibliographic metadata only.")
    rights_parser.add_argument("--processed-dir", default=str(ROOT / "knowledge_base" / "processed"))
    args = parser.parse_args()

    try:
        if args.command == "extract":
            manifest = extract_directory(args.raw_dir, args.output_dir)
            _print_manifest(manifest)
            return 1 if any(doc.get("extraction_status") == "error" for doc in manifest.get("documents", [])) else 0
        if args.command == "inspect":
            manifest = inspect_processed(args.output_dir)
            _print_manifest(manifest)
            return 0
        if args.command == "preflight":
            manifest = update_manifest_rights_preflight(args.processed_dir)
            print(f"Rights preflight recorded for {manifest.get('document_count', 0)} documents.")
            for document in manifest.get("documents", []):
                rights = document.get("rights_preflight", {})
                print(f"- {document.get('document_id')}: {document.get('publication_year')}; {document.get('licensing_status')}; {rights.get('external_embedding_status', 'not reviewed')}")
            print("Only manifest metadata was updated; page text and chunk JSONL files were not changed.")
            return 0
        summary = chunk_processed_directory(
            args.processed_dir,
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
            short_chunk_chars=args.short_chunk_chars,
        )
        _print_chunking(summary)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Document processing command failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
