"""Deterministic page-level chunking for extracted knowledge-base documents."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_TARGET_CHARS = 1300
DEFAULT_OVERLAP_CHARS = 180
DEFAULT_SHORT_CHUNK_CHARS = 200


def chunk_page_text(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    """Split page text on paragraph/sentence/word boundaries with bounded overlap."""
    if target_chars < 100:
        raise ValueError("target_chars must be at least 100")
    if overlap_chars < 0 or overlap_chars >= target_chars // 2:
        raise ValueError("overlap_chars must be non-negative and less than half of target_chars")
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + target_chars, len(text))
        end = hard_end if hard_end == len(text) else _best_chunk_end(text, start, hard_end, target_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = _overlap_start(text, start, end, overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_processed_directory(
    processed_dir: str | Path,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    short_chunk_chars: int = DEFAULT_SHORT_CHUNK_CHARS,
) -> dict[str, Any]:
    """Read the extraction manifest and JSONL files, then overwrite chunk outputs."""
    if target_chars < 100:
        raise ValueError("target_chars must be at least 100")
    if overlap_chars < 0 or overlap_chars >= target_chars // 2:
        raise ValueError("overlap_chars must be non-negative and less than half of target_chars")
    if short_chunk_chars < 1:
        raise ValueError("short_chunk_chars must be positive")

    directory = Path(processed_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Extraction manifest must contain a documents list")

    warnings: list[str] = []
    quality = {
        "empty_chunks": 0,
        "missing_document_references": 0,
        "invalid_page_numbers": 0,
        "duplicate_chunk_ids": 0,
        "oversized_chunks": 0,
        "suspiciously_short_chunks": 0,
        "malformed_page_records": 0,
        "missing_metadata_fields": 0,
    }
    skipped_empty_sections = 0
    chunk_sizes: list[int] = []
    chunks_per_document: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    manifest_ids = {
        document.get("document_id") for document in documents
        if isinstance(document, dict) and isinstance(document.get("document_id"), str)
    }

    for document in documents:
        if not isinstance(document, dict):
            warnings.append("Manifest contains a non-object document entry; skipped it.")
            continue
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            warnings.append("Manifest document is missing document_id; skipped it.")
            continue
        page_file_name = document.get("page_jsonl")
        if not isinstance(page_file_name, str) or not page_file_name:
            warnings.append(f"{document_id}: manifest has no page JSONL reference; no chunks created.")
            chunks_per_document.append({"document_id": document_id, "chunk_count": 0, "chunk_file": None})
            continue
        page_file = directory / page_file_name
        if not page_file.is_file():
            warnings.append(f"{document_id}: page JSONL file is missing: {page_file_name}.")
            chunks_per_document.append({"document_id": document_id, "chunk_count": 0, "chunk_file": None})
            continue

        for field in ("title", "publisher", "publication_year", "filename", "licensing_status"):
            if document.get(field) in (None, ""):
                quality["missing_metadata_fields"] += 1
                warnings.append(f"{document_id}: metadata field '{field}' is unavailable.")
        source_url = document.get("source_url")
        if source_url is not None and (not isinstance(source_url, str) or not source_url.startswith(("http://", "https://"))):
            quality["missing_metadata_fields"] += 1
            warnings.append(f"{document_id}: source_url is malformed; retained as unavailable.")
            source_url = None
        elif source_url is None and document.get("source_url_status") != "unavailable":
            quality["missing_metadata_fields"] += 1
            warnings.append(f"{document_id}: source URL is not available or verified.")

        output_chunks: list[dict[str, Any]] = []
        try:
            page_lines = page_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            warnings.append(f"{document_id}: could not read page JSONL ({type(exc).__name__}).")
            chunks_per_document.append({"document_id": document_id, "chunk_count": 0, "chunk_file": None})
            continue

        for line_number, line in enumerate(page_lines, start=1):
            if not line.strip():
                continue
            try:
                page_record = json.loads(line)
            except json.JSONDecodeError:
                quality["malformed_page_records"] += 1
                warnings.append(f"{document_id}: malformed JSON at page-record line {line_number}; skipped it.")
                continue
            if not isinstance(page_record, dict):
                quality["malformed_page_records"] += 1
                warnings.append(f"{document_id}: non-object page record at line {line_number}; skipped it.")
                continue
            record_document_id = page_record.get("document_id")
            if record_document_id != document_id or record_document_id not in manifest_ids:
                quality["missing_document_references"] += 1
                warnings.append(f"{document_id}: page record {line_number} has a missing or mismatched document reference; skipped it.")
                continue
            page_number = page_record.get("page_number")
            page_count = document.get("page_count")
            if (not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1
                    or (isinstance(page_count, int) and page_number > page_count)):
                quality["invalid_page_numbers"] += 1
                warnings.append(f"{document_id}: invalid page number on record {line_number}; skipped it.")
                continue
            page_text = page_record.get("text")
            if not isinstance(page_text, str):
                quality["malformed_page_records"] += 1
                warnings.append(f"{document_id}: non-text page content on record {line_number}; skipped it.")
                continue
            if not page_text.strip():
                skipped_empty_sections += 1
                continue

            page_chunks = chunk_page_text(page_text, target_chars=target_chars, overlap_chars=overlap_chars)
            for local_index, chunk_text in enumerate(page_chunks, start=1):
                if not chunk_text.strip():
                    quality["empty_chunks"] += 1
                    continue
                chunk_id = _chunk_id(document_id, page_number, local_index, chunk_text)
                if chunk_id in seen_chunk_ids:
                    quality["duplicate_chunk_ids"] += 1
                    warnings.append(f"{document_id}: duplicate chunk ID {chunk_id}; duplicate record skipped.")
                    continue
                seen_chunk_ids.add(chunk_id)
                char_count = len(chunk_text)
                if char_count > target_chars:
                    quality["oversized_chunks"] += 1
                    warnings.append(f"{chunk_id}: chunk exceeds target size ({char_count}>{target_chars}); retained.")
                if char_count < short_chunk_chars:
                    quality["suspiciously_short_chunks"] += 1
                    warnings.append(f"{chunk_id}: short chunk ({char_count} characters); retained for citation fidelity.")
                chunk_sizes.append(char_count)
                output_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "document_title": document.get("title"),
                    "publisher": document.get("publisher"),
                    "publication_year": document.get("publication_year"),
                    "source_url": source_url,
                    "source_filename": document.get("filename"),
                    "page_number": page_number,
                    "page_start": page_number,
                    "page_end": page_number,
                    "text": chunk_text,
                    "character_count": char_count,
                    "licensing_status": document.get("licensing_status", "unverified"),
                    "license": document.get("license"),
                })

        chunk_file_name = f"{document_id}.chunks.jsonl"
        chunk_file = directory / chunk_file_name
        temp_file = directory / f"{chunk_file_name}.tmp"
        temp_file.write_text(
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in output_chunks),
            encoding="utf-8",
        )
        temp_file.replace(chunk_file)
        chunks_per_document.append({
            "document_id": document_id,
            "chunk_count": len(output_chunks),
            "chunk_file": chunk_file_name,
        })
        if document.get("extraction_status") != "succeeded":
            warnings.append(f"{document_id}: source extraction status is {document.get('extraction_status')!r}.")
        if document.get("empty_or_unreadable_pages"):
            warnings.append(f"{document_id}: extraction manifest reports empty or unreadable pages {document['empty_or_unreadable_pages']}.")
        if document.get("sparse_text_pages_under_100_chars"):
            warnings.append(f"{document_id}: extraction manifest reports sparse-text pages {document['sparse_text_pages_under_100_chars']}.")

    total_chunks = len(chunk_sizes)
    summary = {
        "documents_processed": len(documents),
        "chunks_per_document": chunks_per_document,
        "total_chunks": total_chunks,
        "average_chunk_size": round(sum(chunk_sizes) / total_chunks, 2) if total_chunks else 0,
        "maximum_chunk_size": max(chunk_sizes, default=0),
        "target_chunk_size": target_chars,
        "overlap_size": overlap_chars,
        "short_chunk_warning_threshold": short_chunk_chars,
        "skipped_empty_sections": skipped_empty_sections,
        "quality_checks": quality,
        "warnings": warnings,
    }
    summary_path = directory / "chunking_summary.json"
    temp_summary = directory / "chunking_summary.json.tmp"
    temp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_summary.replace(summary_path)
    return summary


def _best_chunk_end(text: str, start: int, hard_end: int, target_chars: int) -> int:
    lower_bound = start + int(target_chars * 0.55)
    for pattern in (r"\n[ \t]*\n", r"(?<=[.!?])\s+", r"\s+"):
        candidates = [match.end() for match in re.compile(pattern).finditer(text, start, hard_end)
                      if match.end() >= lower_bound]
        if candidates:
            return candidates[-1]
    return hard_end


def _overlap_start(text: str, start: int, end: int, overlap_chars: int) -> int:
    desired_start = max(start + 1, end - overlap_chars)
    if overlap_chars == 0:
        return end
    for pattern in (r"\n[ \t]*\n", r"(?<=[.!?])\s+", r"\s+"):
        match = re.search(pattern, text[desired_start:end])
        if match:
            candidate = desired_start + match.end()
            while candidate < end and text[candidate].isspace():
                candidate += 1
            if candidate < end:
                return candidate
    return desired_start


def _chunk_id(document_id: str, page_number: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}-p{page_number:04d}-c{chunk_index:04d}-{digest}"
