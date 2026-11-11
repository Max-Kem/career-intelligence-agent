"""Conservative review and import pipeline for approved ILO chunks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from career_agent.vector_store import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    KNOWLEDGE_NAMESPACE,
    PINECONE_INDEX,
    VECTOR_FIELD,
    SmokeTestError,
    create_openai_client,
    create_pinecone_client,
    embed_texts,
)

ILO_DOCUMENT_ID = "ilo_generative_ai_jobs_2025"
ILO_CHUNK_FILE = f"{ILO_DOCUMENT_ID}.chunks.jsonl"
ILO_LICENSE = "CC BY 4.0"
MAX_LIVE_CHUNKS = 20
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4
EMBEDDING_USD_PER_MILLION_TOKENS = 0.02
EMBEDDING_PRICE_SOURCE = "https://developers.openai.com/api/docs/models/text-embedding-3-small"

# Narrative chunks from the abstract, method, findings, interpretation, and conclusion.
# No tables, references, front matter, or appendices are in this proposed set.
PILOT_CHUNK_IDS = (
    "ilo_generative_ai_jobs_2025-p0004-c0001-f760e5798b11",
    "ilo_generative_ai_jobs_2025-p0004-c0002-99017cedf6d6",
    "ilo_generative_ai_jobs_2025-p0012-c0002-72fad690a1c1",
    "ilo_generative_ai_jobs_2025-p0013-c0001-05121427d2cb",
    "ilo_generative_ai_jobs_2025-p0014-c0001-38187f3a7d1b",
    "ilo_generative_ai_jobs_2025-p0017-c0001-831c02a7b20b",
    "ilo_generative_ai_jobs_2025-p0030-c0002-27b346b7f2f6",
    "ilo_generative_ai_jobs_2025-p0040-c0001-30ce52836d24",
    "ilo_generative_ai_jobs_2025-p0041-c0001-dd3c1d2cfe8a",
    "ilo_generative_ai_jobs_2025-p0041-c0003-4c72cf311482",
    "ilo_generative_ai_jobs_2025-p0043-c0001-bc102c020a99",
    "ilo_generative_ai_jobs_2025-p0044-c0002-cfebec11d60f",
    "ilo_generative_ai_jobs_2025-p0044-c0003-0567f4019514",
    "ilo_generative_ai_jobs_2025-p0047-c0003-1ce0ddcafa11",
    "ilo_generative_ai_jobs_2025-p0047-c0004-37fb75e5020c",
    "ilo_generative_ai_jobs_2025-p0050-c0002-6b02906aa496",
    "ilo_generative_ai_jobs_2025-p0050-c0003-e2a70bcc99a3",
)


class ImportPipelineError(RuntimeError):
    """Invalid approval, source data, credentials, or external API failure."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ImportPipelineError(f"Chunk data line {line_number} is not a JSON object.")
            rows.append(row)
    except FileNotFoundError:
        raise ImportPipelineError(f"Chunk data file is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ImportPipelineError(f"Malformed chunk JSONL at line {exc.lineno} in {path.name}.") from None
    return rows


def _load_source(processed_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = Path(processed_dir)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ImportPipelineError("Processed manifest is missing or invalid.") from None
    matches = [doc for doc in manifest.get("documents", []) if doc.get("document_id") == ILO_DOCUMENT_ID]
    if len(matches) != 1:
        raise ImportPipelineError("Manifest must contain exactly one ILO 2025 document entry.")
    chunk_file = directory / ILO_CHUNK_FILE
    if chunk_file.resolve().parent != directory.resolve():
        raise ImportPipelineError("Chunk file reference must stay inside the processed directory.")
    return matches[0], _jsonl(chunk_file)


def _attribution(document: dict[str, Any]) -> str:
    return (
        "International Labour Office, Generative AI and Jobs: A Refined Global Index of Occupational Exposure, "
        "ILO Working Paper 140, Geneva, 2025. CC BY 4.0; indicate changes. Third-party material is excluded."
    )


def _document_reuse_basis(document: dict[str, Any]) -> bool:
    rights = document.get("rights_preflight") or {}
    return (
        document.get("license") == ILO_LICENSE
        and document.get("licensing_status") == "verified_in_pdf"
        and rights.get("document_notice_status") == "confirmed_in_pdf"
        and rights.get("document_notice_page") == 2
    )


def _load_approval(approval_path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ImportPipelineError(f"Approval file not found: {approval_path}. Run --prepare-review first.") from None
    except json.JSONDecodeError:
        raise ImportPipelineError("Approval file contains invalid JSON.") from None
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("candidates"), list):
        raise ImportPipelineError("Approval file has an unsupported or malformed structure.")
    return data


def prepare_ilo_review(
    processed_dir: str | Path,
    review_dir: str | Path,
) -> tuple[Path, Path, int]:
    """Create pending-only review data and a local text summary; never approve entries."""
    document, chunks = _load_source(processed_dir)
    if not _document_reuse_basis(document):
        raise ImportPipelineError("ILO document lacks the expected verified CC BY 4.0 document notice in its manifest.")
    source_by_id: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        source_by_id.setdefault(chunk.get("chunk_id"), []).append(chunk)
    missing = [chunk_id for chunk_id in PILOT_CHUNK_IDS if len(source_by_id.get(chunk_id, [])) != 1]
    if missing:
        raise ImportPipelineError("Pilot source IDs missing or duplicated in chunk data: " + ", ".join(missing))

    directory = Path(review_dir)
    directory.mkdir(parents=True, exist_ok=True)
    approval_path = directory / "ilo_pilot_approvals.json"
    summary_path = directory / "ilo_pilot_review.md"
    if approval_path.exists():
        approval = _load_approval(approval_path)
        approval_by_id = {entry.get("chunk_id"): entry for entry in approval["candidates"] if isinstance(entry, dict)}
    else:
        approval_by_id = {}

    citation = _attribution(document)
    candidates: list[dict[str, Any]] = []
    for chunk_id in PILOT_CHUNK_IDS:
        chunk = source_by_id[chunk_id][0]
        candidate = approval_by_id.get(chunk_id, {})
        candidates.append({
            "chunk_id": chunk_id,
            "document_id": ILO_DOCUMENT_ID,
            "page_number": chunk.get("page_number"),
            "document_title": document.get("title"),
            "publisher": document.get("publisher"),
            "publication_year": document.get("publication_year"),
            "source_url": document.get("source_url"),
            "source_filename": document.get("filename"),
            "license": ILO_LICENSE,
            "attribution": citation,
            "text_sha256": hashlib.sha256(str(chunk.get("text", "")).encode("utf-8")).hexdigest(),
            "approval_status": candidate.get("approval_status", "pending_review"),
            "approved": candidate.get("approved", False) is True,
            "reviewer": candidate.get("reviewer"),
            "approved_at": candidate.get("approved_at"),
            "third_party_review": candidate.get("third_party_review", "pending"),
            "approval_note": candidate.get("approval_note"),
        })
    # On first preparation create the pending allowlist; later runs never overwrite it.
    if not approval_path.exists():
        approval_path.write_text(json.dumps({
            "schema_version": 1,
            "document_id": ILO_DOCUMENT_ID,
            "document_title": document.get("title"),
            "publisher": document.get("publisher"),
            "publication_year": document.get("publication_year"),
            "source_url": document.get("source_url"),
            "license": ILO_LICENSE,
            "attribution": citation,
            "approval_instructions": "Per chunk, explicitly set approval_status='approved', approved=true, reviewer, approved_at, and third_party_review='cleared' only after reviewing the source chunk and any third-party rights. Leave all others pending or mark rejected.",
            "candidates": candidates,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_review_summary(summary_path, candidates, source_by_id)
    return approval_path, summary_path, len(candidates)


def _write_review_summary(path: Path, candidates: list[dict[str, Any]], source_by_id: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# ILO 2025 Pinecone Pilot Chunk Review",
        "",
        "All entries are pending review. No chunk is approved by default. Review the complete source passage and rights status before editing the allowlist.",
        "",
        "To approve a chunk, edit `ilo_pilot_approvals.json`: set `approval_status` to `approved`, `approved` to `true`, add `reviewer` and `approved_at`, and set `third_party_review` to `cleared` only after checking the passage. Keep all other entries pending or mark them rejected.",
        "",
        f"Proposed candidates: {len(candidates)} | Approved: {sum(c['approved'] and c['approval_status'] == 'approved' for c in candidates)} | Pending/rejected: {sum(not (c['approved'] and c['approval_status'] == 'approved') for c in candidates)}",
        "",
        "The PDF states CC BY 4.0 on page 2. Third-party content is excluded from that license and requires separate review. This list proposes narrative passages; the human reviewer must still check each passage and any embedded rights issue.",
        "",
    ]
    for candidate in candidates:
        chunk = source_by_id[candidate["chunk_id"]][0]
        text = chunk.get("text", "")
        lines.extend([
            f"## Page {candidate['page_number']} — {candidate['chunk_id']}",
            "",
            f"- Status: `{candidate['approval_status']}`; `approved={str(candidate['approved']).lower()}`; third-party review: `{candidate['third_party_review']}`",
            f"- Document: {candidate['document_title']} ({candidate['publication_year']}) — {candidate['publisher']}",
            f"- Source: {candidate['source_url']}",
            f"- Attribution: {candidate['attribution']}",
            "",
            text,
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_import_plan(processed_dir: str | Path, approval_path: str | Path) -> dict[str, Any]:
    """Validate allowlist entries against manifest and current source chunks."""
    document, chunks = _load_source(processed_dir)
    approval = _load_approval(approval_path)
    if approval.get("document_id") != ILO_DOCUMENT_ID:
        raise ImportPipelineError("Approval file document_id does not match the ILO pilot source.")
    if not _document_reuse_basis(document):
        raise ImportPipelineError("ILO reuse basis is missing or unverified; no chunks may be imported.")
    source_by_id: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        source_by_id.setdefault(chunk.get("chunk_id"), []).append(chunk)
    approval_entries = approval["candidates"]
    approval_id_counts: dict[str, int] = {}
    for entry in approval_entries:
        if isinstance(entry, dict) and isinstance(entry.get("chunk_id"), str):
            approval_id_counts[entry["chunk_id"]] = approval_id_counts.get(entry["chunk_id"], 0) + 1

    required_document_fields = ("title", "publisher", "publication_year", "source_url")
    document_missing = [field for field in required_document_fields if document.get(field) in (None, "")]
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in approval_entries:
        if not isinstance(entry, dict):
            rejected.append({"chunk_id": None, "reasons": ["approval_entry_not_object"]})
            continue
        chunk_id = entry.get("chunk_id")
        reasons: list[str] = []
        matches = source_by_id.get(chunk_id, [])
        if not isinstance(chunk_id, str) or not chunk_id:
            reasons.append("missing_chunk_id")
        elif approval_id_counts.get(chunk_id, 0) > 1:
            reasons.append("duplicate_chunk_id_in_approval_file")
        if len(matches) != 1:
            reasons.append("source_chunk_missing_or_duplicate")
        chunk = matches[0] if len(matches) == 1 else None
        if document_missing:
            reasons.append("source_document_missing_metadata:" + ",".join(document_missing))
        if chunk is not None:
            if chunk.get("document_id") != ILO_DOCUMENT_ID or entry.get("document_id") != ILO_DOCUMENT_ID:
                reasons.append("document_id_mismatch")
            if chunk.get("page_number") != entry.get("page_number") or not isinstance(chunk.get("page_number"), int):
                reasons.append("page_reference_mismatch_or_invalid")
            if not isinstance(document.get("source_url"), str) or not document.get("source_url", "").startswith(("https://", "http://")):
                reasons.append("source_url_missing_or_malformed")
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                reasons.append("empty_or_invalid_source_text")
            elif hashlib.sha256(text.encode("utf-8")).hexdigest() != entry.get("text_sha256"):
                reasons.append("source_text_fingerprint_mismatch")
            for approval_field, document_field in (
                ("document_title", "title"), ("publisher", "publisher"),
                ("publication_year", "publication_year"), ("source_url", "source_url"),
                ("source_filename", "filename"),
            ):
                if entry.get(approval_field) != document.get(document_field) or document.get(document_field) in (None, ""):
                    reasons.append(f"metadata_mismatch_or_missing:{approval_field}")
        if entry.get("license") != ILO_LICENSE or entry.get("attribution") != _attribution(document):
            reasons.append("missing_or_invalid_attribution_or_license")
        is_explicitly_approved = entry.get("approved") is True and entry.get("approval_status") == "approved"
        if not is_explicitly_approved:
            reasons.append("not_explicitly_approved")
        if is_explicitly_approved and (not str(entry.get("reviewer", "")).strip() or not str(entry.get("approved_at", "")).strip()):
            reasons.append("approved_entry_missing_reviewer_or_timestamp")
        if is_explicitly_approved and entry.get("third_party_review") != "cleared":
            reasons.append("third_party_rights_review_not_cleared")
        if reasons:
            rejected.append({"chunk_id": chunk_id, "reasons": reasons})
            warnings.extend(f"{chunk_id or 'unknown chunk'}: {reason}" for reason in reasons)
        else:
            approved.append({**entry, "text": chunk["text"]})

    if len(approved) > MAX_LIVE_CHUNKS:
        warnings.append(f"Approved chunk count exceeds the live limit of {MAX_LIVE_CHUNKS}.")
    # Also catch duplicate IDs in the source set, even if a duplicate is not selected.
    duplicate_source_ids = [chunk_id for chunk_id, items in source_by_id.items() if len(items) > 1]
    if duplicate_source_ids:
        warnings.append(f"Source JSONL contains duplicate IDs: {', '.join(duplicate_source_ids)}")
        duplicate_source_set = set(duplicate_source_ids)
        approved = [item for item in approved if item["chunk_id"] not in duplicate_source_set]
    return {
        "candidate_count": len(approval_entries),
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "approved": approved,
        "rejected": rejected,
        "warnings": warnings,
        "document": document,
    }


def _token_estimate(chunks: list[dict[str, Any]]) -> int:
    return sum(math.ceil(len(chunk["text"]) / TOKEN_ESTIMATE_CHARS_PER_TOKEN) for chunk in chunks)


def _estimated_cost(tokens: int) -> float:
    return tokens * EMBEDDING_USD_PER_MILLION_TOKENS / 1_000_000


def run_import(
    processed_dir: str | Path,
    approval_path: str | Path,
    *,
    live: bool = False,
    openai_api_key: str | None = None,
    pinecone_api_key: str | None = None,
    openai_client: Any = None,
    pinecone_client: Any = None,
) -> dict[str, Any]:
    plan = build_import_plan(processed_dir, approval_path)
    if plan["approved_count"] > MAX_LIVE_CHUNKS:
        raise ImportPipelineError(f"Live import is limited to {MAX_LIVE_CHUNKS} approved chunks.")
    approved = plan["approved"]
    # Candidate token budget uses source text, including pending entries, for planning only.
    _, all_source_chunks = _load_source(processed_dir)
    source_by_id = {chunk.get("chunk_id"): chunk for chunk in all_source_chunks}
    approval = _load_approval(approval_path)
    candidate_texts = [source_by_id[e.get("chunk_id")]["text"] for e in approval["candidates"]
                       if isinstance(e, dict) and e.get("chunk_id") in source_by_id]
    candidate_chunks = [{"text": text} for text in candidate_texts]
    plan["estimated_approved_tokens"] = _token_estimate(approved)
    plan["estimated_approved_cost_usd"] = _estimated_cost(plan["estimated_approved_tokens"])
    plan["estimated_full_pilot_tokens"] = _token_estimate(candidate_chunks)
    plan["estimated_full_pilot_cost_usd"] = _estimated_cost(plan["estimated_full_pilot_tokens"])
    plan["embedding_model"] = EMBEDDING_MODEL
    plan["embedding_dimension"] = EMBEDDING_DIMENSION
    plan["index"] = PINECONE_INDEX
    plan["namespace"] = KNOWLEDGE_NAMESPACE
    plan["pricing_source"] = EMBEDDING_PRICE_SOURCE
    if not live:
        plan["mode"] = "dry_run"
        return plan
    if not approved:
        raise ImportPipelineError("No explicitly approved, valid chunks are available; no API calls were made.")
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not pinecone_api_key:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not openai_api_key or not openai_api_key.strip():
        raise ImportPipelineError("OPENAI_API_KEY is missing; no API calls were made.")
    if not pinecone_api_key or not pinecone_api_key.strip():
        raise ImportPipelineError("PINECONE_API_KEY is missing; no API calls were made.")
    try:
        openai_client = openai_client or create_openai_client(openai_api_key)
        pinecone_client = pinecone_client or create_pinecone_client(pinecone_api_key)
        vectors = embed_texts(openai_client, [chunk["text"] for chunk in approved])
        records = []
        for chunk, vector in zip(approved, vectors, strict=True):
            records.append({
                "_id": chunk["chunk_id"],
                VECTOR_FIELD: vector,
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["text"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "publisher": chunk["publisher"],
                "publication_year": chunk["publication_year"],
                "source_url": chunk["source_url"],
                "source_filename": chunk["source_filename"],
                "page_number": chunk["page_number"],
                "attribution": chunk["attribution"],
                "license": chunk["license"],
                "approved_by": chunk["reviewer"],
                "approved_at": chunk["approved_at"],
                "rights_status": "CC BY 4.0; chunk explicitly reviewed and approved; third-party rights review cleared",
            })
        index = pinecone_client.index(PINECONE_INDEX)
        response = index.documents.upsert(namespace=KNOWLEDGE_NAMESPACE, documents=records)
    except SmokeTestError as exc:
        raise ImportPipelineError(str(exc)) from None
    except Exception as exc:
        raise ImportPipelineError(f"Live embedding or Pinecone import failed ({type(exc).__name__}); no credentials or response details shown.") from None
    plan["mode"] = "live"
    plan["upserted_count"] = _get_upserted_count(response, len(records))
    return plan


def _get_upserted_count(response: Any, fallback: int) -> int:
    if isinstance(response, dict):
        return response.get("upserted_count", fallback)
    return getattr(response, "upserted_count", fallback)
