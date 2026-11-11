"""Exact-manifest approval and controlled multi-source knowledge import."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .knowledge_audit import _read_json, _read_chunks
from .knowledge_import import ALREADY_INDEXED_CHUNK_IDS
from .refined_rag_selection import _normalise_whitespace
from .vector_store import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)

# Keep the index contract aligned with the existing v10 Documents API importer.
from .vector_store import KNOWLEDGE_NAMESPACE, PINECONE_INDEX, VECTOR_FIELD, SmokeTestError, create_openai_client, create_pinecone_client, embed_texts


class BatchImportError(RuntimeError):
    """Manifest, approval, source validation, or external API error."""


INDEXED_ILO_IDS = ALREADY_INDEXED_CHUNK_IDS
RISK_ISSUES = {"probable_broken_words", "replacement_or_null_character", "references_only_suspected", "very_short_text"}
EMBEDDING_USD_PER_MILLION_TOKENS = 0.02

# These are the specific records from the reviewed 34-record proposal that
# cannot be repaired into a coherent, useful citation unit.
CURATED_EXCLUSIONS = {
    "cedefop_skills_forecast_germany_2025-p0010-c0002-cf3601a58713": "join_combines_qualification_forecast_with_unrelated_method_explanation",
    "cedefop_skills_forecast_germany_2025-p0013-c0001-2b42695886de": "methodology_not_career_evidence",
    "cedefop_skills_forecast_germany_2025-p0013-c0002-e1a236999c92": "broken_unrelated_methodology_fragment",
    "eu_digital_decade_germany_2026-p0004-c0002-0bd5987025cf": "misaligned_kpi_table_extraction",
    "eu_digital_decade_germany_2026-p0005-c0003-a9ea07dc3607": "join_contains_unrelated_sections_and_repeated_context",
    "eu_digital_decade_germany_2026-p0011-c0003-4eaeddc5ee83": "join_contains_unrelated_broadband_and_semiconductor_detail",
    "eu_digital_decade_germany_2026-p0012-c0003-3f3937e57b7f": "join_contains_unrelated_semiconductor_and_edge_node_sections",
    "eu_digital_decade_germany_2026-p0013-c0002-09db14dd04fc": "join_contains_unrelated_quantum_infrastructure_detail",
    "eu_digital_decade_germany_2026-p0014-c0002-4485c9bdc36d": "join_contains_unrelated_quantum_and_sme_sections",
    "eu_digital_decade_germany_2026-p0016-c0001-02987d73d055": "join_ends_with_incomplete_sentence",
    "eu_digital_decade_germany_2026-p0017-c0001-06aa5c0021d7": "fragmented_ai_adoption_passage",
    "eu_digital_decade_germany_2026-p0017-c0002-fdd2a7c2cb20": "fragmented_ai_adoption_passage",
    "eu_digital_decade_germany_2026-p0017-c0003-5c8fd8e066e1": "fragmented_ai_adoption_passage",
    "eu_digital_decade_germany_2026-p0021-c0002-2b9e78d2eccf": "join_ends_with_incomplete_sentence",
    "eu_digital_decade_germany_2026-p0023-c0001-31aa10283757": "fragmented_education_policy_passage",
    "eu_digital_decade_germany_2026-p0023-c0002-cbb5c909a6a1": "fragmented_ict_specialist_passage",
    "eu_digital_decade_germany_2026-p0024-c0003-3d6379005dda": "join_combines_workforce_policy_with_unrelated_public_services_metrics",
    "eu_digital_decade_germany_2026-p0026-c0001-868dc68ef773": "digital_public_service_architecture_not_career_evidence",
    "eu_digital_decade_germany_2026-p0028-c0002-28f332b57fd7": "unrelated_patient_record_and_greening_sections",
    "eu_digital_decade_germany_2026-p0029-c0001-72290f1ab639": "annex_roadmap_not_career_evidence",
    "eu_digital_decade_germany_2026-p0031-c0001-b97cd95dbe37": "macroeconomic_modelling_not_career_evidence",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _overlap_join(texts: list[str]) -> tuple[str, int]:
    """Join same-page source chunks after removing an exact suffix/prefix overlap."""
    if len(texts) != 2:
        raise BatchImportError("Only two adjacent source chunks can be joined safely.")
    left, right = (_normalise_whitespace(text) for text in texts)
    left_tokens = list(re.finditer(r"\S+", left))
    right_tokens = list(re.finditer(r"\S+", right))
    def comparable(token: str) -> str:
        return re.sub(r"[^\w]+", "", token.casefold())
    overlap = 0
    for count in range(min(len(left_tokens), len(right_tokens)), 7, -1):
        if [comparable(m.group()) for m in left_tokens[-count:]] == [comparable(m.group()) for m in right_tokens[:count]]:
            overlap = count
            break
    if overlap < 8:
        raise BatchImportError("Adjacent chunks do not have a verified overlap of at least eight words.")
    remainder = right[right_tokens[overlap - 1].end():].lstrip()
    return _normalise_whitespace(f"{left} {remainder}"), overlap


def _approval_map(review_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for filename in ("ilo_pilot_approvals.json", "multi_source_pilot_approvals.json"):
        path = review_dir / filename
        if path.exists():
            artifact = _read_json(path, "Approval artifact")
            for row in artifact.get("candidates", []):
                if isinstance(row, dict) and isinstance(row.get("chunk_id"), str):
                    result[row["chunk_id"]] = row
    return result


def _risk_reasons(candidate: dict[str, Any], related_quality_codes: set[str] | None = None) -> list[str]:
    codes = set(candidate.get("quality_issue_codes", []))
    related_quality_codes = related_quality_codes or set()
    reasons = sorted(codes & RISK_ISSUES)
    if candidate.get("unresolved_boundary_codes"):
        reasons.append("unresolved_sentence_boundary")
    if candidate.get("document_id") == "ilo_generative_ai_jobs_2025" and "chart_or_table_residue" in codes:
        reasons.append("table_or_figure_extraction_not_prioritized_for_narrative_batch")
    text = candidate.get("candidate_text", "")
    if candidate.get("document_id") == "ilo_generative_ai_jobs_2025":
        if "chart_or_table_residue" in related_quality_codes:
            reasons.append("derived_text_includes_ilo_table_or_figure_residue")
        if "task justification" in str(text).casefold() or ("task" in str(text).casefold() and "justification" in str(text).casefold() and len(re.findall(r"\b0\.\d+\b", str(text))) >= 2):
            reasons.append("ilo_table_extraction_not_prioritized_for_narrative_batch")
    if not isinstance(text, str) or not text.strip():
        reasons.append("empty_candidate_text")
    # Exclude substantial verbatim quotations; short quoted terms/citations are retained.
    quote_pairs = (("“", "”"), ("‘", "’"), ('"', '"'))
    for opening, closing in quote_pairs:
        start = text.find(opening)
        if start >= 0:
            end = text.find(closing, start + len(opening))
            if end - start - len(opening) >= 240:
                reasons.append("substantial_quotation_needs_separate_rights_review")
                break
    return sorted(set(reasons))


def _rights_basis(document_id: str, document: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rights = document.get("rights_preflight") or {}
    if document_id == "ilo_generative_ai_jobs_2025":
        assumption = "ILO-owned text under the PDF's CC BY 4.0 notice; separately protected third-party content is excluded."
    elif document_id == "cedefop_skills_forecast_germany_2025":
        assumption = "Cedefop's attributed-reproduction policy, subject to document-specific restrictions and separate third-party rights."
    else:
        assumption = "European Commission CC BY 4.0 default applies only to EU-owned content; separately protected third-party content is excluded."
    return {
        "assumption": assumption,
        "manifest_status": document.get("licensing_status"),
        "policy_source_url": document.get("license_source_url"),
        "document_conditions": rights.get("license_or_reuse_conditions"),
        "third_party_status": rights.get("third_party_rights"),
        "specific_chunk_review": candidate.get("third_party_review", "pending"),
    }


def _verify_sources(candidate: dict[str, Any], source_index: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    derived_from = candidate.get("derived_from")
    if not isinstance(derived_from, list) or not derived_from:
        raise BatchImportError(f"Candidate {candidate.get('chunk_id')} has no source fingerprint list.")
    verified: list[dict[str, str]] = []
    texts: list[str] = []
    seen: set[str] = set()
    for item in derived_from:
        if not isinstance(item, dict):
            raise BatchImportError("Candidate source fingerprint entry is malformed.")
        chunk_id, digest = item.get("chunk_id"), item.get("text_sha256")
        if not isinstance(chunk_id, str) or chunk_id in seen or not isinstance(digest, str):
            raise BatchImportError(f"Candidate {candidate.get('chunk_id')} has missing or duplicate source IDs.")
        seen.add(chunk_id)
        matches = source_index.get(chunk_id, [])
        if len(matches) != 1:
            raise BatchImportError(f"Source chunk {chunk_id} is missing or duplicated.")
        chunk = matches[0]
        actual = _sha(str(chunk.get("text", "")))
        if not chunk.get("text", "").strip() or digest != actual:
            raise BatchImportError(f"Source chunk fingerprint mismatch for {chunk_id}.")
        if chunk.get("document_id") != candidate.get("document_id"):
            raise BatchImportError(f"Source document mismatch for {chunk_id}.")
        verified.append({"chunk_id": chunk_id, "text_sha256": actual})
        texts.append(chunk["text"])
    if candidate.get("chunk_id") not in seen:
        raise BatchImportError(f"Primary source ID is absent from provenance for {candidate.get('chunk_id')}.")
    if _sha(str(candidate.get("candidate_text", ""))) != candidate.get("derived_text_sha256"):
        raise BatchImportError(f"Derived text fingerprint mismatch for {candidate.get('chunk_id')}.")
    repair_kind = candidate.get("repair_kind")
    if repair_kind == "normalized_layout_whitespace_only":
        expected_text = _normalise_whitespace(texts[0]) if len(texts) == 1 else None
    elif repair_kind and repair_kind.startswith("joined_exact_adjacent_same_page_text"):
        expected_text = _normalise_whitespace(" ".join(texts))
    elif repair_kind is None and len(texts) == 1:
        expected_text = texts[0]
    else:
        expected_text = None
    if expected_text is None or expected_text != candidate.get("candidate_text"):
        raise BatchImportError(f"Derived text does not reproduce from the verified source chunks: {candidate.get('chunk_id')}.")
    return verified


def _make_record(candidate: dict[str, Any], document: dict[str, Any], approval: dict[str, Any], source_hashes: list[dict[str, str]]) -> dict[str, Any]:
    text = candidate["candidate_text"]
    record = {
        "record_id": candidate["chunk_id"],
        "document_id": candidate["document_id"],
        "document_title": document.get("title"),
        "extracted_chunk_title": candidate.get("extracted_chunk_title"),
        "publisher": document.get("publisher"),
        "publication_year": document.get("publication_year"),
        "source_url": document.get("source_url"),
        "source_filename": document.get("filename"),
        "page_start": candidate.get("page_start", candidate.get("page_number")),
        "page_end": candidate.get("page_end", candidate.get("page_number")),
        "source_chunk_ids": [item["chunk_id"] for item in source_hashes],
        "source_text_fingerprints": source_hashes,
        "candidate_text": text,
        "candidate_text_sha256": candidate["derived_text_sha256"],
        "repair_kind": candidate.get("repair_kind"),
        "overlap_removed_word_count": candidate.get("overlap_removed_word_count", 0),
        "geographic_scope": candidate.get("document_scope"),
        "attribution": (document.get("rights_preflight") or {}).get("attribution_requirements"),
        "license": document.get("license"),
        "rights_basis": _rights_basis(candidate["document_id"], document, candidate),
        "quality_warning_codes": candidate.get("quality_issue_codes", []),
        "prior_individual_approval": {
            "approved": candidate.get("approved", False),
            "reviewer": approval.get("reviewer"),
            "approved_at": approval.get("approved_at"),
            "third_party_review": approval.get("third_party_review", "pending"),
            "text_sha256": approval.get("text_sha256"),
        } if candidate.get("approved") else None,
        "batch_approval_status": "pending_owner_approval",
    }
    record["record_fingerprint"] = _fingerprint(record)
    return record


def build_batch_manifest(processed_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    processed, review = Path(processed_dir), Path(review_dir)
    selection = _read_json(review / "refined_rag_selection.json", "Refined RAG selection")
    source_manifest = _read_json(processed / "manifest.json", "Processed manifest")
    documents = {row.get("document_id"): row for row in source_manifest.get("documents", []) if isinstance(row, dict)}
    source_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document_id in documents:
        for chunk in _read_chunks(processed / f"{document_id}.chunks.jsonl"):
            source_index[chunk.get("chunk_id")].append(chunk)
    approvals = _approval_map(review)
    selection_by_id = {row.get("chunk_id"): row for row in selection.get("selected_candidates", []) if isinstance(row, dict)}
    previous_path = review / "rag_batch_manifest.json"
    previous = _read_json(previous_path, "Existing batch manifest") if previous_path.exists() else {}
    scope_ids = set(previous.get("review_scope_record_ids", [])) or {row.get("record_id") for row in previous.get("records", [])}
    # The batch correction is deliberately constrained to the records already
    # presented for review. Previously blocked/indexed/excluded accounting is retained.
    candidates_to_process = [row for row in selection.get("selected_candidates", []) if isinstance(row, dict) and (not scope_ids or row.get("chunk_id") in scope_ids)]
    included: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = [row for row in previous.get("other_exclusions", []) if row.get("chunk_id") not in scope_ids]
    previously_indexed: list[str] = list(previous.get("previously_indexed_excluded", []))
    blocked: list[dict[str, Any]] = list(previous.get("rights_blocked_records", []))
    for candidate in candidates_to_process:
        chunk_id, document_id = candidate.get("chunk_id"), candidate.get("document_id")
        if document_id == "ba_ikt_arbeitsmarkt_2026":
            blocked.append({"chunk_id": chunk_id, "document_id": document_id, "reason": "reuse_policy_unresolved; no external processing"})
            continue
        if document_id == "ilo_generative_ai_jobs_2025" and chunk_id in INDEXED_ILO_IDS:
            if chunk_id not in previously_indexed:
                previously_indexed.append(chunk_id)
            continue
        related_codes = {
            code
            for source in candidate.get("derived_from", [])
            for code in selection_by_id.get(source.get("chunk_id"), {}).get("quality_issue_codes", [])
        }
        risk_reasons = _risk_reasons(candidate, related_codes)
        source_hashes = _verify_sources(candidate, source_index)
        if document_id == "ilo_generative_ai_jobs_2025" and set(item["chunk_id"] for item in source_hashes) & INDEXED_ILO_IDS:
            risk_reasons.append("would_reupload_previously_indexed_ilo_content")
        if risk_reasons:
            exclusions.append({"chunk_id": chunk_id, "document_id": document_id, "page_number": candidate.get("page_number"), "reasons": sorted(set(risk_reasons))})
            continue
        candidate = dict(candidate)
        if len(source_hashes) == 2:
            source_texts = [source_index[item["chunk_id"]][0]["text"] for item in source_hashes]
            repaired_text, overlap_words = _overlap_join(source_texts)
            candidate["candidate_text"] = repaired_text
            candidate["derived_text_sha256"] = _sha(repaired_text)
            candidate["repair_kind"] = "joined_exact_adjacent_same_page_text_overlap_removed"
            candidate["overlap_removed_word_count"] = overlap_words
        curated_reason = CURATED_EXCLUSIONS.get(chunk_id)
        if curated_reason:
            exclusions.append({"chunk_id": chunk_id, "document_id": document_id, "page_number": candidate.get("page_number"), "reasons": [curated_reason]})
            continue
        document = documents.get(document_id)
        if not document or not candidate.get("candidate_text") or not candidate.get("source_url"):
            raise BatchImportError(f"Required source metadata is missing for {chunk_id}.")
        approval = approvals.get(chunk_id, {})
        if candidate.get("approved") and approval.get("text_sha256") != source_hashes[0]["text_sha256"] and chunk_id in source_index:
            # For a single-source approved record, the individual approval must match its original text.
            if len(source_hashes) == 1:
                exclusions.append({"chunk_id": chunk_id, "document_id": document_id, "page_number": candidate.get("page_number"), "reasons": ["prior_approval_fingerprint_mismatch"]})
                continue
        included.append(_make_record(candidate, document, approval, source_hashes))

    # Suppress overlapping derived candidates, preferring the candidate that
    # preserves the most exact adjacent source text and then the higher score.
    ordered = sorted(included, key=lambda row: (-len(row["source_chunk_ids"]), -next((c.get("relevance_score", 0) for c in selection["selected_candidates"] if c.get("chunk_id") == row["record_id"]), 0), row["document_id"], row["record_id"]))
    deduplicated: list[dict[str, Any]] = []
    used_source_ids: set[str] = set()
    used_text_hashes: set[str] = set()
    for row in ordered:
        if set(row["source_chunk_ids"]) & used_source_ids or row["candidate_text_sha256"] in used_text_hashes:
            exclusions.append({"chunk_id": row["record_id"], "document_id": row["document_id"], "page_number": row["page_start"], "reasons": ["overlapping_or_duplicate_derived_content"]})
            continue
        deduplicated.append(row)
        used_source_ids.update(row["source_chunk_ids"])
        used_text_hashes.add(row["candidate_text_sha256"])

    deduplicated.sort(key=lambda row: (row["document_id"], row["page_start"], row["record_id"]))
    # A stable batch ID binds the record IDs and fingerprints; full manifest hash
    # additionally binds rights assumptions, exclusions, and destination.
    batch_id = _fingerprint([{"record_id": r["record_id"], "record_fingerprint": r["record_fingerprint"]} for r in deduplicated])[:20]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "batch_id": batch_id,
        "status": "pending_owner_approval",
        "destination": {"index": PINECONE_INDEX, "namespace": KNOWLEDGE_NAMESPACE, "vector_field": VECTOR_FIELD, "embedding_model": EMBEDDING_MODEL, "embedding_dimension": EMBEDDING_DIMENSION},
        "rights_policy": {
            "ilo": "CC BY 4.0 for ILO-owned material; third-party content excluded; existing indexed chunks never reuploaded.",
            "cedefop": "Attributed reproduction policy, subject to document-specific restrictions and separate third-party rights.",
            "european_commission": "CC BY 4.0 default for EU-owned content; protected third-party material excluded.",
            "ba": "Blocked pending reuse-policy clarification.",
            "batch_approval": "Required for every proposed record; individual approvals are not extended to derived text.",
        },
        "proposed_record_count": len(deduplicated),
        "refined_selection_count": selection.get("selected_candidate_count", len(selection.get("selected_candidates", []))),
        "overlap_repaired_record_count": sum(bool(row.get("overlap_removed_word_count")) for row in deduplicated),
        "review_scope_record_ids": sorted(scope_ids or {row["record_id"] for row in deduplicated}),
        "records": deduplicated,
        "previously_indexed_excluded": previously_indexed,
        "rights_blocked_records": blocked,
        "other_exclusions": sorted(exclusions, key=lambda row: (row["document_id"], row["chunk_id"])),
    }
    manifest["batch_id"] = batch_id
    manifest["proposed_counts_by_source"] = dict(sorted(Counter(row["document_id"] for row in deduplicated).items()))
    manifest["excluded_counts_by_reason"] = dict(sorted(Counter(reason for row in manifest["other_exclusions"] for reason in row["reasons"]).items()))
    # Counts are part of the approved manifest and therefore included in its digest.
    manifest["manifest_sha256"] = _fingerprint({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return manifest


def write_batch_artifacts(processed_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    review = Path(review_dir)
    manifest = build_batch_manifest(processed_dir, review)
    review.mkdir(parents=True, exist_ok=True)
    (review / "rag_batch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (review / "rag_batch_summary.md").write_text(render_batch_summary(manifest), encoding="utf-8")
    return manifest


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return _fingerprint({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def _load_manifest(review_dir: str | Path) -> dict[str, Any]:
    path = Path(review_dir) / "rag_batch_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise BatchImportError("Batch manifest is missing or invalid. Run --prepare-batch first.") from None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise BatchImportError("Batch manifest has an unsupported schema.")
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise BatchImportError("Batch manifest fingerprint mismatch; regenerate and review the batch.")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("proposed_record_count"):
        raise BatchImportError("Batch record count is invalid.")
    if not records:
        raise BatchImportError("Batch manifest contains no proposed records.")
    ids = [row.get("record_id") for row in records]
    if len(ids) != len(set(ids)):
        raise BatchImportError("Batch manifest contains duplicate record IDs.")
    for row in records:
        expected = row.get("record_fingerprint")
        if not isinstance(expected, str) or expected != _fingerprint({key: value for key, value in row.items() if key != "record_fingerprint"}):
            raise BatchImportError(f"Record fingerprint mismatch: {row.get('record_id')}.")
        sources = row.get("source_text_fingerprints")
        source_ids = row.get("source_chunk_ids")
        if not isinstance(source_ids, list) or not source_ids or not isinstance(sources, list) or [source.get("chunk_id") for source in sources if isinstance(source, dict)] != source_ids:
            raise BatchImportError(f"Source provenance is malformed for {row.get('record_id')}.")
        required = ("document_title", "publisher", "publication_year", "source_url", "source_filename", "attribution", "candidate_text", "candidate_text_sha256", "geographic_scope")
        if any(row.get(key) in (None, "") for key in required) or not str(row.get("source_url", "")).startswith(("https://", "http://")):
            raise BatchImportError(f"Required citation metadata is missing for {row.get('record_id')}.")
        if not isinstance(row.get("page_start"), int) or not isinstance(row.get("page_end"), int) or row["page_start"] < 1 or row["page_end"] < row["page_start"]:
            raise BatchImportError(f"Page references are invalid for {row.get('record_id')}.")
        if row.get("record_id") not in source_ids or row.get("batch_approval_status") != "pending_owner_approval":
            raise BatchImportError(f"Record identity or pending approval status is invalid for {row.get('record_id')}.")
        if row.get("document_id") == "ba_ikt_arbeitsmarkt_2026" or row.get("record_id") in INDEXED_ILO_IDS:
            raise BatchImportError("Manifest includes a prohibited BA or already-indexed ILO record.")
        if row.get("document_id") == "ilo_generative_ai_jobs_2025" and set(row.get("source_chunk_ids", [])) & INDEXED_ILO_IDS:
            raise BatchImportError("Manifest would reupload text from an already-indexed ILO record.")
        if _sha(str(row.get("candidate_text", ""))) != row.get("candidate_text_sha256"):
            raise BatchImportError(f"Candidate text fingerprint mismatch: {row.get('record_id')}.")
    return manifest


def approve_batch(review_dir: str | Path, *, reviewer: str, confirmation_digest: str, approved_at: str | None = None, processed_dir: str | Path | None = None) -> dict[str, Any]:
    manifest = _load_manifest(review_dir)
    if processed_dir is not None:
        _source_hashes_current(manifest, processed_dir)
    if not reviewer.strip():
        raise BatchImportError("Reviewer is required.")
    if confirmation_digest.strip().lower() != manifest["manifest_sha256"]:
        raise BatchImportError("Confirmation does not match the exact batch manifest SHA-256.")
    record = {
        "schema_version": 1,
        "approval_status": "approved",
        "batch_id": manifest["batch_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "reviewer": reviewer.strip(),
        "approved_at": approved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "records": [{"record_id": row["record_id"], "record_fingerprint": row["record_fingerprint"]} for row in manifest["records"]],
    }
    path = Path(review_dir) / "rag_batch_approval.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def _validate_batch_approval(review_dir: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(review_dir)
    path = Path(review_dir) / "rag_batch_approval.json"
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise BatchImportError("No valid batch approval exists; no external calls were made.") from None
    expected = [{"record_id": row["record_id"], "record_fingerprint": row["record_fingerprint"]} for row in manifest["records"]]
    if approval.get("approval_status") != "approved" or not approval.get("reviewer") or not approval.get("approved_at"):
        raise BatchImportError("Batch approval is incomplete; no external calls were made.")
    if approval.get("batch_id") != manifest["batch_id"] or approval.get("manifest_sha256") != manifest["manifest_sha256"] or approval.get("records") != expected:
        raise BatchImportError("Batch approval does not match current manifest contents; no external calls were made.")
    return manifest


def validate_batch_for_import(processed_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    """Validate exact batch approval and current local source contents, without API access."""
    manifest = _validate_batch_approval(review_dir)
    _source_hashes_current(manifest, processed_dir)
    return manifest


def validate_batch_sources(processed_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    """Check a pending manifest against its current local source files."""
    manifest = _load_manifest(review_dir)
    _source_hashes_current(manifest, processed_dir)
    return manifest


def _source_hashes_current(manifest: dict[str, Any], processed_dir: str | Path) -> None:
    processed = Path(processed_dir)
    try:
        source_manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise BatchImportError("Source document manifest is missing or invalid.") from None
    documents = {row.get("document_id"): row for row in source_manifest.get("documents", []) if isinstance(row, dict)}
    chunks_by_doc: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for document_id in {row["document_id"] for row in manifest["records"]}:
        items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in _read_chunks(processed / f"{document_id}.chunks.jsonl"):
            items[chunk.get("chunk_id")].append(chunk)
        chunks_by_doc[document_id] = items
    for record in manifest["records"]:
        document = documents.get(record["document_id"])
        if not document:
            raise BatchImportError(f"Source document disappeared: {record['document_id']}.")
        for record_field, manifest_field in (
            ("document_title", "title"), ("publisher", "publisher"),
            ("publication_year", "publication_year"), ("source_url", "source_url"),
            ("source_filename", "filename"),
        ):
            if record.get(record_field) != document.get(manifest_field):
                raise BatchImportError(f"Source metadata changed after batch review: {record_field} for {record['record_id']}.")
        source_chunks: list[dict[str, Any]] = []
        for fingerprint in record["source_text_fingerprints"]:
            matches = chunks_by_doc[record["document_id"]].get(fingerprint["chunk_id"], [])
            if len(matches) != 1 or _sha(str(matches[0].get("text", ""))) != fingerprint["text_sha256"]:
                raise BatchImportError(f"Source chunk changed after review: {fingerprint['chunk_id']}.")
            source_chunks.append(matches[0])
        repair_kind = record.get("repair_kind")
        if repair_kind == "normalized_layout_whitespace_only" and len(source_chunks) == 1:
            expected_text = _normalise_whitespace(source_chunks[0]["text"])
        elif repair_kind == "joined_exact_adjacent_same_page_text_overlap_removed" and len(source_chunks) == 2:
            expected_text, overlap_words = _overlap_join([chunk["text"] for chunk in source_chunks])
            if overlap_words != record.get("overlap_removed_word_count"):
                raise BatchImportError(f"Recorded overlap repair changed for {record['record_id']}.")
        elif repair_kind and repair_kind.startswith("joined_exact_adjacent_same_page_text"):
            expected_text = _normalise_whitespace(" ".join(chunk["text"] for chunk in source_chunks))
        elif repair_kind is None and len(source_chunks) == 1:
            expected_text = source_chunks[0].get("text")
        else:
            expected_text = None
        if expected_text is None or expected_text != record["candidate_text"]:
            raise BatchImportError(f"Candidate text does not reproduce from its approved source chunks: {record['record_id']}.")
        pages = [chunk.get("page_number") for chunk in source_chunks]
        if any(not isinstance(page, int) for page in pages) or min(pages) != record["page_start"] or max(pages) != record["page_end"]:
            raise BatchImportError(f"Source page references changed after review: {record['record_id']}.")


def run_batch_import(processed_dir: str | Path, review_dir: str | Path, *, live: bool = False, openai_api_key: str | None = None, pinecone_api_key: str | None = None, openai_client: Any = None, pinecone_client: Any = None) -> dict[str, Any]:
    manifest = _load_manifest(review_dir)
    _source_hashes_current(manifest, processed_dir)
    approval: dict[str, Any] | None = None
    try:
        approval_path = Path(review_dir) / "rag_batch_approval.json"
        if approval_path.exists():
            approval = _validate_batch_approval(review_dir)
    except BatchImportError:
        if live:
            raise
    records = manifest["records"]
    token_estimate = sum(math.ceil(len(row["candidate_text"]) / 4) for row in records)
    plan = {
        "mode": "live" if live else "dry_run",
        "included_count": len(records),
        "previously_indexed_count": len(manifest["previously_indexed_excluded"]),
        "rights_blocked_count": len(manifest["rights_blocked_records"]),
        "excluded_count": len(manifest["other_exclusions"]),
        "estimated_tokens": token_estimate,
        "estimated_embedding_cost_usd": token_estimate * EMBEDDING_USD_PER_MILLION_TOKENS / 1_000_000,
        "index": PINECONE_INDEX,
        "namespace": KNOWLEDGE_NAMESPACE,
        "batch_id": manifest["batch_id"],
        "batch_approval_valid": approval is not None,
    }
    if not live:
        return plan
    manifest = validate_batch_for_import(processed_dir, review_dir)
    approval_metadata = json.loads((Path(review_dir) / "rag_batch_approval.json").read_text(encoding="utf-8"))
    if not records:
        raise BatchImportError("Approved batch is empty; no external calls were made.")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if not openai_api_key or not openai_api_key.strip() or not pinecone_api_key or not pinecone_api_key.strip():
        raise BatchImportError("Required OpenAI or Pinecone credential is missing; no external calls were made.")
    try:
        openai_client = openai_client or create_openai_client(openai_api_key)
        pinecone_client = pinecone_client or create_pinecone_client(pinecone_api_key)
        vectors = embed_texts(openai_client, [row["candidate_text"] for row in records])
        if len(vectors) != len(records) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise BatchImportError(f"Embedding output must contain {len(records)} vectors of dimension {EMBEDDING_DIMENSION}.")
        documents = []
        for row, vector in zip(records, vectors, strict=True):
            documents.append({
                "_id": row["record_id"],
                VECTOR_FIELD: vector,
                "chunk_id": row["record_id"],
                "chunk_text": row["candidate_text"],
                "document_id": row["document_id"],
                "document_title": row["document_title"],
                "publisher": row["publisher"],
                "publication_year": row["publication_year"],
                "source_url": row["source_url"],
                "source_filename": row["source_filename"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "geographic_scope": row["geographic_scope"],
                "source_chunk_ids": row["source_chunk_ids"],
                "source_text_fingerprints": json.dumps(row["source_text_fingerprints"], ensure_ascii=False),
                "derived_text_sha256": row["candidate_text_sha256"],
                "attribution": row["attribution"],
                "license": row["license"] or "publisher policy; see rights_basis",
                "rights_basis": json.dumps(row["rights_basis"], ensure_ascii=False),
                "approved_by": approval_metadata["reviewer"],
                "approved_at": approval_metadata["approved_at"],
                "rights_status": "exact batch manifest approved by project owner; source-level terms in rights_basis",
            })
        index = pinecone_client.index(PINECONE_INDEX)
        response = index.documents.upsert(namespace=KNOWLEDGE_NAMESPACE, documents=documents)
    except BatchImportError:
        raise
    except SmokeTestError as exc:
        raise BatchImportError(str(exc)) from None
    except Exception as exc:
        raise BatchImportError(f"Live batch embedding or Pinecone import failed ({type(exc).__name__}); credentials and response payloads are not shown.") from None
    plan["upserted_count"] = response.get("upserted_count", len(documents)) if isinstance(response, dict) else getattr(response, "upserted_count", len(documents))
    return plan


def render_batch_summary(manifest: dict[str, Any]) -> str:
    token_estimate = sum(math.ceil(len(row["candidate_text"]) / 4) for row in manifest["records"])
    cost = token_estimate * EMBEDDING_USD_PER_MILLION_TOKENS / 1_000_000
    counts = manifest["proposed_counts_by_source"]
    lines = [
        "# Proposed Multi-Source RAG Batch",
        "",
        "**Status: pending explicit project-owner approval. No records have been uploaded.**",
        "",
        f"- Batch ID: `{manifest['batch_id']}`",
        f"- Manifest SHA-256 to approve: `{manifest['manifest_sha256']}`",
        f"- Proposed new records: **{manifest['proposed_record_count']}**",
        f"- Previously indexed ILO records excluded: **{len(manifest['previously_indexed_excluded'])}**",
        f"- BA records blocked: **{len(manifest['rights_blocked_records'])}**",
        f"- Other risk/overlap exclusions: **{len(manifest['other_exclusions'])}**",
        f"- Proposed records with exact overlap repair: **{manifest.get('overlap_repaired_record_count', 0)}**",
        f"- Current refined shortlist reconciliation: {manifest.get('refined_selection_count', 'unknown')} = {manifest['proposed_record_count']} proposed + {len(manifest['previously_indexed_excluded'])} previously indexed + {len(manifest['rights_blocked_records'])} BA-blocked + {len(manifest['other_exclusions'])} other exclusions.",
        "- The earlier 90-candidate figure is stale: the current refined selection artifact contains 81. The nine are absent from the current source-of-truth selection, not silently omitted from this batch.",
        f"- Estimated tokens: **{token_estimate:,}** (~1 token per 4 characters)",
        f"- Estimated embedding cost: **${cost:.8f}** at ${EMBEDDING_USD_PER_MILLION_TOKENS:.2f}/1M input tokens; estimate only.",
        "",
        "## Proposed records by source",
        "",
        "| Source | New records |",
        "|---|---:|",
    ]
    for doc_id, count in counts.items():
        lines.append(f"| {doc_id} | {count} |")
    lines.extend(["", "## Rights assumptions and exceptions", ""])
    lines.extend(f"- **{key}:** {value}" for key, value in manifest["rights_policy"].items())
    lines.extend([
        "- Cedefop page 9 retains its individual owner approval and has a chunk-specific third-party review clearance based on the completed page review; this does not clear any other Cedefop passage or source material.",
        "- Commission report passages are treated as EU-owned report narrative under its CC BY default only where no substantial third-party quotation, graphic, or table was identified. Citations to external studies are retained and are not by themselves treated as rights blockers.",
        "- Every new record still requires explicit batch approval. Approval binds IDs, exact candidate text fingerprints, source chunk fingerprints, attribution, destination and exclusions.",
        "",
        "## Exclusions",
        "",
        f"- BA candidate records blocked by unresolved reuse-policy scope: {len(manifest['rights_blocked_records'])}.",
        f"- Existing ILO records excluded from reupload: {len(manifest['previously_indexed_excluded'])}.",
    ])
    lines.extend(f"- `{row['chunk_id']}` ({row['document_id']}): {', '.join(row['reasons'])}" for row in manifest["other_exclusions"])
    lines.extend([
        "",
        "## Review and import procedure",
        "",
        "1. Inspect this summary and every exact record, text, citation, rights note and fingerprint in `rag_batch_manifest.json`.",
        "2. Explicitly approve the reviewed manifest: `python scripts/import_knowledge_base.py --approve-batch --reviewer project_owner`. The CLI prints the manifest SHA-256 and asks you to type the full value. It records the reviewer and UTC timestamp locally.",
        "3. Run the dry-run: `python scripts/import_knowledge_base.py --batch-dry-run`.",
        "4. Only after review, run live upload: `python scripts/import_knowledge_base.py --live-batch`.",
        "",
        "The importer revalidates batch approval, manifest and per-record fingerprints, source IDs, source text hashes, BA exclusion and the seven indexed ILO IDs before any API call. Any manifest change invalidates approval.",
        "",
    ])
    return "\n".join(lines)
