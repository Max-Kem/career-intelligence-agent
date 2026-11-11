"""Offline quality-controlled selection of local PDF chunks for human review."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .knowledge_audit import (
    MIN_RELEVANCE_SCORE,
    TOPIC_TERMS,
    KnowledgeAuditError,
    _load_approval_map,
    _near_duplicate_pairs,
    _quality_issues,
    _read_chunks,
    _read_json,
    _rights_gate,
    _topic_match,
)


REJECTED_CHUNK_IDS = {
    "cedefop_skills_forecast_germany_2025-p0007-c0001-b18b78fd75cb":
        "Previously reviewed as a mixed figure/narrative passage with a broken ending; replace rather than reuse.",
}
REJECTED_PAGES = {("cedefop_skills_forecast_germany_2025", 7)}
MIN_STANDALONE_LENGTH = 100
MIN_TOPIC_GROUPS = 2
MIN_SCORE = max(25.0, MIN_RELEVANCE_SCORE)
GLOBAL_DOCUMENTS = {"ilo_generative_ai_jobs_2025"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    return re.sub(r" *\n\n *", "\n\n", text).strip()


def _scope(document_id: str) -> str:
    if document_id in GLOBAL_DOCUMENTS:
        return "international_global"
    if document_id in {
        "ba_ikt_arbeitsmarkt_2026",
        "cedefop_skills_forecast_germany_2025",
        "eu_digital_decade_germany_2026",
    }:
        return "Germany_focused"
    return "unspecified"


def _matches_for_scope(text: str, scope: str) -> dict[str, list[str]]:
    matches = _topic_match(text)
    # The ILO publication is global. Generic employment language in it is not
    # evidence about Germany's labour market.
    if scope == "international_global":
        matches.pop("german_labour_market", None)
    return matches


def _is_title_only(text: str) -> bool:
    clean = text.strip()
    words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
    has_sentence_end = bool(re.search(r"[.!?](?:[\"'”’)]*)\s*$", clean))
    return len(words) <= 12 and not has_sentence_end


def _is_chart_only(text: str) -> bool:
    clean = text.strip()
    if not re.search(r"\b(?:figure|fig\.|table|tabelle|abbildung)\b", clean, re.I):
        return False
    words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
    sentences = re.findall(r"[^.!?]*[.!?]", clean)
    numeric = len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", clean))
    return len(sentences) == 0 or (numeric >= 6 and len(words) < 45)


def _starts_with_fragment(text: str) -> bool:
    clean = text.strip()
    clean = re.sub(r"^\d{1,3}\s+", "", clean)
    clean = re.sub(r"^(?:ILO Working Paper \d+\s+)?(?:X\s+)?", "", clean, flags=re.I)
    first = re.search(r"[A-Za-zÀ-ÿ]", clean)
    return bool(first and first.group(0).islower())


def _is_references_only(text: str) -> bool:
    clean = re.sub(r"^\s*\d{1,3}\s+ILO Working Paper \d+\s+X?\s*", "", text.strip(), flags=re.I)
    if re.match(r"(?:references|bibliography|literature cited|works cited)\b", clean, flags=re.I):
        return True
    entries = re.findall(r"(?m)^\s*[A-Z][A-Za-zÀ-ÿ'’-]+,\s+[A-Z].{0,90}\b(?:19|20)\d{2}\b", clean)
    return len(entries) >= 3 and len(re.findall(r"[.!?]\s", clean)) < len(entries)


def _chunk_order(chunk_id: str) -> tuple[int, int] | None:
    match = re.search(r"-p(\d+)-c(\d+)-[0-9a-fA-F]{12}$", chunk_id)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _derive_candidate_text(
    index: int,
    records: list[dict[str, Any]],
    issue_codes: set[str],
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Normalize layout whitespace and join only exact adjacent same-page text."""
    current = records[index]
    source_text = current.get("text") if isinstance(current.get("text"), str) else ""
    parts: list[tuple[int, dict[str, Any]]] = [(index, current)]

    def safe_neighbor(other_index: int | None) -> bool:
        if other_index is None:
            return False
        neighbor = records[other_index]
        neighbor_text = neighbor.get("text") if isinstance(neighbor.get("text"), str) else ""
        return len(neighbor_text.strip()) >= MIN_STANDALONE_LENGTH and not _is_title_only(neighbor_text) and not _is_chart_only(neighbor_text)

    order = _chunk_order(str(current.get("chunk_id", "")))
    if order:
        page, chunk_no = order
        if "lowercase_sentence_start" in issue_codes and chunk_no > 1:
            previous = next((i for i, row in enumerate(records)
                             if row.get("_source_document_id") == current.get("_source_document_id")
                             and row.get("page_number") == current.get("page_number")
                             and _chunk_order(str(row.get("chunk_id", ""))) == (page, chunk_no - 1)), None)
            if safe_neighbor(previous):
                parts.insert(0, (previous, records[previous]))
        if issue_codes & {"open_sentence_end", "conjunction_boundary"}:
            following = next((i for i, row in enumerate(records)
                              if row.get("_source_document_id") == current.get("_source_document_id")
                              and row.get("page_number") == current.get("page_number")
                              and _chunk_order(str(row.get("chunk_id", ""))) == (page, chunk_no + 1)), None)
            if safe_neighbor(following):
                parts.append((following, records[following]))

    parts.sort(key=lambda item: (_chunk_order(str(item[1].get("chunk_id", ""))) or (0, 0)))
    texts = [row.get("text", "") for _, row in parts if isinstance(row.get("text"), str) and row.get("text", "").strip()]
    combined = _normalise_whitespace(" ".join(texts))
    source_ids = [str(row.get("chunk_id")) for _, row in parts]
    source_hashes = [_sha(str(row.get("text", ""))) for _, row in parts]
    repair_kind = None
    if len(parts) > 1:
        repair_kind = "joined_exact_adjacent_same_page_text; verify sentence context"
    elif combined != source_text:
        repair_kind = "normalized_layout_whitespace_only"
    provenance = [{"chunk_id": cid, "text_sha256": fingerprint} for cid, fingerprint in zip(source_ids, source_hashes)]
    return combined if combined else source_text, provenance, repair_kind


def _exclusion_reasons(record: dict[str, Any]) -> list[str]:
    text = record["candidate_text"]
    reasons: list[str] = []
    if record["chunk_id"] in REJECTED_CHUNK_IDS:
        reasons.append("previously_rejected_candidate")
    if (record["document_id"], record["page_number"]) in REJECTED_PAGES:
        reasons.append("reviewed_rejected_page_no_replacement_approved")
    if len(text.strip()) < MIN_STANDALONE_LENGTH:
        reasons.append("extremely_short")
    if _is_title_only(text):
        reasons.append("title_only_or_fragment")
    issue_codes = set(record["quality_issue_codes"])
    if "references_only_suspected" in issue_codes:
        reasons.append("references_only")
    if _is_references_only(text):
        reasons.append("references_only")
    if _is_chart_only(text):
        reasons.append("chart_or_figure_only")
    if _starts_with_fragment(text):
        reasons.append("leading_sentence_fragment_without_context")
    if record["blocking_issues"]:
        reasons.append("blocking_source_or_metadata_issue")
    if record["unresolved_boundary_codes"]:
        reasons.append("unresolved_sentence_boundary")
    return sorted(set(reasons))


def _import_eligibility(record: dict[str, Any]) -> tuple[str, bool]:
    if record["document_id"] == "ba_ikt_arbeitsmarkt_2026":
        return "blocked_reuse_policy_unresolved", False
    if record["document_id"] == "cedefop_skills_forecast_germany_2025" and record["third_party_review"] != "cleared":
        return "blocked_third_party_review_unresolved", False
    if record["document_id"] == "eu_digital_decade_germany_2026":
        return "pending_individual_and_third_party_review", False
    if record["approved"] and record["document_id"] == "ilo_generative_ai_jobs_2025":
        return "already_indexed_do_not_reupload", False
    if not record["approved"]:
        return "pending_individual_approval", False
    if record["blocking_issues"] or record["exclusion_reasons"]:
        return "blocked_quality_review", False
    return "eligible_after_review", True


def refine_knowledge_base(processed_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    """Evaluate all local chunks and write a human-review selection offline."""
    processed, review = Path(processed_dir), Path(review_dir)
    manifest = _read_json(processed / "manifest.json", "Processed manifest")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise KnowledgeAuditError("Processed manifest must contain a documents list.")
    doc_map = {d.get("document_id"): d for d in documents if isinstance(d, dict) and d.get("document_id")}
    raw: list[dict[str, Any]] = []
    for document_id, document in sorted(doc_map.items()):
        for chunk in _read_chunks(processed / f"{document_id}.chunks.jsonl"):
            chunk["_source_document_id"] = document_id
            raw.append(chunk)
    id_counts = Counter(row.get("chunk_id") for row in raw)
    dup_pairs = _near_duplicate_pairs(raw)
    duplicate_indices: dict[int, list[int]] = defaultdict(list)
    for left, right, _score, _kind in dup_pairs:
        duplicate_indices[left].append(right)
        duplicate_indices[right].append(left)
    approvals, approval_warnings = _load_approval_map(review)
    analysed: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw):
        document_id = chunk.get("_source_document_id")
        document = doc_map[document_id]
        original_text = chunk.get("text") if isinstance(chunk.get("text"), str) else ""
        quality = _quality_issues(chunk, document, id_counts)
        approval = approvals.get(chunk.get("chunk_id"))
        approved = bool(approval and approval.get("approved") is True and approval.get("approval_status") == "approved")
        codes = {issue["code"] for issue in quality}
        if approved:
            candidate_text = original_text
            provenance = [{"chunk_id": str(chunk.get("chunk_id")), "text_sha256": _sha(original_text)}] if original_text else []
            repair_kind = None
        else:
            candidate_text, provenance, repair_kind = _derive_candidate_text(index, raw, codes)
        unresolved = set(codes & {"lowercase_sentence_start", "open_sentence_end", "conjunction_boundary"})
        if repair_kind and len(provenance) > 1:
            if any(item["chunk_id"] != chunk.get("chunk_id") for item in provenance):
                unresolved.discard("lowercase_sentence_start")
                unresolved.discard("open_sentence_end")
                unresolved.discard("conjunction_boundary")
        scope = _scope(document_id)
        topics = _matches_for_scope(candidate_text, scope)
        points = sum(min(2, len(hits)) for hits in topics.values())
        score = round(points / (len(TOPIC_TERMS) * 2) * 100, 1)
        issue_codes = sorted(codes)
        row = {
            "chunk_id": chunk.get("chunk_id"),
            "document_id": document_id,
            "document_scope": scope,
            "document_title": document.get("title") or chunk.get("document_title"),
            "extracted_chunk_title": chunk.get("document_title"),
            "publisher": document.get("publisher") or chunk.get("publisher"),
            "publication_year": document.get("publication_year") or chunk.get("publication_year"),
            "source_url": document.get("source_url") or chunk.get("source_url"),
            "source_filename": document.get("filename") or chunk.get("source_filename"),
            "page_number": chunk.get("page_number"),
            "page_start": min(item.get("page_number", chunk.get("page_number")) for _, item in [(index, raw[index])]),
            "page_end": chunk.get("page_number"),
            "license": document.get("license"),
            "licensing_status": document.get("licensing_status"),
            "license_source_url": document.get("license_source_url"),
            "attribution": (document.get("rights_preflight") or {}).get("attribution_requirements"),
            "source_text_sha256": _sha(original_text) if original_text else None,
            "source_character_count": len(original_text),
            "candidate_text": candidate_text,
            "derived_text_sha256": _sha(candidate_text) if candidate_text else None,
            "derived_from": provenance,
            "repair_kind": repair_kind,
            "relevance_score": score,
            "topic_matches": topics,
            "quality_issue_codes": issue_codes,
            "quality_warnings": quality,
            "blocking_issues": [issue for issue in quality if issue["severity"] == "blocking"],
            "unresolved_boundary_codes": sorted(unresolved),
            "duplicate_indices": duplicate_indices.get(index, []),
            "approval_status": approval.get("approval_status", "pending_review") if approval else "pending_review",
            "approved": approved,
            "approval_reviewer": approval.get("reviewer") if approval else None,
            "approved_at": approval.get("approved_at") if approval else None,
            "approval_note": approval.get("approval_note") if approval else None,
            "approval_fingerprint_match": bool(approval and approval.get("text_sha256") == _sha(original_text)),
            "third_party_review": approval.get("third_party_review", "pending") if approval else "pending",
            "rights_gate": _rights_gate(document, approval),
        }
        row["exclusion_reasons"] = _exclusion_reasons(row)
        analysed.append(row)

    # Approved records stay visible regardless of lexical score. Pending records
    # must pass quality gates and show at least two topic groups with evidence.
    candidates = [row for row in analysed if (row["approved"] and row["chunk_id"] not in REJECTED_CHUNK_IDS and (row["document_id"], row["page_number"]) not in REJECTED_PAGES) or (
        not row["exclusion_reasons"] and len(row["topic_matches"]) >= MIN_TOPIC_GROUPS and row["relevance_score"] >= MIN_SCORE
    )]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    # Keep each explicit approval, even when it resembles another passage.
    for row in sorted((row for row in candidates if row["approved"]), key=lambda r: (r["document_id"], r["chunk_id"])):
        selected.append(row)
        selected_ids.add(row["chunk_id"])
    pending = [row for row in candidates if not row["approved"]]
    topic_counts: Counter[str] = Counter(topic for row in selected for topic in row["topic_matches"])
    pending.sort(key=lambda row: (
        -sum(topic_counts[t] == 0 for t in row["topic_matches"]),
        -row["relevance_score"], row["document_id"], row["chunk_id"],
    ))
    duplicate_suppressed_ids: set[str] = set()
    for row in pending:
        if row["chunk_id"] in selected_ids:
            continue
        if any(raw[index].get("chunk_id") in selected_ids for index in row["duplicate_indices"]):
            duplicate_suppressed_ids.add(row["chunk_id"])
            continue
        selected.append(row)
        selected_ids.add(row["chunk_id"])
        topic_counts.update(row["topic_matches"].keys())

    for row in selected:
        row["import_eligibility"], row["import_eligible"] = _import_eligibility(row)
    selected_by_doc = Counter(row["document_id"] for row in selected)
    excluded = [row for row in analysed if row["chunk_id"] not in selected_ids]
    excluded_by_doc = Counter(row["document_id"] for row in excluded)
    exclusion_counts: Counter[str] = Counter(reason for row in excluded for reason in row["exclusion_reasons"])
    # Include low-relevance decisions as explicit exclusions too.
    for row in excluded:
        if not row["exclusion_reasons"]:
            reason = "duplicate_of_selected" if row["chunk_id"] in duplicate_suppressed_ids else "below_relevance_threshold"
            row["exclusion_reasons"] = [reason]
            exclusion_counts[reason] += 1

    doc_results: dict[str, Any] = {}
    for document_id, document in sorted(doc_map.items()):
        subset = [row for row in analysed if row["document_id"] == document_id]
        doc_results[document_id] = {
            "title": document.get("title"),
            "geographic_scope": _scope(document_id),
            "total_chunks": len(subset),
            "candidate_count": selected_by_doc[document_id],
            "excluded_count": excluded_by_doc[document_id],
            "approved_in_selection": sum(row["approved"] for row in selected if row["document_id"] == document_id),
            "import_eligible_count": sum(row["import_eligible"] for row in selected if row["document_id"] == document_id),
            "topic_coverage": {topic: sum(topic in row["topic_matches"] for row in selected if row["document_id"] == document_id) for topic in TOPIC_TERMS},
            "issue_counts": dict(sorted(Counter(issue for row in subset for issue in row["quality_issue_codes"]).items())),
            "rights_status": (document.get("rights_preflight") or {}).get("external_embedding_status", "unknown"),
        }

    public_selected = []
    for row in selected:
        public_selected.append({key: value for key, value in row.items() if key != "duplicate_indices"})
    result = {
        "mode": "offline_deterministic_no_credentials_or_network",
        "total_chunks_analyzed": len(analysed),
        "selected_candidate_count": len(selected),
        "selected_counts_by_document": dict(selected_by_doc),
        "excluded_chunk_count": len(excluded),
        "excluded_counts_by_document": dict(excluded_by_doc),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "repaired_candidate_count": sum(bool(row["repair_kind"] and len(row["derived_from"]) > 1) for row in selected),
        "whitespace_normalized_candidate_count": sum(row["repair_kind"] == "normalized_layout_whitespace_only" for row in selected),
        "selected_quality_warning_count": sum(len(row["quality_warnings"]) for row in selected),
        "import_eligible_count": sum(row["import_eligible"] for row in selected),
        "pending_approval_count": sum(not row["approved"] for row in selected),
        "approved_count": sum(row["approved"] for row in selected),
        "document_results": doc_results,
        "topic_coverage": {topic: sum(topic in row["topic_matches"] for row in selected) for topic in TOPIC_TERMS},
        "quality_warning_examples": [
            {"chunk_id": row["chunk_id"], "document_id": row["document_id"], "page_number": row["page_number"], "codes": row["quality_issue_codes"]}
            for row in selected if row["quality_issue_codes"]
        ][:25],
        "excluded_examples": [
            {"chunk_id": row["chunk_id"], "document_id": row["document_id"], "page_number": row["page_number"], "reasons": row["exclusion_reasons"]}
            for row in excluded
        ][:60],
        "selected_candidates": public_selected,
        "approval_warnings": approval_warnings,
    }
    review.mkdir(parents=True, exist_ok=True)
    (review / "refined_rag_selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (review / "refined_rag_selection.md").write_text(render_refined_report(result), encoding="utf-8")
    return result


def render_refined_report(result: dict[str, Any]) -> str:
    lines = [
        "# Refined RAG Selection Review",
        "",
        "Offline deterministic review only; no credentials, network requests, embeddings, or Pinecone operations were used.",
        "",
        f"- Chunks evaluated: **{result['total_chunks_analyzed']}**",
        f"- Review candidates: **{result['selected_candidate_count']}**",
        f"- Excluded from selection: **{result['excluded_chunk_count']}**",
        f"- Candidates repaired by exact adjacent-chunk combination: **{result['repaired_candidate_count']}**",
        f"- Candidates with whitespace-only normalization: **{result['whitespace_normalized_candidate_count']}**",
        f"- Candidates with import eligibility now: **{result['import_eligible_count']}**",
        f"- Existing approvals preserved: **{result['approved_count']}**; unapproved candidates: **{result['pending_approval_count']}**",
        "",
        "## Candidates by source",
        "",
        "| Source | Geographic scope | Chunks | Candidates | Excluded | Approved | Pending approval | Import eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _doc_id, data in result["document_results"].items():
        pending = data["candidate_count"] - data["approved_in_selection"]
        lines.append(f"| {data['title']} | {data['geographic_scope']} | {data['total_chunks']} | {data['candidate_count']} | {data['excluded_count']} | {data['approved_in_selection']} | {pending} | {data['import_eligible_count']} |")
    lines.extend(["", "## Topic coverage by source", "", "Counts are shortlisted passages with at least one lexical match in the topic group. Topic groups overlap; ILO is classified as international/global and does not receive Germany-specific matches from generic employment wording.", "", "| Source | AI transformation | German labour market | ICT demand | Qualifications | Workforce development | Long-term development |", "|---|---:|---:|---:|---:|---:|---:|"])
    topic_columns = ("ai_automation", "german_labour_market", "ict_employment_demand", "skills_qualifications", "upskilling_reskilling", "long_term_career_development")
    for _doc_id, data in result["document_results"].items():
        lines.append(f"| {data['title']} | " + " | ".join(str(data["topic_coverage"][topic]) for topic in topic_columns) + " |")
    lines.extend([
        "",
        "## Quality, derived text, and selection rules",
        "",
        f"Pending chunks shorter than {MIN_STANDALONE_LENGTH} characters, title-only fragments, references-only passages, chart/figure-only extractions, invalid source records, unresolved sentence-boundary fragments, and the previously rejected Cedefop page 7 chunk are excluded. Mixed prose and chart warnings remain review flags rather than automatic exclusions.",
        "",
        "When same-page chunk IDs prove exact adjacency, the candidate text may join the original neighboring passages and normalize layout whitespace. No words, punctuation, statistics, or contextual claims are added. The original chunk ID and SHA-256 remain attached; derived text has its own SHA-256 and source-chunk fingerprints. Combined text still needs human context review.",
        "",
        f"Pending selection requires at least {MIN_TOPIC_GROUPS} topic groups and a lexical score of at least {MIN_SCORE}. Scores are keyword overlap, not semantic relevance or factual confidence. The global ILO report is not treated as Germany-specific. The selector has no source quota or forced target count; near-duplicates are suppressed where possible.",
        "",
        "### Exclusion reasons",
        "",
    ])
    lines.extend(f"- `{reason}`: {count}" for reason, count in result["exclusion_reason_counts"].items())
    lines.extend(["", "### Quality warnings requiring human attention", ""])
    for sample in result["quality_warning_examples"][:12]:
        lines.append(f"- `{sample['chunk_id']}` (PDF p.{sample['page_number']}): {', '.join(sample['codes'])}")
    lines.extend(["", "### Spot-check candidates", ""])
    for doc_id in result["document_results"]:
        samples = [row for row in result["selected_candidates"] if row["document_id"] == doc_id][:2]
        for row in samples:
            excerpt = re.sub(r"\s+", " ", row["candidate_text"]).strip()[:280]
            lines.append(f"- `{row['chunk_id']}` (PDF p.{row['page_number']}; {row['document_scope']}; import: `{row['import_eligibility']}`): “{excerpt}”")
    lines.extend([
        "",
        "## Rights, approvals, and import status",
        "",
        "Quality selection does not grant reuse rights or approval. BA material remains blocked while reuse clarification is unresolved. Cedefop's explicitly owner-approved chunk is not import-ready until its third-party review is cleared. Commission candidates retain EU-owned versus third-party uncertainty and require individual review. The seven approved ILO records are preserved as already indexed and must not be reuploaded. No new approvals were created.",
        "",
        "The European Commission manifest title is used for citation. All 99 chunk records have a differing extracted title; this is surfaced as a metadata warning and does not change source IDs or text.",
        "",
        "Machine-readable details, source text, derived text, source fingerprints, citations, warnings, and per-record eligibility are in `refined_rag_selection.json`. These review files are local and Git-ignored.",
        "",
    ])
    return "\n".join(lines)
