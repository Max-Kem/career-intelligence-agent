"""Deterministic, offline quality and RAG-candidate audit for local chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "ai_automation": (
        "artificial intelligence", "generative ai", "generative artificial intelligence", "\\bai\\b",
        "künstliche intelligenz", "kuenstliche intelligenz", "generative ki", "\bki\b",
        "automation", "automatisierung", "automated", "automatisch", "robotics", "robotic process",
        "algorithmic", "algorithmisch",
    ),
    "german_labour_market": (
        "germany", "german", "deutschland", "arbeitsmarkt", "labour market", "labor market",
        "employment", "beschäftigung", "beschaeftigung", "erwerbstätig", "erwerbstaetig",
        "unemployment", "arbeitslos", "vacancies", "arbeitsstellen", "arbeitskräfte", "arbeitskraefte",
    ),
    "ict_employment_demand": (
        "ict", "ikt", "information and communication technology", "information technology",
        "software development", "software developer", "data analyst", "data science", "it specialist",
        "ict specialist", "ict professionals", "it-fachkräfte", "it-fachkraefte",
    ),
    "skills_qualifications": (
        "skills", "skill requirements", "skill demand", "digital skills", "competenc", "qualification",
        "qualifikation", "kompetenz", "fachkräftebedarf", "fachkraeftebedarf", "digital competence",
    ),
    "upskilling_reskilling": (
        "upskilling", "reskilling", "retraining", "training", "weiterbildung", "qualifizierung",
        "lebenslanges lernen", "lifelong learning", "vocational education", "ausbildung",
    ),
    "long_term_career_development": (
        "career development", "career path", "occupational change", "occupational transition",
        "berufliche entwicklung", "berufsverlauf", "forecast", "forecast period", "2035", "long-term",
        "long term", "future employment", "future labour", "future labor",
    ),
}

MIN_RELEVANCE_SCORE = 16.7  # At least two topic groups with one lexical signal each.
MAX_SHORTLIST_SIZE = 120
MAX_PER_DOCUMENT = 36
SHORT_TEXT_CHARS = 160
LONG_TEXT_CHARS = 2200
NEAR_DUPLICATE_JACCARD = 0.88


class KnowledgeAuditError(RuntimeError):
    """Invalid local manifest, chunk source, or approval artifact."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KnowledgeAuditError(f"{label} not found: {path}") from None
    except json.JSONDecodeError:
        raise KnowledgeAuditError(f"{label} contains invalid JSON: {path.name}") from None
    if not isinstance(value, dict):
        raise KnowledgeAuditError(f"{label} must be a JSON object: {path.name}")
    return value


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise KnowledgeAuditError(f"Could not read chunk file {path.name}: {type(exc).__name__}") from None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise KnowledgeAuditError(f"Malformed JSON in {path.name}, line {line_number}.") from None
        if not isinstance(row, dict):
            raise KnowledgeAuditError(f"Chunk record in {path.name}, line {line_number}, is not an object.")
        rows.append(row)
    return rows


def _normalise_text(text: str) -> str:
    text = text.casefold().replace("\ufffd", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = _normalise_text(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def _near_duplicate_pairs(records: list[dict[str, Any]]) -> list[tuple[int, int, float, str]]:
    """Return exact/near duplicate record pairs using normalized text and word shingles."""
    normalized: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        text = record.get("text")
        if isinstance(text, str) and text.strip():
            normalized[_normalise_text(text)].append(i)
    pairs: list[tuple[int, int, float, str]] = []
    already: set[tuple[int, int]] = set()
    for indices in normalized.values():
        if len(indices) > 1:
            for offset, i in enumerate(indices):
                for j in indices[offset + 1:]:
                    pairs.append((i, j, 1.0, "exact_duplicate"))
                    already.add((i, j))

    nonempty = [i for i, record in enumerate(records) if isinstance(record.get("text"), str) and record["text"].strip()]
    shingle_sets = {i: _shingles(records[i]["text"]) for i in nonempty}
    for pos, i in enumerate(nonempty):
        left = shingle_sets[i]
        if not left:
            continue
        for j in nonempty[pos + 1:]:
            if (i, j) in already:
                continue
            a = records[i]["text"].strip()
            b = records[j]["text"].strip()
            if min(len(a), len(b)) / max(len(a), len(b)) < 0.8:
                continue
            right = shingle_sets[j]
            union = left | right
            similarity = len(left & right) / len(union) if union else 0.0
            if similarity >= NEAR_DUPLICATE_JACCARD:
                pairs.append((i, j, similarity, "near_duplicate"))
    return pairs


def _topic_match(text: str) -> dict[str, list[str]]:
    lowered = text.casefold()
    matched: dict[str, list[str]] = {}
    for topic, terms in TOPIC_TERMS.items():
        hits: list[str] = []
        for term in terms:
            if term.startswith("\\b"):
                pattern = term
            else:
                pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                hits.append(term.strip("\\b"))
        if hits:
            matched[topic] = hits
    return matched


def _quality_issues(chunk: dict[str, Any], document: dict[str, Any], source_counts: Counter[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    def add(code: str, severity: str, detail: str) -> None:
        issues.append({"code": code, "severity": severity, "detail": detail})

    text = chunk.get("text")
    chunk_id = chunk.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        add("missing_chunk_id", "blocking", "Chunk has no stable identifier.")
    elif source_counts[chunk_id] > 1:
        add("duplicate_chunk_id", "blocking", "Chunk identifier occurs more than once in the source set.")

    if chunk.get("document_id") != document.get("document_id"):
        add("document_reference_mismatch", "blocking", "Chunk document_id does not match its source file/manifest entry.")

    page = chunk.get("page_number")
    page_count = document.get("page_count")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1 or (isinstance(page_count, int) and page > page_count):
        add("invalid_page_reference", "blocking", "PDF page reference is missing or outside the manifest page range.")

    required = ("document_title", "publisher", "publication_year", "source_url", "source_filename")
    missing = [field for field in required if chunk.get(field) in (None, "")]
    if missing:
        add("missing_source_metadata", "blocking", "Missing: " + ", ".join(missing) + ".")
    if not isinstance(chunk.get("source_url"), str) or not chunk.get("source_url", "").startswith(("https://", "http://")):
        add("malformed_source_url", "blocking", "Source URL is unavailable or malformed.")

    if not isinstance(text, str) or not text.strip():
        add("empty_text", "blocking", "Chunk text is empty or not a string.")
        return issues

    if len(text.strip()) < SHORT_TEXT_CHARS:
        add("very_short_text", "warning", f"Only {len(text.strip())} characters; may be front matter or an incomplete passage.")
    if len(text) > LONG_TEXT_CHARS:
        add("long_chunk", "warning", f"{len(text)} characters exceeds the audit's {LONG_TEXT_CHARS}-character review threshold.")

    if "\ufffd" in text or "\x00" in text:
        add("replacement_or_null_character", "warning", "Text contains replacement or null characters.")
    if re.search(r" {2,}|\t|\n{3,}", text):
        add("unusual_whitespace", "warning", "Repeated spaces, tabs, or unusually large blank gaps were detected.")
    split_word_hits = re.findall(r"\b(?:a ddress|advis ory|area s|employ ment|qualifi cation|s kills|re quirement)\b", text, flags=re.IGNORECASE)
    if split_word_hits:
        add("probable_broken_words", "warning", "Possible OCR/extraction word splits: " + ", ".join(sorted(set(split_word_hits))) + ".")

    stripped = text.strip()
    if stripped and stripped[0].islower():
        add("lowercase_sentence_start", "warning", "Starts lowercase; may continue a preceding sentence.")
    if stripped and stripped[-1] not in ".!?;:)]}”’\"'":
        add("open_sentence_end", "warning", "Does not end with common sentence punctuation; inspect the next chunk/page.")
    if re.search(r"\b(?:and|or|while|which|that|with|of|as|because|und|oder|während|waehrend|mit|von|dass|sowie)\s*[.!?]?\s*$", stripped, flags=re.IGNORECASE):
        add("conjunction_boundary", "warning", "Ends with a conjunction or dependent phrase.")

    if re.search(r"\b(?:figure|fig\.|abbildung|table|tabelle)\s*\d+", text, flags=re.IGNORECASE) or re.search(r"(?:\b\d+[.,]?\d*\b\s*){8,}", text):
        add("chart_or_table_residue", "warning", "Figure/table labels or dense numeric sequences may be mixed with prose.")
    ref_markers = len(re.findall(r"\b(?:references|bibliography|quelle|sources?|doi|isbn)\b|https?://", text, flags=re.IGNORECASE))
    word_count = len(re.findall(r"\b\w+\b", text))
    if ref_markers >= 2 and word_count < 70:
        add("references_only_suspected", "warning", "Short passage is dominated by references, identifiers, or URLs.")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if isinstance(chunk_id, str) and re.fullmatch(r"[0-9a-fA-F]{12}", chunk_id.rsplit("-", 1)[-1]):
        if not digest.startswith(chunk_id.rsplit("-", 1)[-1].casefold()):
            add("chunk_id_fingerprint_mismatch", "blocking", "ID checksum suffix does not match SHA-256 of the current text.")
    else:
        add("missing_text_fingerprint", "blocking", "Chunk ID does not contain the expected 12-hex text fingerprint suffix.")

    for field, manifest_field in (("document_title", "title"), ("publisher", "publisher"), ("publication_year", "publication_year"), ("source_url", "source_url"), ("source_filename", "filename")):
        if chunk.get(field) != document.get(manifest_field):
            add("manifest_metadata_mismatch", "warning", f"Chunk {field} differs from manifest {manifest_field}; canonical manifest metadata retained in shortlist.")
            break
    return issues


def _load_approval_map(review_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    approval_map: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for filename in ("ilo_pilot_approvals.json", "multi_source_pilot_approvals.json"):
        path = review_dir / filename
        if not path.exists():
            warnings.append(f"Approval artifact missing: {filename}; matching chunks will remain unapproved.")
            continue
        data = _read_json(path, "Approval artifact")
        for entry in data.get("candidates", []):
            if isinstance(entry, dict) and isinstance(entry.get("chunk_id"), str):
                approval_map[entry["chunk_id"]] = entry
    return approval_map, warnings


def _rights_gate(document: dict[str, Any], approval: dict[str, Any] | None) -> str:
    doc_id = document.get("document_id")
    status = (document.get("rights_preflight") or {}).get("external_embedding_status")
    is_approved = bool(approval and approval.get("approved") is True and approval.get("approval_status") == "approved")
    if doc_id == "ilo_generative_ai_jobs_2025":
        if is_approved and approval and approval.get("third_party_review") == "cleared":
            return "approved_by_chunk_review"
        return "pending_chunk_approval_and_third_party_review"
    if doc_id == "ba_ikt_arbeitsmarkt_2026":
        return "blocked_pending_reuse_policy_clarification"
    if doc_id == "cedefop_skills_forecast_germany_2025":
        if is_approved and approval and approval.get("third_party_review") != "cleared":
            return "approved_by_project_owner_third_party_review_not_separately_cleared"
        return "approved_by_project_owner" if is_approved else "pending_chunk_approval"
    if doc_id == "eu_digital_decade_germany_2026":
        return "approved_by_project_owner_with_third_party_review_pending" if is_approved else "pending_rights_and_chunk_review"
    if is_approved:
        return "approved_by_project_owner"
    return f"pending_rights_review:{status or 'unknown'}"


def _select_candidates(records: list[dict[str, Any]], *, limit: int = MAX_SHORTLIST_SIZE) -> list[dict[str, Any]]:
    eligible = [r for r in records if not r["blocking_issues"] and r["relevance_score"] >= MIN_RELEVANCE_SCORE and len(r["topic_matches"]) >= 2]
    # Explicitly approved records are retained in the audit even if a lexical
    # heuristic misses their topic. Approval is a separate human decision.
    approved = [r for r in records if r["approval_status"] == "approved"]
    selected: list[dict[str, Any]] = sorted(approved, key=lambda r: (r["document_id"], r["chunk_id"]))
    selected_ids = {r["chunk_id"] for r in selected}
    doc_counts = Counter(r["document_id"] for r in selected)
    topic_counts = Counter(topic for r in selected for topic in r["topic_matches"])
    remaining = [r for r in eligible if r["chunk_id"] not in selected_ids]

    while remaining and len(selected) < limit:
        selected_ids = {r["chunk_id"] for r in selected}
        remaining = [
            r for r in remaining
            if doc_counts[r["document_id"]] < MAX_PER_DOCUMENT
            and not any(match.get("chunk_id") in selected_ids for match in r["duplicate_matches"])
        ]
        if not remaining:
            break
        def priority(record: dict[str, Any]) -> tuple[float, float, str]:
            novel_topics = sum(1 for topic in record["topic_matches"] if topic_counts[topic] == 0)
            undercovered_topics = sum(1 for topic in record["topic_matches"] if topic_counts[topic] < 5)
            undercovered_doc = max(0, MAX_PER_DOCUMENT - doc_counts[record["document_id"]])
            score = record["relevance_score"] + novel_topics * 10 + undercovered_topics * 1.5 + min(undercovered_doc, 10) * 0.25
            return score, record["relevance_score"], record["chunk_id"]
        chosen = max(remaining, key=priority)
        selected.append(chosen)
        doc_counts[chosen["document_id"]] += 1
        for topic in chosen["topic_matches"]:
            topic_counts[topic] += 1
        remaining.remove(chosen)
    return selected


def audit_knowledge_base(processed_dir: str | Path, review_dir: str | Path) -> dict[str, Any]:
    """Audit every local chunk and write a deterministic report and machine shortlist."""
    processed = Path(processed_dir)
    review = Path(review_dir)
    manifest = _read_json(processed / "manifest.json", "Processed manifest")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise KnowledgeAuditError("Processed manifest must contain a documents list.")
    doc_map = {d.get("document_id"): d for d in documents if isinstance(d, dict) and d.get("document_id")}
    if not doc_map:
        raise KnowledgeAuditError("Processed manifest has no valid document entries.")

    raw_records: list[dict[str, Any]] = []
    for document_id, document in sorted(doc_map.items()):
        filename = processed / f"{document_id}.chunks.jsonl"
        for chunk in _read_chunks(filename):
            chunk["_source_document_id"] = document_id
            raw_records.append(chunk)
    chunk_counts = Counter(r.get("chunk_id") for r in raw_records if isinstance(r.get("chunk_id"), str))
    duplicate_pairs = _near_duplicate_pairs(raw_records)
    duplicate_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, j, similarity, kind in duplicate_pairs:
        duplicate_by_index[i].append({"chunk_id": raw_records[j].get("chunk_id"), "similarity": round(similarity, 4), "kind": kind})
        duplicate_by_index[j].append({"chunk_id": raw_records[i].get("chunk_id"), "similarity": round(similarity, 4), "kind": kind})

    approvals, approval_warnings = _load_approval_map(review)
    analyzed: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw_records):
        document_id = chunk.get("document_id")
        source_document_id = chunk.get("_source_document_id")
        document = doc_map.get(source_document_id)
        if document is None:
            # The source file identity is inferred only for diagnostics; the source reference remains blocking.
            document = next((d for d in doc_map.values() if d.get("document_id") in str(chunk.get("chunk_id", ""))), {})
        if not document:
            document = {"document_id": document_id, "page_count": 0}
        issues = _quality_issues(chunk, document, chunk_counts)
        if duplicate_by_index.get(index):
            issues.append({"code": "duplicate_content", "severity": "warning", "detail": "Matches another chunk by normalized text/shingles; selection avoids repeated candidates where possible."})
        text = chunk.get("text") if isinstance(chunk.get("text"), str) else ""
        topic_hits = _topic_match(text)
        topic_points = sum(min(2, len(hits)) for hits in topic_hits.values())
        score = round(topic_points / (len(TOPIC_TERMS) * 2) * 100, 1)
        approval = approvals.get(chunk.get("chunk_id"))
        approved = bool(approval and approval.get("approved") is True and approval.get("approval_status") == "approved")
        rights_gate = _rights_gate(document, approval)
        analyzed.append({
            "chunk_id": chunk.get("chunk_id"),
            "document_id": document.get("document_id") or document_id,
            "page_number": chunk.get("page_number"),
            "document_title": document.get("title") or chunk.get("document_title"),
            "extracted_chunk_title": chunk.get("document_title"),
            "publisher": document.get("publisher") or chunk.get("publisher"),
            "publication_year": document.get("publication_year") or chunk.get("publication_year"),
            "source_url": document.get("source_url") or chunk.get("source_url"),
            "source_url_status": document.get("source_url_status"),
            "source_filename": document.get("filename") or chunk.get("source_filename"),
            "license": document.get("license"),
            "licensing_status": document.get("licensing_status"),
            "license_source_url": document.get("license_source_url"),
            "attribution": (document.get("rights_preflight") or {}).get("attribution_requirements"),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
            "character_count": len(text),
            "relevance_score": score,
            "topic_matches": topic_hits,
            "quality_issues": issues,
            "blocking_issues": [i for i in issues if i["severity"] == "blocking"],
            "warning_issues": [i for i in issues if i["severity"] == "warning"],
            "duplicate_matches": duplicate_by_index.get(index, []),
            "approval_status": approval.get("approval_status", "pending_review") if approval else "pending_review",
            "approved": approved,
            "approval_reviewer": approval.get("reviewer") if approval else None,
            "approved_at": approval.get("approved_at") if approval else None,
            "third_party_review": approval.get("third_party_review", "pending") if approval else "pending",
            "approval_note": approval.get("approval_note") if approval else None,
            "rights_gate": rights_gate,
            "_source_text": text,
        })

    selected = _select_candidates(analyzed)
    selected_ids = {r["chunk_id"] for r in selected}
    for record in analyzed:
        record["quality_issue_codes"] = sorted({issue["code"] for issue in record["quality_issues"]})
        if record["chunk_id"] not in selected_ids:
            record["selection_status"] = "not_selected"
        elif record["rights_gate"].startswith("blocked"):
            record["selection_status"] = "selected_but_rights_blocked"
        elif record["approval_status"] == "approved":
            record["selection_status"] = "selected_existing_approval"
        else:
            record["selection_status"] = "selected_pending_rights_or_owner_review"

    by_document: dict[str, dict[str, Any]] = {}
    for document_id, document in sorted(doc_map.items()):
        subset = [r for r in analyzed if r["document_id"] == document_id]
        issue_counts = Counter(issue["code"] for r in subset for issue in r["quality_issues"])
        blocking_chunks = sum(bool(r["blocking_issues"]) for r in subset)
        topic_counts = {topic: sum(topic in r["topic_matches"] for r in subset) for topic in TOPIC_TERMS}
        by_document[document_id] = {
            "title": document.get("title"),
            "publisher": document.get("publisher"),
            "publication_year": document.get("publication_year"),
            "source_url": document.get("source_url"),
            "license_source_url": document.get("license_source_url"),
            "licensing_status": document.get("licensing_status"),
            "attribution_requirements": (document.get("rights_preflight") or {}).get("attribution_requirements"),
            "external_embedding_status": (document.get("rights_preflight") or {}).get("external_embedding_status", "unknown"),
            "chunk_count": len(subset),
            "blocking_chunk_count": blocking_chunks,
            "issue_counts": dict(sorted(issue_counts.items())),
            "topic_coverage": topic_counts,
            "selected_candidate_count": sum(r["chunk_id"] in selected_ids for r in subset),
            "selected_approved_count": sum(r["chunk_id"] in selected_ids and r["approval_status"] == "approved" for r in subset),
            "selected_rights_blocked_count": sum(r["chunk_id"] in selected_ids and r["rights_gate"].startswith("blocked") for r in subset),
        }

    all_topics = {topic: sum(topic in r["topic_matches"] for r in analyzed) for topic in TOPIC_TERMS}
    shortlist_records = []
    for record in selected:
        card = {k: v for k, v in record.items() if k not in {"_source_text", "quality_issues", "blocking_issues", "warning_issues", "duplicate_matches"}}
        shortlist_records.append(card)

    results = {
        "audit_version": 1,
        "mode": "offline_deterministic",
        "total_chunk_count": len(analyzed),
        "total_blocking_chunk_count": sum(bool(r["blocking_issues"]) for r in analyzed),
        "total_warning_count": sum(len(r["warning_issues"]) for r in analyzed),
        "exact_duplicate_pair_count": sum(kind == "exact" for _, _, _, kind in duplicate_pairs),
        "near_duplicate_pair_count": sum(kind == "near" for _, _, _, kind in duplicate_pairs),
        "topic_coverage": all_topics,
        "document_results": by_document,
        "selected_candidate_count": len(selected),
        "selected_counts_by_document": {doc_id: data["selected_candidate_count"] for doc_id, data in by_document.items()},
        "selected_approved_count": sum(r["approval_status"] == "approved" for r in selected),
        "selected_pending_count": sum(r["approval_status"] != "approved" for r in selected),
        "selected_rights_blocked_count": sum(r["rights_gate"].startswith("blocked") for r in selected),
        "approval_warnings": approval_warnings,
        "selected_candidates": shortlist_records,
        "_all_records": analyzed,
    }
    review.mkdir(parents=True, exist_ok=True)
    (review / "knowledge_base_shortlist.json").write_text(json.dumps({k: v for k, v in results.items() if not k.startswith("_")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (review / "knowledge_base_audit.md").write_text(render_audit_markdown(results), encoding="utf-8")
    return results


def render_audit_markdown(results: dict[str, Any]) -> str:
    records = results.get("_all_records", [])
    lines = [
        "# Offline Knowledge Base Audit and RAG Selection",
        "",
        f"- Mode: `{results['mode']}` (no credentials, network, embeddings, or Pinecone calls)",
        f"- Chunks analyzed: **{results['total_chunk_count']}**",
        f"- Chunks with blocking quality issues: **{results['total_blocking_chunk_count']}**",
        f"- Quality warnings across chunks: **{results['total_warning_count']}**",
        f"- Duplicate pairs: **{results['exact_duplicate_pair_count']} exact**, **{results['near_duplicate_pair_count']} near-duplicate** (automatically excluded from pending selection where possible)",
        f"- Proposed shortlist: **{results['selected_candidate_count']}** chunks (includes currently approved records and candidates still requiring rights/owner review)",
        f"- Existing approvals represented: **{results['selected_approved_count']}**; pending/unapproved selected candidates: **{results['selected_pending_count']}**",
        f"- Shortlisted records blocked by unresolved rights: **{results['selected_rights_blocked_count']}**",
        "",
        "## Method and limitations",
        "",
        f"The heuristic assigns each of six topic groups up to two points for distinct matched terms; the raw total is normalized to 0–100. A chunk is considered relevant for shortlist selection when it matches at least two topic groups and scores at least {MIN_RELEVANCE_SCORE}. The deterministic selector prioritizes topic and source coverage, limits each source to {MAX_PER_DOCUMENT} candidates, removes near duplicates where detected, and caps output at {MAX_SHORTLIST_SIZE}.",
        "",
        "These scores measure lexical overlap only. They are not semantic judgments, factual confidence, legal clearance, or recommendation scores. Keyword matches can miss relevant passages written with different terms and can overrate irrelevant mentions. Quality warnings are review signals; they do not automatically reject content unless marked blocking.",
        "",
        "## Quality results by source",
        "",
        "| Source | Chunks | Blocking chunks | Selected candidates | Approved in selection | Rights-blocked in selection | Top warnings |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for document_id, data in results["document_results"].items():
        top = sorted(data["issue_counts"].items(), key=lambda p: (-p[1], p[0]))[:4]
        warning_summary = ", ".join(f"{code}={count}" for code, count in top if code not in {"missing_source_metadata", "invalid_page_reference", "chunk_id_fingerprint_mismatch", "duplicate_chunk_id", "empty_text"}) or "none"
        lines.append(f"| {data['title']} | {data['chunk_count']} | {data['blocking_chunk_count']} | {data['selected_candidate_count']} | {data['selected_approved_count']} | {data['selected_rights_blocked_count']} | {warning_summary} |")
    lines.extend(["", "## Topic coverage", "", "Count is chunks with at least one keyword-group match; topic groups may overlap.", "", "| Topic group | Matching chunks |", "|---|---:|"])
    for topic, count in results["topic_coverage"].items():
        lines.append(f"| {topic.replace('_', ' ')} | {count} |")
    lines.extend(["", "### Coverage by source", "", "| Source | " + " | ".join(topic.replace("_", " ") for topic in TOPIC_TERMS) + " |", "|---|" + "---:|" * len(TOPIC_TERMS)])
    for _doc_id, data in results["document_results"].items():
        lines.append("| " + str(data["title"]) + " | " + " | ".join(str(data["topic_coverage"][topic]) for topic in TOPIC_TERMS) + " |")

    lines.extend(["", "## Rights and approval status", "", "Rights are evaluated separately from content quality. Passing quality checks never grants permission or approval.", ""])
    for _document_id, data in results["document_results"].items():
        policy = f"; policy reference [{data['license_source_url']}]({data['license_source_url']})" if data.get("license_source_url") else ""
        attribution = f" Attribution: {data['attribution_requirements']}" if data.get("attribution_requirements") else ""
        lines.append(f"- **{data['title']}**: manifest status `{data['licensing_status']}`; external-embedding preflight `{data['external_embedding_status']}`{policy}; selected `{data['selected_candidate_count']}`, approved `{data['selected_approved_count']}`, rights-blocked `{data['selected_rights_blocked_count']}`.{attribution}")
    lines.extend([
        "- **ILO:** the seven explicitly approved chunks remain approved. The owner-provided context says they are already indexed; this offline audit does not query or verify Pinecone. Other ILO chunks remain unapproved unless a matching approval record explicitly approves them and clears third-party review.",
        "- **BA:** selected for content coverage only where relevant; external processing remains blocked until the reuse-policy ambiguity is resolved.",
        "- **Cedefop:** the specifically owner-approved chunk is recorded as approved; every other chunk remains pending individual review. Apply the existing publisher policy, attribution, and third-party restrictions.",
        "- **European Commission:** Commission-owned content may be covered by the documented CC BY 4.0 default; third-party material still needs separate review. All unapproved chunks remain pending.",
        "",
        "The Cedefop owner-approved chunk retains its explicit approval, but the approval record still marks third-party review as pending; verify the chunk-level context before external processing. The Commission and Cedefop selected candidates also remain subject to document-specific and third-party review.",
        "",
        "The Commission manifest title is used as canonical metadata. All 99 associated chunk records carry a differing extracted title and are flagged as a metadata warning; the source-document association and chunk text were not changed.",
        "",
        "## Representative quality issues",
    ])
    examples = [r for r in records if r["quality_issues"]]
    examples.sort(key=lambda r: (-sum(i["severity"] == "blocking" for i in r["quality_issues"]), -len(r["quality_issues"]), r["chunk_id"] or ""))
    seen_codes: set[str] = set()
    example_count = 0
    for record in examples:
        new_codes = [i for i in record["quality_issues"] if i["code"] not in seen_codes]
        if not new_codes:
            continue
        issue = new_codes[0]
        lines.append(f"- `{record['chunk_id']}` (page {record['page_number']}): `{issue['code']}` ({issue['severity']}) — {issue['detail']}")
        seen_codes.update(i["code"] for i in record["quality_issues"])
        example_count += 1
        if example_count >= 12:
            break
    if not example_count:
        lines.append("- No quality warnings were detected by the configured checks.")

    lines.extend(["", "## Shortlist summary", "", "| Source | Recommended candidates | Existing/project-approved | Pending review | Rights blocked |", "|---|---:|---:|---:|---:|"])
    for document_id, data in results["document_results"].items():
        selected = [r for r in results["selected_candidates"] if r["document_id"] == document_id]
        approved = sum(r["approved"] for r in selected)
        blocked = sum(r["rights_gate"].startswith("blocked") for r in selected)
        lines.append(f"| {data['title']} | {len(selected)} | {approved} | {len(selected)-approved} | {blocked} |")
    lines.extend([
        "",
        "## Recommendations and blockers",
        "",
        "- Review blocking quality issues before selection; inspect warnings in the PDF for sentence context, table/chart reading order, and broken words.",
        "- Resolve BA reuse terms before any external processing; its shortlisted chunks are not ingestible while marked `blocked_pending_reuse_policy_clarification`.",
        "- Keep Cedefop and European Commission attribution and document-level limitations attached to retrieved chunks; review third-party material independently.",
        "- The selector is deterministic and source-balanced, not a fixed-size quota. A shortlist below 60 is retained if the quality/relevance gates support fewer candidates.",
        "- Machine-readable candidate metadata is in `knowledge_base_shortlist.json`. The source JSONL, approvals, fingerprints, and indexed Pinecone records are not rewritten by the audit.",
        "",
    ])
    return "\n".join(lines)
