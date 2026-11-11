"""Local, page-preserving PDF text extraction for the RAG source documents."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from career_agent.document_rights import RIGHTS_PREFLIGHT


DOCUMENT_METADATA: dict[str, dict[str, Any]] = {
    "ba_ikt_arbeitsmarkt_2026": {
        "title": "Der Arbeitsmarkt für IKT-Berufe im Kontext der Transformation",
        "publisher": "Bundesagentur für Arbeit",
        "publication_year": 2026,
        "source_url": "https://statistik.arbeitsagentur.de/DE/Statischer-Content/Statistiken/Themen-im-Fokus/Berufe/Generische-Publikationen/AM-kompakt-IKT.pdf?__blob=publicationFile",
        "source_url_status": "verified",
        "license": None,
        "licensing_status": "unverified",
        "license_source_url": None,
    },
    "cedefop_skills_forecast_germany_2025": {
        "title": "Germany - 2025 Skills forecast",
        "publisher": "European Centre for the Development of Vocational Training (Cedefop)",
        "publication_year": 2025,
        "source_url": "https://www.cedefop.europa.eu/en/country-reports/germany-2025-skills-forecast",
        "source_url_status": "verified",
        "license": None,
        "licensing_status": "unverified",
        "license_source_url": None,
    },
    "eu_digital_decade_germany_2026": {
        "title": "Germany - Digital Decade Country Report 2026",
        "publisher": "European Commission",
        "publication_year": 2026,
        "source_url": "https://digital-strategy.ec.europa.eu/en/factpages/germanys-2026-digital-decade-country-report",
        "source_url_status": "verified",
        "license": None,
        "licensing_status": "unverified",
        "license_source_url": None,
    },
    "ilo_generative_ai_jobs_2025": {
        "title": "Generative AI and Jobs: A Refined Global Index of Occupational Exposure",
        "publisher": "International Labour Organization",
        "publication_year": 2025,
        "source_url": "https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure",
        "source_url_status": "verified",
        "license": "CC BY 4.0",
        "licensing_status": "verified",
        "license_source_url": "https://www.ilo.org/sites/default/files/2025-05/WP140_web.pdf",
    },
}

SPARSE_PAGE_CHAR_THRESHOLD = 100


def extract_directory(raw_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Extract all PDFs under ``raw_dir`` into one page-level JSONL per PDF."""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_path.glob("*.pdf")) if raw_path.exists() else []
    documents = [_extract_one(path, output_path) for path in files]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "documents": documents,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _extract_one(path: Path, output_dir: Path) -> dict[str, Any]:
    document_id = _document_id(path.stem)
    metadata = DOCUMENT_METADATA.get(document_id, {})
    page_records: list[dict[str, Any]] = []
    extraction_errors: list[dict[str, Any]] = []
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ValueError("PDF is encrypted and has no empty-password access")
            except Exception as exc:
                return _failed_document(path, document_id, metadata, f"encrypted_or_unreadable: {type(exc).__name__}")
        page_count = len(reader.pages)
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = text.replace("\x00", "").strip()
                page_records.append({
                    "document_id": document_id,
                    "filename": path.name,
                    "page_number": index,
                    "text": text,
                    "character_count": len(text),
                    "extraction_status": "empty" if not text else "extracted",
                })
                if not text:
                    extraction_errors.append({"page_number": index, "reason": "no extractable text; may be blank, scanned, or unreadable"})
            except Exception as exc:
                extraction_errors.append({"page_number": index, "reason": f"page extraction failed: {type(exc).__name__}"})
                page_records.append({
                    "document_id": document_id,
                    "filename": path.name,
                    "page_number": index,
                    "text": "",
                    "character_count": 0,
                    "extraction_status": "error",
                })
    except Exception as exc:
        return _failed_document(path, document_id, metadata, f"pdf_read_failed: {type(exc).__name__}")

    if page_records:
        jsonl_path = output_dir / f"{document_id}.jsonl"
        temp_path = jsonl_path.with_suffix(".jsonl.tmp")
        temp_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in page_records),
            encoding="utf-8",
        )
        temp_path.replace(jsonl_path)
    else:
        jsonl_path = None

    empty_pages = [record["page_number"] for record in page_records if not record["character_count"]]
    sparse_pages = [
        record["page_number"] for record in page_records
        if 0 < record["character_count"] < SPARSE_PAGE_CHAR_THRESHOLD
    ]
    if page_count == 0:
        status = "empty_document"
    elif extraction_errors:
        status = "partial"
    else:
        status = "succeeded"

    pdf_info = reader.metadata or {}
    title = metadata.get("title") or _optional_string(pdf_info.get("/Title"))
    publisher = metadata.get("publisher") or None
    return {
        "document_id": document_id,
        "filename": path.name,
        "title": title,
        "publisher": publisher,
        "publication_year": metadata.get("publication_year"),
        "source_url": metadata.get("source_url"),
        "source_url_status": metadata.get("source_url_status", "unverified" if metadata.get("source_url") else "unavailable"),
        "page_count": page_count,
        "extraction_status": status,
        "extracted_character_count": sum(record["character_count"] for record in page_records),
        "page_jsonl": jsonl_path.name if jsonl_path else None,
        "empty_or_unreadable_pages": empty_pages,
        "sparse_text_pages_under_100_chars": sparse_pages,
        "page_issues": extraction_errors,
        "licensing_status": metadata.get("licensing_status", "unverified"),
        "license": metadata.get("license"),
        "license_source_url": metadata.get("license_source_url"),
        "rights_preflight": RIGHTS_PREFLIGHT.get(document_id),
    }


def _failed_document(path: Path, document_id: str, metadata: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "filename": path.name,
        "title": metadata.get("title"),
        "publisher": metadata.get("publisher"),
        "publication_year": metadata.get("publication_year"),
        "source_url": metadata.get("source_url"),
        "source_url_status": metadata.get("source_url_status", "unverified" if metadata.get("source_url") else "unavailable"),
        "page_count": None,
        "extraction_status": "error",
        "extracted_character_count": 0,
        "page_jsonl": None,
        "empty_or_unreadable_pages": [],
        "sparse_text_pages_under_100_chars": [],
        "page_issues": [{"reason": error}],
        "licensing_status": metadata.get("licensing_status", "unverified"),
        "license": metadata.get("license"),
        "license_source_url": metadata.get("license_source_url"),
        "rights_preflight": RIGHTS_PREFLIGHT.get(document_id),
    }


def inspect_processed(output_dir: str | Path) -> dict[str, Any]:
    """Return manifest diagnostics and JSONL counts without displaying page text."""
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for document in manifest.get("documents", []):
        relative_path = document.get("page_jsonl")
        if not relative_path:
            document["stored_page_records"] = 0
            continue
        page_file = output_path / relative_path
        document["stored_page_records"] = sum(1 for line in page_file.open(encoding="utf-8") if line.strip())
    return manifest


def _document_id(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", stem.casefold()).strip("_")


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
