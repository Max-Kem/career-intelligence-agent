"""Document-specific reuse preflight metadata for local source PDFs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RIGHTS_PREFLIGHT: dict[str, dict[str, Any]] = {
    "ba_ikt_arbeitsmarkt_2026": {
        "review_date": "2026-11-10",
        "document_notice_status": "confirmed_in_pdf",
        "document_notice_page": 2,
        "license_or_reuse_conditions": "No standard open license is stated. The Impressum permits reuse of data and tables without restriction; other information may be stored, shared, copied, reproduced, and distributed with source citation. Contents must not be changed or falsified. Own calculations are permitted if identified; online publication should link to the BA Statistics homepage.",
        "attribution_requirements": "Cite Statistik der Bundesagentur fuer Arbeit and the report title; for online publication, link to the BA Statistics homepage.",
        "policy_source_url": "https://statistik.arbeitsagentur.de/DE/Statischer-Content/Statistiken/Themen-im-Fokus/Berufe/Generische-Publikationen/AM-kompakt-IKT.pdf?__blob=publicationFile",
        "third_party_rights": "No separate third-party notice identified in the inspected Impressum; rights in embedded third-party material are not independently established.",
        "external_embedding_status": "additional_review_required",
        "review_reason": "The reuse notice does not expressly address transformation into embeddings or third-party vector storage; confirm scope and any third-party component rights before external processing.",
    },
    "cedefop_skills_forecast_germany_2025": {
        "review_date": "2026-11-10",
        "document_notice_status": "no_document_specific_notice_found",
        "license_or_reuse_conditions": "No document-specific reuse notice was found in inspected PDF pages. Cedefop's general policy authorizes reproduction with source acknowledgment unless otherwise stated; any specific prior-permission notice overrides that general permission.",
        "attribution_requirements": "Acknowledge Cedefop and cite the report title and source URL, subject to any document-specific restriction discovered later.",
        "policy_source_url": "https://www26.cedefop.europa.eu/en/content/legal-noticedisclaimer",
        "third_party_rights": "Not independently established for this report.",
        "external_embedding_status": "additional_review_required",
        "review_reason": "General Cedefop reproduction terms are not a document-specific license; verify that no special restriction or third-party right applies before external processing and storage.",
    },
    "eu_digital_decade_germany_2026": {
        "review_date": "2026-11-10",
        "document_notice_status": "no_document_specific_notice_found",
        "license_or_reuse_conditions": "No document-specific reuse notice was found in inspected PDF pages. The European Commission's general Digital Strategy policy applies CC BY 4.0 to EU-owned content unless otherwise indicated; appropriate credit and indication of changes are required. Third-party works and other excluded material are not covered.",
        "attribution_requirements": "Credit the European Union and cite the work title and published Digital Strategy source URL; indicate changes and do not imply Commission endorsement.",
        "policy_source_url": "https://digital-strategy.ec.europa.eu/en/pages/legal-notice",
        "third_party_rights": "Unresolved; the general policy excludes third-party works and content for which the Commission cannot grant reuse.",
        "external_embedding_status": "additional_review_required",
        "review_reason": "The policy is a publisher-level default rather than a PDF-specific license; check report components for third-party rights before external processing and storage.",
    },
    "ilo_generative_ai_jobs_2025": {
        "review_date": "2026-11-10",
        "document_notice_status": "confirmed_in_pdf",
        "document_notice_page": 2,
        "license_or_reuse_conditions": "The PDF states CC BY 4.0. Attribute the work to the ILO and cite its authors, title, ILO Working Paper 140, Geneva, and 2025; indicate changes. The ILO emblem is not licensed, and third-party material is excluded and may require separate permission.",
        "attribution_requirements": "International Labour Office, Generative AI and Jobs: A Refined Global Index of Occupational Exposure, ILO Working Paper 140, Geneva, 2025; indicate changes.",
        "policy_source_url": "https://www.ilo.org/global/copyright/lang--en/index.htm",
        "third_party_rights": "Explicitly excluded from the PDF's CC BY 4.0 grant; check and clear any third-party components individually. The ILO name/emblem may not be used to imply endorsement.",
        "external_embedding_status": "conditional_on_third_party_review",
        "review_reason": "CC BY 4.0 supports reuse of covered ILO material, but third-party content is excluded and must be checked before external processing.",
    },
}


def update_manifest_rights_preflight(processed_dir: str | Path) -> dict[str, Any]:
    """Update bibliographic corrections and rights fields without touching JSONL data."""
    manifest_path = Path(processed_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_fields = {
        "ba_ikt_arbeitsmarkt_2026": {
            "title": "Der Arbeitsmarkt f\u00fcr IKT-Berufe im Kontext der Transformation",
            "publisher": "Bundesagentur f\u00fcr Arbeit",
        },
        "eu_digital_decade_germany_2026": {
            "title": "Germany - Digital Decade Country Report 2026",
        },
    }
    for document in manifest.get("documents", []):
        document_id = document.get("document_id")
        document.update(metadata_fields.get(document_id, {}))
        if document_id in RIGHTS_PREFLIGHT:
            document["rights_preflight"] = RIGHTS_PREFLIGHT[document_id]
            if document_id != "ilo_generative_ai_jobs_2025":
                document["license_source_url"] = RIGHTS_PREFLIGHT[document_id]["policy_source_url"]
            if document_id == "ilo_generative_ai_jobs_2025":
                document["licensing_status"] = "verified_in_pdf"
                document["license"] = "CC BY 4.0"
            elif document_id == "ba_ikt_arbeitsmarkt_2026":
                document["licensing_status"] = "document_reuse_terms_confirmed_no_standard_license"
                document["license"] = None
            else:
                document["licensing_status"] = "publisher_policy_verified_document_license_unverified"
                document["license"] = None
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
