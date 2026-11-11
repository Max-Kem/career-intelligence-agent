from __future__ import annotations

import json

import pytest

from career_agent.document_chunking import chunk_page_text, chunk_processed_directory


def write_page_fixture(directory, *, pages, document_overrides=None):
    document = {
        "document_id": "sample_report",
        "filename": "sample_report.pdf",
        "title": "Sample report",
        "publisher": "Sample publisher",
        "publication_year": 2025,
        "source_url": "https://example.org/report.pdf",
        "source_url_status": "verified",
        "page_count": max((page.get("page_number", 0) for page in pages if isinstance(page, dict)), default=0),
        "extraction_status": "succeeded",
        "page_jsonl": "sample_report.jsonl",
        "licensing_status": "unverified",
        "license": None,
    }
    document.update(document_overrides or {})
    (directory / "manifest.json").write_text(json.dumps({"documents": [document]}), encoding="utf-8")
    (directory / "sample_report.jsonl").write_text(
        "".join(json.dumps(page) + "\n" for page in pages), encoding="utf-8",
    )


def page_record(page_number, text, document_id="sample_report"):
    return {
        "document_id": document_id,
        "filename": "sample_report.pdf",
        "page_number": page_number,
        "text": text,
        "character_count": len(text),
        "extraction_status": "extracted" if text.strip() else "empty",
    }


def test_paragraph_boundaries_and_overlap_are_preserved_within_page():
    first_paragraph = "First paragraph sentence. " + ("alpha " * 8)
    second_paragraph = "Second paragraph sentence. " + ("bravo " * 8)
    text = first_paragraph.strip() + "\n\n" + second_paragraph.strip()

    chunks = chunk_page_text(text, target_chars=100, overlap_chars=15)

    assert len(chunks) == 2
    assert chunks[0] == first_paragraph.strip()
    assert chunks[1] in text
    assert len(chunks[0]) <= 100 and len(chunks[1]) <= 100
    start_second = text.index(chunks[1])
    assert start_second < len(chunks[0])  # page-local overlap exists


def test_long_paragraph_splits_near_sentence_or_word_boundaries_without_loss():
    sentences = [f"Sentence {index} has several words and ends cleanly." for index in range(12)]
    text = " ".join(sentences)
    chunks = chunk_page_text(text, target_chars=120, overlap_chars=20)

    assert len(chunks) > 3
    assert all(chunk and len(chunk) <= 120 for chunk in chunks)
    starts = [text.find(chunk) for chunk in chunks]
    assert starts[0] == 0
    assert all(left < right for left, right in zip(starts, starts[1:]))
    for left, right, previous in zip(starts, starts[1:], chunks):
        assert right < left + len(previous)  # no gap; adjacent chunks overlap
    assert text.endswith(chunks[-1].split()[-1])


def test_empty_pages_are_skipped_but_short_nonempty_content_is_retained(tmp_path):
    write_page_fixture(tmp_path, pages=[page_record(1, "  "), page_record(2, "brief fact")])
    summary = chunk_processed_directory(tmp_path, target_chars=200, overlap_chars=20, short_chunk_chars=30)
    chunks = [json.loads(line) for line in (tmp_path / "sample_report.chunks.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["skipped_empty_sections"] == 1
    assert len(chunks) == 1
    assert chunks[0]["text"] == "brief fact"
    assert chunks[0]["page_number"] == 2
    assert summary["quality_checks"]["suspiciously_short_chunks"] == 1
    assert any("short chunk" in warning for warning in summary["warnings"])


def test_metadata_page_citations_and_chunk_ids_are_stable(tmp_path):
    text = "A cited paragraph with sufficient text. " + ("career evidence " * 20)
    write_page_fixture(tmp_path, pages=[page_record(3, text)], document_overrides={"page_count": 5})
    first_summary = chunk_processed_directory(tmp_path, target_chars=150, overlap_chars=20)
    first_bytes = (tmp_path / "sample_report.chunks.jsonl").read_bytes()
    second_summary = chunk_processed_directory(tmp_path, target_chars=150, overlap_chars=20)
    second_bytes = (tmp_path / "sample_report.chunks.jsonl").read_bytes()
    chunks = [json.loads(line) for line in second_bytes.decode("utf-8").splitlines()]

    assert first_bytes == second_bytes
    assert first_summary == second_summary
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["document_id"] == "sample_report" for chunk in chunks)
    assert all(chunk["document_title"] == "Sample report" for chunk in chunks)
    assert all(chunk["publisher"] == "Sample publisher" for chunk in chunks)
    assert all(chunk["publication_year"] == 2025 for chunk in chunks)
    assert all(chunk["source_url"] == "https://example.org/report.pdf" for chunk in chunks)
    assert all(chunk["source_filename"] == "sample_report.pdf" for chunk in chunks)
    assert all(chunk["page_number"] == chunk["page_start"] == chunk["page_end"] == 3 for chunk in chunks)
    assert all(chunk["licensing_status"] == "unverified" for chunk in chunks)


def test_malformed_records_bad_references_and_invalid_pages_are_reported(tmp_path):
    write_page_fixture(tmp_path, pages=[page_record(1, "valid text")])
    page_path = tmp_path / "sample_report.jsonl"
    with page_path.open("a", encoding="utf-8") as stream:
        stream.write("{broken json}\n")
        stream.write(json.dumps(page_record(2, "wrong doc", "other_document")) + "\n")
        stream.write(json.dumps(page_record(0, "zero page")) + "\n")
    summary = chunk_processed_directory(tmp_path)

    checks = summary["quality_checks"]
    assert checks["malformed_page_records"] == 1
    assert checks["missing_document_references"] == 1
    assert checks["invalid_page_numbers"] == 1
    assert summary["total_chunks"] == 1
    assert len(summary["warnings"]) >= 3


def test_empty_manifest_produces_empty_summary_and_invalid_settings_fail(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"documents": []}), encoding="utf-8")
    summary = chunk_processed_directory(tmp_path)
    assert summary["documents_processed"] == 0
    assert summary["total_chunks"] == 0
    with pytest.raises(ValueError):
        chunk_processed_directory(tmp_path, target_chars=100, overlap_chars=50)
