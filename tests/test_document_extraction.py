from __future__ import annotations

import json

from career_agent import document_extraction
from career_agent.document_extraction import extract_directory, inspect_processed


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, pages, metadata=None):
        self.pages = pages
        self.metadata = metadata or {}
        self.is_encrypted = False


def test_extracts_page_records_with_stable_document_and_page_references(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    source = raw_dir / "sample_document_2025.pdf"
    source.write_bytes(b"mock pdf")
    monkeypatch.setattr(document_extraction, "PdfReader", lambda *_args, **_kwargs: FakeReader([
        FakePage("Page one text."), FakePage("Page two text."),
    ], {"/Title": "Sample title"}))

    manifest = extract_directory(raw_dir, output_dir)
    document = manifest["documents"][0]
    records = [json.loads(line) for line in (output_dir / document["page_jsonl"]).read_text(encoding="utf-8").splitlines()]

    assert document["document_id"] == "sample_document_2025"
    assert document["title"] == "Sample title"
    assert document["page_count"] == 2
    assert document["extraction_status"] == "succeeded"
    assert document["extracted_character_count"] == len("Page one text.Page two text.")
    assert [(record["document_id"], record["page_number"]) for record in records] == [
        ("sample_document_2025", 1), ("sample_document_2025", 2),
    ]
    assert records[0]["text"] == "Page one text."
    assert records[1]["text"] == "Page two text."


def test_empty_pages_are_recorded_and_reported_as_partial(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "empty_page.pdf").write_bytes(b"mock pdf")
    monkeypatch.setattr(document_extraction, "PdfReader", lambda *_args, **_kwargs: FakeReader([
        FakePage(""), FakePage("\n   "),
    ]))

    manifest = extract_directory(raw_dir, output_dir)
    document = manifest["documents"][0]
    record_lines = (output_dir / document["page_jsonl"]).read_text(encoding="utf-8").splitlines()

    assert document["extraction_status"] == "partial"
    assert document["empty_or_unreadable_pages"] == [1, 2]
    assert [json.loads(line)["extraction_status"] for line in record_lines] == ["empty", "empty"]
    assert "may be blank, scanned, or unreadable" in document["page_issues"][0]["reason"]


def test_zero_page_document_and_unreadable_pdf_are_reported(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "zero_page.pdf").write_bytes(b"mock pdf")
    (raw_dir / "bad_pdf.pdf").write_bytes(b"not a pdf")

    def fake_reader(path, **_kwargs):
        if str(path).endswith("bad_pdf.pdf"):
            raise ValueError("malformed test pdf")
        return FakeReader([])

    monkeypatch.setattr(document_extraction, "PdfReader", fake_reader)
    manifest = extract_directory(raw_dir, output_dir)
    by_name = {item["filename"]: item for item in manifest["documents"]}

    assert by_name["zero_page.pdf"]["extraction_status"] == "empty_document"
    assert by_name["zero_page.pdf"]["page_count"] == 0
    assert by_name["bad_pdf.pdf"]["extraction_status"] == "error"
    assert by_name["bad_pdf.pdf"]["page_issues"][0]["reason"] == "pdf_read_failed: ValueError"


def test_inspection_reports_record_counts_without_exposing_page_text(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "inspect_me.pdf").write_bytes(b"mock pdf")
    monkeypatch.setattr(document_extraction, "PdfReader", lambda *_args, **_kwargs: FakeReader([FakePage("private extracted text")]))
    extract_directory(raw_dir, output_dir)

    inspected = inspect_processed(output_dir)
    assert inspected["documents"][0]["stored_page_records"] == 1
    assert "private extracted text" not in json.dumps(inspected)
