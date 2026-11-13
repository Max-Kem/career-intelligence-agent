"""Local, deterministic PDF presentation for completed career reports."""

from __future__ import annotations

import html
import io
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# Match the Streamlit palette while keeping interior pages high contrast on paper.
NAVY = colors.HexColor("#07151C")
NAVY_PANEL = colors.HexColor("#0D2430")
TEAL = colors.HexColor("#0F766E")
DEEP_TEAL = colors.HexColor("#173B42")
MINT = colors.HexColor("#1EEA91")
TURQUOISE = colors.HexColor("#48E0D0")
PAPER = colors.HexColor("#FBFDFC")
PALE_TEAL = colors.HexColor("#EAF7F4")
PALE_BLUE = colors.HexColor("#EFF5F6")
INK = colors.HexColor("#183039")
MUTED = colors.HexColor("#526970")
RULE = colors.HexColor("#D2E0E2")
AMBER = colors.HexColor("#A46A0A")
PAGE_SIZE = A4
_FONT = "Helvetica"


class _StableCanvas(Canvas):
    """Make identical input report data produce stable PDF metadata and bytes."""

    def __init__(self, *args, **kwargs):
        kwargs["invariant"] = 1
        super().__init__(*args, **kwargs)


class _Pill(Flowable):
    """Small rounded score/status badge with measured text bounds."""

    def __init__(self, text: str, *, fill=DEEP_TEAL, foreground=colors.white, font_size=8.5, pad_x=8, pad_y=5):
        super().__init__()
        self.text = text
        self.fill = fill
        self.foreground = foreground
        self.font_size = font_size
        self.pad_x = pad_x
        self.pad_y = pad_y

    def wrap(self, avail_width, avail_height):
        self.width = min(avail_width, pdfmetrics.stringWidth(self.text, "Helvetica-Bold", self.font_size) + 2 * self.pad_x)
        self.height = self.font_size + 2 * self.pad_y
        return self.width, self.height

    def draw(self):
        self.canv.setFillColor(self.fill)
        self.canv.roundRect(0, 0, self.width, self.height, self.height / 2, fill=1, stroke=0)
        self.canv.setFillColor(self.foreground)
        self.canv.setFont("Helvetica-Bold", self.font_size)
        self.canv.drawCentredString(self.width / 2, self.pad_y + 1, self.text)


def _register_font() -> str:
    global _FONT
    candidates = (
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"), Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")),
    )
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            try:
                pdfmetrics.registerFont(TTFont("CareerSans", str(regular)))
                pdfmetrics.registerFont(TTFont("CareerSans-Bold", str(bold)))
                pdfmetrics.registerFontFamily("CareerSans", normal="CareerSans", bold="CareerSans-Bold", italic="CareerSans", boldItalic="CareerSans-Bold")
                _FONT = "CareerSans"
                break
            except (OSError, ValueError):
                continue
    return _FONT


def _safe(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    value = str(value)
    value = (value.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "—")
             .replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"')
             .replace("\u201d", '"').replace("\ufffd", ""))
    if limit is not None and len(value) > limit:
        value = value[:limit - 1].rstrip() + "…"
    return value


def _p(value: Any, style: ParagraphStyle, *, limit: int | None = None) -> Paragraph:
    safe = html.escape(_safe(value, limit), quote=False).replace("\n", "<br/>")
    return Paragraph(safe or " ", style)


def _url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _link(value: Any, label: str, style: ParagraphStyle, *, limit: int = 180) -> Paragraph:
    url = _url(value)
    if not url:
        return _p(label, style, limit=limit)
    return Paragraph(f'<link href="{html.escape(url, quote=True)}" color="#0F766E">{html.escape(_safe(label, limit))}</link>', style)


def _styles() -> dict[str, ParagraphStyle]:
    _register_font()
    sheet = getSampleStyleSheet()
    bold_font = "CareerSans-Bold" if _FONT == "CareerSans" else "Helvetica-Bold"
    body = ParagraphStyle("CIA_Body", parent=sheet["BodyText"], fontName=_FONT, fontSize=9, leading=12.2, textColor=INK, spaceAfter=3, splitLongWords=1, allowWidows=0, allowOrphans=0)
    return {
        "cover_kicker": ParagraphStyle("CIA_Kicker", parent=body, fontName=bold_font, fontSize=8, leading=10, textColor=MINT, spaceAfter=5),
        "cover_title": ParagraphStyle("CIA_CoverTitle", parent=body, fontName=bold_font, fontSize=25, leading=29, textColor=colors.white, spaceAfter=6),
        "cover_subtitle": ParagraphStyle("CIA_CoverSub", parent=body, fontSize=10, leading=14, textColor=colors.HexColor("#D6EAE8"), spaceAfter=0),
        "h1": ParagraphStyle("CIA_H1", parent=sheet["Heading1"], fontName=bold_font, fontSize=17, leading=20, textColor=NAVY, spaceBefore=5, spaceAfter=7, keepWithNext=True),
        "h2": ParagraphStyle("CIA_H2", parent=sheet["Heading2"], fontName=bold_font, fontSize=11, leading=14, textColor=TEAL, spaceBefore=4, spaceAfter=3, keepWithNext=True),
        "h3": ParagraphStyle("CIA_H3", parent=sheet["Heading3"], fontName=bold_font, fontSize=9.5, leading=12, textColor=NAVY, spaceBefore=2, spaceAfter=2, keepWithNext=True),
        "body": body,
        "small": ParagraphStyle("CIA_Small", parent=body, fontSize=7.8, leading=10, textColor=MUTED, spaceAfter=2),
        "tiny": ParagraphStyle("CIA_Tiny", parent=body, fontSize=7, leading=8.6, textColor=MUTED, spaceAfter=1),
        "label": ParagraphStyle("CIA_Label", parent=body, fontName=bold_font, fontSize=7.2, leading=9, textColor=MUTED, spaceAfter=1),
        "table_header": ParagraphStyle("CIA_TableHeader", parent=body, fontName=bold_font, fontSize=7.2, leading=9, textColor=colors.white, spaceAfter=1),
        "score": ParagraphStyle("CIA_Score", parent=body, fontName=bold_font, fontSize=16, leading=18, textColor=TEAL, alignment=TA_LEFT, spaceAfter=0),
        "score_label": ParagraphStyle("CIA_ScoreLabel", parent=body, fontName=bold_font, fontSize=6.8, leading=8, textColor=MUTED, spaceAfter=0),
        "cover_body": ParagraphStyle("CIA_CoverBody", parent=body, textColor=colors.white, fontSize=9, leading=12),
        "callout": ParagraphStyle("CIA_Callout", parent=body, backColor=PALE_TEAL, borderColor=TEAL, borderWidth=0.7, borderPadding=7, borderRadius=5, spaceBefore=3, spaceAfter=4),
        "warning": ParagraphStyle("CIA_Warning", parent=body, backColor=PALE_BLUE, borderColor=AMBER, borderWidth=0.7, borderPadding=6, borderRadius=4, spaceBefore=2, spaceAfter=3),
    }


def _rounded_table(rows: list[list[Any]], widths: list[float], *, background=colors.white, stroke=RULE, radius=7, padding=7, split_rows=1) -> Table:
    radii = [radius] * 4 if isinstance(radius, (int, float)) else radius
    table = Table(rows, colWidths=widths, hAlign="LEFT", splitByRow=1, splitInRow=split_rows, cornerRadii=radii)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.7, stroke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding), ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding), ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
    ]))
    return table


def _section(title: str, styles: dict[str, ParagraphStyle], eyebrow: str | None = None) -> list[Any]:
    result: list[Any] = []
    if eyebrow:
        result.append(_p(eyebrow.upper(), styles["cover_kicker"]))
    result.extend([_p(title, styles["h1"]), HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=7)])
    return result


def _plain_status(status: Any, feature: str) -> str:
    key = str(status or "unknown").casefold()
    if key in {"succeeded", "available", "completed"}:
        return f"{feature} is available in this report."
    if key in {"skipped_no_recommendations"}:
        return f"{feature} was not run because no roles qualified for recommendation."
    if key in {"skipped_offline"}:
        return f"{feature} was not run in offline mode."
    if key in {"skipped_disabled"}:
        return f"{feature} was turned off for this run."
    if key in {"skipped_opt_in_required"}:
        return f"{feature} was not enabled for this run."
    if key in {"not_requested"}:
        return f"{feature} was not requested for this run."
    if key in {"unavailable_no_evidence"}:
        return "No sufficiently complete source passages were available for this section."
    if key in {"unavailable", "failed"}:
        return f"{feature} could not be retrieved; no substitute information was added."
    if key == "partial":
        return f"{feature} is only partly available; use the linked source material for context."
    if key == "no_results":
        return f"{feature} returned no usable results."
    return f"{feature} status was not recorded."


def _evidence_status(item: dict) -> str:
    value = str(item.get("candidate_evidence_status") or "not_documented").casefold()
    return {
        "not_documented": "Not documented — verify",
        "partially_documented": "Partly documented — verify level/context",
        "does_not_meet_requirement": "Explicit mismatch in supplied evidence",
        "meets_requirement": "Documented in profile",
    }.get(value, "Evidence status unclear — verify")


def _concise_strengths(item: dict) -> list[str]:
    """Remove duplicate matching phrases without changing their meaning."""
    output: list[str] = []
    seen: set[str] = set()
    for value in item.get("strengths") or []:
        text = _safe(value).strip().rstrip(".")
        lower = text.casefold()
        if lower.startswith("matched listed skill: "):
            text = f"Documented skill: {text.split(':', 1)[1].strip()}"
        elif lower.startswith("candidate lists "):
            # This repeats the matched-listed-skill evidence; the concise label above is enough.
            continue
        elif lower.startswith("candidate reports "):
            text = text.replace("Candidate reports", "Experience evidence").replace("; job description asks for", "; posting asks for")
            text = re.sub(r"\basks for 1 years\b", "asks for 1 year", text)
        elif lower.startswith("job is explicitly eligible"):
            text = "Location eligibility is confirmed for a preferred country."
        key = text.casefold()
        if key and key not in seen:
            seen.add(key)
            output.append(text)
        if len(output) == 3:
            break
    return output


def _requirement_compact(item: dict, styles: dict[str, ParagraphStyle]) -> list[Any]:
    req = _safe(item.get("requirement") or "Requirement not specified", 180)
    status = _evidence_status(item)
    requirement_type = str(item.get("requirement_type") or "not_specified").replace("_", " ").strip().casefold()
    type_label = {"required": "Required", "preferred": "Preferred", "optional": "Optional"}.get(requirement_type, "Posting does not specify priority")
    posting = _safe(item.get("posting_evidence") or "Posting excerpt not recorded.", 260)
    return [
        _p(f"{req} · {type_label} · {status}", styles["small"]),
        _p(f"Posting: {posting}", styles["tiny"]),
    ]


def _recommendation_card(item: dict, styles: dict[str, ParagraphStyle], width: float) -> Table:
    job = item.get("job") or {}
    title = _safe(job.get("title") or "Position", 150)
    company = _safe(job.get("company") or "Company not provided", 110)
    score = item.get("score")
    evidence = item.get("evidence_coverage")
    role_link = _link(job.get("source_url"), f"{title}  [view posting]", styles["h2"], limit=200)
    score_table = Table([
        [_p(f"{score if score is not None else '—'}", styles["score"]), _p(f"{evidence if evidence is not None else '—'}%", styles["score"])],
        [_p("INDICATIVE ALIGNMENT", styles["score_label"]), _p("EVIDENCE COVERAGE", styles["score_label"])],
    ], colWidths=[38 * mm, 38 * mm], hAlign="LEFT")
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL), ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    location_values = [_safe(v) for v in (job.get("location"), job.get("work_model")) if v]
    if len(location_values) == 2 and location_values[0].casefold() == location_values[1].casefold():
        location_values.pop()
    location = " · ".join(location_values)
    rows: list[list[Any]] = [
        [role_link],
        [_p(f"{company}{'  ·  ' + location if location else ''}", styles["small"])],
        [score_table],
    ]
    strengths = _concise_strengths(item)
    if strengths:
        rows.append([_p("WHY IT ALIGNS", styles["label"])])
        for strength in strengths:
            rows.append([_p(f"• {strength}", styles["small"])])
    reqs = item.get("requirements_to_verify") or []
    if reqs:
        rows.append([_p("TO VERIFY", styles["label"])])
        for requirement in reqs:
            rows.extend([[flow] for flow in _requirement_compact(requirement, styles)])
    if item.get("gaps"):
        rows.append([_p("ROLE FIT NOTE", styles["label"])])
        for gap in item.get("gaps")[:2]:
            rows.append([_p(f"• {_safe(gap, 220)}", styles["small"])])
    rows.append([_p("Score reflects evidence alignment; it is not hiring probability or proof that every qualification is met.", styles["tiny"])])
    return _rounded_table(rows, [width], background=colors.white, stroke=RULE, radius=8, padding=7, split_rows=1)


def _review_reason(item: dict) -> str:
    reasons = item.get("reasons") or []
    for reason in reasons:
        text = _safe(reason).strip()
        if text.startswith("Material job requirement needs verification") or text == "Evidence status: provisional.":
            continue
        text = text.replace("This is not a finding that the candidate lacks it.", "")
        text = text.replace("This is not a conclusion that the candidate cannot perform the work.", "")
        text = " ".join(text.split()).strip()
        if text:
            return _safe(text, 190)
    reqs = item.get("requirements_to_verify") or []
    if reqs:
        return f"{_safe(reqs[0].get('requirement') or 'A material requirement')} needs verification; profile evidence is {_evidence_status(reqs[0]).lower()}."
    return "Evidence did not support recommendation qualification."


def _review_index(rows: list[dict], styles: dict[str, ParagraphStyle], widths: list[float]) -> LongTable:
    data: list[list[Any]] = [[_p("ROLE", styles["table_header"]), _p("SCORE", styles["table_header"]), _p("WHY IT NEEDS REVIEW", styles["table_header"])]]
    for item in rows:
        job = item.get("job") or {}
        role = f"{_safe(job.get('title') or 'Position', 105)}<br/><font color='#526970'>{_safe(job.get('company') or 'Company not provided', 75)}</font>"
        data.append([
            Paragraph(role, styles["small"]),
            _p(f"{item.get('score', '—')}/100", styles["h3"]),
            _p(_review_reason(item), styles["small"]),
        ])
    table = LongTable(data, colWidths=widths, repeatRows=1, splitByRow=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DEEP_TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _append_review_appendix(story: list[Any], reviews: list[dict], styles: dict[str, ParagraphStyle], width: float) -> None:
    story.append(PageBreak())
    story.extend(_section("Review evidence appendix", styles, "Decision support"))
    story.append(_p("The following entries preserve the recorded review reasons and requirement evidence. “Not documented” means the profile supplied no evidence; it is not a confirmed mismatch. Only an explicitly recorded mismatch is described as such.", styles["body"]))
    for index, item in enumerate(reviews, start=1):
        job = item.get("job") or {}
        title = _safe(job.get("title") or "Position", 140)
        company = _safe(job.get("company") or "Company not provided", 100)
        heading = _link(job.get("source_url"), f"{index}. {title} — {company}", styles["h3"], limit=220)
        score = item.get("score")
        coverage = item.get("evidence_coverage")
        score_line = f"Indicative score: {score if score is not None else 'Not scored'} / 100"
        if coverage is not None:
            score_line += f"  ·  Evidence coverage: {coverage}%"
        details: list[list[Any]] = [[heading], [_p(score_line, styles["tiny"])]]
        reasons = [r for r in (item.get("reasons") or []) if not str(r).startswith("Material job requirement needs verification") and str(r) != "Evidence status: provisional."]
        if reasons:
            details.append([_p("Review context", styles["label"])])
            for reason in reasons:
                details.append([_p(f"• {_safe(reason, 360)}", styles["tiny"])])
        requirements = item.get("requirements_to_verify") or []
        if requirements:
            details.append([_p("Material requirements from the posting", styles["label"])])
            for req in requirements:
                requirement_type = str(req.get("requirement_type") or "not_specified").replace("_", " ").strip().casefold()
                type_label = {"required": "required", "preferred": "preferred", "optional": "optional"}.get(requirement_type, "priority not specified")
                details.append([_p(f"{_safe(req.get('requirement') or 'Requirement')} — {type_label} — {_evidence_status(req)}", styles["tiny"])])
                posting = req.get("posting_evidence")
                if posting:
                    details.append([_p(f"Posting evidence: {_safe(posting, 430)}", styles["tiny"])])
                evidence_state = str(req.get("candidate_evidence_status") or "not_documented").casefold()
                candidate = req.get("candidate_evidence")
                if evidence_state == "not_documented":
                    details.append([_p("Profile evidence is not documented; this is not a confirmed mismatch.", styles["tiny"])])
                elif candidate:
                    details.append([_p(f"Profile evidence: {_safe(candidate, 340)}", styles["tiny"])])
        card = _rounded_table(details, [width], background=colors.white, stroke=RULE, radius=6, padding=5, split_rows=1)
        story.append(KeepTogether([card, Spacer(1, 3 * mm)]))


def _append_career_intelligence(story: list[Any], report: dict, styles: dict[str, ParagraphStyle], width: float) -> None:
    story.extend(_section("Career intelligence", styles, "Evidence & implications"))
    market = report.get("market_intelligence") or {}
    synthesis = report.get("intelligence_synthesis") or {}
    story.append(_p(_plain_status(market.get("status"), "Market research"), styles["small"]))
    if synthesis.get("status"):
        story.append(_p(_plain_status(synthesis.get("status"), "Grounded synthesis"), styles["small"]))
    for heading, key in (("Labour-market perspective", "market_intelligence"), ("Personalized career perspective", "personalized_career_intelligence")):
        heading_flowable = _p(heading, styles["h2"])
        paragraphs = synthesis.get(key) or []
        heading_added = False
        if not paragraphs:
            story.append(KeepTogether([heading_flowable, _p("No synthesized insight is available for this section. See the linked evidence below for source context.", styles["small"])]))
            heading_added = True
        hidden_claim_notice_added = False
        for paragraph in paragraphs:
            paragraph_text = str(paragraph.get("text") or "")
            claims_non_documentation_as_failure = (
                key == "personalized_career_intelligence"
                and any(term in paragraph_text.casefold() for term in ("does not currently meet", "does not meet", "lacks the qualification", "lacks qualifications"))
                and any(
                    str(req.get("candidate_evidence_status") or "").casefold() in {"not_documented", "partially_documented"}
                    for section in (report.get("recommendations") or []) + (report.get("potential_matches_requiring_review") or [])
                    for req in section.get("requirements_to_verify") or []
                )
            )
            if claims_non_documentation_as_failure:
                if not hidden_claim_notice_added:
                    notice = "A saved personalized narrative used a broad qualification conclusion. To avoid treating undocumented items as unmet, this summary is omitted here; review each recorded evidence status in the role details and appendix."
                    block = [] if heading_added else [heading_flowable]
                    block.append(_p(notice, styles["warning"]))
                    story.append(KeepTogether(block))
                    heading_added = True
                    hidden_claim_notice_added = True
                continue
            contents: list[list[Any]] = [[_p(paragraph_text, styles["body"])]]
            for citation in paragraph.get("citations") or paragraph.get("sources") or []:
                title = citation.get("document_title") or citation.get("title") or "Source"
                publisher = citation.get("publisher")
                year = citation.get("publication_year") or citation.get("year")
                page = citation.get("page_number") or citation.get("page")
                label = f"{title}{' - ' + str(publisher) if publisher else ''}{' - ' + str(year) if year else ''}{' - PDF page ' + str(page) if page else ''} [source]"
                contents.append([_link(citation.get("source_url") or citation.get("url"), label, styles["tiny"], limit=230)])
            card = _rounded_table(contents, [width], background=PALE_TEAL, stroke=colors.HexColor("#B8DFD7"), radius=6, padding=7)
            block = [] if heading_added else [heading_flowable]
            block.extend([card, Spacer(1, 2 * mm)])
            story.append(KeepTogether(block))
            heading_added = True
    deterministic = report.get("personalized_career_intelligence") or {}
    insights = deterministic.get("insights") or []
    if insights:
        story.append(_p("Practical next steps", styles["h2"]))
        has_personalized_synthesis = bool(synthesis.get("personalized_career_intelligence"))
        for insight in insights:
            card: list[list[Any]] = [[_p(insight.get("topic") or "Career insight", styles["h3"])]]
            if has_personalized_synthesis:
                card.append([_p(_safe(insight.get("suggested_next_step"), 420), styles["small"])])
            else:
                card.append([_p(f"Source finding: {_safe(insight.get('source_finding'), 520)}", styles["small"])])
                card.append([_p(f"Candidate implication: {_safe(insight.get('candidate_implication'), 520)}", styles["small"])])
                card.append([_p(f"Suggested next step: {_safe(insight.get('suggested_next_step'), 420)}", styles["small"])])
            citation = insight.get("citation")
            if isinstance(citation, dict):
                card.append([_link(citation.get("source_url") or citation.get("url"), f"{citation.get('document_title', 'Source')} - {citation.get('publisher', '')} - PDF page {citation.get('page_number', 'not provided')} [source]", styles["tiny"])])
            story.append(KeepTogether([_rounded_table(card, [width], background=colors.white, radius=6, padding=6), Spacer(1, 2 * mm)]))
    evidence = market.get("evidence") or []
    if not evidence:
        story.append(_p("No retrieved market evidence is available; no market conclusion is inferred.", styles["callout"]))
        return

    # Citations already sit beside generated insights. A compact source directory
    # retains traceability without repeating long retrieval excerpts in the PDF.
    source_records: list[dict] = []
    source_records.extend(evidence)
    for key in ("market_intelligence", "personalized_career_intelligence"):
        for paragraph in synthesis.get(key) or []:
            source_records.extend(c for c in (paragraph.get("citations") or paragraph.get("sources") or []) if isinstance(c, dict))
    source_records.extend(
        insight["citation"] for insight in insights
        if isinstance(insight.get("citation"), dict)
    )
    unique: list[dict] = []
    seen: set[tuple] = set()
    for source in source_records:
        url = _url(source.get("source_url") or source.get("url"))
        identity = (url,) if url else (
            source.get("document_title") or source.get("title"),
            source.get("publisher"), source.get("publication_year") or source.get("year"),
            source.get("page_number") or source.get("page"),
        )
        if identity not in seen:
            seen.add(identity)
            unique.append(source)
    story.append(PageBreak())
    story.append(_p("Sources cited", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=7))
    story.append(_p("Citations link to the original source. Page and geographic scope are shown only when the saved report provided them.", styles["small"]))
    rows: list[list[Any]] = [[_p("SOURCE", styles["table_header"]), _p("CITATION DETAILS", styles["table_header"])]]
    for source in unique:
        title = source.get("document_title") or source.get("title") or "Source document"
        publisher = source.get("publisher") or "Publisher not recorded"
        year = source.get("publication_year") or source.get("year") or "Year not recorded"
        page = source.get("page_number") or source.get("page") or source.get("page_start")
        scope = source.get("geographic_scope")
        details = f"{publisher} · {year}"
        if page:
            details += f" · PDF page {page}"
        if scope:
            details += f" · {_safe(scope).replace('_', ' ')}"
        rows.append([
            _link(source.get("source_url") or source.get("url"), f"{title} [open source]", styles["small"], limit=190),
            _p(details, styles["tiny"]),
        ])
    sources_table = LongTable(rows, colWidths=[105 * mm, width - 105 * mm], repeatRows=1, splitByRow=1, hAlign="LEFT")
    sources_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DEEP_TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(sources_table)


def _append_enrichment(story: list[Any], report: dict, styles: dict[str, ParagraphStyle], width: float) -> None:
    story.extend(_section("Research & classification", styles, "Optional enrichment"))
    company = report.get("company_research") or {}
    story.append(_p("Company research", styles["h2"]))
    story.append(_p(_plain_status(company.get("status"), "Company research"), styles["small"]))
    results = company.get("results") or []
    if not results:
        story.append(_p("No company research results are available for this run.", styles["tiny"]))
    for result in results:
        verified = result.get("identity_status") == "verified"
        lines: list[list[Any]] = [[_p(result.get("company") or "Company", styles["h3"])]]
        if verified and result.get("snapshot"):
            lines.append([_p(result["snapshot"], styles["small"], limit=650)])
        elif not verified:
            lines.append([_p("Company identity could not be confirmed, so search snippets are not presented as facts about this company.", styles["tiny"])])
        for uncertainty in (result.get("uncertainties") or [])[:2]:
            lines.append([_p(uncertainty, styles["tiny"], limit=260)])
        for source in (result.get("sources") or [])[:3]:
            lines.append([_link(source.get("url"), f"{source.get('title') or 'Research source'} [source]", styles["tiny"], limit=180)])
        story.append(KeepTogether([_rounded_table(lines, [width], background=colors.white, radius=6, padding=6), Spacer(1, 2 * mm)]))

    esco = report.get("esco_enrichment") or {}
    story.append(_p("ESCO skill and occupation references", styles["h2"]))
    story.append(_p(_plain_status(esco.get("status"), "ESCO lookup"), styles["small"]))
    lookups = esco.get("lookups") or []
    if not lookups:
        story.append(_p("No ESCO concepts were returned for this run.", styles["tiny"]))
    exact_count = alternative_count = ambiguous_count = non_exact_count = 0
    exact_items: list[tuple[str, dict]] = []
    for lookup in lookups:
        result = lookup.get("esco_result") or {}
        relation = result.get("status")
        concepts = result.get("concepts") or []
        for concept in concepts:
            match = concept.get("match_type")
            if match == "exact":
                exact_count += 1
                exact_items.append((lookup.get("term") or "Term", concept))
            elif match == "alternative":
                alternative_count += 1
                exact_items.append((lookup.get("term") or "Term", concept))
            elif match == "ambiguous":
                ambiguous_count += 1
                exact_items.append((lookup.get("term") or "Term", concept))
        if relation == "non_exact":
            non_exact_count += 1
    if lookups:
        story.append(_p(f"{len(lookups)} lookups · {exact_count} exact · {alternative_count} alternative · {ambiguous_count} ambiguous · {non_exact_count} without a verified label match. Search results alone do not establish skill equivalence.", styles["small"]))
    if exact_items:
        data = [[_p("LOOKUP TERM", styles["table_header"]), _p("ESCO CONCEPT", styles["table_header"]), _p("LABEL RELATION", styles["table_header"])]]
        for term, concept in exact_items:
            data.append([_p(term, styles["tiny"]), _link(concept.get("uri"), concept.get("label") or "ESCO concept [source]", styles["tiny"], limit=150), _p(concept.get("match_type") or "not specified", styles["tiny"])])
        table = LongTable(data, colWidths=[50 * mm, 95 * mm, 29 * mm], repeatRows=1, splitByRow=1)
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), DEEP_TEAL), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,PALE_BLUE]), ("VALIGN",(0,0),(-1,-1),"TOP"), ("LINEBELOW",(0,0),(-1,-1),0.3,RULE), ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5), ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story.append(table)


def _header_footer(canvas: Canvas, doc) -> None:
    canvas.saveState()
    width, height = PAGE_SIZE
    # A restrained deep-teal hairline carries the app palette onto white print pages.
    canvas.setFillColor(NAVY)
    canvas.roundRect(doc.leftMargin, height - 12 * mm, 3 * mm, 3 * mm, 1.5 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(doc.leftMargin + 5 * mm, height - 11.5 * mm, "CAREER INTELLIGENCE")
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - doc.rightMargin, height - 11.5 * mm, "CAREER REPORT  ·  PRIVATE")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, height - 15 * mm, width - doc.rightMargin, height - 15 * mm)
    canvas.line(doc.leftMargin, 14 * mm, width - doc.rightMargin, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 9 * mm, "Indicative evidence alignment — not hiring probability or verified qualification")
    canvas.drawRightString(width - doc.rightMargin, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def generate_career_report_pdf(report_data: dict[str, Any]) -> bytes:
    """Render completed report data locally; this function makes no network calls."""
    if not isinstance(report_data, dict):
        raise TypeError("report_data must be a dictionary")
    styles = _styles()
    buffer = io.BytesIO()
    candidate = report_data.get("candidate") or {}
    search = report_data.get("search_scope") or {}
    recommendations = report_data.get("recommendations") or []
    reviews = report_data.get("potential_matches_requiring_review") or []
    evaluated = report_data.get("evaluated_jobs", report_data.get("discovered_jobs", 0))
    width = A4[0] - 36 * mm
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=21 * mm, bottomMargin=19 * mm, title="Career Intelligence Report", author="Career Intelligence Agent", subject="Candidate career intelligence")
    story: list[Any] = []

    # Cover/overview: branded card, compact KPIs, and only reader-relevant context.
    cover = [
        [_p("CAREER INTELLIGENCE  ·  PERSONALIZED REPORT", styles["cover_kicker"])],
        [_p("A clearer view of your next career move", styles["cover_title"])],
        [_p(f"Prepared for {_safe(candidate.get('name') or 'Candidate', 140)}", styles["cover_subtitle"])],
        [_p(_safe(candidate.get("summary") or "Career direction and job evidence summarized from the completed run.", 520), styles["cover_body"])],
    ]
    cover_table = Table(cover, colWidths=[width], cornerRadii=[10, 10, 10, 10])
    cover_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY), ("LINEBEFORE",(0,0),(0,-1),3,MINT), ("LEFTPADDING",(0,0),(-1,-1),13), ("RIGHTPADDING",(0,0),(-1,-1),13), ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.extend([cover_table, Spacer(1, 5 * mm)])

    story.append(_p("Executive overview", styles["h2"]))
    kpis = [
        [ _p(str(evaluated or 0), styles["score"]), _p(str(len(recommendations)), styles["score"]), _p(str(len(reviews)), styles["score"]) ],
        [ _p("JOBS REVIEWED", styles["score_label"]), _p("RECOMMENDED", styles["score_label"]), _p("NEED REVIEW", styles["score_label"]) ],
    ]
    kpi = Table(kpis, colWidths=[width / 3] * 3)
    kpi.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE_TEAL), ("BOX",(0,0),(-1,-1),0.6,RULE), ("INNERGRID",(0,0),(-1,-1),0.5,RULE), ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.extend([kpi, Spacer(1, 4 * mm)])
    story.append(_p("Candidate summary", styles["h2"]))
    target_roles = candidate.get("desired_job_titles") or []
    skills = candidate.get("skills") or []
    experience = candidate.get("years_experience")
    facts = []
    if target_roles:
        facts.append(f"<b>Target roles</b>  {html.escape(', '.join(map(str, target_roles)))}")
    if experience is not None:
        facts.append(f"<b>Experience</b>  {html.escape(str(experience))} years total")
    education = candidate.get("education") or []
    if education:
        education_labels = [" ".join(str(v) for v in (item.get("qualification"), item.get("field")) if v) for item in education]
        facts.append(f"<b>Education</b>  {html.escape('; '.join(education_labels))}")
    if skills:
        facts.append(f"<b>Documented skills</b>  {html.escape(', '.join(map(str, skills[:8])))}")
    story.append(Paragraph("<br/>".join(facts) or "Candidate role and skill details were not provided.", styles["body"]))
    queries = search.get("queries") or []
    providers = search.get("providers") or []
    scope = []
    if queries:
        scope.append("Search themes: " + "; ".join(map(str, queries[:4])))
    if providers:
        scope.append("Job sources: " + ", ".join(map(str, providers)))
    story.append(Spacer(1, 2 * mm))
    story.append(_rounded_table([[_p("SEARCH SCOPE", styles["label"])], [_p("  ·  ".join(scope) or "Search scope was not recorded.", styles["small"])]], [width], background=colors.white, radius=6, padding=6))
    story.append(Spacer(1, 2 * mm))
    disclaimer = "Match scores are indicative evidence-alignment measures. They are not hiring probabilities and do not verify that every employer qualification is met. “Not documented” means the profile did not include evidence; it does not establish that the candidate lacks the qualification. An explicit mismatch is identified separately only when the recorded evidence supports that conclusion."
    story.append(_p(disclaimer, styles["callout"]))

    story.append(PageBreak())
    story.extend(_section("Recommended roles", styles, "Evidence-supported opportunities"))
    if recommendations:
        for index, item in enumerate(recommendations, start=1):
            if index > 1:
                story.append(PageBreak())
            story.append(KeepTogether([
                _p(f"RECOMMENDATION {index} OF {len(recommendations)}", styles["cover_kicker"]),
                _recommendation_card(item, styles, width), Spacer(1, 4 * mm),
            ]))
    else:
        story.append(_p("No roles met the current recommendation criteria. This is a completed result, not a failed run. Potential matches and their review reasons are listed next.", styles["callout"]))
    story.append(_p("A recommendation is an evidence-based lead. Review any listed qualifications before treating it as confirmed fit.", styles["small"]))

    if recommendations:
        story.append(PageBreak())
    story.extend(_section("Potential matches requiring review", styles, "Transparent shortlist"))
    if reviews:
        story.append(_p("These roles did not qualify as recommendations. Scores remain indicative; the review signal explains what needs checking. Full recorded evidence is preserved in the appendix.", styles["small"]))
        story.append(_review_index(reviews, styles, [77 * mm, 20 * mm, width - 97 * mm]))
    else:
        story.append(_p("No roles were flagged for additional review.", styles["callout"]))

    _append_career_intelligence(story, report_data, styles, width)
    _append_enrichment(story, report_data, styles, width)
    if reviews:
        _append_review_appendix(story, reviews, styles, width)
    story.append(Spacer(1, 4 * mm))
    story.append(_p("This document presents the saved report data. Job availability and source information have not been rechecked during PDF export.", styles["tiny"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer, canvasmaker=_StableCanvas)
    return buffer.getvalue()
