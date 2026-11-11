"""Conservative extraction of explicit job qualifications for human review."""

from __future__ import annotations

import html
import re

from .candidate import CandidateProfile


_QUALIFICATION_PATTERNS = (
    ("industry_experience", "Energy-sector experience", r"\b(?:energy(?:[- ]sector| and utilities| & utilities)?|utilities)\s+(?:sector\s+)?experience\b|\bexperience\s+(?:in|within)\s+(?:the\s+)?(?:energy|utilities)(?:\s+(?:sector|industry))?\b", "energy"),
    ("domain_expertise", "Core-banking domain expertise", r"\bcore[- ]banking(?: domain)?\s+(?:expertise|experience|knowledge|background)\b|\b(?:expertise|experience|knowledge|background)\s+(?:in|with)\s+core[- ]banking\b", "core banking"),
    ("specialized_expertise", "Specialized process expertise", r"\b(?:speciali[sz]ed\s+)?process\s+expertise\b|\bexpertise\s+in\s+(?:business\s+)?process(?:es)?\b", "process"),
    ("leadership_experience", "Senior leadership experience", r"\b(?:senior\s+)?leadership\s+(?:experience|responsibilities|roles)\b|\bexperience\s+leading\s+(?:senior\s+)?teams\b", "leadership"),
)
_TECHNICAL_TERMS = ("Python", "SQL", "SAP", "Kubernetes", "Salesforce", "Azure", "AWS", "Power BI")


def requirements_to_verify(description: str, candidate: CandidateProfile | None) -> list[dict]:
    """Return explicit material qualifications that supplied profile data cannot confirm.

    This is intentionally a small lexical helper: absence means ``not_documented``,
    not that the candidate lacks the qualification.
    """
    text = html.unescape(re.sub(r"<[^>]*>", " ", description or ""))
    sentences = [part.strip(" \t\r\n•-") for part in re.split(r"(?<=[.!?])\s+|[\n•]+", text) if part.strip()]
    if not sentences:
        return []
    experience = " ".join(
        [*(entry.title for entry in candidate.professional_experience),
         *(entry.organization or "" for entry in candidate.professional_experience),
         *(" ".join(entry.responsibilities) for entry in candidate.professional_experience)]
    ) if candidate else ""
    education = " ".join(
        f"{item.qualification} {item.field or ''} {item.institution or ''}"
        for item in candidate.education
    ) if candidate else ""
    skills = " ".join(candidate.skills) if candidate else ""
    output: list[dict] = []

    def add(label: str, category: str, sentence: str, evidence_term: str = "", evidence_kind: str = "experience") -> None:
        optional = bool(re.search(r"\b(preferred|ideally|desirable|nice to have|a plus|bonus|advantageous)\b", sentence, re.I))
        required = bool(re.search(r"\b(required|must|minimum|essential|mandatory)\b", sentence, re.I))
        evidence = ""
        status = "not_documented"
        source = {"experience": experience, "education": education, "skills": skills}[evidence_kind]
        if evidence_term and re.search(re.escape(evidence_term), source, re.I):
            status = "partially_documented" if evidence_kind == "experience" else "documented"
            evidence = f"Profile mentions {evidence_term!r}; the job-specific level or context is not independently verified." if status == "partially_documented" else f"Profile documents {evidence_term!r}."
            if status == "documented":
                return
        else:
            evidence = "Not documented in the supplied candidate profile; this does not mean the candidate lacks the qualification."
        if not any(item["requirement"].casefold() == label.casefold() for item in output):
            output.append({
                "requirement": label,
                "category": category,
                "requirement_type": "preferred" if optional else "required" if required else "not_specified",
                "candidate_evidence_status": status,
                "candidate_evidence": evidence,
                "posting_evidence": sentence,
            })

    for sentence in sentences:
        years_match = re.search(
            r"\b(\d+)\s*\+\s*years?\b[^.!?]{0,100}\bexperience\b|\b(?:at least|minimum of)\s+(\d+)\s+years?\b[^.!?]{0,80}\bexperience\b|\b(\d+)\s+years?\b[^.!?]{0,80}\bexperience\b",
            sentence, re.I,
        )
        if years_match:
            years = int(next(group for group in years_match.groups() if group is not None))
            if candidate and candidate.years_experience is not None and candidate.years_experience >= years:
                pass
            else:
                add(f"At least {years} years of experience", "professional_experience", sentence)
                if candidate and candidate.years_experience is not None:
                    output[-1]["candidate_evidence_status"] = "does_not_meet_requirement"
                    output[-1]["candidate_evidence"] = f"Profile reports {candidate.years_experience:g} years; posting states at least {years}."

        for category, label, pattern, evidence_term in _QUALIFICATION_PATTERNS:
            if re.search(pattern, sentence, re.I):
                add(label, category, sentence, evidence_term)

        if re.search(r"\b(certification|certified|certificate|license|licence)\b", sentence, re.I):
            named_credential = re.search(r"\b(?:PMP|CFA|CPA|CISA|CISSP|ITIL|AWS|Azure)\b", sentence, re.I)
            add(sentence[:180], "certification", sentence, named_credential.group(0) if named_credential else "", "education")
        if re.search(r"\b(bachelor|master|doctorate|ph\.?d\.?|degree in)\b", sentence, re.I):
            level = re.search(r"\b(bachelor|master|doctorate|ph\.?d\.?)\b", sentence, re.I)
            field = re.search(r"\bdegree\s+in\s+([a-z][a-z -]{1,40})", sentence, re.I)
            evidence_term = " ".join(part for part in (level.group(0) if level else "", field.group(1).strip() if field else "") if part)
            add(sentence[:180], "education", sentence, evidence_term, "education")
        if re.search(r"\b(fluent|fluency|proficient|proficiency|native|C1|C2|B2)\b", sentence, re.I) and re.search(r"\b(English|German|French|Spanish|language)\b", sentence, re.I):
            language = next((name for name in ("English", "German", "French", "Spanish") if name.casefold() in sentence.casefold()), "language")
            add(f"Language proficiency: {language}", "language", sentence)
        if re.search(r"\b(required|must|minimum|essential|mandatory|preferred|ideally|experience|expertise)\b", sentence, re.I):
            for term in _TECHNICAL_TERMS:
                if re.search(rf"\b{re.escape(term)}\b", sentence, re.I):
                    add(f"{term} qualification", "technical_qualification", sentence, term, "skills")
    return output
