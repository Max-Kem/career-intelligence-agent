"""Conservative extraction of explicit, role-applicable job qualifications."""

from __future__ import annotations

import html
import re

from .candidate import CandidateProfile


_SPECIALIST_PATTERNS = (
    ("seniority_experience", "Senior-level professional experience",
     r"\b(?:solid\s+)?senior[- ]level experience\b|\b(?:at least\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+(?:of\s+)?(?:working|experience|in).*?senior level\b|\bexperience.{0,60}\bworking at a senior level\b",
     ("senior-level", "senior level", "senior data analyst", "senior bi analyst", "senior analyst")),
    ("specialized_expertise", "SaaS analytics experience",
     r"\b(?:data analyst|analytics?)\b.{0,80}\b(?:in|at|within) SaaS\b|\bSaaS analytics experience\b",
     ("SaaS analytics", "SaaS data", "SaaS")),
    ("specialized_expertise", "Experiment design and analysis experience",
     r"\b(?:experimentation depth|design(?:ed|ing)?[ ,]+siz(?:e|ed|ing)[ ,]+and analy[sz](?:e|ed|ing) (?:A/B )?tests?|A/B tests?).{0,120}\b(?:experience|depth|design|analy[sz])?\b",
     ("experimentation", "A/B test", "experiment design")),
    ("specialized_expertise", "Advanced data modeling / analytics engineering experience",
     r"\b(?:strong|advanced|expert|substantial|real) data modell?ing\b|\banalytics engineering\b.{0,80}\b(?:experience|strong|advanced|expert|build|model)\b",
     ("data modeling", "data modelling", "analytics engineering", "dbt")),
    ("specialized_expertise", "Robotics / intralogistics experience",
     r"\b(?:robotics?|intralogistics|warehouse automation|automated guided vehicles?|warehouse robotics)\b.{0,180}\b(?:experience|expertise|industry|years?|knowledge|deploy|implementation)\b|\b(?:experience|expertise|industry|years?|knowledge)\b.{0,140}\b(?:robotics?|intralogistics|warehouse automation|automated guided vehicles?|warehouse robotics)\b",
     ("robotics", "intralogistics", "warehouse automation", "automated guided vehicles")),
    ("project_scale", "Large multi-country / multi-entity program delivery experience",
     r"\b(?:large|complex|global|multi[- ]country|cross[- ]border|multi[- ]entity|multiple countries|multiple entities)\b.{0,180}\b(?:programs?|programmes?|projects?|transformations?|implementations?|experience|delivery)\b|\b(?:programs?|programmes?|projects?|experience|delivery)\b.{0,140}\b(?:multi[- ]country|cross[- ]border|multi[- ]entity|multiple countries|multiple entities)\b",
     ("multi-country", "multi country", "multi-entity", "multiple countries", "multiple entities", "global programs", "global transformations")),
    ("specialized_expertise", "Enterprise software / ERP implementation experience",
     r"\b(?:ERP|enterprise software|enterprise systems?)\b.{0,150}\b(?:implementation|experience|expertise|delivery|rollout)\b|\b(?:implementation|experience|expertise|delivery|rollout)\b.{0,120}\b(?:ERP|enterprise software|enterprise systems?)\b",
     ("ERP implementation", "enterprise software implementation", "enterprise systems implementation")),
    ("domain_expertise", "CPG commercial / RGM and trade-promotion expertise",
     r"\b(?:CPG|consumer packaged goods|consumer goods)\b.{0,180}\b(?:RGM|revenue growth management|TPM|trade promotion|commercial analytics|pricing architecture)\b|\b(?:RGM|revenue growth management|TPM|trade promotion)\b.{0,120}\b(?:CPG|consumer packaged goods|consumer goods)\b",
     ("revenue growth management", "trade promotion", "commercial analytics", "CPG")),
    ("domain_expertise", "Indirect-tax / VAT expertise",
     r"\b(?:indirect tax|VAT|value[- ]added tax|sales tax|GST)\b.{0,150}\b(?:experience|expertise|knowledge|bring|hands[- ]on|deep|years?)\b|\b(?:experience|expertise|knowledge|bring|hands[- ]on|deep)\b.{0,100}\b(?:indirect tax|VAT|value[- ]added tax|sales tax|GST)\b",
     ("indirect tax", "VAT", "sales tax", "GST")),
    ("domain_expertise", "Core-banking systems expertise",
     r"\bcore[- ]banking(?: systems?)?\b.{0,150}\b(?:experience|expertise|knowledge|implement|consult|systems?|platforms?)\b|\b(?:experience|expertise|knowledge)\b.{0,100}\bcore[- ]banking\b",
     ("core banking", "Avaloq", "Temenos")),
    ("specialized_expertise", "Purchase-to-pay / order-to-cash process expertise",
     r"\b(?:purchase[- ]to[- ]pay|P2P|order[- ]to[- ]cash|O2C)\b.{0,150}\b(?:knowledge|expertise|experience|end[- ]to[- ]end)\b|\b(?:knowledge|expertise|experience|end[- ]to[- ]end)\b.{0,100}\b(?:purchase[- ]to[- ]pay|P2P|order[- ]to[- ]cash|O2C)\b",
     ("purchase-to-pay", "order-to-cash", "P2P", "O2C")),
)
_OTHER_QUALIFICATION_PATTERNS = (
    ("industry_experience", "Energy-sector experience", r"\b(?:energy(?:[- ]sector| and utilities| & utilities)?|utilities)\s+(?:sector\s+)?experience\b|\bexperience\s+(?:in|within)\s+(?:the\s+)?(?:energy|utilities)(?:\s+(?:sector|industry))?\b", "energy"),
    ("specialized_expertise", "Specialized process expertise", r"\b(?:speciali[sz]ed\s+)?process\s+expertise\b|\bexpertise\s+in\s+(?:business\s+)?process(?:es)?\b", "process"),
    ("leadership_experience", "Senior leadership experience", r"\b(?:senior\s+)?leadership\s+(?:experience|responsibilities|roles)\b|\bexperience\s+leading\s+(?:senior\s+)?teams\b|\bexecutive presence\b", "leadership"),
)
_TECHNICAL_TERMS = ("Python", "SQL", "SAP", "Kubernetes", "Salesforce", "Azure", "AWS", "Power BI", "Alteryx",
                    "Tableau", "Snowflake", "dbt", "Looker")
_LANGUAGES = ("English", "German", "French", "Spanish", "Italian", "Dutch")
_BLOCK_HEADING = re.compile(r"^(?:qualifications?|what you bring|job requirements?|requirements|your profile)\b", re.I)


def clean_job_description(description: str | None) -> str:
    """Turn provider HTML into readable blocks while decoding entities safely."""
    if not description:
        return ""
    text = description
    # Some job providers escape HTML before returning the description, so one
    # entity decode can leave a second literal layer of tags/entities behind.
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", text)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>|</\s*(?:p|li|div|h[1-6]|tr|section)\s*>", "\n", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"(?i)\b(Your Career Path|As a Principal(?: AI Transformation Consultant)?|What You Bring|Qualifications?|Job Requirements?)\b", r"\n\1\n", text)
    text = text.replace("\xa0", " ").replace("â€¢", "•")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip())


def primary_role_text(description: str | None, job_title: str = "") -> str:
    """Keep the main role and qualification sections; discard other-level career paths.

    Career-path copy commonly lists lower-level years beside a senior vacancy. Those
    figures are not attributed to the advertised role. The role-specific block for
    the advertised level may be retained when its heading matches the title.
    """
    lines = clean_job_description(description).splitlines()
    title_level = _role_level(job_title)
    in_career_path = False
    output: list[str] = []
    sentences = [sentence.strip() for line in lines
                 for sentence in re.split(r"(?<=[.!?])\s+", line) if sentence.strip()]
    for line in sentences:
        normalized = re.sub(r"\s+", " ", line).strip()
        if re.search(r"\bcareer path\b|\bcareer progression\b", normalized, re.I):
            in_career_path = True
            continue
        if in_career_path:
            level = _role_level(normalized)
            if level and title_level and level == title_level:
                in_career_path = False
                output.append(normalized)
                continue
            if _BLOCK_HEADING.match(normalized):
                in_career_path = False
            else:
                continue
        # A separate level-specific bullet is not a requirement of a generic ad.
        if re.match(r"^for\s+(?:the\s+)?(?:senior\s+)?consultant\b", normalized, re.I):
            if not (title_level == "senior" and re.match(r"^for\s+senior", normalized, re.I)):
                continue
        output.append(normalized)
    return "\n".join(output)


def specific_experience_scope(line: str) -> tuple[str, str, tuple[str, ...]] | None:
    """Identify experience wording that cannot be satisfied by total years alone."""
    lowered = line.casefold()
    if re.search(r"\b(?:senior[- ]level|working at a senior level|at a senior level)\b", lowered):
        return "senior-level", "seniority_experience", (
            "senior-level", "senior level", "senior data analyst", "senior bi analyst", "senior analyst",
        )
    if re.search(r"\b(?:robotics?|intralogistics|warehouse automation|automated guided vehicles?)\b", lowered):
        return "robotics or intralogistics", "specialized_expertise", (
            "robotics", "intralogistics", "warehouse automation", "automated guided vehicles",
        )
    if re.search(r"\b(?:multi[- ]country|cross[- ]border|multi[- ]entity|multiple countries|multiple entities|global programs?|global programmes?)\b", lowered):
        return "complex multi-country / multi-entity program delivery", "project_scale", (
            "multi-country", "multi country", "multi-entity", "multiple countries", "multiple entities",
            "global programs", "global transformations",
        )
    if re.search(r"\b(?:steering committee|lead(?:ing)? delivery teams?|senior leadership|team management)\b", lowered):
        return "senior stakeholder, steering committee, or delivery-team leadership", "leadership_experience", (
            "steering committee", "delivery team leadership", "senior leadership", "team management",
        )
    if re.search(r"\b(?:complex projects?|complex programs?|complex programmes?)\b", lowered):
        return "complex project or program delivery", "professional_experience", (
            "complex project delivery", "complex project", "complex program", "complex programme",
        )
    if re.search(r"\b(?:as a|as an)\s+(?:senior\s+)?project manager\b|\bproject management experience\b", lowered):
        return "project-management role", "professional_experience", (
            "project manager", "project management",
        )
    for category, label, pattern, term in _OTHER_QUALIFICATION_PATTERNS:
        if category == "industry_experience" and re.search(pattern, line, re.I):
            return "industry-specific", category, (term,)
    return None


def _role_level(text: str) -> str | None:
    lower = text.casefold()
    if re.search(r"\b(principal|principle|director|vice president|\bvp\b|chief|executive)\b", lower):
        return "principal"
    if re.search(r"\bsenior\b", lower):
        return "senior"
    if re.search(r"\b(junior|graduate|entry[- ]level|intern)\b", lower):
        return "junior"
    return None


def requirements_to_verify(
    description: str, candidate: CandidateProfile | None, job_title: str = "",
) -> list[dict]:
    """Return material explicit qualifications not confirmed by the profile.

    Absence is reported as ``not_documented``; it is never treated as proof the
    candidate does not hold a qualification.
    """
    text = primary_role_text(description, job_title)
    lines = [part.strip(" \t\r\n•-") for part in text.splitlines() if part.strip()]
    if not lines:
        return []
    searchable = "\n".join(lines)
    experience = " ".join([*(e.title for e in candidate.professional_experience),
        *(e.organization or "" for e in candidate.professional_experience),
        *(" ".join(e.responsibilities) for e in candidate.professional_experience)]) if candidate else ""
    education = " ".join(f"{x.qualification} {x.field or ''} {x.institution or ''}" for x in candidate.education) if candidate else ""
    skills = " ".join(candidate.skills) if candidate else ""
    output: list[dict] = []
    preferred_section = False

    def add(label: str, category: str, evidence_text: str, evidence_terms: tuple[str, ...] = (), evidence_kind: str = "experience") -> None:
        optional = preferred_section or bool(re.search(r"\b(preferred|ideally|desirable|nice to have|a plus|bonus|advantageous)\b", evidence_text, re.I))
        conditional = bool(re.search(r"\b(?:for|if you choose|depending on).{0,60}\bpath\b", evidence_text, re.I))
        proficiency_level = bool(re.search(r"\b(expert|advanced|strong|hands[- ]on|practical|real|extensive|solid)\b", evidence_text, re.I))
        required = bool(re.search(r"\b(required|must|minimum|essential|mandatory|at least)\b|\b\d+(?:\.\d+)?\s*\+\s*years?\b", evidence_text, re.I)) or proficiency_level
        source = {"experience": experience, "education": education, "skills": skills}[evidence_kind]
        if category == "language":
            profile_text = " ".join([
                candidate.summary or "",
                " ".join(candidate.skills),
                " ".join(entry.title + " " + " ".join(entry.responsibilities) for entry in candidate.professional_experience),
                " ".join(f"{entry.qualification} {entry.field or ''}" for entry in candidate.education),
            ]) if candidate else ""
            language = label.split(":", 1)[-1].strip()
            proficiency = re.search(
                rf"(?:\b(?:fluent|fluency|native|professional proficiency|business[- ]level|working proficiency|C1|C2|B2)\b.{{0,35}}\b{re.escape(language)}\b|\b{re.escape(language)}\b.{{0,35}}\b(?:fluent|fluency|native|professional proficiency|business[- ]level|working proficiency|C1|C2|B2)\b)",
                profile_text, re.I,
            )
            if proficiency:
                status = "partially_documented"
                candidate_evidence = f"Profile contains an explicit proficiency reference for {language}; compare it with the level stated in the posting."
            else:
                status = "not_documented"
                candidate_evidence = "Language proficiency is not documented in the supplied candidate profile; this does not mean the candidate lacks it."
            output.append({"requirement": label, "category": category,
                "requirement_type": "conditional" if conditional else "preferred" if optional else "required" if required else "not_specified",
                "candidate_evidence_status": status, "candidate_evidence": candidate_evidence,
                "posting_evidence": evidence_text})
            return
        found = next((term for term in evidence_terms if re.search(re.escape(term), source, re.I)), None)
        if found:
            status = "partially_documented" if evidence_kind == "experience" else "documented"
            if category == "technical_qualification":
                evidence_start = evidence_text.casefold().find(found.casefold())
                start = max(evidence_text.rfind(",", 0, evidence_start), evidence_text.rfind(";", 0, evidence_start)) + 1
                end_candidates = [value for value in (evidence_text.find(",", evidence_start), evidence_text.find(";", evidence_start)) if value >= 0]
                end = min(end_candidates) if end_candidates else len(evidence_text)
                tool_clause = evidence_text[start:end]
                if re.search(r"\b(expert(?:[- ]level)?|advanced|strong|hands[- ]on|practical|real|extensive|solid)\b", tool_clause, re.I):
                    status = "partially_documented"
            if evidence_kind == "experience" and category == "domain_expertise":
                status = "documented"
            candidate_evidence = (f"Profile mentions {found!r}; the job-specific level or context is not independently verified."
                                  if status == "partially_documented" else f"Profile documents {found!r} in professional experience.")
            if status == "documented":
                return
        else:
            status = "not_documented"
            candidate_evidence = "Not documented in the supplied candidate profile; this does not mean the candidate lacks the qualification."
        if any(item["requirement"].casefold() == label.casefold() for item in output):
            return
        output.append({"requirement": label, "category": category,
            "requirement_type": "conditional" if conditional else "preferred" if optional else "required" if required else "not_specified",
            "candidate_evidence_status": status, "candidate_evidence": candidate_evidence,
            "posting_evidence": evidence_text})

    # Work line-by-line so credentials in a disclaimer or unrelated level are not
    # promoted into qualifications. Only named credentials or explicit credential
    # requirements are considered certification evidence.
    for line in lines:
        section_heading = re.sub(r"^[^\w]+", "", line, flags=re.UNICODE).strip()
        preferred_heading = re.match(r"^(?:nice to have|bonus points|preferred qualifications|additional advantages)\s*:?\s*", section_heading, re.I)
        if preferred_heading:
            preferred_section = True
            line = section_heading[preferred_heading.end():].strip()
            if not line:
                continue
        required_heading = re.match(r"^(?:what you need|what you bring|what we.re looking for|qualifications|requirements|your profile)\s*:?\s*", section_heading, re.I)
        if required_heading:
            preferred_section = False
            line = section_heading[required_heading.end():].strip()
            if not line:
                continue
        if re.match(r"^(?:what you.ll do|what success looks like|responsibilities|your mission|where you.ll work|benefits|what we offer|why |about the role)\b", section_heading, re.I):
            preferred_section = False
            continue
        years = re.search(
            r"\b(?:(\d+(?:\.\d+)?)\s*(?:-|\u2013|\u2014|\bto\b)\s*(\d+(?:\.\d+)?)|"
            r"exactly\s+(\d+(?:\.\d+)?)|(?:at least|minimum(?: of)?|typically)?\s*(\d+(?:\.\d+)?))"
            r"\s*\+?\s*years?\b(?=[^.!?\n]{0,100}\b(?:experience|in|as|working|delivering|managing|leading)\b)",
            line, re.I,
        )
        if years:
            is_range = years.group(1) is not None
            is_exact = years.group(3) is not None
            required_years = float(years.group(1) or years.group(3) or years.group(4))
            upper_years = float(years.group(2)) if is_range else None
            specific_scope = specific_experience_scope(line)
            requirement_label = (
                f"{required_years:g}–{upper_years:g} years" if is_range
                else f"Exactly {required_years:g} years" if is_exact
                else f"At least {required_years:g} years"
            )
            if specific_scope:
                scope, category, evidence_terms = specific_scope
                label = f"{requirement_label} of {scope} experience"
                add(label, category, line, evidence_terms)
                years_item = next((item for item in reversed(output) if item["requirement"] == label), None)
            elif candidate and candidate.years_experience is not None and candidate.years_experience >= required_years:
                years_item = None
            else:
                label = f"{requirement_label} of experience"
                add(label, "professional_experience", line)
                years_item = next((item for item in reversed(output) if item["requirement"] == label), None)
            if years_item and candidate and candidate.years_experience is not None:
                reported_years = candidate.years_experience
                unmet = (reported_years != required_years if is_exact else reported_years < required_years)
                if unmet:
                    years_item["candidate_evidence_status"] = "does_not_meet_requirement"
                    years_item["candidate_evidence"] = (
                        f"Profile reports {reported_years:g} years; posting states {requirement_label.lower()}."
                    )

        for category, label, pattern, term in _OTHER_QUALIFICATION_PATTERNS:
            if re.search(pattern, line, re.I):
                add(label, category, line, (term,))
        for category, label, pattern, terms in _SPECIALIST_PATTERNS:
            explicit_match = re.search(pattern, line, re.I)
            contradicted_in_sentence = bool(re.search(
                r"\b(?:not|no|never|without)\b.{0,50}\b(?:global|multi[- ]country|multi[- ]entity|robotics?|intralogistics|warehouse automation)\b",
                line, re.I,
            ))
            if explicit_match and not contradicted_in_sentence:
                add(label, category, line, terms)

        credential_names = re.findall(r"\b(?:PMP|CFA|CPA|CISA|CISSP|ITIL|AWS Certified\w*|Azure certification|Avaloq Certified Customizer|ACCP)\b", line, re.I)
        explicit_credential = re.search(r"\b(?:certification|certified|certificate|license|licence)\b", line, re.I)
        credential_requirement = explicit_credential and re.search(r"\b(?:required|preferred|must|plus|advantage|ideally|desirable)\b", line, re.I)
        if credential_requirement:
            name = credential_names[0] if credential_names else "relevant professional certification"
            label = f"{name} certification" if credential_names else name
            add(label, "certification", line, (name,) if credential_names else (), "education")

        if re.search(r"\b(bachelor|master|doctorate|ph\.?d\.?|degree in)\b", line, re.I):
            degree = re.search(r"\b(bachelor|master|doctorate|ph\.?d\.?)\b", line, re.I)
            add(line[:180], "education", line, (degree.group(0),) if degree else (), "education")

        if re.search(r"\b(fluent|fluency|proficient|proficiency|native|C1|C2|B2|business[- ]level|business fluency|working proficiency)\b", line, re.I):
            for language in _LANGUAGES:
                if re.search(rf"\b{language}\b", line, re.I):
                    add(f"Language proficiency: {language}", "language", line, (language,), "skills")

        if re.search(r"\b(required|must|minimum|essential|mandatory|preferred|ideally|experience|expertise|knowledge|expert|advanced|strong|hands[- ]on|practical|real|solid|skills?)\b", line, re.I):
            for term in _TECHNICAL_TERMS:
                if re.search(rf"\b{re.escape(term)}\b", line, re.I):
                    terms = (term,)
                    if term.casefold() == "looker" and re.search(r"\b(?:close|similar|reasonable) equivalent\b", line, re.I):
                        terms = ("Looker", "Power BI", "Tableau")
                    add(f"{term} qualification", "technical_qualification", line, terms, "skills")
    return output


def critical_requirements_to_verify(description: str, candidate: CandidateProfile, job_title: str = "") -> list[dict]:
    """Unverified, material experience/domain requirements that warrant human review."""
    material_categories = {"professional_experience", "industry_experience", "domain_expertise",
                           "specialized_expertise", "leadership_experience", "project_scale",
                           "seniority_experience", "technical_qualification"}
    return [item for item in requirements_to_verify(description, candidate, job_title)
            if item["category"] in material_categories
            and item["requirement_type"] != "preferred"
            and item["candidate_evidence_status"] != "documented"]
