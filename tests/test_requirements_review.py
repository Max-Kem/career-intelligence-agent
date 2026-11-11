from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from career_agent.candidate import CandidateProfile, ExperienceEntry
from career_agent.matching import JobMatch, MatchingResult, SupportStatus
from career_agent.models import Job
from career_agent.requirements_review import requirements_to_verify
from career_agent.workflow import create_report


ROOT = Path(__file__).resolve().parents[1]


def profile_a() -> CandidateProfile:
    return CandidateProfile.model_validate_json((ROOT / "fixtures" / "candidate_profile_a.json").read_text(encoding="utf-8"))


def test_explicit_experience_industry_and_leadership_requirements_are_not_documented_not_assumed_missing():
    candidate = profile_a()
    assert candidate.years_experience is None and not candidate.education
    result = requirements_to_verify(
        "At least 7+ years of relevant experience is required. Experience in the energy sector is required. "
        "Senior leadership responsibilities are expected.", candidate,
    )
    by_name = {item["requirement"]: item for item in result}
    assert by_name["At least 7 years of experience"]["candidate_evidence_status"] == "not_documented"
    assert by_name["Energy-sector experience"]["candidate_evidence_status"] == "not_documented"
    assert by_name["Senior leadership experience"]["candidate_evidence_status"] == "not_documented"
    assert all("does not mean the candidate lacks" in item["candidate_evidence"] for item in result)


def test_core_banking_optional_certification_and_education_are_labeled_without_claiming_mandatory():
    result = requirements_to_verify(
        "Core-banking domain expertise is required. A relevant certification is a plus. "
        "A master's degree in finance is preferred. Fluent in German is required. Python experience is required.", profile_a(),
    )
    by_category = {item["category"]: item for item in result}
    assert by_category["domain_expertise"]["requirement_type"] == "required"
    assert by_category["certification"]["requirement_type"] == "preferred"
    assert by_category["education"]["requirement_type"] == "preferred"
    assert by_category["certification"]["candidate_evidence_status"] == "not_documented"
    assert by_category["language"]["candidate_evidence_status"] == "not_documented"
    assert by_category["technical_qualification"]["candidate_evidence_status"] == "not_documented"


def test_principal_role_requirements_capture_years_process_expertise_and_leadership():
    result = requirements_to_verify(
        "7+ years of relevant experience. Specialized process expertise is expected. "
        "The role includes senior leadership responsibilities.", profile_a(),
    )
    labels = {item["requirement"] for item in result}
    assert "At least 7 years of experience" in labels
    assert "Specialized process expertise" in labels
    assert "Senior leadership experience" in labels


def test_contradictory_years_and_partially_documented_domain_are_distinguished():
    candidate = profile_a().model_copy(update={
        "years_experience": 5,
        "professional_experience": [ExperienceEntry(title="Energy Consultant", responsibilities=["Worked in the energy sector."])],
    })
    result = requirements_to_verify("At least 7+ years of relevant experience is required. Energy-sector experience required.", candidate)
    by_category = {item["category"]: item for item in result}
    assert by_category["professional_experience"]["candidate_evidence_status"] == "does_not_meet_requirement"
    assert by_category["industry_experience"]["candidate_evidence_status"] == "partially_documented"


def test_report_adds_requirement_fields_and_keeps_provisional_job_out_of_recommendations():
    candidate = profile_a()
    eraneos = Job(
        job_id="eraneos", title="Senior Consultant Energy & Utility", company="Eraneos", source="fixture",
        description="Experience in the energy sector is required. At least 7+ years of relevant experience is required.",
        source_url="https://example.test/eraneos",
    )
    procuvia = Job(
        job_id="procuvia", title="Consultant KI & Einkauf", company="ProcuVia", source="fixture",
        description="Implement AI use cases and communicate with stakeholders. Work with procurement teams.",
        source_url="https://example.test/procuvia",
    )
    recommendation = JobMatch(eraneos, 80, [], 85, 4, SupportStatus.SUFFICIENT, "passed")
    provisional = JobMatch(procuvia, 83, [], 55, 2, SupportStatus.PROVISIONAL, "passed")
    matching = MatchingResult([recommendation, provisional], [recommendation], [])
    discovery = SimpleNamespace(jobs=[eraneos, procuvia])
    markdown, data = create_report({
        "candidate": candidate,
        "discovery": discovery,
        "matching": matching,
        "recommendations": [recommendation],
        "provider_diagnostics": {"provider_counts": {}, "filter_exclusions": {}},
        "offline": True,
    })
    requirement = data["recommendations"][0]["requirements_to_verify"][0]
    assert requirement["requirement"] == "Energy-sector experience"
    assert requirement["candidate_evidence_status"] == "not_documented"
    assert data["score_interpretation"].startswith("Indicative alignment")
    assert [item["job"]["job_id"] for item in data["potential_matches_requiring_review"]] == ["procuvia"]
    assert data["potential_matches_requiring_review"][0]["support_status"] == "provisional"
    assert any("coverage is 55%" in reason for reason in data["potential_matches_requiring_review"][0]["reasons"])
    assert "Potential matches requiring review" in markdown
    assert "did not qualify as recommendations" in markdown
    assert "Requirements to verify" in markdown
    # JSON representation keeps the same data without report-only reinterpretation.
    restored = json.loads(json.dumps(data))
    assert restored["recommendations"][0]["requirements_to_verify"] == data["recommendations"][0]["requirements_to_verify"]
