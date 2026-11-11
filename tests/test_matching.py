from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from career_agent.candidate import CandidateProfile, HardRequirements
from career_agent.matching import (
    SKILL_SYNONYMS,
    ScoringWeights,
    SupportStatus,
    canonical_skill,
    extract_job_skills,
    match_candidate_to_jobs,
    score_job,
)
from career_agent.models import Eligibility, Job, WorkModel


ROOT = Path(__file__).resolve().parents[1]


def load_profile(name: str) -> CandidateProfile:
    return CandidateProfile.model_validate_json((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


def load_jobs() -> list[Job]:
    payload = json.loads((ROOT / "fixtures" / "matching_jobs.json").read_text(encoding="utf-8"))
    return [Job.model_validate(item) for item in payload]


def make_job(**kwargs) -> Job:
    values = {"job_id": "job-1", "title": "Data Analyst", "company": "Example", "source": "test"}
    values.update(kwargs)
    return Job(**values)


def test_profile_validation_and_unknown_years_remain_unknown():
    profile = load_profile("candidate_profile_a.json")
    assert profile.candidate_id == "synthetic-profile-a"
    assert profile.years_experience is None
    assert profile.education == []
    with pytest.raises(ValidationError):
        CandidateProfile(candidate_id="x", name="x", years_experience=-1)


def test_skill_matching_is_case_insensitive_synonym_aware_and_word_bounded():
    assert canonical_skill("Stakeholder Engagement") == "stakeholder management"
    assert "stakeholder management" in extract_job_skills("Strong STAKEHOLDER engagement skills required.")
    assert "sql" not in extract_job_skills("NoSQL database experience required.")
    assert "process optimization" in extract_job_skills("Experience with process optimisation.")
    assert SKILL_SYNONYMS["data visualization"]


def test_role_relevance_matches_titles_and_rejects_unrelated_titles():
    profile = load_profile("candidate_profile_b.json")
    analyst = score_job(profile, make_job(title="Junior Data Analyst", description="SQL and Python."))
    sales = score_job(profile, make_job(title="Regional Account Executive", description="Maintain customer accounts."))
    role_score = lambda result: next(d.score for d in result.dimensions if d.name == "role_interests")
    assert role_score(analyst) > 0
    assert role_score(sales) == 0


def test_seniority_mismatch_is_evidenced_and_scores_zero():
    profile = CandidateProfile(candidate_id="s", name="S", seniority_preferences=["junior"], desired_job_titles=["Analyst"])
    result = score_job(profile, make_job(title="Senior Analyst", description="SQL required."))
    dimension = next(d for d in result.dimensions if d.name == "seniority_qualifications")
    assert dimension.score == 0
    assert "senior" in dimension.evidence[0]


def test_junior_profile_gets_no_seniority_credit_for_senior_m_and_a_analyst():
    profile = CandidateProfile(
        candidate_id="junior", name="Junior", skills=["Excel", "SQL"],
        desired_job_titles=["Junior Data Analyst", "Business Intelligence Analyst"],
        seniority_preferences=["junior", "entry-level"],
    )
    job = make_job(
        title="Senior M&A Business Analyst",
        description="Use Excel and SQL. The team includes junior, manager and senior colleagues.",
        seniority=None,
    )
    result = score_job(profile, job)
    seniority = next(d for d in result.dimensions if d.name == "seniority_qualifications")
    role = next(d for d in result.dimensions if d.name == "role_interests")
    assert seniority.score == 0
    assert role.score == 0


def test_title_seniority_overrides_incidental_description_words():
    profile = CandidateProfile(candidate_id="j", name="J", seniority_preferences=["junior"])
    explicitly_senior = make_job(
        title="Senior Analyst", description="Junior, manager and senior stakeholders are supported by this team.",
    )
    incidental_only = make_job(
        title="Business Analyst", description="We work with junior, manager and senior stakeholders.",
    )
    seniority = lambda job: next(
        dimension.score for dimension in score_job(profile, job).dimensions
        if dimension.name == "seniority_qualifications"
    )
    assert seniority(explicitly_senior) == 0
    assert seniority(incidental_only) is None


def test_generic_role_terms_alone_do_not_create_role_relevance():
    profile = CandidateProfile(
        candidate_id="r", name="R", desired_job_titles=["Data Analyst"],
        career_interests=["AI Consultant", "Project Manager"],
    )
    job = make_job(title="Senior M&A Business Analyst")
    role = next(d for d in score_job(profile, job).dimensions if d.name == "role_interests")
    assert role.score == 0


def test_missing_information_is_not_scored_as_zero():
    profile = CandidateProfile(candidate_id="m", name="M", skills=[], desired_job_titles=["Research Scientist"])
    result = score_job(profile, make_job(title="Regional Account Executive", description=None))
    skills = next(d for d in result.dimensions if d.name == "skills")
    experience = next(d for d in result.dimensions if d.name == "experience")
    assert skills.score is None and experience.score is None
    assert result.score is None
    assert result.support_status == SupportStatus.INSUFFICIENT


def test_weight_renormalization_and_coverage_use_only_evaluable_dimensions():
    profile = CandidateProfile(candidate_id="w", name="W", skills=["SQL"], desired_job_titles=["AI Consultant"])
    job = make_job(title="Data Analyst", description="SQL required.")
    result = score_job(profile, job, weights=ScoringWeights())
    assert next(d.score for d in result.dimensions if d.name == "skills") == 100
    assert next(d.score for d in result.dimensions if d.name == "role_interests") == 0
    assert result.evidence_coverage == 55
    assert result.score == 64  # (35*100 + 20*0) / (35+20), rounded to an integer.
    assert result.support_status == SupportStatus.PROVISIONAL


def test_confirmed_hard_filter_violation_excludes_but_unknown_is_retained_unverified():
    profile = CandidateProfile(
        candidate_id="h", name="H", desired_job_titles=["Analyst"],
        required_preferences=HardRequirements(countries=["DE"], work_models=[WorkModel.REMOTE]),
    )
    disallowed = make_job(job_id="no", eligible_countries=["FR"], work_model=WorkModel.REMOTE)
    unknown = make_job(job_id="unknown", eligible_countries=[], work_model=WorkModel.UNKNOWN)
    result = match_candidate_to_jobs(profile, [disallowed, unknown])
    assert [item.job.job_id for item in result.excluded_jobs] == ["no"]
    assert len(result.ranked_matches) == 1
    assert result.ranked_matches[0].hard_filter_status == "unknown"
    assert "country/countries (DE) is unknown" in result.ranked_matches[0].unresolved_requirements[0]
    assert result.top_matches == []


def test_profile_a_hard_geography_excludes_zurich_and_london_but_accepts_verified_remote_de():
    profile = load_profile("candidate_profile_a.json")
    assert profile.required_preferences.countries == ["DE"]
    template = load_jobs()[0]
    zurich = template.model_copy(update={
        "job_id": "zurich-onsite", "location": "Zurich", "country": None,
        "work_model": WorkModel.ONSITE, "eligible_countries": [],
        "source_url": "https://employer.ch/jobs/zurich-role",
    })
    london = template.model_copy(update={
        "job_id": "london-hybrid", "location": "London", "country": None,
        "work_model": WorkModel.HYBRID, "eligible_countries": [],
        "source_url": "https://employer.co.uk/jobs/london-role",
    })
    verified_remote = template.model_copy(update={
        "job_id": "remote-from-zurich", "location": "Zurich", "country": None,
        "work_model": WorkModel.REMOTE, "eligible_countries": ["DE"],
    })
    unknown_from_german_domain = template.model_copy(update={
        "job_id": "unknown-location", "location": "Unknown", "country": None,
        "work_model": WorkModel.REMOTE, "eligible_countries": [],
        "country_eligibility": Eligibility.UNKNOWN,
        "source_url": "https://employer.de/jobs/unknown-location",
    })
    remote_from_germany_unknown = template.model_copy(update={
        "job_id": "remote-country-unknown", "location": "Berlin", "country": None,
        "work_model": WorkModel.REMOTE, "eligible_countries": [],
        "country_eligibility": Eligibility.UNKNOWN,
    })
    result = match_candidate_to_jobs(
        profile, [zurich, london, verified_remote, unknown_from_german_domain, remote_from_germany_unknown],
    )
    assert {item.job.job_id for item in result.excluded_jobs} == {"zurich-onsite", "london-hybrid"}
    assert [item.job.job_id for item in result.top_matches] == ["remote-from-zurich"]
    unknown = next(item for item in result.ranked_matches if item.job.job_id == "unknown-location")
    assert unknown.hard_filter_status == "unknown"
    assert unknown not in result.top_matches
    remote_unknown = next(item for item in result.ranked_matches if item.job.job_id == "remote-country-unknown")
    assert remote_unknown.hard_filter_status == "unknown"
    assert remote_unknown not in result.top_matches


@pytest.mark.parametrize("location", ["Munich", "München", "Munich, Germany", "Berlin"])
def test_explicit_german_on_site_location_satisfies_profile_a_hard_country(location):
    profile = load_profile("candidate_profile_a.json")
    job = make_job(
        job_id=f"local-{location}", title="AI Transformation Consultant", location=location,
        work_model=WorkModel.ONSITE, description="Lead AI project delivery and stakeholder management.",
    )
    result = match_candidate_to_jobs(profile, [job])
    assert result.excluded_jobs == []
    assert len(result.ranked_matches) == 1
    match = result.ranked_matches[0]
    assert match.geographic_status == Eligibility.ELIGIBLE
    assert any("Recognized job location" in item or "Normalized job country" in item for item in match.geographic_evidence)


def test_remote_foreign_location_requires_verified_de_eligibility_and_preserves_geo_reason():
    profile = load_profile("candidate_profile_a.json")
    template = load_jobs()[0]
    foreign_remote_unverified = template.model_copy(update={
        "job_id": "zurich-remote-unknown", "location": "Zurich", "work_model": WorkModel.REMOTE,
        "eligible_countries": [], "country_eligibility": Eligibility.UNKNOWN,
    })
    foreign_remote_verified = template.model_copy(update={
        "job_id": "zurich-remote-de", "location": "Zurich", "work_model": WorkModel.REMOTE,
        "eligible_countries": ["DE"], "country_eligibility": Eligibility.UNKNOWN,
    })
    result = match_candidate_to_jobs(profile, [foreign_remote_unverified, foreign_remote_verified])
    by_id = {match.job.job_id: match for match in result.ranked_matches}
    assert by_id["zurich-remote-unknown"].geographic_status == Eligibility.UNKNOWN
    assert "zurich-remote-unknown" not in {match.job.job_id for match in result.top_matches}
    assert by_id["zurich-remote-de"].geographic_status == Eligibility.ELIGIBLE
    assert "zurich-remote-de" in {match.job.job_id for match in result.top_matches}
    assert any("Provider confirms" in item for item in by_id["zurich-remote-de"].geographic_evidence)


def test_conflicting_locations_remain_unverified_but_german_location_works_with_unknown_model():
    profile = load_profile("candidate_profile_a.json")
    conflict = make_job(
        job_id="conflicting-location", location="London", country="DE",
        work_model=WorkModel.ONSITE,
    )
    unknown_model = make_job(
        job_id="unknown-model-in-berlin", location="Berlin", country=None,
        work_model=WorkModel.UNKNOWN,
    )
    result = match_candidate_to_jobs(profile, [conflict, unknown_model])
    assert result.excluded_jobs == []
    by_id = {match.job.job_id: match for match in result.ranked_matches}
    assert by_id["conflicting-location"].geographic_status == Eligibility.UNKNOWN
    assert any("conflicts with location" in reason for reason in by_id["conflicting-location"].geographic_evidence)
    assert by_id["unknown-model-in-berlin"].geographic_status == Eligibility.ELIGIBLE
    assert "work model remains unknown" in " ".join(by_id["unknown-model-in-berlin"].geographic_evidence)


def test_unknown_work_model_does_not_qualify_foreign_location_or_satisfy_remote_requirement():
    profile = load_profile("candidate_profile_a.json")
    foreign_unknown = make_job(
        job_id="zurich-unknown-model", location="Zurich", work_model=WorkModel.UNKNOWN,
    )
    required_remote_profile = CandidateProfile(
        candidate_id="remote-de", name="Remote DE",
        required_preferences=HardRequirements(countries=["DE"], work_models=[WorkModel.REMOTE]),
    )
    german_unknown = make_job(
        job_id="munich-unknown-model", location="Munich", work_model=WorkModel.UNKNOWN,
    )
    foreign_result = match_candidate_to_jobs(profile, [foreign_unknown])
    remote_result = match_candidate_to_jobs(required_remote_profile, [german_unknown])
    assert foreign_result.ranked_matches[0].geographic_status == Eligibility.UNKNOWN
    assert remote_result.ranked_matches[0].geographic_status == Eligibility.ELIGIBLE
    assert remote_result.ranked_matches[0].hard_filter_status == "unknown"
    assert "Required work model cannot be verified." in remote_result.ranked_matches[0].unresolved_requirements


def test_explicit_applicant_restriction_still_blocks_german_physical_location():
    profile = load_profile("candidate_profile_a.json")
    restricted = make_job(
        job_id="munich-restricted", location="Munich", work_model=WorkModel.UNKNOWN,
        eligible_countries=["FR"],
    )
    result = match_candidate_to_jobs(profile, [restricted])
    assert result.ranked_matches == []
    assert result.excluded_jobs[0].geographic_status == Eligibility.INELIGIBLE


def test_required_seniority_exclusion_keeps_score_and_reason():
    profile = CandidateProfile(
        candidate_id="hard-junior", name="Hard Junior", skills=["SQL"], desired_job_titles=["Analyst"],
        required_preferences=HardRequirements(seniority=["junior"]),
    )
    senior_job = make_job(title="Senior Analyst", description="SQL required.")
    result = match_candidate_to_jobs(profile, [senior_job])
    assert result.top_matches == []
    assert len(result.excluded_jobs) == 1
    exclusion = result.excluded_jobs[0]
    assert exclusion.score is not None
    assert "Job seniority senior" in " ".join(exclusion.violations)
    assert exclusion.explanation


def test_soft_junior_preference_blocks_senior_job_without_strong_relevant_experience_but_keeps_score():
    profile = CandidateProfile(
        candidate_id="soft-junior", name="Soft Junior", skills=["SQL", "Excel"],
        desired_job_titles=["Junior Data Analyst", "Business Intelligence Analyst"],
        seniority_preferences=["junior", "entry-level"],
    )
    senior_ma_job = make_job(
        title="Senior M&A Business Analyst",
        description="Use SQL and Excel for M&A advisory. Support transaction processes.",
        work_model=WorkModel.ONSITE,
    )
    result = match_candidate_to_jobs(profile, [senior_ma_job])
    assert result.excluded_jobs == []  # A flexible preference is not a hard rejection.
    match = result.ranked_matches[0]
    assert match.score is not None
    assert match.recommendation_warnings
    assert match.recommendation_blockers
    assert "Senior-level exception" in match.recommendation_blockers[0]
    assert result.top_matches == []


def test_soft_seniority_mismatch_can_qualify_with_strong_relevant_experience():
    profile = CandidateProfile(
        candidate_id="experienced-junior-pref", name="Experienced Analyst", skills=["SQL"],
        desired_job_titles=["Junior Data Analyst"], seniority_preferences=["junior"],
        professional_experience=[{"title": "Data Analyst", "responsibilities": ["Performed data analysis with SQL and prepared reporting dashboards."]}],
    )
    senior_data_job = make_job(
        title="Senior Data Analyst", description="Use SQL for data analysis and reporting.",
    )
    result = match_candidate_to_jobs(profile, [senior_data_job])
    assert len(result.top_matches) == 1
    assert result.top_matches[0].recommendation_warnings
    assert result.top_matches[0].recommendation_blockers == []


@pytest.mark.parametrize(
    "preference,title,expected",
    [("junior", "Junior Analyst", 100), ("mid-level", "Senior Analyst", 0),
     ("senior", "Senior Analyst", 100), ("junior", "Business Analyst", None)],
)
def test_seniority_cases_are_separated_from_unknown(preference, title, expected):
    profile = CandidateProfile(candidate_id="level", name="Level", seniority_preferences=[preference])
    dimension = next(
        item for item in score_job(profile, make_job(title=title, description="We have junior and senior colleagues.")).dimensions
        if item.name == "seniority_qualifications"
    )
    assert dimension.score == expected


def test_ranking_orders_supported_jobs_and_does_not_fill_top_five_with_weak_evidence():
    profile = load_profile("candidate_profile_b.json")
    result = match_candidate_to_jobs(profile, load_jobs(), top_n=5)
    assert result.top_matches
    assert len(result.top_matches) < 5
    assert all(match.score is not None and match.score >= 50 for match in result.top_matches)
    assert result.top_matches[0].job.job_id in {
        "synthetic-junior-data-analyst", "synthetic-business-intelligence-analyst",
    }
    assert all(match.support_status == SupportStatus.SUFFICIENT for match in result.top_matches)
    assert any(match.support_status == SupportStatus.INSUFFICIENT for match in result.ranked_matches)


def test_profile_a_ranks_transformation_roles_above_analyst_roles():
    result = match_candidate_to_jobs(load_profile("candidate_profile_a.json"), load_jobs())
    assert result.top_matches
    assert result.top_matches[0].job.job_id in {
        "synthetic-ai-transformation-consultant", "synthetic-digital-transformation-pm",
    }


def test_scores_keep_source_traceability_and_whole_number_precision():
    profile = load_profile("candidate_profile_a.json")
    result = score_job(profile, load_jobs()[0])
    assert result.job.source == "synthetic"
    assert result.job.source_url.startswith("https://example.invalid/")
    assert result.score is None or isinstance(result.score, int)
    assert result.evidence_coverage == 100

