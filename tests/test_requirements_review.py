from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from career_agent.candidate import CandidateProfile, ExperienceEntry
from career_agent.matching import JobMatch, MatchingResult, SupportStatus, _required_years, career_direction_alignment, match_candidate_to_jobs
from career_agent.models import Job
from career_agent.requirements_review import critical_requirements_to_verify, requirements_to_verify
from career_agent.workflow import create_report


ROOT = Path(__file__).resolve().parents[1]


def profile_a() -> CandidateProfile:
    return CandidateProfile.model_validate_json((ROOT / "fixtures" / "candidate_profile_a.json").read_text(encoding="utf-8"))


def profile_c() -> CandidateProfile:
    return CandidateProfile.model_validate_json((ROOT / "fixtures" / "candidate_profile_c.json").read_text(encoding="utf-8"))


def profile_d() -> CandidateProfile:
    return CandidateProfile.model_validate_json((ROOT / "fixtures" / "candidate_profile_d.json").read_text(encoding="utf-8"))


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


def test_experience_ranges_are_not_rewritten_as_upper_bound_minimums():
    candidate = profile_a()
    result = requirements_to_verify(
        "5–8 years of relevant experience is required. 2-4 years of consulting experience is preferred.",
        candidate,
    )
    by_name = {item["requirement"]: item for item in result}
    assert "5–8 years of experience" in by_name
    assert "2–4 years of experience" in by_name
    assert by_name["5–8 years of experience"]["candidate_evidence_status"] == "not_documented"
    assert by_name["2–4 years of experience"]["requirement_type"] == "preferred"
    assert all("At least 8" not in name and "At least 4" not in name for name in by_name)
    assert _required_years("5–8 years of relevant experience") == 5
    assert _required_years("2-4 years of professional experience") == 2
    assert _required_years("5+ years of relevant experience") == 5


def test_experience_range_compares_against_lower_bound_while_minimum_and_exact_remain_distinct():
    candidate = profile_a().model_copy(update={"years_experience": 4})
    result = requirements_to_verify(
        "5–8 years of experience is required. At least 5 years of experience is required. "
        "Exactly 4 years of experience is required.", candidate,
    )
    by_name = {item["requirement"]: item for item in result}
    assert by_name["5–8 years of experience"]["candidate_evidence_status"] == "does_not_meet_requirement"
    assert by_name["At least 5 years of experience"]["candidate_evidence_status"] == "does_not_meet_requirement"
    exact = requirements_to_verify("Exactly 4 years of experience is required.", profile_a())
    assert exact[0]["requirement"] == "Exactly 4 years of experience"
    assert exact[0]["candidate_evidence_status"] == "not_documented"


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


def test_specialist_requirements_move_transferable_but_unverified_matches_to_review_without_score_penalty():
    candidate = profile_a()
    samples = [
        ("Senior Strategic Advisor - EMEA", "Qualifications: 6+ years of relevant experience in management consulting or CPG leadership. Deep domain expertise in CPG commercial analytics, pricing architecture, and trade promotion optimization (RGM & TPM)."),
        ("Senior Indirect Tax Manager", "What You Bring: Multiple years of in-house indirect tax experience. Deep, hands-on expertise in European VAT."),
        ("IT Consultant", "Job requirements: Experience with development frameworks and core systems. Work in the Core Banking Systems practice with Avaloq or Temenos."),
        ("Principal AI Transformation Consultant", "Your Career Path: Typically you bring 3+ years in transformation consulting. As a Principal AI Transformation Consultant, you lead executive accounts. What You Bring: Typically 7+ years in transformation consulting. Deep, end-to-end knowledge of purchase-to-pay or order-to-cash. Executive presence. Fluent in German and English."),
    ]
    for title, description in samples:
        critical = critical_requirements_to_verify(description, candidate, title)
        assert critical, title
    principal = requirements_to_verify(samples[-1][1], candidate, samples[-1][0])
    years = [item["requirement"] for item in principal if item["category"] == "professional_experience"]
    assert years == ["At least 7 years of experience"]
    languages = {item["requirement"] for item in principal if item["category"] == "language"}
    assert languages == {"Language proficiency: German", "Language proficiency: English"}


def test_specialist_expertise_documented_in_candidate_history_does_not_trigger_review_blocker():
    candidate = profile_a().model_copy(update={
        "professional_experience": [ExperienceEntry(
            title="Commercial Transformation Consultant",
            responsibilities=["Led Revenue Growth Management and Trade Promotion Management for CPG clients."],
        )],
    })
    text = "Deep domain expertise in CPG commercial analytics, pricing architecture, and trade promotion optimization (RGM & TPM)."
    assert critical_requirements_to_verify(text, candidate, "Senior Strategic Advisor") == []


def test_html_decoding_discards_disclaimer_certification_and_extracts_both_languages():
    html_description = (
        "&amp;lt;div&amp;gt;&amp;lt;p&amp;gt;Fortune 500&amp;reg; is a registered trademark, used under license.&amp;lt;/p&amp;gt;"
        "&amp;lt;h3&amp;gt;What You Bring&amp;lt;/h3&amp;gt;"
        "&amp;lt;p&amp;gt;Fluent in &amp;lt;strong&amp;gt;German&amp;lt;/strong&amp;gt; and English, written and spoken.&amp;lt;/p&amp;gt;&amp;lt;/div&amp;gt;"
    )
    result = requirements_to_verify(html_description, profile_a(), "Consultant")
    assert {item["requirement"] for item in result} == {
        "Language proficiency: German", "Language proficiency: English",
    }
    assert not any(item["category"] == "certification" for item in result)


def test_generic_consulting_requirements_and_optional_certification_do_not_become_specialist_blockers():
    candidate = profile_a()
    text = "What You Bring: Strong project management and stakeholder communication. Avaloq certification is a plus."
    requirements = requirements_to_verify(text, candidate, "Business Consultant")
    assert any(item["category"] == "certification" and item["requirement_type"] == "preferred" for item in requirements)
    assert critical_requirements_to_verify(text, candidate, "Business Consultant") == []


def test_path_specific_avaloq_certification_is_not_a_global_requirement():
    result = requirements_to_verify(
        "For the Avaloq path, an Avaloq Certified Customizer (ACCP) certification is required (or was previously held).",
        profile_a(), "IT Consultant",
    )
    certification = next(item for item in result if item["category"] == "certification")
    assert certification["requirement_type"] == "conditional"
    assert critical_requirements_to_verify(
        "For the Avaloq path, an Avaloq Certified Customizer (ACCP) certification is required (or was previously held).",
        profile_a(), "IT Consultant",
    ) == []


def test_language_proficiency_is_visible_as_not_documented_but_never_a_recommendation_blocker():
    candidate = profile_c()
    text = "Fluent German (required) for customer communication."
    language = next(item for item in requirements_to_verify(text, candidate, "AI Transformation Consultant") if item["category"] == "language")
    assert language["requirement_type"] == "required"
    assert language["candidate_evidence_status"] == "not_documented"
    assert "does not mean" in language["candidate_evidence"]
    assert not critical_requirements_to_verify(text, candidate, "AI Transformation Consultant")


def test_explicit_language_name_without_proficiency_is_not_treated_as_fluency():
    candidate = profile_c().model_copy(update={"skills": [*profile_c().skills, "English"]})
    result = requirements_to_verify("Business-level English is required.", candidate, "AI Transformation Consultant")
    language = next(item for item in result if item["category"] == "language")
    assert language["candidate_evidence_status"] == "not_documented"


def test_specialized_years_and_project_scale_are_not_satisfied_by_total_experience():
    candidate = profile_c()
    description = (
        "Typically you bring 5+ years delivering complex, multi-country programs. "
        "5+ years of experience in intralogistics and/or robotics industry. "
        "Experience leading delivery teams and steering committees is required. "
        "ERP implementation experience is a bonus. Fluent German (required)."
    )
    requirements = requirements_to_verify(description, candidate, "Senior Project Manager")
    by_label = {item["requirement"]: item for item in requirements}
    specialized = by_label["At least 5 years of complex multi-country / multi-entity program delivery experience"]
    robotics = by_label["At least 5 years of robotics or intralogistics experience"]
    assert specialized["candidate_evidence_status"] == "not_documented"
    assert robotics["candidate_evidence_status"] == "not_documented"
    erp = next(item for item in requirements if item["category"] == "specialized_expertise" and "ERP" in item["requirement"])
    assert erp["requirement_type"] == "preferred"
    critical = critical_requirements_to_verify(description, candidate, "Senior Project Manager")
    assert any(item["category"] == "project_scale" for item in critical)
    assert any(item["category"] == "specialized_expertise" for item in critical)
    assert not any(item["category"] == "language" for item in critical)


def test_explicit_statement_that_role_is_not_a_global_program_does_not_create_scale_blocker():
    candidate = profile_c()
    description = "Most implementations are not multi-year global programs; they are delivered for local customers within weeks."
    assert not any(item["category"] == "project_scale"
                   for item in critical_requirements_to_verify(description, candidate, "Project Manager"))


def test_profile_d_senior_data_requirements_and_specialist_tools_are_visible_and_block_review_only():
    candidate = profile_d()
    description = (
        "Solid senior-level experience as a data analyst in SaaS. Real experimentation depth, including designed and sized A/B tests. "
        "Expert SQL, Python for analysis, comfort in Looker (or a close equivalent). "
        "Strong data modelling and analytics engineering, with real dbt mileage."
    )
    requirements = requirements_to_verify(description, candidate, "Senior Data Analyst")
    rows = {item["requirement"]: item for item in requirements}
    assert rows["Senior-level professional experience"]["candidate_evidence_status"] == "not_documented"
    assert rows["SaaS analytics experience"]["candidate_evidence_status"] == "not_documented"
    assert rows["Experiment design and analysis experience"]["candidate_evidence_status"] == "not_documented"
    assert rows["Advanced data modeling / analytics engineering experience"]["candidate_evidence_status"] == "not_documented"
    assert rows["dbt qualification"]["candidate_evidence_status"] == "not_documented"
    assert rows["SQL qualification"]["candidate_evidence_status"] == "partially_documented"
    assert "Python qualification" not in rows  # Documented as a profile skill; no requirement-to-verify record needed.
    assert "Looker qualification" not in rows  # Profile D's Power BI is the posting's stated close equivalent.
    critical = critical_requirements_to_verify(description, candidate, "Senior Data Analyst")
    assert {item["requirement"] for item in critical} >= {
        "Senior-level professional experience", "Experiment design and analysis experience",
        "Advanced data modeling / analytics engineering experience", "dbt qualification",
    }
    assert all("does not mean the candidate lacks" in item["candidate_evidence"]
               for item in requirements if item["candidate_evidence_status"] == "not_documented")


def test_shiftmove_senior_years_tableau_snowflake_and_dbt_are_material_but_nice_to_have_is_optional():
    candidate = profile_d()
    description = (
        "At least three years working at a senior level. Expert-level Tableau dashboard development skills. "
        "Strong SQL and hands-on experience working with Snowflake. Practical experience with dbt. "
        "Nice to have: Experience working with Salesforce, HubSpot or NetSuite data models."
    )
    requirements = requirements_to_verify(description, candidate, "Interim Senior BI Analyst")
    rows = {item["requirement"]: item for item in requirements}
    assert rows["Senior-level professional experience"]["candidate_evidence_status"] == "not_documented"
    assert rows["Tableau qualification"]["candidate_evidence_status"] == "not_documented"
    assert rows["Snowflake qualification"]["candidate_evidence_status"] == "not_documented"
    assert rows["dbt qualification"]["candidate_evidence_status"] == "not_documented"
    assert rows["Salesforce qualification"]["requirement_type"] == "preferred"
    critical = critical_requirements_to_verify(description, candidate, "Interim Senior BI Analyst")
    assert {item["requirement"] for item in critical} >= {
        "Senior-level professional experience", "Tableau qualification", "Snowflake qualification", "dbt qualification",
    }
    assert not any(item["requirement"] == "Salesforce qualification" for item in critical)


def test_total_years_do_not_score_as_specialist_experience_but_generic_years_still_use_total():
    candidate = profile_c()
    specialist_job = Job(
        job_id="robotics", title="Senior Project Manager (Robotics)", company="Example", source="test",
        location="Munich", country="DE", description="5+ years of experience in robotics industry.",
    )
    generic_job = specialist_job.model_copy(update={
        "job_id": "generic-years", "title": "AI Transformation Consultant",
        "description": "At least 5+ years of relevant professional experience is required.",
    })
    specialist_match = match_candidate_to_jobs(candidate, [specialist_job]).ranked_matches[0]
    generic_match = match_candidate_to_jobs(candidate, [generic_job]).ranked_matches[0]
    specialist_experience = next(item for item in specialist_match.dimensions if item.name == "experience")
    generic_experience = next(item for item in generic_match.dimensions if item.name == "experience")
    assert specialist_experience.score is None
    assert generic_experience.score == 100
    assert not match_candidate_to_jobs(candidate, [specialist_job]).top_matches


def test_career_direction_gate_distinguishes_target_adjacent_and_weak_functions():
    candidate = profile_c()
    samples = [
        ("AI Transformation Consultant", "Implement AI adoption and transformation projects.", "aligned"),
        ("Digital Transformation Project Manager", "Deliver digital transformation rollouts and process redesign; lead change adoption.", "adjacent"),
        ("Project Manager", "Plan schedules, manage budgets and publish status updates.", "uncertain"),
        ("Product Marketing Operations Manager", "Automate campaign operations and maintain marketing dashboards.", "weak"),
        ("Senior Customer Success Manager", "Drive account renewals, customer adoption and portfolio retention.", "weak"),
        ("Senior Project Manager (Intralogistics and Robotics)", "Manage robot deployments in warehouses.", "weak"),
    ]
    for title, description, expected in samples:
        status, evidence = career_direction_alignment(candidate, Job(
            job_id=title, title=title, company="Example", source="test", description=description,
        ))
        assert status == expected, (title, evidence)


def test_weak_high_scoring_career_direction_goes_to_review_without_score_change():
    candidate = profile_c()
    job = Job(
        job_id="marketing-ops", title="Product Marketing Operations Manager", company="Example", source="test",
        location="Munich", country="DE", work_model="hybrid",
        description=("Manage project delivery and stakeholders. Automate workflow operations and improve processes. "
                     "Lead marketing operations reporting and campaign execution."),
    )
    match = match_candidate_to_jobs(candidate, [job]).ranked_matches[0]
    assert match.score is not None
    assert match.career_direction_status == "weak"
    assert any("Career-direction alignment is weak" in blocker for blocker in match.recommendation_blockers)
    assert match not in match_candidate_to_jobs(candidate, [job]).top_matches


def test_report_clarifies_matching_evidence_coverage_without_changing_score_fields():
    candidate = profile_a()
    job = Job(job_id="ai", title="AI Consultant", company="Example", source="test", location="Berlin", country="DE", description="AI implementation and project delivery.")
    match = JobMatch(job, 78, [], 80, 4, SupportStatus.SUFFICIENT, "passed")
    markdown, data = create_report({
        "candidate": candidate, "discovery": SimpleNamespace(jobs=[job]), "matching": MatchingResult([match], [match], []),
        "recommendations": [match], "provider_diagnostics": {"provider_counts": {}, "filter_exclusions": {}}, "offline": True,
    })
    assert data["recommendations"][0]["evidence_coverage"] == 80
    assert "percentage of weighted matching signals" in data["evidence_coverage_interpretation"].casefold()
    assert "Matching evidence coverage: 80% of evaluated signals" in markdown
    assert "not a measure of full employer requirement fulfillment" in markdown
