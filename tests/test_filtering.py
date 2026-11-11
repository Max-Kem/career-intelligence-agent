from career_agent.filtering import country_eligibility, filter_jobs
from career_agent.models import Eligibility, EmploymentType, Job, WorkModel


def make_job(**kwargs):
    values = {"job_id": "1", "title": "AI Consultant", "company": "Acme", "source": "test"}
    values.update(kwargs)
    return Job(**values)


def test_keyword_filter_is_case_insensitive_all_terms():
    jobs = [make_job(description="Consulting AI platform"), make_job(job_id="2", title="Data engineer")]
    assert len(filter_jobs(jobs, keywords="ai consulting")) == 1


def test_country_filter_keeps_unknown_by_default_and_strict_drops():
    unknown = make_job()
    germany = make_job(job_id="2", eligible_countries=["DE"])
    france = make_job(job_id="3", eligible_countries=["FR"])
    assert country_eligibility(unknown, "DE") == Eligibility.UNKNOWN
    assert [j.job_id for j in filter_jobs([unknown, germany, france], country="DE")] == ["1", "2"]
    assert [j.job_id for j in filter_jobs([unknown, germany, france], country="DE", strict_unknown=True)] == ["2"]


def test_work_and_employment_filter_drop_confirmed_mismatch_but_keep_unknown():
    jobs = [make_job(work_model=WorkModel.REMOTE, employment_type=EmploymentType.FULL_TIME),
            make_job(job_id="2", work_model=WorkModel.UNKNOWN, employment_type=EmploymentType.UNKNOWN),
            make_job(job_id="3", work_model=WorkModel.ONSITE, employment_type=EmploymentType.PART_TIME)]
    result = filter_jobs(jobs, work_model=WorkModel.REMOTE, employment_type=EmploymentType.FULL_TIME)
    assert [j.job_id for j in result] == ["1", "2"]
    assert [j.job_id for j in filter_jobs(jobs, work_model=WorkModel.REMOTE,
                                         employment_type=EmploymentType.FULL_TIME, strict_unknown=True)] == ["1"]


def test_verified_worldwide_listing_is_country_eligible():
    job = make_job(country_eligibility=Eligibility.ELIGIBLE)
    assert country_eligibility(job, "DE") == Eligibility.ELIGIBLE


def test_job_location_does_not_override_remote_candidate_country_restriction():
    job = make_job(country="DE", eligible_countries=["US"])
    assert country_eligibility(job, "DE") == Eligibility.INELIGIBLE


def test_job_location_alone_does_not_verify_remote_candidate_eligibility():
    job = make_job(country="DE")
    assert country_eligibility(job, "DE") == Eligibility.UNKNOWN
