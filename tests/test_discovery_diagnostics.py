from career_agent.discovery import JobDiscoveryService
from career_agent.models import Job


class StaticAdapter:
    def __init__(self, jobs):
        self.jobs = jobs

    def search(self, *args, **kwargs):
        return self.jobs


def job(job_id, source, *, title="AI Consultant", **kwargs):
    return Job(job_id=job_id, title=title, company=f"Company {job_id}", source=source, **kwargs)


def test_diagnostics_explain_provider_and_keyword_counts_without_changing_filtering():
    arbeitnow_jobs = [job(f"a{i}", "arbeitnow") for i in range(30)]
    himalayas_jobs = [job(f"h{i}", "himalayas", title="Consultant") for i in range(20)]
    result = JobDiscoveryService(
        arbeitnow=StaticAdapter(arbeitnow_jobs), himalayas=StaticAdapter(himalayas_jobs),
    ).search(query="AI Consultant", country="DE")

    assert result.provider_counts == {"arbeitnow": 30, "himalayas": 20}
    assert result.duplicates_removed == 0
    assert result.deduplicated_provider_counts == {"arbeitnow": 30, "himalayas": 20}
    assert result.filter_diagnostics.excluded_counts == {
        "keyword": 20, "country": 0, "work_model": 0, "employment_type": 0,
    }
    assert result.filter_diagnostics.exclusion_reasons["keyword"] == {"keyword_terms_not_found": 20}
    assert result.final_provider_counts == {"arbeitnow": 30, "himalayas": 0}
    assert len(result.jobs) == 30

    # Country is missing for all jobs, but unknown geography stays included under default policy.
    assert result.filter_diagnostics.missing_country_count == 50
    assert result.filter_diagnostics.missing_country_exclusions == 0
    assert result.filter_diagnostics.missing_location_exclusions == 0


def test_diagnostics_identify_strict_country_exclusions_and_unfiltered_location():
    jobs = [job("a", "arbeitnow"), job("b", "arbeitnow", location="Unknown")]
    result = JobDiscoveryService(
        arbeitnow=StaticAdapter(jobs), himalayas=StaticAdapter([]),
    ).search(query="AI Consultant", country="DE", strict_unknown=True)

    assert result.filter_diagnostics.excluded_counts["country"] == 2
    assert result.filter_diagnostics.exclusion_reasons["country"] == {"country_eligibility_unknown_strict": 2}
    assert result.filter_diagnostics.missing_country_exclusions == 2
    assert result.filter_diagnostics.missing_location_count == 2
    assert result.filter_diagnostics.missing_location_exclusions == 0
    assert result.final_provider_counts == {"arbeitnow": 0, "himalayas": 0}
