"""Job discovery foundation for Career Intelligence Agent."""

from .discovery import DiscoveryResult, JobDiscoveryService, SearchPreferences
from .candidate import CandidateProfile, HardRequirements
from .integration import CandidateJobRun, discover_and_match
from .matching import MatchingResult, ScoringWeights, match_candidate_to_jobs
from .models import EmploymentType, Job, WorkModel

__all__ = [
    "CandidateJobRun", "CandidateProfile", "DiscoveryResult", "EmploymentType", "HardRequirements", "Job",
    "JobDiscoveryService", "MatchingResult", "ScoringWeights", "SearchPreferences",
    "WorkModel", "discover_and_match", "match_candidate_to_jobs",
]
