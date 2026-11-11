from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .geography import normalize_country_code


class WorkModel(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class EmploymentType(str, Enum):
    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    UNKNOWN = "unknown"


class Eligibility(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


def _clean(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    result = str(value).strip()
    return result or None


class Job(BaseModel):
    """Normalized provider listing, retaining provider data for traceability."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    title: str
    company: str = "Unknown company"
    location: str = "Unknown"
    country: str | None = None
    work_model: WorkModel = WorkModel.UNKNOWN
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    seniority: str | None = None
    description: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    source: str
    source_url: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    country_eligibility: Eligibility = Eligibility.UNKNOWN
    eligible_countries: list[str] = Field(default_factory=list)
    timezone_restrictions: list[str] = Field(default_factory=list)
    provider_id: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_id", "title", "company", "location", "source", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> str:
        return _clean(value) or "Unknown"

    @field_validator("country", "seniority", "description", "salary_currency", "salary_period", "source_url", "provider_id", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        return _clean(value)

    @field_validator("country", mode="before")
    @classmethod
    def normalize_country(cls, value: Any) -> str | None:
        cleaned = _clean(value)
        return normalize_country_code(cleaned) or cleaned

    @field_validator("eligible_countries", mode="before")
    @classmethod
    def normalize_eligible_countries(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            cleaned = _clean(item)
            canonical = normalize_country_code(cleaned) if cleaned else None
            retained = canonical or cleaned
            if retained and retained not in normalized:
                normalized.append(retained)
        return normalized


def normalize_work_model(value: Any) -> WorkModel:
    if isinstance(value, bool):
        return WorkModel.REMOTE if value else WorkModel.UNKNOWN
    text = (_clean(value) or "").lower()
    if any(word in text for word in ("hybrid", "partly remote", "flexible")):
        return WorkModel.HYBRID
    if any(word in text for word in ("remote", "work from anywhere", "worldwide")):
        return WorkModel.REMOTE
    if any(word in text for word in ("on-site", "onsite", "office-based")):
        return WorkModel.ONSITE
    return WorkModel.UNKNOWN


def normalize_employment_type(value: Any) -> EmploymentType:
    if isinstance(value, list):
        value = value[0] if value else None
    text = (_clean(value) or "").lower().replace("_", "-").replace(" ", "-")
    if text in {"full-time", "fulltime", "permanent"}:
        return EmploymentType.FULL_TIME
    if text in {"part-time", "parttime"}:
        return EmploymentType.PART_TIME
    if any(word in text for word in ("freelance", "self-employed")):
        return EmploymentType.FREELANCE
    if any(word in text for word in ("contract", "contractor", "temporary", "temp")):
        return EmploymentType.CONTRACT
    return EmploymentType.UNKNOWN
