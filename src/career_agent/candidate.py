from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import EmploymentType, WorkModel


class ExperienceEntry(BaseModel):
    title: str
    organization: str | None = None
    responsibilities: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class EducationEntry(BaseModel):
    qualification: str
    field: str | None = None
    institution: str | None = None

    model_config = ConfigDict(extra="ignore")


class HardRequirements(BaseModel):
    """Hard constraints; local roles must be in an allowed country, remote roles eligible from one."""

    countries: list[str] = Field(default_factory=list)
    work_models: list[WorkModel] = Field(default_factory=list)
    employment_types: list[EmploymentType] = Field(default_factory=list)
    seniority: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    name: str
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    professional_experience: list[ExperienceEntry] = Field(default_factory=list)
    years_experience: float | None = Field(default=None, ge=0)
    education: list[EducationEntry] = Field(default_factory=list)
    desired_job_titles: list[str] = Field(default_factory=list)
    career_interests: list[str] = Field(default_factory=list)
    preferred_industries: list[str] = Field(default_factory=list)
    seniority_preferences: list[str] = Field(default_factory=list)
    country_preferences: list[str] = Field(default_factory=list)
    location_preferences: list[str] = Field(default_factory=list)
    work_model_preferences: list[WorkModel] = Field(default_factory=list)
    employment_type_preferences: list[EmploymentType] = Field(default_factory=list)
    required_preferences: HardRequirements = Field(default_factory=HardRequirements)

    @field_validator(
        "skills", "desired_job_titles", "career_interests", "preferred_industries",
        "seniority_preferences", "country_preferences", "location_preferences", mode="before",
    )
    @classmethod
    def clean_string_lists(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return value

    @field_validator("summary", mode="before")
    @classmethod
    def clean_summary(cls, value):
        return value.strip() or None if isinstance(value, str) else value
