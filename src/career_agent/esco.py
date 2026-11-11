from __future__ import annotations

import unicodedata
import time
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field


ESCO_API_BASE = "https://ec.europa.eu/esco/api"
ESCO_SOURCE_URL = "https://ec.europa.eu/esco/api/search"
ESCO_LANGUAGE = "en"
MatchStatus = Literal["exact", "alternative", "ambiguous", "non_exact", "no_match", "unavailable"]


class ESCOConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str | None = None
    uri: str | None = None
    concept_type: str
    language: str
    matched_label: str | None = None
    match_type: Literal["exact", "alternative", "ambiguous", "non_exact"]
    source_reference: str
    description: str | None = None


class ESCOQueryResult(BaseModel):
    term: str
    concept_type: str
    language: str
    status: MatchStatus
    concepts: list[ESCOConcept] = Field(default_factory=list)
    source_reference: str
    error: str | None = None


class ESCOClientError(RuntimeError):
    """Raised for an unavailable or malformed ESCO response."""


class ESCOClient:
    """Small client for ESCO's documented full-text `/search` endpoint."""

    def __init__(
        self,
        *,
        base_url: str = ESCO_API_BASE,
        timeout: float = 6.0,
        max_retries: int = 1,
        result_limit: int = 8,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not 0.1 <= timeout <= 30:
            raise ValueError("timeout must be between 0.1 and 30 seconds")
        if not 0 <= max_retries <= 2:
            raise ValueError("max_retries must be between 0 and 2")
        if not 1 <= result_limit <= 20:
            raise ValueError("result_limit must be between 1 and 20")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.result_limit = result_limit
        self._client = client or httpx.Client(timeout=timeout, headers={"Accept": "application/json"})
        self._owns_client = client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def lookup_skill(self, term: str, *, language: str = ESCO_LANGUAGE) -> ESCOQueryResult:
        return self.lookup(term, concept_type="Skill", language=language)

    def lookup_occupation(self, term: str, *, language: str = ESCO_LANGUAGE) -> ESCOQueryResult:
        return self.lookup(term, concept_type="Occupation", language=language)

    def lookup(self, term: str, *, concept_type: str, language: str = ESCO_LANGUAGE) -> ESCOQueryResult:
        term = term.strip()
        language = language.strip()
        if not term:
            raise ValueError("ESCO lookup term cannot be empty")
        if concept_type not in {"Skill", "Occupation"}:
            raise ValueError("concept_type must be Skill or Occupation")
        if not language:
            raise ValueError("language cannot be empty")

        params = {
            "text": term,
            "language": language,
            "type": {"Skill": "skill", "Occupation": "occupation"}[concept_type],
            "full": "true",
            "limit": self.result_limit,
        }
        source = f"{self.base_url}/search?{urlencode(params)}"
        payload = self._get_json(params)
        items = _extract_items(payload)
        candidates = [_normalize_concept(item, concept_type, language, source) for item in items if isinstance(item, dict)]
        candidates = [item for item in candidates if item.uri or item.label]
        exact = [item for item in candidates if item.label and _label_key(item.label) == _label_key(term)]
        if len(exact) == 1:
            return ESCOQueryResult(term=term, concept_type=concept_type, language=language,
                                   status="exact", concepts=[_with_match(exact[0], term, "exact")], source_reference=source)
        if len(exact) > 1:
            return ESCOQueryResult(term=term, concept_type=concept_type, language=language,
                                   status="ambiguous", concepts=[_with_match(item, term, "ambiguous") for item in exact], source_reference=source)

        alternatives = [
            (item, label) for item, label in _alternative_labels(items, language)
            if _label_key(label) == _label_key(term)
        ]
        if len(alternatives) == 1:
            item, matched_label = alternatives[0]
            concept = _normalize_concept(item, concept_type, language, source)
            return ESCOQueryResult(term=term, concept_type=concept_type, language=language,
                                   status="alternative", concepts=[_with_match(concept, matched_label, "alternative")], source_reference=source)
        if len(alternatives) > 1:
            concepts = []
            for item, matched_label in alternatives:
                concept = _normalize_concept(item, concept_type, language, source)
                concepts.append(_with_match(concept, matched_label, "ambiguous"))
            return ESCOQueryResult(term=term, concept_type=concept_type, language=language,
                                   status="ambiguous", concepts=concepts, source_reference=source)
        if candidates:
            return ESCOQueryResult(
                term=term,
                concept_type=concept_type,
                language=language,
                status="non_exact",
                concepts=[_with_match(item, item.label or "", "non_exact") for item in candidates],
                source_reference=source,
            )
        return ESCOQueryResult(term=term, concept_type=concept_type, language=language,
                               status="no_match", source_reference=source)

    def _get_json(self, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}/search"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                try:
                    return response.json()
                except (ValueError, TypeError) as exc:
                    raise ESCOClientError("ESCO returned malformed JSON") from exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
                if not retryable or attempt >= self.max_retries:
                    # ESCO documents that a 400 response includes a validation
                    # message. Keep a bounded excerpt so invalid query values
                    # are diagnosable without dumping an unbounded payload.
                    detail = " ".join(exc.response.text.split())[:1000]
                    suffix = f": {detail}" if detail else ""
                    raise ESCOClientError(
                        f"ESCO API returned HTTP {exc.response.status_code}{suffix}"
                    ) from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise ESCOClientError(f"ESCO request failed: {type(exc).__name__}") from exc
            if attempt < self.max_retries:
                self._sleep(0.2 * (attempt + 1))
        raise ESCOClientError(f"ESCO request failed: {last_error}")


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ESCOClientError("ESCO search returned an unexpected response shape")
    for key in ("results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    embedded = payload.get("_embedded")
    if isinstance(embedded, dict):
        for key in ("results", "result", "concepts", "resources", "items"):
            value = embedded.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        lists = [value for value in embedded.values() if isinstance(value, list)]
        if len(lists) == 1:
            return [item for item in lists[0] if isinstance(item, dict)]
    raise ESCOClientError("ESCO search response did not include a recognized result collection")


def _normalize_concept(item: dict[str, Any], concept_type: str, language: str, source: str) -> ESCOConcept:
    preferred = item.get("preferredLabel")
    label, label_language = _localized_value(preferred, language)
    label = label or _string(item.get("title")) or _string(item.get("label")) or _string(item.get("name"))
    uri = _string(item.get("uri")) or _link_href(item.get("_links"), "self")
    actual_type = _string(item.get("className")) or _string(item.get("type")) or concept_type
    description, _ = _localized_value(item.get("description"), language)
    return ESCOConcept(
        label=label, uri=uri, concept_type=actual_type,
        language=label_language or language, match_type="exact",
        source_reference=uri or source, description=description,
    )


def _alternative_labels(items: list[dict[str, Any]], language: str):
    for item in items:
        labels = item.get("altLabels", item.get("alternativeLabels", item.get("altLabel", [])))
        if isinstance(labels, dict):
            labels = labels.get(language, [])
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, str):
                    yield item, label


def _localized_value(value: Any, language: str) -> tuple[str | None, str | None]:
    if isinstance(value, str):
        return value, language
    if isinstance(value, dict):
        selected = value.get(language)
        if isinstance(selected, str):
            return selected, language
        for lang, text in value.items():
            if isinstance(text, str):
                return text, str(lang)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text, lang = _localized_value(item, language)
                if text:
                    return text, lang
            elif isinstance(item, str):
                return item, language
    return None, None


def _link_href(links: Any, name: str) -> str | None:
    if not isinstance(links, dict):
        return None
    item = links.get(name)
    if isinstance(item, dict):
        return _string(item.get("href"))
    if isinstance(item, list) and item and isinstance(item[0], dict):
        return _string(item[0].get("href"))
    return None


def _with_match(concept: ESCOConcept, matched_label: str, match_type: str) -> ESCOConcept:
    return concept.model_copy(update={"matched_label": matched_label, "match_type": match_type})


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _label_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def enrich_candidate_and_matches(
    candidate,
    recommendations,
    client: ESCOClient,
    *,
    language: str = ESCO_LANGUAGE,
    max_lookups: int = 32,
) -> dict[str, Any]:
    """Look up listed profile skills and matcher-extracted skills on up to five jobs."""
    if not 1 <= max_lookups <= 32:
        raise ValueError("max_lookups must be between 1 and 32")
    queries: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(term: str, concept_type: str, evidence_source: str, job_id: str | None = None) -> None:
        key = (concept_type, _label_key(term))
        if term.strip() and key not in seen and len(queries) < max_lookups:
            seen.add(key)
            owner = job_id or "candidate"
            queries.append((term.strip(), concept_type, evidence_source, owner))

    for skill in candidate.skills[:12]:
        add(skill, "Skill", "candidate_profile")
    for match in recommendations[:5]:
        job_skills = list(dict.fromkeys([*match.matched_skills, *match.missing_or_unverified_skills]))[:3]
        for skill in job_skills:
            add(skill, "Skill", "existing_job_skill_extraction", match.job.job_id)
        add(match.job.title, "Occupation", "selected_job_title", match.job.job_id)

    outcomes: list[dict[str, Any]] = []
    failures = 0
    for term, concept_type, evidence_source, owner in queries:
        try:
            result = client.lookup(term, concept_type=concept_type, language=language)
        except Exception as exc:
            failures += 1
            result = ESCOQueryResult(
                term=term, concept_type=concept_type, language=language,
                status="unavailable", concepts=[], source_reference=ESCO_SOURCE_URL,
                error=f"{type(exc).__name__}: {exc}",
            )
        outcomes.append({
            "subject": owner,
            "subject_type": "candidate" if owner == "candidate" else "job",
            "evidence_source": evidence_source,
            "term": term,
            "concept_type": concept_type,
            "interpretation": _interpretation_text(result.status),
            "esco_result": result.model_dump(mode="json"),
        })

    if not queries:
        status = "skipped_no_terms"
    elif failures == len(queries):
        status = "unavailable"
    elif failures:
        status = "partial"
    else:
        status = "succeeded"
    return {
        "status": status,
        "language": language,
        "lookup_count": len(queries),
        "failed_lookup_count": failures,
        "lookup_limit": max_lookups,
        "lookups_truncated": _would_exceed_limit(candidate, recommendations, max_lookups),
        "source_reference": ESCO_SOURCE_URL,
        "official_api_documentation": "https://ec.europa.eu/esco/api/doc/esco_api_doc.html",
        "lookups": outcomes,
    }


def _interpretation_text(status: MatchStatus) -> str:
    return {
        "exact": "Our label comparison found one exact preferred-label match in ESCO.",
        "alternative": "Our label comparison found one exact ESCO alternative-label match.",
        "ambiguous": "Our label comparison found multiple matching ESCO concepts; no single concept was selected.",
        "non_exact": "ESCO returned concepts, but none had an exact preferred or alternative label match; equivalence was not inferred.",
        "no_match": "No exact preferred or alternative label match was found in the returned ESCO results.",
        "unavailable": "ESCO lookup was unavailable; no concept equivalence was inferred.",
    }[status]


def _would_exceed_limit(candidate, recommendations, max_lookups: int) -> bool:
    unique: set[tuple[str, str]] = set()
    for skill in candidate.skills[:12]:
        unique.add(("Skill", _label_key(skill)))
    for match in recommendations[:5]:
        for skill in list(dict.fromkeys([*match.matched_skills, *match.missing_or_unverified_skills]))[:3]:
            unique.add(("Skill", _label_key(skill)))
        unique.add(("Occupation", _label_key(match.job.title)))
    return len(unique) > max_lookups
