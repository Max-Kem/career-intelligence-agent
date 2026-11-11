"""Source-grounded company research using Tavily's Search API."""

from __future__ import annotations

import os
import re
import time
import unicodedata
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
CompanyResearchStatus = Literal["succeeded", "no_results", "missing_api_key", "unavailable"]


class CompanySource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    source_type: Literal["possible_company_website", "third_party"] = "third_party"
    identity_status: Literal["verified", "unverified"] = "unverified"
    identity_reason: str = "No identity evidence assessed."


class CompanyResearchResult(BaseModel):
    company: str
    status: CompanyResearchStatus
    identity_status: Literal["verified", "unverified"] = "unverified"
    identity_reason: str | None = None
    snapshot: str | None = None
    sources: list[CompanySource] = Field(default_factory=list)
    raw_results: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    error: str | None = None


class TavilyCompanyResearchClient:
    """Small bounded client; API credentials are never included in diagnostics."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 1,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        dotenv_loader: Callable[..., Any] = load_dotenv,
    ):
        if not 0.1 <= timeout <= 30:
            raise ValueError("timeout must be between 0.1 and 30 seconds")
        if not 0 <= max_retries <= 1:
            raise ValueError("max_retries must be 0 or 1")
        if api_key is None:
            dotenv_loader(override=False)
            api_key = os.getenv("TAVILY_API_KEY")
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def research_company(self, company: str) -> CompanyResearchResult:
        company = " ".join(company.split())
        if not company:
            raise ValueError("company name cannot be empty")
        if not self._api_key:
            return CompanyResearchResult(
                company=company,
                status="missing_api_key",
                uncertainties=["Tavily research was not performed because TAVILY_API_KEY is unavailable."],
            )

        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        query = f'"{company}" company overview products services industry recent developments official'
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
            "include_answer": False,
            "include_raw_content": False,
        }
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    TAVILY_SEARCH_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                if response.status_code in {401, 403}:
                    return self._unavailable(company, f"Tavily authentication failed (HTTP {response.status_code}).")
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_retries:
                        self._sleep(0.2 * (attempt + 1))
                        continue
                    return self._unavailable(company, f"Tavily search failed (HTTP {response.status_code}).")
                if response.is_error:
                    return self._unavailable(company, f"Tavily search failed (HTTP {response.status_code}).")
                try:
                    data = response.json()
                except (ValueError, TypeError):
                    return self._unavailable(company, "Tavily returned malformed JSON.")
                if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
                    return self._unavailable(company, "Tavily returned an unexpected response shape.")
                results = [item for item in data.get("results", []) if isinstance(item, dict)][:3]
                sources = [_normalize_source(item, company) for item in results]
                sources = [source for source in sources if source is not None]
                sources.sort(key=lambda source: source.identity_status != "verified")
                identity_verified = any(source.identity_status == "verified" for source in sources)
                uncertainties = [
                    "Publication dates are shown only when Tavily returned them; undated information may not be current.",
                ]
                if identity_verified:
                    uncertainties.append("Identity is supported by a matching domain and explicit company-name evidence; individual claims remain source snippets.")
                elif sources:
                    uncertainties.append("Returned sources could not be reliably tied to this company; no company facts are attributed.")
                return CompanyResearchResult(
                    company=company,
                    status="succeeded" if sources else "no_results",
                    identity_status="verified" if identity_verified else "unverified",
                    identity_reason=(
                        "At least one source matched the company domain and explicitly named the full company."
                        if identity_verified else "No source met the domain and explicit-name evidence checks."
                    ),
                    snapshot=next((source.snippet for source in sources if source.identity_status == "verified"), None),
                    sources=sources,
                    raw_results=results,
                    uncertainties=uncertainties if sources else ["Tavily returned no usable search results."],
                )
            except httpx.RequestError as exc:
                last_error = type(exc).__name__
                if attempt < self.max_retries:
                    self._sleep(0.2 * (attempt + 1))
                    continue
                return self._unavailable(company, f"Tavily request failed: {last_error}.")
        return self._unavailable(company, f"Tavily request failed: {last_error or 'unknown error'}.")

    @staticmethod
    def _unavailable(company: str, error: str) -> CompanyResearchResult:
        return CompanyResearchResult(
            company=company, status="unavailable", error=error,
            uncertainties=["Company information could not be verified because the search request failed."],
        )


def unique_recommended_companies(recommendations) -> list[str]:
    """Return distinct non-empty company names, preserving recommendation order."""
    companies: list[str] = []
    seen: set[str] = set()
    for match in recommendations[:5]:
        name = " ".join(match.job.company.split())
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            companies.append(name)
    return companies


def _normalize_source(item: dict[str, Any], company: str) -> CompanySource | None:
    title = item.get("title")
    url = item.get("url")
    snippet = item.get("content")
    if not all(isinstance(value, str) and value.strip() for value in (title, url, snippet)):
        return None
    published_date = item.get("published_date")
    if not isinstance(published_date, str) or not published_date.strip():
        published_date = None
    possible_company_domain = _possible_company_domain(url, company)
    explicit_name = _full_company_name_in_text(company, f"{title} {snippet}")
    identity_verified = possible_company_domain and explicit_name
    if identity_verified:
        identity_reason = "Matching domain plus explicit full company-name mention in the title or snippet."
    elif possible_company_domain:
        identity_reason = "Domain resembles the company name, but the title and snippet do not explicitly identify the company."
    else:
        identity_reason = "Source domain does not match the company name."
    return CompanySource(
        title=title.strip(),
        url=url.strip(),
        snippet=" ".join(snippet.split())[:700],
        published_date=published_date,
        source_type="possible_company_website" if possible_company_domain else "third_party",
        identity_status="verified" if identity_verified else "unverified",
        identity_reason=identity_reason,
    )


def _possible_company_domain(url: str, company: str) -> bool:
    """Heuristic source preference only; this is not authoritative verification."""
    hostname = urlparse(url).hostname or ""
    domain_parts = set(re.findall(r"[a-z0-9]+", hostname.casefold()))
    ignored = {"the", "company", "group", "holdings", "inc", "llc", "ltd", "limited", "gmbh", "ag", "plc"}
    company_tokens = re.findall(r"[a-z0-9]+", company.casefold())
    while company_tokens and company_tokens[-1] in ignored:
        company_tokens.pop()
    company_parts = {part for part in company_tokens if part not in ignored and len(part) >= 3}
    company_name = "".join(company_tokens)
    compact_domain = "".join(domain_parts)
    return bool(domain_parts & company_parts) or bool(company_name and company_name in compact_domain)


def _full_company_name_in_text(company: str, text: str) -> bool:
    def normalize(value: str) -> str:
        ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        tokens = re.findall(r"[a-z0-9]+", ascii_text.casefold())
        legal_suffixes = {"inc", "llc", "ltd", "limited", "gmbh", "ag", "plc", "corp", "corporation"}
        while tokens and tokens[-1] in legal_suffixes:
            tokens.pop()
        return " ".join(tokens)

    company_name = normalize(company)
    source_text = normalize(text)
    return bool(company_name and company_name in source_text)
