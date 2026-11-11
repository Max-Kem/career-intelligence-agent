"""Conservative country and work-location normalization shared by sources and matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# Common ISO 3166-1 alpha-2/alpha-3 codes and provider spellings.
COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "DE": ("DE", "DEU", "Germany", "Deutschland"),
    "NL": ("NL", "NLD", "Netherlands", "The Netherlands", "Nederland", "Holland"),
    "CH": ("CH", "CHE", "Switzerland", "Schweiz", "Suisse", "Svizzera"),
    "FR": ("FR", "FRA", "France", "Frankreich"),
    "GB": ("GB", "GBR", "UK", "United Kingdom", "Great Britain", "England", "Scotland", "Wales"),
    "AT": ("AT", "AUT", "Austria", "Österreich"),
    "BE": ("BE", "BEL", "Belgium", "België", "Belgique"),
    "ES": ("ES", "ESP", "Spain", "España"),
    "IT": ("IT", "ITA", "Italy", "Italia"),
    "PL": ("PL", "POL", "Poland", "Polska"),
    "IE": ("IE", "IRL", "Ireland", "Éire"),
    "PT": ("PT", "PRT", "Portugal"),
    "SE": ("SE", "SWE", "Sweden", "Sverige"),
    "DK": ("DK", "DNK", "Denmark", "Danmark"),
    "NO": ("NO", "NOR", "Norway", "Norge"),
    "FI": ("FI", "FIN", "Finland"),
    "LU": ("LU", "LUX", "Luxembourg"),
    "CZ": ("CZ", "CZE", "Czechia", "Czech Republic"),
    "US": ("US", "USA", "United States", "United States of America"),
    "CA": ("CA", "CAN", "Canada"),
    "AU": ("AU", "AUS", "Australia"),
    "IN": ("IN", "IND", "India"),
    "SG": ("SG", "SGP", "Singapore"),
    "JP": ("JP", "JPN", "Japan"),
    "CN": ("CN", "CHN", "China"),
    "BR": ("BR", "BRA", "Brazil", "Brasil"),
    "MX": ("MX", "MEX", "Mexico", "México"),
    "UA": ("UA", "UKR", "Ukraine", "Україна"),
    "TR": ("TR", "TUR", "Türkiye", "Turkey"),
}


CITY_COUNTRIES: dict[str, str] = {
    # Germany
    "berlin": "DE", "munich": "DE", "munchen": "DE", "muenchen": "DE",
    "hamburg": "DE", "frankfurt": "DE", "frankfurt am main": "DE",
    "stuttgart": "DE", "cologne": "DE", "koln": "DE", "duesseldorf": "DE",
    "dusseldorf": "DE", "dortmund": "DE", "essen": "DE", "bochum": "DE",
    "bonn": "DE", "leipzig": "DE", "dresden": "DE", "hannover": "DE",
    "hanover": "DE", "bremen": "DE", "nuernberg": "DE", "nuremberg": "DE",
    "mainz": "DE", "wiesbaden": "DE", "freiburg": "DE", "karlsruhe": "DE",
    "mannheim": "DE", "aachen": "DE", "augsburg": "DE", "regensburg": "DE",
    "bielefeld": "DE", "muenster": "DE", "munster": "DE", "paderborn": "DE",
    # Netherlands
    "amsterdam": "NL", "rotterdam": "NL", "the hague": "NL", "den haag": "NL",
    "utrecht": "NL", "eindhoven": "NL", "groningen": "NL",
    # Switzerland
    "zurich": "CH", "zuerich": "CH", "geneva": "CH", "geneve": "CH",
    "lausanne": "CH", "basel": "CH", "bern": "CH", "lugano": "CH",
    # France and United Kingdom
    "paris": "FR", "lyon": "FR", "marseille": "FR", "toulouse": "FR", "nantes": "FR",
    "london": "GB", "manchester": "GB", "edinburgh": "GB", "birmingham": "GB",
    # Other common European job locations
    "vienna": "AT", "wien": "AT", "brussels": "BE", "bruxelles": "BE",
    "madrid": "ES", "barcelona": "ES", "rome": "IT", "milano": "IT", "milan": "IT",
    "warsaw": "PL", "krakow": "PL", "dublin": "IE", "lisbon": "PT", "stockholm": "SE",
    "copenhagen": "DK", "oslo": "NO", "helsinki": "FI", "prague": "CZ", "praha": "CZ",
}


def _normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def normalize_country_code(value: Any) -> str | None:
    """Normalize codes, canonical names, and common provider country objects."""
    if isinstance(value, dict):
        for key in ("alpha2", "country_code", "countryCode", "code", "alpha3", "name", "slug", "country"):
            if value.get(key) not in (None, ""):
                normalized = normalize_country_code(value[key])
                if normalized:
                    return normalized
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _normalize_text(value)
    for code, aliases in COUNTRY_ALIASES.items():
        if normalized in {_normalize_text(alias) for alias in aliases}:
            return code
    # Unknown but syntactically valid alpha-2 codes are retained as ISO-like codes.
    if re.fullmatch(r"[a-z]{2}", normalized):
        return normalized.upper()
    return None


def normalize_country_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        code = normalize_country_code(value)
        if code and code not in result:
            result.append(code)
    return result


def country_from_location(value: str | None) -> str | None:
    """Resolve explicit country text or a small common-city map; never inspect URLs."""
    if not isinstance(value, str) or not value.strip():
        return None
    location = _normalize_text(value)
    if not location or any(term in location.split() for term in ("remote", "worldwide", "anywhere")):
        return None
    if location in CITY_COUNTRIES:
        return CITY_COUNTRIES[location]
    # Prefer explicit country names/codes after punctuation and city names.
    parts = re.split(r"[,|/()\-]+", value)
    for part in reversed(parts):
        code = normalize_country_code(part.strip())
        if code:
            return code
    padded = f" {location} "
    # Country names are safe when stated as whole words in a human-readable location.
    for code, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_text(alias)
            if len(normalized_alias) > 3 and f" {normalized_alias} " in padded:
                return code
    return None


def country_codes_mentioned(value: str | None) -> set[str]:
    """Find explicit country names in a title or restriction note (not in a URL)."""
    if not isinstance(value, str):
        return set()
    text = f" {_normalize_text(value)} "
    found: set[str] = set()
    for code, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            normalized = _normalize_text(alias)
            # Avoid interpreting short codes as ordinary title words.
            if len(normalized) > 3 and f" {normalized} " in text:
                found.add(code)
                break
    return found
