from __future__ import annotations

import time
from typing import Any

import httpx


class ProviderError(RuntimeError):
    """A provider request or response could not be used."""


class BaseAdapter:
    source: str

    def __init__(self, *, timeout: float = 15.0, retries: int = 2, client: httpx.Client | None = None):
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        self.timeout = timeout
        self.retries = retries
        self._client = client

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(self.retries + 1):
            try:
                if self._client is None:
                    response = httpx.get(url, params=params, timeout=self.timeout, follow_redirects=True)
                else:
                    response = self._client.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.retries:
                        time.sleep(min(0.25 * (2**attempt), 1.0))
                        continue
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as exc:
                    raise ProviderError(f"{self.source} returned invalid JSON") from exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self.retries:
                    raise ProviderError(f"{self.source} request failed: {type(exc).__name__}") from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {429} or status >= 500:
                    if attempt < self.retries:
                        time.sleep(min(0.25 * (2**attempt), 1.0))
                        continue
                raise ProviderError(f"{self.source} returned HTTP {status}") from exc
        raise ProviderError(f"{self.source} request failed after retries")
