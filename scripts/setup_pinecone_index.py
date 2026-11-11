"""Safely check or create the Career Intelligence Pinecone index."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

INDEX_NAME = "career-market-knowledge"
VECTOR_FIELD = "chunk_vector"
DIMENSION = 1536
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
DEPLOYMENT_TYPE = "managed"
PROVISIONING_TIMEOUT_SECONDS = 120
PROVISIONING_POLL_SECONDS = 2


class SetupError(RuntimeError):
    """Expected configuration, compatibility, or Pinecone API failure."""


def _plain(value: Any) -> Any:
    """Normalize SDK response models (including v10 ``to_dict`` models)."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                mapped = method()
            except (TypeError, ValueError):
                continue
            if isinstance(mapped, dict):
                return _plain(mapped)
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return value


def _read(value: Any, key: str, default: Any = None) -> Any:
    normalized = _plain(value)
    if isinstance(normalized, dict):
        return normalized.get(key, default)
    return getattr(value, key, default)


def _named_items(value: Any) -> dict[str, Any]:
    normalized = _plain(value)
    return normalized if isinstance(normalized, dict) else {}


def validate_index(index: Any) -> list[str]:
    """Return configuration mismatches; an empty list means compatible."""
    problems: list[str] = []
    schema = _read(index, "schema")
    fields = _named_items(_read(schema, "fields"))
    vector = fields.get(VECTOR_FIELD)
    if vector is None:
        problems.append(f"missing vector field '{VECTOR_FIELD}'")
    else:
        if str(_read(vector, "type", "")).lower() != "dense_vector":
            problems.append(f"{VECTOR_FIELD} type must be dense_vector")
        if _read(vector, "dimension") != DIMENSION:
            problems.append(f"{VECTOR_FIELD} dimension must be {DIMENSION}")
        if str(_read(vector, "metric", "")).lower() != METRIC:
            problems.append(f"{VECTOR_FIELD} metric must be {METRIC}")

    deployment = _read(index, "deployment")
    for key, expected in (("deployment_type", DEPLOYMENT_TYPE), ("cloud", CLOUD), ("region", REGION)):
        actual = _read(deployment, key)
        if str(actual).lower() != expected:
            problems.append(f"deployment {key} must be {expected} (found {actual!r})")

    read_capacity = _read(index, "read_capacity")
    if read_capacity is None:
        problems.append("read capacity could not be verified as OnDemand")
    else:
        capacity_type = type(read_capacity).__name__.lower()
        mode = _read(read_capacity, "mode")
        if mode is not None:
            is_on_demand = str(mode).replace("_", "").lower() == "ondemand"
        else:
            is_on_demand = "ondemand" in capacity_type
        if not is_on_demand:
            problems.append("read capacity must be OnDemand")

    status = _read(index, "status")
    ready = _read(status, "ready")
    state = str(_read(status, "state", "")).lower()
    if ready is not True and state != "ready":
        problems.append(f"index is not Ready (state={state or 'unknown'})")
    return problems


def _describe_when_ready(indexes: Any) -> Any:
    """Fetch the authoritative description, polling transient create states boundedly."""
    deadline = time.monotonic() + PROVISIONING_TIMEOUT_SECONDS
    while True:
        description = indexes.describe(INDEX_NAME)
        status = _read(description, "status")
        if _read(status, "ready") is True or str(_read(status, "state", "")).lower() == "ready":
            return description
        if time.monotonic() >= deadline:
            state = _read(status, "state", "unknown")
            raise SetupError(f"Index '{INDEX_NAME}' is still provisioning (state={state}); retry the check later.")
        time.sleep(min(PROVISIONING_POLL_SECONDS, max(0, deadline - time.monotonic())))


def _make_client(api_key: str) -> Any:
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise SetupError("Pinecone SDK is missing. Install project requirements first.") from exc
    return Pinecone(api_key=api_key)


def run(*, create: bool, api_key: str | None = None, client: Any = None) -> tuple[bool, str]:
    """Check the target, or create it if explicitly requested and absent."""
    if api_key is None:
        api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or not api_key.strip():
        raise SetupError("PINECONE_API_KEY is not set. Set it in the environment or local .env file.")
    try:
        pc = client if client is not None else _make_client(api_key)
        existing = any(_read(item, "name") == INDEX_NAME for item in pc.indexes.list())
        created = False
        if not existing:
            if not create:
                return False, f"Index '{INDEX_NAME}' was not found. Run with --create to create it."
            pc.indexes.create(
                name=INDEX_NAME,
                schema={"fields": {VECTOR_FIELD: {"type": "dense_vector", "dimension": DIMENSION, "metric": METRIC}}},
                deployment={"deployment_type": DEPLOYMENT_TYPE, "cloud": CLOUD, "region": REGION},
                # Pinecone's documented default when omitted is on-demand read capacity.
                timeout=120,
            )
            created = True
        # The create response is not authoritative; always read the populated description.
        description = _describe_when_ready(pc.indexes) if created else pc.indexes.describe(INDEX_NAME)
    except SetupError:
        raise
    except Exception as exc:
        # Avoid echoing SDK request details, which may contain sensitive headers.
        raise SetupError(f"Pinecone API operation failed ({type(exc).__name__}). Check credentials, permissions, and connectivity.") from None

    mismatches = validate_index(description)
    if mismatches:
        raise SetupError(f"Index '{INDEX_NAME}' has incompatible settings: " + "; ".join(mismatches))
    action = "created and validated" if created else "already exists and is compatible"
    return True, f"Index '{INDEX_NAME}' {action}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check only (default).")
    mode.add_argument("--create", action="store_true", help="Create the target only if it is absent.")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    try:
        _ok, message = run(create=args.create)
    except SetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
