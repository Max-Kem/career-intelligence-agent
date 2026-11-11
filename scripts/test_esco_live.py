"""Run one live ESCO skill lookup for the term Python."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_agent.esco import ESCOClient  # noqa: E402


def main() -> int:
    client = ESCOClient()
    try:
        result = client.lookup_skill("Python")
        print(f"Lookup status: {result.status}")
        if result.error:
            print(f"Error: {result.error}")
        if result.concepts:
            print("ESCO concepts:")
            for concept in result.concepts:
                print(f"- Label: {concept.label or '(no label)'}")
                print(f"  URI: {concept.uri or '(no URI)'}")
        else:
            print("ESCO concepts: none returned")
        return 0 if result.status != "unavailable" else 1
    except Exception as exc:
        print(f"Lookup status: error\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
