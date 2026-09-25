"""Evaluate the legacy rule-based chat selector against varied prompts.

Run from any directory:

    PYTHONPATH=apps/api/src python scripts/chat_routing_baseline.py

This is diagnostic, not a router. The expected labels are human-reviewed in
``apps/api/tests/chat_evidence_cases.py``. A source label describes what a
good answer needs, not what any particular model happens to say.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/api/src"))
sys.path.insert(0, str(ROOT / "apps/api/tests"))

from chat_evidence_cases import EVIDENCE_CASES  # noqa: E402
from chat_evidence_holdout import HOLDOUT_CASES  # noqa: E402
from waqil_api.prompt_scope import user_instruction  # noqa: E402
from waqil_api.web_research import (  # noqa: E402
    is_explicit_web_request,
    is_implicit_web_request,
)


def legacy_sources(prompt: str, scope: str) -> str:
    """Mirror the old `_retrieve` branch with corpus and web available."""
    if scope == "web":
        return "web"
    if scope == "notion":
        return "private"
    return (
        "web"
        if is_explicit_web_request(prompt) or is_implicit_web_request(prompt)
        else "private"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    parser.add_argument("--holdout", action="store_true", help="Use unseen paraphrases")
    args = parser.parse_args()
    cases = HOLDOUT_CASES if args.holdout else EVIDENCE_CASES

    rows = [
        {
            "id": case.id,
            "expected": case.sources,
            "actual": legacy_sources(case.prompt, case.scope),
            "scope": case.scope,
            "prompt": case.prompt,
        }
        for case in cases
    ]
    wrong = [row for row in rows if row["expected"] != row["actual"]]
    # The old web branch passes the user's instruction as the search query
    # whenever it has no direct URL to open. Count explicitly labelled private
    # phrases it would send in those cases.
    privacy_flags = [
        case.id
        for case in cases
        if legacy_sources(case.prompt, case.scope) == "web"
        and "https://" not in user_instruction(case.prompt)
        and any(
            phrase.casefold() in user_instruction(case.prompt).casefold()
            for phrase in case.private_query_terms
        )
    ]
    matrix = Counter((row["expected"], row["actual"]) for row in rows)
    report = {
        "total": len(rows),
        "wrong": len(wrong),
        "accuracy": round(1 - len(wrong) / len(rows), 4),
        "private_query_flags": privacy_flags,
        "confusion": [
            {"expected": expected, "actual": actual, "count": count}
            for (expected, actual), count in sorted(matrix.items())
        ],
        "failures": wrong,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Legacy evidence rules: {len(rows) - len(wrong)}/{len(rows)} correct")
        if privacy_flags:
            print(f"Private phrases in raw web query: {', '.join(privacy_flags)}")
        for row in wrong:
            print(f"{row['id']}: {row['actual']} -> {row['expected']} | {row['prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
