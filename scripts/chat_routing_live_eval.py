"""Measure a real configured ClinePass evidence planner on varied chat prompts.

The labelled corpus lives in ``apps/api/tests/chat_evidence_cases.py``. This
script calls only the source/action planner; it does not browse, generate an
answer, modify user data, or run a project build. All prompts are synthetic.
It prints no configuration values or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/api/src"))
sys.path.insert(0, str(ROOT / "apps/api/tests"))

from chat_evidence_cases import EVIDENCE_CASES, PROJECT_CASES  # noqa: E402
from chat_evidence_holdout import HOLDOUT_CASES  # noqa: E402
from waqil_api.config import Settings  # noqa: E402
from waqil_api.evidence_routing import plan_evidence, scope_plan  # noqa: E402
from waqil_api.model_provider import ClineModelProvider  # noqa: E402


def _source_label(sources: list[str]) -> str:
    names = set(sources)
    if names == {"private", "web"}:
        return "both"
    if names == {"private"}:
        return "private"
    if names == {"web"}:
        return "web"
    if not names:
        return "none"
    return "unexpected"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = [row for row in rows if row["kind"] == "evidence"]
    answered = [row for row in evidence if row.get("actual_sources") is not None]
    correct = [row for row in answered if row["actual_sources"] == row["expected_sources"]]
    durations = sorted(row["latency_seconds"] for row in answered)
    privacy = [row["id"] for row in answered if row.get("private_query_leak")]
    actions = [row for row in rows if row["kind"] == "project" and row.get("actual_action")]
    action_correct = [row for row in actions if row["actual_action"] == row["expected_action"]]
    return {
        "evidence_total": len(evidence),
        "evidence_completed": len(answered),
        "evidence_correct": len(correct),
        "evidence_accuracy": round(len(correct) / len(answered), 4) if answered else None,
        "all_case_accuracy": round(len(correct) / len(evidence), 4) if evidence else None,
        "median_latency_seconds": round(statistics.median(durations), 2) if durations else None,
        "p95_latency_seconds": (
            round(durations[min(len(durations) - 1, int(len(durations) * 0.95))], 2)
            if durations
            else None
        ),
        "query_privacy_flags": privacy,
        "project_action_completed": len(actions),
        "project_action_correct": len(action_correct),
    }


async def _run(args: argparse.Namespace) -> int:
    env_file = Path(args.env_file).expanduser()
    settings = Settings(_env_file=env_file)
    provider = ClineModelProvider(settings)
    if not provider.available:
        print("Configured ClinePass planner is unavailable; no evaluation calls made.")
        return 2

    evidence_cases = HOLDOUT_CASES if args.holdout else EVIDENCE_CASES
    cases = [
        ("evidence", case)
        for case in evidence_cases
        if not args.ids or case.id in args.ids
    ]
    if args.include_project:
        cases.extend(
            ("project", case)
            for case in PROJECT_CASES
            if not args.ids or case.id in args.ids
        )
    semaphore = asyncio.Semaphore(args.concurrency)
    completed: list[dict[str, Any]] = []
    save_lock = asyncio.Lock()
    output = Path(args.output).expanduser()

    async def evaluate(kind: str, case: Any) -> None:
        async with semaphore:
            started = time.monotonic()
            row: dict[str, Any] = {
                "kind": kind,
                "id": case.id,
                "prompt": case.prompt,
                "expected_sources": case.sources if kind == "evidence" else None,
                "expected_action": "answer" if kind == "evidence" else case.expected_action,
            }
            try:
                explicit = scope_plan(case.scope) if kind == "evidence" else None
                if explicit is not None:
                    plan, method = explicit, "explicit_scope"
                else:
                    plan, method = await asyncio.wait_for(
                        plan_evidence(
                            provider,
                            prompt=case.prompt,
                            recent_messages=[],
                            conversation_summary="",
                            has_attachment=False,
                            has_project=bool(kind == "project" and case.selected_project),
                            has_customer=False,
                            model_aliases={"_provider": "cline"},
                        ),
                        timeout=args.timeout,
                    )
                query_text = " ".join(plan.public_queries).casefold()
                forbidden = case.private_query_terms if kind == "evidence" else ()
                row.update(
                    actual_sources=_source_label(plan.sources),
                    actual_action=plan.action,
                    method=method,
                    public_queries=plan.public_queries,
                    focus_terms=plan.focus_terms,
                    needs_verification=plan.needs_verification,
                    private_query_leak=any(term.casefold() in query_text for term in forbidden),
                )
            except Exception as error:  # noqa: BLE001 - keep every failed case visible
                row["error_type"] = type(error).__name__
            row["latency_seconds"] = round(time.monotonic() - started, 3)
            async with save_lock:
                completed.append(row)
                completed.sort(key=lambda item: item["id"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps({"summary": _summary(completed), "cases": completed}, indent=2)
                    + "\n"
                )
            actual = row.get("actual_sources", row.get("error_type", "error"))
            print(f"{case.id}: {actual} ({row['latency_seconds']}s)", flush=True)

    try:
        await asyncio.gather(*(evaluate(kind, case) for kind, case in cases))
    finally:
        await provider.close()
    print(json.dumps(_summary(completed), indent=2))
    print(f"Report: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="Local Metis configuration file (its contents are never printed)",
    )
    parser.add_argument("--output", default="/private/tmp/metis-chat-routing-live.json")
    parser.add_argument("--concurrency", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--include-project", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Use unseen paraphrases")
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
