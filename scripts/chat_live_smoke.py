"""Run two real Metis chat turns in an isolated, disposable data directory.

Only synthetic questions are sent to the configured ClinePass model. The
existing app database and preference files are never opened or modified. A
report with route events, retrieved URLs, answer text, and timing is written
outside the app checkout for review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/api/src"))

from waqil_api.config import Settings  # noqa: E402
from waqil_api.main import create_app  # noqa: E402


QUESTIONS = (
    (
        "cline_recent_harness",
        "What new features did cline release recently for its harness that would "
        "be essential and ahuge improvement for a local AI personal app?",
    ),
    ("stable_list_comprehension", "What is a Python list comprehension?"),
)


class _LoggedWeb:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return self.delegate.available()

    async def retrieve(self, prompt: str, **kwargs: Any):
        started = time.monotonic()
        result = await self.delegate.retrieve(prompt, **kwargs)
        self.calls.append(
            {
                "queries": kwargs.get("queries"),
                "focus_terms": kwargs.get("focus_terms"),
                "include_prompt_urls": kwargs.get("include_prompt_urls", True),
                "latency_seconds": round(time.monotonic() - started, 3),
                "sources": [
                    {
                        "title": item.source_label,
                        "url": item.source_url,
                        "excerpt": item.text[:4000],
                    }
                    for item in result
                ],
            }
        )
        return result


def _events(stream: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in stream.splitlines()
        if line.startswith("data: ")
    ]


def _event_payloads(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [item.get("payload", {}) for item in events if item.get("type") == name]


def _timestamp(item: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00"))


def _run_turn(
    client: TestClient, key: str, question: str, *, timeout: float
) -> dict[str, Any]:
    runtime = client.app.state.runtime
    web = _LoggedWeb(runtime.control_plane.web)
    runtime.control_plane.web = web
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]
    started = time.monotonic()
    accepted = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": question, "attachment_ids": [], "knowledge_scope": "auto"},
    ).json()
    run_id = accepted["run_id"]
    deadline = time.monotonic() + timeout
    run: dict[str, Any] = {}
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in {
            "completed",
            "failed",
            "awaiting_approval",
            "awaiting_input",
        }:
            break
        time.sleep(0.1)
    total_seconds = round(time.monotonic() - started, 3)
    stream = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
    events = _events(stream)
    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages").json()
    assistant = [item["content"] for item in messages if item["role"] == "assistant"]
    first_delta = next(
        (item for item in events if item.get("type") == "message.delta"), None
    )
    first_event = events[0] if events else None
    first_token_seconds = (
        round((_timestamp(first_delta) - _timestamp(first_event)).total_seconds(), 3)
        if first_delta is not None and first_event is not None
        else None
    )
    answer = assistant[-1] if assistant else ""
    milestone_types = {
        "run.started",
        "evidence.planned",
        "evidence.reviewed",
        "context.retrieved",
        "plan.created",
        "policy.evaluated",
        "message.delta",
        "model.response",
        "answer.grounding_reviewed",
        "run.completed",
    }
    milestones: list[dict[str, Any]] = []
    seen_milestones: set[str] = set()
    if first_event is not None:
        for item in events:
            event_type = str(item.get("type", ""))
            if event_type not in milestone_types or event_type in seen_milestones:
                continue
            seen_milestones.add(event_type)
            milestones.append(
                {
                    "event": event_type,
                    "elapsed_seconds": round(
                        (_timestamp(item) - _timestamp(first_event)).total_seconds(), 3
                    ),
                }
            )
    return {
        "id": key,
        "question": question,
        "run_status": run.get("status", "timeout"),
        "error_type": run.get("error_type"),
        "total_seconds": total_seconds,
        "first_answer_token_seconds": first_token_seconds,
        "evidence_plan": _event_payloads(events, "evidence.planned"),
        "evidence_plan_errors": _event_payloads(events, "evidence.plan_failed"),
        "evidence_review": _event_payloads(events, "evidence.reviewed"),
        "context_retrieved": _event_payloads(events, "context.retrieved"),
        "web_calls": web.calls,
        "model_calls": _event_payloads(events, "model.response"),
        "run_failures": _event_payloads(events, "run.failed"),
        "event_types": [item.get("type", "") for item in events],
        "milestones": milestones,
        "answer": answer,
        "answer_links": sorted(set(re.findall(r"https?://[^\s)]+", answer))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="/Users/aliottoman/Developer/metis/.env")
    parser.add_argument("--output", default="/private/tmp/metis-chat-live-smoke.json")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--only", nargs="*", default=[], help="Run only named question IDs"
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    env_file = Path(args.env_file).expanduser()
    with tempfile.TemporaryDirectory(
        prefix="metis-chat-smoke-", dir="/private/tmp"
    ) as temp:
        scratch = Path(temp)
        settings = Settings(
            _env_file=env_file,
            data_dir=scratch / "data",
            repo_root=ROOT,
            asset_roots=[scratch / "projects"],
            project_coding_engine="legacy",
            reference_runner_mode="deterministic",
            allow_test_backends=True,
            model_call_timeout_seconds=150,
            notion_token="",
        )
        with TestClient(create_app(settings)) as client:
            runtime = client.app.state.runtime
            if not runtime.model_preference.cline_available:
                print("Configured ClinePass is unavailable; no chat turns sent.")
                return 2
            runtime.model_preference.save("split", None, provider="cline")
            results: list[dict[str, Any]] = []
            for key, question in QUESTIONS:
                if args.only and key not in args.only:
                    continue
                print(f"Running {key}...", flush=True)
                result = _run_turn(client, key, question, timeout=args.timeout)
                results.append(result)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(results, indent=2) + "\n")
                print(
                    f"{key}: {result['run_status']} in {result['total_seconds']}s; "
                    f"first answer token {result['first_answer_token_seconds']}s; "
                    f"{len(result['answer_links'])} answer link(s)",
                    flush=True,
                )
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
