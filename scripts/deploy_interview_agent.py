"""Deploy the Chiron interview agent to ElevenLabs from the committed spec.

Translates docs/interviews-workflow.json + docs/interviews-agent-prompt.txt
into the agent's live configuration: base prompt, fixed opening line, the
blocking submit_interview_evaluation client tool, the workflow graph, and the
duration cap. Fetch-merge-patch, so settings this spec does not own (LLM,
voice, language) are preserved exactly. The API key is read from the
environment and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://api.elevenlabs.io"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SPEC = REPO_ROOT / "docs" / "interviews-workflow.json"
PROMPT_FILE = REPO_ROOT / "docs" / "interviews-agent-prompt.txt"
EXPORT_FILE = REPO_ROOT / "docs" / "interviews-agent-export.json"

TOOL_NAME = "submit_interview_evaluation"

# The router's fixed opening line. Spoken as the agent's first message rather
# than by a node, so it is deterministic and costs no model call. Must match
# the router prompt's "Say exactly" text in docs/interviews-workflow.json.
OPENING_LINE = (
    "We'll run five questions. Expect me to press on your answers. "
    "I won't coach you during the interview. I'll evaluate you when it is over."
)

# Five answers, up to two follow-up probes each, plus a debrief; the
# platform's 600-second default cuts off mid-answer. 60 minutes, per
# docs/interviews-elevenlabs-agent.md.
MAX_DURATION_SECONDS = 3_600

# The tool blocks while the browser round-trips through the Metis API; that
# takes a second or two, and 60 leaves room for a slow first evaluation.
TOOL_TIMEOUT_SECONDS = 60

IMPROVEMENT_FIELDS = ("what_happened", "evidence", "why_it_hurt", "better_approach")

# What the model submits: five criterion scores and the qualitative material.
# Deliberately no overall score anywhere in this schema — Metis calculates it
# and returns it in the tool response.
EVALUATION_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "required": [
        "specific_evidence",
        "role_depth",
        "relevance",
        "structure",
        "communication",
        "verdict",
        "strongest_answer_quote",
        "strongest_answer_reason",
        "improvements",
        "drill",
        "completed_question_count",
        "incomplete",
    ],
    "properties": {
        "specific_evidence": {
            "type": "integer",
            "description": "Specific evidence, 1-10. 5 is mixed and ordinary, 7 solid, 9 rare.",
        },
        "role_depth": {
            "type": "integer",
            "description": "Role-specific judgment or technical depth, 1-10.",
        },
        "relevance": {"type": "integer", "description": "Relevance, 1-10."},
        "structure": {"type": "integer", "description": "Structure, 1-10."},
        "communication": {"type": "integer", "description": "Communication, 1-10."},
        "verdict": {
            "type": "string",
            "description": "One honest sentence. Do not include an overall score.",
        },
        "strongest_answer_quote": {
            "type": "string",
            "description": "The candidate's exact words from the transcript. Never invented.",
        },
        "strongest_answer_reason": {
            "type": "string",
            "description": "Why that answer worked.",
        },
        "improvements": {
            "type": "array",
            "description": "Exactly three highest-priority problems, in order.",
            "items": {
                "type": "object",
                "required": list(IMPROVEMENT_FIELDS),
                "properties": {
                    "what_happened": {"type": "string", "description": "What the candidate did."},
                    "evidence": {"type": "string", "description": "Evidence from the transcript."},
                    "why_it_hurt": {"type": "string", "description": "Why it hurt the performance."},
                    "better_approach": {
                        "type": "string",
                        "description": "A better approach — not a script to memorise.",
                    },
                },
            },
        },
        "drill": {
            "type": "string",
            "description": "One concrete ten-minute drill, doable alone.",
        },
        "completed_question_count": {
            "type": "integer",
            "description": "How many of the five questions were answered, 0-5.",
        },
        "incomplete": {
            "type": "boolean",
            "description": "True when the interview ended before question five.",
        },
    },
}


# Reads a required variable from the environment, else from an env file.
def require_env(name: str, env_file: Path | None) -> str:
    value = os.environ.get(name, "").strip()
    if not value and env_file is not None and env_file.exists():
        for line in env_file.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{name}="):
                value = stripped.split("=", 1)[1].strip().strip("'\"")
                break
    if not value:
        sys.exit(f"Set {name} in the environment or in {env_file}.")
    return value


# Loads the committed workflow spec and the base prompt.
def load_spec() -> tuple[dict[str, Any], str]:
    spec = json.loads(WORKFLOW_SPEC.read_text())
    prompt = PROMPT_FILE.read_text().strip()
    if not prompt.startswith("You are Chiron"):
        sys.exit(f"{PROMPT_FILE} does not look like the Chiron base prompt.")
    return spec, prompt


# Lays the graph out for the dashboard canvas: one column per route.
def _position(node_id: str, spec: dict[str, Any]) -> dict[str, float]:
    columns = {"hr_recruiter": -520.0, "hiring_manager": 0.0, "technical": 520.0}
    for route, node_ids in spec["routes"].items():
        if node_id in node_ids:
            return {"x": columns[route], "y": 140.0 + 190.0 * node_ids.index(node_id)}
    fixed = {
        "start_node": (0.0, -120.0),
        "evaluation": (0.0, 1_160.0),
        "wrap_up": (0.0, 1_360.0),
        "end_node": (0.0, 1_560.0),
    }
    x, y = fixed[node_id]
    return {"x": x, "y": y}


# Translates the abstract spec graph into ElevenLabs workflow nodes and edges.
def build_workflow(spec: dict[str, Any]) -> dict[str, Any]:
    # The spec's router becomes the start node (its opening line is the
    # agent's first message; its never-re-ask rules live in the base prompt).
    # The spec's end node becomes a wrap-up subagent, because ElevenLabs'
    # `end` node hangs up — and the candidate may have a brief question about
    # the verdict before the call closes.
    node_ids = {"router": "start_node", "end": "wrap_up"}

    edges: dict[str, Any] = {}
    outgoing: dict[str, list[str]] = {}

    def add_edge(source: str, target: str, condition: str, label: str) -> None:
        edge_id = f"{source}__{target}"
        edges[edge_id] = {
            "source": source,
            "target": target,
            "forward_condition": {"type": "llm", "label": label, "condition": condition},
            "backward_condition": None,
        }
        outgoing.setdefault(source, []).append(edge_id)

    for edge in spec["edges"]:
        source = node_ids.get(edge["from"], edge["from"])
        target = node_ids.get(edge["to"], edge["to"])
        label = (
            "Stop early"
            if target == "evaluation" and not edge["from"].endswith("q5")
            else "Next"
        )
        add_edge(source, target, edge["condition"], label)
    # The wrap-up closes the call once the candidate is done.
    add_edge(
        "wrap_up",
        "end_node",
        "The candidate has said they are done or has nothing further to ask.",
        "Done",
    )

    # Early-stop edges outrank the advance edge: a candidate who confirmed
    # they want to stop must reach the evaluation, not question N+1.
    def ordered(node_id: str) -> list[str]:
        edge_ids = outgoing.get(node_id, [])
        return sorted(edge_ids, key=lambda eid: 0 if edges[eid]["target"] == "evaluation" else 1)

    nodes: dict[str, Any] = {
        "start_node": {
            "type": "start",
            "position": _position("start_node", spec),
            "edge_order": outgoing.get("start_node", []),
        }
    }
    for node in spec["nodes"]:
        if node["kind"] == "router":
            continue
        node_id = node_ids.get(node["id"], node["id"])
        nodes[node_id] = {
            "type": "override_agent",
            "label": node.get("focus") or node_id,
            "position": _position(node_id, spec),
            "additional_prompt": node["prompt"],
            "additional_knowledge_base": [],
            "additional_tool_ids": [],
            "edge_order": ordered(node_id),
        }
    nodes["end_node"] = {"type": "end", "position": _position("end_node", spec)}
    return {"nodes": nodes, "edges": edges, "prevent_subagent_loops": False}


# Finds or creates the blocking client tool; returns its id.
def ensure_tool(client: httpx.Client) -> str:
    tool_config = {
        "type": "client",
        "name": TOOL_NAME,
        "description": (
            "Submit the completed interview evaluation. Call exactly once, after "
            "the interview ends. Blocks until Metis returns the calculated "
            "overall score and recommendation; speak those returned values in "
            "the debrief."
        ),
        "expects_response": True,
        "response_timeout_secs": TOOL_TIMEOUT_SECONDS,
        "parameters": EVALUATION_PARAMETERS,
    }
    listing = client.get("/v1/convai/tools")
    listing.raise_for_status()
    for tool in listing.json().get("tools", []):
        if tool.get("tool_config", {}).get("name") == TOOL_NAME:
            tool_id = tool["id"]
            updated = client.patch(
                f"/v1/convai/tools/{tool_id}", json={"tool_config": tool_config}
            )
            updated.raise_for_status()
            print(f"tool: updated {TOOL_NAME} ({tool_id})")
            return tool_id
    created = client.post("/v1/convai/tools", json={"tool_config": tool_config})
    created.raise_for_status()
    tool_id = created.json()["id"]
    print(f"tool: created {TOOL_NAME} ({tool_id})")
    return tool_id


# Fetches the agent, merges this spec's fields in, and patches it back.
def deploy(client: httpx.Client, agent_id: str, dry_run: bool) -> None:
    spec, prompt = load_spec()
    workflow = build_workflow(spec)

    current = client.get(f"/v1/convai/agents/{agent_id}")
    current.raise_for_status()
    fetched = current.json()
    config = fetched["conversation_config"]

    tool_id = "dry-run" if dry_run else ensure_tool(client)
    agent = config.setdefault("agent", {})
    agent["first_message"] = OPENING_LINE
    agent_prompt = agent.setdefault("prompt", {})
    agent_prompt["prompt"] = prompt
    agent_prompt["tool_ids"] = [tool_id]
    # The deprecated inline-tools field conflicts with tool_ids when present.
    agent_prompt.pop("tools", None)
    config.setdefault("conversation", {})["max_duration_seconds"] = MAX_DURATION_SECONDS

    # Delivery analysis reads the call recording back after the interview;
    # recording defaults on, but a dashboard toggle would silently starve it.
    platform = fetched.get("platform_settings") or {}
    platform.setdefault("privacy", {})["record_voice"] = True

    if dry_run:
        print(json.dumps(workflow, indent=2))
        print(f"dry run: {len(workflow['nodes'])} nodes, {len(workflow['edges'])} edges")
        return

    # The workflow is a top-level sibling of conversation_config on the agent
    # resource (the docs' conversation_config.workflow placement is accepted
    # and silently ignored — verified the hard way).
    patched = client.patch(
        f"/v1/convai/agents/{agent_id}",
        json={
            "conversation_config": config,
            "workflow": workflow,
            "platform_settings": platform,
        },
    )
    if patched.status_code >= 400:
        sys.exit(f"ElevenLabs refused the update: HTTP {patched.status_code}: {patched.text[:2000]}")

    verify(client, agent_id, spec, tool_id)


# Reads the agent back and checks the deployment landed whole.
def verify(client: httpx.Client, agent_id: str, spec: dict[str, Any], tool_id: str) -> None:
    fetched = client.get(f"/v1/convai/agents/{agent_id}")
    fetched.raise_for_status()
    data = fetched.json()
    config = data["conversation_config"]
    nodes = (data.get("workflow") or {}).get("nodes", {})
    problems: list[str] = []
    # start_node + 15 question nodes + evaluation + wrap_up + end_node.
    if len(nodes) != 19:
        problems.append(f"expected 19 nodes, agent has {len(nodes)}")
    for route, node_ids in spec["routes"].items():
        missing = [node_id for node_id in node_ids if node_id not in nodes]
        if missing:
            problems.append(f"route {route} is missing nodes: {missing}")
    if not config["agent"]["prompt"]["prompt"].startswith("You are Chiron"):
        problems.append("the base prompt did not land")
    if config["agent"]["first_message"] != OPENING_LINE:
        problems.append("the opening line did not land")
    if tool_id not in config["agent"]["prompt"].get("tool_ids", []):
        problems.append("the evaluation tool is not attached")
    if config["conversation"]["max_duration_seconds"] != MAX_DURATION_SECONDS:
        problems.append("the duration cap did not land")
    privacy = (data.get("platform_settings") or {}).get("privacy") or {}
    if privacy.get("record_voice") is not True:
        # Without a recording the delivery analysis silently lands
        # 'unavailable' on every interview — catch that here, not in a month.
        problems.append("record_voice is not pinned on")
    if problems:
        sys.exit("deployed, but verification failed: " + "; ".join(problems))

    # The export is committed to a public repo: account details (creator name
    # and email) and telephony bindings are not configuration and stay out.
    for private_key in ("access_info", "phone_numbers", "whatsapp_accounts"):
        data.pop(private_key, None)
    EXPORT_FILE.write_text(json.dumps(data, indent=2) + "\n")
    print(f"agent: {data.get('name')} ({agent_id})")
    print(f"workflow: {len(nodes)} nodes, {len((data.get('workflow') or {}).get('edges', {}))} edges")
    print(f"export: {EXPORT_FILE.relative_to(REPO_ROOT)}")
    print(
        "verified: prompt, opening line, blocking tool, workflow, duration cap, recording"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID", "").strip(),
        help="ElevenLabs agent id (default: WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the workflow, change nothing"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="env file to read WAQIL_ELEVENLABS_API_KEY from when it is not exported",
    )
    arguments = parser.parse_args()
    if not arguments.agent_id:
        sys.exit("Pass --agent-id or set WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID.")
    api_key = require_env("WAQIL_ELEVENLABS_API_KEY", arguments.env_file)
    with httpx.Client(
        base_url=BASE_URL, headers={"xi-api-key": api_key}, timeout=30.0
    ) as client:
        deploy(client, arguments.agent_id, arguments.dry_run)


if __name__ == "__main__":
    main()
