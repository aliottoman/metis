"""Verify Metis's configured Ollama planner and code generator."""

from __future__ import annotations

import asyncio
import json

from waqil_api.config import Settings
from waqil_api.contracts import PlanningRequestV1
from waqil_api.diagram_source import validate_diagram_source
from waqil_api.model_provider import OllamaModelProvider


async def main() -> None:
    settings = Settings(context_window=8192, ollama_keep_alive="0")
    provider = OllamaModelProvider(settings)
    health = await provider.health()
    if not health.get("reachable"):
        raise RuntimeError(f"Ollama is not reachable: {health.get('error', 'unknown error')}")
    availability = health.get("configured_available", {})
    missing = [
        role
        for role in ("planner", "coder")
        if not availability.get(role)
    ]
    if missing:
        configured = health.get("configured", {})
        raise RuntimeError(
            "configured Ollama models are unavailable: "
            + ", ".join(f"{role}={configured.get(role)!r}" for role in missing)
        )
    plan = await provider.plan(
        PlanningRequestV1(
            run_id="smoke_run",
            conversation_id="smoke_conversation",
            prompt="Build a reference architecture diagram from this README.",
        )
    )
    if plan.route != "tool_factory" or plan.tool_slug != "reference-architecture-generator":
        raise RuntimeError(f"planner routed the architecture request incorrectly: {plan}")
    print("Planner contract:")
    print(plan.model_dump_json(indent=2))

    spec = await provider.architecture_spec(
        "Build a reference architecture diagram from this README.",
        (
            "# Orders API\n"
            "A browser calls a Python API over HTTPS. The API reads and writes "
            "PostgreSQL over SQL inside a private application boundary."
        ),
        approved_context={
            "memories": ["Use explicit protocol labels."],
            "conversation_summary": "The user prefers left-to-right diagrams.",
            "recent_messages": [],
        },
    )
    print("\nArchitecture contract:")
    print(spec.model_dump_json(indent=2))

    code = await provider.diagram_code(spec)
    evidence = validate_diagram_source(code.diagram_code, spec, ["svg", "png"])
    print(f"\nNorth diagram code ({settings.coder_model}):")
    print(code.diagram_code[:2000])
    print("\nHost validation:")
    print(evidence)
    print("\nSmoke result:")
    print(
        json.dumps(
            {
                "status": "passed",
                "planner_model": settings.planner_model,
                "coder_model": settings.coder_model,
                "route": plan.route,
                "tool_slug": plan.tool_slug,
                "source_bytes": evidence["source_bytes"],
                "host_policy": evidence["policy"],
                "keep_alive": settings.ollama_keep_alive,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
