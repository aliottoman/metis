#!/usr/bin/env python3
"""Run one production-shaped project-build capability evaluation.

Without ``--live`` this prints the exact scenario and model route without
calling a provider.  A live run uses only a disposable project/data directory,
allows a bounded number of repair follow-ups, and executes generated code only
inside Metis's reviewed networkless project-verification container.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.contracts import RoleChainEntryV1
from waqil_api.main import create_app
from waqil_api.model_preference import CLINEPASS_MODELS, ModelPreferenceStore
from waqil_api.project_capability_eval import (
    acceptance_repair_prompt,
    event_dicts,
    project_file_hashes,
    project_tree_digest,
    project_asset_id,
    read_timeline,
    release_gate_passes,
    repair_attempt_evidence,
    run_meridian_acceptance_sync,
    score_evaluation,
    summarize_repair_continuation,
    summarize_run,
    validate_pending_overlay,
)
from waqil_api.project_capability_scenarios import (
    QUALIFICATION_GATE_NAMES,
    SCENARIOS,
    CapabilityScenario,
    get_scenario,
    qualify_scenario,
    run_scenario_acceptance_sync,
)


TERMINAL_OR_PAUSED = {
    "awaiting_approval",
    "awaiting_input",
    "completed",
    "failed",
    "cancelled",
}


def _bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return parse


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _request(client: TestClient, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 300:
        raise RuntimeError(
            f"{method} {path} returned {response.status_code}: {response.text[:800]}"
        )
    if response.status_code == 204:
        return None
    return response.json()


def _wait_for_run(
    client: TestClient,
    run_id: str,
    timeout_seconds: float,
    wanted: set[str] | None = None,
) -> dict[str, Any]:
    wanted = wanted or TERMINAL_OR_PAUSED
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        run = _request(client, "GET", f"/api/v1/runs/{run_id}")
        if run["status"] in wanted:
            return run
        time.sleep(0.5)
    try:
        client.post(f"/api/v1/runs/{run_id}/cancel")
    except Exception:  # noqa: BLE001 - preserve the timeout as the primary result
        pass
    return {
        **_request(client, "GET", f"/api/v1/runs/{run_id}"),
        "status": "timed_out",
        "last_error": f"evaluation turn exceeded {timeout_seconds:g} seconds",
    }


def _pending_approval(client: TestClient, run_id: str) -> dict[str, Any] | None:
    pending = _request(
        client,
        "GET",
        "/api/v1/runs?status=awaiting_approval&limit=200",
    )
    return next(
        (
            dict(item["approval"])
            for item in pending
            if item.get("run", {}).get("id") == run_id and item.get("approval")
        ),
        None,
    )


async def _read_pending_project_overlay(
    runtime: Any,
    conversation_id: str,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Snapshot exact pending bytes from the run's in-process checkpoint."""

    control_plane = runtime.control_plane
    checkpointer = runtime.checkpointer
    if control_plane is None or checkpointer is None:
        raise RuntimeError("project checkpoint runtime is unavailable")
    checkpoint = await checkpointer.aget_tuple(
        control_plane._config(conversation_id, run_id)  # noqa: SLF001 - evaluator introspection
    )
    values = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
    raw_staged = values.get("project_staged") or {}
    if not isinstance(raw_staged, Mapping):
        raise RuntimeError("pending project checkpoint has a malformed staged overlay")
    staged: dict[str, dict[str, Any]] = {}
    for raw_path, raw_entry in raw_staged.items():
        if not isinstance(raw_path, str) or not isinstance(raw_entry, Mapping):
            raise RuntimeError(
                "pending project checkpoint has a malformed staged entry"
            )
        content = raw_entry.get("content")
        if not isinstance(content, str):
            raise RuntimeError(
                f"pending project checkpoint content is not text: {raw_path}"
            )
        staged[raw_path] = dict(raw_entry)
    return staged


def _pending_project_overlay(
    client: TestClient,
    conversation_id: str,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Bridge the synchronous CLI to the TestClient runtime's event loop."""

    portal = client.portal
    if portal is None:
        raise RuntimeError("evaluation runtime portal is unavailable")
    runtime = cast(Any, client.app).state.runtime
    return portal.call(
        _read_pending_project_overlay,
        runtime,
        conversation_id,
        run_id,
    )


def _run_acceptance(
    scenario: CapabilityScenario,
    project: Path,
    settings: Settings,
    *,
    staged: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if scenario.key == "meridian":
        return run_meridian_acceptance_sync(project, settings, staged=staged)
    return run_scenario_acceptance_sync(
        scenario,
        project,
        settings,
        staged=staged,
    )


def _run_aliases(database: Path, run_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT model_aliases_json FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return {}
    try:
        aliases = json.loads(row[0])
    except (TypeError, ValueError):
        return {}
    return aliases if isinstance(aliases, dict) else {}


def _role_chains(
    provider: str,
    orchestrator_model: str | None,
    coder_model: str | None,
    *,
    planner_fallback_models: list[str] | None = None,
    coder_fallback_models: list[str] | None = None,
) -> dict[str, list[RoleChainEntryV1]]:
    """Make the selected provider own all three roles for this evaluation."""

    if provider not in {"local", "oci", "cohere", "cline"}:
        raise ValueError(f"unsupported evaluation provider: {provider}")
    provider_name = cast(Literal["local", "oci", "cohere", "cline"], provider)

    def unique(entries: list[RoleChainEntryV1]) -> list[RoleChainEntryV1]:
        seen: set[tuple[str, str | None]] = set()
        result: list[RoleChainEntryV1] = []
        for entry in entries:
            identity = (entry.provider, entry.model)
            if identity not in seen:
                seen.add(identity)
                result.append(entry)
        return result

    if provider in {"cline", "local"}:
        if not orchestrator_model or not coder_model:
            raise ValueError(f"{provider} requires orchestrator and coder model names")
        planner = [RoleChainEntryV1(provider=provider_name, model=orchestrator_model)]
        coder = [RoleChainEntryV1(provider=provider_name, model=coder_model)]
        if provider == "cline":
            planner.append(
                RoleChainEntryV1(provider="cline", model="cline-pass/glm-5.2")
            )
            coder.extend(
                [
                    RoleChainEntryV1(provider="cline", model="cline-pass/kimi-k3"),
                    RoleChainEntryV1(
                        provider="cline", model="cline-pass/kimi-k2.7-code"
                    ),
                ]
            )
        else:
            planner.extend(
                RoleChainEntryV1(provider="local", model=model)
                for model in (planner_fallback_models or [])
            )
            coder.extend(
                RoleChainEntryV1(provider="local", model=model)
                for model in (coder_fallback_models or [])
            )
        return {
            # A selected backup may also be the default primary.  De-duplicating
            # keeps a timeout from retrying the identical model under the guise
            # of a fallback and makes attribution in the report unambiguous.
            "planner": unique(planner),
            "coder": unique(coder),
            "quality": [
                RoleChainEntryV1(provider=provider_name, model=orchestrator_model)
            ],
        }
    entry = RoleChainEntryV1(provider=provider_name, model=None)
    return {role: [entry] for role in ("planner", "coder", "quality")}


def _models(
    arguments: argparse.Namespace, settings: Settings
) -> tuple[str | None, str | None]:
    planner_fallbacks = list(getattr(arguments, "planner_fallback_models", []) or [])
    coder_fallbacks = list(getattr(arguments, "coder_fallback_models", []) or [])
    if (planner_fallbacks or coder_fallbacks) and arguments.provider != "local":
        raise ValueError(
            "explicit planner/coder fallback models are supported only for the "
            "local Ollama route"
        )
    if arguments.coding_engine == "clinecore" and arguments.provider not in {
        "cline",
        "local",
    }:
        raise ValueError(
            "the ClineCore capability evaluation currently supports only "
            "ClinePass and Ollama provider routes"
        )
    if (
        getattr(arguments, "max_coding_iterations", None) is not None
        and arguments.coding_engine != "clinecore"
    ):
        raise ValueError(
            "--max-coding-iterations is valid only with --coding-engine clinecore"
        )
    if arguments.provider == "cline":
        selected = (
            arguments.orchestrator_model or settings.cline_orchestrator_model,
            arguments.coder_model or settings.cline_coder_model,
        )
        unsupported = [model for model in selected if model not in CLINEPASS_MODELS]
        if unsupported:
            raise ValueError(
                "the Cline capability evaluation accepts only verified "
                "ClinePass models; unsupported or credit-billed model(s): "
                + ", ".join(unsupported)
            )
        return selected
    if arguments.provider == "local":
        return (
            arguments.orchestrator_model or settings.planner_model,
            arguments.coder_model or settings.coder_model,
        )
    return None, None


def _effective_max_coding_iterations(
    arguments: argparse.Namespace,
    settings: Settings,
) -> int | None:
    """Resolve the evaluation-local Cline SDK loop cap, if that engine is active."""

    if arguments.coding_engine != "clinecore":
        return None
    requested = getattr(arguments, "max_coding_iterations", None)
    return requested if requested is not None else settings.cline_sidecar_max_iterations


def _seed_project(
    project: Path,
    project_id: str,
    scenario: CapabilityScenario | None = None,
) -> None:
    scenario = scenario or get_scenario("meridian")
    gitignore = project / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "__pycache__/\n.env\n*.sqlite3\n",
            encoding="utf-8",
        )
    metis = project / ".metis"
    metis.mkdir(exist_ok=True)
    (metis / "project-context.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "project_id": project_id,
                "project_name": scenario.slug,
                "root_name": project.name,
                "revision": 1,
                "bootstrap": {
                    "summary": (
                        f"Disposable seed for the {scenario.name} capability evaluation."
                    ),
                    "architecture": [],
                    "conventions": [],
                    "important_paths": [],
                    "verification": [],
                    "risks": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _describe(arguments: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    scenario = get_scenario(arguments.scenario)
    orchestrator, coder = _models(arguments, settings)
    chains = _role_chains(
        arguments.provider,
        orchestrator,
        coder,
        planner_fallback_models=arguments.planner_fallback_models,
        coder_fallback_models=arguments.coder_fallback_models,
    )
    preview = {
        "scenario": scenario.name,
        "scenario_key": scenario.key,
        "coding_engine": arguments.coding_engine,
        "provider": arguments.provider,
        "orchestrator_model": orchestrator,
        "coder_model": coder,
        "role_chains": {
            role: [entry.model_dump(mode="json") for entry in chain]
            for role, chain in chains.items()
        },
        "planner_fallback_models": list(arguments.planner_fallback_models),
        "coder_fallback_models": list(arguments.coder_fallback_models),
        "max_tokens_per_coding_turn": (
            arguments.max_tokens_per_coding_turn or settings.max_output_tokens
        ),
        "repair_turns": arguments.repair_turns,
        "turn_timeout_seconds": arguments.timeout,
        "model_call_timeout_seconds": arguments.model_call_timeout,
        "max_steps_per_turn": arguments.max_steps,
        "max_coding_iterations": _effective_max_coding_iterations(arguments, settings),
        "required_files": list(scenario.required_files),
        "protected_files": list(scenario.protected_files),
        "seed_files": sorted(scenario.seed_files),
        "request": scenario.request,
        "live": False,
    }
    if scenario.key != "meridian":
        preview["qualification"] = {
            "mode": "binary",
            "pass_score": 100,
            "fail_score": 0,
            "required_gates": list(QUALIFICATION_GATE_NAMES),
        }
    return preview


def _settings(
    arguments: argparse.Namespace,
    *,
    repo_root: Path,
    data_dir: Path,
    project_parent: Path,
) -> Settings:
    env_file = arguments.env_file or repo_root / ".env"
    base = Settings(_env_file=env_file)  # type: ignore[call-arg]
    orchestrator, coder = _models(arguments, base)
    max_coding_iterations = _effective_max_coding_iterations(arguments, base)
    return Settings(  # type: ignore[call-arg]
        _env_file=env_file,
        data_dir=data_dir,
        repo_root=repo_root,
        asset_roots=[project_parent],
        model_backend="ollama",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        project_cloud_coder=False,
        project_coding_engine=arguments.coding_engine,
        project_agent_max_steps=arguments.max_steps,
        # This is a new settings instance passed only to create_app() for this
        # disposable evaluation; no process environment or shared Settings
        # object is mutated.
        cline_sidecar_max_iterations=(
            max_coding_iterations
            if max_coding_iterations is not None
            else base.cline_sidecar_max_iterations
        ),
        project_verify_max_runs=arguments.verify_runs,
        project_sandbox_timeout_seconds=arguments.sandbox_timeout,
        model_call_timeout_seconds=arguments.model_call_timeout,
        cline_orchestrator_model=orchestrator or base.cline_orchestrator_model,
        cline_coder_model=coder or base.cline_coder_model,
        **(
            {"max_output_tokens": arguments.max_tokens_per_coding_turn}
            if arguments.max_tokens_per_coding_turn is not None
            else {}
        ),
    )


def run_live(arguments: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    scenario = get_scenario(arguments.scenario)
    workspace = Path(tempfile.mkdtemp(prefix=f"metis-{scenario.key}-eval-"))
    project_parent = workspace / "Projects"
    project = project_parent / scenario.slug
    data_dir = workspace / "data"
    scenario.initialize(project)
    baseline_protected_hashes = project_file_hashes(project, scenario.protected_files)
    data_dir.mkdir()
    # The asset scanner only treats a child directory as a project when it has
    # a visible file to index; housekeeping dotfiles are intentionally ignored.
    # A starter README is a realistic empty-repository seed and is one of the
    # required deliverables, so the model must still replace its contents.
    settings = _settings(
        arguments,
        repo_root=repo_root,
        data_dir=data_dir,
        project_parent=project_parent,
    )
    orchestrator, coder = _models(arguments, settings)
    chains = _role_chains(
        arguments.provider,
        orchestrator,
        coder,
        planner_fallback_models=arguments.planner_fallback_models,
        coder_fallback_models=arguments.coder_fallback_models,
    )
    attempts: list[dict[str, Any]] = []
    acceptance_history: list[dict[str, Any]] = []
    all_events: dict[str, list[dict[str, Any]]] = {}
    approved = False
    acceptance: dict[str, Any] = {
        "available": False,
        "passed": False,
        "reason": "the build did not reach a clean approved changeset",
        "checks_total": 0,
        "checks_passed": 0,
    }
    conversation_id = ""
    try:
        # Validate the selected lane before the model is called. This catches a
        # missing key/model name locally, while quota and subscription errors
        # remain first-call evidence in the run timeline.
        ModelPreferenceStore(settings).save(
            "split",
            None,
            provider=arguments.provider,
            role_chains=chains,
        )
        with TestClient(create_app(settings)) as client:
            assets = _request(client, "POST", "/api/v1/assets/scan")
            project_id = project_asset_id(assets, project)
            _seed_project(project, project_id, scenario)
            _request(
                client,
                "POST",
                f"/api/v1/projects/{project_id}/open",
                json={"mode": "grok_bootstrap_local"},
            )
            conversation_id = _request(
                client,
                "POST",
                "/api/v1/conversations",
                json={"title": f"{scenario.name} capability evaluation"},
            )["id"]

            prompt = scenario.request
            previous_run_id: str | None = None
            for attempt_index in range(arguments.repair_turns + 1):
                started = time.monotonic()
                accepted = _request(
                    client,
                    "POST",
                    f"/api/v1/conversations/{conversation_id}/messages",
                    json={
                        "content": prompt,
                        "project_id": project_id,
                        "project_mode": "grok_bootstrap_local",
                    },
                )
                run_id = accepted["run_id"]
                _log(
                    f"attempt {attempt_index + 1}/{arguments.repair_turns + 1}: "
                    f"{run_id} ({arguments.coding_engine}, {arguments.provider}, "
                    f"{orchestrator} -> {coder})"
                )
                continuation: dict[str, Any] | None = None
                if previous_run_id is not None:
                    # submit() copies the pending overlay and durably emits this
                    # event before it spawns the graph, so this check is both
                    # immediate and race-free. Without it, waiting for a fresh
                    # rebuild and calling that "repair convergence" would make
                    # the score causally false.
                    continuation = summarize_repair_continuation(
                        read_timeline(data_dir / "waqil.db", run_id),
                        expected_from_run=previous_run_id,
                    )
                if continuation is not None and not continuation["verified"]:
                    try:
                        _request(client, "POST", f"/api/v1/runs/{run_id}/cancel")
                    except Exception:  # noqa: BLE001 - cancellation is best-effort
                        pass
                    run = {
                        **_request(client, "GET", f"/api/v1/runs/{run_id}"),
                        "status": "repair_unavailable",
                        "last_error": continuation["reason"],
                    }
                else:
                    run = _wait_for_run(client, run_id, arguments.timeout)
                duration = time.monotonic() - started
                approval = (
                    _pending_approval(client, run_id)
                    if run.get("status") == "awaiting_approval"
                    else None
                )
                events = read_timeline(data_dir / "waqil.db", run_id)
                summary = summarize_run(
                    run=run,
                    events=events,
                    duration_seconds=duration,
                    approval=approval,
                    required_files=scenario.required_files,
                )
                summary["coding_engine"]["selected"] = arguments.coding_engine
                summary["model_aliases"] = _run_aliases(data_dir / "waqil.db", run_id)
                summary["repair_continuation"] = continuation or {
                    "required": False,
                    "verified": True,
                    "expected_from_run": "",
                    "from_run": "",
                    "files": [],
                    "reason": "initial build",
                }
                attempts.append(summary)
                if arguments.include_events:
                    all_events[run_id] = event_dicts(events)
                blockers = int(summary["verification"]["blocking"])
                _log(
                    f"  status={run.get('status')} writes="
                    f"{summary['writes']['successful']}/{summary['writes']['attempted']} "
                    f"blocking={blockers} refused={summary['writes']['refused']}"
                )

                if continuation is not None and not continuation["verified"]:
                    acceptance = {
                        **acceptance,
                        "reason": (
                            f"repair continuation unavailable: {continuation['reason']}"
                        ),
                    }
                    _log(f"  {acceptance['reason']}")
                    break

                clean_approval = bool(
                    approval and not approval.get("blocked_reason") and blockers == 0
                )
                if clean_approval:
                    if not arguments.approve:
                        break

                    planned_files = list(
                        ((attempts[0].get("plan") or {}).get("files") or [])
                    )
                    scope_contract = attempts[0].get("scope_contract") or {}
                    scope_mode = str(scope_contract.get("mode") or "planner_manifest")
                    authorized_files = list(
                        scope_contract.get("authorized_files") or planned_files
                    )
                    if continuation is not None:
                        repair_evidence = repair_attempt_evidence(
                            summary,
                            planned_paths=authorized_files,
                        )
                        summary["repair_execution_evidence"] = {
                            "continuation": repair_evidence[0],
                            "in_plan_write": repair_evidence[1],
                            "post_write_verification": repair_evidence[2],
                            "verified": all(repair_evidence),
                        }
                        if not all(repair_evidence):
                            acceptance = {
                                **acceptance,
                                "reason": (
                                    "repair follow-up was not eligible for approval: "
                                    "continuity, a real in-plan write, and post-write "
                                    "verification are all required"
                                ),
                            }
                            _log(f"  {acceptance['reason']}")
                            break

                    pending_staged = _pending_project_overlay(
                        client,
                        conversation_id,
                        run_id,
                    )
                    host_scaffold_paths = {
                        str(path)
                        for attempt in attempts
                        for path in (
                            (attempt.get("writes") or {}).get("host_scaffold_paths")
                            or []
                        )
                    }
                    overlay_guard = validate_pending_overlay(
                        pending_staged,
                        required_files=scenario.required_files,
                        protected_files=scenario.protected_files,
                        planned_files=planned_files,
                        scope_mode=scope_mode,
                        contract_observed=bool(scope_contract.get("present")),
                        host_scaffold_paths=host_scaffold_paths,
                        allow_host_scaffold=scenario.allow_host_scaffold,
                        baseline_protected_hashes=baseline_protected_hashes,
                        current_protected_hashes=project_file_hashes(
                            project, scenario.protected_files
                        ),
                    )
                    summary["preapproval_overlay"] = overlay_guard
                    if not overlay_guard["valid"]:
                        acceptance = {
                            **acceptance,
                            "reason": (
                                "pre-approval overlay guard failed: "
                                f"{overlay_guard['reason']}"
                            ),
                            "preapproval_overlay": overlay_guard,
                        }
                        _log(f"  {acceptance['reason']}")
                        break

                    _log(
                        f"  running pre-approval {scenario.name} acceptance "
                        "against the exact pending overlay"
                    )
                    source_tree_before_probe = project_tree_digest(project)
                    preapproval = {
                        **_run_acceptance(
                            scenario,
                            project,
                            settings,
                            staged=pending_staged,
                        ),
                        "phase": "preapproval",
                        "run_id": run_id,
                        "preapproval_overlay": overlay_guard,
                    }
                    protected_after_probe = project_file_hashes(
                        project, scenario.protected_files
                    )
                    source_tree_intact = (
                        project_tree_digest(project) == source_tree_before_probe
                    )
                    source_intact = (
                        protected_after_probe == baseline_protected_hashes
                        and source_tree_intact
                    )
                    preapproval["source_protected_hashes_intact"] = source_intact
                    preapproval["source_tree_unchanged_by_probe"] = source_tree_intact
                    if not source_intact:
                        preapproval = {
                            **preapproval,
                            "passed": False,
                            "checks_passed": 0,
                            "reason": (
                                "protected source hashes changed during disposable "
                                "pre-approval acceptance"
                            ),
                        }
                    acceptance_history.append(preapproval)
                    acceptance = preapproval

                    if preapproval.get("passed"):
                        assert approval is not None
                        _request(
                            client,
                            "POST",
                            f"/api/v1/runs/{run_id}/decisions",
                            json={
                                "approval_id": approval["id"],
                                "decision": "approve",
                                "reason": (
                                    f"Disposable {scenario.name} capability evaluation; "
                                    "exact pending overlay passed host acceptance"
                                ),
                            },
                        )
                        terminal = _wait_for_run(
                            client,
                            run_id,
                            arguments.timeout,
                            {"completed", "failed", "cancelled"},
                        )
                        approved = terminal.get("status") == "completed"
                        break

                    _log(f"  pre-approval acceptance failed: {preapproval['reason']}")
                    if (
                        not preapproval.get("available")
                        or attempt_index >= arguments.repair_turns
                    ):
                        break
                    try:
                        prompt = acceptance_repair_prompt(
                            scenario.repair_request,
                            preapproval,
                        )
                    except ValueError as error:
                        acceptance = {
                            **preapproval,
                            "reason": f"acceptance repair unavailable: {error}",
                        }
                        break
                    previous_run_id = run_id
                    _log("  submitting one bounded acceptance repair follow-up")
                    continue
                if attempt_index >= arguments.repair_turns:
                    break
                if run.get("status") != "awaiting_approval":
                    break
                previous_run_id = run_id
                prompt = scenario.repair_request

        if approved:
            _log(f"running post-approval {scenario.name} acceptance in the sandbox")
            acceptance = {
                **_run_acceptance(scenario, project, settings),
                "phase": "postapproval",
                "source_protected_hashes_intact": (
                    project_file_hashes(project, scenario.protected_files)
                    == baseline_protected_hashes
                ),
            }
            if not acceptance["source_protected_hashes_intact"]:
                acceptance = {
                    **acceptance,
                    "passed": False,
                    "checks_passed": 0,
                    "reason": "protected source hashes changed after approval",
                }
    except Exception as error:  # noqa: BLE001 - the report must survive infrastructure failure
        acceptance = {
            **acceptance,
            "reason": f"evaluation driver failed: {type(error).__name__}: {error}",
        }

    report = {
        "schema_version": "1",
        "scenario": scenario.name,
        "configuration": {
            "coding_engine": arguments.coding_engine,
            "provider": arguments.provider,
            "orchestrator_model": orchestrator,
            "coder_model": coder,
            "role_chains": {
                role: [entry.model_dump(mode="json") for entry in chain]
                for role, chain in chains.items()
            },
            "planner_fallback_models": list(arguments.planner_fallback_models),
            "coder_fallback_models": list(arguments.coder_fallback_models),
            "max_tokens_per_coding_turn": settings.max_output_tokens,
            "repair_turns": arguments.repair_turns,
            "turn_timeout_seconds": arguments.timeout,
            "model_call_timeout_seconds": arguments.model_call_timeout,
            "max_steps_per_turn": arguments.max_steps,
            "max_coding_iterations": (
                settings.cline_sidecar_max_iterations
                if arguments.coding_engine == "clinecore"
                else None
            ),
            "verify_runs_per_turn": arguments.verify_runs,
            "auto_approve_clean_disposable_build": arguments.approve,
        },
        "attempts": attempts,
        "acceptance_history": acceptance_history,
        "acceptance": acceptance,
        "approved": approved,
        "workspace": str(workspace) if arguments.keep_workspace else None,
    }
    report["score"] = score_evaluation(
        cast(list[Mapping[str, Any]], attempts),
        acceptance,
        max_steps=arguments.max_steps,
    )
    if scenario.key != "meridian":
        report["scenario_key"] = scenario.key
        report["qualification"] = qualify_scenario(
            scenario,
            attempts,
            acceptance,
            approved=approved,
        )
    if arguments.include_events:
        report["events"] = all_events
    if not arguments.keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="meridian",
        help="select the disposable production-shaped project fixture",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="spend provider tokens and execute the disposable evaluation",
    )
    parser.add_argument(
        "--provider",
        choices=("cline", "local", "cohere", "oci"),
        default="cline",
    )
    parser.add_argument(
        "--coding-engine",
        choices=("legacy", "clinecore"),
        default="clinecore",
        help=(
            "select the inner project coding loop; clinecore (the default, and "
            "the only engine available for a new production build) runs the "
            "local Cline sidecar while Metis retains planning, verification, "
            "and approval. legacy is retained only as an explicit "
            "historical/test-baseline comparison, never a production route"
        ),
    )
    parser.add_argument(
        "--orchestrator-model",
        help="defaults to WAQIL_CLINE_ORCHESTRATOR_MODEL (cline-pass/qwen3.7-plus)",
    )
    parser.add_argument(
        "--coder-model",
        help="defaults to cline-pass/deepseek-v4-pro for Cline",
    )
    parser.add_argument(
        "--planner-fallback-model",
        dest="planner_fallback_models",
        action="append",
        default=[],
        metavar="MODEL",
        help=(
            "repeatable explicit Ollama planner fallback; valid only with "
            "--provider local"
        ),
    )
    parser.add_argument(
        "--coder-fallback-model",
        dest="coder_fallback_models",
        action="append",
        default=[],
        metavar="MODEL",
        help=(
            "repeatable explicit Ollama coder fallback; valid only with "
            "--provider local"
        ),
    )
    parser.add_argument("--repair-turns", type=int, choices=range(0, 4), default=1)
    # A production-shaped 16-file build reached verification in just under
    # twenty minutes on the ClinePass ladder.  The previous fifteen-minute
    # default cancelled a healthy sequence of writes before the repair loop;
    # keep individual calls tightly bounded, but give the whole causal loop
    # enough wall time to reach its acceptance gate.
    parser.add_argument("--timeout", type=_positive_float, default=1800.0)
    parser.add_argument(
        "--model-call-timeout",
        type=_bounded_integer(60, 900),
        default=300,
        help="advance the role ladder when one provider call exceeds this bound",
    )
    parser.add_argument("--max-steps", type=_bounded_integer(8, 80), default=48)
    parser.add_argument(
        "--max-coding-iterations",
        type=_bounded_integer(1, 200),
        default=None,
        help=(
            "bound ClineCore's inner SDK tool loop independently of --max-steps; "
            "omitted preserves Settings.cline_sidecar_max_iterations"
        ),
    )
    parser.add_argument(
        "--max-tokens-per-coding-turn",
        type=_bounded_integer(4_096, 32_768),
        default=None,
        help=(
            "bound one coding model turn; omitted preserves Settings.max_output_tokens"
        ),
    )
    parser.add_argument("--verify-runs", type=_bounded_integer(1, 12), default=6)
    parser.add_argument(
        "--sandbox-timeout", type=_bounded_integer(30, 600), default=180
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--report", type=Path, help="also write the JSON report here")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument(
        "--no-approve",
        dest="approve",
        action="store_false",
        help="leave a clean disposable changeset at its approval card",
    )
    parser.set_defaults(approve=True)
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help="exit non-zero when the score is below this threshold",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    preview_settings = Settings(  # type: ignore[call-arg]
        _env_file=arguments.env_file or repo_root / ".env"
    )
    try:
        report = (
            run_live(arguments, repo_root)
            if arguments.live
            else _describe(arguments, preview_settings)
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered + "\n", encoding="utf-8")
    if arguments.fail_below is not None:
        if not release_gate_passes(report, arguments.fail_below):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
