"""Permanent, provider-neutral scoring for realistic project-build evaluations.

The live driver lives in ``scripts/project_capability_eval.py``.  This module
keeps the scenario, event accounting, scorecard, and qualification acceptance
probe importable and unit-testable without spending model tokens.

Generated project code is never imported on the host.  Before an evaluator
approval, its exact pending overlay and the host probe are staged only into the
existing networkless project-verification container.  Both disappear with that
container's temporary filesystem; an approved result is replayed once more.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence, cast

from .coding_contracts import CodingSessionState
from .config import Settings
from .project_sandbox import ProjectSandboxService


SCENARIO_NAME = "Meridian Evidence Desk"

REQUIRED_FILES: tuple[str, ...] = (
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/db.py",
    "app/models.py",
    "app/repository.py",
    "app/extraction.py",
    "app/services.py",
    "app/routes/__init__.py",
    "app/routes/documents.py",
    "app/routes/questions.py",
    "app/static/index.html",
    "app/static/app.js",
    "requirements.txt",
    "README.md",
    "tests/test_workflows.py",
)

MERIDIAN_REQUEST = """Build a complete application named \"Meridian Evidence Desk\".

This is a production-shaped acceptance task, not a mockup. Use Python 3.13 and
FastAPI. Serve a responsive same-application web UI and JSON API. Use stdlib
sqlite3 for durable document, review, approval, question, citation, and audit
state. The application must import and its health and TXT workflows must work
without OCI credentials or network access.

Create exactly these application-owned files (Metis may also supply appkit):
app/__init__.py, app/main.py, app/config.py, app/db.py, app/models.py,
app/repository.py, app/extraction.py, app/services.py, app/routes/__init__.py,
app/routes/documents.py, app/routes/questions.py, app/static/index.html,
app/static/app.js, requirements.txt, README.md, tests/test_workflows.py.

Required contracts:
- app.main exposes app and create_app(). Configuration is read lazily. The
  database path comes from MERIDIAN_DB_PATH, defaulting to ./meridian.sqlite3.
- Mount the supplied appkit design language and serve app/static/index.html at
  GET /. The page must show Meridian Evidence Desk, an upload dropzone, a
  review form, extraction status, an audit timeline, and a cited follow-up
  panel. app/static/app.js must wire those controls to the routes below.
- GET /api/health returns HTTP 200 without OCI configuration.
- POST /api/documents accepts multipart field \"file\". Reject unsupported
  executable content with HTTP 415. A UTF-8 TXT invoice is extracted locally
  and synchronously into invoice_number, vendor, invoice_date, currency, and
  total, returning document_id and status \"needs_review\". PNG/JPEG/PDF use
  the supplied appkit.oci_responses.OciResponses adapter; never invent another
  OCI client and never require OCI at import time.
- GET /api/documents/{document_id} returns the document and extracted fields.
- POST /api/documents/{document_id}/review accepts {\"fields\": {...}}, saves
  corrections, and returns status \"reviewed\".
- POST /api/documents/{document_id}/approve returns status \"approved\".
- POST /api/documents/{document_id}/questions accepts {\"question\": \"...\"}
  and returns an evidence-grounded answer plus a non-empty citations array.
  TXT follow-ups must work locally; each citation includes source and quote.
- SQLite has tables named documents and audit_events. documents has id and
  status columns. audit_events has document_id and event_type columns and
  records upload, review, and approve for the workflow.
- requirements.txt declares every non-stdlib dependency. Include meaningful
  workflow tests. Do not use placeholder data, TODO handlers, or pass stubs.

Acceptance workflows the finished project must satisfy:
1. The page and health route work with no OCI environment variables.
2. An executable upload is refused with HTTP 415.
3. Uploading the TXT invoice below extracts INV-1042, Nimbus Systems, AED, and
   431.75; review changes the vendor; approval persists after a fresh request.
4. A follow-up asking for the approved total and invoice number returns 431.75,
   INV-1042, and at least one source quote.
5. SQLite shows the approved document and upload, review, approve audit events.

Do not stop at a plan or describe files that were not staged. Run the real
checks available to you and repair every blocking finding before finishing.
"""

REPAIR_REQUEST = """Continue the Meridian build already staged in this project.
Treat the latest verification findings as the complete repair queue. Fix their
root causes one file at a time, using the actual staged text and the supplied
appkit contracts. Do not redesign working parts, do not add substitute APIs,
and do not finish until verification has zero blocking findings. Re-run the
acceptance workflows after the repairs.
"""


def initialize_meridian_project(project: Path) -> None:
    """Create the smallest realistic repository the asset scanner can see."""

    project.mkdir(parents=True)
    (project / ".gitignore").write_text(
        "__pycache__/\n.env\n*.sqlite3\n",
        encoding="utf-8",
    )
    (project / "README.md").write_text(
        "# Meridian Evidence Desk\n\nStarter repository; replace this README during the build.\n",
        encoding="utf-8",
    )


def project_asset_id(assets: Iterable[Mapping[str, Any]], project: Path) -> str:
    """Find the scanned asset by Metis's path-derived stable identity."""

    target = project.resolve()
    expected = f"asset_{hashlib.sha256(str(target).encode('utf-8')).hexdigest()[:20]}"
    for asset in assets:
        if str(asset.get("id") or "") == expected:
            return expected
    raise LookupError(f"project scan did not return {target}")


def _canonical_evaluation_path(value: str) -> str:
    """Validate an exact project-relative path used by the evaluator.

    Pending bytes come from Metis's own checkpoint, but qualification is a
    security boundary too: a corrupt checkpoint must not make a host-owned
    probe read or materialize outside the disposable project.  Unlike the
    product workspace, this helper deliberately does not normalize malformed
    input.  Qualification evidence should describe the exact checkpoint bytes,
    not a repaired interpretation of them.
    """

    if not isinstance(value, str):
        raise ValueError("pending overlay path is not text")
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 for character in value)
        or len(value.encode("utf-8")) > 1_000
    ):
        raise ValueError(f"invalid pending overlay path: {value!r}")
    return value


def pending_overlay_digest(staged: Mapping[str, Mapping[str, Any]]) -> str:
    """Digest every exact pending path and content byte without exposing code."""

    hasher = hashlib.sha256()
    for raw_path in sorted(staged):
        path = _canonical_evaluation_path(raw_path)
        entry = staged[raw_path]
        if not isinstance(entry, Mapping):
            raise ValueError(f"pending overlay entry is not an object: {path}")
        content = entry.get("content")
        if not isinstance(content, str):
            raise ValueError(f"pending overlay content is not text: {path}")
        encoded_path = path.encode("utf-8")
        encoded_content = content.encode("utf-8")
        hasher.update(len(encoded_path).to_bytes(8, "big"))
        hasher.update(encoded_path)
        hasher.update(len(encoded_content).to_bytes(8, "big"))
        hasher.update(encoded_content)
    return hasher.hexdigest()


def project_file_hashes(root: Path, paths: Iterable[str]) -> dict[str, str]:
    """Hash protected source files, preserving missing/symlink distinctions."""

    hashes: dict[str, str] = {}
    for raw_path in sorted(set(paths)):
        path = _canonical_evaluation_path(raw_path)
        target = root / path
        if target.is_symlink():
            hashes[path] = "symlink"
        elif not target.is_file():
            hashes[path] = "missing"
        else:
            hashes[path] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def project_tree_digest(root: Path) -> str:
    """Digest the visible source tree to prove a sandbox replay was read-only."""

    hasher = hashlib.sha256()
    if not root.is_dir():
        return hasher.hexdigest()
    for target in sorted(root.rglob("*")):
        relative = target.relative_to(root)
        if ".metis" in relative.parts or not target.is_file():
            continue
        path = _canonical_evaluation_path(relative.as_posix())
        encoded_path = path.encode("utf-8")
        hasher.update(len(encoded_path).to_bytes(8, "big"))
        hasher.update(encoded_path)
        if target.is_symlink():
            hasher.update(b"symlink")
        else:
            content = target.read_bytes()
            hasher.update(len(content).to_bytes(8, "big"))
            hasher.update(content)
    return hasher.hexdigest()


def validate_pending_overlay(
    staged: Mapping[str, Mapping[str, Any]],
    *,
    required_files: Iterable[str],
    protected_files: Iterable[str],
    planned_files: Iterable[str],
    scope_mode: str = "planner_manifest",
    contract_observed: bool = True,
    host_scaffold_paths: Iterable[str] = (),
    allow_host_scaffold: bool = False,
    baseline_protected_hashes: Mapping[str, str] | None = None,
    current_protected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail-closed pre-approval scope and source-integrity evidence.

    The model must have staged every requested deliverable and must not have
    touched a protected file. Planner/slice runs additionally prove an exact
    manifest. Direct runs instead prove that they were admitted under the
    durable direct contract and that the independently observed final overlay
    is the exact requested scope. Host scaffolding is accepted only for a
    scenario that explicitly permits it and only for paths independently
    observed in ``project.scaffold_staged``.
    """

    required = {_canonical_evaluation_path(path) for path in required_files}
    protected = {_canonical_evaluation_path(path) for path in protected_files}
    planned = {_canonical_evaluation_path(path) for path in planned_files}
    scaffold = {_canonical_evaluation_path(path) for path in host_scaffold_paths}
    reasons: list[str] = []
    try:
        digest = pending_overlay_digest(staged)
        overlay_paths = {_canonical_evaluation_path(path) for path in staged}
    except ValueError as error:
        return {
            "valid": False,
            "reason": str(error),
            "overlay_digest": "",
            "overlay_paths": [],
            "missing_required_files": sorted(required),
            "out_of_scope_paths": [],
            "protected_paths_touched": [],
            "exact_planned_scope": planned == required,
            "exact_requested_scope": False,
            "scope_mode": scope_mode,
            "contract_observed": contract_observed,
            "source_protected_hashes_intact": False,
        }

    missing = sorted(required - overlay_paths)
    protected_touched = sorted(protected & overlay_paths)
    allowed = required | (scaffold if allow_host_scaffold else set())
    out_of_scope = sorted(overlay_paths - allowed)
    exact_plan = planned == required
    exact_requested_scope = not missing and not out_of_scope and not protected_touched
    if scope_mode not in {"planner_manifest", "direct_contract"}:
        reasons.append(f"unknown scope contract mode: {scope_mode}")
    source_intact = (
        baseline_protected_hashes is None
        or current_protected_hashes is not None
        and dict(current_protected_hashes) == dict(baseline_protected_hashes)
    )
    if not staged:
        reasons.append("pending changeset is empty")
    if scope_mode == "planner_manifest" and not exact_plan:
        reasons.append("planner scope does not exactly match the required files")
    if scope_mode == "direct_contract" and not contract_observed:
        reasons.append("durable direct-build contract was not observed")
    if missing:
        reasons.append("pending changeset omits required files: " + ", ".join(missing))
    if out_of_scope:
        reasons.append(
            "pending changeset contains out-of-scope files: " + ", ".join(out_of_scope)
        )
    if protected_touched:
        reasons.append(
            "pending changeset touches protected files: " + ", ".join(protected_touched)
        )
    if not source_intact:
        reasons.append("protected source hashes changed before approval")
    return {
        "valid": not reasons,
        "reason": "; ".join(reasons),
        "overlay_digest": digest,
        "overlay_paths": sorted(overlay_paths),
        "missing_required_files": missing,
        "out_of_scope_paths": out_of_scope,
        "protected_paths_touched": protected_touched,
        "exact_planned_scope": exact_plan,
        "exact_requested_scope": exact_requested_scope,
        "scope_mode": scope_mode,
        "contract_observed": contract_observed,
        "source_protected_hashes_intact": source_intact,
    }


def acceptance_staged_overlay(
    staged: Mapping[str, Mapping[str, Any]] | None,
    *,
    probe_path: str,
    probe_source: str,
) -> dict[str, dict[str, Any]]:
    """Add a host probe to exact pending bytes without mutating either input."""

    probe = _canonical_evaluation_path(probe_path)
    result: dict[str, dict[str, Any]] = {}
    for raw_path, raw_entry in (staged or {}).items():
        path = _canonical_evaluation_path(raw_path)
        if path == probe:
            raise ValueError(
                "pending changeset collides with the host acceptance probe"
            )
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"pending overlay entry is not an object: {path}")
        content = raw_entry.get("content")
        if not isinstance(content, str):
            raise ValueError(f"pending overlay content is not text: {path}")
        result[path] = dict(raw_entry)
    result[probe] = {"content": probe_source, "origin": "host-evaluation"}
    return result


def bounded_acceptance_findings(
    acceptance: Mapping[str, Any],
    *,
    max_findings: int = 6,
    max_finding_chars: int = 700,
) -> list[str]:
    """Return bounded verbatim host fields suitable for one repair follow-up."""

    candidates: list[str] = []
    probe = acceptance.get("required_probe") or {}
    if isinstance(probe, Mapping) and not probe.get("ok"):
        fields = [
            str(probe.get("name") or "required acceptance probe"),
            str(probe.get("error_type") or "").strip(),
            str(probe.get("detail") or "").strip(),
            str(probe.get("where") or "").strip(),
        ]
        candidates.append(": ".join(field for field in fields if field))
    for finding in acceptance.get("blocking_findings") or []:
        if not isinstance(finding, Mapping):
            continue
        fields = [
            str(finding.get("path") or "").strip(),
            str(finding.get("error") or "").strip(),
        ]
        candidates.append(": ".join(field for field in fields if field))
    if not candidates and acceptance.get("reason"):
        candidates.append(str(acceptance.get("reason")))

    findings: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        # Provider prompts are line-oriented. Preserve the sandbox's words but
        # collapse transport whitespace and enforce a hard information bound.
        bounded = " ".join(candidate.split())[:max_finding_chars]
        if bounded and bounded not in seen:
            seen.add(bounded)
            findings.append(bounded)
        if len(findings) >= max_findings:
            break
    return findings


def acceptance_repair_prompt(
    base_request: str,
    acceptance: Mapping[str, Any],
) -> str:
    """Attach exact sandbox findings to the scenario's bounded repair request."""

    findings = bounded_acceptance_findings(acceptance)
    if not findings:
        raise ValueError("failed acceptance supplied no bounded repair finding")
    lines = [
        base_request.strip(),
        "",
        "The product verifier is clean, but the host-owned qualification probe failed "
        "inside the disposable networkless sandbox. Repair these exact findings:",
        *(f"- {finding}" for finding in findings),
        "",
        "Keep the prior staged overlay and exact host-owned write scope. Make a real "
        "in-scope edit, then finish so Metis can verify the post-write changeset again.",
    ]
    return "\n".join(lines)


# This source is copied into the disposable verifier overlay after an approved
# build.  It is intentionally plain Python rather than pytest: importing it is
# already a bounded, reported sandbox check.  Every assertion names the broken
# user-facing contract so a failure is actionable.
MERIDIAN_ACCEPTANCE_SOURCE = textwrap.dedent(
    '''\
    """Host-owned acceptance probe; staged only in the disposable verifier."""
    from __future__ import annotations

    import ast
    import json
    import os
    import sqlite3
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient


    root = Path.cwd()
    DATABASE = Path("/tmp/meridian-capability-eval.sqlite3")
    try:
        DATABASE.unlink()
    except FileNotFoundError:
        pass
    os.environ["MERIDIAN_DB_PATH"] = str(DATABASE)
    os.environ.pop("OCI_CONFIG_FILE", None)
    os.environ.pop("OCI_CLI_PROFILE", None)
    os.environ.pop("OCI_PROFILE", None)

    # The generic sandbox import checks run before this host-owned probe. They
    # may already have imported app.main under the sandbox's default database
    # path, which would make this probe test verifier process history rather
    # than a fresh production start with MERIDIAN_DB_PATH configured. Reload
    # application-owned modules only when a real on-disk app package exists;
    # trusted in-process fixture tests intentionally supply their own module.
    if (root / "app").is_dir():
        for module_name in tuple(sys.modules):
            if module_name == "app" or module_name.startswith("app."):
                sys.modules.pop(module_name, None)

    from app.main import create_app


    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "starter repository; replace" not in readme.casefold(), (
        "README.md is still the starter placeholder"
    )
    readme_folded = readme.casefold().replace("‑", "-").replace("–", "-")
    missing_readme_guidance = []
    if "uvicorn" not in readme_folded:
        missing_readme_guidance.append("the uvicorn start command")
    if "oci" not in readme_folded:
        missing_readme_guidance.append("OCI image/PDF setup or fallback behavior")
    if not any(
        phrase in readme_folded
        for phrase in ("without oci", "no oci", "no-oci")
    ):
        missing_readme_guidance.append("the credentialless/no-OCI TXT workflow")
    assert not missing_readme_guidance, (
        "README.md omits required setup/fallback guidance: "
        + "; ".join(missing_readme_guidance)
    )

    workflow_tests = (root / "tests" / "test_workflows.py").read_text(
        encoding="utf-8"
    )
    assert workflow_tests.count("def test_") >= 3, (
        "tests/test_workflows.py needs at least three executable workflow tests"
    )
    for marker in ("/api/health", "/api/documents", "415", "citations"):
        assert marker in workflow_tests, (
            f"workflow tests omit required behavior marker {marker!r}"
        )

    requirements = (root / "requirements.txt").read_text(encoding="utf-8").casefold()
    required_dependencies = {"fastapi", "python-multipart", "uvicorn"}
    # FastAPI itself depends on Pydantic, but that does not make Pydantic a
    # direct application dependency. Require an explicit declaration only when
    # application-owned source imports it directly; otherwise a valid dataclass
    # implementation is rejected for omitting a transitive package it never
    # uses.
    for source_path in (root / "app").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.partition(".")[0]]
            else:
                modules = []
            if "pydantic" in modules:
                required_dependencies.add("pydantic")
    for dependency in sorted(required_dependencies):
        assert dependency in requirements, (
            f"requirements.txt does not explicitly declare {dependency}"
        )

    application = create_app()
    invoice = b"""Invoice Number: INV-1042
    Vendor: Nimbus Systems
    Invoice Date: 2026-07-31
    Currency: AED
    Total: 431.75
    Description: Evidence processing subscription
    """

    with TestClient(application) as client:
        health = client.get("/api/health")
        assert health.status_code == 200, (
            f"health must work without OCI credentials; got {health.status_code}"
        )

        page = client.get("/")
        assert page.status_code == 200, f"GET / returned {page.status_code}"
        html = page.text.casefold()
        for phrase in ("meridian evidence desk", "upload", "review", "audit"):
            assert phrase in html, f"the UI does not expose the {phrase!r} workflow"
        assert "viewport" in html, "the UI omits a responsive viewport"
        assert "app.js" in html, "the UI does not load its interaction script"

        script = client.get("/static/app.js")
        assert script.status_code == 200, "the UI interaction script is not served"
        assert "fetch(" in script.text, "the UI script never calls the application API"

        refused = client.post(
            "/api/documents",
            files={"file": ("payload.exe", b"MZ-not-an-invoice", "application/octet-stream")},
        )
        assert refused.status_code == 415, (
            f"unsupported executable upload must return 415, got {refused.status_code}"
        )

        uploaded = client.post(
            "/api/documents",
            files={"file": ("invoice.txt", invoice, "text/plain")},
        )
        assert 200 <= uploaded.status_code < 300, (
            f"TXT upload failed with HTTP {uploaded.status_code}: {uploaded.text[:300]}"
        )
        upload_body = uploaded.json()
        document_id = upload_body.get("document_id")
        assert document_id, "TXT upload response omits document_id"

        fetched = client.get(f"/api/documents/{document_id}")
        assert fetched.status_code == 200, "the uploaded document cannot be fetched"
        extracted_text = json.dumps(fetched.json(), sort_keys=True).casefold()
        for value in ("inv-1042", "nimbus systems", "aed", "431.75"):
            assert value in extracted_text, f"TXT extraction omitted {value!r}"

        reviewed = client.post(
            f"/api/documents/{document_id}/review",
            json={"fields": {"vendor": "Nimbus Systems LLC"}},
        )
        assert 200 <= reviewed.status_code < 300, (
            f"review failed with HTTP {reviewed.status_code}: {reviewed.text[:300]}"
        )
        assert reviewed.json().get("status") == "reviewed", (
            "review does not return status 'reviewed'"
        )

        approved = client.post(f"/api/documents/{document_id}/approve")
        assert 200 <= approved.status_code < 300, (
            f"approval failed with HTTP {approved.status_code}: {approved.text[:300]}"
        )
        assert approved.json().get("status") == "approved", (
            "approval does not return status 'approved'"
        )

        persisted = client.get(f"/api/documents/{document_id}")
        persisted_text = json.dumps(persisted.json(), sort_keys=True).casefold()
        assert persisted.status_code == 200 and "approved" in persisted_text, (
            "approval is not visible on a fresh document request"
        )
        assert "nimbus systems llc" in persisted_text, (
            "reviewed fields are not visible on a fresh document request"
        )

        question = client.post(
            f"/api/documents/{document_id}/questions",
            json={"question": "What is the approved total and invoice number?"},
        )
        assert 200 <= question.status_code < 300, (
            f"follow-up failed with HTTP {question.status_code}: {question.text[:300]}"
        )
        answer = question.json()
        answer_text = json.dumps(answer, sort_keys=True).casefold()
        assert "431.75" in answer_text and "inv-1042" in answer_text, (
            "follow-up answer is not grounded in the approved invoice"
        )
        citations = answer.get("citations")
        assert isinstance(citations, list) and citations, (
            "follow-up answer has no citations"
        )
        assert all(item.get("source") and item.get("quote") for item in citations), (
            "each citation must include source and quote"
        )

    assert DATABASE.is_file(), "the workflow did not create the configured SQLite file"
    with sqlite3.connect(DATABASE) as connection:
        status = connection.execute(
            "SELECT status FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        assert status and status[0] == "approved", (
            "SQLite does not retain the approved document status"
        )
        events = {
            str(row[0]).casefold()
            for row in connection.execute(
                "SELECT event_type FROM audit_events WHERE document_id = ?", (document_id,)
            )
        }
        for expected in ("upload", "review", "approve"):
            assert any(expected in event for event in events), (
                f"SQLite audit history omits the {expected!r} event"
            )
    '''
)


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """One persisted event from one project run."""

    type: str
    payload: dict[str, Any]


def read_timeline(database: Path, run_id: str) -> list[TimelineEvent]:
    """Read one run's durable event trace without opening the SSE endpoint."""

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT type, payload_json FROM run_events WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    events: list[TimelineEvent] = []
    for event_type, raw in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = {}
        events.append(
            TimelineEvent(
                type=str(event_type),
                payload=payload if isinstance(payload, dict) else {},
            )
        )
    return events


def summarize_repair_continuation(
    events: Iterable[TimelineEvent], *, expected_from_run: str
) -> dict[str, Any]:
    """Prove that a repair turn inherited the exact prior run's overlay.

    A follow-up message is a new run, but the product-supported continuation
    path copies the newest undecided project changeset into that run before its
    graph starts. ``project.staged_resumed`` is emitted only after the prior
    checkpoint yielded a non-empty overlay and that overlay was installed in
    the new state. Requiring both its source run and file list prevents the
    evaluator from scoring a fresh rebuild as repair convergence if that path
    ever regresses or selects the wrong pending approval.
    """

    resumed = [
        event.payload for event in events if event.type == "project.staged_resumed"
    ]
    payload = resumed[-1] if resumed else {}
    from_run = str(payload.get("from_run") or "")
    files = sorted(
        {str(path) for path in payload.get("files") or [] if str(path).strip()}
    )
    if not resumed:
        reason = "the follow-up emitted no project.staged_resumed event"
    elif from_run != expected_from_run:
        reason = (
            "the follow-up resumed a different run "
            f"({from_run or 'none'} instead of {expected_from_run})"
        )
    elif not files:
        reason = "the resumed changeset contained no staged files"
    else:
        reason = ""
    return {
        "required": True,
        "verified": not reason,
        "expected_from_run": expected_from_run,
        "from_run": from_run,
        "files": files,
        "reason": reason,
    }


_USAGE_COUNTERS = ("inputTokens", "outputTokens", "totalTokens", "requests")
_TOKEN_COUNTERS = ("inputTokens", "outputTokens", "totalTokens")


def _usage_counter(usage: Mapping[str, Any], name: str) -> int:
    try:
        return max(0, int(usage.get(name) or 0))
    except (TypeError, ValueError):
        return 0


def summarize_coding_usage(
    coding_rounds: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], str, list[dict[str, Any]]]:
    """Count Cline SDK usage once across seeded restart ancestry.

    ClineCore's accumulated token counters are reconstructed from the seeded
    message history after ``restartWithModel``. A child's first token totals
    therefore include its parent's totals, while the sidecar's ``requests``
    counter is child-local. Maxima prevent repeated same-session continuation
    events from double-counting; child token deltas prevent the inherited
    transcript from being charged a second time.

    Older events did not persist ``parent_sidecar_session_id``. Their ancestry
    is safely inferred only from an identity change within the same durable
    Metis coding session, which is how the coordinator serializes restarts.
    """

    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    previous_by_host: dict[str, str] = {}
    missing_identity = False
    inferred_parent = False
    for item in coding_rounds:
        raw_usage = item.get("usage")
        if not isinstance(raw_usage, Mapping):
            continue
        host_session = str(item.get("session_id") or "unknown")
        identity = str(item.get("sidecar_session_id") or "").strip()
        if not identity:
            missing_identity = True
            identity = f"{host_session}:{item.get('model') or 'unknown'}"
        explicit_parent = str(
            item.get("parent_sidecar_session_id") or item.get("parent_session_id") or ""
        ).strip()
        record = records.get(identity)
        if record is None:
            parent = explicit_parent
            previous = previous_by_host.get(host_session, "")
            if not parent and previous and previous != identity:
                parent = previous
                inferred_parent = True
            record = {
                "sidecar_session_id": identity,
                "parent_sidecar_session_id": parent,
                "host_session_id": host_session,
                "model": str(item.get("model") or ""),
                "raw": {name: 0 for name in _USAGE_COUNTERS},
            }
            records[identity] = record
            order.append(identity)
        elif explicit_parent and not record["parent_sidecar_session_id"]:
            record["parent_sidecar_session_id"] = explicit_parent
        raw = cast(dict[str, int], record["raw"])
        for name in _USAGE_COUNTERS:
            raw[name] = max(raw[name], _usage_counter(raw_usage, name))
        previous_by_host[host_session] = identity

    totals = {name: 0 for name in _USAGE_COUNTERS}
    lineage: list[dict[str, Any]] = []
    has_ancestry = False
    for identity in order:
        record = records[identity]
        raw = cast(dict[str, int], record["raw"])
        parent_id = str(record["parent_sidecar_session_id"] or "")
        parent_record = records.get(parent_id)
        counted = dict(raw)
        semantics = "session_root"
        if parent_id:
            has_ancestry = True
            if parent_record is None:
                # The only observed child already contains the ancestry total,
                # so counting it in full is the least lossy non-duplicating view.
                semantics = "cumulative_parent_unobserved"
            else:
                parent_raw = cast(dict[str, int], parent_record["raw"])
                looks_cumulative = bool(
                    any(parent_raw[name] for name in _TOKEN_COUNTERS)
                    and all(raw[name] >= parent_raw[name] for name in _TOKEN_COUNTERS)
                )
                if looks_cumulative:
                    for name in _TOKEN_COUNTERS:
                        counted[name] = raw[name] - parent_raw[name]
                    semantics = "inherited_cumulative_delta"
                else:
                    # Fake/local runtimes may expose genuinely session-local
                    # counters. Never subtract a larger parent and erase work.
                    semantics = "session_local_or_reset"
        # Requests are deliberately never ancestry-subtracted: ClineRuntime
        # persists that counter in Metis metadata and resets it for each child.
        for name in _USAGE_COUNTERS:
            totals[name] += counted[name]
        lineage.append(
            {
                "sidecar_session_id": identity,
                "parent_sidecar_session_id": parent_id,
                "host_session_id": record["host_session_id"],
                "model": record["model"],
                "semantics": semantics,
                "raw": raw,
                "counted": counted,
            }
        )

    if not records:
        evidence = "not_observed"
    elif missing_identity:
        evidence = "estimated_host_session_model_ancestry"
    elif inferred_parent:
        evidence = "inferred_sidecar_ancestry_deltas"
    elif has_ancestry:
        evidence = "sidecar_ancestry_deltas"
    else:
        evidence = "sidecar_session_maxima"
    return totals, evidence, lineage


async def release_evaluation_artifacts(
    coordinator: Any,
    *,
    run_id: str,
    workspace_root: Path,
) -> dict[str, Any]:
    """Release exactly the sessions and mirrors this evaluation created.

    A repair probe that times out or is cancelled used to leave its sidecar
    session and disposable mirror behind, because cleanup only ran on the
    success path.  This belongs in a ``finally``.

    Scoping is doubled deliberately.  The evaluation's own run id selects the
    sessions, and every one of them must also live under the evaluation's own
    disposable workspace.  A session that fails either test is left completely
    alone: the developer's running Metis app keeps its sidecar and its data
    directory, and an evaluator crash is never allowed to take them down.
    """

    root = workspace_root.resolve()
    released: list[str] = []
    skipped: list[str] = []
    failures: list[str] = []
    try:
        sessions = await coordinator.sessions.for_run(run_id)
    except Exception as error:  # A store that never opened has nothing to release.
        return {
            "released": [],
            "skipped": [],
            "failures": [f"{type(error).__name__}: {error}"],
            "clean": False,
        }
    for session in sessions:
        try:
            owned = Path(session.workspace_path).resolve().is_relative_to(root)
        except (OSError, ValueError):
            owned = False
        if not owned:
            skipped.append(session.id)
            continue
        try:
            await coordinator.abort(session.id, discard=False)
        except Exception:
            # Aborting is best effort; releasing below is the durable step.
            pass
        try:
            await coordinator.release(session.id, CodingSessionState.ABORTED)
            released.append(session.id)
        except Exception as error:
            failures.append(f"{session.id}: {type(error).__name__}: {error}")
    return {
        "released": released,
        "skipped": skipped,
        "failures": failures,
        "clean": not failures,
    }


def summarize_coding_attempts(events: Iterable[TimelineEvent]) -> dict[str, Any]:
    """Coder/repair inference attempts, including ones that never came back.

    A round that reaches ``project.coding_started`` and then times out or
    loses its transport emits no ``project.coding_round`` at all, so every
    token-counting path saw nothing and reported zero. A live repair probe
    did exactly that: the sidecar reached iteration_start, the call never
    returned, and the evidence said "0 tokens" -- which reads as free rather
    than as unknown. An attempt that started is an attempt that happened.
    """

    trace = list(events)
    started = [
        event.payload for event in trace if event.type == "project.coding_started"
    ]
    # A rejected round still came back with a decision and a usage record, so
    # it settles. Only silence counts as unaccounted.
    settled = [
        event.payload
        for event in trace
        if event.type in {"project.coding_round", "project.coding_rejected"}
    ]
    settled_sessions = {
        str(item.get("session_id") or "") for item in settled if item.get("session_id")
    }
    unaccounted = [
        str(item.get("session_id") or "")
        for item in started
        if str(item.get("session_id") or "") not in settled_sessions
    ]
    return {
        "started": len(started),
        "settled": len(settled),
        "attempted_requests": len(started),
        "unaccounted": len(unaccounted),
        "unaccounted_sessions": unaccounted,
        # An attempt with no settled round produced no diff it could prove, so
        # it can never be credited as a repair or as convergence.
        "repair_credit": not unaccounted,
        "convergence_credit": not unaccounted,
    }


def summarize_planner_usage(
    events: Iterable[TimelineEvent],
) -> dict[str, Any]:
    """Planner spend for one run, kept separate from the coder's.

    Planner calls do not run through the sidecar, so for a long time a run's
    "total tokens" silently meant "coder tokens" and a spending ceiling could
    be passed without a planner ever being counted. Providers that expose a
    usage block are summed here; the ones that do not are counted as calls
    with unknown cost, and ``complete`` says so rather than letting a missing
    number read as zero.
    """

    trace = list(events)
    # run.planner_attempt covers every inference attempt, failed ones
    # included; run.planner_usage is the older success-only event and is
    # still read so a retained report from before this change still parses.
    calls = [
        event.payload
        for event in trace
        if event.type in {"run.planner_attempt", "run.planner_usage"}
    ]
    totals = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    unknown = 0
    failed = 0
    for call in calls:
        if call.get("ok") is False:
            failed += 1
        usage = call.get("usage") or {}
        if not usage:
            # A call that cost something the provider did not report. Counting
            # it as zero would turn "unknown" into "free", which is the one
            # reading a spending ceiling must never be given.
            unknown += 1
            continue
        totals["inputTokens"] += _usage_counter(usage, "prompt_tokens")
        totals["outputTokens"] += _usage_counter(usage, "completion_tokens")
        totals["totalTokens"] += _usage_counter(usage, "total_tokens")
    normalizations = [
        event.payload for event in trace if event.type == "project.plan_normalized"
    ]
    synthesized = [
        event.payload for event in trace if event.type == "project.plan_synthesized"
    ]
    return {
        "calls": len(calls),
        "failed_calls": failed,
        "calls_without_usage": unknown,
        # Planner quality: a plan the host had to canonicalize is not the same
        # as one the planner got right, even though both build.
        "normalizations": len(normalizations),
        "normalization_codes": sorted(
            {code for item in normalizations for code in item.get("codes") or []}
        ),
        "normalized_paths": sorted(
            {
                str(path)
                for item in normalizations
                for moved in item.get("moved") or []
                for path in moved.get("paths") or []
            }
        ),
        # A slice list Metis supplied is planner quality too: the plan built,
        # but the planner did not describe how.
        "slices_synthesized": len(synthesized),
        "synthesized_slice_files": sorted(
            {str(path) for item in synthesized for path in item.get("files") or []}
        ),
        "plan_accepted_as_written": not normalizations and not synthesized,
        "complete": bool(calls) and unknown == 0,
        "models": list(
            dict.fromkeys(
                str(call.get("model") or "") for call in calls if call.get("model")
            )
        ),
        "fallbacks": sum(1 for call in calls if call.get("fallback")),
        "corrections": sum(1 for call in calls if call.get("corrected")),
        "failure_stages": sorted(
            {
                str(call.get("failure_stage") or "")
                for call in calls
                if call.get("failure_stage")
            }
        ),
        "failure_reasons": sorted(
            {str(call.get("reason") or "") for call in calls if call.get("reason")}
        ),
        "latency_ms": sum(int(call.get("latency_ms") or 0) for call in calls),
        "usage": totals,
    }


def combined_observed_tokens(
    coding_usage: Mapping[str, Any],
    planner_usage: Mapping[str, Any],
    coding_attempts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Coder + repair + planner tokens, with honesty about what is missing.

    ``partially_unknown`` is the point of this helper: a ceiling compared
    against a total that silently omits an uncounted planner is not a ceiling.
    """

    coder = _usage_counter(coding_usage, "totalTokens")
    planner = _usage_counter(planner_usage.get("usage") or {}, "totalTokens")
    unaccounted = int((coding_attempts or {}).get("unaccounted") or 0)
    return {
        "coder_total_tokens": coder,
        "planner_total_tokens": planner,
        "total_tokens": coder + planner,
        # Unknown, never free: a planner call whose provider reported no usage
        # and a coder round that started and never settled both make the total
        # a floor rather than the bill.
        "partially_unknown": bool(planner_usage.get("calls_without_usage"))
        or bool(unaccounted),
        "planner_calls_without_usage": int(
            planner_usage.get("calls_without_usage") or 0
        ),
        "coder_attempts_without_usage": unaccounted,
    }


def summarize_run(
    *,
    run: Mapping[str, Any],
    events: Iterable[TimelineEvent],
    duration_seconds: float,
    approval: Mapping[str, Any] | None = None,
    required_files: Sequence[str] = REQUIRED_FILES,
) -> dict[str, Any]:
    """Turn the event stream into the few model/build signals that matter."""

    trace = list(events)
    plans = [item.payload for item in trace if item.type == "project.build_planned"]
    direct_contracts = [
        item.payload for item in trace if item.type == "project.direct_contract"
    ]
    revisions = [item.payload for item in trace if item.type == "project.plan_revised"]
    plan = dict(plans[-1]) if plans else {}
    if revisions and revisions[-1].get("files") is not None:
        # A revision changes the executable file contract, not the intent or
        # acceptance scenarios established by the planner. The host emits the
        # merged list, so omitted commitments are already retained here.
        plan["files"] = list(revisions[-1].get("files") or [])
    planned_files = [str(path) for path in plan.get("files") or []]
    planned_scenarios = [str(name) for name in plan.get("scenarios") or []]
    tool_results = [
        item.payload for item in trace if item.type == "project.tool_result"
    ]
    host_scaffold_paths = sorted(
        {
            str(path)
            for item in trace
            if item.type == "project.scaffold_staged"
            for path in item.payload.get("files") or []
            if str(path).strip()
        }
    )
    write_tools = {"create_file", "apply_patch", "replace_lines"}
    legacy_writes = [item for item in tool_results if item.get("tool") in write_tools]
    # A settled sidecar operation can be rejected by the host before its bytes
    # enter the staged overlay (for example, a safe mirror import refusal).
    # It still consumed provider tokens and is still model/session/state truth,
    # but it is not a coding round, write, or repair.  De-duplicate a defensive
    # replay of the same settled operation across accepted/rejected event types
    # by the coordinator's exact operation + sidecar-session identity.
    coding_observations: list[dict[str, Any]] = []
    coding_observation_indexes: dict[tuple[str, str], int] = {}
    for event_index, item in enumerate(trace):
        if item.type not in {"project.coding_round", "project.coding_rejected"}:
            continue
        payload = {**item.payload, "_observation_type": item.type}
        operation_id = str(payload.get("operation_id") or "").strip()
        sidecar_id = str(payload.get("sidecar_session_id") or "").strip()
        if operation_id and sidecar_id:
            identity = (operation_id, sidecar_id)
            prior_index = coding_observation_indexes.get(identity)
            if prior_index is not None:
                # Rejection is the final host disposition and therefore the
                # more useful state record if an older build emitted both.
                if item.type == "project.coding_rejected":
                    coding_observations[prior_index] = payload
                continue
            coding_observation_indexes[identity] = len(coding_observations)
        else:
            # Older events have no operation identity. Keep each one rather
            # than guessing that two provider calls were the same operation.
            payload["_legacy_observation_index"] = event_index
        coding_observations.append(payload)
    coding_rounds = [
        item
        for item in coding_observations
        if item.get("_observation_type") == "project.coding_round"
    ]
    coding_rejections = [
        item
        for item in coding_observations
        if item.get("_observation_type") == "project.coding_rejected"
    ]
    coding_path_writes = [
        {"tool": "external_workspace", "ok": True, "path": str(path)}
        for item in coding_rounds
        for path in item.get("changed_paths") or []
        if str(path).strip()
    ]
    writes = [*legacy_writes, *coding_path_writes]
    successful_writes = [item for item in writes if item.get("ok")]
    refused_writes = [item for item in writes if not item.get("ok")]
    successful_path_events = [
        str(item.get("path") or "").strip()
        for item in successful_writes
        if str(item.get("path") or "").strip()
    ]
    successful_paths = sorted(set(successful_path_events))
    successful_write_indexes = [
        index
        for index, item in enumerate(trace)
        if item.type == "project.tool_result"
        and item.payload.get("tool") in write_tools
        and item.payload.get("ok")
    ]
    successful_write_indexes.extend(
        index
        for index, item in enumerate(trace)
        if item.type == "project.coding_round"
        and any(str(path).strip() for path in item.payload.get("changed_paths") or [])
    )
    verifications = [
        item.payload for item in trace if item.type == "project.staged_verified"
    ]
    verification_indexes = [
        index
        for index, item in enumerate(trace)
        if item.type == "project.staged_verified"
    ]
    last_verification = verifications[-1] if verifications else {}
    model_steps = [item.payload for item in trace if item.type == "project.agent_step"]
    blocked = [item.payload for item in trace if item.type == "project.step_blocked"]
    direction_failures = [
        item.payload for item in trace if item.type == "project.direction_failed"
    ]
    fallbacks = [item.payload for item in trace if item.type == "run.model_fallback"]
    exhausted_models = [
        item.payload for item in trace if item.type == "run.model_exhausted"
    ]
    coding_starts = [
        item.payload for item in trace if item.type == "project.coding_started"
    ]
    coding_usage, usage_evidence, usage_lineage = summarize_coding_usage(
        coding_observations
    )
    planner_usage = summarize_planner_usage(trace)
    coding_attempts = summarize_coding_attempts(trace)
    combined_usage = combined_observed_tokens(
        coding_usage, planner_usage, coding_attempts
    )
    coding_models = list(
        dict.fromkeys(
            str(item.get("model") or "")
            for item in [*coding_starts, *coding_observations]
            if str(item.get("model") or "")
        )
    )
    coding_sessions = list(
        dict.fromkeys(
            str(item.get("session_id") or "")
            for item in [*coding_starts, *coding_observations]
            if str(item.get("session_id") or "")
        )
    )
    coding_states = [
        str(item.get("state") or "")
        for item in coding_observations
        if str(item.get("state") or "")
    ]
    # Preserve raw SDK terminal states, but distinguish a genuine failure from
    # the one terminal-shaped result that the host intentionally knows how to
    # settle.  ClineCore 0.0.72 reports its configured iteration boundary as a
    # failed/error result.  The sidecar marks that exact result as a controlled
    # stop; Metis may then import its non-empty, in-plan overlay and run the
    # independent verifier.  Only that complete event chain is a recovery.  A
    # marker without changed bytes, without the following verifier event, or
    # any ordinary failed/aborted result remains terminal and fails closed.
    unhandled_terminal_rounds: list[tuple[str, str, str]] = []
    fallback_recovered_terminal_states: list[str] = []
    controlled_stop_recovered_terminal_states: list[str] = []
    pending_controlled_stop: tuple[str, str, str] | None = None
    for event in trace:
        if event.type == "project.coding_round":
            # A verifier must settle the controlled stop before another coding
            # round begins; a later verifier cannot retroactively bless it.
            pending_controlled_stop = None
            state = str(event.payload.get("state") or "")
            if state in {"aborted", "failed"}:
                terminal = (
                    str(event.payload.get("session_id") or ""),
                    state,
                    str(event.payload.get("sidecar_session_id") or ""),
                )
                unhandled_terminal_rounds.append(terminal)
                changed_paths = {
                    str(path).strip()
                    for path in event.payload.get("changed_paths") or []
                    if str(path).strip()
                }
                if (
                    state == "failed"
                    and event.payload.get("controlled_stop") is True
                    and str(event.payload.get("controlled_stop_reason") or "")
                    == "max_iterations"
                    and changed_paths
                ):
                    pending_controlled_stop = terminal
            continue
        if event.type == "project.staged_verified" and pending_controlled_stop:
            for index in range(len(unhandled_terminal_rounds) - 1, -1, -1):
                if unhandled_terminal_rounds[index] == pending_controlled_stop:
                    _, recovered_state, _ = unhandled_terminal_rounds.pop(index)
                    controlled_stop_recovered_terminal_states.append(recovered_state)
                    break
            pending_controlled_stop = None
            continue
        if event.type in {"project.coding_failed", "run.model_exhausted"}:
            pending_controlled_stop = None
        if event.type != "run.model_fallback":
            continue
        recovered_state = str(event.payload.get("terminal_state") or "")
        recovered_session = str(event.payload.get("session_id") or "")
        if recovered_state not in {"aborted", "failed"} or not recovered_session:
            continue
        for index in range(len(unhandled_terminal_rounds) - 1, -1, -1):
            terminal_session, terminal_state, _ = unhandled_terminal_rounds[index]
            if (terminal_session, terminal_state) == (
                recovered_session,
                recovered_state,
            ):
                unhandled_terminal_rounds.pop(index)
                fallback_recovered_terminal_states.append(recovered_state)
                break
    terminal_coding_failures = list(
        dict.fromkeys(state for _, state, _ in unhandled_terminal_rounds)
    )
    recorded_status = str(run.get("status") or "")
    status = (
        "failed"
        if terminal_coding_failures
        and recorded_status not in {"failed", "cancelled", "timed_out"}
        else recorded_status
    )
    last_error = str(run.get("last_error") or "")
    if terminal_coding_failures and not last_error:
        last_error = "Cline coding round ended in terminal failure state: " + ", ".join(
            terminal_coding_failures
        )
    required = set(required_files)
    planned = set(planned_files)
    successful_planned_paths = sorted(planned & set(successful_paths))
    direct_contract = dict(direct_contracts[-1]) if direct_contracts else {}
    direct_writable_roots = [
        str(path) for path in direct_contract.get("writable_roots") or []
    ]
    direct_protected_files = {
        str(path) for path in direct_contract.get("protected_files") or []
    }
    direct_unresolved = [str(path) for path in direct_contract.get("unresolved") or []]

    def direct_path_is_writable(path: str) -> bool:
        candidate = PurePosixPath(path)
        for raw_root in direct_writable_roots:
            root = PurePosixPath(raw_root)
            if raw_root == "." or candidate == root or root in candidate.parents:
                return path not in direct_protected_files
        return False

    direct_required_coverage = (
        round(
            len({path for path in required if direct_path_is_writable(path)})
            / len(required),
            4,
        )
        if required
        else 1.0
    )
    scope_mode = (
        "planner_manifest"
        if plans
        else "direct_contract"
        if direct_contracts
        else "unobserved"
    )
    if not successful_writes:
        path_evidence = "complete"
    elif len(successful_path_events) == len(successful_writes):
        path_evidence = "complete"
    elif successful_paths:
        path_evidence = "partial"
    else:
        path_evidence = "unavailable"
    blocked_reason = str((approval or {}).get("blocked_reason") or "")
    return {
        "run_id": str(run.get("id") or ""),
        # ``recorded_status`` preserves the database evidence when evaluating
        # an older/inconsistent run. The effective status fails closed so an
        # aborted SDK round can never look like a successful attempt merely
        # because a host bug previously committed RunStatus.COMPLETED.
        "status": status,
        "recorded_status": recorded_status,
        "last_error": last_error,
        "duration_seconds": round(duration_seconds, 3),
        "plan": {
            "present": bool(plans),
            "intent": str(plan.get("intent") or ""),
            "scope": str(plan.get("scope") or ""),
            "files": planned_files,
            "scenarios": planned_scenarios,
            "revisions": len(revisions),
            "required_file_coverage": round(len(required & planned) / len(required), 4),
            "missing_required_files": sorted(required - planned),
        },
        # One durable admission/scope shape for both build paths. The planner
        # manifest remains honest above; a direct run is not made to look as if
        # it called a planner. Evaluation, approval and qualification consume
        # this contract instead.
        "scope_contract": {
            "present": bool(plans or direct_contracts),
            "mode": scope_mode,
            "source_event": (
                "project.build_planned"
                if plans
                else "project.direct_contract"
                if direct_contracts
                else ""
            ),
            "admitted": bool(plans) or bool(direct_contracts and not direct_unresolved),
            "requested_files": list(required_files),
            "authorized_files": (
                planned_files
                if plans
                else list(required_files)
                if direct_contracts
                else []
            ),
            "required_file_coverage": (
                round(len(required & planned) / len(required), 4)
                if plans and required
                else direct_required_coverage
                if direct_contracts
                else 0.0
            ),
            "writable_roots": direct_writable_roots,
            "protected_files": sorted(direct_protected_files),
            "unresolved": direct_unresolved,
            "approval_required": bool(direct_contract.get("approval_required", True)),
            "check_budget": int(direct_contract.get("check_budget") or 0),
        },
        "tool_calls": len(tool_results),
        "model_steps": len(model_steps) + len(coding_rounds),
        "writes": {
            "attempted": len(writes),
            "successful": len(successful_writes),
            "refused": len(refused_writes),
            "unique_successful": len(successful_paths),
            "unique_successful_paths": successful_paths,
            "planned_successful": len(successful_planned_paths),
            "missing_planned_paths": sorted(planned - set(successful_paths)),
            "host_scaffold_paths": host_scaffold_paths,
            "path_evidence": path_evidence,
        },
        "reads": len(tool_results) - len(legacy_writes),
        # Planner spend, reported apart from the coder's, and the combined
        # figure a spending ceiling must actually be compared against.
        "planner": planner_usage,
        "coding_attempts": coding_attempts,
        "combined_usage": combined_usage,
        "directions": sum(item.type == "project.direction" for item in trace),
        "direction_failures": len(direction_failures),
        "fallbacks": fallbacks,
        "model_exhaustions": exhausted_models,
        "coding_engine": {
            "observed": (
                str(coding_starts[-1].get("engine") or "clinecore")
                if coding_starts
                else "clinecore"
                if coding_observations
                else "not_observed"
            ),
            "sessions": coding_sessions,
            "rounds": len(coding_rounds),
            "rejections": len(coding_rejections),
            "rejection_reasons": [
                str(item.get("reason") or item.get("rejection_reason") or "")[:1_000]
                for item in coding_rejections
                if str(item.get("reason") or item.get("rejection_reason") or "")
            ],
            "models": coding_models,
            "states": coding_states,
            "terminal_failure": bool(terminal_coding_failures),
            "terminal_failure_states": terminal_coding_failures,
            "fallback_recovered_terminal_states": (fallback_recovered_terminal_states),
            "controlled_stop_recovered_terminal_states": list(
                dict.fromkeys(controlled_stop_recovered_terminal_states)
            ),
            "usage": coding_usage,
            "usage_evidence": usage_evidence,
            "requests_semantics": "sidecar_turns_not_provider_iterations",
            "usage_lineage": usage_lineage,
        },
        "blocked_steps": blocked,
        "verification": {
            "attempts": len(verifications),
            "after_last_successful_write": bool(
                successful_write_indexes
                and verification_indexes
                and verification_indexes[-1] > successful_write_indexes[-1]
            ),
            "blocking": int(last_verification.get("errors") or 0),
            "warnings": int(last_verification.get("warnings") or 0),
            "checks_run": int(last_verification.get("ran") or 0),
            "notes": list(last_verification.get("notes") or []),
            "findings": list(last_verification.get("findings") or [])[:12],
        },
        "approval": {
            "offered": approval is not None,
            "blocked": bool(blocked_reason),
            "blocked_reason": blocked_reason[:4_000],
        },
    }


def _ratio(numerator: float, denominator: float) -> float:
    return max(0.0, min(1.0, numerator / denominator if denominator else 0.0))


def authorized_scope_paths(attempt: Mapping[str, Any]) -> list[str]:
    """Return the exact evaluator scope without inventing a planner manifest."""

    plan = attempt.get("plan") or {}
    planned = [str(path) for path in plan.get("files") or [] if str(path)]
    if planned:
        return planned
    contract = attempt.get("scope_contract") or {}
    if str(contract.get("mode") or "") != "direct_contract":
        return []
    return [str(path) for path in contract.get("authorized_files") or [] if str(path)]


def repair_attempt_evidence(
    attempt: Mapping[str, Any], *, planned_paths: Iterable[str]
) -> tuple[bool, bool, bool]:
    """Return causal carry, in-plan mutation, and post-mutation verification.

    Copying an undecided overlay into a follow-up proves continuity, not a
    repair.  Credit additionally requires an observed model step, at least one
    exact successful path inside the host plan, and verification after that
    final write.  A no-op or a mirror import rejected for an out-of-plan path
    therefore cannot masquerade as convergence.
    """

    continuation = attempt.get("repair_continuation") or {}
    writes = attempt.get("writes") or {}
    verification = attempt.get("verification") or {}
    allowed = {str(path).strip() for path in planned_paths if str(path).strip()}
    changed = {
        str(path).strip()
        for path in writes.get("unique_successful_paths") or []
        if str(path).strip()
    }
    continuation_verified = bool(continuation.get("verified"))
    execution_verified = bool(
        int(attempt.get("model_steps") or 0) > 0
        and int(writes.get("successful") or 0) > 0
        and changed
        and changed <= allowed
    )
    post_write_verification_verified = bool(
        int(verification.get("attempts") or 0) > 0
        and verification.get("after_last_successful_write") is True
    )
    return (
        continuation_verified,
        execution_verified,
        post_write_verification_verified,
    )


def score_evaluation(
    attempts: list[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    *,
    max_steps: int,
) -> dict[str, Any]:
    """Score one initial build plus its bounded repair turns, transparently."""

    if not attempts:
        return {
            "total": 0.0,
            "verdict": "no run",
            "categories": {},
            "weights": {},
        }
    first = attempts[0]
    # A repair may replace the prior result only when the evidence establishes
    # the complete causal chain: it inherited the immediately preceding staged
    # changeset, the model ran and successfully changed a file, and verification
    # ran after the final successful change. Merely copying the overlay is not a
    # repair. In particular, an exhausted provider can otherwise leave an empty
    # ``blocking=0`` summary that falsely erases the prior run's blockers.
    scoreable_attempts = [first]
    repair_attempts = attempts[1:]
    authorized_paths = authorized_scope_paths(first)
    repair_evidence = [
        repair_attempt_evidence(
            attempt,
            planned_paths=authorized_paths,
        )
        for attempt in repair_attempts
    ]
    for attempt, evidence in zip(repair_attempts, repair_evidence, strict=True):
        if not all(evidence):
            break
        scoreable_attempts.append(attempt)
    final = scoreable_attempts[-1]
    repair_continuation_verified = all(item[0] for item in repair_evidence)
    repair_execution_verified = bool(repair_evidence) and all(
        item[1] for item in repair_evidence
    )
    post_repair_verification_verified = bool(repair_evidence) and all(
        item[2] for item in repair_evidence
    )
    repair_chain_verified = not repair_evidence or all(
        all(item) for item in repair_evidence
    )
    plan = first.get("plan") or {}
    scope_contract = first.get("scope_contract") or {}
    if str(scope_contract.get("mode") or "") == "direct_contract":
        planning_ratio = (
            0.25 * float(bool(scope_contract.get("present")))
            + 0.15 * float(bool(scope_contract.get("admitted")))
            + 0.10
            * float(
                "."
                in {str(path) for path in scope_contract.get("writable_roots") or []}
            )
            + 0.35 * float(scope_contract.get("required_file_coverage") or 0.0)
            + 0.15
            * float(
                bool(scope_contract.get("approval_required"))
                and int(scope_contract.get("check_budget") or 0) > 0
            )
        )
    else:
        planning_ratio = (
            0.25 * float(bool(plan.get("present")))
            + 0.15 * float(plan.get("intent") == "build")
            + 0.10 * float(plan.get("scope") == "whole_app")
            + 0.35 * float(plan.get("required_file_coverage") or 0.0)
            + 0.15 * _ratio(len(plan.get("scenarios") or []), 5)
        )

    writes = first.get("writes") or {}
    planned_count = max(1, len(authorized_paths))
    path_evidence = str(writes.get("path_evidence") or "unavailable")
    # A path-less success is evidence that *something* was written, never that
    # another planned file was completed. Older events remain readable, but
    # cannot turn repeated edits to one unknown target into full manifest
    # coverage. Partial traces receive credit only for the unique planned paths
    # they actually identify.
    if str(scope_contract.get("mode") or "") == "direct_contract":
        written_paths = {
            str(path) for path in writes.get("unique_successful_paths") or []
        }
        completion_count = float(len(set(authorized_paths) & written_paths))
    else:
        completion_count = float(writes.get("planned_successful") or 0)
    write_completion = _ratio(completion_count, planned_count)
    attempted = int(writes.get("attempted") or 0)
    efficiency = (
        1.0 - _ratio(float(writes.get("refused") or 0), attempted) if attempted else 0.0
    )
    boundedness = (
        1.0 if attempted and int(first.get("model_steps") or 0) <= max_steps else 0.0
    )
    execution_ratio = 0.65 * write_completion + 0.25 * efficiency + 0.10 * boundedness

    verification = final.get("verification") or {}
    verified = float(int(verification.get("attempts") or 0) > 0)
    final_approval = final.get("approval") or {}
    clean = float(
        verified
        and bool(final_approval.get("offered"))
        and int(verification.get("blocking") or 0) == 0
        and not bool(final_approval.get("blocked"))
    )
    verification_ratio = 0.25 * verified + 0.75 * clean

    initial_blocking = int((first.get("verification") or {}).get("blocking") or 0)
    final_blocking = int(verification.get("blocking") or 0)
    if initial_blocking == 0:
        convergence_ratio = clean
    else:
        convergence_ratio = _ratio(initial_blocking - final_blocking, initial_blocking)
    # An inference that started and never settled left no diff behind. Whatever
    # the blocking count looks like afterwards, this run did not demonstrate
    # that a model converged on anything.
    unsettled_inference = not all(
        (attempt.get("coding_attempts") or {}).get("convergence_credit", True)
        for attempt in scoreable_attempts
    )
    if unsettled_inference:
        convergence_ratio = 0.0

    acceptance_total = int(acceptance.get("checks_total") or 0)
    acceptance_passed = int(acceptance.get("checks_passed") or 0)
    acceptance_ratio = (
        _ratio(acceptance_passed, acceptance_total)
        if (
            acceptance.get("available")
            and acceptance.get("passed")
            and acceptance_total
        )
        else 0.0
    )
    if not repair_chain_verified:
        # A passing sandbox probe without the complete causal repair chain is
        # useful diagnosis, but it is not evidence that the model repaired and
        # verified the staged changeset.
        acceptance_ratio = 0.0

    weights = {
        "planning": 20.0,
        "execution": 25.0,
        "verification": 20.0,
        "repair_convergence": 15.0,
        "acceptance": 20.0,
    }
    ratios = {
        "planning": planning_ratio,
        "execution": execution_ratio,
        "verification": verification_ratio,
        "repair_convergence": convergence_ratio,
        "acceptance": acceptance_ratio,
    }
    categories = {key: round(weights[key] * ratios[key], 2) for key in weights}
    total = round(sum(categories.values()), 2)
    if not acceptance.get("available"):
        verdict = "required acceptance unavailable"
    elif not acceptance.get("passed"):
        verdict = "failed required acceptance"
    elif path_evidence != "complete":
        verdict = "insufficient write-path evidence"
    elif total >= 85:
        verdict = "ready on this scenario"
    elif total >= 70:
        verdict = "promising but not dependable"
    elif total >= 50:
        verdict = "incomplete"
    else:
        verdict = "failed this scenario"
    return {
        "total": total,
        "verdict": verdict,
        "categories": categories,
        "weights": weights,
        "signals": {
            "write_completion": round(write_completion, 4),
            "write_completion_evidence": (
                "unique_planned_paths"
                if path_evidence == "complete"
                else "partial_unique_planned_paths"
                if path_evidence == "partial"
                else "unavailable"
            ),
            "write_path_evidence_complete": path_evidence == "complete",
            "write_efficiency": round(efficiency, 4),
            "initial_blocking": initial_blocking,
            "final_blocking": final_blocking,
            "blocking_reduction": round(convergence_ratio, 4),
            "convergence_credit_withheld": unsettled_inference,
            "acceptance_ratio": round(acceptance_ratio, 4),
            "repair_attempts": max(0, len(attempts) - 1),
            "verified_repair_attempts": max(0, len(scoreable_attempts) - 1),
            "repair_continuation_verified": repair_continuation_verified,
            "repair_execution_verified": repair_execution_verified,
            "post_repair_verification_verified": (post_repair_verification_verified),
            "repair_chain_verified": repair_chain_verified,
            "required_acceptance_passed": bool(
                acceptance.get("available") and acceptance.get("passed")
            ),
            "scope_contract_mode": str(
                scope_contract.get("mode") or "planner_manifest"
            ),
        },
    }


def release_gate_passes(report: Mapping[str, Any], threshold: float) -> bool:
    """Whether a report has enough evidence to act as a release gate.

    The numeric score is useful diagnosis, but it is not a substitute for the
    required product workflow, materialization, or exact write attribution.
    Keeping those conditions in one shared predicate prevents the CLI from
    quietly treating an 80-point run with zero acceptance credit as a pass.
    """

    score = report.get("score") or {}
    acceptance = report.get("acceptance") or {}
    qualification = report.get("qualification") or {}
    attempts = list(report.get("attempts") or [])
    complete_path_evidence = bool(attempts) and all(
        str((attempt.get("writes") or {}).get("path_evidence") or "") == "complete"
        for attempt in attempts
    )

    def has_terminal_coding_failure(attempt: Mapping[str, Any]) -> bool:
        engine = attempt.get("coding_engine") or {}
        if not isinstance(engine, Mapping):
            return False
        # New summaries correlate an availability fallback to the exact
        # terminal session/state. Older reports have no such proof, so retain
        # the conservative raw-state check for backward-compatible fail-close.
        if "terminal_failure" in engine:
            return bool(engine.get("terminal_failure"))
        return any(
            str(state) in {"aborted", "failed"} for state in engine.get("states") or []
        )

    terminal_coding_failure = any(
        has_terminal_coding_failure(attempt) for attempt in attempts
    )
    try:
        total = float(score.get("total") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        total >= threshold
        and report.get("approved")
        and acceptance.get("available")
        and acceptance.get("passed")
        and complete_path_evidence
        and not terminal_coding_failure
        and (not qualification or qualification.get("passed") is True)
    )


async def run_meridian_acceptance(
    project_root: Path,
    settings: Settings,
    *,
    staged: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Exercise approved or exact pending bytes in the networkless container."""

    paths = [
        str(path.relative_to(project_root))
        for path in sorted(project_root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and ".metis" not in path.relative_to(project_root).parts
    ]
    staged_requirements = (staged or {}).get("requirements.txt", {}).get("content")
    if isinstance(staged_requirements, str):
        requirements = staged_requirements
    else:
        try:
            requirements = (project_root / "requirements.txt").read_text(
                encoding="utf-8"
            )
        except OSError:
            requirements = ""
    try:
        verifier_overlay = acceptance_staged_overlay(
            staged,
            probe_path="metis_eval_acceptance.py",
            probe_source=MERIDIAN_ACCEPTANCE_SOURCE,
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": f"could not materialize exact pending acceptance overlay: {error}",
            "passed": False,
            "checks_total": 1,
            "checks_passed": 0,
            "sandbox_checks_total": 0,
            "sandbox_checks_passed": 0,
            "required_probe": {},
            "blocking_findings": [],
            "warnings": [],
            "routes": [],
            "skipped_modules": [],
        }
    service = ProjectSandboxService(settings)
    try:
        outcome = await service.verify(
            root=project_root,
            staged=verifier_overlay,
            project_paths=paths,
            requirements=requirements,
        )
    finally:
        await service.release_machine(reason="capability evaluation complete")

    required_check = next(
        (
            check
            for check in outcome.checks
            if check.get("name") == "import metis_eval_acceptance"
        ),
        None,
    )
    sandbox_checks_total = len(outcome.checks)
    sandbox_checks_passed = sum(bool(check.get("ok")) for check in outcome.checks)
    blocking = [
        finding for finding in outcome.findings if finding.get("severity") != "warning"
    ]
    required_probe_passed = bool(required_check and required_check.get("ok"))
    passed = bool(outcome.available and required_probe_passed and not blocking)
    if not outcome.available:
        reason = outcome.reason or "the sandbox acceptance probe was unavailable"
    elif required_check is None:
        reason = "the sandbox did not execute the required Meridian probe"
    elif not required_probe_passed:
        reason = str(
            required_check.get("detail") or "the required Meridian probe failed"
        )
    elif blocking:
        reason = "the sandbox reported blocking findings outside the required probe"
    else:
        reason = ""
    return {
        "available": outcome.available,
        "reason": reason,
        "passed": passed,
        # These are deliberately the one required product probe, not every
        # generic module import and route smoke check the sandbox happened to
        # run beside it. A report saying "19/20 acceptance checks" after the
        # fail-fast probe stopped at its first assertion is causally false.
        "checks_total": 1,
        "checks_passed": int(passed),
        "sandbox_checks_total": sandbox_checks_total,
        "sandbox_checks_passed": sandbox_checks_passed,
        "required_probe": dict(required_check or {}),
        "blocking_findings": blocking,
        "warnings": [
            finding
            for finding in outcome.findings
            if finding.get("severity") == "warning"
        ],
        "routes": outcome.routes,
        "skipped_modules": outcome.skipped_modules,
    }


def run_meridian_acceptance_sync(
    project_root: Path,
    settings: Settings,
    *,
    staged: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Synchronous entrypoint for the command-line evaluation driver."""

    return asyncio.run(run_meridian_acceptance(project_root, settings, staged=staged))


def event_dicts(events: Iterable[TimelineEvent]) -> list[dict[str, Any]]:
    """JSON-ready trace helper for reports that opt into retaining evidence."""

    return [asdict(event) for event in events]
