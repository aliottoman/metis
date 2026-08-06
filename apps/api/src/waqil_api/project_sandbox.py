"""Run a staged project inside the reviewed container and report what breaks.

The wiring gate proves that a changeset's files refer to each other correctly;
this proves they actually run. It materializes the overlay into a throwaway
copy, hands that copy to the Podman wrapper, and turns the sandbox's checks into
the same ``{path, error, severity}`` findings the rest of the build loop speaks.

No project code is executed on the host, ever: the wrapper at
``infra/sandbox/project-verify/run_project_verify.py`` owns every container flag
and there is no host fallback. When the sandbox cannot run — Podman is down, the
image is missing, the project is too large — verification degrades to the static
checks and says why, because a gate that quietly passes is worse than no gate.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .project_wiring import ERROR, WARNING, declared_distributions, module_name_for

# Directories that are never part of what a build should be judged on.
_SKIP_DIRECTORIES = frozenset(
    {".git", ".hg", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".metis"}
)

# Where a project's entrypoint conventionally lives, best first. Everything the
# turn staged is imported too; this only decides what gets imported first, and
# therefore which module is asked for the application object.
_ENTRYPOINT_HINTS = ("app/main.py", "main.py", "src/main.py", "app.py", "app/api.py")

_MAX_INPUT_FILES = 2_048
_MAX_INPUT_BYTES = 32_000_000

# Sandbox error codes that mean "the sandbox did not run", as opposed to "the
# project is broken". These degrade to the static gate instead of being reported
# to the model, which cannot do anything about any of them.
_UNAVAILABLE_CODES = frozenset(
    {
        "PODMAN_UNAVAILABLE",
        "PODMAN_START_FAILED",
        "PODMAN_INFO_INVALID",
        "ROOTLESS_REQUIRED",
        "UNAPPROVED_IMAGE",
        "POLICY_INVALID",
        "POLICY_UNREADABLE",
        "INPUT_TOO_LARGE",
        "INPUT_UNAVAILABLE",
        "INPUT_UNSAFE",
        "INVALID_TOOL_RESULT",
        "REQUEST_TOO_LARGE",
        "REQUEST_INVALID",
    }
)


@dataclass(frozen=True, slots=True)
class SandboxOutcome:
    """What running the changeset proved, or why nothing could be proved."""

    available: bool = False
    reason: str = ""
    findings: list[dict[str, str]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    routes: list[dict[str, str]] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)


def _relative_parts_are_visible(relative: Path) -> bool:
    """Whether a project path belongs in the copy the sandbox will import."""
    return not any(part in _SKIP_DIRECTORIES for part in relative.parts)


def materialize(
    root: Path,
    staged: dict[str, dict[str, Any]],
    destination: Path,
    *,
    max_files: int = _MAX_INPUT_FILES,
    max_bytes: int = _MAX_INPUT_BYTES,
) -> int:
    """Copy the project and lay the staged overlay on top of the copy.

    The result is what the user's disk would look like if they approved the
    changeset — which is the only thing worth verifying. Returns the file count,
    or raises when the project is past what the sandbox policy accepts.
    """
    destination.mkdir(parents=True, exist_ok=True)
    files = 0
    total = 0
    if root.is_dir():
        for source in sorted(root.rglob("*")):
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(root)
            if not _relative_parts_are_visible(relative):
                continue
            size = source.stat().st_size
            files += 1
            total += size
            if files > max_files or total > max_bytes:
                raise ValueError("project exceeds the sandbox input limits")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for path in sorted(staged):
        relative = Path(path)
        if relative.is_absolute() or not _relative_parts_are_visible(relative):
            continue
        content = str(staged[path].get("content", ""))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        files += 1
        total += len(content.encode("utf-8"))
        if files > max_files or total > max_bytes:
            raise ValueError("project exceeds the sandbox input limits")
    return files


def import_order(paths: list[str], staged: dict[str, dict[str, Any]], limit: int) -> list[str]:
    """The modules to import, entrypoints first, then everything staged.

    Entrypoints come first so the application object is found on the module that
    is meant to expose it. The rest of the changeset follows, because a module
    nothing imports yet can still be the one that will not compile.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in (*_ENTRYPOINT_HINTS, *sorted(staged), *sorted(paths)):
        if candidate not in staged and candidate not in paths:
            continue
        dotted = module_name_for(candidate)
        if dotted is None or dotted in seen or dotted.endswith("__init__"):
            continue
        seen.add(dotted)
        ordered.append(dotted)
        if len(ordered) >= limit:
            break
    return ordered


def _finding(path: str, error: str, severity: str = ERROR) -> dict[str, str]:
    """One reportable defect, in the shape every gate in the loop returns."""
    return {"path": path, "error": error, "severity": severity}


def _where_path(check: dict[str, Any], fallback: str) -> tuple[str, str]:
    """The project file a failed check points at, and how to say where in it."""
    where = str(check.get("where") or "")
    if not where:
        return fallback, ""
    path, _, location = where.partition(" line ")
    return path or fallback, f" (line {location})" if location else ""


class ProjectSandboxService:
    """Executes a staged changeset in the container and classifies the result."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._machine_checked = False
        # Only a machine Metis started is a machine Metis may stop. Podman is a
        # shared tool: if it was already up it belongs to whatever else the user
        # is doing, and quietly shutting that down is not ours to do.
        self._machine_started_here = False
        self._last_used = time.monotonic()

    async def verify(
        self,
        *,
        root: Path,
        staged: dict[str, dict[str, Any]],
        project_paths: list[str] | None = None,
        requirements: str = "",
        scenarios: list[dict[str, Any]] | None = None,
    ) -> SandboxOutcome:
        """Import the staged project in the sandbox and report what failed."""
        if self.settings.model_backend == "deterministic":
            # The hermetic test backend must not reach for a container: a suite
            # whose result depends on whether a VM happens to be running is not
            # a suite. The empty reason keeps it off approval cards too.
            return SandboxOutcome(reason="")
        if not self.settings.project_sandbox_enabled:
            return SandboxOutcome(reason="the execution sandbox is turned off")
        runner = self.settings.project_sandbox_runner
        if not runner.is_file():
            return SandboxOutcome(reason=f"sandbox runner unavailable: {runner.name}")
        paths = list(project_paths or [])
        modules = import_order(paths, staged, self.settings.project_sandbox_max_modules)
        if not modules:
            return SandboxOutcome(
                available=True, reason="no Python module in this changeset to import"
            )
        return await asyncio.to_thread(
            self._run, runner, root, staged, paths, requirements, modules,
            list(scenarios or []),
        )

    def _run(
        self,
        runner: Path,
        root: Path,
        staged: dict[str, dict[str, Any]],
        paths: list[str],
        requirements: str,
        modules: list[str],
        scenarios: list[dict[str, Any]] | None = None,
    ) -> SandboxOutcome:
        """Materialize, invoke the wrapper, and read back its envelope."""
        self._last_used = time.monotonic()
        self.settings.run_dir.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="metis-verify-", dir=str(self.settings.run_dir)))
        project = workspace / "project"
        try:
            try:
                materialize(root, staged, project)
            except (OSError, ValueError) as exc:
                return SandboxOutcome(reason=f"could not stage the project for verification: {exc}")
            if self.settings.project_sandbox_autostart:
                self._ensure_machine()
            request = json.dumps(
                {
                    "schema_version": "1",
                    "modules": modules,
                    "app_attribute": "app",
                    "scenarios": list(scenarios or []),
                }
            ).encode("utf-8")
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "--project-dir",
                        str(project),
                        "--image",
                        self.settings.project_sandbox_image,
                    ],
                    input=request,
                    capture_output=True,
                    timeout=self.settings.project_sandbox_timeout_seconds,
                    check=False,
                )
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                return SandboxOutcome(
                    reason=f"sandbox runner did not complete: {type(exc).__name__}"
                )
            try:
                envelope = json.loads(completed.stdout.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                detail = completed.stderr.decode("utf-8", errors="replace")[:300]
                return SandboxOutcome(reason=f"sandbox returned no result: {detail}")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        return classify_envelope(
            envelope, staged=staged, project_paths=paths, requirements=requirements
        )

    def _ensure_machine(self) -> None:
        """Start the Podman VM once per process if it is not already running."""
        if self._machine_checked:
            return
        self._machine_checked = True
        try:
            probe = subprocess.run(
                ["podman", "info", "--format", "{{.Host.Security.Rootless}}"],
                capture_output=True,
                timeout=20,
                check=False,
            )
            if probe.returncode == 0:
                return
            started = subprocess.run(
                ["podman", "machine", "start"], capture_output=True, timeout=120, check=False
            )
            self._machine_started_here = started.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            # A VM that will not start is not an error here: the wrapper reports
            # it as unavailable and the loop degrades to the static gate.
            return

    @property
    def idle_seconds(self) -> float:
        """How long since a verification last used the sandbox."""
        return time.monotonic() - self._last_used

    async def release_if_idle(self, after_seconds: float) -> bool:
        """Stop the VM once nothing has verified for a while."""
        if after_seconds <= 0 or self.idle_seconds < after_seconds:
            return False
        return await self.release_machine(reason="idle")

    async def release_machine(self, *, reason: str = "shutdown") -> bool:
        """Give the VM's memory back, but only if Metis is the one who took it."""
        if not self._machine_started_here:
            return False
        self._machine_started_here = False
        self._machine_checked = False
        try:
            stopped = await asyncio.to_thread(
                subprocess.run,
                ["podman", "machine", "stop"],
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return stopped.returncode == 0


# The error shapes of configuration read at import time. Matched against the
# import check's exception type and text; request-time failures never land
# here, so a route that correctly 503s on missing config stays advisory.
_CONFIG_SHAPED = re.compile(
    r"ConfigError|EnvironmentError|is not set|Field required|"
    r"validation error for|environment variable|\bOCI_[A-Z][A-Z0-9_]*|\bWAQIL_[A-Z][A-Z0-9_]*"
)


def classify_envelope(
    envelope: dict[str, Any],
    *,
    staged: dict[str, dict[str, Any]],
    project_paths: list[str] | None = None,
    requirements: str = "",
) -> SandboxOutcome:
    """Turn the sandbox's checks into findings the build loop can act on.

    The distinction that matters is between a defect the model can fix and a
    limit of the verifier. A package the project declares but the offline image
    does not carry is the second kind: reporting it as a defect would send the
    model off to "fix" code that is perfectly correct.
    """
    paths = list(project_paths or [])
    if envelope.get("status") != "succeeded":
        error = envelope.get("error") or {}
        code = str(error.get("code") or "SANDBOX_FAILED")
        message = str(error.get("message") or "the sandbox did not run")
        if code in _UNAVAILABLE_CODES:
            return SandboxOutcome(reason=f"{message} ({code})")
        if code == "SANDBOX_TIMEOUT":
            return SandboxOutcome(
                available=True,
                checks=[],
                findings=[
                    _finding(
                        _first_entrypoint(staged),
                        f"the project did not finish importing inside the sandbox: {message}",
                    )
                ],
            )
        return SandboxOutcome(reason=f"{message} ({code})")

    checks = list(envelope.get("checks") or [])
    module_paths = {
        module_name_for(path): path for path in (*staged, *paths) if module_name_for(path)
    }
    local_roots = {name.split(".")[0] for name in module_paths if name}
    declared = declared_distributions(requirements)
    fallback = _first_entrypoint(staged)

    findings: list[dict[str, str]] = []
    skipped: list[str] = []
    for check in checks:
        if check.get("ok"):
            continue
        missing = str(check.get("missing_module") or "")
        path, location = _where_path(check, fallback)
        if missing:
            root = missing.split(".")[0]
            if root in local_roots:
                findings.append(
                    _finding(
                        path,
                        f"imports {missing}{location}, but no file in this project "
                        "provides that module",
                    )
                )
            elif root.replace("_", "-").casefold() in declared:
                # Declared, but the offline verify image does not carry it.
                # The code may be perfectly correct, so this is reported to
                # the user and never sent back to the model as a defect.
                skipped.append(root)
            else:
                findings.append(
                    _finding(
                        path,
                        f"imports {root}{location}, which is neither declared in the "
                        "project's requirements nor available to the verifier",
                    )
                )
            continue
        detail = str(check.get("detail") or "failed")
        name = str(check.get("name") or "check")
        error_type = str(check.get("error_type") or "")
        if check.get("kind") == "import" and _CONFIG_SHAPED.search(
            f"{error_type}: {detail}"
        ):
            # Under the lazy-config contract (the reference doc and appkit both
            # teach it) an import that dies reading configuration is a code
            # defect, not an environment gap: the app must import and serve its
            # health route with no environment at all.
            findings.append(
                _finding(
                    path,
                    f"{name} raised at import over missing configuration: {detail}"
                    f"{location}. Read settings lazily at use time "
                    "(appkit.config.require) so the app imports and serves with "
                    "no environment, and the missing value fails only the "
                    "feature that needs it",
                )
            )
            continue
        if check.get("kind") == "acceptance":
            # The spec's own claim, replayed. A crash or wrong status class is
            # a provable defect; a response that merely fails to *mention*
            # something stays advisory, because the scenario — not the app —
            # may be the wrong party, and false blocks cost more than they
            # catch.
            findings.append(
                _finding(
                    path,
                    f"{name} failed: {detail}{location}",
                    WARNING if check.get("content_miss") else ERROR,
                )
            )
        elif check.get("kind") == "request":
            findings.append(
                _finding(path, f"{name} failed when the project ran: {detail}{location}")
            )
        else:
            findings.append(_finding(path, f"{name} failed: {detail}{location}"))

    if skipped:
        findings.append(
            _finding(
                fallback,
                "could not be fully verified: "
                + ", ".join(sorted(set(skipped)))
                + " is declared but not installed in the offline verify image",
                WARNING,
            )
        )
    return SandboxOutcome(
        available=True,
        findings=findings,
        checks=checks,
        routes=list(envelope.get("routes") or []),
        skipped_modules=sorted(set(skipped)),
    )


def _first_entrypoint(staged: dict[str, dict[str, Any]]) -> str:
    """A sensible file to attribute a whole-project failure to."""
    for hint in _ENTRYPOINT_HINTS:
        if hint in staged:
            return hint
    for path in sorted(staged):
        if path.endswith(".py"):
            return path
    return next(iter(sorted(staged)), "the project")
