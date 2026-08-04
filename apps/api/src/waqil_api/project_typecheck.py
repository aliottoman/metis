"""Ruff and mypy over a staged changeset, resolved against the real packages.

The hand-written rungs check the code against itself: it parses, its imports
point at files the project has. Neither can know that `AsyncOpenAI` takes no
`auth` argument, because that fact lives in the installed package. A type
checker resolving against the same site-packages the project will run on knows
it immediately — and that one defect was invented independently by a frontier
model and a local one, and survived every gate until a reference document was
written by hand to prevent it.

Neither tool executes the code it reads, which is what makes this safe on a file
the model wrote a moment ago. Both are run with configuration discovery turned
off: a model-authored `pyproject.toml` can register a mypy plugin, and a plugin
is imported, so honouring project config here would hand the changeset exactly
the code execution the sandbox exists to contain.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ERROR = "error"
WARNING = "warning"

# Rules precise enough to withhold the Approve button. Every one means "this
# name or call cannot work as written", not "this is untidy" — a checker's
# opinion about style must never block a build.
_BLOCKING_MYPY_CODES = frozenset(
    {
        "call-arg",       # a keyword or positional the callee does not accept
        "attr-defined",   # a name that is not in the module or object
        "name-defined",   # an undefined name
        "call-overload",  # no signature matches the call
        "return-value",   # returns something the annotation forbids
        "assignment",     # assigns something the annotation forbids
    }
)
# Advisory: real, but routinely true of correct freshly written code.
_ADVISORY_MYPY_CODES = frozenset({"arg-type", "union-attr", "index", "operator"})

_BLOCKING_RUFF_CODES = frozenset({"F821", "F811", "F822", "E999"})
_ADVISORY_RUFF_CODES = frozenset({"F401", "F841"})

_MYPY_LINE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?:\d+:)?\s*(?P<severity>error|note|warning):\s*"
    r"(?P<message>.*?)(?:\s*\[(?P<code>[a-z-]+)\])?$"
)


def _finding(path: str, error: str, severity: str) -> dict[str, str]:
    """One reportable defect, in the shape every other rung returns."""
    return {"path": path, "error": error, "severity": severity}


def _relative_to(reported: str, root: Path) -> str:
    """A checker's path as a project-relative one.

    Both tools report absolute paths in some cases, and the temp tree is behind
    a symlink on macOS (/var → /private/var), so string-stripping the root is
    not enough on its own.
    """
    if not reported:
        return ""
    candidate = Path(reported)
    for base in {root, root.resolve()}:
        try:
            return str(candidate.relative_to(base))
        except ValueError:
            continue
    return reported


def _materialize(staged: dict[str, dict[str, Any]], root: Path) -> list[str]:
    """Write the overlay to a throwaway tree and return its Python files."""
    written: list[str] = []
    for relative, entry in staged.items():
        target = root / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(entry.get("content", "")), encoding="utf-8")
        except OSError:
            continue
        if target.suffix.lower() in {".py", ".pyi"}:
            written.append(relative)
    return written


async def _run(
    command: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None
) -> tuple[int, str]:
    """Run one checker, returning its exit code and combined output."""
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env={**os.environ, **(env or {})},
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, ""
    return process.returncode or 0, stdout.decode("utf-8", "replace")


def _ruff_findings(output: str, owned: set[str], root: Path) -> list[dict[str, str]]:
    """Parse ruff's JSON report into findings for staged files only.

    Ruff reports absolute paths even when handed a relative target, so they are
    made relative before matching. Without that every ruff finding is silently
    discarded and the rung reports clean.
    """
    try:
        entries = json.loads(output or "[]")
    except json.JSONDecodeError:
        return []
    findings: list[dict[str, str]] = []
    for entry in entries if isinstance(entries, list) else []:
        code = str(entry.get("code") or "")
        relative = _relative_to(str(entry.get("filename") or ""), root)
        if relative not in owned:
            continue
        if code in _BLOCKING_RUFF_CODES:
            severity = ERROR
        elif code in _ADVISORY_RUFF_CODES:
            severity = WARNING
        else:
            continue
        line = (entry.get("location") or {}).get("row")
        where = f" (line {line})" if line else ""
        findings.append(
            _finding(relative, f"{code}: {entry.get('message', '')}{where}", severity)
        )
    return findings


def _mypy_findings(output: str, owned: set[str], root: Path) -> list[dict[str, str]]:
    """Parse mypy's text report into findings for staged files only."""
    findings: list[dict[str, str]] = []
    for raw in output.splitlines():
        match = _MYPY_LINE.match(raw.strip())
        if not match or match.group("severity") != "error":
            continue
        relative = _relative_to(match.group("path"), root)
        if relative not in owned:
            continue
        code = match.group("code") or ""
        if code in _BLOCKING_MYPY_CODES:
            severity = ERROR
        elif code in _ADVISORY_MYPY_CODES:
            severity = WARNING
        else:
            continue
        findings.append(
            _finding(
                relative,
                f"{code}: {match.group('message')} (line {match.group('line')})",
                severity,
            )
        )
    return findings


async def staged_static_analysis(
    staged: dict[str, dict[str, Any]], *, timeout_seconds: float = 60.0
) -> list[dict[str, str]]:
    """Everything ruff and mypy can prove about this changeset.

    Findings are limited to staged files: the changeset is what the user is
    about to approve, and faults it did not introduce are not its to answer for.
    A checker that is missing, times out, or crashes yields nothing rather than
    failing the build — an unavailable rung must never be indistinguishable
    from a clean one, and the caller reports which rungs ran.
    """
    if not staged:
        return []
    with tempfile.TemporaryDirectory(prefix="metis-typecheck-") as directory:
        root = Path(directory)
        owned = set(_materialize(staged, root))
        if not owned:
            return []
        # --isolated / --config-file: a model-authored config is part of the
        # changeset under review, so it does not get to configure its reviewer.
        ruff_code, ruff_output = await _run(
            [
                sys.executable, "-m", "ruff", "check",
                "--isolated", "--output-format=json", "--exit-zero", ".",
            ],
            root,
            timeout_seconds,
        )
        empty_config = root / ".metis-mypy.ini"
        empty_config.write_text("[mypy]\n", encoding="utf-8")
        # Explicit files with --explicit-package-bases, never the directory: a
        # project without __init__.py makes mypy abort the whole tree with
        # "Source file found twice under different module names" before it
        # checks anything, and it silently reported clean on a changeset whose
        # phantom keyword argument it catches in a second when asked directly.
        _, mypy_output = await _run(
            [
                sys.executable, "-m", "mypy",
                "--config-file", str(empty_config),
                "--ignore-missing-imports",
                "--no-error-summary",
                "--no-incremental",
                "--no-color-output",
                "--hide-error-context",
                "--explicit-package-bases",
                *sorted(owned),
            ],
            root,
            timeout_seconds,
            env={"MYPYPATH": str(root)},
        )
        findings = [
            *(_ruff_findings(ruff_output, owned, root) if ruff_code != 124 else []),
            *_mypy_findings(mypy_output, owned, root),
        ]
    findings.sort(key=lambda item: (item["severity"] != ERROR, item["path"]))
    return findings
