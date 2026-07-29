"""Reviewed verification recipes for project workspaces.

A project declares named checks in `.metis/verify.json`. The agent may only
*name* a declared check; it never supplies a command, so the set of programs
Metis can start in a project is fixed by a file the user reviewed rather than by
whatever a model decided to type. The recipe is approved once by fingerprint,
exactly like an asset launch manifest, and any edit to the file revokes that
approval.

Running a check is trusted host execution, not the generated-code sandbox: the
child runs as the user's own account with that account's filesystem and network
access. `explain_recipe` exists so the approval card can say that in plain
English, including for commands whose argv is not self-explanatory.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings


_MANIFEST_LIMIT = 32 * 1024
_CHECK_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SAFE_ID = re.compile(r"^asset_[0-9a-f]{20}$")
_MAX_CHECKS = 12
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Marks where the middle of an over-long stream was dropped while draining.
_ELISION = b"\x00\x00METIS-OUTPUT-ELIDED\x00\x00"

# A verification child inherits the user's account, so the recipe is the only
# boundary. Say so on the approval card rather than implying a sandbox.
BOUNDARY_NOTICE = (
    "Checks run as your own macOS user account, in the project folder, with "
    "your normal filesystem and network access. Metis never lets the model "
    "invent a command: it can only name one of the checks below, exactly as "
    "written here. Editing .metis/verify.json cancels this approval."
)


class ProjectVerificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    command: tuple[str, ...]
    description: str = ""
    timeout_seconds: int = 0

    @property
    def display_command(self) -> str:
        return shlex.join(self.command)


@dataclass(frozen=True, slots=True)
class VerificationRecipe:
    checks: tuple[VerificationCheck, ...] = ()
    fingerprint: str = ""
    present: bool = False
    error: str = ""
    _by_name: dict[str, VerificationCheck] = field(default_factory=dict, repr=False)

    def get(self, name: str) -> VerificationCheck | None:
        return self._by_name.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks)


@dataclass(frozen=True, slots=True)
class VerificationRun:
    name: str
    command: str
    ok: bool
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    output: str
    truncated: bool


# ── Plain-English explanation ────────────────────────────────────────────────
#
# The approval card has to be readable by someone who did not write the recipe,
# so every explanation is derived deterministically from the argv. No model call
# is involved: an explanation that needed the network would be unavailable in
# exactly the offline case where the user still has to decide whether to trust a
# command.

_SCRIPT_RUNNERS = {"npm": "npm", "pnpm": "pnpm", "yarn": "Yarn", "bun": "Bun"}
_TEST_TOOLS = {
    "pytest": "the Python test suite with pytest",
    "vitest": "the JavaScript test suite with Vitest",
    "jest": "the JavaScript test suite with Jest",
    "mocha": "the JavaScript test suite with Mocha",
    "phpunit": "the PHP test suite with PHPUnit",
    "rspec": "the Ruby test suite with RSpec",
}
_LINTERS = {
    "ruff": "Ruff",
    "eslint": "ESLint",
    "flake8": "Flake8",
    "pylint": "Pylint",
    "clippy": "Clippy",
    "biome": "Biome",
    "prettier": "Prettier",
}
_TYPE_CHECKERS = {
    "tsc": "TypeScript",
    "mypy": "mypy",
    "pyright": "Pyright",
}
_FLAG_NOTES = {
    "--noEmit": "without writing any output files",
    "--no-emit": "without writing any output files",
    "--check": "reporting problems without rewriting files",
    "--frozen-lockfile": "refusing to update the lockfile",
    "--frozen": "refusing to update the lockfile",
    "--locked": "refusing to update the lockfile",
    "-q": "with quiet output",
    "--quiet": "with quiet output",
    "--fix": "and automatically fixes what it can",
    "--watch": "and keeps watching for changes",
}


def _positional(command: tuple[str, ...], start: int = 1) -> list[str]:
    return [item for item in command[start:] if not item.startswith("-")]


def _flag_notes(command: tuple[str, ...]) -> str:
    notes = [_FLAG_NOTES[item] for item in command if item in _FLAG_NOTES]
    unique = list(dict.fromkeys(notes))
    return f", {' and '.join(unique)}" if unique else ""


# Recipes may use the same placeholders the runner substitutes. Resolving them
# here matters more than it looks: an unresolved `{python}` falls through to the
# generic fallback, so the one command the user most needs explained — "run this
# script with Metis's own interpreter" — would be the one described worst.
_PLACEHOLDERS = {"{python}": "python", "{uv}": "uv"}


def _describe_program(command: tuple[str, ...]) -> str:
    """One sentence naming what the first token actually does."""
    head = _PLACEHOLDERS.get(command[0], command[0])
    command = (head,) + command[1:]
    program = Path(head).name
    rest = _positional(command)
    target = rest[0] if rest else ""

    if program == "make":
        targets = rest or ["the default"]
        joined = ", ".join(f"`{item}`" for item in targets)
        return f"Runs the {joined} target from the project's Makefile"
    if program in _SCRIPT_RUNNERS:
        label = _SCRIPT_RUNNERS[program]
        if rest[:1] == ["run"] and rest[1:]:
            return f"Runs the project's `{rest[1]}` script with {label}"
        if target in {"install", "ci"}:
            return f"Installs the project's dependencies with {label}"
        if target:
            return f"Runs the project's `{target}` script with {label}"
        return f"Runs the project's default {label} script"
    if program in {"uv", "poetry", "pipenv", "hatch", "rye"}:
        if rest[:1] == ["run"] and rest[1:]:
            inner = _describe_program(tuple(command[command.index("run") + 1 :]))
            return f"{inner}, inside the project's {program}-managed environment"
        return f"Runs a {program} command in the project environment"
    if program == "npx":
        if rest:
            inner = _describe_program(tuple(rest))
            return f"{inner}, using a locally resolved tool"
        return "Runs a locally resolved Node tool"
    if program in _TEST_TOOLS:
        return f"Runs {_TEST_TOOLS[program]}"
    if program == "cargo":
        if target == "test":
            return "Compiles the Rust crate and runs its tests"
        if target in {"build", "check"}:
            return f"Compiles the Rust crate to check that it {'builds' if target == 'build' else 'type-checks'}"
        if target == "clippy":
            return "Lints the Rust crate with Clippy"
        return f"Runs `cargo {target or 'build'}` on the Rust crate"
    if program == "go":
        if target == "test":
            return "Runs the Go test suite"
        if target in {"build", "vet"}:
            return f"Runs `go {target}` over the Go packages"
        return f"Runs `go {target}` in the project"
    if program in _TYPE_CHECKERS:
        return f"Type-checks the sources with {_TYPE_CHECKERS[program]}"
    if program in _LINTERS:
        return f"Lints the sources with {_LINTERS[program]}"
    if program in {"python", "python3", "py"}:
        if command[1:2] == ("-m",) and command[2:3]:
            return f"Runs the Python module `{command[2]}`"
        if "-c" in command:
            # `-c` is inline source, not a path; calling it a script would point
            # a reviewer at a file that does not exist.
            return "Runs a short Python program written inline in the recipe"
        return f"Runs the Python script `{target}`" if target else "Runs a Python command"
    if program in {"gradle", "gradlew", "./gradlew"}:
        return f"Runs the Gradle `{target or 'build'}` task"
    if program in {"mvn", "maven"}:
        return f"Runs the Maven `{target or 'verify'}` phase"
    if program in {"swift", "xcodebuild"}:
        return f"Builds and checks the {program} project"
    if program in {"dotnet"}:
        return f"Runs `dotnet {target or 'build'}` on the solution"
    if program in {"bash", "sh", "zsh"}:
        script = target or "a shell script"
        return f"Runs the shell script `{script}` from the project"

    # A path rather than a bare name means the project ships the script itself,
    # which is the part a reviewer most needs to notice: its contents are not
    # covered by the fingerprint, only its invocation is.
    subject = (
        f"the project's own script `{command[0]}` (its contents are not pinned by "
        "this approval)"
        if "/" in command[0]
        else f"the program `{program}`"
    )
    arguments = " ".join(command[1:])
    if arguments:
        return f"Runs {subject} with the arguments `{arguments}`"
    return f"Runs {subject}"


def explain_command(command: tuple[str, ...]) -> str:
    """Plain-English sentence describing what one check will do."""
    if not command:
        return "Does nothing: this check has no command."
    return f"{_describe_program(command)}{_flag_notes(command)}."


def explain_recipe(recipe: VerificationRecipe) -> str:
    """Plain-English summary of every check the approval would authorize."""
    if not recipe.present:
        return "This project has not declared any verification checks yet."
    if recipe.error:
        return f"This project's verification file could not be used: {recipe.error}"
    lines = [
        f"Approving this lets Metis run {len(recipe.checks)} reviewed "
        f"{'check' if len(recipe.checks) == 1 else 'checks'} in this project:"
    ]
    for check in recipe.checks:
        lines.append(f"• {check.name} — `{check.display_command}`")
        lines.append(f"  {explain_command(check.command)}")
        if check.description:
            lines.append(f"  The project describes it as: {check.description}")
    return "\n".join(lines)


# ── Recipe file ──────────────────────────────────────────────────────────────


def _clean_text(value: object, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


def _valid_command(value: object) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 32
        or not all(type(item) is str for item in value)
        or sum(len(item) for item in value) > 8_192
    ):
        return None
    for item in value:
        if (
            not item
            or len(item) > 1_024
            or any(character in item for character in ("\x00", "\r", "\n"))
        ):
            return None
    return tuple(value)


def _fingerprint(project: Path, checks: tuple[VerificationCheck, ...]) -> str:
    payload = {
        "project_path": str(project.resolve(strict=False)),
        "checks": [
            {
                "name": check.name,
                "command": list(check.command),
                "timeout_seconds": check.timeout_seconds,
            }
            for check in checks
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def read_recipe(project: Path, settings: Settings) -> VerificationRecipe:
    """Parse `.metis/verify.json`, or explain why it is unusable.

    A malformed file is reported rather than ignored: silently treating it as
    "no checks configured" is how a user ends up believing verification is
    running when it never was.
    """
    metis_dir = project / ".metis"
    path = metis_dir / "verify.json"
    try:
        if metis_dir.is_symlink() or path.is_symlink():
            return VerificationRecipe(error="the verification file may not be a symlink")
        if not path.is_file():
            return VerificationRecipe()
        if path.stat().st_size > _MANIFEST_LIMIT:
            return VerificationRecipe(present=True, error="the verification file is too large")
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return VerificationRecipe(error="the verification file could not be read")

    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return VerificationRecipe(present=True, error="the verification file is not valid JSON")
    if not isinstance(body, dict) or body.get("schema_version") != "1":
        return VerificationRecipe(
            present=True, error='the verification file needs "schema_version": "1"'
        )
    declared = body.get("checks")
    if not isinstance(declared, list) or not declared:
        return VerificationRecipe(present=True, error="the verification file declares no checks")

    checks: list[VerificationCheck] = []
    seen: set[str] = set()
    for item in declared[:_MAX_CHECKS]:
        if not isinstance(item, dict):
            return VerificationRecipe(present=True, error="every check must be a JSON object")
        name = _clean_text(item.get("name"), 32).casefold()
        if not _CHECK_NAME.fullmatch(name):
            return VerificationRecipe(
                present=True,
                error=f"check name {name or '(missing)'!r} must be lowercase letters, digits, - or _",
            )
        if name in seen:
            return VerificationRecipe(present=True, error=f"check {name!r} is declared twice")
        command = _valid_command(item.get("command"))
        if command is None:
            return VerificationRecipe(
                present=True,
                error=f"check {name!r} needs a bounded argv array, not a shell string",
            )
        raw_timeout = item.get("timeout_seconds", settings.project_verify_timeout_seconds)
        try:
            timeout = int(raw_timeout)
        except (TypeError, ValueError):
            return VerificationRecipe(
                present=True, error=f"check {name!r} has a non-numeric timeout_seconds"
            )
        timeout = max(5, min(timeout, settings.project_verify_timeout_seconds))
        seen.add(name)
        checks.append(
            VerificationCheck(
                name=name,
                command=command,
                description=_clean_text(item.get("description"), 240),
                timeout_seconds=timeout,
            )
        )

    ordered = tuple(checks)
    return VerificationRecipe(
        checks=ordered,
        fingerprint=_fingerprint(project, ordered),
        present=True,
        _by_name={check.name: check for check in ordered},
    )


# ── Approval store and execution ─────────────────────────────────────────────


class ProjectVerificationService:
    """Owns recipe approval and bounded execution of reviewed checks."""

    def __init__(self, settings: Settings, *, approval_path: Path | None = None) -> None:
        self._settings = settings
        self._approval_path = approval_path
        self._approvals = self._load_approvals()
        self._lock = asyncio.Lock()

    def recipe(self, project: Path) -> VerificationRecipe:
        return read_recipe(project, self._settings)

    def is_approved(self, project_id: str, recipe: VerificationRecipe) -> bool:
        return bool(
            recipe.fingerprint
            and self._approvals.get(project_id) == recipe.fingerprint
        )

    async def approve(self, project_id: str, recipe: VerificationRecipe) -> None:
        if not recipe.present:
            raise ProjectVerificationError(
                "this project has no .metis/verify.json recipe to approve"
            )
        if recipe.error or not recipe.fingerprint:
            raise ProjectVerificationError(
                f"the verification recipe cannot be approved: {recipe.error}"
            )
        async with self._lock:
            self._approvals[project_id] = recipe.fingerprint
            self._save_approvals()

    async def revoke(self, project_id: str) -> None:
        async with self._lock:
            if self._approvals.pop(project_id, None) is not None:
                self._save_approvals()

    async def run(
        self, project: Path, check: VerificationCheck
    ) -> VerificationRun:
        """Execute one reviewed check, always returning evidence.

        A failing check is a normal, expected outcome — it is the signal the
        agent needs — so process failure is reported rather than raised. Only a
        refusal to start is exceptional, and even that comes back as output the
        next step can read.
        """
        argv = self._substituted(check.command)
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=project,
                env=self._environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                close_fds=True,
                start_new_session=os.name == "posix",
            )
        except (OSError, ValueError) as exc:
            return VerificationRun(
                name=check.name,
                command=check.display_command,
                ok=False,
                exit_code=None,
                timed_out=False,
                duration_seconds=round(time.monotonic() - started, 3),
                output=(
                    f"{check.display_command!r} could not be started: "
                    f"{type(exc).__name__}. Check that the program is installed "
                    "and on PATH."
                ),
                truncated=False,
            )

        timed_out = False
        try:
            raw = await asyncio.wait_for(
                self._read_bounded(process), timeout=check.timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            raw = b""
            self._terminate(process)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                self._kill(process)
        await process.wait()

        text, truncated = self._bounded_text(raw)
        if timed_out:
            text = (
                f"Timed out after {check.timeout_seconds}s and was stopped.\n{text}"
            ).strip()
        return VerificationRun(
            name=check.name,
            command=check.display_command,
            ok=process.returncode == 0 and not timed_out,
            exit_code=process.returncode,
            timed_out=timed_out,
            duration_seconds=round(time.monotonic() - started, 3),
            output=text,
            truncated=truncated,
        )

    def _substituted(self, command: tuple[str, ...]) -> tuple[str, ...]:
        runtime_python = str(Path(sys.executable).resolve(strict=False))
        runtime_uv = str(Path(sys.executable).with_name("uv").resolve(strict=False))
        return tuple(
            item.replace("{python}", runtime_python).replace("{uv}", runtime_uv)
            for item in command
        )

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = {"PATH": os.environ.get("PATH", os.defpath)}
        for key in ("HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"):
            value = os.environ.get(key)
            if value:
                environment[key] = value
        # Tools that colour output for a TTY produce escape codes that waste the
        # model's context and read as noise in the timeline.
        environment.update({"CI": "1", "NO_COLOR": "1", "TERM": "dumb"})
        return environment

    async def _read_bounded(self, process: asyncio.subprocess.Process) -> bytes:
        """Drain the child, keeping only as much as the caller can use.

        A test run that prints a megabyte per second must not be able to grow
        the host's memory, and the model will never read more than the bounded
        window anyway.
        """
        cap = max(self._settings.project_verify_output_chars * 4, 8_192)
        assert process.stdout is not None
        head = bytearray()
        tail = bytearray()
        overflowed = False
        while True:
            block = await process.stdout.read(16_384)
            if not block:
                break
            if len(head) < cap:
                room = cap - len(head)
                head.extend(block[:room])
                block = block[room:]
            if block:
                overflowed = True
                tail.extend(block)
                if len(tail) > cap:
                    del tail[: len(tail) - cap]
        if not overflowed:
            return bytes(head)
        return bytes(head) + _ELISION + bytes(tail)

    def _bounded_text(self, raw: bytes) -> tuple[str, bool]:
        """Decode, then keep the head and the tail.

        Build and test output puts the invocation at the top and the failure at
        the bottom; dropping either end is what makes a truncated log useless.
        """
        limit = self._settings.project_verify_output_chars
        text = raw.decode("utf-8", errors="replace")
        dropped_bytes = _ELISION.decode() in text
        text = text.replace(_ELISION.decode(), "\n")
        text = _ANSI.sub("", text).strip()
        if len(text) <= limit:
            return text, dropped_bytes
        head = text[: limit // 3].rstrip()
        tail = text[-(limit - limit // 3) :].lstrip()
        omitted = len(text) - len(head) - len(tail)
        return f"{head}\n\n… {omitted} characters of output omitted …\n\n{tail}", True

    @staticmethod
    def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass

    def _load_approvals(self) -> dict[str, str]:
        path = self._approval_path
        if path is None:
            return {}
        try:
            if path.is_symlink():
                return {}
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            project_id: fingerprint
            for project_id, fingerprint in value.items()
            if isinstance(project_id, str)
            and _SAFE_ID.fullmatch(project_id)
            and isinstance(fingerprint, str)
            and re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        }

    def _save_approvals(self) -> None:
        path = self._approval_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ProjectVerificationError(
                "the verification approval store may not be a symbolic link"
            )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(self._approvals, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if os.name == "posix":
                temporary.chmod(0o600)
            os.replace(temporary, path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ProjectVerificationError(
                "the verification approval could not be saved"
            ) from exc
