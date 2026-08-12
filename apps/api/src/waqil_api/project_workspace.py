"""Explicit, bounded project workspaces for Metis chat.

The saved Asset catalog is the grant boundary: selecting a project never scans
for new siblings. A deterministic local manifest and an evolving METIS.md live
inside the project. Models can inspect the grant through narrow tools; exact
mutations are executed only after the control plane records user approval.
"""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .asset_library import AssetLibraryError, AssetManager
from .config import Settings
from .contracts import (
    PROJECT_HOST_TOOLS,
    ProjectBootstrapV1,
    ProjectCheckV1,
    ProjectToolCallV1,
    ProjectVerificationV1,
    ProjectWorkspaceV1,
)
from .project_conformance import staged_conformance_errors
from .project_contracts import cross_file_findings, is_stylesheet, style_gaps
from .project_lookup import LookupError_, inspect_installed_api
from .project_env import CAPABILITY_VARS, capabilities_of_tree
from .project_patch import EXACT, PatchProblem, locate_patch
from .project_scaffold import SCAFFOLD_VERSION, scaffold_sources
from .project_sandbox import ProjectSandboxService, SandboxOutcome
from .project_typecheck import staged_static_analysis
from .project_verification import (
    BOUNDARY_NOTICE,
    ProjectVerificationService,
    explain_command,
    explain_recipe,
)
from .project_wiring import staged_wiring_errors
from . import repo_map


_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".next",
        ".nuxt",
        ".turbo",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "coverage",
        "target",
        "__pycache__",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".md",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".css": "CSS",
    ".scss": "SCSS",
    ".html": "HTML",
    ".md": "Markdown",
}
_PRIORITY_FILES = (
    "README.md",
    "README",
    "AGENTS.md",
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "docker-compose.yml",
    "compose.yml",
)
_SECRETISH = re.compile(
    r"(?i)(password|passwd|secret|private[_ -]?key|api[_ -]?key|access[_ -]?token)"
)
_SENSITIVE_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx", ".jks"})

# Env files that hold no secrets: they exist to *name* the variables a project
# needs, which is the opposite of hiding them. Blocking these along with real
# `.env` files was a silent, total failure — every build was asked for
# `.env.example`, every plan listed it, every model tried to write it, and the
# host refused all of them. Ten builds across three models each ended one file
# short for this reason and it read as the models forgetting.
_ENV_TEMPLATE_NAMES = frozenset(
    {".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"}
)
_MANAGED_HEADER = "<!-- metis-project-context:v1 -->"
_LEARNINGS_START = "<!-- metis-learnings:start -->"
_LEARNINGS_END = "<!-- metis-learnings:end -->"

_PLAN_START = "<!-- metis-plan:start -->"
_PLAN_END = "<!-- metis-plan:end -->"

# A coding engine receives a useful project, not an unbounded clone of an
# arbitrary filesystem tree. The file count follows the user-configured
# manifest boundary; this independent byte ceiling prevents a project made of
# a few enormous artifacts from filling the run volume while it is mirrored.
_EXTERNAL_MIRROR_MAX_BYTES = 128_000_000
_EXTERNAL_WORKSPACE_PREFIX = "metis-code-workspace-"
_EXTERNAL_WORKSPACE_NAME = re.compile(
    rf"^{re.escape(_EXTERNAL_WORKSPACE_PREFIX)}[A-Za-z0-9_-]{{6,64}}$"
)
_EXTERNAL_SECRET_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "credentials.toml",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "service-account.json",
        "service_account.json",
    }
)
_EXTERNAL_SENSITIVE_SUFFIXES = _SENSITIVE_SUFFIXES | frozenset(
    {".cer", ".crt", ".der", ".gpg", ".keystore", ".kdbx"}
)


@dataclass(frozen=True, slots=True)
class ExternalMirrorFile:
    """One immutable byte fact from an external workspace's starting tree."""

    path: str
    sha256: str
    bytes: int
    source: str
    disk_sha256: str


@dataclass(frozen=True, slots=True)
class ExternalWorkspaceMirror:
    """A disposable project mirror and the baseline its diff is measured from.

    ``project_root`` is the only path an external coding engine should receive.
    The baseline lives in host memory, outside that editable directory, so the
    engine cannot make a rewrite look unchanged by editing its own manifest.
    """

    id: str
    asset_id: str
    source_root: Path
    project_root: Path
    tree_sha256: str
    overlay_sha256: str
    files: tuple[ExternalMirrorFile, ...]
    excluded_count: int
    excluded_paths: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class ExternalChangeProvenance:
    """The durable identity attached to every imported external edit."""

    engine: str
    session_id: str
    run_id: str = ""


class ProjectWorkspaceError(RuntimeError):
    """A refused project tool call.

    ``argument_shape`` marks the refusals a model can fix by resending the same
    tool with better arguments — a missing required key, a block that does not
    match. Those are worth narrowing the next step's grammar to. Everything else
    is semantic (the target is unavailable, the file already exists), where
    pinning the model to the same call would loop it until the budget is gone.

    ``wrong_target`` marks the one semantic refusal that does have a mechanical
    answer: a create_file aimed at a path that already exists. The tool is right
    and the arguments are well formed — only the target is wrong — so the next
    step can be pinned to the files the build still owes instead of leaving the
    model to guess, which measurably means re-sending the same path.

    ``repair_strategy`` names a safer edit primitive after the requested one
    has proved brittle. Exact-block and line-range failures use ``whole_file``:
    the next coder rewrites the complete current file through one replace_lines
    call instead of guessing another patch or range.
    """

    def __init__(
        self,
        message: str,
        *,
        argument_shape: bool = False,
        wrong_target: bool = False,
        repair_strategy: str = "",
        repairable_external: bool = False,
        repair_path: str = "",
    ) -> None:
        super().__init__(message)
        self.argument_shape = argument_shape
        self.wrong_target = wrong_target
        self.repair_strategy = repair_strategy
        # External mirror import is atomic, so a syntax-invalid in-scope file
        # cannot enter the approval overlay. It is nevertheless model-repairable
        # inside that same private mirror. This marker is deliberately absent
        # from scope, CAS, deletion, symlink, secret, and destructive-rewrite
        # refusals, which remain terminal safety boundaries.
        self.repairable_external = repairable_external
        self.repair_path = repair_path


class VerificationNotApprovedError(ProjectWorkspaceError):
    """The project declares checks the user has not reviewed yet.

    Distinct from a plain tool error so the control plane can raise a one-time
    approval instead of handing the agent a dead end it cannot act on.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded_line(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


def _bootstrap_aliases(preference: Any | None) -> dict[str, str] | None:
    """The routing hint for a project map, or None when there is no preference.

    Cline uses it to retain the selected planner rung; the Ollama fallback uses
    its local planner alias. A preference store that cannot answer is not worth
    failing a map over, so any error degrades to role defaults.
    """
    if preference is None:
        return None
    try:
        return dict(preference.resolve_aliases())
    except Exception:  # noqa: BLE001 - a routing hint is never worth a failure
        return None


def _preferred_planner(preference: Any | None, settings: Settings) -> str:
    """The model name to RECORD for a map the Ollama lane wrote."""
    aliases = _bootstrap_aliases(preference) or {}
    return str(aliases.get("planner") or settings.planner_model)


def _read_window(
    text: str, arguments: dict[str, Any], max_characters: int
) -> dict[str, Any]:
    """One bounded slice of a file, returned as the exact bytes it contains.

    The content is verbatim, and that is the whole point. Reads used to be
    line-numbered — ``f"{n:>6}  {line}"`` — while apply_patch requires
    ``original`` to appear in the file exactly once. The model copied what it
    was shown, numbers and all, so the patch matched zero times, every time; the
    two halves of the read/edit contract could not both be satisfied. Measured
    against a real model: 0 of 4 patches could apply from a numbered view, 4 of
    4 from this one.

    The numbers stay available as ``start_line``/``end_line`` beside the text
    rather than running through it. A slice that hits the character budget says
    so, because silently truncating text a model is about to quote back is how
    an exact match becomes impossible to make.
    """
    start = min(max(int(arguments.get("start_line", 1)), 1), 1_000_000)
    end = min(max(int(arguments.get("end_line", start + 399)), start), start + 999)
    lines = text.splitlines()
    selected = lines[start - 1 : end]
    content = "\n".join(selected)
    return {
        "start_line": start,
        "end_line": start + len(selected) - 1,
        "total_lines": len(lines),
        "truncated": len(content) > max_characters,
        "content": content[:max_characters],
    }


def _splice_lines(text: str, arguments: dict[str, Any]) -> str:
    """The file text with one 1-indexed inclusive line range replaced.

    The forgiving edit primitive. apply_patch requires a byte-exact quote of
    the current text, and that is precisely what weak models cannot produce:
    across the measured repair turns, 0 of 8 attempted fixes landed, every
    miss on whitespace. Line coordinates come from read_file's own
    ``start_line``/``end_line`` — nothing here depends on quoting — and the
    optional ``expect`` guard turns a mis-aimed range into a refusal that
    shows the model what the range actually holds, instead of a silent wrong
    edit the verifier has to catch later.
    """
    try:
        start = int(arguments.get("start_line", 0))
        end = int(arguments.get("end_line", 0))
    except (TypeError, ValueError):
        raise ProjectWorkspaceError(
            "replace_lines needs integer start_line and end_line",
            argument_shape=True,
            repair_strategy="whole_file",
        ) from None
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if start < 1 or end < start:
        raise ProjectWorkspaceError(
            f"replace_lines needs 1 <= start_line <= end_line; you sent {start}..{end}",
            argument_shape=True,
            repair_strategy="whole_file",
        )
    # A present but empty file still has a valid whole-file insertion point.
    # The recovery grammar deliberately uses 1..WHOLE_FILE_END_LINE for every
    # file, and clamping that range to zero lines should replace the empty body.
    if start > total and not (total == 0 and start == 1):
        raise ProjectWorkspaceError(
            f"replace_lines range {start}..{end} starts past the end of the "
            f"file, which has {total} line(s); read_file the target first",
            argument_shape=True,
            repair_strategy="whole_file",
        )
    end = min(end, total)
    doomed = "".join(lines[start - 1 : end])
    expect = str(arguments.get("expect", "") or "")
    if expect and expect not in doomed:
        raise ProjectWorkspaceError(
            f"replace_lines refused: lines {start}..{end} do not contain the "
            f"expected text {expect[:120]!r}. That range currently holds:\n"
            f"{doomed[:400]}\nAim start_line/end_line at the block you meant.",
            argument_shape=True,
            repair_strategy="whole_file",
        )
    replacement = str(arguments.get("replacement", ""))
    # A replacement that stops mid-line would glue itself onto the next line
    # and manufacture a syntax error the model never wrote.
    if replacement and not replacement.endswith("\n") and end < total:
        replacement += "\n"
    return "".join(lines[: start - 1]) + replacement + "".join(lines[end:])


class _HtmlStructureMeasure(HTMLParser):
    """Measure document structure while excluding inline script bodies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.weight = 0
        self._script_depth = 0
        self.loads_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        folded = tag.casefold()
        if folded == "script":
            self._script_depth += 1
            self.loads_script = self.loads_script or any(
                name.casefold() == "src" and bool(value and value.strip())
                for name, value in attrs
            )
        # Tags and attributes are part of the page's structural contract even
        # inside script markup; only JavaScript data is intentionally ignored.
        self.weight += len(tag) + sum(
            len(name) + len(value or "") for name, value in attrs
        )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() == "script" and self._script_depth:
            self._script_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        self.weight += len(tag)
        if tag.casefold() == "script" and self._script_depth:
            self._script_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._script_depth:
            self.weight += len(data)


def _html_structure(document: str) -> tuple[int, bool]:
    parser = _HtmlStructureMeasure()
    parser.feed(document)
    parser.close()
    return parser.weight, parser.loads_script


def _guard_whole_file_repair(
    relative: str,
    before: str,
    after: str,
    arguments: Mapping[str, Any],
) -> None:
    """Refuse a repair that satisfies the checker by deleting the file.

    A live three-model micro-benchmark exposed the otherwise invisible failure:
    one coder honored the whole-file tool schema and cleared mypy by replacing a
    958-byte repository with 67 bytes. Parsing and the targeted static check
    both passed, but most of the module's public API was gone. The host-selected
    recovery strategy uses a sentinel end line, while models may spell the same
    full rewrite with the actual last line. In either form its contract is a
    narrow repair, so retaining the surrounding file is an enforceable invariant
    rather than a stylistic preference.
    """
    try:
        start = int(arguments.get("start_line", 0))
        end = int(arguments.get("end_line", 0))
    except (TypeError, ValueError):
        return
    line_count = len(before.splitlines(keepends=True))
    # The sentinel is how the host requests a recovery rewrite, but a model can
    # express the identical operation with the file's real last line (notably
    # 1..1 for minified HTML). Guard the operation's actual span, not one spelling
    # of it. Empty files have no surface to preserve.
    if start != 1 or not line_count or end < line_count:
        return
    before_bytes = len(before.encode("utf-8"))
    after_bytes = len(after.encode("utf-8"))
    severe_shrink = before_bytes >= 256 and after_bytes * 2 < before_bytes
    if severe_shrink and relative.lower().endswith((".html", ".htm")):
        # Extracting a large inline application script into its already-planned
        # static asset can legitimately remove most of an HTML file's bytes.
        # Compare browser-visible structure with script bodies excluded. This
        # admits that refactor while a replacement containing only a script tag
        # still loses almost all of the original page's structural weight.
        before_structure, _ = _html_structure(before)
        after_structure, loads_script = _html_structure(after)
        severe_shrink = not (
            loads_script
            and before_structure >= 128
            and after_structure * 2 >= before_structure
        )
    if severe_shrink:
        raise ProjectWorkspaceError(
            f"whole-file repair for {relative} is incomplete: it shrank the file "
            f"from {before_bytes} to {after_bytes} bytes. Send the complete corrected "
            "file, preserving unrelated code and public interfaces.",
            argument_shape=True,
            repair_strategy="whole_file",
        )
    if not relative.lower().endswith(".py"):
        return
    try:
        old_tree = ast.parse(before)
        new_tree = ast.parse(after)
    except SyntaxError:
        return  # the ordinary parse gate below reports the precise syntax error

    def public_surface(tree: ast.Module) -> set[str]:
        return {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }

    missing = sorted(public_surface(old_tree) - public_surface(new_tree))
    if missing:
        raise ProjectWorkspaceError(
            f"whole-file repair for {relative} removed public definitions: "
            f"{', '.join(missing[:12])}. Send the complete corrected file and retain "
            "every unrelated public interface.",
            argument_shape=True,
            repair_strategy="whole_file",
        )


# Every spelling a model reaches for when it means "the project root". Left
# unrecognized, each one filters the listing down to nothing and the project
# reads as empty — which is what sends an agent into a re-listing loop.
_ROOT_ALIASES = {"", ".", "./", "/", "*", "**", "./*"}


def _project_prefix(value: Any) -> str:
    """Normalize a list/search path prefix, treating root spellings as no filter."""
    prefix = _bounded_line(value, 1_000).strip()
    if prefix in _ROOT_ALIASES:
        return ""
    while prefix.startswith("./"):
        prefix = prefix[2:]
    return prefix.strip("/")


def _suggest_relative(relative: str) -> str:
    """The project-relative form of a rejected path, for the error message.

    A model that writes an absolute path has almost always prefixed a workspace
    root of its own invention, so the tail after that prefix is what it meant.
    """
    parts = [part for part in Path(relative).parts if part not in {"/", "..", "."}]
    for index, part in enumerate(parts):
        if part in {"app", "src", "lib", "tests", "test", "docs", "scripts"}:
            return "/".join(parts[index:])
    return "/".join(parts[1:]) if len(parts) > 1 else "/".join(parts)


def _is_text_file(path: Path) -> bool:
    return path.name in _PRIORITY_FILES or path.suffix.casefold() in _TEXT_SUFFIXES


def _external_secret_path(relative: Path) -> bool:
    """Whether a path is too likely to carry credentials to leave the host.

    Source modules may legitimately be named ``secrets.py`` or
    ``credential_store.ts``; those contain handling logic, not necessarily
    values. Data/config files with the same names are excluded fail-closed.
    """
    name = relative.name.casefold()
    suffix = relative.suffix.casefold()
    if name.startswith(".env") and name not in _ENV_TEMPLATE_NAMES:
        return True
    if name in _EXTERNAL_SECRET_NAMES or suffix in _EXTERNAL_SENSITIVE_SUFFIXES:
        return True
    source_suffixes = frozenset(
        {
            ".c",
            ".cc",
            ".cpp",
            ".go",
            ".h",
            ".java",
            ".js",
            ".jsx",
            ".kt",
            ".php",
            ".py",
            ".rb",
            ".rs",
            ".swift",
            ".ts",
            ".tsx",
            ".vue",
        }
    )
    credentialish = re.search(
        r"(?i)(credential|password|passwd|secret|private[_ -]?key|"
        r"api[_ -]?key|access[_ -]?token|auth[_ -]?token)",
        name,
    )
    return bool(credentialish and suffix not in source_suffixes)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write_text(target: Path, content: str) -> None:
    """Write ``target`` via temp-file + fsync + atomic rename.

    A plain ``write_text`` truncates the destination in place: a crash
    mid-write leaves a partial file on disk, and the next materialize pass
    reports it as merely "changed after staging" rather than corrupted. The
    rename step means a crash can only ever leave the previous complete
    content or the new complete content, never a half-written file.
    """

    try:
        mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    descriptor, temporary = tempfile.mkstemp(
        prefix=".metis-apply-", dir=str(target.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _external_tree_digest(files: Iterable[ExternalMirrorFile]) -> str:
    """Bind a baseline to every path, byte length, hash and source."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda candidate: candidate.path):
        for value in (
            item.path,
            str(item.bytes),
            item.sha256,
            item.source,
            item.disk_sha256,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _staged_state_digest(staged: Mapping[str, Mapping[str, Any]]) -> str:
    """Bind an external session to the exact overlay it was given.

    The ordinary approval digest intentionally covers only the bytes a user
    will apply. An external coding session also needs to retain the CAS origin
    and prior provenance: swapping a ``create`` for a ``patch`` with identical
    content changes whether materialization may overwrite a disk path.
    """
    digest = hashlib.sha256()
    for relative in sorted(staged):
        entry = staged[relative]
        provenance = json.dumps(
            entry.get("provenance", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for value in (
            relative,
            str(entry.get("content", "")),
            str(entry.get("origin", "")),
            str(entry.get("base_sha256", "")),
            str(entry.get("bytes", "")),
            provenance,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _decode_external_text(relative: str, content: bytes) -> str:
    """Decode an edited file without guessing that arbitrary bytes are text."""
    if b"\x00" in content:
        raise ProjectWorkspaceError(
            f"external change {relative} is binary; binary changes are not supported"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectWorkspaceError(
            f"external change {relative} is binary or not UTF-8; only text changes "
            "can be staged"
        ) from exc
    # A UTF-8 decode alone is not sufficient: compact binary formats can be
    # ASCII-heavy. Permit ordinary whitespace, but refuse the C0 controls that
    # never belong in source/configuration files.
    if any(ord(character) < 32 and character not in "\t\n\r\f" for character in text):
        raise ProjectWorkspaceError(
            f"external change {relative} contains binary control bytes; only text "
            "changes can be staged"
        )
    return text


def _normalized_external_provenance(
    provenance: ExternalChangeProvenance,
) -> dict[str, str]:
    engine = _bounded_line(provenance.engine, 120)
    session_id = _bounded_line(provenance.session_id, 240)
    run_id = _bounded_line(provenance.run_id, 240)
    if not engine or not session_id:
        raise ProjectWorkspaceError(
            "external changes require a bounded engine and session_id provenance"
        )
    return {
        "engine": engine,
        "session_id": session_id,
        **({"run_id": run_id} if run_id else {}),
    }


# How much on-disk Python the static gate will read to resolve a changeset's
# imports. Enough to see an ordinary project, small enough that checking a
# ten-file build never costs more than the build did.
_STATIC_SOURCE_BYTES = 2_000_000


def _requirements_from(
    staged: dict[str, dict[str, Any]], on_disk: dict[str, str]
) -> str:
    """The project's declared dependencies, with the changeset taking precedence."""
    texts = [text for text in on_disk.values() if text]
    for name in ("requirements.txt", "pyproject.toml"):
        entry = staged.get(name)
        if entry:
            texts.append(str(entry.get("content", "")))
    return "\n".join(texts)


def parse_error(path: str, content: str) -> str | None:
    """Why this file will not parse, or None when the host cannot fault it.

    Only languages with a safe, dependency-free stdlib parser are checked, and
    parsing executes nothing — no import runs, no bytecode is evaluated — so
    this is safe to run on code the model just wrote. Any other language is
    passed through unchecked rather than guessed at, so None means "nothing the
    host can validate is broken", not "this file is correct".
    """
    suffix = Path(path).suffix.lower()
    try:
        if suffix in {".py", ".pyi"}:
            ast.parse(content, filename=path)
        elif suffix == ".json":
            json.loads(content)
        return None
    except json.JSONDecodeError as exc:
        return f"JSONDecodeError: {exc.msg} (line {exc.lineno})"
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        return f"{type(exc).__name__}: {exc.msg}{where}"
    except ValueError as exc:
        # ast.parse rejects source containing a NUL byte with a ValueError;
        # treat anything unparseable as a reportable error, not a crash.
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def staged_syntax_errors(staged: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Parse-check each staged file, one ``{path, error}`` per failure.

    Since writes are parse-checked as they are staged, a changeset reaching
    here should already be clean. This stays as the changeset-wide backstop:
    it covers overlays assembled by any path that did not go through
    ``_stage_write``, and it is what the approval card reports against.
    """
    errors: list[dict[str, str]] = []
    for path in sorted(staged):
        error = parse_error(path, str(staged[path].get("content", "")))
        if error:
            errors.append({"path": path, "error": error})
    return errors


class ProjectWorkspaceService:
    def __init__(
        self,
        settings: Settings,
        assets: AssetManager,
        bootstrap_model: Any,
        verification: ProjectVerificationService | None = None,
        sandbox: ProjectSandboxService | None = None,
        preference: Any | None = None,
    ) -> None:
        self.settings = settings
        self.assets = assets
        self.bootstrap_model = bootstrap_model
        self.verification = verification
        self.sandbox = sandbox
        # The user's chosen model, so a map that falls back to the Ollama lane
        # runs on the model they actually picked rather than on the settings
        # default — which is a local 35B that is usually not even loaded.
        # Optional: a service built without one behaves exactly as before.
        self.preference = preference
        self._lock = asyncio.Lock()
        # Repo-map extraction, per project, keyed by (mtime_ns, size) so a
        # multi-step build turn parses each file exactly once.
        self._repo_map_cache: dict[
            str, dict[str, tuple[tuple[int, int], repo_map.FileFacts]]
        ] = {}

    async def list(self) -> list[ProjectWorkspaceV1]:
        projects: list[ProjectWorkspaceV1] = []
        for asset in await self.assets.list():
            initialized = False
            revision = 0
            file_count = 0
            updated_at = None
            try:
                root = await self.assets.project_path(asset.id)
                manifest = self._read_manifest(root)
                initialized = bool(manifest)
                revision = int(manifest.get("revision", 0)) if manifest else 0
                file_count = int(manifest.get("file_count", 0)) if manifest else 0
                raw_updated = manifest.get("updated_at") if manifest else None
                updated_at = raw_updated if isinstance(raw_updated, str) else None
            except (AssetLibraryError, OSError, ValueError, TypeError):
                pass
            projects.append(
                ProjectWorkspaceV1(
                    id=asset.id,
                    name=asset.name,
                    summary=asset.summary,
                    framework=asset.framework,
                    initialized=initialized,
                    manifest_revision=revision,
                    file_count=file_count,
                    updated_at=updated_at,
                )
            )
        return projects

    async def open(self, asset_id: str) -> ProjectWorkspaceV1:
        """Create or refresh local context; a model maps only the first access."""
        async with self._lock:
            root = await self.assets.project_path(asset_id)
            metadata = await self._asset_metadata(asset_id)
            metis_dir = root / ".metis"
            manifest_path = metis_dir / "project-context.json"
            notes_path = metis_dir / "METIS.md"
            if metis_dir.exists() and metis_dir.is_symlink():
                raise ProjectWorkspaceError(
                    "the project's .metis folder may not be a symlink"
                )
            if manifest_path.is_symlink() or notes_path.is_symlink():
                raise ProjectWorkspaceError(
                    "Metis project context files may not be symlinks"
                )
            metis_dir.mkdir(parents=True, exist_ok=True)

            prior = self._read_manifest(root)
            if manifest_path.exists() and not prior:
                raise ProjectWorkspaceError(
                    ".metis/project-context.json already exists but is not a valid Metis manifest"
                )
            snapshot, sample = await asyncio.to_thread(self._snapshot, root)
            bootstrap_raw = prior.get("bootstrap") if prior else None
            bootstrap = (
                ProjectBootstrapV1.model_validate(bootstrap_raw)
                if isinstance(bootstrap_raw, dict)
                else None
            )
            # Which provider will actually write the map, so the manifest can
            # record it truthfully rather than always claiming Grok.
            bootstrapper = "oci-grok"
            bootstrap_model_name = self.settings.oci_grok_model
            if bootstrap is None:
                aliases = _bootstrap_aliases(self.preference)
                available = getattr(self.bootstrap_model, "available", None)
                if available is None and hasattr(self.bootstrap_model, "oci"):
                    # Mirror the router's own order (a selected Cline planner,
                    # then Grok, Cohere, and the Ollama lane) so a healthy
                    # Cline-only setup can open a project without borrowing an
                    # unrelated provider or a local model session.
                    #
                    # The Ollama fallback is why this no longer refuses: making
                    # a map used to require an OCI or Cohere key, so with the
                    # Grok lane off and a spent Cohere quota — the real state of
                    # this install — no project could be opened at all.
                    cohere = getattr(self.bootstrap_model, "cohere", None)
                    cline = getattr(self.bootstrap_model, "cline", None)
                    selected_provider = str((aliases or {}).get("_provider") or "local")
                    cline_ready = bool(
                        selected_provider == "cline"
                        and cline is not None
                        and getattr(cline, "available", False)
                    )
                    oci_ready = bool(
                        getattr(self.bootstrap_model.oci, "available", False)
                    )
                    cohere_ready = bool(
                        cohere is not None and getattr(cohere, "available", False)
                    )
                    if cline_ready:
                        available = True
                        bootstrapper = "cline"
                        bootstrap_model_name = self.settings.cline_orchestrator_model
                    elif oci_ready:
                        available = True
                    elif cohere_ready:
                        available = True
                        bootstrapper = "cohere"
                        bootstrap_model_name = self.settings.cohere_model
                    else:
                        available = True
                        bootstrapper = "ollama"
                        bootstrap_model_name = _preferred_planner(
                            self.preference, self.settings
                        )
                if available is False:
                    raise ProjectWorkspaceError(
                        "A project map needs a model backend that can answer a "
                        "structured request; none is configured"
                    )
                request = {
                    "project": metadata,
                    "manifest": snapshot,
                    "bounded_file_samples": sample,
                }
                try:
                    bootstrap = await self.bootstrap_model.bootstrap_project(
                        request, model_aliases=aliases
                    )
                except TypeError:
                    # A provider (or a test double) whose bootstrap_project
                    # predates the aliases keyword. The map is worth more than
                    # the routing hint, so it still gets made.
                    bootstrap = await self.bootstrap_model.bootstrap_project(request)
                self._write_initial_notes(notes_path, metadata, bootstrap)
            elif not notes_path.is_file():
                self._write_initial_notes(notes_path, metadata, bootstrap)

            timestamp = _now()
            revision = int(prior.get("revision", 0)) + 1 if prior else 1
            manifest = {
                "schema_version": "1",
                "project_id": asset_id,
                "project_name": metadata["name"],
                "root_name": root.name,
                "revision": revision,
                "created_at": prior.get("created_at", timestamp)
                if prior
                else timestamp,
                "updated_at": timestamp,
                "bootstrap_provider": prior.get("bootstrap_provider", bootstrapper)
                if prior
                else bootstrapper,
                "bootstrap_model": prior.get("bootstrap_model", bootstrap_model_name)
                if prior
                else bootstrap_model_name,
                **snapshot,
                "bootstrap": bootstrap.model_dump(mode="json"),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return ProjectWorkspaceV1(
                id=asset_id,
                name=metadata["name"],
                summary=metadata["summary"],
                framework=metadata.get("framework"),
                initialized=True,
                manifest_revision=revision,
                file_count=int(snapshot["file_count"]),
                updated_at=timestamp,
            )

    async def context(self, asset_id: str) -> dict[str, Any]:
        root = await self.assets.project_path(asset_id)
        manifest = self._read_manifest(root)
        notes_path = root / ".metis" / "METIS.md"
        if not manifest or not notes_path.is_file() or notes_path.is_symlink():
            raise ProjectWorkspaceError(
                "open this project once before starting a project chat"
            )
        notes = notes_path.read_text(encoding="utf-8")[:40_000]
        return {
            "project_id": asset_id,
            "project_name": manifest.get("project_name", root.name),
            "manifest": manifest,
            "metis_md": notes,
            "verification": await self._verification_context(asset_id, root),
            "settings": self.read_project_settings(root),
        }

    @staticmethod
    def settings_path(root: Path) -> Path:
        return root / ".metis" / "project-settings.json"

    def read_project_settings(self, root: Path) -> dict[str, Any]:
        """Durable per-project settings a person set. Never model-writable.

        Stores path identities and patterns only. The bytes behind a protected
        path are hashed per run at admission, so a file edited on disk between
        runs cannot inherit a stale hash.
        """

        path = self.settings_path(root)
        if not path.is_file() or path.is_symlink():
            return {"protected_files": []}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt settings file protects nothing rather than failing the
            # project open; the resolved contract is emitted visibly either way.
            return {"protected_files": []}
        if not isinstance(raw, Mapping):
            return {"protected_files": []}
        stored = raw.get("protected_files")
        return {
            "protected_files": sorted(
                {
                    str(item).replace("\\", "/").strip()
                    for item in (stored if isinstance(stored, list) else [])
                    if str(item).strip()
                }
            )[:256]
        }

    async def write_project_settings(
        self, asset_id: str, protected_files: Sequence[str]
    ) -> dict[str, Any]:
        """Replace the per-project protections. Callers are people, not models."""

        root = (await self.assets.project_path(asset_id)).resolve()
        cleaned = sorted(
            {
                str(item).replace("\\", "/").strip()
                for item in protected_files
                if str(item).strip()
            }
        )[:256]
        for candidate in cleaned:
            # A protection is a project-relative identity or pattern. An
            # absolute path or a traversal is a different instruction.
            if candidate.startswith("/") or ".." in PurePosixPath(candidate).parts:
                raise ProjectWorkspaceError(
                    f"a protected path must stay inside the project: {candidate}"
                )
        target = self.settings_path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            target,
            json.dumps({"protected_files": cleaned}, indent=2) + "\n",
        )
        return {"protected_files": cleaned}

    async def preview_project_settings(
        self, asset_id: str, protected_files: Sequence[str]
    ) -> dict[str, Any]:
        """What those patterns resolve to right now, without saving them."""

        root = (await self.assets.project_path(asset_id)).resolve()
        manifest = self._read_manifest(root) or {}
        tree = [str(path) for path in manifest.get("file_tree") or []]
        resolved: list[str] = []
        unmatched: list[str] = []
        for pattern in protected_files:
            needle = str(pattern).replace("\\", "/").strip()
            if not needle:
                continue
            hits = [
                path
                for path in tree
                if path == needle
                or PurePosixPath(path).name == needle
                or fnmatch.fnmatch(path, needle)
                or fnmatch.fnmatch(PurePosixPath(path).name, needle)
            ]
            if hits:
                resolved.extend(hits)
            else:
                unmatched.append(needle)
        return {
            "resolved": sorted(set(resolved)),
            "unmatched": sorted(set(unmatched)),
        }

    async def repo_map(
        self, asset_id: str, *, request: str = "", max_chars: int = 6_000
    ) -> str:
        """A ranked symbol map of this project, biased toward `request`.

        Separate from `context()` on purpose: the map is only worth ranking if
        it knows what is being asked, and `context()` is called in places where
        no request exists yet. Extraction is cached per file by (mtime, size),
        so a build turn re-parses only what actually changed between steps —
        ranking and rendering are pure and cost microseconds.
        """
        if max_chars <= 0:
            return ""
        root = await self.assets.project_path(asset_id)
        return await asyncio.to_thread(
            self._repo_map_sync, asset_id, root, request, max_chars
        )

    async def interface_map(
        self,
        asset_id: str,
        *,
        target_path: str,
        dependency_paths: Sequence[str] = (),
        staged: Mapping[str, Mapping[str, Any]] | None = None,
        max_chars: int = 6_000,
    ) -> str:
        """Exact callable/import context for one directed file.

        Unlike the ranked overview, this reads the staged overlay as the source
        of truth. Earlier files in a build are not materialized until approval,
        so a disk-only map would teach the next coder that their interfaces do
        not exist. Canonical appkit modules are included when they are actually
        present on disk or in the overlay; absent optional capabilities are
        never advertised.
        """
        if max_chars <= 0 or not target_path:
            return ""
        root = await self.assets.project_path(asset_id)
        return await asyncio.to_thread(
            self._interface_map_sync,
            root,
            target_path,
            tuple(dependency_paths),
            dict(staged or {}),
            max_chars,
        )

    def _interface_map_sync(
        self,
        root: Path,
        target_path: str,
        dependency_paths: Sequence[str],
        staged: Mapping[str, Mapping[str, Any]],
        max_chars: int,
    ) -> str:
        canonical_appkit = {
            path
            for path in scaffold_sources({"oci_responses", "web_ui"})
            if path.startswith("appkit/")
        }
        selected = {
            target_path,
            *(str(path) for path in dependency_paths if path),
            *canonical_appkit,
        }
        sources: dict[str, str] = {}
        for path, relative in self._iter_files(root):
            rel = relative.as_posix()
            if rel not in selected or not _is_text_file(path):
                continue
            try:
                sources[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
        # Overlay bytes are newer than disk bytes by definition. A repair must
        # preserve the interface the approval card currently carries, not the
        # older file beneath it.
        for rel, entry in staged.items():
            if rel not in selected or not isinstance(entry, Mapping):
                continue
            content = entry.get("content")
            if isinstance(content, str):
                sources[rel] = content
        facts = [repo_map.extract(path, source) for path, source in sources.items()]
        return repo_map.render_interfaces(
            facts,
            target_path=target_path,
            dependency_paths=dependency_paths,
            max_chars=max_chars,
        )

    def _repo_map_sync(
        self, asset_id: str, root: Path, request: str, max_chars: int
    ) -> str:
        cache = self._repo_map_cache.setdefault(asset_id, {})
        facts: list[repo_map.FileFacts] = []
        seen: set[str] = set()
        for path, relative in self._iter_files(root):
            rel = relative.as_posix()
            if not _is_text_file(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            seen.add(rel)
            key = (stat.st_mtime_ns, stat.st_size)
            cached = cache.get(rel)
            if cached is not None and cached[0] == key:
                facts.append(cached[1])
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            entry = repo_map.extract(rel, source)
            cache[rel] = (key, entry)
            facts.append(entry)
        # A deleted file must not keep contributing symbols that no longer
        # exist — a map that names a path the model then cannot read is exactly
        # the confidently-wrong map this is supposed to replace.
        for stale in set(cache) - seen:
            cache.pop(stale, None)
        if not facts:
            return ""
        scores = repo_map.rank(facts, repo_map.focus_terms(request))
        return repo_map.render(facts, scores, max_chars=max_chars)

    async def _verification_context(self, asset_id: str, root: Path) -> dict[str, Any]:
        """What the agent is allowed to know about `run_check`.

        Only names and descriptions cross this boundary. The agent never sees
        the argv, because it can never supply one — telling it the command would
        only invite it to propose variations the host would refuse.
        """
        if self.verification is None or not self.settings.project_verify_enabled:
            return {"available": False, "reason": "verification checks are disabled"}
        recipe = await asyncio.to_thread(self.verification.recipe, root)
        if not recipe.present:
            return {"available": False, "reason": "this project declares no checks"}
        if recipe.error:
            return {"available": False, "reason": recipe.error}
        if not self.verification.is_approved(asset_id, recipe):
            return {
                "available": False,
                "reason": "the user has not approved this project's checks yet",
                "checks": [check.name for check in recipe.checks],
            }
        return {
            "available": True,
            "checks": [
                {"name": check.name, "description": check.description}
                for check in recipe.checks
            ],
        }

    async def preview(self, asset_id: str, call: ProjectToolCallV1) -> dict[str, str]:
        root = await self.assets.project_path(asset_id)
        if call.name not in {"apply_patch", "replace_lines", "create_file"}:
            raise ProjectWorkspaceError("that project tool does not mutate files")
        relative = _bounded_line(call.arguments.get("path"), 1_000)
        target = self._safe_target(root, relative, write=True)
        if call.name == "apply_patch":
            original = str(call.arguments.get("original", ""))
            replacement = str(call.arguments.get("replacement", ""))
            detail = (
                f"Replace {len(original)} characters with {len(replacement)} characters "
                f"in {relative}."
            )
        elif call.name == "replace_lines":
            replacement = str(call.arguments.get("replacement", ""))
            detail = (
                f"Replace lines {call.arguments.get('start_line')}–"
                f"{call.arguments.get('end_line')} of {relative} with "
                f"{len(replacement)} characters."
            )
        else:
            content = str(call.arguments.get("content", ""))
            detail = f"Create {relative} with {len(content.encode('utf-8'))} bytes."
        digest = hashlib.sha256(
            json.dumps(call.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "path": str(target.relative_to(root)),
            "summary": detail,
            "digest": digest,
        }

    async def execute(self, asset_id: str, call: ProjectToolCallV1) -> dict[str, Any]:
        if call.name in PROJECT_HOST_TOOLS:
            # Roster tools, but host affordances: the loop pauses on ask_user,
            # publishes respond, and re-gates on revise_plan, all itself.
            # Reaching the workspace means a routing bug, and a loud refusal
            # beats a silent misfile.
            raise ProjectWorkspaceError(
                f"{call.name} is a talk tool the loop handles; it never runs "
                "in the workspace"
            )
        root = await self.assets.project_path(asset_id)
        if call.name == "list_files":
            return await asyncio.to_thread(self._list_files, root, call.arguments)
        if call.name == "search_code":
            return await asyncio.to_thread(self._search_code, root, call.arguments)
        if call.name == "read_file":
            return await asyncio.to_thread(self._read_file, root, call.arguments)
        if call.name == "apply_patch":
            async with self._lock:
                result = await asyncio.to_thread(
                    self._apply_patch, root, call.arguments
                )
                await asyncio.to_thread(self._record_mutation, root, call, result)
                return result
        if call.name == "replace_lines":
            async with self._lock:
                result = await asyncio.to_thread(
                    self._replace_lines, root, call.arguments
                )
                await asyncio.to_thread(self._record_mutation, root, call, result)
                return result
        if call.name == "create_file":
            async with self._lock:
                result = await asyncio.to_thread(
                    self._create_file, root, call.arguments
                )
                await asyncio.to_thread(self._record_mutation, root, call, result)
                return result
        if call.name == "run_check":
            return await self._run_check(asset_id, root, call.arguments)
        if call.name == "inspect_api":
            return await self._inspect_api(call)
        raise ProjectWorkspaceError(f"unsupported project tool: {call.name}")

    # ── Staged builds ──────────────────────────────────────────────────────
    # The act→observe→decide loop runs against an overlay: writes land in a
    # dict the graph state carries, reads consult the overlay before the disk,
    # and nothing touches the real tree until the user approves the whole
    # changeset once. A staged entry is {content, origin, base_sha256, bytes} —
    # base_sha256 pins the disk text a patch was computed against, so a file
    # that changed under a pending approval is skipped rather than clobbered.

    async def create_external_mirror(
        self,
        asset_id: str,
        staged: dict[str, dict[str, Any]],
        *,
        workspace_parent: Path | None = None,
    ) -> ExternalWorkspaceMirror:
        """Materialize disk + overlay into a disposable, editable project.

        The mirror is a capability boundary for a local coding engine. Secret
        and internal paths are omitted before the engine sees the tree; staged
        entries are CAS-checked first, and the immutable baseline is retained
        by the host rather than written inside the editable project.
        """
        root = (await self.assets.project_path(asset_id)).resolve()
        parent = Path(workspace_parent or self.settings.run_dir)

        async with self._lock:
            return await asyncio.to_thread(
                self._create_external_mirror,
                asset_id,
                root,
                dict(staged),
                parent,
            )

    def _create_external_mirror(
        self,
        asset_id: str,
        root: Path,
        staged: dict[str, dict[str, Any]],
        parent: Path,
    ) -> ExternalWorkspaceMirror:
        self._validate_external_overlay(root, staged)
        parent.mkdir(parents=True, exist_ok=True)
        container = Path(
            tempfile.mkdtemp(prefix=_EXTERNAL_WORKSPACE_PREFIX, dir=str(parent))
        )
        project_root = container / "project"
        project_root.mkdir(mode=0o700)
        records: dict[str, ExternalMirrorFile] = {}
        byte_sizes: dict[str, int] = {}
        excluded: set[str] = set()

        try:
            count = 0
            total = 0
            for current, directories, filenames in os.walk(root, followlinks=False):
                current_path = Path(current)
                kept: list[str] = []
                for directory in sorted(directories):
                    source = current_path / directory
                    relative = source.relative_to(root)
                    rel = relative.as_posix() + "/"
                    if source.is_symlink():
                        excluded.add(rel)
                        continue
                    if directory in _IGNORE_DIRS or directory.startswith("."):
                        excluded.add(rel)
                        continue
                    kept.append(directory)
                directories[:] = kept

                for filename in sorted(filenames):
                    source = current_path / filename
                    relative = source.relative_to(root)
                    rel = relative.as_posix()
                    if source.is_symlink() or not source.is_file():
                        excluded.add(rel)
                        continue
                    if _external_secret_path(relative):
                        excluded.add(rel)
                        continue
                    content = source.read_bytes()
                    count += 1
                    total += len(content)
                    if count > self.settings.project_manifest_max_files:
                        raise ProjectWorkspaceError(
                            "project exceeds the external workspace file limit"
                        )
                    if total > _EXTERNAL_MIRROR_MAX_BYTES:
                        raise ProjectWorkspaceError(
                            "project exceeds the external workspace byte limit"
                        )
                    destination = project_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    destination.chmod(source.stat().st_mode & 0o777)
                    digest = _sha256_bytes(content)
                    records[rel] = ExternalMirrorFile(
                        path=rel,
                        sha256=digest,
                        bytes=len(content),
                        source="disk",
                        disk_sha256=digest,
                    )
                    byte_sizes[rel] = len(content)

            # The overlay is authoritative: replace the copied disk bytes with
            # precisely what the coding session is meant to observe.
            for rel in sorted(staged):
                entry = staged[rel]
                content = str(entry["content"]).encode("utf-8")
                previous_size = byte_sizes.get(rel, 0)
                if rel not in records:
                    count += 1
                total += len(content) - previous_size
                if count > self.settings.project_manifest_max_files:
                    raise ProjectWorkspaceError(
                        "project exceeds the external workspace file limit"
                    )
                if total > _EXTERNAL_MIRROR_MAX_BYTES:
                    raise ProjectWorkspaceError(
                        "project exceeds the external workspace byte limit"
                    )
                relative = Path(rel)
                destination = project_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                prior = records.get(rel)
                records[rel] = ExternalMirrorFile(
                    path=rel,
                    sha256=_sha256_bytes(content),
                    bytes=len(content),
                    source="staged",
                    disk_sha256=prior.disk_sha256 if prior else "",
                )
                byte_sizes[rel] = len(content)

            files = tuple(records[path] for path in sorted(records))
            return ExternalWorkspaceMirror(
                id=f"ewm_{uuid.uuid4().hex}",
                asset_id=asset_id,
                source_root=root,
                project_root=project_root,
                tree_sha256=_external_tree_digest(files),
                overlay_sha256=_staged_state_digest(staged),
                files=files,
                excluded_count=len(excluded),
                excluded_paths=tuple(sorted(excluded)[:256]),
                created_at=_now(),
            )
        except Exception:
            shutil.rmtree(container, ignore_errors=True)
            raise

    def _validate_external_overlay(
        self, root: Path, staged: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Prove every staged entry still names the disk state it was based on."""
        for relative in sorted(staged):
            entry = staged[relative]
            target = self._safe_target(root, relative, write=True, framework=True)
            canonical = target.relative_to(root).as_posix()
            if canonical != relative:
                raise ProjectWorkspaceError(
                    f"staged path {relative!r} is not canonical; expected {canonical!r}"
                )
            content = str(entry.get("content", ""))
            encoded = content.encode("utf-8")
            try:
                recorded_bytes = int(entry.get("bytes", len(encoded)))
            except (TypeError, ValueError):
                raise ProjectWorkspaceError(
                    f"staged entry {relative} has invalid byte provenance"
                ) from None
            if recorded_bytes != len(encoded):
                raise ProjectWorkspaceError(
                    f"staged entry {relative} no longer matches its recorded byte count"
                )
            origin = str(entry.get("origin", ""))
            if origin == "create":
                if target.exists():
                    raise ProjectWorkspaceError(
                        f"cannot mirror {relative}: a file appeared after it was staged"
                    )
                continue
            if origin != "patch":
                raise ProjectWorkspaceError(
                    f"staged entry {relative} has unknown origin provenance"
                )
            if not target.is_file() or target.is_symlink():
                raise ProjectWorkspaceError(
                    f"cannot mirror {relative}: its disk source is unavailable"
                )
            disk_sha256 = _sha256_bytes(target.read_bytes())
            if disk_sha256 != str(entry.get("base_sha256", "")):
                raise ProjectWorkspaceError(
                    f"cannot mirror {relative}: its disk source changed after staging"
                )

    def _validate_external_mirror_identity(
        self, asset_id: str, root: Path, mirror: ExternalWorkspaceMirror
    ) -> None:
        if mirror.asset_id != asset_id or mirror.source_root != root:
            raise ProjectWorkspaceError(
                "external workspace does not belong to the selected project"
            )
        if mirror.project_root.is_symlink() or not mirror.project_root.is_dir():
            raise ProjectWorkspaceError(
                "external workspace is unavailable or redirected"
            )
        container = mirror.project_root.parent
        if mirror.project_root.name != "project" or not container.name.startswith(
            _EXTERNAL_WORKSPACE_PREFIX
        ):
            raise ProjectWorkspaceError("external workspace identity is invalid")
        if mirror.tree_sha256 != _external_tree_digest(mirror.files):
            raise ProjectWorkspaceError("external workspace baseline metadata changed")

    def _read_external_tree(
        self,
        mirror: ExternalWorkspaceMirror,
        *,
        baseline: Mapping[str, ExternalMirrorFile],
    ) -> dict[str, bytes]:
        """Read an untrusted engine workspace without following filesystem aliases."""
        current: dict[str, bytes] = {}
        total = 0
        for current_dir, directories, filenames in os.walk(
            mirror.project_root, followlinks=False
        ):
            current_path = Path(current_dir)
            for name in sorted([*directories, *filenames]):
                path = current_path / name
                relative = path.relative_to(mirror.project_root)
                rel = relative.as_posix()
                if path.is_symlink():
                    raise ProjectWorkspaceError(
                        f"external workspace contains a symbolic link: {rel}"
                    )
                if any(part in {".git", ".metis"} for part in relative.parts):
                    raise ProjectWorkspaceError(
                        f"external workspace changed protected internals: {rel}"
                    )
                if _external_secret_path(relative):
                    raise ProjectWorkspaceError(
                        f"external workspace contains a secret-bearing path: {rel}"
                    )
            directories[:] = sorted(directories)
            for filename in sorted(filenames):
                path = current_path / filename
                relative = path.relative_to(mirror.project_root)
                rel = relative.as_posix()
                if not path.is_file():
                    raise ProjectWorkspaceError(
                        f"external workspace contains a non-regular file: {rel}"
                    )
                stat = path.stat()
                if stat.st_nlink != 1:
                    raise ProjectWorkspaceError(
                        f"external workspace contains a hard-linked file: {rel}"
                    )
                if len(current) >= (
                    self.settings.project_manifest_max_files
                    + self.settings.project_staged_max_files
                ):
                    raise ProjectWorkspaceError(
                        "external workspace exceeds the file import limit"
                    )
                if (
                    stat.st_size > self.settings.project_max_write_bytes
                    and rel not in baseline
                ):
                    raise ProjectWorkspaceError(
                        f"external create {rel} exceeds the per-file write limit"
                    )
                content = path.read_bytes()
                total += len(content)
                if total > (
                    _EXTERNAL_MIRROR_MAX_BYTES + self.settings.project_staged_max_bytes
                ):
                    raise ProjectWorkspaceError(
                        "external workspace exceeds the byte import limit"
                    )
                current[rel] = content
        return current

    async def rebase_external_mirror(
        self,
        mirror: ExternalWorkspaceMirror,
        staged: dict[str, dict[str, Any]],
    ) -> ExternalWorkspaceMirror:
        """Advance one persistent engine session to its just-imported overlay.

        Import deliberately invalidates the old overlay digest. Rebase proves
        that every current mirror byte is either unchanged from the old baseline
        or exactly present in the newly staged overlay, then returns a new
        immutable baseline for the same directory. No project bytes are copied.
        """
        root = (await self.assets.project_path(mirror.asset_id)).resolve()
        async with self._lock:
            return await asyncio.to_thread(
                self._rebase_external_mirror, root, mirror, dict(staged)
            )

    def _rebase_external_mirror(
        self,
        root: Path,
        mirror: ExternalWorkspaceMirror,
        staged: dict[str, dict[str, Any]],
    ) -> ExternalWorkspaceMirror:
        self._validate_external_mirror_identity(mirror.asset_id, root, mirror)
        self._validate_external_overlay(root, staged)
        old = {item.path: item for item in mirror.files}
        current = self._read_external_tree(mirror, baseline=old)
        missing = sorted(set(old) - set(current))
        if missing:
            raise ProjectWorkspaceError(
                f"cannot rebase an external workspace with a deletion: {missing[0]}"
            )
        unexpected = sorted(set(current) - set(old) - set(staged))
        if unexpected:
            raise ProjectWorkspaceError(
                f"cannot rebase an unimported external change: {unexpected[0]}"
            )

        records: list[ExternalMirrorFile] = []
        for rel in sorted(current):
            content = current[rel]
            digest = _sha256_bytes(content)
            entry = staged.get(rel)
            if entry is not None:
                expected = str(entry.get("content", "")).encode("utf-8")
                if content != expected:
                    raise ProjectWorkspaceError(
                        f"cannot rebase unimported bytes for {rel}"
                    )
                origin = str(entry.get("origin", ""))
                records.append(
                    ExternalMirrorFile(
                        path=rel,
                        sha256=digest,
                        bytes=len(content),
                        source="staged",
                        disk_sha256=(
                            str(entry.get("base_sha256", ""))
                            if origin == "patch"
                            else ""
                        ),
                    )
                )
                continue
            prior = old.get(rel)
            if prior is None or digest != prior.sha256:
                raise ProjectWorkspaceError(f"cannot rebase unimported bytes for {rel}")
            records.append(prior)

        files = tuple(records)
        return ExternalWorkspaceMirror(
            id=mirror.id,
            asset_id=mirror.asset_id,
            source_root=mirror.source_root,
            project_root=mirror.project_root,
            tree_sha256=_external_tree_digest(files),
            overlay_sha256=_staged_state_digest(staged),
            files=files,
            excluded_count=mirror.excluded_count,
            excluded_paths=mirror.excluded_paths,
            created_at=_now(),
        )

    async def preview_external_overlay(
        self,
        asset_id: str,
        mirror: ExternalWorkspaceMirror,
        staged: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """A read-only view of the mirror as it stands right now.

        ``run_check`` has to verify what the model has written *so far*, in the
        middle of its own turn. Running the real import to get that would
        journal, rebase and advance the overlay digest mid-round, which is the
        state the round's own drift proof depends on. This reads the same tree
        through the same alias-safe reader and returns an overlay-shaped dict
        that is handed to the verifier and then thrown away.

        Nothing here mutates workspace state, and nothing outside the mirror is
        read: the walk is bounded by the mirror's own baseline exactly as the
        authoritative import is.
        """

        root = (await self.assets.project_path(asset_id)).resolve()

        def _read() -> dict[str, dict[str, Any]]:
            self._validate_external_mirror_identity(asset_id, root, mirror)
            baseline = {item.path: item for item in mirror.files}
            current = self._read_external_tree(mirror, baseline=baseline)
            preview: dict[str, dict[str, Any]] = {
                path: dict(entry) for path, entry in staged.items()
            }
            for path, content in current.items():
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    # A binary file cannot be verified as source; the existing
                    # overlay entry (if any) stands.
                    continue
                previous = baseline.get(path)
                if previous is not None and previous.sha256 == _sha256_bytes(content):
                    continue
                preview[path] = {
                    "content": text,
                    "bytes": len(content),
                    "origin": "preview",
                }
            return preview

        async with self._lock:
            return await asyncio.to_thread(_read)

    async def import_external_changes(
        self,
        asset_id: str,
        mirror: ExternalWorkspaceMirror,
        staged: dict[str, dict[str, Any]],
        *,
        provenance: ExternalChangeProvenance,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Atomically turn a mirror's byte diff into the ordinary staged overlay.

        The external process never supplies a patch or a claimed file list. The
        host walks the resulting tree, compares every byte hash to its retained
        baseline, then sends each accepted create/modify through ``_stage_write``.
        Any refusal aborts the whole import and the caller's overlay is untouched.
        """
        root = (await self.assets.project_path(asset_id)).resolve()
        normalized_provenance = _normalized_external_provenance(provenance)

        async with self._lock:
            return await asyncio.to_thread(
                self._import_external_changes,
                asset_id,
                root,
                mirror,
                dict(staged),
                normalized_provenance,
            )

    def _import_external_changes(
        self,
        asset_id: str,
        root: Path,
        mirror: ExternalWorkspaceMirror,
        staged: dict[str, dict[str, Any]],
        provenance: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        self._validate_external_mirror_identity(asset_id, root, mirror)
        if mirror.overlay_sha256 != _staged_state_digest(staged):
            raise ProjectWorkspaceError(
                "staged overlay changed while the external coding session was running"
            )
        self._validate_external_overlay(root, staged)

        baseline = {item.path: item for item in mirror.files}
        current = self._read_external_tree(mirror, baseline=baseline)

        missing = sorted(set(baseline) - set(current))
        if missing:
            missing_hashes = {baseline[path].sha256 for path in missing}
            replacement = next(
                (
                    path
                    for path, content in current.items()
                    if path not in baseline and _sha256_bytes(content) in missing_hashes
                ),
                "",
            )
            detail = f" (possible rename to {replacement})" if replacement else ""
            raise ProjectWorkspaceError(
                f"external deletions and renames are not supported: {missing[0]}{detail}"
            )

        changes: list[tuple[str, str, bytes, str]] = []
        for rel in sorted(current):
            before = baseline.get(rel)
            after = current[rel]
            after_sha256 = _sha256_bytes(after)
            if before is not None and before.sha256 == after_sha256:
                continue
            changes.append(
                (rel, "modify" if before is not None else "create", after, after_sha256)
            )

        # Preflight every path, payload and target CAS before constructing any
        # candidate entries. This makes a late forbidden file no different from
        # an early one: neither can partially enter the returned overlay.
        decoded: dict[str, str] = {}
        for rel, change, content, _ in changes:
            target = self._safe_target(root, rel, write=True)
            decoded[rel] = _decode_external_text(rel, content)
            if len(content) > self.settings.project_max_write_bytes:
                raise ProjectWorkspaceError(
                    f"external change {rel} exceeds the per-file write limit"
                )
            before = baseline.get(rel)
            if change == "create":
                if target.exists() or rel in staged:
                    raise ProjectWorkspaceError(
                        f"external create {rel} collided with a project file"
                    )
            elif before is None:
                raise ProjectWorkspaceError(f"external baseline is missing {rel}")
            elif before.source == "disk":
                if (
                    not target.is_file()
                    or _sha256_bytes(target.read_bytes()) != before.disk_sha256
                ):
                    raise ProjectWorkspaceError(
                        f"external change {rel} is stale because its disk source changed"
                    )

        next_staged = dict(staged)
        imported: list[dict[str, Any]] = []
        for rel, change, _, after_sha256 in changes:
            before = baseline.get(rel)
            if change == "create":
                call = ProjectToolCallV1(
                    name="create_file",
                    arguments={"path": rel, "content": decoded[rel]},
                )
            else:
                before_text = (
                    str(next_staged[rel]["content"])
                    if rel in next_staged
                    else root.joinpath(rel).read_text(encoding="utf-8")
                )
                call = ProjectToolCallV1(
                    name="replace_lines",
                    arguments={
                        "path": rel,
                        "start_line": 1,
                        "end_line": max(1, len(before_text.splitlines(keepends=True))),
                        "replacement": decoded[rel],
                    },
                )
            _, candidate = self._stage_write(root, call, next_staged)
            entry = dict(candidate[rel])
            file_provenance: dict[str, Any] = {
                "kind": "external_workspace",
                **provenance,
                "workspace_id": mirror.id,
                "workspace_baseline_sha256": mirror.tree_sha256,
                "file_before_sha256": before.sha256 if before else "",
                "file_after_sha256": after_sha256,
                "excluded_path_count": mirror.excluded_count,
                "excluded_paths": list(mirror.excluded_paths),
            }
            entry["provenance"] = file_provenance
            candidate[rel] = entry
            next_staged = candidate
            imported.append(
                {
                    "path": rel,
                    "change": change,
                    "before_sha256": before.sha256 if before else "",
                    "after_sha256": after_sha256,
                    "bytes": len(decoded[rel].encode("utf-8")),
                }
            )

        total = sum(int(entry["bytes"]) for entry in next_staged.values())
        result: dict[str, Any] = {
            "workspace_id": mirror.id,
            "baseline_sha256": mirror.tree_sha256,
            "overlay_sha256": mirror.overlay_sha256,
            "provenance": provenance,
            "excluded_path_count": mirror.excluded_count,
            "excluded_paths": list(mirror.excluded_paths),
            "changes": imported,
            "staged_files": len(next_staged),
            "staged_bytes": total,
        }
        return result, next_staged

    async def discard_external_mirror(self, mirror: ExternalWorkspaceMirror) -> None:
        """Remove only a temporary container created by ``create_external_mirror``."""
        container = mirror.project_root.parent
        if mirror.project_root.name != "project" or not container.name.startswith(
            _EXTERNAL_WORKSPACE_PREFIX
        ):
            raise ProjectWorkspaceError("refusing to remove an unknown workspace path")
        await asyncio.to_thread(shutil.rmtree, container, True)

    async def discard_unreferenced_external_mirrors(
        self,
        referenced_workspace_paths: set[Path],
        *,
        older_than: datetime,
    ) -> tuple[Path, ...]:
        """Collect abandoned mirrors that crashed before a session row existed.

        This deliberately accepts only old, direct children with the exact
        ``mkdtemp`` shape.  A symlink, recent write, referenced mirror, or path
        outside the private workspace parent turns deletion into a no-op.
        """

        if older_than.tzinfo is None:
            raise ValueError("orphan workspace cutoff must be timezone-aware")
        return await asyncio.to_thread(
            self._discard_unreferenced_external_mirrors,
            referenced_workspace_paths,
            older_than,
        )

    def _discard_unreferenced_external_mirrors(
        self,
        referenced_workspace_paths: set[Path],
        older_than: datetime,
    ) -> tuple[Path, ...]:
        parent = self.settings.coding_workspace_dir
        if not parent.is_dir() or parent.is_symlink():
            return ()
        parent = parent.resolve()
        referenced = {
            Path(os.path.abspath(str(path))) for path in referenced_workspace_paths
        }
        removed: list[Path] = []
        with os.scandir(parent) as entries:
            for entry in entries:
                if not _EXTERNAL_WORKSPACE_NAME.fullmatch(
                    entry.name
                ) or not entry.is_dir(follow_symlinks=False):
                    continue
                container = Path(entry.path)
                project_root = container / "project"
                if (
                    Path(os.path.abspath(str(project_root))) in referenced
                    or not project_root.is_dir()
                    or project_root.is_symlink()
                    or self._external_workspace_is_recent_or_linked(
                        container,
                        older_than,
                    )
                ):
                    continue
                # Recheck after the scan narrows the race with a newly-created
                # mirror. A current mirror is always recent even before its row
                # has been committed.
                if self._external_workspace_is_recent_or_linked(
                    container,
                    older_than,
                ):
                    continue
                shutil.rmtree(container)
                removed.append(container)
        return tuple(removed)

    @staticmethod
    def _external_workspace_is_recent_or_linked(
        container: Path,
        older_than: datetime,
    ) -> bool:
        cutoff = older_than.timestamp()
        try:
            if container.lstat().st_mtime >= cutoff:
                return True
            for root, directories, files in os.walk(container, followlinks=False):
                root_path = Path(root)
                if root_path.is_symlink() or root_path.lstat().st_mtime >= cutoff:
                    return True
                for name in (*directories, *files):
                    candidate = root_path / name
                    stat = candidate.lstat()
                    if candidate.is_symlink() or stat.st_mtime >= cutoff:
                        return True
        except FileNotFoundError:
            return True
        return False

    async def execute_staged(
        self,
        asset_id: str,
        call: ProjectToolCallV1,
        staged: dict[str, dict[str, Any]],
        next_paths: Sequence[str] = (),
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
        """Run one tool call through the overlay.

        Returns the tool result and, for writes, the replacement overlay;
        reads return None as the second element and leave staging untouched.
        ``next_paths`` is the files the turn planned and has not written, so a
        refused write can name the one the build actually still owes.
        """
        if call.name in PROJECT_HOST_TOOLS:
            raise ProjectWorkspaceError(
                f"{call.name} is a talk tool the loop handles; it never runs "
                "in the workspace"
            )
        root = await self.assets.project_path(asset_id)
        if call.name == "list_files":
            return (
                await asyncio.to_thread(
                    self._list_files_staged, root, call.arguments, staged
                ),
                None,
            )
        if call.name == "search_code":
            return (
                await asyncio.to_thread(
                    self._search_code_staged, root, call.arguments, staged
                ),
                None,
            )
        if call.name == "read_file":
            return (
                await asyncio.to_thread(
                    self._read_file_staged, root, call.arguments, staged
                ),
                None,
            )
        if call.name == "inspect_api":
            return await self._inspect_api(call), None
        if call.name in {"apply_patch", "replace_lines", "create_file"}:
            return await asyncio.to_thread(
                self._stage_write, root, call, dict(staged), tuple(next_paths)
            )
        raise ProjectWorkspaceError(
            f"project tool {call.name} cannot run against staged changes"
        )

    async def _inspect_api(self, call: ProjectToolCallV1) -> dict[str, Any]:
        """The real shape of an installed library, read rather than recalled.

        ``appkit`` is redirected to the canonical scaffold package. It is
        vendored into the project rather than installed, so the probe — which
        deliberately runs with no project on its path — could not see it, and
        a model asking for the signature of a helper it was misusing got told
        to "declare it in requirements". That is the correct question to ask
        and the worst possible answer: measured live, a repair turn asked
        three times, was refused, and never fixed the call. The vendored bytes
        are copied from this package, so the signature it reports is the one
        the project actually holds.
        """
        module = str(call.arguments.get("module", "")).strip()
        probed = module
        if module == "appkit" or module.startswith("appkit."):
            probed = f"{__package__}.scaffold.{module}"
        try:
            found = await inspect_installed_api(
                probed,
                str(call.arguments.get("symbol", "") or ""),
                project_roots=tuple(self.settings.asset_roots),
            )
        except LookupError_ as exc:
            raise ProjectWorkspaceError(
                str(exc).replace(probed, module), argument_shape=True
            ) from exc
        if probed != module and isinstance(found.get("module"), str):
            # Report the name the project imports, not Metis's internal path.
            found["module"] = module
        return found

    def _overlay_text(
        self, root: Path, relative: str, staged: dict[str, dict[str, Any]]
    ) -> str | None:
        """The staged text for a path, whichever way the model spelled it.

        Overlay keys are canonical (``str(target.relative_to(root))``), but the
        lookup used to take the raw argument — so a model that staged
        ``app/x.py`` and re-read ``./app/x.py`` missed its own work and was told
        the file was unavailable. Canonicalizing through the same jail the write
        used makes the two spellings agree.
        """
        entry = staged.get(relative)
        if entry is None:
            try:
                canonical = str(self._safe_target(root, relative).relative_to(root))
            except (ProjectWorkspaceError, ValueError):
                return None
            entry = staged.get(canonical)
        return str(entry["content"]) if entry else None

    def _list_files_staged(
        self, root: Path, arguments: dict[str, Any], staged: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        prefix = _project_prefix(arguments.get("path", ""))
        limit = min(max(int(arguments.get("limit", 200)), 1), 500)

        def matches(rel: str) -> bool:
            return (
                not prefix or rel == prefix or rel.startswith(prefix.rstrip("/") + "/")
            )

        names = {rel for rel in staged if matches(rel)}
        for _, relative in self._iter_files(root):
            rel = relative.as_posix()
            if matches(rel):
                names.add(rel)
                if len(names) >= limit * 2:
                    break
        files = sorted(names)[:limit]
        return {"files": files, "truncated": len(names) > len(files)}

    def _read_file_staged(
        self, root: Path, arguments: dict[str, Any], staged: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        overlay = self._overlay_text(root, relative, staged)
        if overlay is None:
            return self._read_file(root, arguments)
        # Same shape as a disk read, so the model cannot tell staged text from
        # committed text — which is the point of observing it.
        self._safe_target(root, relative)
        return {
            **_read_window(overlay, arguments, self.settings.project_tool_result_chars),
            "path": relative,
            "staged": True,
        }

    def _search_code_staged(
        self, root: Path, arguments: dict[str, Any], staged: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        query = _bounded_line(arguments.get("query"), 300)
        if not query:
            raise ProjectWorkspaceError("search_code requires a query")
        limit = min(max(int(arguments.get("limit", 80)), 1), 200)
        case_sensitive = bool(arguments.get("case_sensitive", False))
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []

        def scan(rel: str, lines: list[str]) -> bool:
            for number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append({"path": rel, "line": number, "text": line[:500]})
                    if len(matches) >= limit:
                        return True
            return False

        # Overlay first — the model is usually looking for what it just wrote —
        # then the disk, skipping any file the overlay shadows.
        for rel in sorted(staged):
            if scan(rel, str(staged[rel]["content"]).splitlines()):
                return {"matches": matches, "truncated": True}
        for path, relative in self._iter_files(root):
            rel = relative.as_posix()
            if rel in staged or not _is_text_file(path):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            if scan(rel, lines):
                return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _stage_write(
        self,
        root: Path,
        call: ProjectToolCallV1,
        staged: dict[str, dict[str, Any]],
        next_paths: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        relative = _bounded_line(call.arguments.get("path"), 1_000)
        target = self._safe_target(root, relative, write=True)
        rel = str(target.relative_to(root))
        existing = staged.get(rel)
        # Set when a patch matched on something looser than exact text, so the
        # result can say so rather than letting a forgiven quote look verbatim.
        matched_how = ""

        if call.name == "create_file":
            content = str(call.arguments.get("content", ""))
            if (
                not content
                or len(content.encode("utf-8")) > self.settings.project_max_write_bytes
            ):
                raise ProjectWorkspaceError(
                    "new project file is empty or exceeds the write limit. "
                    'create_file needs the complete file text in "content"; '
                    f"you sent {sorted(call.arguments)}.",
                    argument_shape=True,
                )
            if target.exists() or existing is not None:
                # Naming the alternative matters: a model told only that this is
                # refused will re-send it, and re-send it, until the step budget
                # is gone. "A different path" was still a guess, though — one
                # live build spent 43 create_file calls to produce 11 files. The
                # host knows exactly which file the build still owes, so it says
                # so, and the next step's grammar is pinned to that list.
                owed = next_paths[0] if next_paths else ""
                nudge = (
                    f"write the next file you planned: {owed}."
                    if owed
                    else "write a different path."
                )
                raise ProjectWorkspaceError(
                    "create_file refuses to overwrite an existing or already-staged "
                    f"file. {rel} already exists — change it with apply_patch, or "
                    f"{nudge}",
                    wrong_target=True,
                )
            entry = {
                "content": content,
                "origin": "create",
                "base_sha256": "",
                "bytes": len(content.encode("utf-8")),
            }
        elif call.name == "replace_lines":
            replacement = str(call.arguments.get("replacement", ""))
            if len(replacement.encode("utf-8")) > self.settings.project_max_write_bytes:
                raise ProjectWorkspaceError(
                    "project replacement exceeds the write limit"
                )
            if existing is not None:
                text = str(existing["content"])
                origin = str(existing["origin"])
                base = str(existing["base_sha256"])
            elif target.is_file() and _is_text_file(target):
                text = target.read_text(encoding="utf-8")
                origin = "patch"
                base = hashlib.sha256(text.encode("utf-8")).hexdigest()
            else:
                raise ProjectWorkspaceError(
                    "replace_lines target must be an existing or staged text file"
                )
            updated = _splice_lines(text, call.arguments)
            _guard_whole_file_repair(rel, text, updated, call.arguments)
            entry = {
                "content": updated,
                "origin": origin,
                "base_sha256": base,
                "bytes": len(updated.encode("utf-8")),
            }
        else:
            original = str(call.arguments.get("original", ""))
            replacement = str(call.arguments.get("replacement", ""))
            if not original:
                raise ProjectWorkspaceError(
                    "apply_patch requires a non-empty exact original block. It takes "
                    '{"path","original","replacement"} and is not a diff; you sent '
                    f"{sorted(call.arguments)}. If quoting the block exactly keeps "
                    "failing, use replace_lines with the line range instead.",
                    argument_shape=True,
                    repair_strategy="whole_file",
                )
            if len(replacement.encode("utf-8")) > self.settings.project_max_write_bytes:
                raise ProjectWorkspaceError(
                    "project replacement exceeds the write limit"
                )
            if existing is not None:
                text = str(existing["content"])
                origin = str(existing["origin"])
                base = str(existing["base_sha256"])
            elif target.is_file() and _is_text_file(target):
                text = target.read_text(encoding="utf-8")
                origin = "patch"
                base = hashlib.sha256(text.encode("utf-8")).hexdigest()
            else:
                raise ProjectWorkspaceError(
                    "apply_patch target must be an existing or staged text file"
                )
            located = locate_patch(text, original, replacement)
            if isinstance(located, PatchProblem):
                # The old advice was "refresh the file and narrow it", which
                # made things worse while reads were line-numbered: refreshing
                # returned the same numbered text the model had just copied.
                #
                # A zero match is usually a model patching a file it never read
                # and inventing the block it expects to find. The host has the
                # real text right here, so it sends the opening of it back
                # rather than asking for a read that costs another whole step.
                if located.count == 0:
                    advice = (
                        "nothing in the file matches it, even ignoring whitespace. "
                        "Copy a block out of the current text below — whole lines, "
                        f"no line numbers — or read_file {rel} for the rest of it.\n"
                        f"--- {rel} begins ---\n{text[:400]}"
                    )
                else:
                    advice = "extend it with surrounding lines until it appears once."
                raise ProjectWorkspaceError(
                    f"patch context matched {located.count} times "
                    f"({located.how}); {advice}",
                    argument_shape=True,
                    repair_strategy="whole_file",
                )
            matched_how = located.how
            updated = text[: located.start] + located.replacement + text[located.end :]
            if len(updated.encode("utf-8")) > self.settings.project_max_write_bytes:
                raise ProjectWorkspaceError(
                    "updated project file exceeds the write limit"
                )
            entry = {
                "content": updated,
                "origin": origin,
                "base_sha256": base,
                "bytes": len(updated.encode("utf-8")),
            }

        # The overlay only ever holds text that parses. Checking here rather than
        # at the end of the turn is the difference between one wasted step and a
        # wasted build: a 48-step run once ended at the approval card with a file
        # spliced together from two drafts, and every step after the corruption
        # was spent on a project that could never have run. A refusal is cheap,
        # arrives while the model still has the file in hand, and is classified
        # argument_shape so the next step is pinned to this same tool.
        broken = parse_error(rel, str(entry["content"]))
        if broken:
            hint = (
                "Send the complete corrected file."
                if call.name == "create_file"
                else "Nothing was staged, so the file is unchanged; patch it again."
            )
            raise ProjectWorkspaceError(
                f"{rel} was not staged because it does not parse — {broken}. {hint}",
                argument_shape=True,
                repair_strategy=(
                    "whole_file"
                    if call.name in {"apply_patch", "replace_lines"}
                    else ""
                ),
                repairable_external=True,
                repair_path=rel,
            )

        next_staged = dict(staged)
        next_staged[rel] = entry
        if len(next_staged) > self.settings.project_staged_max_files:
            raise ProjectWorkspaceError(
                f"this turn's staged changeset is capped at "
                f"{self.settings.project_staged_max_files} files; finish and let "
                "the user apply what you have"
            )
        total = sum(int(item["bytes"]) for item in next_staged.values())
        if total > self.settings.project_staged_max_bytes:
            raise ProjectWorkspaceError(
                "this turn's staged changeset exceeds its byte budget; finish "
                "and let the user apply what you have"
            )
        result = {
            "path": rel,
            "staged": True,
            "bytes": int(entry["bytes"]),
            "staged_files": len(next_staged),
            "staged_bytes": total,
        }
        if matched_how and matched_how != EXACT:
            # Visible to the model, the trace and the approval card: the block
            # was found by forgiving whitespace, not by matching what it sent.
            result["matched"] = matched_how
        return result, next_staged

    def staged_summary(
        self, staged: dict[str, dict[str, Any]]
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """The approval card's text, a stable digest, and per-file facts.

        The digest covers every path and its exact staged content, so the
        approval the user grants is bound to precisely these bytes.
        """
        files = [
            {
                "path": rel,
                "origin": str(staged[rel]["origin"]),
                "bytes": int(staged[rel]["bytes"]),
            }
            for rel in sorted(staged)
        ]
        hasher = hashlib.sha256()
        for rel in sorted(staged):
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(str(staged[rel]["content"]).encode("utf-8"))
            hasher.update(b"\x00")
        lines = [
            f"- {'create' if item['origin'] == 'create' else 'modify'} "
            f"`{item['path']}` · {item['bytes']:,} bytes"
            for item in files
        ]
        total = sum(item["bytes"] for item in files)
        summary = (
            f"{len(files)} file(s), {total:,} bytes staged. Nothing has been "
            "written yet; approving applies all of it, rejecting discards all of it.\n\n"
            + "\n".join(lines)
        )
        return summary[:8_000], hasher.hexdigest(), files

    async def materialize_staged(
        self, asset_id: str, staged: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Write an approved changeset to the real tree.

        Every file passes the same jail as a direct write. A file whose disk
        state no longer matches what the model worked from is skipped and
        reported, never overwritten — approval covered the staged bytes, not
        whatever arrived on disk since.
        """
        root = await self.assets.project_path(asset_id)

        def apply() -> dict[str, Any]:
            applied: list[str] = []
            skipped: list[dict[str, str]] = []
            for rel in sorted(staged):
                entry = staged[rel]
                try:
                    # framework=True: overlay provenance already guarantees any
                    # appkit/ entry is host-authored — model writes there are
                    # refused at staging time — so materialize may pass.
                    target = self._safe_target(root, rel, write=True, framework=True)
                except ProjectWorkspaceError as exc:
                    skipped.append({"path": rel, "reason": str(exc)})
                    continue
                origin = str(entry["origin"])
                if origin == "create" and target.exists():
                    skipped.append(
                        {"path": rel, "reason": "a file appeared here after staging"}
                    )
                    continue
                if origin == "patch":
                    if not target.is_file():
                        skipped.append(
                            {
                                "path": rel,
                                "reason": "the file disappeared after staging",
                            }
                        )
                        continue
                    disk_sha = hashlib.sha256(
                        target.read_text(encoding="utf-8").encode("utf-8")
                    ).hexdigest()
                    if disk_sha != str(entry["base_sha256"]):
                        skipped.append(
                            {"path": rel, "reason": "the file changed after staging"}
                        )
                        continue
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(target, str(entry["content"]))
                applied.append(rel)
            if applied:
                self._record_batch_mutation(root, applied)
            return {"applied": applied, "skipped": skipped}

        async with self._lock:
            return await asyncio.to_thread(apply)

    async def ensure_asset_manifest(self, asset_id: str) -> str:
        """Write the framework-owned launch manifest for a just-applied build.

        Models are prohibited from writing under .metis, so the path from
        "reviewed changeset" to "launchable asset" is the host's to provide:
        after an approved apply, Metis derives the manifest from what actually
        reached disk — entry point, dependency file, detected capabilities —
        and writes it itself. An existing manifest is never overwritten (a
        human wrote or reviewed it), and writing one grants nothing by
        itself: launch still requires the separate fingerprint approval.

        Returns the manifest's project-relative path, or "" when nothing was
        written.
        """
        root = await self.assets.project_path(asset_id)

        def write() -> str:
            manifest_path = root / ".metis" / "asset.json"
            if manifest_path.exists():
                return ""
            entry_module = ""
            for candidate in ("app/main.py", "main.py", "app.py", "server.py"):
                file = root / candidate
                if not file.is_file():
                    continue
                try:
                    text = file.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if "FastAPI(" in text:
                    entry_module = candidate[:-3].replace("/", ".") + ":app"
                    break
            if not entry_module:
                return ""  # nothing recognisably launchable; leave it manual
            if (root / "requirements.txt").is_file():
                # uv prepares an isolated environment from the project's own
                # requirements on first launch and reuses its cache after.
                command = [
                    "{uv}",
                    "run",
                    "--with-requirements",
                    "requirements.txt",
                    "uvicorn",
                    entry_module,
                    "--host",
                    "{host}",
                    "--port",
                    "{port}",
                ]
            else:
                command = [
                    "{python}",
                    "-m",
                    "uvicorn",
                    entry_module,
                    "--host",
                    "{host}",
                    "--port",
                    "{port}",
                ]
            capabilities = capabilities_of_tree(root)
            env_keys = sorted(
                {var.name for cap in capabilities for var in CAPABILITY_VARS[cap]}
            )
            manifest = {
                "name": root.name[:120] or "Metis project",
                "summary": "Metis-built application; manifest generated after an approved build",
                "category": "Generated",
                "tags": ["metis-build"],
                "env": env_keys,
                "launch": {"command": command, "path": "/"},
                "metis": {
                    "generated_by": "metis-build",
                    "scaffold_version": SCAFFOLD_VERSION
                    if (root / "appkit").is_dir()
                    else "",
                },
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            return str(manifest_path.relative_to(root))

        async with self._lock:
            return await asyncio.to_thread(write)

    async def stage_scaffold(
        self,
        asset_id: str,
        staged: dict[str, dict[str, Any]],
        capabilities: Iterable[str],
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Seed or upgrade the framework-owned scaffold in the build overlay.

        Existing appkit files are Metis-owned, so canonical changes are staged
        as approval-visible patches pinned to their exact disk hashes. Files
        outside appkit remain seed-only, and unknown appkit extras are left
        alone. All-or-nothing against the changeset budgets: a build too large
        to hold the scaffold proceeds without it rather than dying before the
        first model step.
        """
        root = await self.assets.project_path(asset_id)

        def seed() -> tuple[dict[str, dict[str, Any]], list[str]]:
            def canonical_target(rel: str) -> Path | None:
                lexical = root / rel
                try:
                    resolved = self._safe_target(root, rel, write=True, framework=True)
                except ProjectWorkspaceError:
                    return None
                # `_safe_target` resolves every component. Equality therefore
                # proves no appkit path is being redirected through a symlink.
                return lexical if resolved == lexical else None

            effective_capabilities = set(capabilities)
            existing_paths = set(staged)
            oci_path = canonical_target("appkit/oci_responses.py")
            if oci_path is not None and oci_path.is_file():
                existing_paths.add("appkit/oci_responses.py")
            for rel in ("appkit/web.py", "appkit/static/theme.css"):
                optional_path = canonical_target(rel)
                if optional_path is not None and optional_path.is_file():
                    existing_paths.add(rel)
            if "appkit/oci_responses.py" in existing_paths:
                effective_capabilities.add("oci_responses")
            if existing_paths & {"appkit/web.py", "appkit/static/theme.css"}:
                effective_capabilities.add("web_ui")

            sources = scaffold_sources(effective_capabilities)
            next_staged = dict(staged)
            added: list[str] = []
            for rel in sorted(sources):
                if rel in next_staged:
                    continue
                content = sources[rel]
                target = canonical_target(rel)
                if target is None:
                    continue
                if target.exists():
                    # .env.example and any future non-appkit seed are
                    # project-owned once present; only the declared Metis
                    # boundary is eligible for automatic upgrades.
                    if not rel.startswith("appkit/") or not target.is_file():
                        continue
                    try:
                        disk_content = target.read_text(encoding="utf-8")
                    except (OSError, UnicodeError):
                        continue
                    if disk_content == content:
                        continue
                    origin = "patch"
                    base_sha256 = hashlib.sha256(
                        disk_content.encode("utf-8")
                    ).hexdigest()
                else:
                    origin = "create"
                    base_sha256 = ""
                next_staged[rel] = {
                    "content": content,
                    "origin": origin,
                    "base_sha256": base_sha256,
                    "bytes": len(content.encode("utf-8")),
                }
                added.append(rel)
            total = sum(int(item["bytes"]) for item in next_staged.values())
            if (
                len(next_staged) > self.settings.project_staged_max_files
                or total > self.settings.project_staged_max_bytes
            ):
                return dict(staged), []
            return next_staged, added

        # Reads the vendored sources and stats the project tree; off the loop
        # like every other filesystem step in this service.
        return await asyncio.to_thread(seed)

    async def verify_staged_syntax(
        self, staged: dict[str, dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Parse-check the staged changeset off the event loop; empty when clean.

        Runs before the build approval so a changeset that would not parse is
        sent back to the model to fix instead of being offered for the user to
        apply. Reads nothing from disk — it inspects only the overlay bytes the
        approval would write.
        """
        if not staged:
            return []
        return await asyncio.to_thread(staged_syntax_errors, staged)

    async def verify_staged_wiring(
        self, asset_id: str, staged: dict[str, dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Check that the staged files fit together, without running any of them.

        Parsing proves each file is valid on its own; this proves the changeset
        is a working whole — imports resolve, names exist, what was built got
        wired up. Resolution sees the project on disk as well as the overlay, so
        an edit turn importing untouched code is not reported as broken.
        """
        if not staged or not self.settings.project_wiring_gate_enabled:
            return []
        paths, sources, requirements = await self._static_context(asset_id, staged)
        return await asyncio.to_thread(
            staged_wiring_errors,
            staged,
            sources=sources,
            project_paths=paths,
            requirements=requirements,
        )

    async def verify_staged_types(
        self, staged: dict[str, dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Run ruff and mypy over the changeset against the real packages.

        Nothing is imported or executed — both tools read source only — so this
        is safe on code the model wrote a moment ago, and it is the only rung
        that can see a call the installed library would reject.
        """
        if not staged or not self.settings.project_typecheck_enabled:
            return []
        return await staged_static_analysis(
            staged, timeout_seconds=self.settings.project_typecheck_timeout_seconds
        )

    async def verify_staged_conformance(
        self,
        asset_id: str,
        staged: dict[str, dict[str, Any]],
        planned: list[str] | None = None,
        required: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Check the changeset against what this turn set out to do.

        The rungs below prove the code is well-formed. This one proves it is the
        code the turn committed to: every planned file written, every file the
        user's own request explicitly required, and no frontend call whose body
        the backend would never read. Pure inspection, like the wiring gate —
        the files on disk are consulted so a turn that only edits part of a
        project is not told to re-write what is already there.
        """
        if not staged:
            return []
        paths, sources, _ = await self._static_context(asset_id, staged)

        def check() -> list[dict[str, str]]:
            return [
                *staged_conformance_errors(
                    staged,
                    planned=list(planned or []),
                    required=list(required or []),
                    on_disk=paths,
                ),
                # Contracts BETWEEN files: the defects where every file is
                # individually well-formed and the changeset is still broken.
                *cross_file_findings(staged, on_disk=paths, disk_sources=sources),
            ]

        return await asyncio.to_thread(check)

    async def style_gaps(
        self, asset_id: str, staged: dict[str, dict[str, Any]]
    ) -> dict[str, list[str]]:
        """The classes and custom properties this project's markup needs and lacks.

        Given to the coder when it has been directed at a stylesheet, so writing
        one is a list to satisfy rather than a repository to re-read.
        """
        _, sources, _ = await self._static_context(asset_id, staged)
        sheets = {path: text for path, text in sources.items() if is_stylesheet(path)}
        for path, entry in staged.items():
            if is_stylesheet(path):
                sheets[path] = str(entry.get("content", ""))
        markup = {
            path: text
            for path, text in sources.items()
            if Path(path).suffix.lower()
            in {".jsx", ".tsx", ".vue", ".svelte", ".html", ".htm"}
        }
        if not sheets or not markup:
            return {"classes": [], "variables": []}
        return await asyncio.to_thread(style_gaps, sheets, markup)

    async def verify_staged_runtime(
        self,
        asset_id: str,
        staged: dict[str, dict[str, Any]],
        *,
        scenarios: list[dict[str, Any]] | None = None,
    ) -> SandboxOutcome:
        """Import the staged changeset inside the reviewed container.

        This is the only place model-authored project code is ever executed, and
        it happens in a network-less, read-only, non-root container against a
        throwaway copy — never against the user's project. An unavailable
        sandbox degrades to the static gates and reports why.
        """
        if not staged or self.sandbox is None:
            return SandboxOutcome(available=False, reason="")
        try:
            root = await self.assets.project_path(asset_id)
        except AssetLibraryError as exc:
            return SandboxOutcome(available=False, reason=str(exc))
        paths, _, requirements = await self._static_context(
            asset_id, staged, with_sources=False
        )
        return await self.sandbox.verify(
            root=root,
            staged=staged,
            project_paths=paths,
            requirements=requirements,
            scenarios=scenarios,
        )

    async def _static_context(
        self,
        asset_id: str,
        staged: dict[str, dict[str, Any]],
        *,
        with_sources: bool = True,
    ) -> tuple[list[str], dict[str, str], str]:
        """What the project already contains, as the checkers need to see it.

        Bounded on purpose: a gate that reads an entire large repository to
        judge a ten-file changeset would cost more than the build it is checking.
        """

        try:
            root = await self.assets.project_path(asset_id)
        except AssetLibraryError:
            # An unreadable project simply contributes no extra context; the
            # overlay alone is still worth checking.
            return [], {}, _requirements_from(staged, {})

        def collect() -> tuple[list[str], dict[str, str], str]:
            paths: list[str] = []
            sources: dict[str, str] = {}
            budget = _STATIC_SOURCE_BYTES
            for path, relative in self._iter_files(root):
                text_path = relative.as_posix()
                paths.append(text_path)
                # Python for the wiring gate; stylesheets and package
                # manifests for the cross-file contracts, which cannot answer
                # "does this class exist" from the overlay alone.
                # Python for the wiring gate; stylesheets, markup and package
                # manifests for the cross-file contracts. Markup is read because
                # the check runs BOTH ways: a turn that stages only a stylesheet
                # has to be judged against the components already on disk, which
                # are the ones about to render against it.
                if not with_sources or (
                    path.suffix.lower()
                    not in {
                        ".py",
                        ".pyi",
                        ".css",
                        ".scss",
                        ".sass",
                        ".less",
                        ".jsx",
                        ".tsx",
                        ".vue",
                        ".svelte",
                        ".html",
                        ".htm",
                    }
                    and path.name != "package.json"
                ):
                    continue
                try:
                    size = path.stat().st_size
                    if size > budget:
                        continue
                    sources[text_path] = path.read_text(encoding="utf-8")
                    budget -= size
                except (OSError, UnicodeError):
                    continue
            disk_requirements = ""
            for name in ("requirements.txt", "pyproject.toml"):
                candidate = root / name
                try:
                    if candidate.is_file():
                        disk_requirements += (
                            candidate.read_text(encoding="utf-8") + "\n"
                        )
                except (OSError, UnicodeError):
                    continue
            return paths, sources, _requirements_from(staged, {"": disk_requirements})

        return await asyncio.to_thread(collect)

    def _record_batch_mutation(self, root: Path, applied: list[str]) -> None:
        """One manifest revision and one work-log line for the whole changeset."""
        manifest = self._read_manifest(root)
        timestamp = _now()
        if manifest:
            snapshot, _ = self._snapshot(root)
            manifest.update(snapshot)
            manifest["revision"] = int(manifest.get("revision", 0)) + 1
            manifest["updated_at"] = timestamp
            history = list(manifest.get("recent_changes", []))[-19:]
            history.append(
                {
                    "at": timestamp,
                    "tool": "apply_build",
                    "path": _bounded_line(", ".join(applied), 1_000),
                }
            )
            manifest["recent_changes"] = history
            self._manifest_path(root).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        notes_path = root / ".metis" / "METIS.md"
        if notes_path.is_file() and not notes_path.is_symlink():
            text = notes_path.read_text(encoding="utf-8").rstrip()
            listed = _bounded_line(", ".join(f"`{item}`" for item in applied), 1_000)
            notes_path.write_text(
                f"{text}\n- {timestamp[:10]} · Approved a staged build touching "
                f"{len(applied)} file(s): {listed}.\n",
                encoding="utf-8",
            )

    async def _run_check(
        self, asset_id: str, root: Path, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one check the project declared and the user approved."""
        if self.verification is None or not self.settings.project_verify_enabled:
            raise ProjectWorkspaceError("verification checks are disabled")
        recipe = await asyncio.to_thread(self.verification.recipe, root)
        if not recipe.present:
            raise ProjectWorkspaceError(
                "this project declares no checks; add .metis/verify.json with a "
                'checks array of {"name", "command"} entries to enable verification'
            )
        if recipe.error:
            raise ProjectWorkspaceError(
                f"this project's .metis/verify.json cannot be used: {recipe.error}"
            )
        if not self.verification.is_approved(asset_id, recipe):
            raise VerificationNotApprovedError(
                "the verification recipe for this project has not been approved yet"
            )
        name = _bounded_line(arguments.get("name"), 32).casefold()
        check = recipe.get(name)
        if check is None:
            available = ", ".join(recipe.names) or "none"
            raise ProjectWorkspaceError(
                f"unknown check {name or '(missing)'!r}; this project declares: {available}"
            )
        run = await self.verification.run(root, check)
        return {
            "name": run.name,
            "command": run.command,
            "ok": run.ok,
            "exit_code": run.exit_code,
            "timed_out": run.timed_out,
            "duration_seconds": run.duration_seconds,
            "output": run.output,
            "truncated": run.truncated,
        }

    async def verification_view(self, asset_id: str) -> ProjectVerificationV1:
        """Everything the approval card needs, including the plain-English text."""
        root = await self.assets.project_path(asset_id)
        if self.verification is None:
            return ProjectVerificationV1(
                project_id=asset_id, error="verification checks are disabled"
            )
        recipe = await asyncio.to_thread(self.verification.recipe, root)
        return ProjectVerificationV1(
            project_id=asset_id,
            configured=recipe.present and not recipe.error,
            approved=self.verification.is_approved(asset_id, recipe),
            fingerprint=recipe.fingerprint or None,
            checks=[
                ProjectCheckV1(
                    name=check.name,
                    command=list(check.command),
                    description=check.description,
                    explanation=explain_command(check.command),
                    timeout_seconds=check.timeout_seconds,
                )
                for check in recipe.checks
            ],
            explanation=explain_recipe(recipe),
            boundary=BOUNDARY_NOTICE,
            error=recipe.error or None,
        )

    async def approve_verification(self, asset_id: str) -> ProjectVerificationV1:
        root = await self.assets.project_path(asset_id)
        if self.verification is None:
            raise ProjectWorkspaceError("verification checks are disabled")
        recipe = await asyncio.to_thread(self.verification.recipe, root)
        await self.verification.approve(asset_id, recipe)
        return await self.verification_view(asset_id)

    async def revoke_verification(self, asset_id: str) -> ProjectVerificationV1:
        if self.verification is None:
            raise ProjectWorkspaceError("verification checks are disabled")
        await self.verification.revoke(asset_id)
        return await self.verification_view(asset_id)

    async def record_plan(
        self,
        asset_id: str,
        plan: dict[str, Any],
        *,
        done: list[str] | None = None,
    ) -> None:
        """The current build plan, as a file the next context window can read.

        A plan used to live only in graph state — real enough to gate the
        turn, invisible to the user, and gone from the model's view the moment
        the trace window slid past it. Written down it becomes what the
        published harness work calls a control object: reviewable in the
        project, versioned with it, and — because METIS.md rides into every
        step's context — the one part of the plan that survives any reset.

        `done` marks the files that have actually landed. Best-effort by
        design: a plan that cannot be written must never cost the turn that
        made it, so every failure is swallowed here.
        """
        files = [str(item) for item in (plan.get("files") or [])][:24]
        finished = {str(item) for item in (done or [])}
        intent = _bounded_line(plan.get("intent"), 20) or "build"
        scope = _bounded_line(plan.get("scope"), 20) or "narrow"
        reason = _bounded_line(plan.get("reason"), 300)
        lines = [
            f"### Current build plan · {intent} ({scope}) · {_now()[:10]}",
        ]
        if reason:
            lines.append(f"_Revised: {reason}_")
        lines.extend(
            f"- [{'x' if item in finished else ' '}] `{item}`" for item in files
        )
        if not files:
            lines.append("- (no new files planned)")
        body = "\n".join(lines)
        try:
            async with self._lock:
                root = await self.assets.project_path(asset_id)
                metis_dir = root / ".metis"
                if not metis_dir.is_dir():
                    return
                (metis_dir / "plan.json").write_text(
                    json.dumps(
                        {**plan, "files": files, "done": sorted(finished)}, indent=2
                    ),
                    encoding="utf-8",
                )
                path = metis_dir / "METIS.md"
                if not path.is_file() or path.is_symlink():
                    return
                text = path.read_text(encoding="utf-8")
                block = f"{_PLAN_START}\n{body}\n{_PLAN_END}"
                start = text.find(_PLAN_START)
                end = text.find(_PLAN_END)
                if start >= 0 and end > start:
                    text = text[:start] + block + text[end + len(_PLAN_END) :]
                else:
                    text = text.rstrip() + "\n\n" + block + "\n"
                path.write_text(text, encoding="utf-8")
        except Exception:  # noqa: BLE001 - the plan file is never worth a turn
            return

    async def record_learnings(
        self, asset_id: str, run_id: str, learnings: list[str]
    ) -> None:
        clean = []
        for raw in learnings[:16]:
            value = _bounded_line(raw, 600)
            if value and not _SECRETISH.search(value):
                clean.append(value)
        if not clean:
            return
        async with self._lock:
            root = await self.assets.project_path(asset_id)
            path = root / ".metis" / "METIS.md"
            if not path.is_file() or path.is_symlink():
                return
            text = path.read_text(encoding="utf-8")
            start = text.find(_LEARNINGS_START)
            end = text.find(_LEARNINGS_END)
            if start < 0 or end < start:
                return
            existing = text[start + len(_LEARNINGS_START) : end]
            known = {
                line[2:].strip().casefold()
                for line in existing.splitlines()
                if line.startswith("- ")
            }
            additions = [item for item in clean if item.casefold() not in known]
            if not additions:
                return
            body = (
                existing.rstrip()
                + "\n"
                + "\n".join(f"- {item}" for item in additions)
                + "\n"
            )
            text = text[: start + len(_LEARNINGS_START)] + body + text[end:]
            log = f"\n- {_now()[:10]} · `{run_id}` · captured {len(additions)} durable learning(s).\n"
            text = text.rstrip() + log
            path.write_text(text, encoding="utf-8")

    async def _asset_metadata(self, asset_id: str) -> dict[str, Any]:
        for asset in await self.assets.list():
            if asset.id == asset_id:
                return {
                    "id": asset.id,
                    "name": asset.name,
                    "summary": asset.summary,
                    "framework": asset.framework,
                    "entrypoint": asset.entrypoint,
                    "tags": asset.tags,
                }
        raise AssetLibraryError("project is not in the saved Asset catalog")

    def _manifest_path(self, root: Path) -> Path:
        return root / ".metis" / "project-context.json"

    def _read_manifest(self, root: Path) -> dict[str, Any]:
        path = self._manifest_path(root)
        try:
            if path.is_symlink():
                return {}
            value = json.loads(path.read_text(encoding="utf-8"))
            return (
                value
                if isinstance(value, dict) and value.get("schema_version") == "1"
                else {}
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _iter_files(self, root: Path):
        count = 0
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirs[:] = sorted(
                item
                for item in dirs
                if item not in _IGNORE_DIRS
                and not item.startswith(".")
                and not (current_path / item).is_symlink()
            )
            for name in sorted(files):
                path = current_path / name
                if (
                    path.is_symlink()
                    or name.startswith(".env")
                    or path.suffix.casefold() in _SENSITIVE_SUFFIXES
                ):
                    continue
                relative = path.relative_to(root)
                if relative.parts[:1] == (".metis",):
                    continue
                yield path, relative
                count += 1
                if count >= self.settings.project_manifest_max_files:
                    return

    def _snapshot(self, root: Path) -> tuple[dict[str, Any], str]:
        paths: list[str] = []
        languages: Counter[str] = Counter()
        total_bytes = 0
        digest = hashlib.sha256()
        for path, relative in self._iter_files(root):
            rel = relative.as_posix()
            paths.append(rel)
            try:
                stat = path.stat()
            except OSError:
                continue
            total_bytes += stat.st_size
            digest.update(rel.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            language = _LANGUAGE_BY_SUFFIX.get(path.suffix.casefold())
            if language:
                languages[language] += 1

        samples: list[str] = []
        remaining = self.settings.project_manifest_sample_chars
        priority = list(_PRIORITY_FILES)
        priority.extend(
            path for path in paths if path.endswith(("/AGENTS.md", "/README.md"))
        )
        for relative in dict.fromkeys(priority):
            if remaining <= 0:
                break
            path = root / relative
            if not path.is_file() or path.is_symlink():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            excerpt = content[: min(remaining, 16_000)]
            samples.append(f"--- {relative} ---\n{excerpt}")
            remaining -= len(excerpt)
        snapshot = {
            "file_count": len(paths),
            "total_bytes": total_bytes,
            "tree_digest": digest.hexdigest(),
            "languages": dict(languages.most_common()),
            "key_files": [item for item in paths if Path(item).name in _PRIORITY_FILES][
                :80
            ],
            "file_tree": paths[:2_000],
            "truncated": len(paths) >= self.settings.project_manifest_max_files,
        }
        return snapshot, "\n\n".join(samples)

    def _initial_notes(
        self, metadata: dict[str, Any], bootstrap: ProjectBootstrapV1
    ) -> str:
        def bullets(values: list[str], empty: str) -> str:
            cleaned = [
                _bounded_line(item, 600) for item in values if _bounded_line(item, 600)
            ]
            return "\n".join(f"- {item}" for item in cleaned) or f"- {empty}"

        return (
            f"{_MANAGED_HEADER}\n"
            f"# {metadata['name']} — Metis project context\n\n"
            "> Metis created this local working map on first access. Keep durable "
            "project facts here; never place credentials or secret values in this file.\n\n"
            "## Project overview\n\n"
            f"{bootstrap.summary.strip()}\n\n"
            "## Architecture\n\n"
            f"{bullets(bootstrap.architecture, 'Inspect the project further as work requires.')}\n\n"
            "## Important paths\n\n"
            f"{bullets(bootstrap.important_paths, 'No important paths recorded yet.')}\n\n"
            "## Working conventions\n\n"
            f"{bullets(bootstrap.conventions, 'Follow the repository’s existing patterns.')}\n\n"
            "## Verification\n\n"
            f"{bullets(bootstrap.verification, 'Confirm the relevant checks before finishing a change.')}\n\n"
            "## Known risks and open questions\n\n"
            f"{bullets(bootstrap.risks, 'No project-specific risks recorded yet.')}\n\n"
            "## Durable learnings\n\n"
            f"{_LEARNINGS_START}\n"
            "- Initial structure mapped by Grok through OCI Responses.\n"
            f"{_LEARNINGS_END}\n\n"
            "## Work log\n\n"
            f"- {_now()[:10]} · Initial manifest and working context created.\n"
        )

    def _write_initial_notes(
        self,
        path: Path,
        metadata: dict[str, Any],
        bootstrap: ProjectBootstrapV1,
    ) -> None:
        generated = self._initial_notes(metadata, bootstrap)
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if _MANAGED_HEADER in existing:
                # A missing/corrupt JSON manifest must not erase a surviving
                # project memory file. Its managed markers remain usable.
                return
            generated = existing.rstrip() + "\n\n---\n\n" + generated
        path.write_text(generated, encoding="utf-8")

    def _record_mutation(
        self, root: Path, call: ProjectToolCallV1, result: dict[str, Any]
    ) -> None:
        """Keep the deterministic map and work log current after an approved edit."""
        manifest = self._read_manifest(root)
        if manifest:
            snapshot, _ = self._snapshot(root)
            manifest.update(snapshot)
            manifest["revision"] = int(manifest.get("revision", 0)) + 1
            manifest["updated_at"] = _now()
            history = list(manifest.get("recent_changes", []))[-19:]
            history.append(
                {
                    "at": manifest["updated_at"],
                    "tool": call.name,
                    "path": _bounded_line(result.get("path"), 1_000),
                }
            )
            manifest["recent_changes"] = history
            self._manifest_path(root).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        notes_path = root / ".metis" / "METIS.md"
        if notes_path.is_file() and not notes_path.is_symlink():
            text = notes_path.read_text(encoding="utf-8").rstrip()
            path = _bounded_line(result.get("path"), 1_000)
            notes_path.write_text(
                f"{text}\n- {_now()[:10]} · Approved `{call.name}` on `{path}`.\n",
                encoding="utf-8",
            )

    def _safe_target(
        self, root: Path, relative: str, *, write: bool = False, framework: bool = False
    ) -> Path:
        if not relative or len(relative) > 1_000:
            raise ProjectWorkspaceError("project paths must be non-empty and bounded")
        while relative.startswith("./"):
            relative = relative[2:]
        candidate_path = Path(relative)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            # A refusal the model cannot act on is a refusal it repeats. Naming
            # the convention and showing the corrected form turns a dead end
            # into a step it can retry, which is the difference between one
            # wasted call and a build that gives up on writing anything.
            raise ProjectWorkspaceError(
                f"'{relative[:120]}' is not a project path. Paths are relative to "
                "the project root, with no leading slash and no '..' — write "
                f"'{_suggest_relative(relative)}' instead."
            )
        if candidate_path.parts and candidate_path.parts[0] in {".git", ".metis"}:
            raise ProjectWorkspaceError(
                "models cannot directly change Metis or source-control internals"
            )
        # Model writes stop at the scaffold boundary; reads pass. `framework`
        # marks the host's own writes — staging the scaffold, materializing an
        # approved overlay — which are the only way appkit/ content ever moves.
        if (
            write
            and not framework
            and candidate_path.parts
            and candidate_path.parts[0] == "appkit"
        ):
            raise ProjectWorkspaceError(
                "appkit/ is Metis-owned scaffold: import it from your application "
                "modules instead of editing it. Write your changes elsewhere."
            )
        name = candidate_path.name.casefold()
        if (
            name.startswith(".env") and name not in _ENV_TEMPLATE_NAMES
        ) or candidate_path.suffix.casefold() in _SENSITIVE_SUFFIXES:
            raise ProjectWorkspaceError(
                "environment and secret files are not exposed to project tools. "
                "An example file that only names the variables — .env.example — "
                "is allowed; a real .env is not."
            )
        candidate = (root / candidate_path).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ProjectWorkspaceError(
                "project path escaped the selected project"
            ) from exc
        if candidate.exists() and candidate.is_symlink():
            raise ProjectWorkspaceError("project tools do not follow symbolic links")
        if write and candidate.exists() and not candidate.is_file():
            raise ProjectWorkspaceError(
                "project mutation target must be a regular file"
            )
        return candidate

    def _list_files(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        prefix = _project_prefix(arguments.get("path", ""))
        limit = min(max(int(arguments.get("limit", 200)), 1), 500)
        files = []
        for _, relative in self._iter_files(root):
            rel = relative.as_posix()
            if not prefix or rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
                files.append(rel)
                if len(files) >= limit:
                    break
        return {"files": files, "truncated": len(files) >= limit}

    def _read_file(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        path = self._safe_target(root, relative)
        if not path.is_file() or not _is_text_file(path):
            raise ProjectWorkspaceError(
                "project file is unavailable or not readable text"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProjectWorkspaceError(
                "project file could not be decoded as UTF-8"
            ) from exc
        return {
            **_read_window(text, arguments, self.settings.project_tool_result_chars),
            "path": relative,
        }

    def _search_code(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _bounded_line(arguments.get("query"), 300)
        if not query:
            raise ProjectWorkspaceError("search_code requires a query")
        limit = min(max(int(arguments.get("limit", 80)), 1), 200)
        case_sensitive = bool(arguments.get("case_sensitive", False))
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        for path, relative in self._iter_files(root):
            if not _is_text_file(path):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append(
                        {
                            "path": relative.as_posix(),
                            "line": number,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _replace_lines(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        path = self._safe_target(root, relative, write=True)
        if not path.is_file() or not _is_text_file(path):
            raise ProjectWorkspaceError(
                "replace_lines target must be an existing text file"
            )
        if (
            len(str(arguments.get("replacement", "")).encode("utf-8"))
            > self.settings.project_max_write_bytes
        ):
            raise ProjectWorkspaceError("project replacement exceeds the write limit")
        text = path.read_text(encoding="utf-8")
        updated = _splice_lines(text, arguments)
        _guard_whole_file_repair(relative, text, updated, arguments)
        if len(updated.encode("utf-8")) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("updated project file exceeds the write limit")
        path.write_text(updated, encoding="utf-8")
        return {
            "path": relative,
            "changed": True,
            "before_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "after_sha256": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
        }

    def _apply_patch(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        original = str(arguments.get("original", ""))
        replacement = str(arguments.get("replacement", ""))
        if not original:
            raise ProjectWorkspaceError(
                "apply_patch requires a non-empty exact original block"
            )
        if len(replacement.encode("utf-8")) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("project replacement exceeds the write limit")
        path = self._safe_target(root, relative, write=True)
        if not path.is_file() or not _is_text_file(path):
            raise ProjectWorkspaceError(
                "apply_patch target must be an existing text file"
            )
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(original)
        if occurrences != 1:
            raise ProjectWorkspaceError(
                f"exact patch context matched {occurrences} times; refresh the file and narrow it"
            )
        updated = text.replace(original, replacement, 1)
        if len(updated.encode("utf-8")) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("updated project file exceeds the write limit")
        path.write_text(updated, encoding="utf-8")
        return {
            "path": relative,
            "changed": True,
            "before_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "after_sha256": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
        }

    def _create_file(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        content = str(arguments.get("content", ""))
        encoded = content.encode("utf-8")
        if not content or len(encoded) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError(
                "new project file is empty or exceeds the write limit"
            )
        path = self._safe_target(root, relative, write=True)
        if path.exists():
            raise ProjectWorkspaceError(
                "create_file refuses to overwrite an existing file"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": relative,
            "created": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
