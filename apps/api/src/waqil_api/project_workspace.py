"""Explicit, bounded project workspaces for Metis chat.

The saved Asset catalog is the grant boundary: selecting a project never scans
for new siblings. A deterministic local manifest and an evolving METIS.md live
inside the project. Models can inspect the grant through narrow tools; exact
mutations are executed only after the control plane records user approval.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

from .asset_library import AssetLibraryError, AssetManager
from .config import Settings
from .contracts import (
    ProjectBootstrapV1,
    ProjectCheckV1,
    ProjectToolCallV1,
    ProjectVerificationV1,
    ProjectWorkspaceV1,
)
from .project_conformance import staged_conformance_errors
from .project_lookup import LookupError_, inspect_installed_api
from .project_env import CAPABILITY_VARS, capabilities_of_tree
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
    """

    def __init__(
        self, message: str, *, argument_shape: bool = False, wrong_target: bool = False
    ) -> None:
        super().__init__(message)
        self.argument_shape = argument_shape
        self.wrong_target = wrong_target


class VerificationNotApprovedError(ProjectWorkspaceError):
    """The project declares checks the user has not reviewed yet.

    Distinct from a plain tool error so the control plane can raise a one-time
    approval instead of handing the agent a dead end it cannot act on.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded_line(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


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
    ) -> None:
        self.settings = settings
        self.assets = assets
        self.bootstrap_model = bootstrap_model
        self.verification = verification
        self.sandbox = sandbox
        self._lock = asyncio.Lock()

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
        """Create or refresh local context; Grok is called only on first access."""
        async with self._lock:
            root = await self.assets.project_path(asset_id)
            metadata = await self._asset_metadata(asset_id)
            metis_dir = root / ".metis"
            manifest_path = metis_dir / "project-context.json"
            notes_path = metis_dir / "METIS.md"
            if metis_dir.exists() and metis_dir.is_symlink():
                raise ProjectWorkspaceError("the project's .metis folder may not be a symlink")
            if manifest_path.is_symlink() or notes_path.is_symlink():
                raise ProjectWorkspaceError("Metis project context files may not be symlinks")
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
            if bootstrap is None:
                available = getattr(self.bootstrap_model, "available", None)
                if available is None and hasattr(self.bootstrap_model, "oci"):
                    available = getattr(self.bootstrap_model.oci, "available", False)
                if available is False:
                    raise ProjectWorkspaceError(
                        "Grok project bootstrap requires OCI Responses to be configured"
                    )
                bootstrap = await self.bootstrap_model.bootstrap_project(
                    {
                        "project": metadata,
                        "manifest": snapshot,
                        "bounded_file_samples": sample,
                    }
                )
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
                "created_at": prior.get("created_at", timestamp) if prior else timestamp,
                "updated_at": timestamp,
                "bootstrap_provider": "oci-grok",
                "bootstrap_model": self.settings.oci_grok_model,
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
            raise ProjectWorkspaceError("open this project once before starting a project chat")
        notes = notes_path.read_text(encoding="utf-8")[:40_000]
        return {
            "project_id": asset_id,
            "project_name": manifest.get("project_name", root.name),
            "manifest": manifest,
            "metis_md": notes,
            "verification": await self._verification_context(asset_id, root),
        }

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
        if call.name not in {"apply_patch", "create_file"}:
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
        else:
            content = str(call.arguments.get("content", ""))
            detail = f"Create {relative} with {len(content.encode('utf-8'))} bytes."
        digest = hashlib.sha256(
            json.dumps(call.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {"path": str(target.relative_to(root)), "summary": detail, "digest": digest}

    async def execute(self, asset_id: str, call: ProjectToolCallV1) -> dict[str, Any]:
        root = await self.assets.project_path(asset_id)
        if call.name == "list_files":
            return await asyncio.to_thread(self._list_files, root, call.arguments)
        if call.name == "search_code":
            return await asyncio.to_thread(self._search_code, root, call.arguments)
        if call.name == "read_file":
            return await asyncio.to_thread(self._read_file, root, call.arguments)
        if call.name == "apply_patch":
            async with self._lock:
                result = await asyncio.to_thread(self._apply_patch, root, call.arguments)
                await asyncio.to_thread(self._record_mutation, root, call, result)
                return result
        if call.name == "create_file":
            async with self._lock:
                result = await asyncio.to_thread(self._create_file, root, call.arguments)
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
        if call.name in {"apply_patch", "create_file"}:
            return await asyncio.to_thread(
                self._stage_write, root, call, dict(staged), tuple(next_paths)
            )
        raise ProjectWorkspaceError(
            f"project tool {call.name} cannot run against staged changes"
        )

    async def _inspect_api(self, call: ProjectToolCallV1) -> dict[str, Any]:
        """The real shape of an installed library, read rather than recalled."""
        try:
            return await inspect_installed_api(
                str(call.arguments.get("module", "")),
                str(call.arguments.get("symbol", "") or ""),
                project_roots=tuple(self.settings.asset_roots),
            )
        except LookupError_ as exc:
            raise ProjectWorkspaceError(str(exc), argument_shape=True) from exc

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
            return not prefix or rel == prefix or rel.startswith(prefix.rstrip("/") + "/")

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

        if call.name == "create_file":
            content = str(call.arguments.get("content", ""))
            if not content or len(content.encode("utf-8")) > self.settings.project_max_write_bytes:
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
        else:
            original = str(call.arguments.get("original", ""))
            replacement = str(call.arguments.get("replacement", ""))
            if not original:
                raise ProjectWorkspaceError(
                    "apply_patch requires a non-empty exact original block. It takes "
                    '{"path","original","replacement"} and is not a diff; you sent '
                    f"{sorted(call.arguments)}.",
                    argument_shape=True,
                )
            if len(replacement.encode("utf-8")) > self.settings.project_max_write_bytes:
                raise ProjectWorkspaceError("project replacement exceeds the write limit")
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
            occurrences = text.count(original)
            if occurrences != 1:
                # The old advice was "refresh the file and narrow it", which
                # made things worse while reads were line-numbered: refreshing
                # returned the same numbered text the model had just copied.
                #
                # A zero match is usually a model patching a file it never read
                # and inventing the block it expects to find. The host has the
                # real text right here, so it sends the opening of it back
                # rather than asking for a read that costs another whole step.
                if occurrences == 0:
                    advice = (
                        "nothing in the file matches it. Copy an exact block out of "
                        "the current text below — no line numbers, exact indentation "
                        f"— or read_file {rel} for the rest of it.\n"
                        f"--- {rel} begins ---\n{text[:400]}"
                    )
                else:
                    advice = "extend it with surrounding lines until it appears once."
                raise ProjectWorkspaceError(
                    f"exact patch context matched {occurrences} times; {advice}",
                    argument_shape=True,
                )
            updated = text.replace(original, replacement, 1)
            if len(updated.encode("utf-8")) > self.settings.project_max_write_bytes:
                raise ProjectWorkspaceError("updated project file exceeds the write limit")
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
                            {"path": rel, "reason": "the file disappeared after staging"}
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
                target.write_text(str(entry["content"]), encoding="utf-8")
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
                    "{uv}", "run", "--with-requirements", "requirements.txt",
                    "uvicorn", entry_module, "--host", "{host}", "--port", "{port}",
                ]
            else:
                command = [
                    "{python}", "-m", "uvicorn", entry_module,
                    "--host", "{host}", "--port", "{port}",
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
        """Seed the framework-owned scaffold into a build turn's overlay.

        Only paths absent from both the overlay and the disk are added, so a
        rebuild in a project that already carries appkit stages nothing and
        the approval card stays about the model's own work. All-or-nothing
        against the changeset budgets: a build too large to hold the scaffold
        proceeds without it rather than dying before the first model step.
        """
        root = await self.assets.project_path(asset_id)

        def seed() -> tuple[dict[str, dict[str, Any]], list[str]]:
            sources = scaffold_sources(capabilities)
            next_staged = dict(staged)
            added: list[str] = []
            for rel in sorted(sources):
                if rel in next_staged or (root / rel).exists():
                    continue
                content = sources[rel]
                next_staged[rel] = {
                    "content": content,
                    "origin": "create",
                    "base_sha256": "",
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
    ) -> list[dict[str, str]]:
        """Check the changeset against what this turn set out to do.

        The rungs below prove the code is well-formed. This one proves it is the
        code the turn committed to: every planned file written, and no frontend
        call whose body the backend would never read. Pure inspection, like the
        wiring gate — the files on disk are consulted so a turn that only edits
        part of a project is not told to re-write what is already there.
        """
        if not staged:
            return []
        paths, _, _ = await self._static_context(asset_id, staged)
        return await asyncio.to_thread(
            staged_conformance_errors,
            staged,
            planned=list(planned or []),
            on_disk=paths,
        )

    async def verify_staged_runtime(
        self, asset_id: str, staged: dict[str, dict[str, Any]]
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
        paths, _, requirements = await self._static_context(asset_id, staged, with_sources=False)
        return await self.sandbox.verify(
            root=root, staged=staged, project_paths=paths, requirements=requirements
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
                if not with_sources or path.suffix.lower() not in {".py", ".pyi"}:
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
                        disk_requirements += candidate.read_text(encoding="utf-8") + "\n"
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
            known = {line[2:].strip().casefold() for line in existing.splitlines() if line.startswith("- ")}
            additions = [item for item in clean if item.casefold() not in known]
            if not additions:
                return
            body = existing.rstrip() + "\n" + "\n".join(f"- {item}" for item in additions) + "\n"
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
            return value if isinstance(value, dict) and value.get("schema_version") == "1" else {}
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
        priority.extend(path for path in paths if path.endswith(("/AGENTS.md", "/README.md")))
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
            "key_files": [item for item in paths if Path(item).name in _PRIORITY_FILES][:80],
            "file_tree": paths[:2_000],
            "truncated": len(paths) >= self.settings.project_manifest_max_files,
        }
        return snapshot, "\n\n".join(samples)

    def _initial_notes(
        self, metadata: dict[str, Any], bootstrap: ProjectBootstrapV1
    ) -> str:
        def bullets(values: list[str], empty: str) -> str:
            cleaned = [_bounded_line(item, 600) for item in values if _bounded_line(item, 600)]
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
            generated = (
                existing.rstrip()
                + "\n\n---\n\n"
                + generated
            )
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
            raise ProjectWorkspaceError("models cannot directly change Metis or source-control internals")
        # Model writes stop at the scaffold boundary; reads pass. `framework`
        # marks the host's own writes — staging the scaffold, materializing an
        # approved overlay — which are the only way appkit/ content ever moves.
        if write and not framework and candidate_path.parts and candidate_path.parts[0] == "appkit":
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
            raise ProjectWorkspaceError("project path escaped the selected project") from exc
        if candidate.exists() and candidate.is_symlink():
            raise ProjectWorkspaceError("project tools do not follow symbolic links")
        if write and candidate.exists() and not candidate.is_file():
            raise ProjectWorkspaceError("project mutation target must be a regular file")
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
            raise ProjectWorkspaceError("project file is unavailable or not readable text")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProjectWorkspaceError("project file could not be decoded as UTF-8") from exc
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
                        {"path": relative.as_posix(), "line": number, "text": line[:500]}
                    )
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _apply_patch(self, root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = _bounded_line(arguments.get("path"), 1_000)
        original = str(arguments.get("original", ""))
        replacement = str(arguments.get("replacement", ""))
        if not original:
            raise ProjectWorkspaceError("apply_patch requires a non-empty exact original block")
        if len(replacement.encode("utf-8")) > self.settings.project_max_write_bytes:
            raise ProjectWorkspaceError("project replacement exceeds the write limit")
        path = self._safe_target(root, relative, write=True)
        if not path.is_file() or not _is_text_file(path):
            raise ProjectWorkspaceError("apply_patch target must be an existing text file")
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
            raise ProjectWorkspaceError("new project file is empty or exceeds the write limit")
        path = self._safe_target(root, relative, write=True)
        if path.exists():
            raise ProjectWorkspaceError("create_file refuses to overwrite an existing file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": relative,
            "created": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
