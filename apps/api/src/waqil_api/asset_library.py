from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import socket
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import AssetEnvVarV1, AssetLogsV1, AssetStatus, AssetV1


_README_LIMIT = 64 * 1024
_MANIFEST_LIMIT = 32 * 1024
_METADATA_LIMIT = 64 * 1024
_LOG_LIMIT = 64 * 1024
_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
# A folder name, never a path: no separators, no leading dot, bounded length.
_PROJECT_FOLDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")
_SAFE_ID = re.compile(r"^asset_[0-9a-f]{20}$")
_RESERVED_ENV_EXACT = {
    "BASH_ENV",
    "ENV",
    "HOME",
    "HOST",
    "IFS",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "METIS_ASSET_ID",
    "METIS_HOST",
    "METIS_PORT",
    "NODE_OPTIONS",
    "PATH",
    "PORT",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "SHELL",
    "TMPDIR",
}
_RESERVED_ENV_PREFIXES = ("DYLD_", "LD_", "METIS_")
_ENV_FILE_LIMIT = 256 * 1024
_ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_ENV_BARE_VALUE = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")
_SENSITIVE_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")


class AssetLibraryError(ValueError):
    """A safe, user-facing asset-library conflict."""


class UnknownAssetError(AssetLibraryError):
    pass


class LaunchNotApprovedError(AssetLibraryError):
    pass


class AssetEnvironmentError(AssetLibraryError):
    pass


@dataclass(frozen=True, slots=True)
class AssetRecord:
    id: str
    path: Path
    name: str
    summary: str
    category: str
    tags: tuple[str, ...]
    framework: str | None
    entrypoint: str | None
    env_keys: tuple[str, ...]
    command: tuple[str, ...] | None
    launch_fingerprint: str | None = None
    launch_path: str = ""


@dataclass(slots=True)
class _ProcessRun:
    process: asyncio.subprocess.Process
    port: int
    url: str
    secrets: tuple[str, ...]
    logs: str = ""
    truncated: bool = False
    requested_stop: bool = False
    ready: bool = False
    startup_failed: bool = False
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    reader_task: asyncio.Task[None] | None = None
    readiness_task: asyncio.Task[None] | None = None

    def append_logs(self, text: str) -> None:
        if not text:
            return
        self.logs += text
        if len(self.logs) > _LOG_LIMIT:
            self.logs = self.logs[-_LOG_LIMIT:]
            self.truncated = True


@dataclass(slots=True)
class _ManifestMetadata:
    name: str | None = None
    summary: str | None = None
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    env_keys: list[str] = field(default_factory=list)
    entrypoint: str | None = None
    command: tuple[str, ...] | None = None
    launch_path: str = ""


def _is_reserved_env(key: str) -> bool:
    upper = key.upper()
    return upper in _RESERVED_ENV_EXACT or upper.startswith(_RESERVED_ENV_PREFIXES)


def _single_line(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return text[:limit].rstrip()


def _slug_tag(value: object) -> str | None:
    text = _single_line(value, 40)
    if text is None:
        return None
    slug = re.sub(r"[^a-z0-9+.#-]+", "-", text.casefold()).strip("-")
    return slug[:32].rstrip("-") or None


def _read_text(path: Path, limit: int, *, reject_oversize: bool = False) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except (OSError, PermissionError):
        return None
    if reject_oversize and len(raw) > limit:
        return None
    return raw[:limit].decode("utf-8", errors="replace").replace("\x00", "")


def _read_project_file(project: Path, relative: str, limit: int) -> str | None:
    """Read one known metadata file without following links outside the asset."""
    candidate = project.joinpath(*relative.split("/"))
    try:
        current = project
        for part in relative.split("/"):
            current = current / part
            if current.is_symlink():
                return None
        candidate.resolve(strict=True).relative_to(project.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return _read_text(candidate, limit)


def _readme_text(project: Path) -> str:
    try:
        candidates = sorted(
            (
                item
                for item in project.iterdir()
                if item.name.casefold().startswith("readme")
                and not item.name.startswith(".")
                and item.is_file()
                and not item.is_symlink()
            ),
            key=lambda item: (
                item.name.casefold() != "readme.md",
                item.name.casefold() != "readme",
                item.name.casefold(),
            ),
        )
    except (OSError, PermissionError):
        return ""
    if not candidates:
        return ""
    return _read_text(candidates[0], _README_LIMIT) or ""


def _readme_title(readme: str) -> str | None:
    for line in readme.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return _single_line(re.sub(r"[`*_]", "", match.group(1)), 120)
    return None


def _readme_summary(readme: str) -> str | None:
    paragraph: list[str] = []
    in_fence = False
    for raw_line in readme.splitlines():
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line:
            if paragraph:
                break
            continue
        if (
            line.startswith("#")
            or line.startswith("![")
            or line.startswith("<")
            or re.match(r"^\[!?[^]]*\]:", line)
            or re.fullmatch(r"[-=_]{3,}", line)
        ):
            continue
        line = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"^[>*+-]\s+", "", line)
        line = re.sub(r"[`*_~]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            paragraph.append(line)
        if len(" ".join(paragraph)) >= 240:
            break
    return _single_line(" ".join(paragraph), 240) if paragraph else None


def _folder_name(name: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:120] or "Untitled asset"


def _manifest(project: Path) -> _ManifestMetadata:
    metadata = _ManifestMetadata()
    manifest_dir = project / ".metis"
    manifest_path = manifest_dir / "asset.json"
    try:
        if manifest_dir.is_symlink() or manifest_path.is_symlink():
            return metadata
        manifest_path.resolve(strict=True).relative_to(project.resolve(strict=True))
    except (OSError, ValueError):
        return metadata
    raw = _read_text(manifest_path, _MANIFEST_LIMIT, reject_oversize=True)
    if raw is None:
        return metadata
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError):
        return metadata
    if not isinstance(body, dict):
        return metadata

    metadata.name = _single_line(body.get("name"), 120)
    metadata.summary = _single_line(
        body.get("summary", body.get("description")), 240
    )
    metadata.category = _single_line(body.get("category"), 60)
    metadata.entrypoint = _single_line(body.get("entrypoint"), 240)
    if isinstance(body.get("tags"), list):
        metadata.tags = [
            tag for item in body["tags"] if (tag := _slug_tag(item)) is not None
        ][:16]

    declared_env: list[object] = []
    for field_name in ("env_keys", "env"):
        value = body.get(field_name)
        if isinstance(value, list):
            declared_env.extend(value)
        elif isinstance(value, dict):
            declared_env.extend(value.keys())

    launch = body.get("launch")
    if isinstance(launch, dict):
        launch_env = launch.get("env")
        if isinstance(launch_env, list):
            declared_env.extend(launch_env)
        elif isinstance(launch_env, dict):
            declared_env.extend(launch_env.keys())

    env_keys: set[str] = set()
    invalid_env = False
    for item in declared_env:
        if isinstance(item, dict):
            item = item.get("key", item.get("name"))
        if not isinstance(item, str) or not _ENV_KEY.fullmatch(item):
            invalid_env = True
            continue
        if _is_reserved_env(item):
            invalid_env = True
            continue
        env_keys.add(item)
    metadata.env_keys = sorted(env_keys)

    if not isinstance(launch, dict):
        return metadata

    command = launch.get("command")
    if (
        invalid_env
        or not isinstance(command, list)
        or not 1 <= len(command) <= 32
        or not all(type(item) is str for item in command)
        or any(
            not item
            or len(item) > 1_024
            or any(character in item for character in ("\x00", "\r", "\n"))
            for item in command
        )
        or sum(len(item) for item in command) > 8_192
    ):
        return metadata
    metadata.command = tuple(command)
    launch_path = launch.get("path", launch.get("url_path", ""))
    if (
        isinstance(launch_path, str)
        and len(launch_path) <= 240
        and (not launch_path or launch_path.startswith("/"))
        and not launch_path.startswith("//")
        and "\r" not in launch_path
        and "\n" not in launch_path
    ):
        metadata.launch_path = launch_path
    return metadata


def _infer_env_keys(project: Path, readme: str) -> list[str]:
    keys: set[str] = set()
    for filename in (".env.example", ".env.sample", "example.env"):
        text = _read_project_file(project, filename, _METADATA_LIMIT)
        if text:
            for line in text.splitlines():
                match = re.match(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]{0,63})\s*=", line)
                if match and not _is_reserved_env(match.group(1)):
                    keys.add(match.group(1))
    for pattern in (
        r"\$\{([A-Z][A-Z0-9_]{2,63})\}",
        r"\$([A-Z][A-Z0-9_]{2,63})\b",
        r"`([A-Z][A-Z0-9_]*_[A-Z0-9_]+)`",
    ):
        for key in re.findall(pattern, readme):
            if not _is_reserved_env(key):
                keys.add(key)

    # Source references reveal configuration names without ever reading values
    # from a real .env file. Keep this deliberately shallow and bounded.
    source_files: list[Path] = []
    allowed_suffixes = {".py", ".js", ".jsx", ".ts", ".tsx"}
    try:
        source_files.extend(
            item
            for item in project.iterdir()
            if item.is_file()
            and not item.is_symlink()
            and item.suffix.casefold() in allowed_suffixes
        )
        for child in project.iterdir():
            if (
                child.is_dir()
                and not child.is_symlink()
                and not child.name.startswith(".")
                and child.name not in {"node_modules", "venv", ".venv", "dist", "build"}
            ):
                source_files.extend(
                    item
                    for item in child.iterdir()
                    if item.is_file()
                    and not item.is_symlink()
                    and item.suffix.casefold() in allowed_suffixes
                )
    except (OSError, PermissionError):
        pass
    source_patterns = (
        r"os\.getenv\(\s*['\"]([A-Z][A-Z0-9_]{1,63})['\"]",
        r"os\.environ(?:\.get)?\(\s*['\"]([A-Z][A-Z0-9_]{1,63})['\"]",
        r"os\.environ\[\s*['\"]([A-Z][A-Z0-9_]{1,63})['\"]\s*\]",
        r"process\.env\.([A-Z][A-Z0-9_]{1,63})\b",
        r"process\.env\[\s*['\"]([A-Z][A-Z0-9_]{1,63})['\"]\s*\]",
    )
    for source in sorted(source_files, key=lambda item: str(item).casefold())[:40]:
        text = _read_text(source, _METADATA_LIMIT) or ""
        for pattern in source_patterns:
            for key in re.findall(pattern, text):
                if not _is_reserved_env(key):
                    keys.add(key)
    return sorted(keys)[:64]


def _is_sensitive_env(key: str) -> bool:
    return _SENSITIVE_ENV.search(key.upper()) is not None


def _env_file_path(project: Path) -> Path:
    return project / ".env"


def _split_env_value(raw: str) -> tuple[str, str]:
    """Split an assignment's right-hand side into its value and trailing note.

    Scanning from the left is what lets `KEY="a b"  # note` read correctly: the
    quote closes the value, so the note is never folded into it.
    """
    text = raw.strip()
    if text[:1] in ("'", '"'):
        quote = text[0]
        buffer: list[str] = []
        index = 1
        while index < len(text):
            character = text[index]
            if character == "\\" and quote == '"' and index + 1 < len(text):
                following = text[index + 1]
                buffer.append({"n": "\n", "t": "\t"}.get(following, following))
                index += 2
                continue
            if character == quote:
                return "".join(buffer), text[index + 1 :].strip()
            buffer.append(character)
            index += 1
        # An unterminated quote is not a value Metis can reason about; keep it whole.
        return text, ""
    # An unquoted value ends at an inline comment, matching dotenv readers.
    match = re.search(r"\s+#", text)
    if match is None:
        return text, ""
    return text[: match.start()].strip(), text[match.start() :].strip()


def _parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=value pairs, keeping file order and dotenv's last-wins rule.

    Only keys Metis is willing to hand to a child process survive, so a project
    cannot smuggle PATH or DYLD_* through its own .env.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_ASSIGNMENT.match(line)
        if match is None:
            continue
        key = match.group(1)
        if not _ENV_KEY.fullmatch(key) or _is_reserved_env(key):
            continue
        values[key] = _split_env_value(match.group(2))[0]
    return values


def _read_env_file(project: Path) -> dict[str, str] | None:
    """Return the project's own .env values, or None when there is no file.

    Values stay server-side: only `_env_presence` is ever serialized to a client.
    """
    path = _env_file_path(project)
    try:
        if path.is_symlink() or not path.is_file():
            return None
    except OSError:
        return None
    text = _read_text(path, _ENV_FILE_LIMIT)
    if text is None:
        return None
    return _parse_env_file(text)


def _env_presence(project: Path) -> list[tuple[str, bool]]:
    values = _read_env_file(project)
    if values is None:
        return []
    return [(key, bool(value)) for key, value in values.items()][:64]


def _quote_env_value(value: str) -> str:
    if value and _ENV_BARE_VALUE.fullmatch(value):
        return value
    return '"{}"'.format(value.replace("\\", "\\\\").replace('"', '\\"'))


def _render_env_file(original: str, updates: dict[str, str]) -> str:
    """Rewrite assignments in place so comments, order, and spacing survive."""
    applied: set[str] = set()
    lines: list[str] = []
    for line in original.splitlines():
        match = _ENV_ASSIGNMENT.match(line)
        key = match.group(1) if match is not None else None
        if key is None or key not in updates or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        # A duplicated key is rewritten at every position: dotenv takes the last
        # one, so leaving a stale earlier line would silently win on some readers.
        prefix = "export " if line.lstrip().startswith("export ") else ""
        note = _split_env_value(match.group(2))[1]
        suffix = f"   {note}" if note else ""
        lines.append(f"{prefix}{key}={_quote_env_value(updates[key])}{suffix}")
        applied.add(key)
    lines.extend(
        f"{key}={_quote_env_value(updates[key])}"
        for key in sorted(updates)
        if key not in applied
    )
    body = "\n".join(lines).strip("\n")
    return f"{body}\n" if body else ""


def _write_env_file(path: Path, text: str) -> None:
    """Replace the file atomically, keeping its mode and never following a link."""
    mode = 0o600
    try:
        mode = stat.S_IMODE(path.lstat().st_mode)
    except OSError:
        pass
    temp = path.with_name(f".env.metis-{os.getpid()}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _framework_and_entrypoint(project: Path) -> tuple[str | None, str | None, set[str]]:
    tags: set[str] = set()
    framework: str | None = None
    entrypoint: str | None = None
    package_text = _read_project_file(project, "package.json", _METADATA_LIMIT)
    package: dict[str, Any] = {}
    if package_text:
        try:
            parsed = json.loads(package_text)
            if isinstance(parsed, dict):
                package = parsed
        except json.JSONDecodeError:
            pass
        dependencies: set[str] = set()
        for key in ("dependencies", "devDependencies"):
            if isinstance(package.get(key), dict):
                dependencies.update(str(item).casefold() for item in package[key])
        for dependency, label in (
            ("next", "Next.js"),
            ("@sveltejs/kit", "SvelteKit"),
            ("nuxt", "Nuxt"),
            ("vite", "Vite"),
            ("react", "React"),
            ("vue", "Vue"),
            ("express", "Express"),
        ):
            if dependency in dependencies:
                framework = label
                break
        framework = framework or "Node.js"
        tags.update(("javascript", "web"))
        entrypoint = "package.json"

    python_metadata = "\n".join(
        filter(
            None,
            (
                _read_project_file(project, "pyproject.toml", _METADATA_LIMIT),
                _read_project_file(project, "requirements.txt", _METADATA_LIMIT),
            ),
        )
    ).casefold()
    try:
        python_files = sorted(
            (
                item
                for item in project.iterdir()
                if item.suffix.casefold() == ".py"
                and item.is_file()
                and not item.is_symlink()
                and not item.name.startswith(".")
            ),
            key=lambda item: item.name.casefold(),
        )
    except (OSError, PermissionError):
        python_files = []
    support_modules = {
        "config.py",
        "constants.py",
        "models.py",
        "prompts.py",
        "schemas.py",
        "settings.py",
        "utils.py",
        "validators.py",
    }
    entrypoint_candidates = [
        item for item in python_files if item.name.casefold() not in support_modules
    ]

    def entrypoint_rank(path: Path) -> tuple[int, str]:
        name = path.name.casefold()
        exact = {
            "app.py": 0,
            "main.py": 1,
            "run.py": 2,
            "server.py": 3,
            "demo.py": 4,
        }
        if name in exact:
            return exact[name], name
        stem = path.stem.casefold()
        if stem.endswith("_app") or stem.startswith("app_"):
            return 10, name
        if any(
            token in stem
            for token in ("chatbot", "dashboard", "demo", "agent", "generator", "extractor")
        ):
            return 20, name
        return 50, name

    detected_python_entrypoint = (
        min(entrypoint_candidates, key=entrypoint_rank)
        if entrypoint_candidates
        else None
    )
    source_hint = (
        _read_text(detected_python_entrypoint, _METADATA_LIMIT)
        if detected_python_entrypoint is not None
        else ""
    ) or ""
    python_signals = f"{python_metadata}\n{source_hint}".casefold()
    if python_metadata or detected_python_entrypoint is not None:
        tags.add("python")
        for token, label in (
            ("streamlit", "Streamlit"),
            ("gradio", "Gradio"),
            ("nicegui", "NiceGUI"),
            ("reflex", "Reflex"),
            ("chainlit", "Chainlit"),
            ("fastapi", "FastAPI"),
            ("flask", "Flask"),
            ("django", "Django"),
        ):
            if token in python_signals:
                framework = label
                break
        framework = framework or "Python"
        if detected_python_entrypoint is not None:
            entrypoint = detected_python_entrypoint.name

    if framework is None:
        index = project / "index.html"
        if index.is_file() and not index.is_symlink():
            framework = "Static HTML"
            entrypoint = "index.html"
            tags.update(("html", "web"))
        elif (project / "Cargo.toml").is_file():
            framework = "Rust"
            entrypoint = "Cargo.toml"
            tags.add("rust")
        elif (project / "go.mod").is_file():
            framework = "Go"
            entrypoint = "go.mod"
            tags.add("go")
    return framework, entrypoint, tags


def _category_and_tags(
    name: str,
    summary: str,
    readme: str,
    framework: str | None,
    explicit_category: str | None,
    explicit_tags: list[str],
    detected_tags: set[str],
) -> tuple[str, tuple[str, ...]]:
    text = f"{name} {summary} {readme[:8_000]}".casefold()
    if explicit_category:
        category = explicit_category
    elif any(token in text for token in ("invoice", "document extraction", "document understanding", "ocr", "data extraction")):
        category = "Document AI"
    elif any(token in text for token in ("voice", "audio", "speech", "tts", "whisper")):
        category = "Voice & Audio"
    elif any(token in text for token in ("video", "image", "vision", "vlm", "multimodal")):
        category = "Vision & Media"
    elif any(token in text for token in ("health", "medical", "patient", "clinical")):
        category = "Healthcare"
    elif any(token in text for token in ("insurance", "claims", "financial", "funding", "earnings")):
        category = "Finance & Insurance"
    elif any(token in text for token in ("legal", "policy", "compliance", "tender", "due diligence")):
        category = "Legal & Policy"
    elif any(token in text for token in ("architecture", "diagram", "topology")):
        category = "Architecture"
    elif any(token in text for token in ("analytics", "dashboard", "metric", "report")):
        category = "Analytics"
    elif any(token in text for token in ("rag", "vector", "database", "sql", "data pipeline")):
        category = "Data & Knowledge"
    elif any(token in text for token in ("agent", "generative ai", " llm", "artificial intelligence")):
        category = "AI & Agents"
    elif any(token in text for token in ("api", "developer", "sdk", "backend")):
        category = "Developer Tools"
    elif framework in {"Next.js", "SvelteKit", "Nuxt", "Vite", "React", "Vue", "Static HTML"}:
        category = "Web Apps"
    else:
        category = "Other"

    tags = set(detected_tags)
    tags.update(explicit_tags)
    if framework:
        framework_tag = _slug_tag(framework)
        if framework_tag:
            tags.add(framework_tag)
    for token, tag in (
        ("oracle cloud", "oci"),
        (" oci", "oci"),
        ("agent", "agent"),
        ("rag", "rag"),
        ("dashboard", "dashboard"),
        ("database", "database"),
        (" api", "api"),
        ("document", "documents"),
        ("invoice", "invoice"),
        ("audio", "audio"),
        ("voice", "voice"),
        ("video", "video"),
        ("image", "image"),
        ("insurance", "insurance"),
        ("health", "healthcare"),
    ):
        if token in text:
            tags.add(tag)
    return category[:60], tuple(sorted(tags, key=str.casefold)[:16])


def _launch_fingerprint(
    project: Path,
    command: tuple[str, ...],
    env_keys: list[str] | tuple[str, ...],
    launch_path: str,
) -> str:
    launch_payload = {
        "asset_path": str(project.resolve(strict=False)),
        "command": list(command),
        "env_keys": list(env_keys)[:64],
        "launch_path": launch_path,
    }
    return hashlib.sha256(
        json.dumps(
            launch_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _scan_record(project: Path) -> AssetRecord:
    readme = _readme_text(project)
    manifest = _manifest(project)
    framework, detected_entrypoint, detected_tags = _framework_and_entrypoint(project)
    if manifest.command is not None:
        launch_tokens = {token.casefold() for token in manifest.command}
        if "streamlit" in launch_tokens:
            framework = "Streamlit"
        elif "uvicorn" in launch_tokens:
            framework = "FastAPI"
        elif "http.server" in launch_tokens:
            framework = "Static HTML"
            detected_tags.update(("html", "web"))
    name = manifest.name or _readme_title(readme) or _folder_name(project.name)
    summary = (
        manifest.summary
        or _readme_summary(readme)
        or f"Local {framework or 'project'} asset."
    )
    category, tags = _category_and_tags(
        name,
        summary,
        readme,
        framework,
        manifest.category,
        manifest.tags,
        detected_tags,
    )
    env_keys = sorted(set(manifest.env_keys) | set(_infer_env_keys(project, readme)))
    stable_path = str(project.resolve(strict=True)).encode("utf-8")
    asset_id = f"asset_{hashlib.sha256(stable_path).hexdigest()[:20]}"
    command = manifest.command
    launch_fingerprint = (
        _launch_fingerprint(project, command, env_keys, manifest.launch_path)
        if command is not None
        else None
    )
    return AssetRecord(
        id=asset_id,
        path=project,
        name=name,
        summary=summary,
        category=category,
        tags=tags,
        framework=framework,
        entrypoint=manifest.entrypoint or detected_entrypoint,
        env_keys=tuple(env_keys[:64]),
        command=command,
        launch_fingerprint=launch_fingerprint,
        launch_path=manifest.launch_path,
    )


class AssetManager:
    """Discovers assets and owns explicitly trusted local child processes.

    Metis assigns a loopback preview URL and injects loopback host/port hints,
    but a direct host process is trusted code and can ignore those hints. The UI
    discloses that boundary before fingerprint approval.
    """

    def __init__(
        self,
        roots: list[Path],
        *,
        approval_path: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self._roots = tuple(roots)
        self._approval_path = approval_path
        self._catalog_path = catalog_path
        self._approvals = self._load_approvals()
        self._catalog: dict[str, AssetRecord] = self._load_catalog()
        self._runs: dict[str, _ProcessRun] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def list(self) -> list[AssetV1]:
        """Return the saved snapshot without touching configured project roots."""
        async with self._lock:
            return [
                self._view(record)
                for record in sorted(
                    self._catalog.values(),
                    key=lambda item: (item.name.casefold(), item.id),
                )
            ]

    async def project_path(self, asset_id: str) -> Path:
        """Resolve one saved catalog entry as an explicit project grant.

        Project workspaces deliberately reuse the manually refreshed Asset
        catalog: opening chat or loading this path never discovers new folders.
        The path must still be the same non-symlinked immediate child of a
        configured root that produced the catalog identity.
        """
        async with self._lock:
            record = self._lookup(asset_id)
            try:
                project = record.path.resolve(strict=True)
            except OSError as exc:
                raise AssetLibraryError(
                    "project source is unavailable; scan Assets for updates"
                ) from exc
            allowed_roots = {
                root.expanduser().resolve(strict=False) for root in self._roots
            }
            if (
                record.path.is_symlink()
                or not project.is_dir()
                or project.parent not in allowed_roots
            ):
                raise AssetLibraryError("project no longer matches its saved grant")
            expected_id = f"asset_{hashlib.sha256(str(project).encode('utf-8')).hexdigest()[:20]}"
            if expected_id != asset_id:
                raise AssetLibraryError("project identity no longer matches its saved grant")
            return project

    async def create(self, name: str) -> AssetV1:
        """Create one empty project folder and return its catalog entry.

        The folder always lands in the *first* configured root — the same
        place every other project lives — and then goes through the ordinary
        scan, so a created project is indistinguishable from a discovered one:
        same identity hash, same grant checks, no side channel into the
        catalog. The name is the only input, and it is held to a shape that
        cannot express a path: no separators, no leading dot, 64 characters.
        """
        cleaned = " ".join(str(name or "").split())
        if not _PROJECT_FOLDER_NAME.match(cleaned):
            raise AssetLibraryError(
                "project names use letters, numbers, spaces, dots, dashes and "
                "underscores, start with a letter or number, and stay under 64 "
                "characters"
            )
        if not self._roots:
            raise AssetLibraryError("no projects folder is configured")
        try:
            root = self._roots[0].expanduser().resolve(strict=True)
            if not root.is_dir():
                raise OSError("configured asset root is not a directory")
            existing = {child.name.casefold() for child in root.iterdir()}
        except (OSError, PermissionError) as exc:
            raise AssetLibraryError(
                "the configured projects folder is unavailable"
            ) from exc
        if cleaned.casefold() in existing:
            raise AssetLibraryError(f'a project named "{cleaned}" already exists')
        try:
            (root / cleaned).mkdir()
        except (OSError, PermissionError) as exc:
            raise AssetLibraryError("the project folder could not be created") from exc
        await self.scan()
        created_id = (
            f"asset_{hashlib.sha256(str((root / cleaned).resolve()).encode('utf-8')).hexdigest()[:20]}"
        )
        async with self._lock:
            record = self._catalog.get(created_id)
            if record is None:
                raise AssetLibraryError(
                    "the new project folder did not survive a catalog scan"
                )
            return self._view(record)

    async def scan(self) -> list[AssetV1]:
        records: dict[str, AssetRecord] = {}
        for configured_root in self._roots:
            try:
                root = configured_root.expanduser().resolve(strict=True)
                if not root.is_dir():
                    raise OSError("configured asset root is not a directory")
                children = sorted(root.iterdir(), key=lambda item: item.name.casefold())
            except (OSError, PermissionError) as exc:
                # Scans are atomic per root, so an unavailable folder cannot empty the catalog.
                raise AssetLibraryError(
                    "a configured projects folder is unavailable; the saved catalog was kept"
                ) from exc
            for child in children:
                if child.name.startswith(".") or child.is_symlink():
                    continue
                try:
                    resolved = child.resolve(strict=True)
                    if not resolved.is_dir() or resolved.parent != root:
                        continue
                    record = _scan_record(resolved)
                except (OSError, PermissionError, ValueError):
                    continue
                records.setdefault(record.id, record)
        async with self._lock:
            # Keep a tombstone for a running child whose folder moved, to preserve stop
            # control. It disappears on the first scan after the process exits.
            for asset_id, run in self._runs.items():
                previous = self._catalog.get(asset_id)
                if (
                    asset_id not in records
                    and previous is not None
                    and run.process.returncode is None
                ):
                    records[asset_id] = previous
            self._catalog = records
            self._save_catalog()
            return [
                self._view(record)
                for record in sorted(
                    records.values(), key=lambda item: (item.name.casefold(), item.id)
                )
            ]

    async def start(self, asset_id: str, provided_env: dict[str, str]) -> AssetV1:
        await self._refresh_known_record(asset_id)
        async with self._lock:
            record = self._lookup(asset_id)
            if record.command is None:
                raise LaunchNotApprovedError(
                    "asset launch is not configured; add a valid .metis/asset.json argv manifest"
                )
            if not self._is_approved(record):
                raise LaunchNotApprovedError(
                    "asset launch recipe must be explicitly reviewed and trusted before it can run"
                )
            existing = self._runs.get(asset_id)
            if existing is not None and existing.process.returncode is None:
                raise AssetLibraryError("asset is already running")
            environment = self._validated_environment(record, provided_env)
            port = self._reserve_port()
            runtime_python = str(Path(sys.executable).resolve(strict=False))
            runtime_uv = str(Path(sys.executable).with_name("uv").resolve(strict=False))
            argv = tuple(
                item.replace("{port}", str(port))
                .replace("{host}", "127.0.0.1")
                .replace("{python}", runtime_python)
                .replace("{uv}", runtime_uv)
                for item in record.command
            )
            # The project's own .env is the source of truth for runtime settings.
            # It sits under the request overrides and under the Metis pins, and
            # `_parse_env_file` has already dropped every reserved key, so a .env
            # can configure the app but never redirect PATH or preload a library.
            project_env = _read_env_file(record.path) or {}
            child_env = self._base_environment()
            child_env.update(project_env)
            child_env.update(environment)
            child_env.update(
                {
                    "HOST": "127.0.0.1",
                    "PORT": str(port),
                    "METIS_HOST": "127.0.0.1",
                    "METIS_PORT": str(port),
                    "METIS_ASSET_ID": record.id,
                }
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=record.path,
                    env=child_env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=os.name == "posix",
                )
            except (OSError, ValueError) as exc:
                raise AssetLibraryError(f"asset failed to start: {type(exc).__name__}") from exc
            url = f"http://127.0.0.1:{port}{record.launch_path}"
            run = _ProcessRun(
                process=process,
                port=port,
                url=url,
                # Only secret-looking .env values join the redaction set. Redacting
                # every value would rewrite ordinary words out of the log stream.
                secrets=tuple(
                    sorted(
                        {value for value in provided_env.values() if value}
                        | {
                            value
                            for key, value in project_env.items()
                            if len(value) >= 8 and _is_sensitive_env(key)
                        },
                        key=len,
                        reverse=True,
                    )
                ),
            )
            run.reader_task = asyncio.create_task(
                self._capture_output(run), name=f"metis-asset-log-{asset_id}"
            )
            run.readiness_task = asyncio.create_task(
                self._watch_readiness(run), name=f"metis-asset-ready-{asset_id}"
            )
            self._runs[asset_id] = run
        # Fast apps bind within this window; slower ones stay `starting` until probed.
        try:
            await asyncio.wait_for(asyncio.shield(run.ready_event.wait()), timeout=2.0)
        except TimeoutError:
            pass
        async with self._lock:
            return self._view(record)

    async def approve(self, asset_id: str) -> AssetV1:
        """Trust the exact current launch fingerprint; any manifest drift revokes it."""
        await self._refresh_known_record(asset_id)
        async with self._lock:
            record = self._lookup(asset_id)
            if record.command is None or record.launch_fingerprint is None:
                raise LaunchNotApprovedError(
                    "asset has no valid .metis/asset.json launch recipe to approve"
                )
            self._approvals[record.id] = record.launch_fingerprint
            self._save_approvals()
            return self._view(record)

    async def revoke(self, asset_id: str) -> AssetV1:
        async with self._lock:
            record = self._lookup(asset_id)
            if record.id in self._approvals:
                del self._approvals[record.id]
                self._save_approvals()
            return self._view(record)

    async def stop(self, asset_id: str) -> AssetV1:
        async with self._lock:
            record = self._lookup(asset_id)
            run = self._runs.get(asset_id)
            if run is None or run.process.returncode is not None:
                return self._view(record)
            run.requested_stop = True
            self._terminate(run.process)
        try:
            await asyncio.wait_for(run.process.wait(), timeout=3.0)
        except TimeoutError:
            self._kill(run.process)
            await run.process.wait()
        if run.reader_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(run.reader_task), timeout=1.0)
            except TimeoutError:
                run.reader_task.cancel()
        async with self._lock:
            return self._view(record)

    async def write_env(self, asset_id: str, values: dict[str, str]) -> AssetV1:
        """Persist runtime settings into the project's own .env file.

        This is the only path that writes inside a project folder. It rewrites
        assignments in place, so comments and unrelated keys survive, and it
        refuses reserved keys so a saved value can never alter how Metis launches
        the process. A launch recipe is unaffected, so approval is not disturbed.
        """
        await self._refresh_known_record(asset_id)
        async with self._lock:
            record = self._lookup(asset_id)
            for key, value in values.items():
                if not _ENV_KEY.fullmatch(key) or _is_reserved_env(key):
                    raise AssetEnvironmentError(
                        f"environment key {key!r} is reserved or invalid"
                    )
                if (
                    not isinstance(value, str)
                    or len(value) > 16_384
                    or any(character in value for character in ("\x00", "\r", "\n"))
                ):
                    raise AssetEnvironmentError(
                        f"environment value for {key!r} is invalid"
                    )
            if not values:
                return self._view(record)

            path = _env_file_path(record.path)
            try:
                if path.is_symlink():
                    raise AssetEnvironmentError(
                        "the project .env is a symbolic link; Metis will not write through it"
                    )
                exists = path.is_file()
                # Refuse rather than truncate: a partial read would silently drop
                # whatever sat past the limit when the file is written back.
                if exists and path.stat().st_size > _ENV_FILE_LIMIT:
                    raise AssetEnvironmentError(
                        "the project .env file is too large for Metis to edit safely"
                    )
                original = _read_text(path, _ENV_FILE_LIMIT) if exists else ""
                if original is None:
                    raise AssetEnvironmentError("the project .env file could not be read")
                _write_env_file(path, _render_env_file(original, values))
            except OSError as exc:
                raise AssetEnvironmentError(
                    "the project .env file could not be written"
                ) from exc
            return self._view(record)

    async def logs(self, asset_id: str) -> AssetLogsV1:
        async with self._lock:
            record = self._lookup(asset_id)
            run = self._runs.get(asset_id)
            if run is None:
                return AssetLogsV1(asset_id=asset_id, status=self._status(record), logs="")
            text = run.logs
            for secret in run.secrets:
                text = text.replace(secret, "[REDACTED]")
            if len(text) > _LOG_LIMIT:
                text = text[-_LOG_LIMIT:]
            return AssetLogsV1(
                asset_id=asset_id,
                status=self._status(record),
                logs=text,
                truncated=run.truncated,
                return_code=run.process.returncode,
            )

    async def shutdown(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            active = [run for run in self._runs.values() if run.process.returncode is None]
            for run in active:
                run.requested_stop = True
                self._terminate(run.process)
        for run in active:
            try:
                await asyncio.wait_for(run.process.wait(), timeout=3.0)
            except TimeoutError:
                self._kill(run.process)
                await run.process.wait()
        reader_tasks = [run.reader_task for run in self._runs.values() if run.reader_task]
        if reader_tasks:
            await asyncio.gather(*reader_tasks, return_exceptions=True)
        readiness_tasks = [
            run.readiness_task for run in self._runs.values() if run.readiness_task
        ]
        if readiness_tasks:
            await asyncio.gather(*readiness_tasks, return_exceptions=True)

    async def _refresh_known_record(self, asset_id: str) -> AssetRecord:
        """Revalidate one saved asset without discovering any sibling folders."""
        async with self._lock:
            saved = self._lookup(asset_id)
            path = saved.path
        try:
            if path.is_symlink():
                raise OSError("symbolic-link asset")
            resolved = path.resolve(strict=True)
            roots = {
                configured.expanduser().resolve(strict=True)
                for configured in self._roots
                if configured.expanduser().exists()
            }
            if not resolved.is_dir() or resolved.parent not in roots:
                raise OSError("asset is outside the configured roots")
            refreshed = _scan_record(resolved)
            if refreshed.id != asset_id:
                raise OSError("asset identity changed")
        except (OSError, PermissionError, ValueError) as exc:
            raise AssetLibraryError(
                "asset source is unavailable or moved; search for project updates"
            ) from exc
        async with self._lock:
            if asset_id not in self._catalog:
                raise UnknownAssetError("unknown asset")
            self._catalog[asset_id] = refreshed
            self._save_catalog()
            return refreshed

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
            asset_id: fingerprint
            for asset_id, fingerprint in value.items()
            if isinstance(asset_id, str)
            and _SAFE_ID.fullmatch(asset_id)
            and isinstance(fingerprint, str)
            and re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        }

    def _save_approvals(self) -> None:
        path = self._approval_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise AssetLibraryError("asset approval store may not be a symbolic link")
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
            raise AssetLibraryError("asset launch approval could not be saved") from exc

    def _root_signature(self) -> list[str]:
        return sorted(
            str(root.expanduser().resolve(strict=False)) for root in self._roots
        )

    def _load_catalog(self) -> dict[str, AssetRecord]:
        path = self._catalog_path
        if path is None:
            return {}
        try:
            if path.is_symlink():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "1"
            or payload.get("roots") != self._root_signature()
            or not isinstance(payload.get("assets"), list)
        ):
            return {}
        records: dict[str, AssetRecord] = {}
        for raw in payload["assets"][:10_000]:
            record = self._record_from_cache(raw)
            if record is not None:
                records.setdefault(record.id, record)
        return records

    def _record_from_cache(self, raw: object) -> AssetRecord | None:
        if not isinstance(raw, dict):
            return None
        asset_id = raw.get("id")
        raw_path = raw.get("path")
        if (
            not isinstance(asset_id, str)
            or not _SAFE_ID.fullmatch(asset_id)
            or not isinstance(raw_path, str)
            or not raw_path
            or len(raw_path) > 4_096
        ):
            return None
        project = Path(raw_path).expanduser().resolve(strict=False)
        allowed_parents = {Path(root) for root in self._root_signature()}
        expected_id = f"asset_{hashlib.sha256(str(project).encode('utf-8')).hexdigest()[:20]}"
        if project.parent not in allowed_parents or expected_id != asset_id:
            return None

        name = _single_line(raw.get("name"), 120)
        summary = _single_line(raw.get("summary"), 240)
        category = _single_line(raw.get("category"), 60)
        if name is None or summary is None or category is None:
            return None
        framework = _single_line(raw.get("framework"), 60)
        entrypoint = _single_line(raw.get("entrypoint"), 240)
        tags = tuple(
            tag
            for item in (raw.get("tags") if isinstance(raw.get("tags"), list) else [])
            if (tag := _slug_tag(item)) is not None
        )[:16]
        env_keys = tuple(sorted({
            item
            for item in (
                raw.get("env_keys") if isinstance(raw.get("env_keys"), list) else []
            )
            if isinstance(item, str)
            and _ENV_KEY.fullmatch(item)
            and not _is_reserved_env(item)
        }))[:64]

        raw_command = raw.get("command")
        command: tuple[str, ...] | None = None
        if (
            isinstance(raw_command, list)
            and 1 <= len(raw_command) <= 32
            and all(type(item) is str for item in raw_command)
            and all(
                item
                and len(item) <= 1_024
                and not any(character in item for character in ("\x00", "\r", "\n"))
                for item in raw_command
            )
            and sum(len(item) for item in raw_command) <= 8_192
        ):
            command = tuple(raw_command)
        raw_launch_path = raw.get("launch_path", "")
        launch_path = (
            raw_launch_path
            if isinstance(raw_launch_path, str)
            and len(raw_launch_path) <= 240
            and (not raw_launch_path or raw_launch_path.startswith("/"))
            and not raw_launch_path.startswith("//")
            and "\r" not in raw_launch_path
            and "\n" not in raw_launch_path
            else ""
        )
        fingerprint = (
            _launch_fingerprint(project, command, env_keys, launch_path)
            if command is not None
            else None
        )
        return AssetRecord(
            id=asset_id,
            path=project,
            name=name,
            summary=summary,
            category=category,
            tags=tags,
            framework=framework,
            entrypoint=entrypoint,
            env_keys=env_keys,
            command=command,
            launch_fingerprint=fingerprint,
            launch_path=launch_path,
        )

    def _save_catalog(self) -> None:
        path = self._catalog_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise AssetLibraryError("asset catalog store may not be a symbolic link")
        payload = {
            "schema_version": "1",
            "roots": self._root_signature(),
            "assets": [
                {
                    "id": record.id,
                    "path": str(record.path),
                    "name": record.name,
                    "summary": record.summary,
                    "category": record.category,
                    "tags": list(record.tags),
                    "framework": record.framework,
                    "entrypoint": record.entrypoint,
                    "env_keys": list(record.env_keys),
                    "command": list(record.command) if record.command is not None else None,
                    "launch_path": record.launch_path,
                }
                for record in sorted(self._catalog.values(), key=lambda item: item.id)
            ],
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
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
            raise AssetLibraryError("asset catalog could not be saved") from exc

    def _is_approved(self, record: AssetRecord) -> bool:
        return bool(
            record.launch_fingerprint
            and self._approvals.get(record.id) == record.launch_fingerprint
        )

    def _lookup(self, asset_id: str) -> AssetRecord:
        if not _SAFE_ID.fullmatch(asset_id):
            raise UnknownAssetError("unknown asset")
        record = self._catalog.get(asset_id)
        if record is None:
            raise UnknownAssetError("unknown asset")
        return record

    def _validated_environment(
        self, record: AssetRecord, provided: dict[str, str]
    ) -> dict[str, str]:
        allowed = set(record.env_keys)
        for key, value in provided.items():
            if not _ENV_KEY.fullmatch(key) or _is_reserved_env(key):
                raise AssetEnvironmentError(f"environment key {key!r} is reserved or invalid")
            if key not in allowed:
                raise AssetEnvironmentError(f"environment key {key!r} is not declared by the asset")
            if not isinstance(value, str) or len(value) > 16_384 or "\x00" in value:
                raise AssetEnvironmentError(f"environment value for {key!r} is invalid")
        return dict(provided)

    def _view(self, record: AssetRecord) -> AssetV1:
        run = self._runs.get(record.id)
        active = run is not None and run.process.returncode is None
        launch_approved = self._is_approved(record)
        # Read live rather than from the catalog: the project's .env is edited
        # outside Metis too, and a stale "not set" badge would be misleading.
        env_file = _read_env_file(record.path)
        return AssetV1(
            id=record.id,
            name=record.name,
            summary=record.summary,
            category=record.category,
            tags=list(record.tags),
            framework=record.framework,
            entrypoint=record.entrypoint,
            env_keys=list(record.env_keys),
            env_file=[
                AssetEnvVarV1(key=key, is_set=bool(value), sensitive=_is_sensitive_env(key))
                for key, value in list((env_file or {}).items())[:64]
            ],
            env_file_present=env_file is not None,
            launch_configured=record.command is not None,
            launch_approved=launch_approved,
            launch_command=list(record.command or ()),
            status=self._status(record),
            # The loopback route exists before the child binds, so the preview target is
            # stable while status communicates readiness.
            url=run.url if active else None,
        )

    def _status(self, record: AssetRecord) -> AssetStatus:
        run = self._runs.get(record.id)
        if run is None:
            if record.command is None:
                return AssetStatus.UNCONFIGURED
            return (
                AssetStatus.READY
                if self._is_approved(record)
                else AssetStatus.NEEDS_APPROVAL
            )
        if run.startup_failed:
            return AssetStatus.FAILED
        if run.process.returncode is None:
            return AssetStatus.RUNNING if run.ready else AssetStatus.STARTING
        if run.requested_stop or run.process.returncode == 0:
            return AssetStatus.STOPPED
        return AssetStatus.FAILED

    async def _capture_output(self, run: _ProcessRun) -> None:
        stream = run.process.stdout
        if stream is not None:
            while True:
                chunk = await stream.read(4_096)
                if not chunk:
                    break
                run.append_logs(chunk.decode("utf-8", errors="replace"))
        await run.process.wait()

    async def _watch_readiness(self, run: _ProcessRun) -> None:
        deadline = asyncio.get_running_loop().time() + 30.0
        try:
            while asyncio.get_running_loop().time() < deadline:
                if run.process.returncode is not None:
                    run.startup_failed = not run.requested_stop
                    return
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection("127.0.0.1", run.port), timeout=0.4
                    )
                except (OSError, TimeoutError):
                    await asyncio.sleep(0.15)
                    continue
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
                del reader
                run.ready = True
                return
            if run.process.returncode is None and not run.requested_stop:
                run.startup_failed = True
                run.append_logs("\n[Metis] Startup timed out before the loopback port became ready.\n")
                self._terminate(run.process)
        finally:
            run.ready_event.set()

    @staticmethod
    def _reserve_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _base_environment() -> dict[str, str]:
        environment = {"PATH": os.environ.get("PATH", os.defpath)}
        for key in ("HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"):
            value = os.environ.get(key)
            if value:
                environment[key] = value
        return environment

    @staticmethod
    def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
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
        except ProcessLookupError:
            pass
