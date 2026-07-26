"""JSON-stdio architecture renderer used inside Metis's Podman sandbox."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, NoReturn

try:
    from .validator import (
        SourceValidationError,
        validate_generated_source,
        validate_generated_source_v2,
        validate_source_against_spec,
        validate_source_against_spec_v2,
    )
except ImportError:  # Executed as a standalone file in the sandbox image.
    from validator import (
        SourceValidationError,
        validate_generated_source,
        validate_generated_source_v2,
        validate_source_against_spec,
        validate_source_against_spec_v2,
    )


ALLOWED_VALIDATION_PROFILES = {"diagrams-render-v1", "diagrams-draw-v2"}


SCHEMA_VERSION = "1"
MAX_STDIN_BYTES = 1_048_576
MAX_ARTIFACT_BYTES = 104_857_600
ALLOWED_FORMATS = {"svg", "png"}
ALLOWED_RENDER_MODES = {"auto", "diagrams", "fallback"}
ALLOWED_PROVIDERS = {
    "generic",
    "aws",
    "azure",
    "gcp",
    "oci",
    "kubernetes",
    "onprem",
    "hybrid",
}
ALLOWED_DIRECTIONS = {"LR", "RL", "TB", "BT"}
IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
SAFE_ARTIFACTS = {
    "architecture-spec.json": "application/json",
    "diagram.py": "text/x-python",
    "architecture.svg": "image/svg+xml",
    "architecture.png": "image/png",
    "validation-report.json": "application/json",
}


class ToolFailure(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}


def _fail(
    code: str,
    message: str,
    *,
    exit_code: int = 2,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    raise ToolFailure(code, message, exit_code=exit_code, details=details)


def _strict_keys(value: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail("INVALID_INPUT", f"{where} contains unknown fields", details={"fields": unknown})


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("INVALID_INPUT", f"{where} must be an object")
    return value


def _array(value: Any, where: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        _fail("INVALID_INPUT", f"{where} must be an array")
    if len(value) > maximum:
        _fail("INVALID_INPUT", f"{where} exceeds {maximum} entries")
    return value


def _text(
    value: Any,
    where: str,
    *,
    minimum: int = 0,
    maximum: int,
) -> str:
    if not isinstance(value, str):
        _fail("INVALID_INPUT", f"{where} must be a string")
    if len(value) < minimum or len(value) > maximum:
        _fail("INVALID_INPUT", f"{where} length must be between {minimum} and {maximum}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail("INVALID_INPUT", f"{where} contains a control character")
    return value


def _identifier(value: Any, where: str) -> str:
    text = _text(value, where, minimum=1, maximum=64)
    if not IDENTIFIER_RE.fullmatch(text):
        _fail("INVALID_INPUT", f"{where} is not a valid identifier")
    return text


def _string_list(value: Any, where: str) -> list[str]:
    items = _array(value, where, maximum=32)
    return [
        _text(item, f"{where}[{index}]", minimum=1, maximum=500)
        for index, item in enumerate(items)
    ]


def _diagram_code(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("INVALID_INPUT", "diagram_code must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        _fail("INVALID_INPUT", "diagram_code must be valid Unicode encodable as UTF-8")
    if len(encoded) > 100_000:
        _fail("INVALID_INPUT", "diagram_code exceeds 100000 UTF-8 bytes")
    if "\x00" in value or "\r" in value:
        _fail("INVALID_INPUT", "diagram_code must use LF line endings and contain no NUL bytes")
    return value


def validate_request(value: Any) -> dict[str, Any]:
    request = _object(value, "request")
    _strict_keys(
        request,
        {
            "schema_version", "spec", "output_formats", "render_mode",
            "diagram_code", "validation_profile",
        },
        "request",
    )
    if request.get("schema_version") != SCHEMA_VERSION:
        _fail("UNSUPPORTED_SCHEMA", "schema_version must be '1'")

    render_mode = request.get("render_mode", "auto")
    if not isinstance(render_mode, str) or render_mode not in ALLOWED_RENDER_MODES:
        _fail("INVALID_INPUT", "render_mode must be auto, diagrams, or fallback")

    validation_profile = request.get("validation_profile", "diagrams-render-v1")
    if (
        not isinstance(validation_profile, str)
        or validation_profile not in ALLOWED_VALIDATION_PROFILES
    ):
        _fail("INVALID_INPUT", "validation_profile is not a recognized policy")

    raw_formats = _array(request.get("output_formats", ["svg", "png"]), "output_formats", maximum=2)
    if not raw_formats:
        _fail("INVALID_INPUT", "output_formats must not be empty")
    if any(not isinstance(item, str) or item not in ALLOWED_FORMATS for item in raw_formats):
        _fail("INVALID_INPUT", "output_formats may contain only svg and png")
    if len(set(raw_formats)) != len(raw_formats):
        _fail("INVALID_INPUT", "output_formats must not contain duplicates")
    output_formats = [item for item in ("svg", "png") if item in raw_formats]

    spec = _object(request.get("spec"), "spec")
    _strict_keys(
        spec,
        {
            "title",
            "provider",
            "direction",
            "components",
            "edges",
            "boundaries",
            "assumptions",
            "unresolved_ambiguities",
        },
        "spec",
    )
    title = _text(spec.get("title"), "spec.title", minimum=1, maximum=160)
    provider = spec.get("provider", "generic")
    if not isinstance(provider, str) or provider not in ALLOWED_PROVIDERS:
        _fail("INVALID_INPUT", f"unsupported provider {provider!r}")
    direction = spec.get("direction", "LR")
    if not isinstance(direction, str) or direction not in ALLOWED_DIRECTIONS:
        _fail("INVALID_INPUT", f"unsupported direction {direction!r}")

    raw_components = _array(spec.get("components"), "spec.components", maximum=64)
    if not raw_components:
        _fail("INVALID_INPUT", "spec.components must contain at least one component")
    components: list[dict[str, str]] = []
    component_ids: set[str] = set()
    for index, raw in enumerate(raw_components):
        item = _object(raw, f"spec.components[{index}]")
        _strict_keys(item, {"id", "label", "kind"}, f"spec.components[{index}]")
        component_id = _identifier(item.get("id"), f"spec.components[{index}].id")
        if component_id in component_ids:
            _fail("INVALID_INPUT", f"duplicate component id {component_id!r}")
        component_ids.add(component_id)
        components.append(
            {
                "id": component_id,
                "label": _text(item.get("label"), f"spec.components[{index}].label", minimum=1, maximum=120),
                "kind": _text(item.get("kind"), f"spec.components[{index}].kind", minimum=1, maximum=64),
            }
        )

    raw_edges = _array(spec.get("edges"), "spec.edges", maximum=256)
    edges: list[dict[str, str]] = []
    for index, raw in enumerate(raw_edges):
        item = _object(raw, f"spec.edges[{index}]")
        _strict_keys(item, {"source", "target", "label"}, f"spec.edges[{index}]")
        source = _identifier(item.get("source"), f"spec.edges[{index}].source")
        target = _identifier(item.get("target"), f"spec.edges[{index}].target")
        if source not in component_ids or target not in component_ids:
            _fail(
                "INVALID_INPUT",
                f"edge {index} references an unknown component",
                details={"source": source, "target": target},
            )
        edges.append(
            {
                "source": source,
                "target": target,
                "label": _text(item.get("label", ""), f"spec.edges[{index}].label", maximum=120),
            }
        )

    raw_boundaries = _array(spec.get("boundaries", []), "spec.boundaries", maximum=16)
    boundaries: list[dict[str, Any]] = []
    boundary_ids: set[str] = set()
    assigned_components: set[str] = set()
    for index, raw in enumerate(raw_boundaries):
        item = _object(raw, f"spec.boundaries[{index}]")
        _strict_keys(item, {"id", "label", "component_ids"}, f"spec.boundaries[{index}]")
        boundary_id = _identifier(item.get("id"), f"spec.boundaries[{index}].id")
        if boundary_id in boundary_ids:
            _fail("INVALID_INPUT", f"duplicate boundary id {boundary_id!r}")
        boundary_ids.add(boundary_id)
        raw_members = _array(item.get("component_ids"), f"spec.boundaries[{index}].component_ids", maximum=64)
        if not raw_members:
            _fail("INVALID_INPUT", f"boundary {boundary_id!r} must contain at least one component")
        members = [_identifier(member, f"spec.boundaries[{index}].component_ids") for member in raw_members]
        if len(set(members)) != len(members):
            _fail("INVALID_INPUT", f"boundary {boundary_id!r} contains duplicate component ids")
        unknown = sorted(set(members) - component_ids)
        if unknown:
            _fail("INVALID_INPUT", f"boundary {boundary_id!r} references unknown components", details={"ids": unknown})
        overlap = sorted(set(members) & assigned_components)
        if overlap:
            _fail("INVALID_INPUT", "components may belong to only one boundary", details={"ids": overlap})
        assigned_components.update(members)
        boundaries.append(
            {
                "id": boundary_id,
                "label": _text(item.get("label"), f"spec.boundaries[{index}].label", minimum=1, maximum=120),
                "component_ids": sorted(members),
            }
        )

    normalized_spec: dict[str, Any] = {
        "title": title,
        "provider": provider,
        "direction": direction,
        "components": sorted(components, key=lambda item: item["id"]),
        "edges": sorted(edges, key=lambda item: (item["source"], item["target"], item["label"])),
        "boundaries": sorted(boundaries, key=lambda item: item["id"]),
        "assumptions": _string_list(spec.get("assumptions", []), "spec.assumptions"),
        "unresolved_ambiguities": _string_list(
            spec.get("unresolved_ambiguities", []), "spec.unresolved_ambiguities"
        ),
    }
    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "spec": normalized_spec,
        "output_formats": output_formats,
        "render_mode": render_mode,
        "validation_profile": validation_profile,
    }
    if "diagram_code" in request:
        normalized["diagram_code"] = _diagram_code(request["diagram_code"])
    return normalized


def generate_diagram_source(spec: dict[str, Any], output_formats: list[str]) -> str:
    components = spec["components"]
    variable_by_id = {
        component["id"]: f"node_{index:03d}" for index, component in enumerate(components)
    }
    component_by_id = {component["id"]: component for component in components}
    grouped_ids = {
        component_id
        for boundary in spec["boundaries"]
        for component_id in boundary["component_ids"]
    }

    lines = [
        "# Generated deterministically by Metis. Do not add executable input.",
        "from pathlib import Path",
        "from diagrams import Cluster, Diagram, Edge",
        "from diagrams.generic.blank import Blank",
        "",
        'OUTPUT_STEM = str(Path(__file__).resolve().parent / "architecture")',
        "",
        (
            f"with Diagram({spec['title']!r}, filename=OUTPUT_STEM, "
            f"outformat={output_formats!r}, show=False, direction={spec['direction']!r}):"
        ),
    ]

    for boundary in spec["boundaries"]:
        lines.append(f"    with Cluster({boundary['label']!r}):")
        for component_id in boundary["component_ids"]:
            component = component_by_id[component_id]
            label = f"{component['label']}\n[{component['kind']}]"
            lines.append(f"        {variable_by_id[component_id]} = Blank({label!r})")

    for component in components:
        if component["id"] not in grouped_ids:
            label = f"{component['label']}\n[{component['kind']}]"
            lines.append(f"    {variable_by_id[component['id']]} = Blank({label!r})")

    for edge in spec["edges"]:
        source = variable_by_id[edge["source"]]
        target = variable_by_id[edge["target"]]
        if edge["label"]:
            lines.append(f"    {source} >> Edge(label={edge['label']!r}) >> {target}")
        else:
            lines.append(f"    {source} >> {target}")

    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: bytes) -> None:
    if path.name not in SAFE_ARTIFACTS:
        _fail("POLICY_VIOLATION", f"artifact {path.name!r} is not allowed", exit_code=3)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        _fail("POLICY_VIOLATION", f"artifact target {path.name!r} is not a regular file", exit_code=3)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _prepare_output_directory(path: Path) -> Path:
    if not path.is_absolute():
        _fail("POLICY_VIOLATION", "output directory must be absolute", exit_code=3)
    if path.is_symlink() or not path.exists() or not path.is_dir():
        _fail("POLICY_VIOLATION", "output directory must be an existing non-symlink directory", exit_code=3)
    resolved = path.resolve(strict=True)
    if resolved == Path("/"):
        _fail("POLICY_VIOLATION", "output directory may not be the filesystem root", exit_code=3)
    for name in SAFE_ARTIFACTS:
        candidate = resolved / name
        if candidate.exists():
            if candidate.is_symlink() or not candidate.is_file():
                _fail("POLICY_VIOLATION", f"existing {name!r} is not a regular file", exit_code=3)
            candidate.unlink()
    return resolved


def _render_with_diagrams(source_path: Path, output_dir: Path) -> None:
    environment = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(source_path)],
            cwd=output_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolFailure("RENDER_TIMEOUT", "diagram renderer exceeded 90 seconds", exit_code=4) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[:4_000]
        _fail(
            "RENDER_FAILED",
            "Python diagrams or Graphviz failed",
            exit_code=4,
            details={"returncode": completed.returncode, "stderr": stderr},
        )


def _fallback_positions(spec: dict[str, Any]) -> tuple[dict[str, tuple[int, int]], int, int]:
    ordered_groups: list[tuple[str, list[str]]] = [
        (boundary["label"], boundary["component_ids"]) for boundary in spec["boundaries"]
    ]
    grouped = {item for _, members in ordered_groups for item in members}
    ungrouped = [component["id"] for component in spec["components"] if component["id"] not in grouped]
    if ungrouped:
        ordered_groups.append(("", ungrouped))

    positions: dict[str, tuple[int, int]] = {}
    y = 100
    width = 860
    for _, members in ordered_groups:
        columns = min(3, max(1, len(members)))
        rows = math.ceil(len(members) / columns)
        for index, component_id in enumerate(members):
            column = index % columns
            row = index // columns
            positions[component_id] = (90 + column * 250, y + row * 120)
        y += rows * 120 + 70
    return positions, width, max(320, y + 20)


def _render_fallback_svg(spec: dict[str, Any], output_path: Path) -> None:
    positions, width, height = _fallback_positions(spec)
    components = {component["id"]: component for component in spec["components"]}
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">'
        ),
        f'<title id="title">{html.escape(spec["title"])}</title>',
        "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"4\" orient=\"auto\"><path d=\"M0,0 L8,4 L0,8 z\" fill=\"#64748b\"/></marker></defs>",
        f'<text x="40" y="42" font-family="sans-serif" font-size="24" font-weight="600" fill="#0f172a">{html.escape(spec["title"])}</text>',
        f'<text x="40" y="66" font-family="sans-serif" font-size="12" fill="#64748b">Provider: {html.escape(spec["provider"])}</text>',
    ]

    for boundary in spec["boundaries"]:
        member_positions = [positions[item] for item in boundary["component_ids"]]
        min_x = min(item[0] for item in member_positions) - 25
        max_x = max(item[0] for item in member_positions) + 205
        min_y = min(item[1] for item in member_positions) - 32
        max_y = max(item[1] for item in member_positions) + 82
        parts.append(
            f'<rect x="{min_x}" y="{min_y}" width="{max_x - min_x}" height="{max_y - min_y}" rx="14" fill="#f8fafc" stroke="#94a3b8" stroke-dasharray="6 4"/>'
        )
        parts.append(
            f'<text x="{min_x + 10}" y="{min_y + 19}" font-family="sans-serif" font-size="12" fill="#475569">{html.escape(boundary["label"])}</text>'
        )

    for edge in spec["edges"]:
        source_x, source_y = positions[edge["source"]]
        target_x, target_y = positions[edge["target"]]
        x1, y1 = source_x + 180, source_y + 30
        x2, y2 = target_x, target_y + 30
        if x2 <= x1:
            x1, y1 = source_x + 90, source_y + 60
            x2, y2 = target_x + 90, target_y
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        if edge["label"]:
            label_x = (x1 + x2) // 2
            label_y = (y1 + y2) // 2 - 6
            parts.append(
                f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#334155">{html.escape(edge["label"])}</text>'
            )

    for component_id in sorted(components):
        component = components[component_id]
        x, y = positions[component_id]
        parts.append(
            f'<rect id="component-{html.escape(component_id)}" x="{x}" y="{y}" width="180" height="60" rx="10" fill="#ffffff" stroke="#2563eb" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + 90}" y="{y + 26}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="600" fill="#0f172a">{html.escape(component["label"])}</text>'
        )
        parts.append(
            f'<text x="{x + 90}" y="{y + 46}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{html.escape(component["kind"])}</text>'
        )
    parts.append("</svg>")
    _atomic_write(output_path, ("\n".join(parts) + "\n").encode("utf-8"))


def _validate_svg(path: Path) -> None:
    content = path.read_bytes()
    if not content or len(content) > MAX_ARTIFACT_BYTES:
        _fail("INVALID_ARTIFACT", "SVG is empty or too large", exit_code=4)
    lowered = content.lower()
    if b"<svg" not in lowered:
        _fail("INVALID_ARTIFACT", "SVG root is missing", exit_code=4)
    for forbidden in (b"<script", b"<foreignobject", b"javascript:", b"<!entity", b"http-equiv"):
        if forbidden in lowered:
            _fail("INVALID_ARTIFACT", f"SVG contains forbidden content {forbidden!r}", exit_code=4)


def _validate_png(path: Path) -> None:
    content = path.read_bytes()
    if not content or len(content) > MAX_ARTIFACT_BYTES:
        _fail("INVALID_ARTIFACT", "PNG is empty or too large", exit_code=4)
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        _fail("INVALID_ARTIFACT", "PNG signature is invalid", exit_code=4)


def _artifact(path: Path) -> dict[str, Any]:
    if path.name not in SAFE_ARTIFACTS or path.is_symlink() or not path.is_file():
        _fail("INVALID_ARTIFACT", f"invalid artifact {path.name!r}", exit_code=4)
    content = path.read_bytes()
    if not content or len(content) > MAX_ARTIFACT_BYTES:
        _fail("INVALID_ARTIFACT", f"artifact {path.name!r} is empty or too large", exit_code=4)
    return {
        "name": path.name,
        "path": path.name,
        "media_type": SAFE_ARTIFACTS[path.name],
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def execute(request: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    normalized = validate_request(request)
    spec = normalized["spec"]
    output_formats = normalized["output_formats"]
    render_mode = normalized["render_mode"]
    validation_profile = normalized["validation_profile"]
    output_dir = _prepare_output_directory(output_directory)

    spec_path = output_dir / "architecture-spec.json"
    source_path = output_dir / "diagram.py"
    _atomic_write(
        spec_path,
        (json.dumps({"schema_version": SCHEMA_VERSION, "spec": spec}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    supplied_source = normalized.get("diagram_code")
    source_origin = "model-generated" if supplied_source is not None else "deterministic-fallback"
    source = supplied_source if supplied_source is not None else generate_diagram_source(spec, output_formats)
    if validation_profile == "diagrams-draw-v2":
        static_validator, semantic_validator = (
            validate_generated_source_v2,
            validate_source_against_spec_v2,
        )
    else:
        static_validator, semantic_validator = (
            validate_generated_source,
            validate_source_against_spec,
        )
    try:
        static_evidence = static_validator(source)
    except SourceValidationError as exc:
        raise ToolFailure("STATIC_VALIDATION_FAILED", str(exc), exit_code=3) from exc
    try:
        semantic_evidence = semantic_validator(source, spec, output_formats)
    except SourceValidationError as exc:
        raise ToolFailure("SEMANTIC_VALIDATION_FAILED", str(exc), exit_code=3) from exc
    _atomic_write(source_path, source.encode("utf-8"))

    diagrams_available = importlib.util.find_spec("diagrams") is not None and shutil.which("dot") is not None
    warnings: list[str] = []
    if render_mode == "diagrams" and not diagrams_available:
        _fail(
            "RENDERER_UNAVAILABLE",
            "render_mode diagrams requires the diagrams package and Graphviz",
            exit_code=4,
        )

    if render_mode != "fallback" and diagrams_available:
        renderer = "python-diagrams-0.25.1+graphviz"
        _render_with_diagrams(source_path, output_dir)
    else:
        renderer = "deterministic-svg-fallback-v1"
        if "png" in output_formats:
            _fail(
                "PNG_RENDERER_UNAVAILABLE",
                "PNG output requires the sandbox image with Python diagrams and Graphviz",
                exit_code=4,
                details={"svg_fallback_available": True},
            )
        _render_fallback_svg(spec, output_dir / "architecture.svg")
        warnings.append("Rendered with the deterministic generic SVG fallback; provider icons are unavailable.")

    if "svg" in output_formats:
        svg_path = output_dir / "architecture.svg"
        if not svg_path.exists() or svg_path.is_symlink():
            _fail("MISSING_ARTIFACT", "renderer did not produce architecture.svg", exit_code=4)
        _validate_svg(svg_path)
    if "png" in output_formats:
        png_path = output_dir / "architecture.png"
        if not png_path.exists() or png_path.is_symlink():
            _fail("MISSING_ARTIFACT", "renderer did not produce architecture.png", exit_code=4)
        _validate_png(png_path)

    validation = {
        "schema": {"status": "passed", "schema_version": SCHEMA_VERSION},
        "static_code": static_evidence,
        "semantic_code": semantic_evidence,
        "source_origin": source_origin,
        "artifacts": {"status": "passed", "formats": output_formats},
        "counts": {
            "components": len(spec["components"]),
            "edges": len(spec["edges"]),
            "boundaries": len(spec["boundaries"]),
        },
    }
    report_path = output_dir / "validation-report.json"
    _atomic_write(
        report_path,
        (
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "renderer": renderer,
                    "warnings": warnings,
                    "validation": validation,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )

    artifact_names = ["architecture-spec.json", "diagram.py"]
    artifact_names.extend(f"architecture.{item}" for item in output_formats)
    artifact_names.append("validation-report.json")
    artifacts = [_artifact(output_dir / name) for name in artifact_names]
    if sum(item["size_bytes"] for item in artifacts) > MAX_ARTIFACT_BYTES:
        _fail("ARTIFACT_LIMIT_EXCEEDED", "combined artifacts exceed 100 MiB", exit_code=4)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "succeeded",
        "renderer": renderer,
        "artifacts": artifacts,
        "warnings": warnings,
        "validation": validation,
    }


def _read_stdin() -> Any:
    data = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
    if len(data) > MAX_STDIN_BYTES:
        _fail("INPUT_TOO_LARGE", f"stdin exceeds {MAX_STDIN_BYTES} bytes")
    if not data.strip():
        _fail("INVALID_JSON", "stdin must contain one JSON object")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("INVALID_JSON", "stdin is not valid UTF-8 JSON", details={"reason": str(exc)})


def _failure_envelope(exc: ToolFailure) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one ArchitectureSpec JSON request from stdin")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = execute(_read_stdin(), arguments.output_dir)
        exit_code = 0
    except ToolFailure as exc:
        result = _failure_envelope(exc)
        exit_code = exc.exit_code
    except Exception as exc:  # Never leak a traceback or local paths through the tool protocol.
        result = _failure_envelope(
            ToolFailure("INTERNAL_ERROR", "unexpected renderer failure", exit_code=5, details={"type": type(exc).__name__})
        )
        exit_code = 5
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
