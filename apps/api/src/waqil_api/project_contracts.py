"""Contracts between files that no single file can violate on its own.

The wiring gate proves each staged file is well-formed and that its Python
imports resolve. That leaves whole categories of "the changeset is broken" it
cannot see, because the defect lives in the *gap between* two files — and those
are the ones that reach the user looking clean.

The case that produced this module: a revamp rewrote eleven React components in
one turn and failed to write the stylesheet they were written for. Every file
parsed. Every Python import resolved. The card said the static checks passed.
On disk, 121 of the 187 CSS classes the components use are defined nowhere, and
the application renders unstyled.

So the checks here are all of the same shape — something is *referenced* in one
staged file and *defined* in no file, staged or on disk:

* a CSS class or custom property used in JSX/HTML that no stylesheet defines
* a local module imported by JS/JSX that does not exist
* a JSON file that does not parse
* a package imported by JS/JSX that package.json does not declare

Severity follows the house rule: something the host can be *sure* about is an
error, and something where the check itself might be wrong is a warning. A
stylesheet can legitimately be generated, and a class can legitimately come from
a library, so the style contract reports a warning unless the proportion is so
high that no other reading is plausible.
"""
from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

ERROR = "error"
WARNING = "warning"

_STYLE_SUFFIXES = frozenset({".css", ".scss", ".sass", ".less"})
_MARKUP_SUFFIXES = frozenset({".jsx", ".tsx", ".html", ".htm", ".vue", ".svelte"})
_SCRIPT_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})

# `class="a b"`, `className="a b"`, and the className={`a ${x}`} form, whose
# static leading tokens are still real class names.
_CLASS_ATTRIBUTE = re.compile(
    r"""\bclass(?:Name)?\s*=\s*(?:\{\s*)?[`"']([^`"'{}]*)""", re.I
)
_VAR_USE = re.compile(r"var\(\s*(--[\w-]+)")
_CSS_CLASS_DEF = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
_CSS_VAR_DEF = re.compile(r"(--[\w-]+)\s*:")
_IMPORT_FROM = re.compile(r"""^\s*import\s+(?:.+?\s+from\s+)?['"]([^'"]+)['"]""", re.M)
_REQUIRE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")
# Tailwind and friends: the stylesheet is a set of directives, and the classes
# it will define do not exist until the build runs.
_UTILITY_DIRECTIVE = re.compile(r"@(tailwind|apply|use\s+['\"]tailwind)", re.I)

# Class names that are never a stylesheet's job.
_IGNORED_CLASSES = frozenset({"", "true", "false", "null", "undefined"})

# How much of a file's class vocabulary may be undefined before this stops
# looking like a library convention and starts looking like a missing file.
_UNDEFINED_SHARE_FOR_ERROR = 0.5


def _finding(path: str, error: str, severity: str = ERROR) -> dict[str, str]:
    return {"path": path, "error": error, "severity": severity}


def _content(entry: Any) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("content", ""))
    return str(entry or "")


def _tokens(text: str) -> set[str]:
    found: set[str] = set()
    for match in _CLASS_ATTRIBUTE.findall(text):
        for token in match.split():
            if token not in _IGNORED_CLASSES and not token.startswith(("$", "{")):
                found.add(token)
    return found


def style_contract_findings(
    staged: Mapping[str, Any],
    stylesheets: Mapping[str, str],
) -> list[dict[str, str]]:
    """Classes and custom properties used by staged markup and defined nowhere.

    ``stylesheets`` is every stylesheet the project has — staged ones override
    their on-disk namesakes, because the overlay is what approval would apply.
    """
    defined_classes: set[str] = set()
    defined_vars: set[str] = set()
    utility_framework = False
    for text in stylesheets.values():
        defined_classes.update(_CSS_CLASS_DEF.findall(text))
        defined_vars.update(_CSS_VAR_DEF.findall(text))
        # A utility framework generates its class vocabulary at build time, so
        # the stylesheet on disk defines almost nothing the markup uses. Judging
        # a Tailwind project against it would report every file as broken —
        # the check would be wrong about the whole project, every time.
        if _UTILITY_DIRECTIVE.search(text):
            utility_framework = True

    findings: list[dict[str, str]] = []
    for path, entry in sorted(staged.items()):
        if Path(path).suffix.lower() not in _MARKUP_SUFFIXES:
            continue
        text = _content(entry)
        used_classes = set() if utility_framework else _tokens(text)
        used_vars = set(_VAR_USE.findall(text))
        missing_classes = sorted(used_classes - defined_classes)
        missing_vars = sorted(used_vars - defined_vars)
        if not missing_classes and not missing_vars:
            continue
        # A missing custom property is unambiguous: `var(--sage)` resolves to
        # nothing and the declaration is simply dropped by the browser.
        share = len(missing_classes) / max(len(used_classes), 1)
        severity = (
            ERROR
            if missing_vars or share >= _UNDEFINED_SHARE_FOR_ERROR
            else WARNING
        )
        parts = []
        if missing_classes:
            shown = ", ".join(missing_classes[:8])
            more = f" and {len(missing_classes) - 8} more" if len(missing_classes) > 8 else ""
            parts.append(f"{len(missing_classes)} CSS class(es) no stylesheet defines: {shown}{more}")
        if missing_vars:
            parts.append(
                f"{len(missing_vars)} CSS variable(s) never declared: "
                + ", ".join(missing_vars[:6])
            )
        findings.append(
            _finding(
                path,
                "; ".join(parts)
                + ". The markup will render unstyled unless the stylesheet that "
                "defines these is part of this change.",
                severity,
            )
        )
    return findings


def local_import_findings(
    staged: Mapping[str, Any],
    on_disk: Iterable[str],
) -> list[dict[str, str]]:
    """Relative imports in staged JS/JSX that resolve to no file.

    The Python half of this has existed in the wiring gate since the beginning;
    a project whose frontend is half its code deserves the same check.
    """
    known = {str(path) for path in on_disk} | {str(path) for path in staged}
    findings: list[dict[str, str]] = []
    for path, entry in sorted(staged.items()):
        suffix = Path(path).suffix.lower()
        if suffix not in _SCRIPT_SUFFIXES:
            continue
        text = _content(entry)
        here = Path(path).parent
        for target in [*_IMPORT_FROM.findall(text), *_REQUIRE.findall(text)]:
            if not target.startswith("."):
                continue  # a package, not a file — see package_findings
            # `posixpath.normpath`, not `Path`: Path preserves `..` as a
            # literal segment, so `components/../api.js` never matched
            # `api.js` and every relative import out of a subdirectory was
            # reported missing. Normalising is pure string work — it must not
            # touch the filesystem, because the overlay is not on it.
            resolved = posixpath.normpath((here / target).as_posix())
            candidates = {resolved}
            if not Path(resolved).suffix:
                for extension in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".json", ".css"):
                    candidates.add(f"{resolved}{extension}")
                    candidates.add(f"{resolved}/index{extension}")
            if candidates & known:
                continue
            findings.append(
                _finding(
                    path,
                    f"imports '{target}', which resolves to no file in this "
                    "project or this changeset",
                )
            )
    return findings


def json_findings(staged: Mapping[str, Any]) -> list[dict[str, str]]:
    """A staged .json file that does not parse.

    Python gets a stage-time parse gate, so a build cannot end on a Python
    syntax error. Configuration, fixtures, presets and package manifests had no
    equivalent, and a JSON file that will not load fails at runtime rather than
    at review.
    """
    findings: list[dict[str, str]] = []
    for path, entry in sorted(staged.items()):
        if Path(path).suffix.lower() != ".json":
            continue
        text = _content(entry)
        if not text.strip():
            continue
        try:
            json.loads(text)
        except ValueError as error:
            findings.append(_finding(path, f"is not valid JSON: {str(error)[:200]}"))
    return findings


def package_findings(
    staged: Mapping[str, Any],
    package_manifests: Mapping[str, str],
) -> list[dict[str, str]]:
    """Packages imported by staged JS/JSX that no package.json declares.

    A warning, never an error: monorepo aliases, path mappings and transitive
    imports are all legitimate, and the manifest read here may not be the one
    that governs this file. The undeclared *Python* dependency check made the
    same call for the same reason.
    """
    declared: set[str] = set()
    for text in package_manifests.values():
        try:
            manifest = json.loads(text)
        except ValueError:
            continue
        for field in ("dependencies", "devDependencies", "peerDependencies"):
            section = manifest.get(field)
            if isinstance(section, dict):
                declared.update(section)
    if not declared:
        return []

    findings: list[dict[str, str]] = []
    for path, entry in sorted(staged.items()):
        if Path(path).suffix.lower() not in _SCRIPT_SUFFIXES:
            continue
        missing: list[str] = []
        for target in _IMPORT_FROM.findall(_content(entry)):
            if target.startswith((".", "/", "@/", "~")):
                continue
            # "@scope/name/sub" declares as "@scope/name"; "name/sub" as "name".
            parts = target.split("/")
            root = "/".join(parts[:2]) if target.startswith("@") else parts[0]
            if root not in declared and root not in missing:
                missing.append(root)
        if missing:
            findings.append(
                _finding(
                    path,
                    "imports package(s) package.json does not declare: "
                    + ", ".join(sorted(missing)[:6]),
                    WARNING,
                )
            )
    return findings


def cross_file_findings(
    staged: Mapping[str, Any],
    *,
    on_disk: Iterable[str] = (),
    disk_sources: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Every cross-file contract this changeset breaks.

    ``disk_sources`` carries the non-Python files the checks need to read —
    stylesheets and package manifests — because "which classes exist" cannot be
    answered from the overlay alone when the project already has a design system.
    """
    sources = dict(disk_sources or {})
    stylesheets = {
        path: text
        for path, text in sources.items()
        if Path(path).suffix.lower() in _STYLE_SUFFIXES
    }
    # A staged stylesheet is what approval would apply, so it wins.
    for path, entry in staged.items():
        if Path(path).suffix.lower() in _STYLE_SUFFIXES:
            stylesheets[path] = _content(entry)
    manifests = {
        path: text for path, text in sources.items() if Path(path).name == "package.json"
    }
    for path, entry in staged.items():
        if Path(path).name == "package.json":
            manifests[path] = _content(entry)

    findings = [*json_findings(staged), *local_import_findings(staged, on_disk)]
    # Without a stylesheet anywhere there is no contract to check against, and
    # every class would be reported — a project styled entirely by a framework
    # would be nothing but false positives.
    if stylesheets:
        findings.extend(style_contract_findings(staged, stylesheets))
        # And the other direction. A turn that stages ONLY a stylesheet is the
        # repair turn — the markup is already on disk and already broken — so
        # checking staged markup finds nothing and the card calls it clean. It
        # happened: a repair asked to define 121 missing classes staged a
        # fifty-one-byte edit and was reported as a clean one-file change.
        staged_styles = [
            path for path in staged if Path(path).suffix.lower() in _STYLE_SUFFIXES
        ]
        if staged_styles:
            findings.extend(
                _existing_markup_findings(staged_styles, stylesheets, sources)
            )
    findings.extend(package_findings(staged, manifests))
    return findings


def _existing_markup_findings(
    staged_styles: list[str],
    stylesheets: Mapping[str, str],
    disk_sources: Mapping[str, str],
) -> list[dict[str, str]]:
    """What the markup already in the project still needs from this stylesheet.

    Reported against the staged stylesheet rather than against each file that
    uses it: the stylesheet is what this turn is changing, and one finding
    naming the shortfall is more use than a dozen naming its consequences.
    """
    markup = {
        path: text
        for path, text in disk_sources.items()
        if Path(path).suffix.lower() in _MARKUP_SUFFIXES
    }
    if not markup:
        return []
    combined = "\n".join(stylesheets.values())
    if _UTILITY_DIRECTIVE.search(combined):
        return []
    defined_classes = set(_CSS_CLASS_DEF.findall(combined))
    defined_vars = set(_CSS_VAR_DEF.findall(combined))

    missing_classes: set[str] = set()
    missing_vars: set[str] = set()
    affected: set[str] = set()
    for path, text in markup.items():
        gaps = _tokens(text) - defined_classes
        holes = set(_VAR_USE.findall(text)) - defined_vars
        if gaps or holes:
            affected.add(path)
        missing_classes |= gaps
        missing_vars |= holes
    if not missing_classes and not missing_vars:
        return []

    detail = []
    if missing_classes:
        detail.append(
            f"{len(missing_classes)} class(es) still undefined "
            f"({', '.join(sorted(missing_classes)[:8])}…)"
        )
    if missing_vars:
        detail.append(
            f"{len(missing_vars)} variable(s) still undeclared "
            f"({', '.join(sorted(missing_vars)[:6])})"
        )
    return [
        _finding(
            staged_styles[0],
            "; ".join(detail)
            + f" — used by {len(affected)} file(s) already in this project. "
            "This change does not yet give them a stylesheet to render against.",
            ERROR,
        )
    ]


def style_gaps(
    stylesheets: Mapping[str, str], markup: Mapping[str, str]
) -> dict[str, list[str]]:
    """Exactly which classes and custom properties the markup needs and lacks.

    The host can answer this with two regexes in milliseconds. Asking the model
    to "read the components and collect every className" instead is asking it to
    re-derive, across eleven files and inside one output budget, something
    already known — and a repair turn measured doing that produced a fifty-one
    byte edit. Handing over the list turns authorship into the mechanical task
    it should be.
    """
    combined = "\n".join(stylesheets.values())
    if _UTILITY_DIRECTIVE.search(combined):
        return {"classes": [], "variables": []}
    defined_classes = set(_CSS_CLASS_DEF.findall(combined))
    defined_vars = set(_CSS_VAR_DEF.findall(combined))
    classes: set[str] = set()
    variables: set[str] = set()
    for text in markup.values():
        classes |= _tokens(text) - defined_classes
        variables |= set(_VAR_USE.findall(text)) - defined_vars
    return {"classes": sorted(classes), "variables": sorted(variables)}


def is_stylesheet(path: str) -> bool:
    return Path(path).suffix.lower() in _STYLE_SUFFIXES
