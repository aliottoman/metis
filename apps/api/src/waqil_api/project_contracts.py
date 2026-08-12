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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping

ERROR = "error"
WARNING = "warning"

_STYLE_SUFFIXES = frozenset({".css", ".scss", ".sass", ".less"})
_MARKUP_SUFFIXES = frozenset({".jsx", ".tsx", ".html", ".htm", ".vue", ".svelte"})
_SCRIPT_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})

# The whole attribute value, delimiter and all, so the value can be processed
# rather than truncated. The previous pattern stopped at the first `{`, which
# both lost every literal class after an interpolation and kept the fragment
# before it — `className={`rail-btn${x}`}` was read as the class `rail-btn$`.
_CLASS_ATTRIBUTE = re.compile(
    r"""\bclass(?:Name)?\s*=\s*(?:\{\s*)?(?P<q>["'`])(?P<value>(?:\\.|(?!\1).)*)\1""",
    re.I | re.S,
)
# A `${ ... }` span inside a template literal, including nested braces one deep
# (`${a ? "x" : "y"}`), replaced wholesale before tokenising.
_INTERPOLATION = re.compile(r"\$\{(?:[^{}]|\{[^{}]*\})*\}")
# Stands in for an interpolation while tokenising: not whitespace, so it stays
# joined to whatever it was joined to, which is exactly what marks a fragment.
_DYNAMIC = "\x00"
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
    """The class names a file uses literally, with every dynamic part discarded.

    An interpolation is replaced by a non-space sentinel before splitting, so
    adjacency survives the substitution and decides the outcome:

    * ``className={`rail-btn ${extra}`}`` — the space means ``rail-btn`` is a
      complete class, and it is kept.
    * ``className={`rail-btn${suffix}`}`` — no space, so the real class is
      ``rail-btn`` plus something unknowable. Reporting ``rail-btn`` (or worse,
      ``rail-btn$``) as missing would be an invented finding, so the whole
      token is dropped.

    Anything still carrying a brace, a dollar or the sentinel is a fragment of
    an expression rather than a name, and never becomes a finding.
    """
    found: set[str] = set()
    for match in _CLASS_ATTRIBUTE.finditer(text):
        value = _INTERPOLATION.sub(_DYNAMIC, match.group("value"))
        for token in value.split():
            if token in _IGNORED_CLASSES:
                continue
            if _DYNAMIC in token or any(ch in token for ch in "${}"):
                continue
            found.add(token)
    return found


class _InlineStyleParser(HTMLParser):
    """Collect real ``<style>`` element bodies from an HTML-like document.

    ``HTMLParser`` deliberately does the element recognition here. A regex over
    the whole document would also accept examples inside comments, escaped
    markup, and JavaScript strings, allowing text that the browser never treats
    as CSS to satisfy the contract.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[list[str]] = []
        self._style_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() == "style":
            self._style_depth += 1
            self.blocks.append([])

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self.blocks[-1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "style" and self._style_depth:
            self._style_depth -= 1


def _inline_style_blocks(document: str) -> list[str]:
    parser = _InlineStyleParser()
    parser.feed(document)
    parser.close()
    return ["".join(chunks) for chunks in parser.blocks]


class _ScriptSourceParser(HTMLParser):
    """Collect ``src`` values from real script elements only.

    HTML comments and JavaScript string literals can contain convincing
    ``<script src=...>`` examples. ``HTMLParser`` keeps those as comment/data
    nodes, so only markup the browser would actually load can satisfy the
    contract.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "script":
            return
        for name, value in attrs:
            if name.casefold() == "src" and value:
                self.sources.append(value.strip())


def _script_sources(document: str) -> list[str]:
    parser = _ScriptSourceParser()
    parser.feed(document)
    parser.close()
    return parser.sources


def _loads_sibling_script(html_path: str, src: str, sibling: str) -> bool:
    """Whether a browser-facing script URL names this HTML file's sibling."""
    # Queries and fragments do not change the asset being loaded. Protocol and
    # protocol-relative URLs are not local project paths and must not happen to
    # pass merely because their suffix is ``/static/app.js``.
    reference = src.split("#", 1)[0].split("?", 1)[0].strip()
    if not reference or "://" in reference or reference.startswith("//"):
        return False

    sibling = posixpath.normpath(sibling)
    if reference.startswith("/"):
        public_path = posixpath.normpath(reference.lstrip("/"))
        aliases = {sibling}
        sibling_parts = sibling.split("/")
        # Common Python web layout: ``app/static/index.html`` is served from
        # ``/static/index.html``. The first filesystem package is not part of
        # the public URL, while ``/app/static/app.js`` remains acceptable too.
        if len(sibling_parts) > 1:
            aliases.add("/".join(sibling_parts[1:]))
        return public_path in aliases

    resolved = posixpath.normpath(
        posixpath.join(posixpath.dirname(html_path), reference)
    )
    return resolved == sibling


def script_asset_findings(
    staged: Mapping[str, Any],
    on_disk: Iterable[str],
    disk_sources: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Require a changed conventional ``app.js``/HTML pair to be connected.

    This is intentionally not a general "load every JavaScript file" rule.
    It applies when app-owned HTML is staged beside a known ``app.js``, or when
    ``app.js`` itself is staged beside existing HTML. That narrow contract covers
    the generated-app convention without guessing about bundles, worker scripts,
    test helpers, or framework-owned ``appkit`` assets.
    """
    known = {str(path) for path in on_disk} | {str(path) for path in staged}
    sources = dict(disk_sources or {})
    staged_scripts = {
        posixpath.normpath(str(path))
        for path in staged
        if Path(path).name.casefold() == "app.js"
    }
    documents: dict[str, str] = {
        str(path): _content(entry)
        for path, entry in staged.items()
        if Path(path).suffix.lower() in {".html", ".htm"}
    }
    # The reverse direction matters too: a turn may extract an existing page's
    # inline behavior into app.js without touching the page. Inspect untouched
    # HTML only when its exact sibling app.js is staged, so an unrelated JS/CSS
    # or backend edit never turns into a surprise frontend contract.
    for path, content in sources.items():
        if path in documents or Path(path).suffix.lower() not in {".html", ".htm"}:
            continue
        sibling = posixpath.normpath(posixpath.join(posixpath.dirname(path), "app.js"))
        if sibling in staged_scripts:
            documents[path] = content

    findings: list[dict[str, str]] = []
    for path, content in sorted(documents.items()):
        file_path = Path(path)
        if file_path.parts and file_path.parts[0].casefold() == "appkit":
            continue
        sibling = posixpath.normpath(posixpath.join(posixpath.dirname(path), "app.js"))
        if sibling not in known:
            continue
        if any(
            _loads_sibling_script(path, src, sibling)
            for src in _script_sources(content)
        ):
            continue
        findings.append(
            _finding(
                path,
                f"does not load its sibling '{sibling}' through a real "
                "<script src=...> element; inline JavaScript does not connect "
                "that asset to the page",
            )
        )
    return findings


def _style_symbols(texts: Iterable[str]) -> tuple[set[str], set[str], bool]:
    combined = "\n".join(texts)
    return (
        set(_CSS_CLASS_DEF.findall(combined)),
        set(_CSS_VAR_DEF.findall(combined)),
        bool(_UTILITY_DIRECTIVE.search(combined)),
    )


def _is_app_owned_stylesheet(path: str) -> bool:
    """Whether an authored sheet belongs to the generated application.

    ``appkit/`` is Metis' staged framework scaffold. Directing a product-style
    repair into its shared theme would mutate infrastructure instead of the app
    that owns the markup.
    """
    parts = Path(path).parts
    return (
        Path(path).suffix.lower() in _STYLE_SUFFIXES
        and bool(parts)
        and parts[0].casefold() != "appkit"
    )


def style_contract_findings(
    staged: Mapping[str, Any],
    stylesheets: Mapping[str, str],
) -> list[dict[str, str]]:
    """Classes and custom properties used by staged markup and defined nowhere.

    ``stylesheets`` is every stylesheet the project has — staged ones override
    their on-disk namesakes, because the overlay is what approval would apply.
    Real inline ``<style>`` elements are read from each staged document too.
    """
    defined_classes, defined_vars, utility_framework = _style_symbols(
        stylesheets.values()
    )
    inline_classes, inline_vars, inline_utility = _style_symbols(
        block
        for path, entry in staged.items()
        if Path(path).suffix.lower() in _MARKUP_SUFFIXES
        for block in _inline_style_blocks(_content(entry))
    )
    defined_classes |= inline_classes
    defined_vars |= inline_vars
    utility_framework = utility_framework or inline_utility
    # If this turn authors exactly one app-owned stylesheet, that file is the
    # only deterministic repair owner. Framework styles are context, not a
    # product repair target. With zero or several, retain the markup path:
    # guessing among several sheets is worse than asking the author to choose.
    staged_styles = sorted(path for path in staged if _is_app_owned_stylesheet(path))
    repair_target = staged_styles[0] if len(staged_styles) == 1 else None

    findings: list[dict[str, str]] = []
    for path, entry in sorted(staged.items()):
        if Path(path).suffix.lower() not in _MARKUP_SUFFIXES:
            continue
        text = _content(entry)
        # A utility framework generates its class vocabulary at build time, so
        # the stylesheet on disk defines almost nothing the markup uses. Judging
        # a Tailwind project against it would report every file as broken —
        # the check would be wrong about the whole project, every time.
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
            ERROR if missing_vars or share >= _UNDEFINED_SHARE_FOR_ERROR else WARNING
        )
        parts = []
        if missing_classes:
            shown = ", ".join(missing_classes[:8])
            more = (
                f" and {len(missing_classes) - 8} more"
                if len(missing_classes) > 8
                else ""
            )
            parts.append(
                f"{len(missing_classes)} CSS class(es) no stylesheet defines: {shown}{more}"
            )
        if missing_vars:
            parts.append(
                f"{len(missing_vars)} CSS variable(s) never declared: "
                + ", ".join(missing_vars[:6])
            )
        findings.append(
            _finding(
                repair_target or path,
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
                for extension in (
                    ".js",
                    ".jsx",
                    ".ts",
                    ".tsx",
                    ".mjs",
                    ".json",
                    ".css",
                ):
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
    disk_paths = tuple(str(path) for path in on_disk)
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
    # Inline style elements participate in the same project-wide cascade as an
    # external sheet. Include existing documents too, unless this turn replaces
    # that document (the staged body is already collected by the checker).
    for path, text in sources.items():
        if path in staged or Path(path).suffix.lower() not in _MARKUP_SUFFIXES:
            continue
        for index, block in enumerate(_inline_style_blocks(text)):
            stylesheets[f"{path}#inline-style-{index}"] = block
    manifests = {
        path: text
        for path, text in sources.items()
        if Path(path).name == "package.json"
    }
    for path, entry in staged.items():
        if Path(path).name == "package.json":
            manifests[path] = _content(entry)

    findings = [
        *json_findings(staged),
        *local_import_findings(staged, disk_paths),
        *script_asset_findings(staged, disk_paths, sources),
    ]
    # Without a stylesheet file or inline style element there is no contract to
    # check against, and every class would be reported — a project styled
    # entirely by a framework would be nothing but false positives.
    staged_inline_styles = any(
        _inline_style_blocks(_content(entry))
        for path, entry in staged.items()
        if Path(path).suffix.lower() in _MARKUP_SUFFIXES
    )
    if stylesheets or staged_inline_styles:
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

    Reported against the sole staged app stylesheet when there is one: that
    stylesheet is what this turn is changing, and one finding naming the
    shortfall is more use than a dozen naming its consequences. With no app
    sheet or several, attribution stays on affected markup rather than guessing
    or directing a product repair into the shared ``appkit/`` theme.
    """
    markup = {
        path: text
        for path, text in disk_sources.items()
        if Path(path).suffix.lower() in _MARKUP_SUFFIXES
    }
    if not markup:
        return []
    defined_classes, defined_vars, utility_framework = _style_symbols(
        stylesheets.values()
    )
    inline_classes, inline_vars, inline_utility = _style_symbols(
        block for text in markup.values() for block in _inline_style_blocks(text)
    )
    defined_classes |= inline_classes
    defined_vars |= inline_vars
    utility_framework = utility_framework or inline_utility
    if utility_framework:
        return []

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

    app_styles = [path for path in staged_styles if _is_app_owned_stylesheet(path)]
    repair_target = app_styles[0] if len(app_styles) == 1 else sorted(affected)[0]

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
            repair_target,
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

    The host can answer this with one safe HTML parse and two regexes in
    milliseconds. Asking the model to "read the components and collect every
    className" instead is asking it to re-derive, across eleven files and inside
    one output budget, something already known — and a repair turn measured
    doing that produced a fifty-one byte edit. Handing over the list turns
    authorship into the mechanical task it should be.
    """
    defined_classes, defined_vars, utility_framework = _style_symbols(
        stylesheets.values()
    )
    inline_classes, inline_vars, inline_utility = _style_symbols(
        block for text in markup.values() for block in _inline_style_blocks(text)
    )
    defined_classes |= inline_classes
    defined_vars |= inline_vars
    utility_framework = utility_framework or inline_utility
    if utility_framework:
        return {"classes": [], "variables": []}
    classes: set[str] = set()
    variables: set[str] = set()
    for text in markup.values():
        classes |= _tokens(text) - defined_classes
        variables |= set(_VAR_USE.findall(text)) - defined_vars
    return {"classes": sorted(classes), "variables": sorted(variables)}


def is_stylesheet(path: str) -> bool:
    return Path(path).suffix.lower() in _STYLE_SUFFIXES
