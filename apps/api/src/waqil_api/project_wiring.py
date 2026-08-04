"""Static cross-file checks over a staged project changeset.

The syntax gate proves that each staged file parses. This proves the files fit
together: that an import points at a file the project actually has, that a name
it pulls in is really defined there, that a build wired up what it constructed,
and that nothing was left as a stub. Every check is pure AST inspection —
nothing is imported and nothing is executed — so it is safe on code the model
wrote a moment ago, it costs microseconds, and it still works when the container
sandbox is unavailable.
"""
from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

ERROR = "error"
WARNING = "warning"


# Import roots whose installable distribution is named differently from the
# module they provide. Deliberately short: this table only has to cover what a
# Metis build actually reaches for, and a guessed entry would produce exactly
# the false warning the table exists to prevent.
_DISTRIBUTION_ALIASES: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "jwt": "pyjwt",
    "multipart": "python-multipart",
    "pil": "pillow",
    "yaml": "pyyaml",
}

# A requirement line's distribution name, before any version or extras marker.
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Directories whose contents are not part of the project's own import namespace.
_SKIP_DIRECTORIES = frozenset({".venv", "venv", "node_modules", "__pycache__", "build", "dist"})


def _finding(path: str, error: str, severity: str = ERROR) -> dict[str, str]:
    """One reportable defect, in the same shape the syntax gate returns."""
    return {"path": path, "error": error, "severity": severity}


def module_name_for(path: str) -> str | None:
    """The dotted module a project-relative .py path provides, if any."""
    parts = Path(path).as_posix().split("/")
    if any(part in _SKIP_DIRECTORIES for part in parts[:-1]):
        return None
    name = parts[-1]
    if not name.endswith(".py"):
        return None
    if name == "__init__.py":
        parts = parts[:-1]
    else:
        parts = [*parts[:-1], name[: -len(".py")]]
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _module_index(paths: Iterable[str]) -> tuple[dict[str, str], set[str]]:
    """Every dotted module the project provides, plus its package prefixes.

    Package prefixes are collected from the module paths themselves rather than
    from `__init__.py` files, because an implicit namespace package imports
    perfectly well without one — treating it as unresolvable would report a
    working import as broken.
    """
    modules: dict[str, str] = {}
    packages: set[str] = set()
    for path in paths:
        dotted = module_name_for(path)
        if dotted is None:
            continue
        modules[dotted] = path
        segments = dotted.split(".")
        for index in range(1, len(segments)):
            packages.add(".".join(segments[:index]))
    return modules, packages


def _package_of(path: str, dotted: str) -> str:
    """The package a module resolves relative imports against."""
    if Path(path).name == "__init__.py":
        return dotted
    return dotted.rpartition(".")[0]


def _absolute_module(package: str, level: int, module: str | None) -> str | None:
    """Turn a relative `from .. import x` into the module it actually names."""
    if level <= 0:
        return module
    base = package
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    if not base and level > 1:
        return None
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def _top_level_names(tree: ast.Module) -> set[str]:
    """Every name a module binds at its top level, as an importer would see it."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_bound_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_bound_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.If | ast.Try):
            # Conditional definitions (a try/except import fallback, a
            # TYPE_CHECKING block) still bind names an importer can reach.
            for child in ast.walk(node):
                if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    names.add(child.name)
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        names.update(_bound_names(target))
    return names


def _bound_names(target: ast.expr) -> set[str]:
    """The plain names an assignment target binds, ignoring attributes."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in target.elts:
            names.update(_bound_names(element))
        return names
    return set()


def _exports_opaquely(tree: ast.Module) -> bool:
    """Whether a module's exported names cannot be read off its top level.

    A star import or a module-level `__getattr__` means names appear that no
    static reader can enumerate, so the symbol check has to stand down rather
    than report a name it simply cannot see.
    """
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ):
            return True
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            return True
    return False


def _is_stub_body(body: list[ast.stmt]) -> bool:
    """Whether a function body does nothing but announce that it does nothing."""
    for statement in body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.Raise):
            exception = statement.exc
            if isinstance(exception, ast.Call):
                exception = exception.func
            if isinstance(exception, ast.Name) and exception.id == "NotImplementedError":
                continue
            return False
        else:
            return False
    return True


def _declares_an_interface(tree: ast.Module) -> bool:
    """Whether a module is deliberately abstract, so empty bodies are correct."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if name in {"ABC", "Protocol"}:
                    return True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = (
                    target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                )
                if name in {"abstractmethod", "abstractproperty", "overload"}:
                    return True
    return False


def _calls(tree: ast.Module, name: str) -> list[ast.Call]:
    """Every call of a function or method with this name, at any depth."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else ""
        )
        if called == name:
            found.append(node)
    return found


def declared_distributions(requirements: str) -> set[str]:
    """The distribution names a requirements or pyproject text declares."""
    declared: set[str] = set()
    for line in requirements.splitlines():
        stripped = line.strip().strip('",\'')
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = _REQUIREMENT_NAME.match(stripped)
        if match:
            declared.add(match.group(1).replace("_", "-").casefold())
    return declared


def _distribution_for(root: str) -> str:
    """The distribution name an import root most likely comes from."""
    return _DISTRIBUTION_ALIASES.get(root.casefold(), root.replace("_", "-").casefold())


def staged_wiring_errors(
    staged: dict[str, dict[str, Any]],
    *,
    sources: dict[str, str] | None = None,
    project_paths: Iterable[str] = (),
    requirements: str = "",
) -> list[dict[str, str]]:
    """Cross-file defects in a staged changeset, worst kind first.

    ``sources`` is every Python file the host can read — the staged text plus
    whatever already exists on disk — so an import into an untouched part of the
    project still resolves. Only staged files are reported on: the changeset is
    what the user is about to approve, and faults it did not introduce are not
    its to answer for.

    A clean result means "nothing checkable is wrong", never "correct". Findings
    carry a severity: ``error`` is something the host can prove, ``warning`` is
    worth a human's eye but is never worth sending the model back to fix.
    """
    texts = dict(sources or {})
    for path, entry in staged.items():
        texts[path] = str(entry.get("content", ""))
    paths = {*texts, *project_paths, *staged}
    modules, packages = _module_index(paths)

    trees: dict[str, ast.Module] = {}
    for path, text in texts.items():
        if not path.endswith((".py", ".pyi")):
            continue
        try:
            trees[path] = ast.parse(text, filename=path)
        except (SyntaxError, ValueError):
            # The syntax gate owns unparseable files and reports them with the
            # real error; re-reporting them here would only duplicate it.
            continue

    declared = declared_distributions(requirements)
    findings: list[dict[str, str]] = []
    for path in sorted(staged):
        tree = trees.get(path)
        if tree is None:
            continue
        dotted = module_name_for(path)
        package = _package_of(path, dotted) if dotted else ""
        findings.extend(
            _import_findings(
                path,
                tree,
                package=package,
                modules=modules,
                packages=packages,
                trees=trees,
                declared=declared,
                has_requirements=bool(declared),
            )
        )
        findings.extend(_stub_findings(path, tree))
        findings.extend(_multipart_findings(path, tree, declared))
    findings.extend(_unwired_findings(staged, trees))
    findings.extend(_static_mount_findings(staged, trees, paths))
    findings.extend(_asset_reference_findings(staged, texts, paths, trees))
    findings.sort(key=lambda item: (item["severity"] != ERROR, item["path"]))
    return findings


# The FastAPI names whose presence makes python-multipart a hard runtime
# requirement. Import analysis cannot see it — nothing ever imports the
# package by name — which is exactly why it needs its own rule.
_MULTIPART_MARKS = frozenset({"UploadFile", "File", "Form"})


def _multipart_findings(
    path: str, tree: ast.Module, declared: set[str]
) -> list[dict[str, str]]:
    """A FastAPI upload route whose implicit dependency is undeclared.

    FastAPI resolves File/Form/UploadFile parameters through python-multipart
    and raises at startup — or 500s on the first real upload — when it is
    missing. The rule fires on usage (importing those names from fastapi),
    never on the mere absence of the requirement, so an app without uploads
    cannot false-alarm.
    """
    if not declared or "python-multipart" in declared:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "fastapi":
            used = sorted({alias.name for alias in node.names} & _MULTIPART_MARKS)
            if used:
                return [
                    _finding(
                        path,
                        f"uses fastapi.{used[0]} on line {node.lineno}, which "
                        "needs python-multipart declared in requirements — "
                        "without it FastAPI fails at startup or the upload "
                        "route returns 500",
                    )
                ]
    return []


def _import_findings(
    path: str,
    tree: ast.Module,
    *,
    package: str,
    modules: dict[str, str],
    packages: set[str],
    trees: dict[str, ast.Module],
    declared: set[str],
    has_requirements: bool,
) -> list[dict[str, str]]:
    """Imports that name a module the project lacks, or a name it never defines."""
    findings: list[dict[str, str]] = []
    local_roots = {name.split(".")[0] for name in (*modules, *packages)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                findings.extend(
                    _module_finding(
                        path, alias.name, node.lineno, modules, packages, local_roots
                    )
                )
                findings.extend(
                    _dependency_finding(
                        path,
                        alias.name.split(".")[0],
                        node.lineno,
                        local_roots,
                        declared,
                        has_requirements,
                    )
                )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _absolute_module(package, node.level, node.module)
        if target is None:
            findings.append(
                _finding(
                    path,
                    f"relative import on line {node.lineno} reaches above the project root",
                )
            )
            continue
        root = target.split(".")[0]
        if node.level == 0:
            findings.extend(
                _dependency_finding(
                    path, root, node.lineno, local_roots, declared, has_requirements
                )
            )
        if root not in local_roots:
            continue
        if target not in modules and target not in packages:
            findings.append(
                _finding(
                    path,
                    f"imports {target} on line {node.lineno}, but no file in this "
                    "project provides that module",
                )
            )
            continue
        provider = trees.get(modules.get(target, ""))
        if provider is None or _exports_opaquely(provider):
            continue
        available = _top_level_names(provider)
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name in available or f"{target}.{alias.name}" in modules:
                continue
            if f"{target}.{alias.name}" in packages:
                continue
            findings.append(
                _finding(
                    path,
                    f"imports {alias.name} from {target} on line {node.lineno}, "
                    f"but {target} does not define it",
                )
            )
    return findings


def _module_finding(
    path: str,
    name: str,
    lineno: int,
    modules: dict[str, str],
    packages: set[str],
    local_roots: set[str],
) -> list[dict[str, str]]:
    """A plain `import a.b` that names a local module the project does not have."""
    if name.split(".")[0] not in local_roots:
        return []
    if name in modules or name in packages:
        return []
    return [
        _finding(
            path,
            f"imports {name} on line {lineno}, but no file in this project "
            "provides that module",
        )
    ]


def _dependency_finding(
    path: str,
    root: str,
    lineno: int,
    local_roots: set[str],
    declared: set[str],
    has_requirements: bool,
) -> list[dict[str, str]]:
    """A third-party import the project never declares as a dependency.

    A warning rather than an error on purpose: mapping an import root back to
    its distribution is the one heuristic here, and a false positive would send
    the model off to "fix" code that is perfectly correct.
    """
    if not has_requirements or root in local_roots:
        return []
    if root in sys.stdlib_module_names or root.startswith("_"):
        return []
    if _distribution_for(root) in declared or root.replace("_", "-").casefold() in declared:
        return []
    return [
        _finding(
            path,
            f"imports {root} on line {lineno}, which the project's requirements "
            "do not declare",
            WARNING,
        )
    ]


def _stub_findings(path: str, tree: ast.Module) -> list[dict[str, str]]:
    """A module whose every function is an empty placeholder."""
    if Path(path).name == "__init__.py" or _declares_an_interface(tree):
        return []
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not functions or not all(_is_stub_body(node.body) for node in functions):
        return []
    return [
        _finding(
            path,
            f"defines {len(functions)} function(s) and implements none of them — "
            "every body is empty, `pass`, or NotImplementedError",
        )
    ]


_FILE_SENTINEL = "\x00__file__"


def _toplevel_assigns(tree: ast.Module) -> dict[str, ast.expr]:
    """Top-level `NAME = expr` bindings, for resolving a mount's directory arg."""
    assigns: dict[str, ast.expr] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigns[node.targets[0].id] = node.value
    return assigns


def _module_dir(path: str) -> str:
    """The project-relative directory a module file sits in ("" at the root)."""
    posix = Path(path).as_posix()
    return posix.rsplit("/", 1)[0] if "/" in posix else ""


def _norm_reldir(value: str) -> str | None:
    """A project-relative directory, or None if it is absolute / unusable."""
    value = value.strip()
    if not value or value.startswith(("/", "~")):
        return None
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def _resolve_dir(
    expr: ast.expr | None, module_dir: str, assigns: dict[str, ast.expr], depth: int = 0
) -> str | None:
    """Best-effort project-relative directory an expression names.

    Covers the shapes real FastAPI code uses for a static mount: a literal, a
    name bound to one, `Path(__file__).parent / "static"`, and
    `os.path.join(os.path.dirname(__file__), "static")`. Anything it cannot read
    with certainty returns None, so an unresolvable directory is never a finding.
    Returns the sentinel for an expression that denotes ``__file__`` itself.
    """
    if expr is None or depth > 6:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return _norm_reldir(expr.value)
    if isinstance(expr, ast.Name):
        if expr.id == "__file__":
            return _FILE_SENTINEL
        return _resolve_dir(assigns.get(expr.id), module_dir, assigns, depth + 1)
    if isinstance(expr, ast.Attribute):
        if expr.attr == "parent":
            base = _resolve_dir(expr.value, module_dir, assigns, depth + 1)
            if base == _FILE_SENTINEL:
                return module_dir
            if isinstance(base, str):
                return base.rsplit("/", 1)[0] if "/" in base else ""
            return None
        if expr.attr in {"resolve", "absolute"}:
            return _resolve_dir(expr.value, module_dir, assigns, depth + 1)
        return None
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        base = _resolve_dir(expr.left, module_dir, assigns, depth + 1)
        base = module_dir if base == _FILE_SENTINEL else base
        right = expr.right
        if (
            isinstance(base, str)
            and isinstance(right, ast.Constant)
            and isinstance(right.value, str)
        ):
            segment = right.value.strip("/")
            return _norm_reldir(f"{base}/{segment}" if base else segment)
        return None
    if isinstance(expr, ast.Call):
        func = expr.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "Path" and expr.args:
            return _resolve_dir(expr.args[0], module_dir, assigns, depth + 1)
        if name in {"resolve", "absolute"} and isinstance(func, ast.Attribute):
            return _resolve_dir(func.value, module_dir, assigns, depth + 1)
        if name == "dirname" and expr.args:  # os.path.dirname(__file__)
            inner = _resolve_dir(expr.args[0], module_dir, assigns, depth + 1)
            return module_dir if inner == _FILE_SENTINEL else None
        if name == "join" and expr.args:  # os.path.join(base, "static", ...)
            base = _resolve_dir(expr.args[0], module_dir, assigns, depth + 1)
            base = module_dir if base == _FILE_SENTINEL else base
            if not isinstance(base, str):
                return None
            parts = [base] if base else []
            for arg in expr.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(arg.value.strip("/"))
                else:
                    return None
            return _norm_reldir("/".join(parts))
        return None
    return None


def _static_mount_findings(
    staged: dict[str, dict[str, Any]],
    trees: dict[str, ast.Module],
    all_paths: set[str],
) -> list[dict[str, str]]:
    """A StaticFiles mount whose directory the project never provides.

    `StaticFiles(directory=...)` stats that directory the moment it is built
    (``check_dir`` defaults to True), so a mount at a path no file in the
    changeset or the project on disk creates raises at import and the app cannot
    start. This is environment-independent — the directory is part of what the
    build ships, not of the machine it runs on — so blocking on it never rejects
    correct code. It only fires when the directory is resolvable AND provably
    absent AND ``check_dir`` was not turned off, which is why it is an error the
    sandbox would otherwise only rediscover by failing to import (and, when a
    declared-but-unbaked dependency imports first, not even then).
    """

    def directory_exists(directory: str) -> bool:
        if directory == "":
            return True
        prefix = f"{directory}/"
        return any(path == directory or path.startswith(prefix) for path in all_paths)

    findings: list[dict[str, str]] = []
    for path in sorted(p for p in trees if p in staged):
        tree = trees[path]
        assigns = _toplevel_assigns(tree)
        module_dir = _module_dir(path)
        for call in _calls(tree, "StaticFiles"):
            if any(
                keyword.arg == "check_dir"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in call.keywords
            ):
                continue
            argument = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "directory"),
                call.args[0] if call.args else None,
            )
            directory = _resolve_dir(argument, module_dir, assigns)
            if directory is None or directory_exists(directory):
                continue
            findings.append(
                _finding(
                    path,
                    f"mounts StaticFiles at directory '{directory}', but no file in the "
                    "project creates it — the app raises at startup and never serves",
                )
            )
    return findings


# References a browser will request: enough coverage to catch a frontend
# pointing at files nobody wrote, small enough to never misread a template.
_HTML_REFS = (
    re.compile(r"<script\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"<link\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"<source\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
)
_CSS_REF = re.compile(r"url\(\s*[\"']?([^\"'()]+?)[\"']?\s*\)", re.IGNORECASE)
# A scheme (http:, https:, data:, mailto:…), protocol-relative //, or anchor.
_EXTERNAL_REF = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.IGNORECASE)


def _posix_dir(path: str) -> str:
    parent = str(PurePosixPath(path).parent)
    return "" if parent == "." else parent


def _normalize_reference(pieces: Iterable[str]) -> str | None:
    """Collapse ./ and ../ segments; None when the path escapes the project."""
    out: list[str] = []
    for piece in pieces:
        if piece in ("", "."):
            continue
        if piece == "..":
            if not out:
                return None
            out.pop()
        else:
            out.append(piece)
    return "/".join(out) or None


def _static_mounts(trees: dict[str, ast.Module]) -> list[tuple[str, str]]:
    """Every StaticFiles mount the project declares, as (url prefix, directory)."""
    mounts: list[tuple[str, str]] = []
    for path, tree in trees.items():
        assigns = _toplevel_assigns(tree)
        module_dir = _module_dir(path)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "mount"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            static_call = next(
                (
                    argument
                    for argument in node.args[1:]
                    if isinstance(argument, ast.Call)
                    and (
                        getattr(argument.func, "id", "") == "StaticFiles"
                        or getattr(argument.func, "attr", "") == "StaticFiles"
                    )
                ),
                None,
            )
            if static_call is None:
                continue
            argument = next(
                (kw.value for kw in static_call.keywords if kw.arg == "directory"),
                static_call.args[0] if static_call.args else None,
            )
            directory = _resolve_dir(argument, module_dir, assigns)
            if directory is None:
                continue
            prefix = node.args[0].value
            mounts.append((prefix if prefix.startswith("/") else f"/{prefix}", directory))
    return sorted(mounts, key=lambda mount: -len(mount[0]))


def _resolve_reference(
    raw: str, referrer: str, mounts: list[tuple[str, str]]
) -> tuple[str | None, bool]:
    """(project path a reference points at, whether the answer is authoritative).

    Authoritative means a miss is provable: the reference was relative to the
    referencing file, or an absolute path a parsed StaticFiles mount claims.
    An absolute path no mount claims may be a server-generated route, so it
    can pass by existing but never fail by being absent.
    """
    ref = raw.strip()
    if not ref or _EXTERNAL_REF.match(ref) or "{" in ref or "<" in ref:
        return None, False
    ref = ref.split("?", 1)[0].split("#", 1)[0]
    if not ref:
        return None, False
    if ref.startswith("/"):
        for prefix, directory in mounts:
            if ref == prefix or ref.startswith(prefix.rstrip("/") + "/"):
                remainder = ref[len(prefix) :].lstrip("/")
                pieces = [*directory.split("/"), *remainder.split("/")]
                return _normalize_reference(pieces), True
        return _normalize_reference(ref.split("/")), False
    pieces = [*_posix_dir(referrer).split("/"), *ref.split("/")]
    return _normalize_reference(pieces), True


def _asset_reference_findings(
    staged: dict[str, dict[str, Any]],
    texts: dict[str, str],
    all_paths: set[str],
    trees: dict[str, ast.Module],
) -> list[dict[str, str]]:
    """A staged frontend referencing a local asset nobody provides.

    The browser resolves these references at page load, so a miss is a broken
    page regardless of environment — provable from the changeset alone, like
    every other error in this gate. External URLs, data URLs, anchors and
    template expressions are not checkable text and never fire; an absolute
    path no mount claims may be a dynamically served route, so it is only
    consulted, never blamed.
    """
    mounts = _static_mounts(trees)
    findings: list[dict[str, str]] = []
    for path in sorted(p for p in staged if p.endswith((".html", ".htm", ".css"))):
        text = texts.get(path) or str(staged[path].get("content", ""))
        if path.endswith(".css"):
            references = [(m.start(1), m.group(1)) for m in _CSS_REF.finditer(text)]
        else:
            references = [
                (m.start(1), m.group(1))
                for pattern in _HTML_REFS
                for m in pattern.finditer(text)
            ]
        for offset, raw in sorted(references):
            target, authoritative = _resolve_reference(raw, path, mounts)
            if target is None or not authoritative or target in all_paths:
                continue
            line = text.count("\n", 0, offset) + 1
            findings.append(
                _finding(
                    path,
                    f"references '{raw.strip()}' on line {line}, but no file in "
                    f"this project provides {target} — the page loads broken",
                )
            )
    return findings


def _unwired_findings(
    staged: dict[str, dict[str, Any]], trees: dict[str, ast.Module]
) -> list[dict[str, str]]:
    """Things a build constructed and then never connected to the application.

    Scoped to turns that staged the application object itself. Without that, the
    wiring may well live in a file this turn never touched, and reporting it
    would be a guess rather than a fact.
    """
    staged_trees = {path: tree for path, tree in trees.items() if path in staged}
    if not any(_calls(tree, "FastAPI") for tree in staged_trees.values()):
        return []
    findings: list[dict[str, str]] = []
    mounted = any(_calls(tree, "mount") for tree in trees.values())
    included = any(_calls(tree, "include_router") for tree in trees.values())
    for path, tree in sorted(staged_trees.items()):
        if not mounted and _calls(tree, "StaticFiles"):
            findings.append(
                _finding(
                    path,
                    "builds StaticFiles but nothing in the project calls .mount() — "
                    "the static files will 404",
                )
            )
        if not included and _calls(tree, "APIRouter"):
            findings.append(
                _finding(
                    path,
                    "builds an APIRouter but nothing in the project calls "
                    "include_router() — those routes are unreachable",
                )
            )
    return findings
