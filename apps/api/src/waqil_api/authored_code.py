"""Model-authored tool code — the ``pure-python-authored`` safety profile.

This is the trust boundary for tools whose *implementation is written by a model*
for an unseen task. Unlike the diagram profiles (which allow only a tiny fixed
DSL), authored tools need real logic — control flow, comprehensions, a slice of
the standard library — so this validator is a **denylist over a broad allowlist**:

- The authored source must define exactly one top-level ``def run(input, model):``
  (helpers allowed). It never does I/O itself — a trusted harness injects the
  ``input`` dict and a ``model()`` callable and takes the returned value.
- Every AST node type must be in a broad allowed set (statements, control flow,
  comprehensions, arithmetic, calls); anything else — ``with``, ``class``,
  ``async``, ``yield``, ``match`` — is rejected.
- Imports are limited to a reviewed stdlib allowlist (no ``os``/``sys``/
  ``subprocess``/``socket``/network/filesystem).
- The classic escape hatches are banned outright: ``eval``/``exec``/``compile``/
  ``__import__``/``open``/``getattr``/``setattr`` and **any dunder attribute or
  name** (``__class__``/``__subclasses__``/``__globals__``/``__builtins__`` …).

The validator is host code, mirrored in-container for the Podman path, and fails
closed. It returns JSON-able evidence recorded with the build/run.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 20_000
MAX_AST_NODES = 4_000
RUN_FUNCTION = "run"
# Injected params. Not named `input`, so the builtin can stay fully banned.
REQUIRED_PARAMS = ("inputs", "model")

# Reviewed stdlib the authored code may import. Pure-compute + text only; nothing
# that can touch the network, filesystem, processes, or the interpreter internals.
_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "json",
        "re",
        "math",
        "statistics",
        "decimal",
        "fractions",
        "random",
        "collections",
        "collections.abc",
        "itertools",
        "functools",
        "operator",
        "datetime",
        "string",
        "textwrap",
        "unicodedata",
        "html",
        "difflib",
        "bisect",
        "heapq",
        "enum",
        "typing",
        "dataclasses",
        "numbers",
        "csv",
        "base64",
        "binascii",
        "hashlib",
        "hmac",
        "urllib.parse",  # parsing only — NOT urllib.request (network)
        "calendar",
        "zoneinfo",
    }
)

# Callable names that are never allowed, anywhere.
_BANNED_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "memoryview",
        "bytearray",  # arbitrary memory
        "breakpoint",
        "help",
        "exit",
        "quit",
        "copyright",
        "credits",
        "license",
        "__builtins__",
        "object",  # object().__subclasses__() escape
        "type",    # type(...) metaclass tricks / type().__subclasses__()
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "print",   # I/O — the harness owns stdio
    }
)

# The AST nodes authored code may use; anything else is an escape surface.
_ALLOWED_NODE_NAMES = (
    "Module Expression FunctionDef arguments arg Return Assign AugAssign AnnAssign "
    "For While If Try TryStar ExceptHandler Raise Assert Import ImportFrom alias Expr "
    "Pass Break Continue Global Nonlocal Delete "
    "BoolOp BinOp UnaryOp Lambda IfExp Dict Set ListComp SetComp DictComp GeneratorExp "
    "comprehension Compare Call keyword FormattedValue JoinedStr Constant Attribute "
    "Subscript Starred Name List Tuple Slice NamedExpr "
    "Load Store Del "
    "And Or Add Sub Mult MatMult Div Mod Pow LShift RShift BitOr BitXor BitAnd FloorDiv "
    "Invert Not UAdd USub Eq NotEq Lt LtE Gt GtE Is IsNot In NotIn"
)
_ALLOWED_NODE_TYPES: tuple[type, ...] = tuple(
    getattr(ast, name) for name in _ALLOWED_NODE_NAMES.split() if hasattr(ast, name)
)


class AuthoredCodeError(ValueError):
    """Model-authored source violates the pure-python-authored profile."""


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _import_allowed(dotted: str) -> bool:
    if dotted in _ALLOWED_IMPORTS:
        return True
    # Allow a submodule of an allowed package (e.g. collections.abc) only if its
    # top-level root is allowed AND the exact dotted path is allowed above.
    return False


class _Auditor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.imports: list[str] = []
        self.defines_run = False
        self.run_params: list[str] = []

    def _v(self, message: str, node: ast.AST) -> None:
        self.violations.append(f"{message} (line {getattr(node, 'lineno', '?')})")

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            self._v(f"disallowed syntax: {type(node).__name__}", node)
            return
        super().generic_visit(node)

    # ── module shape ────────────────────────────────────────────────────────
    def visit_Module(self, node: ast.Module) -> None:
        # No top-level side effects, so nothing runs at exec time.
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                if isinstance(stmt, ast.FunctionDef) and stmt.name == RUN_FUNCTION:
                    self.defines_run = True
                    self.run_params = [arg.arg for arg in stmt.args.args]
            else:
                self._v(
                    f"top-level {type(stmt).__name__} not allowed — only imports and "
                    "function definitions may appear at module level",
                    stmt,
                )
        self.generic_visit(node)

    # ── imports ─────────────────────────────────────────────────────────────
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _import_allowed(alias.name):
                self._v(f"import not allowed: {alias.name}", node)
            else:
                self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or not _import_allowed(module):
            self._v(f"import not allowed: from {module}", node)
        for alias in node.names:
            if alias.name == "*":
                self._v("wildcard import is not allowed", node)
            elif _is_dunder(alias.name):
                self._v(f"dunder import not allowed: {alias.name}", node)
        if _import_allowed(module):
            self.imports.append(module)
        self.generic_visit(node)

    # ── names / attributes ──────────────────────────────────────────────────
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BANNED_NAMES:
            self._v(f"use of banned name: {node.id}", node)
        elif _is_dunder(node.id):
            self._v(f"dunder name not allowed: {node.id}", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr):
            self._v(f"dunder attribute not allowed: .{node.attr}", node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_dunder(node.name):
            self._v(f"dunder function name not allowed: {node.name}", node)
        self.generic_visit(node)


def validate_authored_source(source: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate model-authored ``run(input, model)`` source against the
    pure-python-authored profile. Raises ``AuthoredCodeError`` on any violation
    and otherwise returns JSON-able evidence."""
    if "\x00" in source or "\r" in source:
        raise AuthoredCodeError("source must use LF line endings and contain no NUL bytes")
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise AuthoredCodeError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise AuthoredCodeError(f"source is not valid Python: {exc}") from exc

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        raise AuthoredCodeError(f"source exceeds {MAX_AST_NODES} AST nodes")

    auditor = _Auditor()
    auditor.visit(tree)
    if not auditor.defines_run:
        auditor.violations.append(f"missing a top-level `def {RUN_FUNCTION}(input, model)`")
    else:
        missing = [p for p in REQUIRED_PARAMS if p not in auditor.run_params]
        if missing:
            auditor.violations.append(
                f"`{RUN_FUNCTION}` must accept parameters {REQUIRED_PARAMS}; missing {missing}"
            )
    if auditor.violations:
        raise AuthoredCodeError("; ".join(sorted(set(auditor.violations))[:8]))
    return {
        "profile": "pure-python-authored-v1",
        "node_count": node_count,
        "imports": sorted(set(auditor.imports)),
        "defines_run": True,
    }


# Restricted executor. Every dangerous builtin is absent; the harness injects a
# guarded __import__. Exception classes stay so code can raise and except.
SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "abs", "all", "any", "ascii", "bin", "bool", "bytes", "chr", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "oct", "ord", "pow", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "zip", "True", "False", "None", "NotImplemented", "Ellipsis",
        # exception hierarchy (so try/except/raise work)
        "Exception", "ArithmeticError", "AssertionError", "AttributeError",
        "EOFError", "FloatingPointError", "IndexError", "KeyError", "LookupError",
        "MemoryError", "NameError", "NotImplementedError", "OverflowError",
        "RecursionError", "RuntimeError", "StopIteration", "TypeError",
        "ValueError", "ZeroDivisionError", "UnicodeError", "UnicodeDecodeError",
        "UnicodeEncodeError",
    }
)

_HARNESS_PATH = Path(__file__).resolve().parent / "authored_harness.py"


class AuthoredExecutionError(RuntimeError):
    """A model-authored tool failed to run (crash, timeout, or bad output)."""


def _rlimits(timeout_seconds: int, memory_mb: int) -> dict[str, int]:
    limits = {"RLIMIT_CPU": max(1, int(timeout_seconds)), "RLIMIT_FSIZE": 0}
    if memory_mb:
        limits["RLIMIT_AS"] = int(memory_mb) * 1024 * 1024
    return limits


async def execute_authored(
    source: str,
    inputs: dict[str, Any],
    *,
    on_model_request: Callable[[dict[str, Any]], Awaitable[str]] | None = None,
    timeout_seconds: int = 10,
    memory_mb: int = 512,
    max_frame_bytes: int = 1_000_000,
    model_call_timeout_seconds: int = 0,
    model_call_budget: int = 0,
) -> dict[str, Any]:
    """Run AST-gated ``run(inputs, model)`` in a restricted host subprocess.

    ``model()`` calls inside the tool are bridged to ``on_model_request`` (which
    the caller wires to the budget-enforced Model Broker). Raises
    ``AuthoredExecutionError`` on any crash, timeout, non-object output, or if the
    source fails the profile.

    The subprocess is driven with a plain ``Popen`` on a worker thread and the
    broker call is hopped back to the running loop — deliberately avoiding
    asyncio's subprocess child-watcher, which is unreliable on non-main-thread
    event loops (e.g. a server's request-portal). This is the same frame protocol
    the production Podman host-wrapper speaks.
    """
    validate_authored_source(source)  # gate before executing (defense in depth)
    loop = asyncio.get_running_loop()
    workdir = Path(tempfile.mkdtemp(prefix="metis-authored-"))
    bootstrap = workdir / "bootstrap.json"
    bootstrap.write_text(
        json.dumps(
            {
                "source": source,
                "inputs": inputs,
                "allowed_imports": sorted(_ALLOWED_IMPORTS),
                "safe_builtins": sorted(SAFE_BUILTINS),
                "rlimits": _rlimits(timeout_seconds, memory_mb),
            }
        ),
        encoding="utf-8",
    )

    # Time spent blocked on the host's own model call is not untrusted compute, so
    # it gets its own allowance instead of eating the code budget. RLIMIT_CPU still
    # pins actual CPU burn to `timeout_seconds`, which is the runaway-code guard;
    # a tool that legitimately calls a model just needs wall-clock to wait.
    model_wait = max(0, int(model_call_timeout_seconds)) * max(0, int(model_call_budget))
    wall_clock_budget = timeout_seconds + model_wait

    def bridge(params: dict[str, Any]) -> str:
        # Runs on the driver thread; hop to the loop for the async broker call.
        if on_model_request is None:
            raise RuntimeError("this tool has no model access")
        future = asyncio.run_coroutine_threadsafe(on_model_request(params), loop)
        return future.result(timeout=model_call_timeout_seconds or timeout_seconds)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _drive_sync, bootstrap, workdir, bridge, wall_clock_budget, max_frame_bytes
            ),
            timeout=wall_clock_budget + 12,
        )
    except asyncio.TimeoutError as exc:
        raise AuthoredExecutionError("authored tool exceeded its time budget") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def _drive_sync(
    bootstrap: Path,
    workdir: Path,
    bridge: Callable[[dict[str, Any]], str],
    timeout_seconds: int,
    max_frame_bytes: int,
) -> dict[str, Any]:
    proc = subprocess.Popen(
        [sys.executable, "-I", str(_HARNESS_PATH), str(bootstrap)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", "")},
        cwd=str(workdir),
        start_new_session=True,
    )
    # Hard wall-clock backstop: a hung/CPU-bound child is force-killed, so
    # readline() below always returns (EOF) instead of blocking forever.
    watchdog = threading.Timer(timeout_seconds + 8, _kill_group, args=(proc,))
    watchdog.daemon = True
    watchdog.start()
    try:
        assert proc.stdout is not None and proc.stdin is not None and proc.stderr is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                stderr = proc.stderr.read()[:1000].decode("utf-8", "replace").strip()
                raise AuthoredExecutionError(
                    f"authored tool exited without a result. {stderr}".strip()
                )
            if len(line) > max_frame_bytes:
                raise AuthoredExecutionError("authored tool emitted an oversized frame")
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore any stray non-frame line
            kind = frame.get("frame")
            if kind == "model_request":
                response: dict[str, Any] = {"frame": "model_response"}
                try:
                    content = bridge(frame.get("params", {}))
                    # A bridge that answers with anything but text is a host bug,
                    # not tool output — never hand it to the tool as a reply.
                    if not isinstance(content, str):
                        raise TypeError("model bridge returned a non-text reply")
                    response["content"] = content
                except Exception as error:  # noqa: BLE001 — broker/budget → typed error frame
                    # `str(TimeoutError())` is empty, and an empty message reads
                    # as "no error" on the far side, so a timed-out model call
                    # silently became an empty successful reply. Always send a
                    # non-empty reason: an error must never look like success.
                    response["error"] = str(error)[:200] or type(error).__name__
                try:
                    proc.stdin.write((json.dumps(response) + "\n").encode("utf-8"))
                    proc.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise AuthoredExecutionError("authored tool closed the model channel") from exc
            elif kind == "result":
                output = frame.get("output")
                if not isinstance(output, dict):
                    raise AuthoredExecutionError("authored tool returned a non-object output")
                return output
            elif kind == "error":
                raise AuthoredExecutionError(str(frame.get("error", "authored tool failed")))
    finally:
        watchdog.cancel()
        _kill_group(proc)
