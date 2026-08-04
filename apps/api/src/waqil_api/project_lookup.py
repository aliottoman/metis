"""Let a build read the real signature of an installed package.

Every time this session found a build at fault it was really the harness: a
model could not check anything, so it guessed, and a confident guess is
indistinguishable from knowledge. Both a frontier model and a local one wrote
`AsyncOpenAI(auth=...)` — a parameter that has never existed — and one invented
`load_client_config` in a package that does not export it. Neither could have
discovered otherwise, because the project tools are sealed to the project.

This is the one authorised door. It reads *installed* modules only: what a
module exports, and what one of its functions or classes actually takes. The
project's own files stay out of scope — they are what read_file is for, and
importing them would execute model-authored code in the host process.

Introspection runs in a subprocess. Importing any module runs its top-level
code, and a package that hangs or crashes on import must cost one tool call
rather than the API.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Long enough for a heavy SDK's import, short enough that a hang is a tool
# error rather than a stalled turn.
_TIMEOUT_SECONDS = 20.0
_MAX_EXPORTS = 200
_MAX_DOC_CHARS = 800

# Run inside the child: import, describe, print JSON. Nothing is written and
# nothing is returned to the caller but text.
_PROBE = r"""
import importlib, inspect, json, sys

module_name, symbol = sys.argv[1], (sys.argv[2] or None)
out = {"module": module_name, "importable": False}
try:
    module = importlib.import_module(module_name)
except BaseException as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out)); raise SystemExit(0)

out["importable"] = True
out["file"] = getattr(module, "__file__", "") or ""
out["version"] = str(getattr(module, "__version__", "") or "")
exported = list(getattr(module, "__all__", []) or [])
if not exported:
    exported = [name for name in dir(module) if not name.startswith("_")]
out["exports"] = sorted(exported)[:%(max_exports)d]

if symbol:
    if not hasattr(module, symbol):
        out["has_symbol"] = False
    else:
        target = getattr(module, symbol)
        out["has_symbol"] = True
        out["kind"] = type(target).__name__
        try:
            subject = target.__init__ if inspect.isclass(target) else target
            out["signature"] = f"{symbol}{inspect.signature(subject)}"
        except (TypeError, ValueError) as exc:
            out["signature_error"] = str(exc)
        doc = inspect.getdoc(target) or ""
        out["doc"] = doc[:%(max_doc)d]
        if inspect.isclass(target):
            out["members"] = sorted(
                name for name in dir(target) if not name.startswith("_")
            )[:%(max_exports)d]
print(json.dumps(out))
""" % {"max_exports": _MAX_EXPORTS, "max_doc": _MAX_DOC_CHARS}


class LookupError_(RuntimeError):
    """The lookup was refused or could not be performed."""


def _is_project_module(module: str, project_roots: tuple[Path, ...]) -> bool:
    """Whether this name would resolve to a file inside a granted project.

    Checked before the subprocess runs: importing a project module executes
    code the model just wrote, which is precisely what the sandbox exists to
    contain, and read_file already covers that need without executing anything.
    """
    head = module.split(".", 1)[0]
    for root in project_roots:
        try:
            candidates = [root / f"{head}.py", root / head / "__init__.py"]
        except (OSError, ValueError):
            continue
        if any(candidate.exists() for candidate in candidates):
            return True
    return False


async def inspect_installed_api(
    module: str, symbol: str = "", *, project_roots: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """What an installed module exports, and what one of its symbols takes."""
    module = (module or "").strip()
    symbol = (symbol or "").strip()
    if not module or not all(part.isidentifier() for part in module.split(".")):
        raise LookupError_(
            f"{module!r} is not a module name. Send a dotted import path, "
            'e.g. "openai" or "oci_genai_auth".'
        )
    if symbol and not symbol.isidentifier():
        raise LookupError_(f"{symbol!r} is not a symbol name.")
    if _is_project_module(module, project_roots):
        raise LookupError_(
            f"{module} is a file in this project, not an installed package. "
            "Use read_file for the project's own code; inspect_api is for "
            "libraries you did not write."
        )

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _PROBE,
        module,
        symbol,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        stdin=asyncio.subprocess.DEVNULL,
        # A project on sys.path would let a crafted name import project code.
        cwd=os.path.dirname(sys.executable) or "/",
        env={**os.environ, "PYTHONPATH": "", "PYTHONSTARTUP": ""},
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise LookupError_(f"importing {module} took too long and was stopped") from None

    try:
        result: dict[str, Any] = json.loads(stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError:
        raise LookupError_(f"{module} could not be inspected") from None
    if not result:
        raise LookupError_(f"{module} could not be inspected")
    if not result.get("importable"):
        raise LookupError_(
            f"{module} is not installed here ({result.get('error', 'import failed')}). "
            "Declare it in requirements and write against its documented API, or "
            "choose a package that is available."
        )
    if symbol and result.get("has_symbol") is False:
        result["hint"] = (
            f"{module} does not define {symbol}. Its exported names are listed "
            "in `exports` — use one of those."
        )
    return result
