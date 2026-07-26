#!/usr/bin/env python3
"""Untrusted-code harness — runs a model-authored ``run(inputs, model)`` function
in a locked-down namespace and bridges its ``model()`` calls to the host over a
newline-delimited stdio frame protocol.

Standalone (imports only stdlib, no project code) so it runs under ``python -I``.
The trusted host (``authored_code.execute_authored``) launches it with a bootstrap
file path as argv[1]; everything the authored code can touch is the restricted
builtins + injected ``inputs``/``model``. This is the same frame protocol the
Podman host-wrapper will speak, so the sandbox path reuses it unchanged.

Frames (one JSON object per line):
  child → host : {"frame":"model_request","params":{...}}
  host → child : {"frame":"model_response","content":"..."}  | {"frame":...,"error":"..."}
  child → host : {"frame":"result","output":{...}}  | {"frame":"error","error":"..."}
"""
import builtins as _builtins
import json
import sys

_REAL_IMPORT = _builtins.__import__


def _frame_out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _model(params):
    """The injected bridge: send a model_request frame, block for the response."""
    if not isinstance(params, dict):
        raise TypeError("model(params) requires a dict of template parameters")
    _frame_out({"frame": "model_request", "params": params})
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("model channel closed by host")
    message = json.loads(line)
    if message.get("error"):
        raise RuntimeError(f"model call rejected: {message['error']}")
    return message.get("content", "")


def _make_safe_import(allowed: set) -> callable:
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level != 0 or (name not in allowed and root not in allowed):
            raise ImportError(f"import of {name!r} is not permitted in an authored tool")
        return _REAL_IMPORT(name, globals, locals, fromlist, level)

    return _safe_import


def _apply_rlimits(rlimits: dict) -> None:
    try:
        import resource
    except Exception:
        return
    for name, soft in rlimits.items():
        try:
            resource.setrlimit(getattr(resource, name), (int(soft), int(soft)))
        except Exception:
            # Best-effort: some limits (RLIMIT_AS) are unreliable on macOS. The
            # host's wall-clock timeout and the AST ban on network/process imports
            # are the load-bearing guards.
            pass


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as handle:
        bootstrap = json.load(handle)
    _apply_rlimits(bootstrap.get("rlimits", {}))

    allowed = set(bootstrap["allowed_imports"])
    safe_builtins = {
        name: getattr(_builtins, name)
        for name in bootstrap["safe_builtins"]
        if hasattr(_builtins, name)
    }
    safe_builtins["__import__"] = _make_safe_import(allowed)
    namespace = {"__builtins__": safe_builtins, "__name__": "authored_tool"}

    try:
        exec(compile(bootstrap["source"], "<authored_tool>", "exec"), namespace)
        run = namespace.get("run")
        if not callable(run):
            raise RuntimeError("authored source defines no run()")
        output = run(bootstrap["inputs"], _model)
        json.dumps(output)  # ensure serializable before framing
        _frame_out({"frame": "result", "output": output})
    except Exception as exc:  # noqa: BLE001 — report, never leak a traceback
        _frame_out({"frame": "error", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})


if __name__ == "__main__":
    main()
