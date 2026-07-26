"""P11 — the pure-python-authored AST safety profile (the code-authoring trust boundary).

Table-tests the denylist: real tool logic passes; every classic escape hatch
(imports of os/subprocess/socket, eval/exec/compile/open/__import__, getattr,
dunder attribute/name access, with/class/async/yield) is rejected. The host
profile is the primary control on what authored code may ever exist.
"""
from __future__ import annotations

import pytest

from waqil_api import capability_profiles
from waqil_api.authored_code import AuthoredCodeError, validate_authored_source

VALID = '''
import json
import re
from collections import Counter


def _words(text):
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 2]


def run(inputs, model):
    text = str(inputs.get("text", ""))
    counts = Counter(_words(text))
    top = [w for w, _ in counts.most_common(5)]
    hint = model({"prompt": "one-line gist", "text": text[:200]})
    result = {"top_words": top, "total": sum(counts.values()), "gist": hint}
    if not top:
        result["gist"] = "no content"
    return result
'''


def test_valid_authored_code_passes() -> None:
    evidence = validate_authored_source(VALID)
    assert evidence["defines_run"] is True
    assert "re" in evidence["imports"] and "json" in evidence["imports"]
    # And it passes through the named capability profile too.
    assert capability_profiles.validate("pure-python-authored-v1", VALID, {})["defines_run"]


def _wrap(body: str, params: str = "inputs, model") -> str:
    """Indent a run() body by 4 spaces. `body` lines must be given UN-indented."""
    return f"def run({params}):\n" + "\n".join("    " + line for line in body.splitlines())


REJECTED = {
    "import os": _wrap("import os\n    return {}"),
    "import sys": _wrap("import sys\n    return {}"),
    "import subprocess": _wrap("import subprocess\n    return {}"),
    "import socket": _wrap("import socket\n    return {}"),
    "urllib.request (network)": _wrap("from urllib.request import urlopen\n    return {}"),
    "import pathlib": _wrap("import pathlib\n    return {}"),
    "import importlib": _wrap("import importlib\n    return {}"),
    "eval": _wrap("return eval('1+1')"),
    "exec": _wrap("exec('x=1')\n    return {}"),
    "compile": _wrap("return compile('1', '<s>', 'eval')"),
    "__import__": _wrap("return __import__('os')"),
    "open": _wrap("return open('/etc/passwd').read()"),
    "getattr escape": _wrap("return getattr(inputs, 'foo')"),
    "setattr": _wrap("setattr(inputs, 'x', 1)\n    return {}"),
    "globals": _wrap("return globals()"),
    "input builtin (stdin)": _wrap("return input()"),
    "__class__ dunder attr": _wrap("return inputs.__class__"),
    "__subclasses__ chain": _wrap("return type.__subclasses__(object)"),
    "__builtins__ name": _wrap("return __builtins__"),
    "object subclasses": _wrap("return object.__subclasses__()"),
    "print (io)": _wrap("print('hi')\n    return {}"),
    "wildcard import": "from math import *\ndef run(inputs, model):\n    return {}",
    "with statement": _wrap("with inputs:\n        pass\n    return {}"),
    "class def": "class Sneaky:\n    pass\ndef run(inputs, model):\n    return {}",
    "async def": "async def run(inputs, model):\n    return {}",
    "yield generator": _wrap("yield 1"),
    "lambda getattr": _wrap("f = lambda o: getattr(o, 'x')\n    return f(inputs)"),
    "top-level side effect": "x = 1 + 1\ndef run(inputs, model):\n    return {}",
    "missing run": "def helper(x):\n    return x",
    "wrong params": _wrap("return {}", params="data"),
}


@pytest.mark.parametrize("name,source", list(REJECTED.items()), ids=list(REJECTED))
def test_escape_hatches_are_rejected(name: str, source: str) -> None:
    with pytest.raises(AuthoredCodeError):
        validate_authored_source(source)
    # And the named profile rejects it too (fails closed).
    with pytest.raises(capability_profiles.CodeProfileError):
        capability_profiles.validate("pure-python-authored-v1", source, {})


def test_oversized_source_is_rejected() -> None:
    big = _wrap("x = 1\n    " + "x = x + 1\n    " * 5000 + "return {}")
    with pytest.raises(AuthoredCodeError, match="exceeds"):
        validate_authored_source(big)


def test_null_and_crlf_rejected() -> None:
    with pytest.raises(AuthoredCodeError):
        validate_authored_source("def run(input, model):\r\n    return {}")
    with pytest.raises(AuthoredCodeError):
        validate_authored_source("def run(input, model):\n    return {}\x00")


def test_allowed_stdlib_and_control_flow_pass() -> None:
    source = _wrap(
        "import math\n"
        "from itertools import chain\n"
        "total = 0\n"
        "for i in range(int(inputs.get('n', 3))):\n"
        "    total += math.factorial(i)\n"
        "pairs = {k: v for k, v in enumerate(chain([1], [2]))}\n"
        "return {'total': total, 'pairs': pairs}"
    )
    evidence = validate_authored_source(source)
    assert "math" in evidence["imports"]
