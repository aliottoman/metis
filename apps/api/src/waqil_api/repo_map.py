"""A ranked, symbol-level map of a project, for the context of a build step.

The project agent used to be handed a flat list of paths and a prose note, and
everything else it wanted to know it had to *read*. That is the measured cause
of the dominant build failure: a model reading twenty-two files and writing
none, another re-reading one file forty-four times. It was not confused about
what to write; it was looking for what already exists.

This module answers that question up front. It extracts the definitions a file
declares, ranks files by how much the rest of the project depends on them —
biased toward whatever the request is actually about — and renders the top of
that ranking into a fixed character budget.

Three design commitments, each learned from something that went wrong before:

* **Every line is attributable.** A symbol is printed with its file and its line
  number, so the model can always go and check. A map that cannot be verified is
  worse than no map, because a confident wrong map is followed.
* **Truncation is stated, never silent.** The map ends by naming how much it did
  not show. "Covered everything" is a claim the renderer must earn.
* **Extraction is cheap and fail-soft.** Python goes through the exact ``ast``
  extractor the code graph already uses; the other languages are line-oriented
  and deliberately approximate. A file that cannot be parsed contributes
  nothing and never raises. The bar is "names, and where they live" — not a
  compiler.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from . import code_graph

# Files big enough to be generated rather than written. Scanning a 2 MB bundle
# costs real time and contributes nothing a person put there on purpose.
_MAX_SOURCE_BYTES = 400_000

# How many symbols any one file may contribute. A 4,000-line module would
# otherwise spend the entire budget on itself and crowd out the other forty
# files the model needs to know exist.
_MAX_SYMBOLS_PER_FILE = 24

# How many design tokens / selectors / anchors one file may spend lines on.
# They are worth naming and not worth enumerating: a stylesheet with 135 custom
# properties took the entire top of a real project's map.
_MAX_LEAF_SYMBOLS = 6

_PAGERANK_DAMPING = 0.85
_PAGERANK_ITERATIONS = 24

# How many files may declare a name before a reference to it stops being
# evidence about any of them. `run`, `get` and `value` are declared everywhere.
_MAX_NAME_OWNERS = 3

# Kinds that appear in the map but never attract rank. A method name is not a
# module-level identity, and an anchor or a colour token is not something other
# files "depend on" in any sense PageRank should reward.
_NON_ATTRACTING_KINDS = frozenset({"method", "anchor", "token", "selector"})

# Tokens in a request that could plausibly be an identifier. Three characters
# is the floor at which a match stops being noise ("app" is a signal; "an" is
# not), and the split keeps dotted/dashed/underscored spellings together.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL = re.compile(r"[a-z0-9](?=[A-Z])")


@dataclass(frozen=True)
class Symbol:
    """One definition, with the file and line that prove it exists."""

    path: str
    kind: str  # module | class | function | method | component | selector | token | anchor
    name: str
    line: int
    detail: str = ""  # the short signature-ish tail, when the language gives one

    def render(self) -> str:
        if self.kind in ("class", "component"):
            return f"class {self.name}{self.detail}"
        if self.kind in ("function", "method"):
            return f"def {self.name}{self.detail}"
        if self.kind == "type":
            return f"type {self.name}{self.detail}"
        return f"{self.name}{self.detail}"


@dataclass(frozen=True)
class FileFacts:
    """What one file declares, and what it names from elsewhere.

    ``references`` are *simple names* only. Resolving them properly would mean
    type inference; matching by name is ambiguous when two symbols share one,
    and that is an accepted trade for an extractor that costs nothing and can
    never be wrong about a file it did not read. The same trade the code graph
    already makes.
    """

    path: str
    symbols: tuple[Symbol, ...]
    references: tuple[str, ...]


# ── Extraction, one dialect at a time ──────────────────────────────────────


def _is_surface(node: code_graph.GraphNode, module_depth: int) -> bool:
    """Whether a definition is reachable from outside its own file.

    Top-level anything, and the methods and inner classes of a top-level class.
    Everything deeper is a closure — real code, but not a name another file can
    hold, which is all this map is for.
    """
    nesting = len(node.qualname.split(".")) - module_depth
    if nesting <= 1:
        return True
    return nesting == 2 and node.kind in ("method", "class")


def _python_facts(path: str, source: str) -> FileFacts:
    nodes, edges = code_graph.extract(source, path)
    depth = len(code_graph.module_qualname(path).split("."))
    symbols = tuple(
        Symbol(path, node.kind, node.name, node.start_line)
        for node in nodes
        # The module node names the file, which the map already prints as a
        # heading; repeating it as its own symbol wastes a line per file.
        if node.kind != "module"
        # A closure is not the file's surface. `on_token` defined inside a
        # function or a method is invisible to every other file, and listing it
        # spent map budget describing internals nobody can call. A METHOD sits
        # at the same nesting depth as a closure and is kept, which is why the
        # rule reads the kind rather than the depth alone.
        and _is_surface(node, depth)
    )
    references = tuple(
        edge.dst_name
        for edge in edges
        # An import names something specific. A *bare* call — `create_app()` —
        # names something in this module's namespace, so it too is real
        # evidence. An ATTRIBUTE call is not: `self.db.get(...)` says nothing
        # about which class was meant, and letting it through made the one
        # module with a top-level function called `get` collect a vote from
        # all forty-seven files that call `.get()` on anything at all.
        if edge.kind == "imports" or (edge.kind == "calls" and "." not in edge.dst_raw)
    )
    return FileFacts(path, symbols, references)


# One line-oriented pattern per declaration form TS/JS actually uses. A parser
# would be more correct and would cost a dependency plus a build step; these
# find the names, and every name carries the line that proves it.
#
# Every pattern is anchored to a TOP-LEVEL declaration — column zero, or an
# explicit `export`. Allowing indented ones was measured and was much worse
# than useless: it pulled in every local inside every function (`body`,
# `preview`, `router`, `value`), which both drowned the real declarations and
# corrupted the ranking, because a hundred files referencing `value` all voted
# for whichever component happened to declare a local by that name.
_TOP_LEVEL = r"^(?:export\s+(?:default\s+)?)?"
_TS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("class", re.compile(_TOP_LEVEL + r"(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
    ("function", re.compile(_TOP_LEVEL + r"(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)")),
    ("function", re.compile(_TOP_LEVEL + r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>")),
    ("type", re.compile(_TOP_LEVEL + r"(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)")),
    ("token", re.compile(_TOP_LEVEL + r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*[:=](?!.*=>)")),
)
_TS_IMPORT = re.compile(r"""^\s*import\s+(?:(.+?)\s+from\s+)?['"]([^'"]+)['"]""")
_TS_REQUIRE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")


def _typescript_facts(path: str, source: str) -> FileFacts:
    symbols: list[Symbol] = []
    references: list[str] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if len(line) > 400:  # a minified or generated line names nothing useful
            continue
        imported = _TS_IMPORT.match(line)
        if imported is not None:
            clause, module = imported.groups()
            references.append(_module_leaf(module))
            references.extend(_TOKEN.findall(clause or ""))
            continue
        for required in _TS_REQUIRE.findall(line):
            references.append(_module_leaf(required))
        for kind, pattern in _TS_PATTERNS:
            found = pattern.match(line)
            if found is not None:
                name = found.group(1)
                # A React component is a function whose name is capitalised;
                # saying so is worth a word, because "which file owns the
                # sidebar" is the question a UI task actually asks.
                if kind == "function" and name[:1].isupper():
                    kind = "component"
                symbols.append(Symbol(path, kind, name, number))
                break
    return FileFacts(path, tuple(symbols), tuple(references))


_HTML_ID = re.compile(r"""\bid\s*=\s*['"]([^'"]+)['"]""")
_HTML_REF = re.compile(r"""\b(?:src|href|action)\s*=\s*['"]([^'"#?][^'"]*)['"]""")


def _html_facts(path: str, source: str) -> FileFacts:
    symbols: list[Symbol] = []
    references: list[str] = []
    for number, line in enumerate(source.splitlines(), start=1):
        for anchor in _HTML_ID.findall(line):
            symbols.append(Symbol(path, "anchor", f"#{anchor}", number))
        for target in _HTML_REF.findall(line):
            references.append(_module_leaf(target))
    return FileFacts(path, tuple(symbols), tuple(references))


_CSS_SELECTOR = re.compile(r"^([.#]?[A-Za-z_][\w.#:>\-\s,\[\]=\"']*?)\s*\{")
_CSS_TOKEN = re.compile(r"^\s*(--[\w-]+)\s*:")


def _css_facts(path: str, source: str) -> FileFacts:
    symbols: list[Symbol] = []
    for number, line in enumerate(source.splitlines(), start=1):
        token = _CSS_TOKEN.match(line)
        if token is not None:
            symbols.append(Symbol(path, "token", token.group(1), number))
            continue
        selector = _CSS_SELECTOR.match(line)
        if selector is not None:
            text = " ".join(selector.group(1).split())
            if text:
                symbols.append(Symbol(path, "selector", text[:80], number))
    return FileFacts(path, tuple(symbols), ())


_EXTRACTORS = {
    ".py": _python_facts,
    ".pyi": _python_facts,
    ".ts": _typescript_facts,
    ".tsx": _typescript_facts,
    ".js": _typescript_facts,
    ".jsx": _typescript_facts,
    ".mjs": _typescript_facts,
    ".cjs": _typescript_facts,
    ".html": _html_facts,
    ".htm": _html_facts,
    ".css": _css_facts,
}


def _module_leaf(reference: str) -> str:
    """The last meaningful segment of an import path or asset URL.

    ``./lib/api`` -> ``api``; ``../styles/theme.css`` -> ``theme``. References
    are matched by simple name, so the leaf is the part that can match.
    """
    cleaned = reference.split("?")[0].split("#")[0].rstrip("/")
    leaf = cleaned.rsplit("/", 1)[-1]
    return leaf.rsplit(".", 1)[0] if "." in leaf else leaf


def _module_aliases(path: str) -> set[str]:
    """The names an import of this file could plausibly use.

    ``lib/model-control.tsx`` is imported as ``model-control``, but a request
    and a sibling may equally spell it ``model_control``; both are cheap to
    accept and neither can collide with anything else meaningful.
    """
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    if not stem or stem == "__init__":
        return set()
    lowered = stem.lower()
    return {lowered, lowered.replace("-", "_"), lowered.replace("_", "-")}


def extract(path: str, source: str) -> FileFacts:
    """Facts for one file. Never raises; an unknown suffix contributes nothing."""
    if len(source) > _MAX_SOURCE_BYTES:
        return FileFacts(path, (), ())
    suffix = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path else ""
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        return FileFacts(path, (), ())
    try:
        facts = extractor(path, source)
    except Exception:  # noqa: BLE001 - one odd file must never fail a build turn
        return FileFacts(path, (), ())
    # References are DISTINCT names, not every occurrence. "This file depends on
    # that one" is a single fact; counting each call site made one module that
    # happens to declare a frequently-called helper collect 1,335 votes and win
    # every ranking regardless of the request.
    return FileFacts(
        facts.path, facts.symbols, tuple(dict.fromkeys(facts.references))
    )


# ── Ranking ────────────────────────────────────────────────────────────────


def focus_terms(text: str) -> set[str]:
    """The identifiers a request could be about, lowercased.

    Both spellings of a camel-cased name are kept (``createApp`` also yields
    ``create`` and ``app``), because a user writes ``create app`` and the code
    says ``createApp``, and the map is worth nothing if those do not meet.
    """
    terms: set[str] = set()
    for token in _TOKEN.findall(text or ""):
        terms.add(token.lower())
        for part in _CAMEL.sub(lambda match: match.group(0) + " ", token).split():
            if len(part) >= 3:
                terms.add(part.lower())
        for part in token.split("_"):
            if len(part) >= 3:
                terms.add(part.lower())
    return terms


def _personalisation(facts: Sequence[FileFacts], terms: set[str]) -> dict[str, float]:
    """Seed weight per file, from how much the request names what it holds.

    This is the whole reason the map is worth ranking rather than listing: two
    requests against the same repository should not get the same map. With no
    terms it degrades to uniform, which is the ordinary "what is this project"
    map.
    """
    weights: dict[str, float] = {}
    for entry in facts:
        score = 1.0
        if terms:
            path_parts = {
                part.lower()
                for chunk in entry.path.replace("\\", "/").split("/")
                for part in re.split(r"[._-]", chunk)
                if len(part) >= 3
            }
            # A path the request names is the strongest signal there is — the
            # user said the filename — and it has to outweigh structural mass
            # by a wide margin or it is decorative. Measured: at a weight of 4
            # the file the request named did not reach the top ten.
            score += 25.0 * len(path_parts & terms)
            score += 6.0 * sum(
                1 for symbol in entry.symbols if symbol.name.lower().lstrip("#.") in terms
            )
        weights[entry.path] = score
    total = sum(weights.values()) or 1.0
    return {path: weight / total for path, weight in weights.items()}


def rank(facts: Sequence[FileFacts], terms: set[str] | None = None) -> dict[str, float]:
    """PageRank over "file A names something file B defines", personalised.

    Importance here is *structural*: a module everything imports matters even
    when nothing in the request mentions it, which is exactly the knowledge a
    model lacks on its first step and would otherwise buy with reads.
    """
    terms = terms or set()
    owners: dict[str, list[str]] = defaultdict(list)
    for entry in facts:
        # A file is named by what it declares AND by its own filename, because
        # `from . import repo_map` and `import x from "@/lib/api"` name the
        # MODULE, not a symbol inside it. Without this the strongest and
        # cleanest signal in the graph — the import — resolved to nothing, and
        # ranking was decided entirely by incidental name collisions.
        for alias in _module_aliases(entry.path):
            owners[alias].append(entry.path)
        for symbol in entry.symbols:
            # Methods are listed in the map but never attract rank: a call to
            # `.get()` or `.run()` says nothing about which class was meant, so
            # counting it hands importance to whoever used the most ordinary
            # verb. Measured — it put a tool registry above the module every
            # other module imports.
            if symbol.kind in _NON_ATTRACTING_KINDS:
                continue
            owners[symbol.name.lower()].append(entry.path)

    # A name many files declare identifies none of them. Matching by simple name
    # is the accepted approximation here, but an ambiguous name is not a weak
    # signal — it is noise, and left in it hands rank to whichever file happens
    # to declare the most ordinary words.
    ambiguous = {
        name for name, paths in owners.items() if len(set(paths)) > _MAX_NAME_OWNERS
    }

    out_edges: dict[str, dict[str, float]] = {entry.path: {} for entry in facts}
    for entry in facts:
        for reference in entry.references:
            if reference.lower() in ambiguous:
                continue
            for owner in owners.get(reference.lower(), ()):
                if owner == entry.path:
                    continue  # a file citing itself says nothing about importance
                out_edges[entry.path][owner] = out_edges[entry.path].get(owner, 0.0) + 1.0

    seed = _personalisation(facts, terms)
    scores = dict(seed)
    for _ in range(_PAGERANK_ITERATIONS):
        incoming: dict[str, float] = {path: 0.0 for path in scores}
        drained = 0.0
        for path, score in scores.items():
            targets = out_edges.get(path) or {}
            weight = sum(targets.values())
            if not weight:
                # A file that names nothing hands its rank back to the seed
                # rather than to everyone, so leaves cannot flatten the result.
                drained += score
                continue
            for target, count in targets.items():
                incoming[target] += score * (count / weight)
        scores = {
            path: (1 - _PAGERANK_DAMPING) * seed[path]
            + _PAGERANK_DAMPING * (incoming[path] + drained * seed[path])
            for path in scores
        }
    return scores


# ── Rendering ──────────────────────────────────────────────────────────────


def _trimmed(symbols: Sequence[Symbol], budget_chars: int) -> tuple[Symbol, ...]:
    """The symbols of one file that fit, in source order, public ones first.

    Order on the page stays by line number — a map you read top-to-bottom
    against the file is the point. But when the cap bites, a private helper is
    what gets dropped: `_as_text` is not what another file is going to call.
    """
    # Design tokens, selectors and anchors are worth *naming* — knowing a
    # stylesheet defines `--ink` is how a model reuses it instead of inventing
    # a colour — but they are one fact each, and a theme with 135 of them spent
    # the entire top of a real project's map on CSS variables. A handful plus an
    # honest count says the same thing in a fifth of the space.
    leaves = [item for item in symbols if item.kind in _NON_ATTRACTING_KINDS]
    if len(leaves) > _MAX_LEAF_SYMBOLS:
        allowed = {id(item) for item in leaves[:_MAX_LEAF_SYMBOLS]}
        symbols = [
            item
            for item in symbols
            if item.kind not in _NON_ATTRACTING_KINDS or id(item) in allowed
        ]
    keep = list(symbols[:_MAX_SYMBOLS_PER_FILE])
    if len(symbols) > _MAX_SYMBOLS_PER_FILE:
        public = [item for item in symbols if not item.name.startswith("_")]
        if len(public) >= _MAX_SYMBOLS_PER_FILE:
            keep = public[:_MAX_SYMBOLS_PER_FILE]
        elif public:
            private = [item for item in symbols if item.name.startswith("_")]
            keep = sorted(
                public + private[: _MAX_SYMBOLS_PER_FILE - len(public)],
                key=lambda item: item.line,
            )
    while keep and sum(len(item.render()) + 9 for item in keep) > budget_chars:
        keep.pop()
    return tuple(keep)


def render(
    facts: Sequence[FileFacts],
    scores: Mapping[str, float],
    *,
    max_chars: int,
) -> str:
    """The map as text, highest-ranked first, inside a character budget.

    Files that declare nothing extractable are not dropped — they are listed by
    path at the end, because "this project has a Dockerfile" is a fact worth one
    line and the alternative is a model reading the tree to find out.
    """
    if max_chars <= 0 or not facts:
        return ""
    named = [entry for entry in facts if entry.symbols]
    bare = sorted(entry.path for entry in facts if not entry.symbols)
    ordered = sorted(
        named, key=lambda entry: (-scores.get(entry.path, 0.0), entry.path)
    )

    lines: list[str] = []
    used = 0
    shown = 0
    hidden_symbols = 0
    # No one file may take more than a share of the budget. The top-ranked file
    # is usually the most-imported module in the project, which is genuinely
    # worth knowing about and is almost never what the request is about — left
    # uncapped it spent half a small budget on itself and pushed the file the
    # user actually named off the map.
    per_file_chars = max(240, max_chars // 3)
    for entry in ordered:
        symbols = _trimmed(entry.symbols, per_file_chars)
        dropped = len(entry.symbols) - len(symbols)
        block = [entry.path]
        block.extend(f"  {symbol.line:>5}  {symbol.render()}" for symbol in symbols)
        if dropped:
            block.append(f"        … {dropped} more in this file")
        text = "\n".join(block)
        # Always show at least one file: a budget too small for the top-ranked
        # file should still say what the top-ranked file is.
        if used + len(text) + 1 > max_chars and shown:
            hidden_symbols += len(entry.symbols)
            continue
        lines.append(text)
        used += len(text) + 1
        shown += 1

    hidden_files = len(ordered) - shown
    if hidden_files > 0:
        lines.append(
            f"… {hidden_files} more file{'' if hidden_files == 1 else 's'} with "
            f"{hidden_symbols} definitions not shown — read them if you need them."
        )
    if bare:
        room = max_chars - used
        listed = []
        for path in bare:
            if room - len(path) - 2 <= 0:
                break
            listed.append(path)
            room -= len(path) + 2
        if listed:
            tail = f"Other files: {', '.join(listed)}"
            if len(listed) < len(bare):
                tail += f", and {len(bare) - len(listed)} more"
            lines.append(tail)
    return "\n".join(lines)


def build(
    sources: Iterable[tuple[str, str]],
    *,
    request: str = "",
    max_chars: int = 6_000,
) -> str:
    """Extract, rank and render in one call — the whole map, from file contents."""
    facts = [extract(path, source) for path, source in sources]
    if not facts:
        return ""
    return render(facts, rank(facts, focus_terms(request)), max_chars=max_chars)
