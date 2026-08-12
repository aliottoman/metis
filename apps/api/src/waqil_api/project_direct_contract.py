"""The contract a `cline_direct` run is admitted under, and how it is resolved.

The direct path does not ask a model to enumerate files before work starts --
that was the manifest problem, and enumerating the existing tree instead would
be the same mistake wearing a different hat: it would forbid the model from
creating a file that did not exist yet.

So the run is admitted under a *contract* rather than a file list. It says
where writing is allowed (normally the whole project root), what may never
change, and what the frozen bytes of those protected files were at admission.
Everything else is discovered afterwards from the independent mirror diff.

Protections come from two places and are unioned:

* persistent per-project protections a person set in the UI, and
* task protections resolved deterministically from explicit wording in the
  request itself.

No model decides what is protected. The resolver below reads the request, and
where it cannot be certain it refuses to guess: an explicit protection naming
something that matches nothing, or matching so much that including all of it
would be a different instruction, stops the run before any inference and asks.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


# Explicit protection wording. Deliberately a closed list of unambiguous
# imperatives: "be careful with x" is not a protection, and guessing that it
# might be is exactly the judgment this module refuses to make.
PROTECTION_PHRASES: tuple[str, ...] = (
    "do not touch",
    "don't touch",
    "dont touch",
    "do not modify",
    "don't modify",
    "dont modify",
    "do not change",
    "don't change",
    "dont change",
    "do not edit",
    "don't edit",
    "dont edit",
    "do not alter",
    "don't alter",
    "must not change",
    "must not be changed",
    "must not be modified",
    "preserve unchanged",
    "leave unchanged",
    "leave untouched",
    "keep unchanged",
)

# Stripped before a term is read as a reference.
_ARTICLES = frozenset(
    {"the", "a", "an", "any", "all", "my", "our", "its", "their", "this", "that"}
)
# A term longer than this many content words is prose, not a file reference.
_MAX_TERM_WORDS = 3
# A bare-word term (no extension, slash or glob) matching more than this many
# files is ambiguous: including all of them may not be what was meant.
_MAX_BARE_WORD_MATCHES = 6

_TERM_SPLIT = re.compile(r",|\bor\b|\band\b|;", re.IGNORECASE)
# A sentence-ending dot is followed by whitespace or the end of the text; an
# extension dot is followed immediately by a character. Without that
# distinction "dont touch extractor.py, excel_writer.py" ended at the first
# filename and protected one file out of four.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_GLOB_CHARS = frozenset("*?[")

# Framework-owned and control paths a direct session may never write, whatever
# the contract says. These are host property, not the project's.
FRAMEWORK_PREFIXES: tuple[str, ...] = ("appkit/", ".metis/", ".git/")
FRAMEWORK_PATHS: frozenset[str] = frozenset({"appkit", ".metis", ".git"})
# Suffixes whose bytes a source-level verifier cannot meaningfully check, and
# which a coding session has no business rewriting.
BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".xlsx",
        ".xls",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".zip",
        ".gz",
        ".tar",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".so",
        ".dylib",
        ".dll",
        ".pyc",
        ".db",
        ".sqlite3",
    }
)


def is_framework_path(path: str) -> bool:
    """Host-owned: supplied by Metis, never the session's to change."""

    # NOT lstrip("./"): that strips characters, not a prefix, and turned
    # ".git/config" into "git/config" -- unprotecting the control directory.
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized in FRAMEWORK_PATHS or any(
        normalized.startswith(prefix) for prefix in FRAMEWORK_PREFIXES
    )


def is_binary_path(path: str) -> bool:
    return PurePosixPath(str(path)).suffix.casefold() in BINARY_SUFFIXES


@dataclass(frozen=True, slots=True)
class ProtectionResolution:
    """What the request's explicit protection wording resolved to."""

    resolved: tuple[str, ...] = ()
    # Terms that named something concrete and matched nothing in the tree.
    unmatched: tuple[str, ...] = ()
    # Terms matching so broadly that including everything may not be meant.
    ambiguous: tuple[str, ...] = ()

    @property
    def needs_user(self) -> bool:
        """Whether the run must stop and ask before any inference."""

        return bool(self.unmatched or self.ambiguous)

    def question(self) -> str:
        """The exact question to put to the user, or ''."""

        parts: list[str] = []
        if self.unmatched:
            joined = ", ".join(f"“{item}”" for item in self.unmatched)
            parts.append(
                f"You asked me not to change {joined}, but I cannot find "
                "anything by that name in this project."
            )
        if self.ambiguous:
            joined = ", ".join(f"“{item}”" for item in self.ambiguous)
            parts.append(
                f"You asked me not to change {joined}, and that matches many "
                "files — I do not want to guess which you meant."
            )
        if not parts:
            return ""
        return (
            " ".join(parts)
            + " Tell me the exact paths to protect and I will start, or say to "
            "continue without protecting them."
        )


def _clause_after(text: str, phrase: str) -> str:
    """The stretch of text a protection phrase governs."""

    lowered = text.casefold()
    marker = phrase.casefold()
    start = lowered.find(marker)
    if start < 0:
        return ""
    rest = text[start + len(marker) :]
    ended = _SENTENCE_END.search(rest)
    return rest[: ended.start()] if ended else rest


def _content_words(term: str) -> list[str]:
    return [
        word
        for word in _WORD.findall(term)
        if word.casefold() not in _ARTICLES and len(word) > 1
    ]


def _looks_concrete(term: str) -> bool:
    """Whether a term names a path rather than describing one in prose."""

    stripped = term.strip()
    return (
        "/" in stripped
        or any(character in _GLOB_CHARS for character in stripped)
        or bool(re.search(r"[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,6}\b", stripped))
    )


def _match_concrete(term: str, tree: Sequence[str]) -> list[str]:
    """Match a path, filename or glob against the project tree."""

    needle = term.strip().strip("`'\"")
    while needle.startswith("./"):
        needle = needle[2:]
    if not needle:
        return []
    matches: list[str] = []
    for path in tree:
        name = PurePosixPath(path).name
        if (
            path == needle
            or name == needle
            or fnmatch.fnmatch(path, needle)
            or fnmatch.fnmatch(name, needle)
        ):
            matches.append(path)
    return sorted(set(matches))


def _match_words(term: str, tree: Sequence[str]) -> list[str]:
    """Match a short bare-word term against filenames, all words required.

    "the DHL template" protects ``assets/DHL_Template.xlsx`` because that
    filename contains both words. Prose like "the extraction" matches nothing
    and is discarded rather than reported: a sentence continuing past the file
    list is not a failed protection.
    """

    words = [word.casefold() for word in _content_words(term)]
    if not words or len(words) > _MAX_TERM_WORDS:
        return []
    matches = [
        path
        for path in tree
        if all(word in PurePosixPath(path).name.casefold() for word in words)
    ]
    return sorted(set(matches))


def resolve_task_protections(prompt: str, tree: Iterable[str]) -> ProtectionResolution:
    """Resolve explicit protection wording against the current project tree.

    Deterministic by construction: phrase matching, path/glob matching and
    filename word matching, with no model in the loop. A term that names
    something concrete and matches nothing is an error worth stopping for; a
    term that names nothing concrete and matches nothing is ordinary prose and
    is ignored.
    """

    paths = sorted({str(item).replace("\\", "/") for item in tree if str(item)})
    resolved: set[str] = set()
    unmatched: list[str] = []
    ambiguous: list[str] = []
    seen_terms: set[str] = set()

    for phrase in PROTECTION_PHRASES:
        clause = _clause_after(prompt or "", phrase)
        if not clause:
            continue
        for raw in _TERM_SPLIT.split(clause):
            term = raw.strip().strip("`'\"")
            if not term or term.casefold() in seen_terms:
                continue
            seen_terms.add(term.casefold())
            concrete = _looks_concrete(term)
            matches = (
                _match_concrete(term, paths) if concrete else _match_words(term, paths)
            )
            if matches:
                if not concrete and len(matches) > _MAX_BARE_WORD_MATCHES:
                    ambiguous.append(term)
                    continue
                resolved.update(matches)
            elif concrete:
                # It named a file, and there is no such file. Ask.
                unmatched.append(term)
    return ProtectionResolution(
        resolved=tuple(sorted(resolved)),
        unmatched=tuple(dict.fromkeys(unmatched)),
        ambiguous=tuple(dict.fromkeys(ambiguous)),
    )


@dataclass(frozen=True, slots=True)
class DirectBuildContract:
    """What one direct run is admitted under. Frozen at admission."""

    writable_roots: tuple[str, ...] = (".",)
    task_protected: tuple[str, ...] = ()
    project_protected: tuple[str, ...] = ()
    protected_hashes: Mapping[str, str] = field(default_factory=dict)
    acceptance: tuple[Mapping[str, Any], ...] = ()
    max_iterations: int = 0
    max_rounds: int = 0
    check_budget: int = 0
    approval_required: bool = True

    @property
    def protected_files(self) -> tuple[str, ...]:
        """Task and persistent protections, unioned."""

        return tuple(sorted(set(self.task_protected) | set(self.project_protected)))

    def as_state(self) -> dict[str, Any]:
        """The shape carried in the run checkpoint."""

        return {
            "writable_roots": list(self.writable_roots),
            "task_protected": list(self.task_protected),
            "project_protected": list(self.project_protected),
            "protected_files": list(self.protected_files),
            "protected_hashes": dict(self.protected_hashes),
            "acceptance": [dict(item) for item in self.acceptance],
            "max_iterations": self.max_iterations,
            "max_rounds": self.max_rounds,
            "check_budget": self.check_budget,
            "approval_required": self.approval_required,
            "authorized_scope": authorized_scope_text(self),
        }


def authorized_scope_text(contract: DirectBuildContract) -> str:
    roots = ", ".join(contract.writable_roots) or "."
    where = (
        "Any file in this project"
        if roots == "."
        else f"Any file under {roots} in this project"
    )
    return (
        f"{where}, including files that do not exist yet — create them where "
        "they belong. You may not change the protected files listed above, "
        "appkit/, .git, .metis, or any secret-bearing file. Metis rejects a "
        "change to any of them independently of this instruction."
    )


def freeze_protected_hashes(project: Path, protected: Sequence[str]) -> dict[str, str]:
    """SHA-256 of every protected file that exists, taken at admission.

    Recomputed per run on purpose: a persistent setting stores identities, not
    bytes, so a file edited on disk between runs cannot inherit a stale hash
    and pass a drift check it should fail.
    """

    frozen: dict[str, str] = {}
    for relative in protected:
        candidate = project / relative
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            frozen[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
    return frozen


def protected_drift(project: Path, frozen: Mapping[str, str]) -> list[str]:
    """Protected files whose bytes no longer match the frozen contract."""

    drifted: list[str] = []
    for relative, digest in sorted(frozen.items()):
        candidate = project / relative
        try:
            if candidate.is_symlink() or not candidate.is_file():
                drifted.append(relative)
                continue
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                drifted.append(relative)
        except OSError:
            drifted.append(relative)
    return drifted


def build_direct_contract(
    *,
    prompt: str,
    project: Path,
    tree: Iterable[str],
    project_protected: Sequence[str] = (),
    acceptance: Sequence[Mapping[str, Any]] = (),
    max_iterations: int = 0,
    max_rounds: int = 0,
    check_budget: int = 0,
    writable_roots: Sequence[str] = (".",),
) -> tuple[DirectBuildContract, ProtectionResolution]:
    """Resolve, union and freeze one run's contract.

    Returns the contract and the resolution, so a caller can stop and ask when
    an explicit protection could not be resolved safely.
    """

    resolution = resolve_task_protections(prompt, tree)
    persistent = tuple(
        sorted({str(item).replace("\\", "/") for item in project_protected if item})
    )
    contract = DirectBuildContract(
        writable_roots=tuple(writable_roots or (".",)),
        task_protected=resolution.resolved,
        project_protected=persistent,
        acceptance=tuple(dict(item) for item in acceptance),
        max_iterations=max_iterations,
        max_rounds=max_rounds,
        check_budget=check_budget,
    )
    frozen = freeze_protected_hashes(project, contract.protected_files)
    return (
        DirectBuildContract(
            writable_roots=contract.writable_roots,
            task_protected=contract.task_protected,
            project_protected=contract.project_protected,
            protected_hashes=frozen,
            acceptance=contract.acceptance,
            max_iterations=contract.max_iterations,
            max_rounds=contract.max_rounds,
            check_budget=contract.check_budget,
        ),
        resolution,
    )
