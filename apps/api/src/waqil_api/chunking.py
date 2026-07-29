"""Structure-aware chunking for the personal corpus.

A fixed char window shreds functions mid-body and buries the symbol a query is
looking for, which is why naive code RAG retrieves poorly. Instead we split
Python along AST function/class boundaries and Markdown along headings, carrying
the symbol name and start line so a retrieved passage can cite `path::symbol`.
Everything else falls back to a line-aware character window. Pure and
dependency-free so it can be unit-tested without a model.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

# Extension -> language label. Also the allow-list of indexable text files.
TEXT_EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown", ".rst": "rst",
    ".txt": "text", ".text": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".env": "ini",
    ".css": "css", ".scss": "css", ".html": "html",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".sql": "sql", ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin", ".lua": "lua",
}


# Upper bound on the heading path prepended to a Markdown chunk.
_MAX_CONTEXT_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    text: str
    symbol: str | None
    start_line: int


def lang_for(suffix: str) -> str | None:
    """Language label for a file suffix, or None if it is not an indexable text type."""
    return TEXT_EXTENSIONS.get(suffix.lower())


def _window(
    text: str,
    start_line: int,
    symbol: str | None,
    max_chars: int,
    overlap: int,
    *,
    context: str = "",
) -> list[Chunk]:
    """Split `text` into overlapping character windows, tracking line numbers.

    `context` is prepended to every window. A passage halfway down a page has no
    idea which page it is on, so a query naming the document ("interactions with
    Acme") cannot match it; carrying the heading path into each window makes the
    whole document findable by name, not just its opening section."""
    text = text.strip("\n")
    if not text.strip():
        return []
    prefix = f"{context}\n\n" if context else ""
    # Never let a long heading path starve the passage it is supposed to label.
    budget = max(max_chars // 2, max_chars - len(prefix))
    if len(text) <= budget:
        return [Chunk(prefix + text, symbol, start_line)]
    step = max(1, budget - overlap)
    chunks: list[Chunk] = []
    index = 0
    while index < len(text):
        piece = text[index : index + budget]
        line = start_line + text.count("\n", 0, index)
        if piece.strip():
            chunks.append(Chunk(prefix + piece, symbol, line))
        index += step
    return chunks


def _chunk_python(text: str, max_chars: int, overlap: int) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _window(text, 1, None, max_chars, overlap)
    lines = text.splitlines()
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_start = 1

    def flush() -> None:
        nonlocal buffer
        block = "\n".join(buffer)
        if block.strip():
            chunks.extend(_window(block, buffer_start, None, max_chars, overlap))
        buffer = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush()
            decorators = [d.lineno for d in getattr(node, "decorator_list", [])]
            start = (min(decorators) if decorators else node.lineno) - 1
            end = getattr(node, "end_lineno", node.lineno)
            segment = "\n".join(lines[start:end])
            chunks.extend(_window(segment, start + 1, node.name, max_chars, overlap))
            buffer_start = end + 1
        else:
            if not buffer:
                buffer_start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            buffer.extend(lines[node.lineno - 1 : end])
    flush()
    return chunks or _window(text, 1, None, max_chars, overlap)


def _heading_level(line: str) -> int:
    """ATX heading depth (1-6), or 0 when the line is not a heading."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return 0
    level = len(stripped) - len(stripped.lstrip("#"))
    return level if level <= 6 else 0


def _breadcrumb(trail: list[tuple[int, str]]) -> str:
    """`Page title > Section > Subsection`, shortened from the middle when long
    so the document title — the part a query is most likely to name — survives."""
    names = [name for _, name in trail]
    if not names:
        return ""
    path = " > ".join(names)
    if len(path) > _MAX_CONTEXT_CHARS and len(names) > 2:
        path = f"{names[0]} > … > {names[-1]}"
    return path[:_MAX_CONTEXT_CHARS]


def _chunk_markdown(text: str, max_chars: int, overlap: int) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    trail: list[tuple[int, str]] = []
    heading: str | None = None
    context = ""
    section_start = 1
    section: list[str] = []

    def flush() -> None:
        block = "\n".join(section)
        if block.strip():
            chunks.extend(
                _window(
                    block, section_start, heading, max_chars, overlap, context=context
                )
            )

    for number, line in enumerate(lines, start=1):
        level = _heading_level(line)
        if level:
            flush()
            heading = line.strip("# ").strip()[:120] or None
            # Pop siblings and deeper headings so the trail holds only ancestors.
            while trail and trail[-1][0] >= level:
                trail.pop()
            if heading:
                trail.append((level, heading))
            context = _breadcrumb(trail)
            section_start = number
            section = [line]
        else:
            section.append(line)
    flush()
    return chunks or _window(text, 1, None, max_chars, overlap)


def chunk_text(text: str, lang: str | None, *, max_chars: int, overlap: int) -> list[Chunk]:
    """Chunk `text` by the best strategy for `lang`. Never returns empty-only chunks."""
    if not text.strip():
        return []
    if lang == "python":
        return _chunk_python(text, max_chars, overlap)
    if lang in ("markdown", "rst"):
        return _chunk_markdown(text, max_chars, overlap)
    return _window(text, 1, None, max_chars, overlap)
