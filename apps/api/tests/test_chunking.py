from __future__ import annotations

from waqil_api.chunking import chunk_text, lang_for


def test_lang_for_maps_and_rejects() -> None:
    assert lang_for(".py") == "python"
    assert lang_for(".MD") == "markdown"
    assert lang_for(".tsx") == "typescript"
    assert lang_for(".bin") is None


def test_python_splits_by_symbol_with_lines() -> None:
    source = (
        "import os\n"
        "\n"
        "def alpha():\n"
        "    return os.getpid()\n"
        "\n"
        "class Beta:\n"
        "    def method(self):\n"
        "        return 1\n"
    )
    chunks = chunk_text(source, "python", max_chars=1000, overlap=50)
    symbols = [chunk.symbol for chunk in chunks]
    assert "alpha" in symbols
    assert "Beta" in symbols
    # The module preamble (imports/constants) is kept as a symbol-less chunk.
    assert any(chunk.symbol is None and "import os" in chunk.text for chunk in chunks)
    alpha = next(chunk for chunk in chunks if chunk.symbol == "alpha")
    assert "return os.getpid()" in alpha.text
    assert alpha.start_line == 3


def test_python_decorated_function_includes_decorator() -> None:
    source = "import functools\n\n@functools.cache\ndef cached():\n    return 2\n"
    chunks = chunk_text(source, "python", max_chars=1000, overlap=50)
    cached = next(chunk for chunk in chunks if chunk.symbol == "cached")
    assert "@functools.cache" in cached.text


def test_python_syntax_error_falls_back_to_window() -> None:
    chunks = chunk_text("def broken(:\n", "python", max_chars=1000, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].symbol is None


def test_markdown_splits_by_heading() -> None:
    markdown = "# Intro\nhello there\n\n## Billing\nStripe handles invoices\n"
    chunks = chunk_text(markdown, "markdown", max_chars=1000, overlap=50)
    headings = [chunk.symbol for chunk in chunks]
    assert "Intro" in headings
    assert "Billing" in headings
    billing = next(chunk for chunk in chunks if chunk.symbol == "Billing")
    assert "Stripe handles invoices" in billing.text


def test_char_window_overlaps_and_tracks_lines() -> None:
    text = "\n".join(f"line{i}" for i in range(100))
    chunks = chunk_text(text, "text", max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert all(chunk.symbol is None for chunk in chunks)
    # Later windows start on a later line than the first.
    assert chunks[1].start_line > chunks[0].start_line


def test_blank_text_yields_no_chunks() -> None:
    assert chunk_text("   \n  \n", "python", max_chars=100, overlap=10) == []
