"""The document factory: intent detection and deterministic rendering.

The renderer is first-party code, so unlike an authored tool it can be pinned
exactly — same outline in, valid file out, every time and on every provider.
"""
from __future__ import annotations

import pytest

from waqil_api import document_factory as factory
from waqil_api.contracts import DocumentOutlineV1, DocumentSectionV1


def _outline() -> DocumentOutlineV1:
    return DocumentOutlineV1(
        title="BAPCO — DAC expansion",
        subtitle="Account brief",
        sections=[
            DocumentSectionV1(
                heading="Where we are",
                body="Live on 2xH100 since August.",
                bullets=["$110k ARR booked", "Two open actions"],
                notes="Lead with the ARR.",
            ),
            DocumentSectionV1(
                heading="Shapes",
                body=(
                    "Validated shapes:\n"
                    "| Shape | Units | $/hr |\n"
                    "|---|---|---|\n"
                    "| H100_X2 | 1 | 12.40 |\n"
                    "| A100_80G_X2 | 2 | 8.10 |"
                ),
            ),
        ],
        sources=["https://example.com/bench"],
    )


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("make me a deck on this account", True),
        ("build a one-pager for BAPCO", True),
        ("turn this into a pdf brief", True),
        ("give me a deck", True),
        ("summarize this as a pdf", True),
        ("what makes a good deck?", False),
        ("summarize this document", False),
        ("explain the pdf I attached", False),
        # Ali's real prompt: a "brief" is content he wants in the chat, not a
        # file. Ambiguous content nouns must never trigger a download.
        ("give me a very short brief on the benchmarks of these models", False),
        ("draft a short report from these notes", False),
        ("write up a summary of this call", False),
    ],
)
def test_document_request_detection(prompt: str, expected: bool) -> None:
    assert factory.is_explicit_document_request(prompt) is expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("make me a deck", "pptx"),
        ("build the slides", "pptx"),
        ("a deck I can send as a pdf", "pptx"),  # slides win ties
        ("write a one-pager", "pdf"),
        ("produce a brief", "pdf"),
    ],
)
def test_requested_format(prompt: str, expected: str) -> None:
    assert factory.requested_format(prompt) == expected


def test_filename_is_slugged_and_bounded() -> None:
    assert factory.filename_for("BAPCO — DAC expansion!", "pptx") == "bapco-dac-expansion.pptx"
    assert factory.filename_for("", "pdf") == "metis-document.pdf"
    assert len(factory.filename_for("x" * 200, "pdf")) <= 64


def test_pptx_render_is_a_real_deck() -> None:
    pptx = pytest.importorskip("pptx")
    from io import BytesIO

    data = factory.render(_outline(), "pptx")
    assert data[:2] == b"PK"  # OOXML is a zip
    deck = pptx.Presentation(BytesIO(data))
    # Cover + one slide per section + a sources slide.
    assert len(deck.slides) == 4
    texts = [
        shape.text_frame.text
        for slide in deck.slides
        for shape in slide.shapes
        if shape.has_text_frame
    ]
    assert any("BAPCO" in text for text in texts)
    assert any("$110k ARR booked" in text for text in texts)
    assert any(shape.has_table for slide in deck.slides for shape in slide.shapes)
    assert deck.slides[1].notes_slide.notes_text_frame.text == "Lead with the ARR."


def test_pdf_render_is_a_real_pdf() -> None:
    pytest.importorskip("reportlab")
    data = factory.render(_outline(), "pdf")
    assert data[:5] == b"%PDF-"
    assert data.rstrip().endswith(b"%%EOF")


def test_markup_in_content_cannot_break_the_pdf() -> None:
    """Platypus reads paragraphs as mini-HTML: an unescaped '<' from model
    prose would drop the line or fail the build outright."""
    pytest.importorskip("reportlab")
    outline = DocumentOutlineV1(
        title="a < b & c > d",
        sections=[DocumentSectionV1(heading="<b>raw</b>", body="1 < 2 & 3 > 2")],
    )
    assert factory.render(outline, "pdf")[:5] == b"%PDF-"


def test_markdown_table_is_parsed_out_of_the_body() -> None:
    """Tables ride in the body as markdown because typed column/row arrays
    make Command A+ emit tool arguments its own platform rejects."""
    blocks = factory.split_body(
        "Intro line\n| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\nAfter"
    )
    assert [type(block).__name__ for block in blocks] == [
        "TextBlock", "TableBlock", "TextBlock",
    ]
    assert blocks[0].text == "Intro line"
    assert blocks[1].columns == ["A", "B", "C"]
    assert blocks[1].rows == [["1", "2", "3"]]
    assert blocks[2].text == "After"


def test_ragged_table_rows_are_padded_not_dropped() -> None:
    blocks = factory.split_body("| A | B | C |\n|---|---|---|\n| 1 | 2 |\n| 1 | 2 | 3 | 4 |")
    assert blocks[0].rows == [["1", "2", ""], ["1", "2", "3"]]


def test_body_without_a_table_is_one_text_block() -> None:
    blocks = factory.split_body("Just prose.")
    assert len(blocks) == 1 and blocks[0].text == "Just prose."


def test_empty_outline_still_renders_both_formats() -> None:
    pytest.importorskip("reportlab")
    pytest.importorskip("pptx")
    empty = DocumentOutlineV1()
    assert factory.render(empty, "pdf")[:5] == b"%PDF-"
    assert factory.render(empty, "pptx")[:2] == b"PK"


def test_unknown_format_is_refused() -> None:
    with pytest.raises(factory.DocumentRenderError):
        factory.render(_outline(), "docx")
