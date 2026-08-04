"""Citation rendering: Notion pages link, invented markers do not survive.

These exercise the pure helpers directly. The end-to-end citation path already
has coverage in `test_grounding_review.py`; what needs pinning here is the two
things a model can get wrong — pointing at a page the reader cannot open, and
pointing at a source that was never there."""
from __future__ import annotations

from waqil_api.control_plane import (
    _append_cited_sources,
    _format_knowledge,
    _notion_page_title,
    _notion_page_url,
    _source_display,
    _strip_dangling_markers,
)

_PAGE_ID = "1ad1d12c506280b1a995c38271e105ce"
_MIRROR_FILE = f"sr-volvo--{_PAGE_ID}.md"


def _notion_snippet(**overrides) -> dict:
    snippet = {
        "source_label": "Notion",
        "provider": "notion",
        "rel_path": _MIRROR_FILE,
        "symbol": "Prep meeting",
        # The chunker prepends the heading breadcrumb to every Markdown window.
        "text": "SR - Volvo > Prep meeting\n\nACEs: Herman Ericsson.",
        "score": 0.9,
    }
    snippet.update(overrides)
    return snippet


def _local_snippet(**overrides) -> dict:
    snippet = {
        "source_label": "proj",
        "provider": "local",
        "rel_path": "apps/api/src/waqil_api/corpus.py",
        "symbol": "retrieve",
        "text": "async def retrieve(...)",
        "score": 0.9,
    }
    snippet.update(overrides)
    return snippet


# ── Notion links ─────────────────────────────────────────────────────────────


def test_page_url_is_recovered_from_the_mirror_filename() -> None:
    assert _notion_page_url(_MIRROR_FILE) == f"https://www.notion.so/{_PAGE_ID}"


def test_page_url_is_none_for_anything_that_is_not_a_mirror_file() -> None:
    # No id means no link, and no link means the old path::symbol form is kept.
    assert _notion_page_url("notes.md") is None
    assert _notion_page_url("sr-volvo--not-a-page-id.md") is None


def test_title_comes_from_the_breadcrumb_not_the_lossy_slug() -> None:
    # "SR - Volvo" does not survive a round trip through the filename slug.
    assert _notion_page_title(_notion_snippet()) == "SR - Volvo"


def test_title_ignores_a_breadcrumb_that_does_not_match_the_filename() -> None:
    # A passage with no breadcrumb starts with body text. Promoting that to a
    # page title would invent provenance, so the filename wins instead.
    titled = _notion_page_title(
        _notion_snippet(text="Data of accounts and contracts sits on master data")
    )
    assert titled == "Sr Volvo"


def test_notion_source_displays_as_title_and_section() -> None:
    assert _source_display(_notion_snippet()) == "SR - Volvo › Prep meeting"


def test_page_level_passage_is_not_rendered_as_title_twice() -> None:
    snippet = _notion_snippet(symbol="SR - Volvo", text="SR - Volvo\n\nCurrent status")
    assert _source_display(snippet) == "SR - Volvo"


def test_export_noise_is_cleaned_out_of_the_section_name() -> None:
    # Both seen on real mirrored pages: a tab-indented heading keeps its `#`
    # because the chunker detects it on the lstripped line but slices the raw
    # one, and Notion's export leaves block attributes on the end.
    snippet = _notion_snippet(symbol='\t## Prep meeting {toggle="true"}')
    assert _source_display(snippet) == "SR - Volvo › Prep meeting"


def test_local_source_keeps_the_path_form() -> None:
    assert _source_display(_local_snippet()) == (
        "apps/api/src/waqil_api/corpus.py::retrieve"
    )


def test_cited_notion_source_renders_as_an_openable_link() -> None:
    answer, dropped = _append_cited_sources(
        "The ACEs are Herman and Daniel [1].", [_notion_snippet()]
    )
    assert dropped == []
    assert answer.endswith(
        f"[1] Notion — [SR - Volvo › Prep meeting](https://www.notion.so/{_PAGE_ID})"
    )


def test_cited_local_source_is_not_linked() -> None:
    answer, _ = _append_cited_sources("Retrieval lives here [1].", [_local_snippet()])
    assert "**Sources**\n[1] proj — apps/api/src/waqil_api/corpus.py::retrieve" in answer
    assert "](" not in answer


def test_prompt_side_knowledge_block_carries_no_url() -> None:
    # A URL in the prompt is a URL the model can paste into prose, where nothing
    # checks it points at the passage being described.
    block = _format_knowledge([_notion_snippet()])
    assert block.startswith("[1] Notion — SR - Volvo › Prep meeting\n")
    assert "notion.so" not in block


# ── Dangling markers ─────────────────────────────────────────────────────────


def test_marker_past_the_end_of_the_sources_is_removed() -> None:
    text, dropped = _strip_dangling_markers("Volvo wants an MVP [9].", 2)
    assert text == "Volvo wants an MVP."
    assert dropped == [9]


def test_valid_markers_are_untouched() -> None:
    text, dropped = _strip_dangling_markers("Both [1] and [2] agree.", 2)
    assert text == "Both [1] and [2] agree."
    assert dropped == []


def test_an_answer_with_no_markers_is_returned_byte_identical() -> None:
    original = "The launch code is ORCHID-73."
    text, dropped = _strip_dangling_markers(original, 0)
    assert text == original
    assert dropped == []


def test_fenced_code_is_left_exact() -> None:
    original = "Use this:\n\n```python\nrows[0] = items[1]\n```\n\nThat is all."
    text, dropped = _strip_dangling_markers(original, 0)
    assert text == original
    assert dropped == []


def test_inline_code_is_left_exact() -> None:
    text, dropped = _strip_dangling_markers("Read `chunks[0]` first [4].", 1)
    assert text == "Read `chunks[0]` first."
    assert dropped == [4]


def test_a_markdown_link_is_not_mistaken_for_a_citation() -> None:
    original = "See [1](https://example.com/spec) for the schema."
    text, dropped = _strip_dangling_markers(original, 0)
    assert text == original
    assert dropped == []


def test_every_marker_invented_means_no_sources_block_is_appended() -> None:
    # The regression this guards: the marker survived, so `_ground_review` read
    # the answer as grounded and skipped the revision that would have fixed it.
    answer, dropped = _append_cited_sources(
        "Per your Notion notes [3], the MVP is scoped.", [_notion_snippet()]
    )
    assert answer == "Per your Notion notes, the MVP is scoped."
    assert dropped == [3]
    assert "**Sources**" not in answer


def test_real_and_invented_markers_in_one_answer() -> None:
    answer, dropped = _append_cited_sources(
        "Herman is the ACE [1], and the deadline is Friday [7].", [_notion_snippet()]
    )
    assert "Herman is the ACE [1], and the deadline is Friday." in answer
    assert dropped == [7]
    assert "**Sources**\n[1] Notion — [SR - Volvo › Prep meeting](" in answer
