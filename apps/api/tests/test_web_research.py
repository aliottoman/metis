"""Web research: the pure parsing layer, tested without a network.

The live search endpoint and result pages are moving targets; what must stay
correct forever is how Metis decodes DuckDuckGo's redirect wrapping, drops ad
slots, and flattens HTML into promptable text.
"""
from waqil_api.config import Settings
from waqil_api.web_research import WebResearch, _decode_result_href, _strip_html


def test_decode_unwraps_duckduckgo_redirect() -> None:
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc123"
    assert _decode_result_href(href) == "https://example.com/page"


def test_decode_drops_ad_slots() -> None:
    assert _decode_result_href("//duckduckgo.com/y.js?ad_domain=x") == ""


def test_decode_passes_direct_links_and_rejects_junk() -> None:
    assert _decode_result_href("https://example.com/a") == "https://example.com/a"
    assert _decode_result_href("javascript:alert(1)") == ""


def test_strip_html_removes_scripts_and_collapses_whitespace() -> None:
    page = (
        "<head><title>T</title></head><body><script>var x = 1;</script>"
        "<p>Hello&nbsp;<b>world</b></p>\n\n<style>p{}</style>  done</body>"
    )
    assert _strip_html(page) == "Hello world done"


def test_availability_follows_the_setting() -> None:
    assert WebResearch(Settings()).available()
    assert not WebResearch(Settings(web_research_enabled=False)).available()
