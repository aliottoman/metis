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


def test_explicit_web_request_detection() -> None:
    from waqil_api.web_research import is_explicit_web_request

    assert is_explicit_web_request(
        "Research online and give me a very short brief on the benchmarks of X"
    )
    assert is_explicit_web_request("search the web for the latest Ollama release")
    assert is_explicit_web_request("can you look this up on the internet?")
    assert not is_explicit_web_request("summarize the attached document")
    assert not is_explicit_web_request("what does this error mean?")


def test_substantive_prompt_walks_past_bare_retries() -> None:
    from waqil_api.control_plane import _substantive_prompt

    state = {
        "prompt": "Try again",
        "recent_messages": [
            {"role": "user", "content": "Research the benchmarks of model X"},
            {"role": "assistant", "content": "drafted a tool"},
            {"role": "user", "content": "Build it"},
        ],
    }
    assert _substantive_prompt(state) == "Research the benchmarks of model X"
    state["prompt"] = "Compare model X and model Y"
    assert _substantive_prompt(state) == "Compare model X and model Y"
