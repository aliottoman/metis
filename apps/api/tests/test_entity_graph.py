from __future__ import annotations

from waqil_api.entity_graph import EntityExtraction, build_prompt, parse


def test_prompt_frames_document_as_untrusted_and_requests_json() -> None:
    prompt = build_prompt("Metis uses Cohere.")
    assert "DOCUMENT:" in prompt
    assert "untrusted" in prompt.lower()
    assert '"entities"' in prompt and '"relations"' in prompt


def test_parses_clean_json() -> None:
    raw = (
        '{"entities":[{"name":"Metis","kind":"project"},'
        '{"name":"Ali","kind":"person"}],'
        '"relations":[{"source":"Ali","relation":"builds","target":"Metis"}]}'
    )
    result = parse(raw)
    assert {(e.name, e.kind) for e in result.entities} == {
        ("Metis", "project"),
        ("Ali", "person"),
    }
    assert (result.relations[0].source, result.relations[0].relation) == ("Ali", "builds")


def test_strips_code_fence_and_prose() -> None:
    raw = 'Sure!\n```json\n{"entities":[{"name":"Cohere","kind":"organization"}],"relations":[]}\n```'
    result = parse(raw)
    assert [e.name for e in result.entities] == ["Cohere"]


def test_unknown_kind_is_coerced_and_duplicates_dropped() -> None:
    raw = (
        '{"entities":[{"name":"OCI","kind":"vendor"},{"name":"OCI","kind":"vendor"}],'
        '"relations":[{"source":"OCI","relation":"hosts the model","target":"Command A"},'
        '{"source":"OCI","relation":"hosts the model","target":"Command A"}]}'
    )
    result = parse(raw)
    assert [(e.name, e.kind) for e in result.entities] == [("OCI", "other")]
    # relation verb is normalized (lowercased, spaces -> underscores) and deduped
    assert [r.relation for r in result.relations] == ["hosts_the_model"]


def test_malformed_output_is_fail_soft() -> None:
    assert parse("the model refused to answer") == EntityExtraction()
    assert parse("") == EntityExtraction()
    assert parse('{"entities": "not a list"}') == EntityExtraction()


def test_incomplete_records_are_skipped() -> None:
    raw = (
        '{"entities":[{"kind":"person"},{"name":"Valid","kind":"person"}],'
        '"relations":[{"source":"A","target":"B"},{"source":"A","relation":"knows","target":"B"}]}'
    )
    result = parse(raw)
    assert [e.name for e in result.entities] == ["Valid"]  # nameless entity skipped
    assert [r.relation for r in result.relations] == ["knows"]  # relationless edge skipped
