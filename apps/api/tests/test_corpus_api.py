from __future__ import annotations


def test_corpus_status_reports_cloud_off_by_default(client) -> None:
    response = client.get("/api/v1/corpus/status")
    assert response.status_code == 200
    body = response.json()
    # Cloud retrieval is opt-in; a default install reports it unavailable.
    assert body["available"] is False
    assert body["cloud_embeddings_enabled"] is False


def test_profile_get_and_put(client) -> None:
    assert client.get("/api/v1/profile").json()["content"] == ""
    put = client.put(
        "/api/v1/profile", json={"content": "I am Ali; concise British English."}
    )
    assert put.status_code == 200
    assert "British English" in put.json()["content"]
    assert client.get("/api/v1/profile").json()["characters"] > 0


def test_notion_connection_is_local_secret_safe_and_manual(client) -> None:
    initial = client.get("/api/v1/corpus/notion")
    assert initial.status_code == 200
    assert initial.json()["configured"] is False

    configured = client.put(
        "/api/v1/corpus/notion",
        json={
            "access_token": "secret_test_notion_token",
            "root_page_ids": [
                "https://www.notion.so/Launch-11111111222233334444555555555555"
            ],
            "label": "Work Notion",
        },
    )
    assert configured.status_code == 200
    body = configured.json()
    assert body["configured"] is True
    assert body["token_configured"] is True
    assert body["source"]["provider"] == "notion"
    assert body["root_page_ids"] == ["11111111-2222-3333-4444-555555555555"]
    assert "secret_test_notion_token" not in configured.text

    # Saving without a token preserves the locally stored credential.
    updated = client.put(
        "/api/v1/corpus/notion",
        json={"root_page_ids": [], "label": "Everything shared"},
    )
    assert updated.status_code == 200
    assert updated.json()["token_configured"] is True
    assert updated.json()["root_page_ids"] == []


def test_corpus_source_lifecycle_and_consent_gate(client, tmp_path) -> None:
    source_dir = tmp_path / "myproj"
    source_dir.mkdir()
    (source_dir / "a.py").write_text("x = 1\n")

    # A non-existent path is rejected before anything is registered.
    bad = client.post("/api/v1/corpus/sources", json={"root_path": "/does/not/exist"})
    assert bad.status_code == 400

    created = client.post(
        "/api/v1/corpus/sources",
        json={"root_path": str(source_dir), "label": "myproj", "kind": "code"},
    )
    assert created.status_code == 201
    source = created.json()
    source_id = source["id"]
    assert source["consent"] is False
    assert source["status"] == "pending"

    # Registering the same path twice is a conflict.
    assert (
        client.post(
            "/api/v1/corpus/sources", json={"root_path": str(source_dir)}
        ).status_code
        == 409
    )

    # Indexing is refused until consent is granted (the egress boundary).
    assert (
        client.post(f"/api/v1/corpus/sources/{source_id}/reindex").status_code == 409
    )

    granted = client.post(
        f"/api/v1/corpus/sources/{source_id}/consent",
        json={"consent": True, "reason": "mine"},
    )
    assert granted.status_code == 200
    assert granted.json()["consent"] is True

    # With consent but no OCI configured in tests, indexing reports unavailable.
    assert (
        client.post(f"/api/v1/corpus/sources/{source_id}/reindex").status_code == 503
    )

    listing = client.get("/api/v1/corpus/sources").json()
    assert any(item["id"] == source_id for item in listing)

    # Search degrades to empty (rather than erroring) when unavailable.
    search = client.post("/api/v1/corpus/search", json={"query": "anything"})
    assert search.status_code == 200
    assert search.json() == []

    deleted = client.delete(f"/api/v1/corpus/sources/{source_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/v1/corpus/sources/{source_id}").status_code == 404


def test_code_graph_endpoints_are_available_and_typed(client) -> None:
    # The code graph is local-only, so its endpoints answer even with no cloud
    # and an empty graph — returning valid, zeroed contract shapes.
    stats = client.get("/api/v1/corpus/graph/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["node_count"] == 0
    assert body["edge_count"] == 0
    assert body["nodes_by_kind"] == {}

    lookup = client.get("/api/v1/corpus/graph/symbol/anything")
    assert lookup.status_code == 200
    payload = lookup.json()
    assert payload["name"] == "anything"
    assert payload["definitions"] == []
    assert payload["callers"] == []
    assert payload["callees"] == []


def test_entity_graph_is_opt_in_and_endpoints_are_typed(client) -> None:
    # Stage 2 is off by default; status reflects it and the endpoints answer
    # with valid, empty contract shapes.
    status = client.get("/api/v1/corpus/status").json()
    assert status["entity_graph_enabled"] is False

    stats = client.get("/api/v1/corpus/entities/stats")
    assert stats.status_code == 200
    assert stats.json()["node_count"] == 0

    lookup = client.get("/api/v1/corpus/entities/Cohere")
    assert lookup.status_code == 200
    body = lookup.json()
    assert body["name"] == "Cohere"
    assert body["relations_out"] == []
    assert body["relations_in"] == []
