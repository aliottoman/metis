from __future__ import annotations

from fastapi.testclient import TestClient


def test_customer_capture_review_save_and_markdown_output(client: TestClient) -> None:
    created = client.post(
        "/api/v1/customers",
        json={"name": "Acme", "aliases": ["ACME"], "industry": "Retail", "region": "UAE"},
    )
    assert created.status_code == 201
    account_id = created.json()["id"]

    session = client.post(
        "/api/v1/model-session/launch",
        json={
            "model": "deterministic",
            "idle_timeout_seconds": 300,
            "context_window": 32768,
        },
    )
    assert session.status_code == 200
    assert session.json()["state"] == "ready"

    captured = client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": account_id,
            "title": "Discovery call",
            "source_kind": "meeting",
            "content": "They need a private deployment.\nAction: Ali to send sizing guidance.",
        },
    )
    assert captured.status_code == 201
    assert captured.json()["status"] == "waiting"
    source_id = captured.json()["id"]

    duplicate = client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": account_id,
            "title": "Same note",
            "source_kind": "note",
            "content": "They need a private deployment.\nAction: Ali to send sizing guidance.",
        },
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["status"] == "duplicate"
    assert duplicate.json()["id"] == source_id

    analyzed = client.post(f"/api/v1/customers/sources/{source_id}/analyze")
    assert analyzed.status_code == 200
    proposal = analyzed.json()
    assert proposal["status"] == "review"
    assert proposal["extraction"]["actions"][0]["description"] == "Ali to send sizing guidance."

    saved = client.put(
        f"/api/v1/customers/proposals/{proposal['id']}/save",
        json={"extraction": proposal["extraction"]},
    )
    assert saved.status_code == 200
    assert saved.json()["status"] == "approved"

    detail = client.get(f"/api/v1/customers/{account_id}")
    assert detail.status_code == 200
    assert len(detail.json()["interactions"]) == 1
    assert detail.json()["actions"][0]["description"] == "Ali to send sizing guidance."

    tracker = client.put(
        "/api/v1/customer-settings",
        json={"tracker_url": "https://company.example/activity", "activity_template": ""},
    )
    assert tracker.status_code == 200
    output = client.post(
        f"/api/v1/customers/{account_id}/outputs",
        json={"kind": "activity_tracker"},
    )
    assert output.status_code == 201
    assert "Acme" in output.json()["content"]
    assert output.json()["tracker_url"] == "https://company.example/activity"


def test_customer_win_lifecycle_and_dashboard_tracker(client: TestClient) -> None:
    first = client.post(
        "/api/v1/customers",
        json={"name": "Northwind Authority", "industry": "Government", "region": "UAE"},
    )
    second = client.post("/api/v1/customers", json={"name": "Harbor Health"})
    assert first.status_code == 201 and second.status_code == 201
    first_id, second_id = first.json()["id"], second.json()["id"]

    recorded = client.post(
        f"/api/v1/customers/{first_id}/wins",
        json={
            "title": "Identity service Model Import DAC live",
            "brief": "Model Import DAC connected through the OpenAI-compatible endpoint.",
            "services": ["Generative AI Services", "DAC", "Model-Import"],
            "dac_shape": "Model Import DAC (2xA100-40G)",
            "yearly_arr": 42000,
            "won_at": "2025-12-30T00:00:00Z",
            "source_ref": "https://notes.example/northwind",
        },
    )
    assert recorded.status_code == 201
    win = recorded.json()
    assert win["account_name"] == "Northwind Authority"
    assert win["services"] == ["Generative AI Services", "DAC", "Model-Import"]

    other = client.post(
        f"/api/v1/customers/{second_id}/wins",
        json={"title": "PAYG contract signed", "services": ["On-demand"]},
    )
    assert other.status_code == 201
    assert other.json()["yearly_arr"] is None

    missing = client.post(
        "/api/v1/customers/cust_00000000000000000000/wins",
        json={"title": "Orphan win"},
    )
    assert missing.status_code == 404

    dashboard = client.get("/api/v1/customers/dashboard").json()
    assert dashboard["total_wins"] == 2
    assert dashboard["dac_wins"] == 1
    assert dashboard["total_yearly_arr"] == 42000
    assert dashboard["wins_by_service"]["DAC"] == 1
    assert dashboard["wins_by_service"]["On-demand"] == 1
    assert dashboard["recent_wins"][0]["account_name"] in {"Northwind Authority", "Harbor Health"}

    detail = client.get(f"/api/v1/customers/{first_id}").json()
    assert len(detail["wins"]) == 1
    assert detail["account"]["wins"] == 1

    updated = client.put(
        f"/api/v1/customers/wins/{win['id']}",
        json={
            "title": "Identity service Model Import DAC live",
            "services": ["Generative AI Services", "DAC"],
            "yearly_arr": 90000,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["yearly_arr"] == 90000

    deleted = client.delete(f"/api/v1/customers/wins/{win['id']}")
    assert deleted.status_code == 204
    assert client.delete(f"/api/v1/customers/wins/{win['id']}").status_code == 404
    assert client.get("/api/v1/customers/dashboard").json()["total_wins"] == 1

    # Deleting the account cascades to its remaining wins.
    assert client.delete(f"/api/v1/customers/{second_id}").status_code == 204
    assert client.get("/api/v1/customers/dashboard").json()["total_wins"] == 0


def test_delete_customer_removes_account_scoped_data(client: TestClient) -> None:
    created = client.post("/api/v1/customers", json={"name": "Disposable account"})
    assert created.status_code == 201
    account_id = created.json()["id"]
    captured = client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": account_id,
            "title": "Temporary note",
            "source_kind": "note",
            "content": "This account and its source should be removed.",
        },
    )
    assert captured.status_code == 201

    deleted = client.delete(f"/api/v1/customers/{account_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/customers/{account_id}").status_code == 404
    assert account_id not in {item["id"] for item in client.get("/api/v1/customers").json()}


def test_dac_shape_alone_counts_as_a_dac_win(client: TestClient) -> None:
    """The record-win form offers the DAC shape as the way to mark a DAC win, so
    a shape with no explicit DAC service tag must still reach the tracker tile."""
    account_id = client.post("/api/v1/customers", json={"name": "Shape only"}).json()["id"]
    recorded = client.post(
        f"/api/v1/customers/{account_id}/wins",
        json={
            "title": "Model Import DAC live",
            "services": ["Model-Import"],
            "dac_shape": "Model Import DAC (2xA100-40G)",
        },
    )
    assert recorded.status_code == 201
    dashboard = client.get("/api/v1/customers/dashboard").json()
    assert dashboard["total_wins"] == 1
    assert dashboard["dac_wins"] == 1


def test_win_dates_are_stored_as_utc_so_ordering_is_chronological(client: TestClient) -> None:
    """won_at rows are compared as text, so a preserved +04:00 offset would sort
    against a UTC row lexicographically rather than chronologically."""
    account_id = client.post("/api/v1/customers", json={"name": "Ordering"}).json()["id"]
    # 2026-01-01T00:00+04:00 is 2025-12-31T20:00Z — earlier than the second win.
    earlier = client.post(
        f"/api/v1/customers/{account_id}/wins",
        json={"title": "Gulf-offset win", "won_at": "2026-01-01T00:00:00+04:00"},
    )
    later = client.post(
        f"/api/v1/customers/{account_id}/wins",
        json={"title": "UTC win", "won_at": "2025-12-31T22:00:00Z"},
    )
    assert earlier.status_code == 201 and later.status_code == 201
    # Normalized to UTC on the way in, not kept at its original +04:00 offset.
    assert earlier.json()["won_at"] == "2025-12-31T20:00:00Z"

    titles = [item["title"] for item in client.get(f"/api/v1/customers/{account_id}").json()["wins"]]
    assert titles == ["UTC win", "Gulf-offset win"], titles
    recent = [item["title"] for item in client.get("/api/v1/customers/dashboard").json()["recent_wins"]]
    assert recent[:2] == ["UTC win", "Gulf-offset win"], recent


def test_naive_win_date_is_read_as_utc(client: TestClient) -> None:
    account_id = client.post("/api/v1/customers", json={"name": "Naive date"}).json()["id"]
    recorded = client.post(
        f"/api/v1/customers/{account_id}/wins",
        json={"title": "Date only", "won_at": "2025-12-30T00:00:00"},
    )
    assert recorded.status_code == 201
    assert recorded.json()["won_at"] == "2025-12-30T00:00:00Z"
