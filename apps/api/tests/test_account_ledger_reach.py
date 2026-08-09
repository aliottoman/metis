"""An account named in an ordinary chat still gets its own record.

The failure this pins was measured live on both hosted lanes: asked "what is
the current state of the BAPCO account — what have I written down about it, and
what is outstanding?", an unscoped chat answered from Notion pages and run logs
and reported "no saved facts or actions" for an account holding twenty-two
reviewed facts and seven open ones. Scoping a conversation to the account had
always worked; naming it in a sentence had no reach into the ledger at all, so
the confident emptiness read as evidence.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _account_with_a_record(client: TestClient, name: str) -> str:
    created = client.post("/api/v1/customers", json={"name": name})
    assert created.status_code == 201, created.text
    account_id = str(created.json()["id"])
    for index in range(12):
        added = client.post(
            f"/api/v1/customers/{account_id}/facts",
            json={"kind": "requirement", "content": f"Recorded requirement {index}"},
        )
        assert added.status_code in (200, 201), added.text
    for index in range(10):
        added = client.post(
            f"/api/v1/customers/{account_id}/actions",
            json={"description": f"Open commitment {index}"},
        )
        assert added.status_code in (200, 201), added.text
    return account_id


def _evidence(client: TestClient, account_id: str, *, compact: bool) -> list:
    customers = client.app.state.runtime.customers  # type: ignore[attr-defined]
    return client.portal.call(  # type: ignore[attr-defined]
        lambda: customers.evidence(account_id, compact=compact)
    )


def test_the_full_ledger_is_unchanged_for_a_scoped_chat(client: TestClient) -> None:
    account_id = _account_with_a_record(client, "Ledger Full")
    snippets = _evidence(client, account_id, compact=False)
    kinds = [item.source_label for item in snippets]
    assert kinds.count("Reviewed fact") == 12
    assert kinds.count("Open action") == 10


def test_a_compact_ledger_caps_each_kind_rather_than_truncating_the_list(
    client: TestClient,
) -> None:
    """The cap has to be per kind. A plain head would have cut every open
    action off the end of a long fact list — and "what is outstanding?" is
    exactly the question being answered."""
    account_id = _account_with_a_record(client, "Ledger Compact")
    snippets = _evidence(client, account_id, compact=True)
    kinds = [item.source_label for item in snippets]
    assert kinds.count("Reviewed fact") == 10
    assert kinds.count("Open action") == 8


def test_an_unscoped_message_naming_one_account_resolves_it(client: TestClient) -> None:
    account_id = _account_with_a_record(client, "Northwind Energy")
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    resolved = client.portal.call(  # type: ignore[attr-defined]
        cp._named_account,
        {"prompt": "what is outstanding on Northwind Energy right now?"},
    )
    assert resolved is not None and str(resolved["id"]) == account_id


def test_a_message_naming_no_account_resolves_nothing(client: TestClient) -> None:
    """Reading a ledger is safer than writing to one, but it is still the wrong
    record to read. The bar is the same resolver the write path uses."""
    _account_with_a_record(client, "Southwind Energy")
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    resolved = client.portal.call(  # type: ignore[attr-defined]
        cp._named_account,
        {"prompt": "what is the minimum commitment for a dedicated AI cluster?"},
    )
    assert resolved is None
