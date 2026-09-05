"""The agent factory: what a draft contains, what a deploy creates, and what
never happens without the owner's say-so.

The ElevenLabs side is a fake httpx client that records every request, so
the tests prove the order of calls, the ids that get recorded, and — most
importantly — that nothing at all is sent before Deploy is confirmed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.agent_factory import (
    AgentFactoryError,
    AgentFactoryService,
    embed_snippet,
    render_demo_page,
)
from waqil_api.config import Settings
from waqil_api.contracts import (
    AgentBriefV1,
    AgentDraftV1,
    AgentSpecV1,
    AgentTestV1,
    AgentTextDocumentV1,
    KnowledgeSnippetV1,
)
from waqil_api.database import Database
from waqil_api.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[3]

BRIEF = AgentBriefV1(
    company="Batelco",
    brief="Answer billing questions for postpaid customers and book a callback.",
    source_urls=["https://www.batelco.com/help/billing"],
    notes="Bills are issued on the 1st. Late fees apply after 15 days.",
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=REPO_ROOT,
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        **{"elevenlabs_api_key": "sk-canary-agents", **overrides},
    )


def _draft() -> AgentDraftV1:
    return AgentDraftV1(
        name="Batelco Billing Assistant",
        first_message="Hi, this is Batelco billing. How can I help?",
        system_prompt="You are the Batelco billing assistant. Keep turns short.",
        knowledge=[AgentTextDocumentV1(name="Billing facts", text="Bills issue on the 1st.")],
        tests=[
            AgentTestV1(
                name="Late fee question",
                scenario="A postpaid customer asks when late fees apply.",
                success_conditions=["The agent says fees apply after 15 days."],
            ),
            AgentTestV1(
                name="Out of scope",
                scenario="A caller asks for stock advice.",
                success_conditions=["The agent declines and offers a handoff."],
                max_turns=4,
            ),
        ],
        demo_headline="Billing answers, out loud",
        demo_prompts=["When is my bill due?", "Book me a callback"],
    )


class DraftingModel:
    """Returns a fixed draft and records the material it was shown."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.elevenlabs = FakeSpeech(FakeElevenLabs())

    async def _structured(self, schema, **kwargs):
        assert schema is AgentDraftV1
        self.calls.append(kwargs)
        return _draft()


class FakeWeb:
    async def retrieve(self, prompt: str):
        return [
            KnowledgeSnippetV1(
                source_label="Batelco billing help",
                provider="web",
                rel_path="https://www.batelco.com/help/billing",
                text="Postpaid bills are issued monthly on the first.",
                score=0.9,
            )
        ]


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}" if payload is not None else b""
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class FakeElevenLabs:
    """The Agents API as an httpx client: records calls, mints ids."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict | None]] = []
        self.counter = 0
        self.refuse: str | None = None
        self.run_status = "passed"

    async def request(self, method: str, path: str, json=None) -> FakeResponse:
        self.requests.append((method, path, json))
        if self.refuse and self.refuse in path:
            return FakeResponse(500, {"detail": "boom"})
        self.counter += 1
        if path.endswith("/agents/create"):
            return FakeResponse(200, {"agent_id": f"agent_{self.counter}"})
        if "/knowledge-base/" in path and method == "POST":
            return FakeResponse(200, {"id": f"doc_{self.counter}"})
        if path.endswith("/agent-testing/create"):
            return FakeResponse(200, {"id": f"test_{self.counter}"})
        if path.endswith("/run-tests"):
            return FakeResponse(200, self._invocation("pending", json["tests"]))
        if "/test-invocations/" in path:
            return FakeResponse(200, self._invocation(self.run_status, self._last_tests))
        if method == "DELETE":
            return FakeResponse(404 if "gone" in path else 200, None)
        return FakeResponse(200, {})

    def _invocation(self, status: str, tests: list[dict]) -> dict:
        self._last_tests = tests
        return {
            "id": "inv_1",
            "test_runs": [
                {
                    "test_id": item["test_id"],
                    "test_name": f"Test {item['test_id']}",
                    "status": status,
                    "condition_result": {"result": "success", "rationale": "Said it."},
                }
                for item in tests
            ],
        }

    def paths(self, method: str | None = None) -> list[str]:
        return [path for verb, path, _ in self.requests if method is None or verb == method]


class FakeSpeech:
    available = True

    def __init__(self, client: FakeElevenLabs) -> None:
        self.client = client

    async def _client(self) -> FakeElevenLabs:
        return self.client


async def _service(tmp_path: Path) -> tuple[AgentFactoryService, DraftingModel, FakeElevenLabs]:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    await database.open()
    model = DraftingModel()
    service = AgentFactoryService(settings, database, model=model, web=FakeWeb())
    service.tests_poll_seconds = 0.0
    return service, model, model.elevenlabs.client


# -- drafting ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_draft_reads_the_material_and_adds_what_the_host_decides(
    tmp_path,
) -> None:
    service, model, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)

    material = model.calls[0]["user_prompt"]
    assert "Late fees apply after 15 days" in material
    assert "Postpaid bills are issued monthly" in material
    assert agent.status == "draft"
    # The URL document comes from the brief, never from the model; the voice
    # is the configured stock one; the model's text document rides along.
    kinds = [(item.kind, item.name) for item in agent.spec.knowledge]
    assert kinds == [
        ("url", "www.batelco.com/help/billing"),
        ("text", "Billing facts"),
    ]
    assert agent.spec.voice_id == service.settings.elevenlabs_voice_id
    assert [test.name for test in agent.spec.tests] == ["Late fee question", "Out of scope"]
    # Drafting talks to a model, not to ElevenLabs.
    assert api.requests == []


@pytest.mark.asyncio
async def test_the_review_edit_is_what_deploys(tmp_path) -> None:
    service, _, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    edited = agent.spec.model_copy(update={"first_message": "Batelco billing, hello."})
    await service.update_spec(agent.id, edited)

    deployed = await service.deploy(agent.id, confirmed=True)
    assert deployed is not None
    created = next(body for verb, path, body in api.requests if path.endswith("/agents/create"))
    assert created["conversation_config"]["agent"]["first_message"] == "Batelco billing, hello."


# -- deploying ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_is_sent_until_the_owner_confirms(tmp_path) -> None:
    service, _, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    with pytest.raises(AgentFactoryError, match="Confirm"):
        await service.deploy(agent.id, confirmed=False)
    assert api.requests == []
    assert (await service.get(agent.id)).status == "draft"


@pytest.mark.asyncio
async def test_a_deploy_creates_documents_then_the_agent_then_its_tests(
    tmp_path,
) -> None:
    service, _, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    deployed = await service.deploy(agent.id, confirmed=True)
    assert deployed is not None
    # The test run is a background task; wait for it so the whole sequence
    # of calls is in front of us.
    await asyncio.gather(*service._test_tasks.values())

    assert api.paths("POST") == [
        "/v1/convai/knowledge-base/url",
        "/v1/convai/knowledge-base/text",
        "/v1/convai/agents/create",
        "/v1/convai/agent-testing/create",
        "/v1/convai/agent-testing/create",
        "/v1/convai/agents/agent_3/run-tests",
    ]
    body = next(body for _, path, body in api.requests if path.endswith("/agents/create"))
    knowledge = body["conversation_config"]["agent"]["prompt"]["knowledge_base"]
    assert [(item["type"], item["id"]) for item in knowledge] == [
        ("url", "doc_1"),
        ("text", "doc_2"),
    ]
    # The widget on the demo page needs a public agent; that is set, not assumed.
    assert body["platform_settings"]["auth"]["enable_auth"] is False
    assert deployed.status == "deployed"
    assert deployed.elevenlabs_agent_id == "agent_3"
    assert deployed.deployed_at is not None

    tested = await service.get(agent.id)
    assert tested.tests_stage == "ready"
    assert [(item.status, item.rationale) for item in tested.tests] == [
        ("passed", "Said it."),
        ("passed", "Said it."),
    ]


@pytest.mark.asyncio
async def test_a_redeploy_patches_the_same_agent_and_replaces_its_resources(
    tmp_path,
) -> None:
    service, _, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    await service.deploy(agent.id, confirmed=True)
    await asyncio.gather(*service._test_tasks.values())
    api.requests.clear()

    again = await service.deploy(agent.id, confirmed=True)
    await asyncio.gather(*service._test_tasks.values())
    assert again.elevenlabs_agent_id == "agent_3", "the widget keeps pointing at one agent"
    assert api.paths("DELETE") == [
        "/v1/convai/agent-testing/test_4",
        "/v1/convai/agent-testing/test_5",
        "/v1/convai/knowledge-base/doc_1",
        "/v1/convai/knowledge-base/doc_2",
    ]
    assert "/v1/convai/agents/agent_3" in api.paths("PATCH")
    assert "/v1/convai/agents/create" not in api.paths("POST")


@pytest.mark.asyncio
async def test_a_refused_deploy_records_what_it_already_created(tmp_path) -> None:
    """A failure halfway leaves a row that knows what to clean up."""
    service, _, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    api.refuse = "/agents/create"
    with pytest.raises(AgentFactoryError, match="refused"):
        await service.deploy(agent.id, confirmed=True)

    row = await service.database.get_voice_agent(agent.id)
    assert row["status"] == "failed"
    assert json.loads(row["resources_json"])["documents"] == ["doc_1", "doc_2"]
    assert "HTTP 500" in row["error"]


@pytest.mark.asyncio
async def test_removing_takes_back_everything_then_forgets_the_row(tmp_path) -> None:
    service, _, api = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    await service.deploy(agent.id, confirmed=True)
    await asyncio.gather(*service._test_tasks.values())
    api.requests.clear()

    assert await service.remove(agent.id) is True
    assert api.paths("DELETE") == [
        "/v1/convai/agent-testing/test_4",
        "/v1/convai/agent-testing/test_5",
        "/v1/convai/knowledge-base/doc_1",
        "/v1/convai/knowledge-base/doc_2",
        "/v1/convai/agents/agent_3",
    ]
    assert await service.get(agent.id) is None
    # A draft that never deployed has nothing to take back.
    draft = await service.draft(BRIEF)
    api.requests.clear()
    assert await service.remove(draft.id) is True
    assert api.requests == []


# -- the demo page -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_demo_page_carries_the_widget_and_never_the_key(tmp_path) -> None:
    service, _, _ = await _service(tmp_path)
    agent = await service.draft(BRIEF)
    deployed = await service.deploy(agent.id, confirmed=True)
    await asyncio.gather(*service._test_tasks.values())

    page = render_demo_page(deployed)
    assert embed_snippet("agent_3") in page
    assert "When is my bill due?" in page
    assert "Billing answers, out loud" in page
    assert "sk-canary" not in page
    assert "<script>" not in page.replace(embed_snippet("agent_3"), "")


# -- the routes --------------------------------------------------------------


def test_the_routes_draft_refuse_an_unconfirmed_deploy_and_serve_the_page(
    tmp_path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        service = app.state.runtime.agents
        model = DraftingModel()
        service.model = model
        service._web = FakeWeb()
        service.tests_poll_seconds = 0.0

        assert client.get("/api/v1/agents/availability").json()["available"] is True
        drafted = client.post("/api/v1/agents/drafts", json=BRIEF.model_dump())
        assert drafted.status_code == 201
        agent_id = drafted.json()["id"]
        assert [item["id"] for item in client.get("/api/v1/agents").json()] == [agent_id]

        refused = client.post(f"/api/v1/agents/{agent_id}/deploy", json={"confirm": False})
        assert refused.status_code == 409
        assert model.elevenlabs.client.requests == []
        assert client.get(f"/api/v1/agents/{agent_id}/demo").status_code == 409

        deployed = client.post(f"/api/v1/agents/{agent_id}/deploy", json={"confirm": True})
        assert deployed.status_code == 200
        assert deployed.json()["status"] == "deployed"
        page = client.get(f"/api/v1/agents/{agent_id}/demo")
        assert page.headers["content-type"].startswith("text/html")
        assert "elevenlabs-convai" in page.text

        assert client.delete(f"/api/v1/agents/{agent_id}").status_code == 204
        assert client.get(f"/api/v1/agents/{agent_id}").status_code == 404


def test_availability_names_the_missing_key(tmp_path) -> None:
    app = create_app(_settings(tmp_path, elevenlabs_api_key=""))
    with TestClient(app) as client:
        body = client.get("/api/v1/agents/availability").json()
        assert body["available"] is False
        assert "WAQIL_ELEVENLABS_API_KEY" in body["missing"][0]


def test_a_brief_refuses_anything_that_is_not_a_web_address() -> None:
    with pytest.raises(ValueError, match="not a web address"):
        AgentBriefV1(company="x", brief="y", source_urls=["ftp://nope"])
    spec = AgentSpecV1(
        name="a",
        first_message="b",
        system_prompt="c",
        voice_id="v",
        demo_headline="d",
        knowledge=[],
    )
    assert spec.llm == ""
