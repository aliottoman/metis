from __future__ import annotations

import sqlite3

import pytest

from waqil_api.contracts import (
    ApprovalRequestV1,
    EvalReportV1,
    EvalResultV1,
    ProposalStatus,
    RiskLevel,
    ToolManifestV1,
    ToolState,
)
from waqil_api.database import Database, SCHEMA_V1


@pytest.mark.asyncio
async def test_event_sequences_and_tool_activation_are_idempotent(tmp_path) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        conversation = await database.create_conversation("Test")
        message = await database.add_message(conversation.id, "user", "diagram")
        run = await database.create_run(
            conversation.id, message.id, graph_schema_version="1", model_aliases={}
        )
        first = await database.append_event(run.id, conversation.id, "one")
        second = await database.append_event(run.id, conversation.id, "two")
        assert (first.sequence, second.sequence) == (1, 2)

        manifest = ToolManifestV1(
            slug="reference-architecture-generator",
            name="Reference Architecture Generator",
            description="test",
            version="0.1.0",
            entrypoint="runner.py",
            risk_level=RiskLevel.R2,
            permissions=[],
            dependencies=[],
            input_schema={},
            output_schema={},
            content_hash="a" * 64,
        )
        report = EvalReportV1(
            passed=True,
            score=1,
            results=[EvalResultV1(case_id="smoke", passed=True)],
        )
        tool, version, proposal = await database.create_tool_candidate(
            manifest, report, run.id, "/bundle"
        )
        replay_tool, replay_version, replay_proposal = await database.create_tool_candidate(
            manifest, report, run.id, "/bundle"
        )
        assert (replay_tool.id, replay_version.id, replay_proposal.id) == (
            tool.id,
            version.id,
            proposal.id,
        )
        assert len(await database.list_tool_proposals()) == 1

        approval = ApprovalRequestV1(
            id="appr_original",
            run_id=run.id,
            action_id="activate:stable-action",
            kind="activate_tool",
            title="Activate",
            summary="Activate tested version",
            risk_level=RiskLevel.R3,
            proposal_id=proposal.id,
            tool_version_id=version.id,
            input_digest="b" * 64,
        )
        persisted = await database.create_approval(approval)
        replayed = await database.create_approval(
            approval.model_copy(update={"id": "appr_random_replay"})
        )
        assert persisted.id == replayed.id == "appr_original"

        one = await database.decide_tool_proposal(
            proposal.id, ProposalStatus.APPROVED, None, "same-action"
        )
        two = await database.decide_tool_proposal(
            proposal.id, ProposalStatus.APPROVED, None, "same-action"
        )
        assert one == two
        tools = await database.list_tools()
        assert tools[0].active_version_id == version.id
        assert tools[0].id == tool.id

        published, created = await database.add_assistant_message_once(
            conversation.id, "done", run.id
        )
        replayed_message, replay_created = await database.add_assistant_message_once(
            conversation.id, "different replay text", run.id
        )
        assert created is True
        assert replay_created is False
        assert replayed_message.id == published.id
        assert replayed_message.content == "done"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_migrations_upgrade_real_v1_are_idempotent_and_reject_future(tmp_path) -> None:
    path = tmp_path / "legacy-v1.db"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA_V1)
    connection.executescript(
        """
        DROP TABLE conversation_summaries;
        DROP TABLE tool_improvement_proposals;
        DROP TABLE tool_version_activation_log;
        DROP INDEX idx_messages_one_assistant_per_run;
        DROP INDEX idx_artifacts_run_name_hash;
        DROP INDEX idx_tool_proposals_source_version;
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        INSERT INTO schema_migrations VALUES (1, 'legacy');
        """
    )
    connection.close()

    database = Database(path)
    await database.open()
    await database.close()
    reopened = Database(path)
    await reopened.open()
    await reopened.close()

    connection = sqlite3.connect(path)
    versions = [row[0] for row in connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    )]
    objects = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        )
    }
    connection.close()
    assert versions == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert {
        "conversation_summaries",
        "tool_improvement_proposals",
        "tool_version_activation_log",
        "tool_revision_requests",
        "tool_improvement_decisions",
        "idx_messages_one_assistant_per_run",
        "idx_artifacts_run_name_hash",
        "idx_tool_proposals_source_version",
        "corpus_sources",
        "corpus_files",
        "corpus_chunks",
        "code_graph_nodes",
        "code_graph_edges",
        "entity_nodes",
        "entity_edges",
        "tool_definitions",
        "tool_definition_proposals",
        "idx_tool_definitions_active",
        "tool_definition_builds",
        "conversation_projects",
        "idx_tool_definition_builds_active",
    } <= objects

    future_path = tmp_path / "future.db"
    connection = sqlite3.connect(future_path)
    connection.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    connection.execute("INSERT INTO schema_migrations VALUES (999, 'future')")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="newer than supported"):
        await Database(future_path).open()


@pytest.mark.asyncio
async def test_rollback_atomically_selects_prior_immutable_version(tmp_path) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        conversation = await database.create_conversation("Rollback")
        message = await database.add_message(conversation.id, "user", "diagram")
        run = await database.create_run(
            conversation.id, message.id, graph_schema_version="1", model_aliases={}
        )
        report = EvalReportV1(
            passed=True,
            score=1,
            results=[EvalResultV1(case_id="smoke", passed=True)],
        )

        def manifest(version: str, content_hash: str) -> ToolManifestV1:
            return ToolManifestV1(
                slug="reference-architecture-generator",
                name="Reference Architecture Generator",
                description="test",
                version=version,
                entrypoint="runner.py",
                runner_image="image@sha256:" + content_hash,
                risk_level=RiskLevel.R2,
                content_hash=content_hash,
                input_schema={},
                output_schema={},
            )

        tool, first, first_proposal = await database.create_tool_candidate(
            manifest("1.0.0", "a" * 64), report, run.id, "/bundle/a"
        )
        await database.decide_tool_proposal(
            first_proposal.id, ProposalStatus.APPROVED, None, "activate:first"
        )
        second_message = await database.add_message(conversation.id, "user", "revise")
        second_run = await database.create_run(
            conversation.id,
            second_message.id,
            graph_schema_version="1",
            model_aliases={},
        )
        _, second, second_proposal = await database.create_tool_candidate(
            manifest("2.0.0", "b" * 64), report, second_run.id, "/bundle/b"
        )
        await database.decide_tool_proposal(
            second_proposal.id, ProposalStatus.APPROVED, None, "activate:second"
        )
        result = await database.activate_tool_version(
            tool.id,
            first.id,
            action_id="rollback:stable",
            reason="Regression detected",
        )
        replay = await database.activate_tool_version(
            tool.id,
            first.id,
            action_id="rollback:stable",
            reason="ignored replay",
        )
        assert result == replay
        assert result["prior_version_id"] == second.id
        assert (await database.list_tools())[0].active_version_id == first.id
        states = {item.id: item.state for item in await database.list_tool_versions(tool.id)}
        assert states[first.id] == ToolState.ACTIVE
        assert states[second.id] == ToolState.APPROVED
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_tool_improvement_decisions_queue_or_activate_exact_evaluated_revision(
    tmp_path,
) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        conversation = await database.create_conversation("Improve")
        message = await database.add_message(conversation.id, "user", "draw")
        run = await database.create_run(
            conversation.id, message.id, graph_schema_version="1", model_aliases={}
        )
        report = EvalReportV1(
            passed=True,
            score=1,
            results=[EvalResultV1(case_id="smoke", passed=True)],
        )

        def manifest(version: str, content_hash: str) -> ToolManifestV1:
            return ToolManifestV1(
                slug="reference-architecture-generator",
                name="Reference Architecture Generator",
                description="test",
                version=version,
                entrypoint="runner.py",
                runner_image="image@sha256:" + content_hash,
                risk_level=RiskLevel.R2,
                content_hash=content_hash,
                input_schema={},
                output_schema={},
            )

        tool, base, base_proposal = await database.create_tool_candidate(
            manifest("1.0.0", "a" * 64), report, run.id, "/bundle/a"
        )
        await database.decide_tool_proposal(
            base_proposal.id, ProposalStatus.APPROVED, "initial", "activate:base"
        )
        await database.pin_tool_version(
            run.id,
            slug="reference-architecture-generator",
            version_id=base.id,
            version=base.version,
            content_hash=base.content_hash,
        )
        queued_proposal = (
            await database.create_tool_improvements_for_run(run.id, "Use private SQL")
        )[0]
        queued = await database.decide_tool_improvement(
            queued_proposal.id,
            "approve",
            "Turn the correction into a tested revision",
            "improvement:queue",
        )
        replay = await database.decide_tool_improvement(
            queued_proposal.id,
            "approve",
            "Ignored on replay",
            "improvement:queue",
        )
        assert queued == replay
        assert queued["outcome"] == "revision_queued"
        request = await database.get_tool_revision_request(queued["revision_request_id"])
        assert request is not None
        assert request.status == "queued"
        assert request.base_version_id == base.id
        assert (await database.list_tools())[0].active_version_id == base.id

        next_message = await database.add_message(conversation.id, "user", "draw again")
        next_run = await database.create_run(
            conversation.id,
            next_message.id,
            graph_schema_version="1",
            model_aliases={},
        )
        await database.pin_tool_version(
            next_run.id,
            slug="reference-architecture-generator",
            version_id=base.id,
            version=base.version,
            content_hash=base.content_hash,
        )
        activation_proposal = (
            await database.create_tool_improvements_for_run(
                next_run.id, "Keep the database private"
            )
        )[0]
        candidate_message = await database.add_message(conversation.id, "user", "repair")
        candidate_run = await database.create_run(
            conversation.id,
            candidate_message.id,
            graph_schema_version="1",
            model_aliases={},
        )
        _, candidate, _ = await database.create_tool_candidate(
            manifest("1.1.0", "b" * 64), report, candidate_run.id, "/bundle/b"
        )
        activated = await database.decide_tool_improvement(
            activation_proposal.id,
            "approve",
            "The exact revision passed its regression suite",
            "improvement:activate",
            target_version_id=candidate.id,
        )
        assert activated["outcome"] == "revision_activated"
        assert activated["activated_version_id"] == candidate.id
        assert activated["prior_version_id"] == base.id
        assert (await database.list_tools())[0].active_version_id == candidate.id

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            database._connection().execute(
                "UPDATE tool_improvement_decisions SET reason = 'changed'"
            )
    finally:
        await database.close()
