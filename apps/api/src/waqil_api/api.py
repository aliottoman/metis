from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from .asset_library import AssetLibraryError
from .attachment_text import (
    AttachmentExtractionError,
    AttachmentTextTooLargeError,
    extract_attachment_text,
    image_attachment_media_type,
    supports_attachment,
)
from .blob_store import BlobTooLargeError
from .contracts import (
    ApprovalDecisionV1,
    AssetLogsV1,
    AssetStartV1,
    AssetV1,
    CodeGraphLookupV1,
    CodeGraphStatsV1,
    EntityGraphLookupV1,
    EntityGraphStatsV1,
    ConversationCreateV1,
    ConversationProjectV1,
    ConversationV1,
    CorpusConsentDecisionV1,
    CorpusReindexResultV1,
    CorpusSearchV1,
    CorpusSourceCreateV1,
    CorpusSourceV1,
    Decision,
    FeedbackV1,
    HealthV1,
    KnowledgeSnippetV1,
    MemoryDecisionV1,
    MemoryProposalCreateV1,
    MemoryProposalV1,
    NotionConnectionUpdateV1,
    NotionConnectionV1,
    NotionSyncResultV1,
    ModelPreferenceV1,
    ModelPreferenceUpdateV1,
    PersonalProfileV1,
    PersonalProfileUpdateV1,
    ProjectOpenV1,
    ProjectWorkspaceV1,
    MessageAcceptedV1,
    MessageCreateV1,
    MessageV1,
    ProposalDecisionV1,
    ProposalStatus,
    RecoverableRunV1,
    RunStatus,
    RunV1,
    ToolDefinitionBuildV1,
    ToolDefinitionProposalV1,
    ToolDefinitionRecordV1,
    ToolProposalV1,
    ToolImprovementDecisionResultV1,
    ToolImprovementDecisionV1,
    ToolImprovementEvidenceV1,
    ToolImprovementProposalV1,
    ToolV1,
    ToolVersionV1,
    ToolVersionEvidenceV1,
    ToolVersionActivationV1,
    UploadV1,
)
from .control_plane import GRAPH_SCHEMA_VERSION, initial_state
from .embeddings import CohereUnavailable
from .notion import NotionError
from .reference_architecture import ReferenceRunnerError
from .project_workspace import ProjectWorkspaceError
from .runtime import AppRuntime
from .tool_evidence import build_tool_version_evidence


router = APIRouter(prefix="/api/v1")
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


def runtime(request: Request) -> AppRuntime:
    return request.app.state.runtime


def not_found(kind: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{kind} not found")


def _extract_upload_record(record: dict, max_bytes: int) -> str:
    content = Path(record["blob_path"]).read_bytes()
    return extract_attachment_text(
        str(record["filename"]),
        str(record["media_type"]),
        content,
        max_bytes=max_bytes,
    )


@router.get("/health", response_model=HealthV1)
async def health(request: Request) -> HealthV1:
    app = runtime(request)
    database_ok, model_health = await asyncio.gather(
        app.database.ping(), app.model.health()
    )
    runner_available = (
        app.settings.reference_runner_mode == "deterministic"
        or app.settings.reference_sandbox_runner.is_file()
    )
    return HealthV1(
        status="ok"
        if database_ok and app.checkpointer is not None and model_health.get("reachable")
        else "degraded",
        version="0.1.0",
        database=database_ok,
        checkpoints=app.checkpointer is not None,
        model_backend=app.model.name,
        reference_runner=app.settings.reference_runner_mode,
        details={
            "runner_available": runner_available,
            "deep_worker_available": app.deep_worker_factory is not None,
            "model": model_health,
        },
    )


@router.get("/settings/model", response_model=ModelPreferenceV1)
async def get_model_preference(request: Request) -> ModelPreferenceV1:
    return runtime(request).model_preference.load()


@router.put("/settings/model", response_model=ModelPreferenceV1)
async def put_model_preference(
    body: ModelPreferenceUpdateV1, request: Request
) -> ModelPreferenceV1:
    try:
        return runtime(request).model_preference.save(
            body.mode,
            body.model,
            provider=body.provider,
            oci_tools=body.oci_tools,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


# ── Metis local asset library ────────────────────────────────────────────────


@router.get("/assets", response_model=list[AssetV1])
async def list_assets(request: Request) -> list[AssetV1]:
    # Ordinary reads are snapshot-only. New folders are detected solely by the
    # explicit POST /assets/scan action initiated from the Assets page.
    return await runtime(request).assets.list()


@router.post("/assets/scan", response_model=list[AssetV1])
async def scan_assets(request: Request) -> list[AssetV1]:
    try:
        return await runtime(request).assets.scan()
    except AssetLibraryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/assets/{asset_id}/start", response_model=AssetV1)
async def start_asset(
    asset_id: str, request: Request, body: AssetStartV1 | None = None
) -> AssetV1:
    try:
        return await runtime(request).assets.start(asset_id, body.env if body else {})
    except AssetLibraryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/assets/{asset_id}/approval", response_model=AssetV1)
async def approve_asset(asset_id: str, request: Request) -> AssetV1:
    try:
        return await runtime(request).assets.approve(asset_id)
    except AssetLibraryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.delete("/assets/{asset_id}/approval", response_model=AssetV1)
async def revoke_asset_approval(asset_id: str, request: Request) -> AssetV1:
    try:
        return await runtime(request).assets.revoke(asset_id)
    except AssetLibraryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/assets/{asset_id}/stop", response_model=AssetV1)
async def stop_asset(asset_id: str, request: Request) -> AssetV1:
    try:
        return await runtime(request).assets.stop(asset_id)
    except AssetLibraryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/assets/{asset_id}/logs", response_model=AssetLogsV1)
async def asset_logs(asset_id: str, request: Request) -> AssetLogsV1:
    try:
        return await runtime(request).assets.logs(asset_id)
    except AssetLibraryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ── Explicit project workspaces ─────────────────────────────────────────────


@router.get("/projects", response_model=list[ProjectWorkspaceV1])
async def list_projects(request: Request) -> list[ProjectWorkspaceV1]:
    projects = runtime(request).projects
    return await projects.list() if projects is not None else []


@router.post("/projects/{project_id}/open", response_model=ProjectWorkspaceV1)
async def open_project(
    project_id: str, body: ProjectOpenV1, request: Request
) -> ProjectWorkspaceV1:
    app = runtime(request)
    if app.projects is None:
        raise HTTPException(status_code=503, detail="project workspaces are unavailable")
    try:
        # `mode` is persisted with the conversation on first send. Opening is a
        # project-level operation and Grok bootstraps only if local context is absent.
        return await app.projects.open(project_id)
    except (AssetLibraryError, ProjectWorkspaceError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/conversations", response_model=ConversationV1, status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    body: ConversationCreateV1, request: Request
) -> ConversationV1:
    return await runtime(request).database.create_conversation(body.title)


@router.get("/conversations", response_model=list[ConversationV1])
async def list_conversations(
    request: Request, limit: Annotated[int, Query(ge=1, le=200)] = 100
) -> list[ConversationV1]:
    return await runtime(request).database.list_conversations(limit)


@router.get("/conversations/{conversation_id}", response_model=ConversationV1)
async def get_conversation(conversation_id: str, request: Request) -> ConversationV1:
    value = await runtime(request).database.get_conversation(conversation_id)
    if value is None:
        raise not_found("conversation")
    return value


@router.get(
    "/conversations/{conversation_id}/project",
    response_model=ConversationProjectV1 | None,
)
async def get_conversation_project(
    conversation_id: str, request: Request
) -> ConversationProjectV1 | None:
    app = runtime(request)
    if await app.database.get_conversation(conversation_id) is None:
        raise not_found("conversation")
    return await app.database.get_conversation_project(conversation_id)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageV1])
async def list_messages(conversation_id: str, request: Request) -> list[MessageV1]:
    app = runtime(request)
    if await app.database.get_conversation(conversation_id) is None:
        raise not_found("conversation")
    return await app.database.list_messages(conversation_id)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageAcceptedV1,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_message(
    conversation_id: str, body: MessageCreateV1, request: Request
) -> MessageAcceptedV1:
    app = runtime(request)
    if await app.database.get_conversation(conversation_id) is None:
        raise not_found("conversation")
    if not await app.database.validate_upload_ids(body.attachment_ids):
        raise HTTPException(status_code=422, detail="one or more attachment IDs are invalid")
    attachment_records = await asyncio.gather(
        *(app.database.get_upload_record(item) for item in body.attachment_ids)
    )
    try:
        extracted = await asyncio.gather(*(
            asyncio.to_thread(
                _extract_upload_record,
                item,
                app.settings.max_text_attachment_bytes,
            )
            for item in attachment_records
            if item is not None
        ))
    except AttachmentTextTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (AttachmentExtractionError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"attachment text could not be read: {exc}") from exc
    total_attachment_bytes = sum(len(item.encode("utf-8")) for item in extracted)
    if total_attachment_bytes > app.settings.max_text_attachment_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "aggregate attachment text exceeds the v1 context budget of "
                f"{app.settings.max_text_attachment_bytes} bytes"
            ),
        )
    model_aliases = app.model_preference.resolve_aliases()
    model_aliases["_knowledge_scope"] = body.knowledge_scope
    project_fields_supplied = bool(
        {"project_id", "project_mode"} & body.model_fields_set
    )
    if project_fields_supplied and (body.project_id is None) != (body.project_mode is None):
        raise HTTPException(
            status_code=422,
            detail="project_id and project_mode must be supplied together",
        )
    project_session = None
    if project_fields_supplied and body.project_id and body.project_mode:
        if app.projects is None:
            raise HTTPException(status_code=503, detail="project workspaces are unavailable")
        if (
            body.project_mode == "grok_continuous"
            and not app.model_preference.oci_available
            and app.settings.model_backend != "deterministic"
        ):
            raise HTTPException(status_code=409, detail="continuous Grok mode is unavailable")
        try:
            await app.projects.context(body.project_id)
        except (AssetLibraryError, ProjectWorkspaceError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        project_session = await app.database.set_conversation_project(
            conversation_id, body.project_id, body.project_mode
        )
    elif project_fields_supplied:
        await app.database.clear_conversation_project(conversation_id)
    else:
        project_session = await app.database.get_conversation_project(conversation_id)
    if project_session is not None:
        if (
            project_session.mode == "grok_continuous"
            and not app.model_preference.oci_available
            and app.settings.model_backend != "deterministic"
        ):
            raise HTTPException(status_code=409, detail="continuous Grok mode is unavailable")
        model_aliases.update(
            {
                "_project_id": project_session.project_id,
                "_project_mode": project_session.mode,
                "_provider": "oci"
                if project_session.mode == "grok_continuous"
                else "local",
            }
        )
    user_message = await app.database.add_message(
        conversation_id, "user", body.content, body.attachment_ids
    )
    run = await app.database.create_run(
        conversation_id,
        user_message.id,
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        model_aliases=model_aliases,
    )
    await app.database.link_message_run(user_message.id, run.id)
    assert app.control_plane is not None
    await app.control_plane.submit(
        initial_state(
            run_id=run.id,
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            prompt=body.content,
            attachment_ids=body.attachment_ids,
            model_aliases=model_aliases,
        )
    )
    return MessageAcceptedV1(
        message_id=user_message.id, run_id=run.id, status=RunStatus.QUEUED
    )


@router.get("/runs", response_model=list[RecoverableRunV1])
async def list_runs(
    request: Request,
    run_status: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[RecoverableRunV1]:
    app = runtime(request)
    runs = await app.database.list_runs(run_status, limit=limit)
    return [
        RecoverableRunV1(
            run=run,
            approval=await app.database.get_pending_approval(run.id)
            if run.status == RunStatus.AWAITING_APPROVAL
            else None,
        )
        for run in runs
    ]


@router.get("/runs/{run_id}", response_model=RunV1)
async def get_run(run_id: str, request: Request) -> RunV1:
    value = await runtime(request).database.get_run(run_id)
    if value is None:
        raise not_found("run")
    return value


def _sse(event_type: str, event_id: str | int | None, data: dict) -> bytes:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines.extend(f"data: {line}" for line in payload.splitlines() or [""])
    return ("\n".join(lines) + "\n\n").encode("utf-8")


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    after: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    app = runtime(request)
    if await app.database.get_run(run_id) is None:
        raise not_found("run")

    async def body() -> AsyncIterator[bytes]:
        async for event in app.events.stream(run_id, after=after):
            if await request.is_disconnected():
                return
            if event is None:
                yield b": heartbeat\n\n"
            else:
                yield _sse(
                    event.type,
                    event.sequence,
                    event.model_dump(mode="json"),
                )

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/decisions", response_model=RunV1)
async def decide_run(
    run_id: str, body: ApprovalDecisionV1, request: Request
) -> RunV1:
    app = runtime(request)
    run = await app.database.get_run(run_id)
    if run is None:
        raise not_found("run")
    if run.status != RunStatus.AWAITING_APPROVAL:
        raise HTTPException(status_code=409, detail="run is not awaiting approval")
    approval = await app.database.get_pending_approval(run_id)
    if approval is None:
        raise HTTPException(status_code=409, detail="run has no pending approval")
    if body.approval_id is not None and body.approval_id != approval.id:
        raise HTTPException(status_code=409, detail="approval ID does not match pending action")
    changed = await app.database.record_approval_decision(
        approval.id, body.decision, body.reason
    )
    if not changed:
        raise HTTPException(status_code=409, detail="approval has already been decided")
    await app.events.emit(
        run_id,
        run.conversation_id,
        "approval.decided",
        {
            "approval_id": approval.id,
            "decision": body.decision,
            "reason": body.reason,
        },
    )
    assert app.control_plane is not None
    await app.control_plane.resume(
        run_id,
        run.conversation_id,
        ApprovalDecisionV1(
            approval_id=approval.id, decision=body.decision, reason=body.reason
        ),
    )
    return (await app.database.get_run(run_id))  # type: ignore[return-value]


@router.post("/runs/{run_id}/cancel", response_model=RunV1)
async def cancel_run(run_id: str, request: Request) -> RunV1:
    app = runtime(request)
    run = await app.database.get_run(run_id)
    if run is None:
        raise not_found("run")
    assert app.control_plane is not None
    changed = await app.control_plane.cancel(run_id)
    if not changed and run.status not in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }:
        raise HTTPException(status_code=409, detail="run could not be cancelled")
    return (await app.database.get_run(run_id))  # type: ignore[return-value]


@router.post("/runs/{run_id}/feedback", status_code=status.HTTP_201_CREATED)
async def add_feedback(run_id: str, body: FeedbackV1, request: Request) -> dict:
    if body.run_id != run_id:
        raise HTTPException(status_code=422, detail="body run_id must match URL")
    app = runtime(request)
    run = await app.database.get_run(run_id)
    if run is None:
        raise not_found("run")
    feedback_id = await app.database.add_feedback(run_id, body.rating, body.correction)
    memory_proposal_id = None
    tool_improvement_ids: list[str] = []
    if body.correction:
        proposal = await app.database.create_memory_proposal(
            "project", body.correction, run_id, confidence=0.8
        )
        memory_proposal_id = proposal.id
        improvements = await app.database.create_tool_improvements_for_run(
            run_id, body.correction
        )
        tool_improvement_ids = [item.id for item in improvements]
    return {
        "id": feedback_id,
        "memory_proposal_id": memory_proposal_id,
        "tool_improvement_proposal_ids": tool_improvement_ids,
    }


@router.post("/uploads", response_model=UploadV1, status_code=status.HTTP_201_CREATED)
async def upload_file(
    request: Request, file: Annotated[UploadFile, File(...)]
) -> UploadV1:
    app = runtime(request)
    raw_name = Path(file.filename or "upload.bin").name
    raw_lowered_name = raw_name.strip().rstrip(".").casefold()
    if (
        raw_lowered_name == ".env"
        or raw_lowered_name.startswith(".env.")
        or Path(raw_lowered_name).suffix == ".env"
    ):
        raise HTTPException(
            status_code=415,
            detail="secret-bearing environment files are not accepted in v1",
        )
    filename = _SAFE_FILENAME.sub("_", raw_name).strip(". ")[:240] or "upload.bin"
    if any(filename.lower().endswith(suffix) for suffix in _ARCHIVE_SUFFIXES):
        raise HTTPException(status_code=415, detail="archive uploads are not supported in v1")
    media_type = (file.content_type or "application/octet-stream").split(";", 1)[0].lower()
    if not supports_attachment(filename, media_type):
        raise HTTPException(
            status_code=415,
            detail=(
                "attachments may be text, source code, PDF, DOCX, PPTX, XLSX, "
                "PNG, JPEG, WebP, or GIF"
            ),
        )
    media_type = image_attachment_media_type(filename, media_type) or media_type

    async def chunks() -> AsyncIterator[bytes]:
        while chunk := await file.read(64 * 1024):
            yield chunk

    def validate_staged(path: Path) -> None:
        extract_attachment_text(
            filename,
            media_type,
            path.read_bytes(),
            max_bytes=app.settings.max_text_attachment_bytes,
        )

    try:
        blob = await app.blobs.put_stream(
            chunks(),
            max_bytes=app.settings.max_upload_bytes,
            validate_staged=validate_staged,
        )
    except BlobTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except AttachmentTextTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except AttachmentExtractionError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        await file.close()
    return await app.database.create_upload(
        blob.sha256,
        filename,
        media_type,
        blob.size,
        str(blob.path),
    )


@router.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str, request: Request) -> FileResponse:
    app = runtime(request)
    record = await app.database.get_artifact_record(artifact_id)
    if record is None:
        raise not_found("artifact")
    path = app.blobs.path_for(record["sha256"])
    if not path.is_file():
        raise not_found("artifact content")
    return FileResponse(
        path,
        media_type=record["media_type"],
        filename=record["filename"],
        content_disposition_type="attachment",
    )


@router.get("/tools", response_model=list[ToolV1])
async def list_tools(request: Request) -> list[ToolV1]:
    return await runtime(request).database.list_tools()


@router.get("/tools/{tool_id}/versions", response_model=list[ToolVersionV1])
async def list_tool_versions(tool_id: str, request: Request) -> list[ToolVersionV1]:
    return await runtime(request).database.list_tool_versions(tool_id)


@router.get(
    "/tools/{tool_id}/versions/{version_id}/evidence",
    response_model=ToolVersionEvidenceV1,
)
async def get_tool_version_evidence(
    tool_id: str, version_id: str, request: Request
) -> ToolVersionEvidenceV1:
    app = runtime(request)
    record = await app.database.get_tool_version_record(tool_id, version_id)
    if record is None:
        raise not_found("tool version")
    active = await app.database.get_active_tool_version_record(tool_id)
    try:
        return await asyncio.to_thread(
            build_tool_version_evidence,
            record,
            verifier=app.reference_runner,
            compared_record=active,
        )
    except (ValueError, ReferenceRunnerError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/tools/{tool_id}/versions/{version_id}/activate")
async def activate_tool_version(
    tool_id: str,
    version_id: str,
    body: ToolVersionActivationV1,
    request: Request,
) -> dict:
    app = runtime(request)
    record = await app.database.get_tool_version_record(tool_id, version_id)
    if record is None:
        raise not_found("tool version")
    manifest = record["manifest"]
    image_ref = manifest.get("runner_image")
    if not image_ref:
        raise HTTPException(status_code=409, detail="version has no pinned runner image")
    try:
        app.reference_runner.verify_snapshot(
            record["bundle_path"], record["content_hash"], image_ref
        )
        return await app.database.activate_tool_version(
            tool_id,
            version_id,
            action_id=(
                f"manual-version-activation:{tool_id}:{version_id}:"
                f"{body.idempotency_key}"
            ),
            reason=body.reason,
        )
    except KeyError as exc:
        raise not_found("tool version") from exc
    except (ValueError, ReferenceRunnerError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# Read-only registry browser and gate inboxes. Both gates are decided through
# the generic run-approval flow, so there are no definition-specific endpoints.


@router.get("/tool-definitions", response_model=list[ToolDefinitionRecordV1])
async def list_tool_definitions(request: Request) -> list[ToolDefinitionRecordV1]:
    """The live tool registry with host-derived state (capability profiles,
    budgets, runnable/buildable/disabled). Returns [] when no registry is wired."""
    app = runtime(request)
    if app.registry is None:
        return []
    return await app.registry.records()


@router.get(
    "/tool-definition-proposals", response_model=list[ToolDefinitionProposalV1]
)
async def list_tool_definition_proposals(
    request: Request,
    proposal_status: Annotated[str | None, Query(alias="status")] = None,
) -> list[ToolDefinitionProposalV1]:
    """Gate-1 inbox: drafted definitions and their approve/reject history."""
    return await runtime(request).database.list_tool_definition_proposals(proposal_status)


@router.get("/tool-definition-builds", response_model=list[ToolDefinitionBuildV1])
async def list_tool_definition_builds(
    request: Request,
    build_status: Annotated[str | None, Query(alias="status")] = None,
) -> list[ToolDefinitionBuildV1]:
    """Gate-2 inbox for declarative tools: evaluated builds and their evidence."""
    return await runtime(request).database.list_tool_definition_builds(build_status)


@router.get("/tool-proposals", response_model=list[ToolProposalV1])
async def list_tool_proposals(
    request: Request,
    proposal_status: Annotated[str | None, Query(alias="status")] = None,
) -> list[ToolProposalV1]:
    return await runtime(request).database.list_tool_proposals(proposal_status)


@router.get(
    "/tool-improvement-proposals", response_model=list[ToolImprovementProposalV1]
)
async def list_tool_improvement_proposals(
    request: Request,
    proposal_status: Annotated[str | None, Query(alias="status")] = ProposalStatus.PENDING,
) -> list[ToolImprovementProposalV1]:
    return await runtime(request).database.list_tool_improvements(proposal_status)


@router.get(
    "/tool-proposals/{proposal_id}/evidence",
    response_model=ToolVersionEvidenceV1,
)
async def get_tool_proposal_evidence(
    proposal_id: str, request: Request
) -> ToolVersionEvidenceV1:
    app = runtime(request)
    proposal = await app.database.get_tool_proposal(proposal_id)
    if proposal is None:
        raise not_found("tool proposal")
    record = await app.database.get_tool_version_record(
        proposal.tool_id, proposal.tool_version_id
    )
    if record is None:
        raise not_found("tool version")
    active = await app.database.get_active_tool_version_record(proposal.tool_id)
    try:
        return await asyncio.to_thread(
            build_tool_version_evidence,
            record,
            verifier=app.reference_runner,
            compared_record=active,
        )
    except (ValueError, ReferenceRunnerError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/tool-improvement-proposals/{proposal_id}/evidence",
    response_model=ToolImprovementEvidenceV1,
)
async def get_tool_improvement_evidence(
    proposal_id: str, request: Request
) -> ToolImprovementEvidenceV1:
    app = runtime(request)
    proposal = await app.database.get_tool_improvement(proposal_id)
    if proposal is None:
        raise not_found("tool improvement proposal")
    base_record = await app.database.get_tool_version_record(
        proposal.tool_id, proposal.tool_version_id
    )
    if base_record is None:
        raise not_found("base tool version")
    eligible = await app.database.list_eligible_tool_revision_records(proposal_id)
    try:
        base_evidence = await asyncio.to_thread(
            build_tool_version_evidence,
            base_record,
            verifier=app.reference_runner,
        )
        revision_evidence = [
            await asyncio.to_thread(
                build_tool_version_evidence,
                record,
                verifier=app.reference_runner,
                compared_record=base_record,
            )
            for record in eligible
        ]
    except (ValueError, ReferenceRunnerError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ToolImprovementEvidenceV1(
        proposal=proposal,
        base_version=base_evidence,
        eligible_revisions=revision_evidence,
    )


@router.post(
    "/tool-improvement-proposals/{proposal_id}/decision",
    response_model=ToolImprovementDecisionResultV1,
)
async def decide_tool_improvement(
    proposal_id: str, body: ToolImprovementDecisionV1, request: Request
) -> ToolImprovementDecisionResultV1:
    app = runtime(request)
    proposal = await app.database.get_tool_improvement(proposal_id)
    if proposal is None:
        raise not_found("tool improvement proposal")

    if body.target_version_id is not None:
        candidates = await app.database.list_eligible_tool_revision_records(proposal_id)
        target = next(
            (item for item in candidates if item["id"] == body.target_version_id), None
        )
        if target is None:
            raise HTTPException(
                status_code=409,
                detail="target must be an immutable evaluated revision created after the proposal",
            )
        try:
            await asyncio.to_thread(
                build_tool_version_evidence,
                target,
                verifier=app.reference_runner,
                compared_record=await app.database.get_tool_version_record(
                    proposal.tool_id, proposal.tool_version_id
                ),
            )
        except (ValueError, ReferenceRunnerError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        result = await app.database.decide_tool_improvement(
            proposal_id,
            body.decision,
            body.reason,
            f"tool-improvement:{proposal_id}:{body.idempotency_key}",
            target_version_id=body.target_version_id,
        )
    except KeyError as exc:
        raise not_found("tool improvement proposal") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    decided = await app.database.get_tool_improvement(proposal_id)
    if decided is None:
        raise RuntimeError("tool improvement decision was not persisted")
    revision_request = (
        await app.database.get_tool_revision_request(result["revision_request_id"])
        if result.get("revision_request_id")
        else None
    )
    return ToolImprovementDecisionResultV1(
        proposal=decided,
        outcome=result["outcome"],
        revision_request=revision_request,
        activated_version_id=result.get("activated_version_id"),
        prior_version_id=result.get("prior_version_id"),
    )


async def _decide_tool_from_endpoint(
    request: Request,
    proposal_id: str,
    decision: Literal["approve", "reject"],
    reason: str | None,
) -> ToolProposalV1:
    app = runtime(request)
    proposal = await app.database.get_tool_proposal(proposal_id)
    if proposal is None:
        raise not_found("tool proposal")
    run = await app.database.get_run(proposal.source_run_id)
    approval = (
        await app.database.get_pending_approval(proposal.source_run_id) if run else None
    )
    if run and run.status == RunStatus.AWAITING_APPROVAL and approval:
        await decide_run(
            run.id,
            ApprovalDecisionV1(
                approval_id=approval.id, decision=decision, reason=reason
            ),
            request,
        )
        for _ in range(100):
            current = await app.database.get_tool_proposal(proposal_id)
            if current and current.status != ProposalStatus.PENDING:
                return current
            await asyncio.sleep(0.01)
    else:
        mapped = (
            ProposalStatus.APPROVED if decision == "approve" else ProposalStatus.REJECTED
        )
        try:
            await app.database.decide_tool_proposal(
                proposal_id,
                mapped,
                reason,
                f"proposal-api:{proposal_id}:{decision}",
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return (await app.database.get_tool_proposal(proposal_id))  # type: ignore[return-value]


@router.post("/tool-proposals/{proposal_id}/approve", response_model=ToolProposalV1)
async def approve_tool_proposal(
    proposal_id: str, body: ProposalDecisionV1, request: Request
) -> ToolProposalV1:
    return await _decide_tool_from_endpoint(request, proposal_id, "approve", body.reason)


@router.post("/tool-proposals/{proposal_id}/reject", response_model=ToolProposalV1)
async def reject_tool_proposal(
    proposal_id: str, body: ProposalDecisionV1, request: Request
) -> ToolProposalV1:
    return await _decide_tool_from_endpoint(request, proposal_id, "reject", body.reason)


@router.get("/memory/proposals", response_model=list[MemoryProposalV1])
async def list_memory_proposals(
    request: Request,
    proposal_status: Annotated[str | None, Query(alias="status")] = ProposalStatus.PENDING,
) -> list[MemoryProposalV1]:
    return await runtime(request).database.list_memory_proposals(proposal_status)


@router.post(
    "/memory/proposals",
    response_model=MemoryProposalV1,
    status_code=status.HTTP_201_CREATED,
)
async def create_memory_proposal(
    body: MemoryProposalCreateV1, request: Request
) -> MemoryProposalV1:
    app = runtime(request)
    if body.source_run_id and await app.database.get_run(body.source_run_id) is None:
        raise HTTPException(status_code=422, detail="source run does not exist")
    return await app.database.create_memory_proposal(
        body.kind,
        body.content.strip(),
        body.source_run_id,
        confidence=1.0,
    )


@router.post("/memory/proposals/{proposal_id}/decision", response_model=MemoryProposalV1)
async def decide_memory_proposal(
    proposal_id: str, body: MemoryDecisionV1, request: Request
) -> MemoryProposalV1:
    try:
        mapped = {
            Decision.APPROVE: ProposalStatus.APPROVED,
            Decision.REJECT: ProposalStatus.REJECTED,
            Decision.DRAFT: ProposalStatus.DRAFT,
        }[body.decision]
        return await runtime(request).database.decide_memory_proposal(
            proposal_id, mapped, body.reason
        )
    except KeyError as exc:
        raise not_found("memory proposal") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ── Personal knowledge: Tier-1 corpus (RAG) + Tier-0 profile ─────────────────


@router.get("/corpus/status")
async def corpus_status(request: Request) -> dict:
    app_runtime = runtime(request)
    return {
        "available": app_runtime.corpus.available(),
        "cloud_embeddings_enabled": app_runtime.settings.allow_cloud_embeddings,
        "embed_model": app_runtime.settings.oci_embed_model,
        "rerank_model": app_runtime.settings.oci_rerank_model,
        "entity_graph_enabled": app_runtime.settings.corpus_entity_graph,
    }


@router.get("/corpus/sources", response_model=list[CorpusSourceV1])
async def list_corpus_sources(request: Request) -> list[CorpusSourceV1]:
    return await runtime(request).corpus.list_sources()


@router.get("/corpus/notion", response_model=NotionConnectionV1)
async def get_notion_connection(request: Request) -> NotionConnectionV1:
    return await runtime(request).notion.status()


@router.put("/corpus/notion", response_model=NotionConnectionV1)
async def configure_notion(
    body: NotionConnectionUpdateV1, request: Request
) -> NotionConnectionV1:
    try:
        return await runtime(request).notion.configure(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/corpus/notion/sync", response_model=NotionSyncResultV1)
async def sync_notion(request: Request) -> NotionSyncResultV1:
    try:
        return await runtime(request).notion.sync()
    except NotionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CohereUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/corpus/sources",
    response_model=CorpusSourceV1,
    status_code=status.HTTP_201_CREATED,
)
async def create_corpus_source(
    body: CorpusSourceCreateV1, request: Request
) -> CorpusSourceV1:
    try:
        return await runtime(request).corpus.register_source(
            body.root_path, body.label, body.kind
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/corpus/sources/{source_id}", response_model=CorpusSourceV1)
async def get_corpus_source(source_id: str, request: Request) -> CorpusSourceV1:
    source = await runtime(request).corpus.get_source(source_id)
    if source is None:
        raise not_found("corpus source")
    return source


@router.post("/corpus/sources/{source_id}/consent", response_model=CorpusSourceV1)
async def decide_corpus_consent(
    source_id: str, body: CorpusConsentDecisionV1, request: Request
) -> CorpusSourceV1:
    try:
        return await runtime(request).corpus.set_consent(
            source_id, body.consent, body.reason
        )
    except KeyError as exc:
        raise not_found("corpus source") from exc


@router.post(
    "/corpus/sources/{source_id}/reindex", response_model=CorpusReindexResultV1
)
async def reindex_corpus_source(
    source_id: str, request: Request
) -> CorpusReindexResultV1:
    try:
        return await runtime(request).corpus.index_source(source_id)
    except KeyError as exc:
        raise not_found("corpus source") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CohereUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/corpus/sources/{source_id}")
async def delete_corpus_source(source_id: str, request: Request) -> dict:
    deleted = await runtime(request).corpus.delete_source(source_id)
    if not deleted:
        raise not_found("corpus source")
    return {"id": source_id, "deleted": True}


@router.post("/corpus/search", response_model=list[KnowledgeSnippetV1])
async def search_corpus(
    body: CorpusSearchV1, request: Request
) -> list[KnowledgeSnippetV1]:
    try:
        return await runtime(request).corpus.retrieve(body.query, body.limit)
    except CohereUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a CORS-less 500
        # An unhandled error here 500s without CORS headers, which the browser reports
        # as an opaque failure, so return a real message instead.
        raise HTTPException(
            status_code=502, detail=f"retrieval failed: {str(exc)[:300]}"
        ) from exc


@router.get("/corpus/graph/stats", response_model=CodeGraphStatsV1)
async def code_graph_stats(request: Request) -> CodeGraphStatsV1:
    """Counts over the deterministic code graph. Local-only — no cloud call."""
    return await runtime(request).corpus.graph_stats()


@router.get("/corpus/graph/symbol/{name}", response_model=CodeGraphLookupV1)
async def code_graph_symbol(name: str, request: Request) -> CodeGraphLookupV1:
    """Definitions, callers, and callees for one symbol name (one-hop, by name)."""
    return await runtime(request).corpus.graph_lookup(name)


@router.get("/corpus/entities/stats", response_model=EntityGraphStatsV1)
async def entity_graph_stats(request: Request) -> EntityGraphStatsV1:
    """Counts over the cloud-extracted entity graph (Stage 2). Empty unless
    entity extraction is enabled and a prose source has been indexed."""
    return await runtime(request).corpus.entity_stats()


@router.get("/corpus/entities/{name}", response_model=EntityGraphLookupV1)
async def entity_graph_lookup(name: str, request: Request) -> EntityGraphLookupV1:
    """Kinds and one-hop relationships for one entity name (both directions)."""
    return await runtime(request).corpus.entity_lookup(name)


@router.get("/profile", response_model=PersonalProfileV1)
async def get_profile(request: Request) -> PersonalProfileV1:
    return runtime(request).profile.load()


@router.put("/profile", response_model=PersonalProfileV1)
async def put_profile(
    body: PersonalProfileUpdateV1, request: Request
) -> PersonalProfileV1:
    return runtime(request).profile.save(body.content)
