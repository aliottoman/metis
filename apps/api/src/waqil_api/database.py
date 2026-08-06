from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from .contracts import (
    PROJECT_MODES,
    ApprovalRequestV1,
    ConversationProjectV1,
    ConversationV1,
    CorpusSourceV1,
    EvalReportV1,
    MemoryProposalV1,
    MessageV1,
    ProposalStatus,
    RiskLevel,
    RunEventV1,
    RunStatus,
    RunV1,
    ToolDefinitionBuildV1,
    ToolDefinitionProposalV1,
    ToolDefinitionV1,
    ToolManifestV1,
    ToolRevisionRequestV1,
    ToolImprovementProposalV1,
    ToolProposalV1,
    ToolState,
    ToolV1,
    ToolVersionV1,
    UploadV1,
)

T = TypeVar("T")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def memory_content_hash(content: str) -> str:
    """Binds a stored vector to the exact text it was built from.

    Editing a memory must invalidate its embedding; otherwise the old wording
    keeps deciding whether the new wording is retrieved.
    """
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _loads(value: str | None, default: T) -> Any | T:
    return default if value is None else json.loads(value)


def _corpus_source(row: sqlite3.Row) -> CorpusSourceV1:
    """Map a corpus_sources row to its public contract, dropping internal columns."""
    return CorpusSourceV1(
        id=row["id"],
        root_path=row["root_path"],
        label=row["label"],
        kind=row["kind"],
        provider=row["provider"] if "provider" in row.keys() else "local",
        consent=bool(row["consent"]),
        status=row["status"],
        file_count=row["file_count"],
        chunk_count=row["chunk_count"],
        last_indexed_at=row["last_indexed_at"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size INTEGER NOT NULL CHECK(size >= 0),
    blob_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    attachments_json TEXT NOT NULL DEFAULT '[]',
    run_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_one_assistant_per_run
    ON messages(run_id) WHERE role = 'assistant' AND run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL REFERENCES messages(id),
    status TEXT NOT NULL,
    graph_schema_version TEXT NOT NULL,
    model_aliases_json TEXT NOT NULL,
    prompt_versions_json TEXT NOT NULL,
    tool_versions_json TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size INTEGER NOT NULL CHECK(size >= 0),
    blob_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_run_name_hash
    ON artifacts(run_id, filename, sha256);

CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    active_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_versions (
    id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    state TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    bundle_path TEXT NOT NULL,
    eval_report_json TEXT,
    source_run_id TEXT REFERENCES runs(id),
    created_at TEXT NOT NULL,
    UNIQUE(tool_id, version),
    UNIQUE(tool_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_tool_versions_tool ON tool_versions(tool_id, created_at);

CREATE TABLE IF NOT EXISTS tool_proposals (
    id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    tool_version_id TEXT NOT NULL REFERENCES tool_versions(id) ON DELETE CASCADE,
    source_run_id TEXT NOT NULL REFERENCES runs(id),
    status TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    summary TEXT NOT NULL,
    decision_reason TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_proposals_status
    ON tool_proposals(status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_proposals_source_version
    ON tool_proposals(source_run_id, tool_version_id);

CREATE TABLE IF NOT EXISTS tool_improvement_proposals (
    id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL REFERENCES runs(id),
    tool_id TEXT NOT NULL REFERENCES tools(id),
    tool_version_id TEXT NOT NULL REFERENCES tool_versions(id),
    content_hash TEXT NOT NULL,
    correction TEXT NOT NULL,
    regression_eval_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_improvements_status
    ON tool_improvement_proposals(status, created_at);

CREATE TABLE IF NOT EXISTS tool_version_activation_log (
    id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL UNIQUE,
    tool_id TEXT NOT NULL REFERENCES tools(id),
    target_version_id TEXT NOT NULL REFERENCES tool_versions(id),
    prior_version_id TEXT REFERENCES tool_versions(id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    proposal_id TEXT REFERENCES tool_proposals(id),
    action_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    decision_json TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id, status);

CREATE TABLE IF NOT EXISTS idempotency_actions (
    action_id TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source_run_id TEXT REFERENCES runs(id),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memory_items',
    content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory_items BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS memory_proposals (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source_run_id TEXT REFERENCES runs(id),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL,
    decision_reason TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_proposals_status
    ON memory_proposals(status, created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rating TEXT NOT NULL,
    correction TEXT,
    created_at TEXT NOT NULL
);
"""

# Idempotent so a fresh database uses the baseline above while a real v1
# database receives every later object transactionally.
SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_improvement_proposals (
    id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL REFERENCES runs(id),
    tool_id TEXT NOT NULL REFERENCES tools(id),
    tool_version_id TEXT NOT NULL REFERENCES tool_versions(id),
    content_hash TEXT NOT NULL,
    correction TEXT NOT NULL,
    regression_eval_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_improvements_status
    ON tool_improvement_proposals(status, created_at);

CREATE TABLE IF NOT EXISTS tool_version_activation_log (
    id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL UNIQUE,
    tool_id TEXT NOT NULL REFERENCES tools(id),
    target_version_id TEXT NOT NULL REFERENCES tool_versions(id),
    prior_version_id TEXT REFERENCES tool_versions(id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_one_assistant_per_run
    ON messages(run_id) WHERE role = 'assistant' AND run_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_run_name_hash
    ON artifacts(run_id, filename, sha256);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_proposals_source_version
    ON tool_proposals(source_run_id, tool_version_id);
"""

SCHEMA_V3 = """
ALTER TABLE tool_improvement_proposals ADD COLUMN decision_reason TEXT;
ALTER TABLE tool_improvement_proposals ADD COLUMN decided_at TEXT;
ALTER TABLE tool_improvement_proposals ADD COLUMN outcome TEXT;
ALTER TABLE tool_improvement_proposals ADD COLUMN revision_request_id TEXT;
ALTER TABLE tool_improvement_proposals ADD COLUMN target_version_id TEXT;

CREATE TABLE IF NOT EXISTS tool_revision_requests (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE REFERENCES tool_improvement_proposals(id),
    tool_id TEXT NOT NULL REFERENCES tools(id),
    base_version_id TEXT NOT NULL REFERENCES tool_versions(id),
    base_content_hash TEXT NOT NULL,
    correction TEXT NOT NULL,
    regression_eval_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status = 'queued'),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_improvement_decisions (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE REFERENCES tool_improvement_proposals(id),
    action_id TEXT NOT NULL UNIQUE,
    decision TEXT NOT NULL CHECK(decision IN ('approve','reject')),
    reason TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('revision_queued','revision_activated','rejected')),
    target_version_id TEXT REFERENCES tool_versions(id),
    revision_request_id TEXT REFERENCES tool_revision_requests(id),
    prior_version_id TEXT REFERENCES tool_versions(id),
    created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS tool_improvement_decisions_no_update
BEFORE UPDATE ON tool_improvement_decisions BEGIN
    SELECT RAISE(ABORT, 'tool improvement decisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS tool_improvement_decisions_no_delete
BEFORE DELETE ON tool_improvement_decisions BEGIN
    SELECT RAISE(ABORT, 'tool improvement decisions are immutable');
END;
"""

# Corpus tables. A source stays unconsented until the user grants it, and the
# indexer refuses to send text for any unconsented source.
SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS corpus_sources (
    id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('code','docs','notes','mixed')),
    consent INTEGER NOT NULL DEFAULT 0 CHECK(consent IN (0,1)),
    consent_reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','indexing','indexed','error','revoked')),
    file_count INTEGER NOT NULL DEFAULT 0 CHECK(file_count >= 0),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
    embed_model TEXT,
    last_indexed_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_files (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    lang TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
    indexed_at TEXT NOT NULL,
    UNIQUE(source_id, rel_path)
);
CREATE INDEX IF NOT EXISTS idx_corpus_files_source ON corpus_files(source_id);

CREATE TABLE IF NOT EXISTS corpus_chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES corpus_files(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    symbol TEXT,
    start_line INTEGER,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL CHECK(dim > 0),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_source ON corpus_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_file ON corpus_chunks(file_id);
"""

# Code graph, derived locally and keyed by file_id so it replaces incrementally
# and cascade-deletes with the file, source, or consent.
SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS code_graph_nodes (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES corpus_files(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('module','class','function','method')),
    name TEXT NOT NULL,
    qualname TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cg_nodes_source ON code_graph_nodes(source_id);
CREATE INDEX IF NOT EXISTS idx_cg_nodes_file ON code_graph_nodes(file_id);
CREATE INDEX IF NOT EXISTS idx_cg_nodes_name ON code_graph_nodes(name);
CREATE INDEX IF NOT EXISTS idx_cg_nodes_qualname ON code_graph_nodes(qualname);

CREATE TABLE IF NOT EXISTS code_graph_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES corpus_files(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('contains','imports','calls')),
    src TEXT NOT NULL,
    dst_name TEXT NOT NULL,
    dst_raw TEXT NOT NULL,
    line INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cg_edges_source ON code_graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_cg_edges_file ON code_graph_edges(file_id);
CREATE INDEX IF NOT EXISTS idx_cg_edges_src ON code_graph_edges(src);
CREATE INDEX IF NOT EXISTS idx_cg_edges_dst ON code_graph_edges(dst_name);
CREATE INDEX IF NOT EXISTS idx_cg_edges_kind ON code_graph_edges(kind);
"""

# Entity graph. Kept separate from the code graph because its provenance is
# cloud extraction; same fail-closed cascade lifecycle.
SCHEMA_V6 = """
CREATE TABLE IF NOT EXISTS entity_nodes (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES corpus_files(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_source ON entity_nodes(source_id);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_file ON entity_nodes(file_id);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_name ON entity_nodes(name);

CREATE TABLE IF NOT EXISTS entity_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES corpus_files(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    src_name TEXT NOT NULL,
    relation TEXT NOT NULL,
    dst_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_file ON entity_edges(file_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_src ON entity_edges(src_name);
CREATE INDEX IF NOT EXISTS idx_entity_edges_dst ON entity_edges(dst_name);
"""

SCHEMA_V7 = """
-- Tool Factory v2: declarative, immutable, content-hashed tool definitions.
-- The reference-architecture tool is seeded as entry #1 by the registry at
-- startup (idempotent), so no version-specific JSON is baked into this DDL.
CREATE TABLE IF NOT EXISTS tool_definitions (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('draft','proposed','defined','retired')),
    content_hash TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    source_run_id TEXT REFERENCES runs(id),
    created_at TEXT NOT NULL,
    UNIQUE(slug, version),
    UNIQUE(slug, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_tool_definitions_slug
    ON tool_definitions(slug, created_at);
-- At most one active definition version per slug.
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_definitions_active
    ON tool_definitions(slug) WHERE active = 1;

-- Gate-1 decisions over definitions (approve / reject-tombstone), mirroring the
-- Gate-2 tool_proposals table for built versions.
CREATE TABLE IF NOT EXISTS tool_definition_proposals (
    id TEXT PRIMARY KEY,
    definition_id TEXT NOT NULL REFERENCES tool_definitions(id) ON DELETE CASCADE,
    source_run_id TEXT REFERENCES runs(id),
    status TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    summary TEXT NOT NULL,
    decision_reason TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_definition_proposals_status
    ON tool_definition_proposals(status, created_at);
"""

SCHEMA_V8 = """
-- Gate-2 build records for declarative tool definitions.
-- Image-based tools keep using tools/tool_versions/tool_proposals; declarative
-- tools (no runner image — e.g. the README summary card) are built, evaluated,
-- and activated entirely within the registry. A build is the immutable,
-- content-hashed, eval-backed candidate that Gate-2 activation pins as the
-- runnable version of a defined tool.
CREATE TABLE IF NOT EXISTS tool_definition_builds (
    id TEXT PRIMARY KEY,
    definition_id TEXT NOT NULL REFERENCES tool_definitions(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('evaluated','active','rejected','superseded')),
    eval_report_json TEXT NOT NULL,
    source_run_id TEXT REFERENCES runs(id),
    created_at TEXT NOT NULL,
    decided_at TEXT,
    UNIQUE(definition_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_tool_definition_builds_slug
    ON tool_definition_builds(slug, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_definition_builds_status
    ON tool_definition_builds(status, created_at);
-- At most one active build per slug (the runnable declarative version).
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_definition_builds_active
    ON tool_definition_builds(slug) WHERE status = 'active';
"""

SCHEMA_V9 = """
-- A build may pin a model-authored, AST-gated
-- `run(inputs, model)` implementation (empty for declarative tools) plus the
-- optional OCI Grok review evidence.
ALTER TABLE tool_definition_builds ADD COLUMN implementation TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_definition_builds ADD COLUMN code_review_json TEXT;
"""

SCHEMA_V10 = """
-- A conversation may pin one explicitly opened project workspace. The project
-- itself remains identified by the manually refreshed Asset catalog; no host
-- path is copied into chat state or discovered automatically.
CREATE TABLE IF NOT EXISTS conversation_projects (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('grok_bootstrap_local','grok_continuous')),
    updated_at TEXT NOT NULL
);
"""

SCHEMA_V11 = """
-- External knowledge connectors materialize into the same local corpus, but
-- retain their provider identity so a chat turn can deliberately restrict
-- retrieval to one trusted source family (for example, Notion only).
ALTER TABLE corpus_sources ADD COLUMN provider TEXT NOT NULL DEFAULT 'local'
    CHECK(provider IN ('local','notion'));
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_sources_single_external_provider
    ON corpus_sources(provider) WHERE provider != 'local';
"""

SCHEMA_V12 = """
-- Long-term memory gets the same retrieval quality as the corpus: a local
-- float32 vector per active memory, embedded only after an explicit, separate
-- consent. Keyed by memory id so deactivating or deleting a memory takes its
-- vector with it, and revoking consent purges the table wholesale.
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL CHECK(dim > 0),
    embed_model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Consent for memory is deliberately its own record rather than a reuse of the
-- corpus per-source flag: memories are the user's own words about themselves,
-- so opting a directory into cloud embedding must never silently opt in these.
CREATE TABLE IF NOT EXISTS memory_settings (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    cloud_consent INTEGER NOT NULL DEFAULT 0 CHECK(cloud_consent IN (0,1)),
    consent_reason TEXT,
    updated_at TEXT NOT NULL
);
"""

SCHEMA_V13 = """
-- Customer intelligence is deliberately separate from personal memory and the
-- general corpus. Every derived record remains account-scoped and traceable to
-- the original note so one customer's context cannot silently leak to another.
CREATE TABLE IF NOT EXISTS customer_accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    industry TEXT NOT NULL DEFAULT '',
    region TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active','paused','archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS customer_sources (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '',
    occurred_at TEXT,
    status TEXT NOT NULL DEFAULT 'waiting'
        CHECK(status IN ('waiting','review','saved','duplicate')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(account_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_customer_sources_account
    ON customer_sources(account_id, created_at DESC);
CREATE TABLE IF NOT EXISTS customer_update_proposals (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES customer_sources(id) ON DELETE CASCADE,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'review'
        CHECK(status IN ('review','approved','rejected')),
    extraction_json TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS customer_interactions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL UNIQUE REFERENCES customer_sources(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS customer_people (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    organization TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(account_id, name)
);
CREATE TABLE IF NOT EXISTS customer_facts (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    interaction_id TEXT REFERENCES customer_interactions(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active','superseded','disputed')),
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_customer_facts_account
    ON customer_facts(account_id, status, created_at DESC);
CREATE TABLE IF NOT EXISTS customer_actions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    interaction_id TEXT REFERENCES customer_interactions(id) ON DELETE SET NULL,
    description TEXT NOT NULL,
    owner TEXT NOT NULL DEFAULT '',
    due_at TEXT,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK(status IN ('open','done','cancelled')),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_customer_actions_account
    ON customer_actions(account_id, status, due_at);
CREATE TABLE IF NOT EXISTS customer_outputs (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    interaction_id TEXT REFERENCES customer_interactions(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS customer_settings (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    tracker_url TEXT NOT NULL DEFAULT '',
    activity_template TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""

SCHEMA_V14 = """
-- Customer wins are explicit, user-recorded outcomes on an account. They power
-- the customers-page win tracker and stay account-scoped like every other
-- customer record: deleting the account removes its wins with it.
CREATE TABLE IF NOT EXISTS customer_wins (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    brief TEXT NOT NULL DEFAULT '',
    services_json TEXT NOT NULL DEFAULT '[]',
    dac_shape TEXT NOT NULL DEFAULT '',
    yearly_arr REAL CHECK(yearly_arr IS NULL OR yearly_arr >= 0),
    won_at TEXT,
    source_ref TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_customer_wins_account
    ON customer_wins(account_id, won_at DESC);
"""

SCHEMA_V15 = """
-- An estimated value for a win whose ARR nobody recorded. Kept in its own table
-- rather than as a column on customer_wins so an estimate can never be mistaken
-- for the figure a human confirmed: `customer_wins.yearly_arr` stays the single
-- source of truth for money, and a row here only reaches it once accepted.
CREATE TABLE IF NOT EXISTS customer_win_valuations (
    id TEXT PRIMARY KEY,
    win_id TEXT NOT NULL UNIQUE REFERENCES customer_wins(id) ON DELETE CASCADE,
    estimated_yearly_arr REAL CHECK(
        estimated_yearly_arr IS NULL OR estimated_yearly_arr >= 0
    ),
    currency TEXT NOT NULL DEFAULT 'USD',
    lines_json TEXT NOT NULL DEFAULT '[]',
    explanation TEXT NOT NULL DEFAULT '',
    confidence TEXT NOT NULL DEFAULT 'low',
    unpriced_json TEXT NOT NULL DEFAULT '[]',
    rates_verified INTEGER NOT NULL DEFAULT 0,
    model_used TEXT,
    prompt_version TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'proposed'
        CHECK(status IN ('proposed', 'accepted', 'dismissed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

SCHEMA_V16 = """
-- A note written straight onto an account. Deliberately not a customer_source:
-- a source is raw material awaiting extraction, whereas a note is already the
-- knowledge the user meant to keep. It needs no model, never enters the review
-- queue, and pinned notes join the account context a scoped chat is given.
CREATE TABLE IF NOT EXISTS customer_notes (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES customer_accounts(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0 CHECK(pinned IN (0,1)),
    origin TEXT NOT NULL DEFAULT 'manual' CHECK(origin IN ('manual','chat')),
    origin_ref TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_customer_notes_account
    ON customer_notes(account_id, pinned DESC, updated_at DESC);
"""

SCHEMA_V17 = """
-- Command A+ joins the project modes. The mode column is guarded by a CHECK,
-- and SQLite cannot alter one in place, so the table is rebuilt around the
-- widened constraint. Rows carry over untouched: both existing modes remain
-- legal, so no stored session changes meaning.
CREATE TABLE conversation_projects_v17 (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL,
    mode TEXT NOT NULL
        CHECK(mode IN ('grok_bootstrap_local','grok_continuous','cohere_continuous')),
    updated_at TEXT NOT NULL
);
INSERT INTO conversation_projects_v17 (conversation_id, project_id, mode, updated_at)
    SELECT conversation_id, project_id, mode, updated_at FROM conversation_projects;
DROP TABLE conversation_projects;
ALTER TABLE conversation_projects_v17 RENAME TO conversation_projects;
"""

SCHEMA_V18 = """
-- Rewinding a thread: editing a message you already sent has to remove what
-- followed it from the model's view, or the edit changes nothing.
--
-- Marked, never deleted. Runs point at their originating message, and memory
-- proposals, tool proposals and artifacts point at those runs WITHOUT a
-- cascade — so a hard delete either fails on a foreign key or takes governed
-- records with it. A superseded message stays on disk and out of context,
-- which is also the honest answer for an app whose whole posture is that
-- history is auditable.
ALTER TABLE messages ADD COLUMN superseded_at TEXT;
CREATE INDEX IF NOT EXISTS idx_messages_conversation_live
    ON messages(conversation_id, created_at) WHERE superseded_at IS NULL;
"""

MIGRATIONS: dict[int, str] = {
    1: SCHEMA_V1,
    2: SCHEMA_V2,
    3: SCHEMA_V3,
    4: SCHEMA_V4,
    5: SCHEMA_V5,
    6: SCHEMA_V6,
    7: SCHEMA_V7,
    8: SCHEMA_V8,
    9: SCHEMA_V9,
    10: SCHEMA_V10,
    11: SCHEMA_V11,
    12: SCHEMA_V12,
    13: SCHEMA_V13,
    14: SCHEMA_V14,
    15: SCHEMA_V15,
    16: SCHEMA_V16,
    17: SCHEMA_V17,
    18: SCHEMA_V18,
}
SUPPORTED_SCHEMA_VERSION = max(MIGRATIONS)


class Database:
    """Small async facade over one serialized SQLite connection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def operation() -> None:
            conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
                )"""
            )
            applied = {
                int(row[0])
                for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }
            if applied and max(applied) > SUPPORTED_SCHEMA_VERSION:
                conn.close()
                raise RuntimeError(
                    f"database schema {max(applied)} is newer than supported "
                    f"schema {SUPPORTED_SCHEMA_VERSION}"
                )
            for version, migration in sorted(MIGRATIONS.items()):
                if version in applied:
                    continue
                try:
                    conn.executescript(
                        "BEGIN IMMEDIATE;\n"
                        + migration
                        + "\nINSERT INTO schema_migrations(version, applied_at) "
                        + f"VALUES ({version}, CURRENT_TIMESTAMP);\nCOMMIT;"
                    )
                except BaseException:
                    if conn.in_transaction:
                        conn.rollback()
                    conn.close()
                    raise
            self._conn = conn

        await asyncio.to_thread(operation)

    async def close(self) -> None:
        if self._conn is None:
            return
        conn, self._conn = self._conn, None
        await asyncio.to_thread(conn.close)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("database is not open")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()

    async def _call(self, function: Callable[[], T]) -> T:
        return await asyncio.to_thread(function)

    async def ping(self) -> bool:
        def operation() -> bool:
            with self._lock:
                return self._connection().execute("SELECT 1").fetchone()[0] == 1

        return await self._call(operation)

    async def create_conversation(self, title: str | None) -> ConversationV1:
        conversation_id, timestamp = _id("conv"), _now()
        title = (title or "New conversation").strip() or "New conversation"

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                    (conversation_id, title, timestamp, timestamp),
                )

        await self._call(operation)
        return ConversationV1(
            id=conversation_id, title=title, created_at=timestamp, updated_at=timestamp
        )

    async def list_conversations(self, limit: int = 100) -> list[ConversationV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
                    )
                    .fetchall()
                )

        return [ConversationV1.model_validate(dict(row)) for row in await self._call(operation)]

    async def get_conversation(self, conversation_id: str) -> ConversationV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()

        row = await self._call(operation)
        return ConversationV1.model_validate(dict(row)) if row else None

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all data that is scoped to it."""

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def set_conversation_project(
        self, conversation_id: str, project_id: str, mode: str
    ) -> ConversationProjectV1:
        if mode not in PROJECT_MODES:
            raise ValueError("unsupported project mode")
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO conversation_projects
                    (conversation_id, project_id, mode, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        project_id = excluded.project_id,
                        mode = excluded.mode,
                        updated_at = excluded.updated_at""",
                    (conversation_id, project_id, mode, timestamp),
                )

        await self._call(operation)
        return ConversationProjectV1(
            conversation_id=conversation_id,
            project_id=project_id,
            mode=mode,
            updated_at=timestamp,
        )

    async def get_conversation_project(
        self, conversation_id: str
    ) -> ConversationProjectV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM conversation_projects WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()

        row = await self._call(operation)
        return ConversationProjectV1.model_validate(dict(row)) if row else None

    async def clear_conversation_project(self, conversation_id: str) -> None:
        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    "DELETE FROM conversation_projects WHERE conversation_id = ?",
                    (conversation_id,),
                )

        await self._call(operation)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        attachment_ids: list[str] | None = None,
        run_id: str | None = None,
    ) -> MessageV1:
        message_id, timestamp = _id("msg"), _now()
        attachments = attachment_ids or []

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO messages
                    (id, conversation_id, role, content, attachments_json, run_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        message_id,
                        conversation_id,
                        role,
                        content,
                        _json(attachments),
                        run_id,
                        timestamp,
                    ),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (timestamp, conversation_id),
                )

        await self._call(operation)
        return MessageV1(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            attachment_ids=attachments,
            run_id=run_id,
            created_at=timestamp,
        )

    async def add_assistant_message_once(
        self, conversation_id: str, content: str, run_id: str
    ) -> tuple[MessageV1, bool]:
        """Publish at most one assistant message for a run across graph replays."""

        message_id, timestamp = _id("msg"), _now()

        def operation() -> tuple[dict[str, Any], bool]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT * FROM messages WHERE run_id = ? AND role = 'assistant'",
                    (run_id,),
                ).fetchone()
                if existing:
                    return dict(existing), False
                conn.execute(
                    """INSERT INTO messages
                    (id, conversation_id, role, content, attachments_json, run_id, created_at)
                    VALUES (?, ?, 'assistant', ?, '[]', ?, ?)""",
                    (message_id, conversation_id, content, run_id, timestamp),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (timestamp, conversation_id),
                )
                return {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": content,
                    "attachments_json": "[]",
                    "run_id": run_id,
                    "created_at": timestamp,
                }, True

        data, created = await self._call(operation)
        data["attachment_ids"] = _loads(data.pop("attachments_json"), [])
        data.pop("superseded_at", None)  # see list_messages
        return MessageV1.model_validate(data), created

    async def list_messages(self, conversation_id: str) -> list[MessageV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        """SELECT * FROM messages
                        WHERE conversation_id = ? AND superseded_at IS NULL
                        ORDER BY created_at, rowid""",
                        (conversation_id,),
                    )
                    .fetchall()
                )

        result: list[MessageV1] = []
        for row in await self._call(operation):
            data = dict(row)
            data["attachment_ids"] = _loads(data.pop("attachments_json"), [])
            # Rewind bookkeeping, not part of the wire contract: every row
            # this returns is live by construction, so the column would only
            # ever carry NULL to a client that has no use for it.
            data.pop("superseded_at", None)
            result.append(MessageV1.model_validate(data))
        return result

    async def supersede_messages_from(
        self, conversation_id: str, message_id: str
    ) -> int:
        """Retire a message and everything after it. Returns how many were retired.

        The ordering is `created_at, rowid` — the same pair `list_messages`
        reads by — because two messages written in the same clock tick are
        ordered only by rowid, and a rewind that used the timestamp alone
        would leave one of a pair behind.

        Already-superseded messages are counted out, so rewinding twice to the
        same point is a no-op rather than a growing number.
        """
        timestamp = _now()

        def operation() -> int:
            with self._transaction() as conn:
                anchor = conn.execute(
                    "SELECT created_at, rowid FROM messages WHERE id = ? AND conversation_id = ?",
                    (message_id, conversation_id),
                ).fetchone()
                if anchor is None:
                    raise LookupError("message not found in this conversation")
                cursor = conn.execute(
                    """UPDATE messages SET superseded_at = ?
                    WHERE conversation_id = ? AND superseded_at IS NULL
                      AND (created_at, rowid) >= (?, ?)""",
                    (timestamp, conversation_id, anchor["created_at"], anchor["rowid"]),
                )
                # The rolling summary was written over text that is no longer
                # in the thread, so it would smuggle the rewound turns back
                # into context. It is derived state and rebuilds itself.
                conn.execute(
                    "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                    (conversation_id,),
                )
                return int(cursor.rowcount)

        return await self._call(operation)

    async def recent_messages(
        self, conversation_id: str, *, limit: int = 20, max_characters: int = 12_000
    ) -> list[dict[str, str]]:
        messages, _ = await self.recent_messages_with_metadata(
            conversation_id, limit=limit, max_characters=max_characters
        )
        return messages

    async def recent_messages_with_metadata(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
        max_characters: int = 12_000,
        exclude_message_id: str | None = None,
    ) -> tuple[list[dict[str, str]], bool]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                if exclude_message_id is not None:
                    return list(
                        self._connection().execute(
                            """SELECT role, content FROM messages
                            WHERE conversation_id = ? AND id != ?
                              AND superseded_at IS NULL
                            ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                            (conversation_id, exclude_message_id, limit + 1),
                        )
                    )
                return list(
                    self._connection().execute(
                        """SELECT role, content FROM messages
                        WHERE conversation_id = ? AND superseded_at IS NULL
                        ORDER BY created_at DESC, rowid DESC
                        LIMIT ?""",
                        (conversation_id, limit + 1),
                    )
                )

        selected = await self._call(operation)
        result: list[dict[str, str]] = []
        remaining = max_characters
        truncated = len(selected) > limit
        for index, message in enumerate(selected[:limit]):
            if remaining <= 0:
                truncated = True
                break
            original = str(message["content"])
            content = original[-remaining:]
            if len(content) < len(original):
                truncated = True
            if not content:
                continue
            result.append({"role": str(message["role"]), "content": content})
            remaining -= len(content)
            if remaining <= 0:
                if index + 1 < min(len(selected), limit):
                    truncated = True
                break
        result.reverse()
        return result, truncated

    async def get_conversation_summary(self, conversation_id: str) -> str:
        def operation() -> str:
            with self._lock:
                row = self._connection().execute(
                    "SELECT summary FROM conversation_summaries WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                return row["summary"] if row else ""

        return await self._call(operation)

    async def refresh_conversation_summary(
        self, conversation_id: str, *, max_characters: int = 8_000
    ) -> str:
        messages = await self.recent_messages(
            conversation_id, limit=20, max_characters=max_characters
        )
        summary = "\n".join(
            f"{item['role']}: {item['content']}" for item in messages
        )[-max_characters:]
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO conversation_summaries
                    (conversation_id, summary, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                    summary = excluded.summary, updated_at = excluded.updated_at""",
                    (conversation_id, summary, timestamp),
                )

        await self._call(operation)
        return summary

    async def link_message_run(self, message_id: str, run_id: str) -> None:
        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    "UPDATE messages SET run_id = ? WHERE id = ?", (run_id, message_id)
                )

        await self._call(operation)

    async def create_upload(
        self, sha256: str, filename: str, media_type: str, size: int, blob_path: str
    ) -> UploadV1:
        upload_id, timestamp = _id("upl"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (upload_id, sha256, filename, media_type, size, blob_path, timestamp),
                )

        await self._call(operation)
        return UploadV1(
            id=upload_id,
            filename=filename,
            media_type=media_type,
            size=size,
            sha256=sha256,
            created_at=timestamp,
        )

    async def get_upload_record(self, upload_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM uploads WHERE id = ?", (upload_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def validate_upload_ids(self, upload_ids: list[str]) -> bool:
        if not upload_ids:
            return True
        unique = sorted(set(upload_ids))
        placeholders = ",".join("?" for _ in unique)

        def operation() -> bool:
            with self._lock:
                count = self._connection().execute(
                    f"SELECT COUNT(*) FROM uploads WHERE id IN ({placeholders})", unique
                ).fetchone()[0]
                return count == len(unique)

        return await self._call(operation)

    async def create_run(
        self,
        conversation_id: str,
        user_message_id: str,
        *,
        graph_schema_version: str,
        model_aliases: dict[str, str],
    ) -> RunV1:
        run_id, timestamp = _id("run"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO runs
                    (id, conversation_id, user_message_id, status, graph_schema_version,
                     model_aliases_json, prompt_versions_json, tool_versions_json,
                     cancel_requested, result_json, last_error, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)""",
                    (
                        run_id,
                        conversation_id,
                        user_message_id,
                        RunStatus.QUEUED,
                        graph_schema_version,
                        _json(model_aliases),
                        _json(
                            {
                                "planner": "1",
                                "response": "1",
                                "architecture": "1",
                                "diagram_code": "1",
                                "deep_worker": "1",
                            }
                        ),
                        _json({}),
                        timestamp,
                        timestamp,
                    ),
                )

        await self._call(operation)
        return RunV1(
            id=run_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            status=RunStatus.QUEUED,
            graph_schema_version=graph_schema_version,
            cancel_requested=False,
            created_at=timestamp,
            updated_at=timestamp,
        )

    async def get_run(self, run_id: str) -> RunV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM runs WHERE id = ?", (run_id,)
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["cancel_requested"] = bool(data["cancel_requested"])
        data["result"] = _loads(data.pop("result_json"), None)
        for field in ("model_aliases_json", "prompt_versions_json", "tool_versions_json"):
            data.pop(field, None)
        return RunV1.model_validate(data)

    async def get_run_execution_record(self, run_id: str) -> dict[str, Any] | None:
        """Return persisted inputs and pins needed to recover a durable run."""

        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT r.*, m.content AS prompt, m.attachments_json
                    FROM runs r JOIN messages m ON m.id = r.user_message_id
                    WHERE r.id = ?""",
                    (run_id,),
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["model_aliases"] = _loads(data.pop("model_aliases_json"), {})
        data["prompt_versions"] = _loads(data.pop("prompt_versions_json"), {})
        data["tool_versions"] = _loads(data.pop("tool_versions_json"), {})
        data["attachment_ids"] = _loads(data.pop("attachments_json"), [])
        data["cancel_requested"] = bool(data["cancel_requested"])
        data["result"] = _loads(data.pop("result_json"), None)
        return data

    async def list_recoverable_execution_records(self) -> list[dict[str, Any]]:
        runs = await self.list_runs(limit=10_000)
        records: list[dict[str, Any]] = []
        for run in runs:
            if run.status in {
                RunStatus.QUEUED,
                RunStatus.RUNNING,
                RunStatus.AWAITING_APPROVAL,
            }:
                record = await self.get_run_execution_record(run.id)
                if record is not None:
                    records.append(record)
        return records

    async def has_active_runs(self) -> bool:
        """Whether any run is still in flight, including one paused for approval.

        A run waiting on the user has made no model call for as long as they
        have taken to answer, so the idle clock must not treat it as finished.
        """
        def operation() -> int:
            with self._lock:
                row = self._connection().execute(
                    "SELECT COUNT(*) FROM runs WHERE status IN (?, ?, ?)",
                    (
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                        RunStatus.AWAITING_APPROVAL.value,
                    ),
                ).fetchone()
                return int(row[0]) if row else 0

        return await self._call(operation) > 0

    async def list_runs(
        self, status: str | None = None, *, limit: int = 100
    ) -> list[RunV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                if status:
                    return list(
                        self._connection().execute(
                            "SELECT * FROM runs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                            (status, limit),
                        )
                    )
                return list(
                    self._connection().execute(
                        "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
                    )
                )

        result: list[RunV1] = []
        for row in await self._call(operation):
            data = dict(row)
            data["cancel_requested"] = bool(data["cancel_requested"])
            data["result"] = _loads(data.pop("result_json"), None)
            for field in (
                "model_aliases_json",
                "prompt_versions_json",
                "tool_versions_json",
            ):
                data.pop(field, None)
            result.append(RunV1.model_validate(data))
        return result

    async def latest_awaiting_project_approval(
        self, conversation_id: str
    ) -> tuple[str, str] | None:
        """The newest run of this conversation parked at an approval, with its project.

        Returns (run_id, project_id) — the project read from the run's own
        aliases rather than the conversation's current pin, so a changeset
        staged for one project can never be carried into a run on another.
        """

        def operation() -> sqlite3.Row | None:
            with self._lock:
                return (
                    self._connection()
                    .execute(
                        "SELECT id, model_aliases_json FROM runs "
                        "WHERE conversation_id = ? AND status = ? "
                        "ORDER BY updated_at DESC LIMIT 1",
                        (conversation_id, str(RunStatus.AWAITING_APPROVAL)),
                    )
                    .fetchone()
                )

        row = await self._call(operation)
        if row is None:
            return None
        aliases = _loads(row["model_aliases_json"], {}) or {}
        return str(row["id"]), str(aliases.get("_project_id", ""))

    async def set_run_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """UPDATE runs SET status = ?, result_json = COALESCE(?, result_json),
                    last_error = ?, updated_at = ? WHERE id = ?""",
                    (
                        str(status),
                        _json(result) if result is not None else None,
                        error,
                        timestamp,
                        run_id,
                    ),
                )

        await self._call(operation)

    async def pin_tool_version(
        self,
        run_id: str,
        *,
        slug: str,
        version_id: str,
        version: str,
        content_hash: str,
    ) -> None:
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT tool_versions_json FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if not row:
                    raise KeyError("run not found")
                pinned = _loads(row["tool_versions_json"], {})
                pinned[slug] = {
                    "version_id": version_id,
                    "version": version,
                    "content_hash": content_hash,
                }
                conn.execute(
                    "UPDATE runs SET tool_versions_json = ?, updated_at = ? WHERE id = ?",
                    (_json(pinned), timestamp, run_id),
                )

        await self._call(operation)

    async def request_cancel(self, run_id: str) -> bool:
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE runs SET cancel_requested = 1, updated_at = ?
                    WHERE id = ? AND status NOT IN ('completed','failed','cancelled')""",
                    (timestamp, run_id),
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def is_cancel_requested(self, run_id: str) -> bool:
        run = await self.get_run(run_id)
        return bool(run and run.cancel_requested)

    async def append_event(
        self,
        run_id: str,
        thread_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> RunEventV1:
        event_id, timestamp = _id("evt"), _now()
        payload = payload or {}

        def operation() -> int:
            with self._transaction() as conn:
                sequence = conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO run_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        run_id,
                        sequence,
                        thread_id,
                        checkpoint_id,
                        event_type,
                        _json(payload),
                        timestamp,
                    ),
                )
                return int(sequence)

        sequence = await self._call(operation)
        return RunEventV1(
            id=event_id,
            sequence=sequence,
            run_id=run_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            type=event_type,
            timestamp=timestamp,
            payload=payload,
        )

    async def list_events(self, run_id: str, after: int = 0) -> list[RunEventV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        """SELECT * FROM run_events WHERE run_id = ? AND sequence > ?
                        ORDER BY sequence""",
                        (run_id, after),
                    )
                    .fetchall()
                )

        events: list[RunEventV1] = []
        for row in await self._call(operation):
            data = dict(row)
            data["timestamp"] = data.pop("created_at")
            data["payload"] = _loads(data.pop("payload_json"), {})
            events.append(RunEventV1.model_validate(data))
        return events

    async def create_artifact(
        self,
        run_id: str,
        sha256: str,
        filename: str,
        media_type: str,
        size: int,
        blob_path: str,
    ) -> dict[str, Any]:
        artifact_id, timestamp = _id("art"), _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    """SELECT * FROM artifacts
                    WHERE run_id = ? AND filename = ? AND sha256 = ?""",
                    (run_id, filename, sha256),
                ).fetchone()
                if existing:
                    return dict(existing)
                data = {
                    "id": artifact_id,
                    "run_id": run_id,
                    "sha256": sha256,
                    "filename": filename,
                    "media_type": media_type,
                    "size": size,
                    "blob_path": blob_path,
                    "created_at": timestamp,
                }
                conn.execute(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(data.values()),
                )
                return data

        return await self._call(operation)

    async def get_idempotency_result(self, action_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()

        row = await self._call(operation)
        return _loads(row["result_json"], {}) if row else None

    async def put_idempotency_result(
        self, action_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO idempotency_actions VALUES (?, ?, ?)",
                    (action_id, _json(result), timestamp),
                )
                row = conn.execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if not row:
                    raise RuntimeError("idempotency action was not persisted")
                return _loads(row["result_json"], {})

        return await self._call(operation)

    async def get_artifact_record(self, artifact_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def list_active_tools(self) -> list[dict[str, Any]]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        """SELECT t.id, t.slug, t.name, t.description, t.active_version_id,
                        v.version, v.content_hash, v.manifest_json, v.bundle_path
                        FROM tools t JOIN tool_versions v ON v.id = t.active_version_id
                        WHERE v.state = 'active'
                        ORDER BY t.slug"""
                    )
                    .fetchall()
                )

        result = []
        for row in await self._call(operation):
            data = dict(row)
            data["manifest"] = _loads(data.pop("manifest_json"), {})
            result.append(data)
        return result

    async def is_tool_hash_rejected(self, slug: str, content_hash: str) -> bool:
        def operation() -> bool:
            with self._lock:
                row = self._connection().execute(
                    """SELECT 1 FROM tool_proposals p
                    JOIN tools t ON t.id = p.tool_id
                    JOIN tool_versions v ON v.id = p.tool_version_id
                    WHERE t.slug = ? AND v.content_hash = ? AND p.status = 'rejected'
                    LIMIT 1""",
                    (slug, content_hash),
                ).fetchone()
                return row is not None

        return await self._call(operation)

    async def list_tools(self) -> list[ToolV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection().execute("SELECT * FROM tools ORDER BY slug").fetchall()
                )

        return [ToolV1.model_validate(dict(row)) for row in await self._call(operation)]

    async def list_tool_versions(self, tool_id: str) -> list[ToolVersionV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        "SELECT * FROM tool_versions WHERE tool_id = ? ORDER BY created_at DESC",
                        (tool_id,),
                    )
                    .fetchall()
                )

        result: list[ToolVersionV1] = []
        for row in await self._call(operation):
            data = dict(row)
            data["manifest"] = _loads(data.pop("manifest_json"), {})
            data["eval_report"] = _loads(data.pop("eval_report_json"), None)
            data.pop("bundle_path", None)
            result.append(ToolVersionV1.model_validate(data))
        return result

    async def get_tool_version_record(
        self, tool_id: str, version_id: str
    ) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT * FROM tool_versions
                    WHERE id = ? AND tool_id = ?""",
                    (version_id, tool_id),
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["manifest"] = _loads(data.pop("manifest_json"), {})
        data["eval_report"] = _loads(data.pop("eval_report_json"), None)
        return data

    async def get_active_tool_version_record(
        self, tool_id: str
    ) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT v.* FROM tools t JOIN tool_versions v
                    ON v.id = t.active_version_id WHERE t.id = ?""",
                    (tool_id,),
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["manifest"] = _loads(data.pop("manifest_json"), {})
        data["eval_report"] = _loads(data.pop("eval_report_json"), None)
        return data

    async def activate_tool_version(
        self,
        tool_id: str,
        version_id: str,
        *,
        action_id: str,
        reason: str,
    ) -> dict[str, Any]:
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if existing:
                    return _loads(existing["result_json"], {})
                tool = conn.execute(
                    "SELECT active_version_id FROM tools WHERE id = ?", (tool_id,)
                ).fetchone()
                version = conn.execute(
                    "SELECT state FROM tool_versions WHERE id = ? AND tool_id = ?",
                    (version_id, tool_id),
                ).fetchone()
                if not tool or not version:
                    raise KeyError("tool or version not found")
                if version["state"] not in {ToolState.APPROVED, ToolState.ACTIVE}:
                    raise ValueError("only an approved prior version can be activated")
                prior = tool["active_version_id"]
                if prior and prior != version_id:
                    conn.execute(
                        "UPDATE tool_versions SET state = ? WHERE id = ?",
                        (ToolState.APPROVED, prior),
                    )
                conn.execute(
                    "UPDATE tool_versions SET state = ? WHERE id = ?",
                    (ToolState.ACTIVE, version_id),
                )
                conn.execute(
                    "UPDATE tools SET active_version_id = ?, updated_at = ? WHERE id = ?",
                    (version_id, timestamp, tool_id),
                )
                result = {
                    "tool_id": tool_id,
                    "active_version_id": version_id,
                    "prior_version_id": prior,
                }
                conn.execute(
                    """INSERT INTO tool_version_activation_log
                    (id, action_id, tool_id, target_version_id, prior_version_id,
                     reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _id("tvact"),
                        action_id,
                        tool_id,
                        version_id,
                        prior,
                        reason,
                        timestamp,
                    ),
                )
                conn.execute(
                    "INSERT INTO idempotency_actions VALUES (?, ?, ?)",
                    (action_id, _json(result), timestamp),
                )
                return result

        return await self._call(operation)

    async def create_tool_candidate(
        self,
        manifest: ToolManifestV1,
        eval_report: EvalReportV1,
        source_run_id: str,
        bundle_path: str,
    ) -> tuple[ToolV1, ToolVersionV1, ToolProposalV1]:
        timestamp = _now()

        def operation() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            with self._transaction() as conn:
                tool_row = conn.execute(
                    "SELECT * FROM tools WHERE slug = ?", (manifest.slug,)
                ).fetchone()
                if tool_row:
                    tool = dict(tool_row)
                    conn.execute(
                        "UPDATE tools SET name = ?, description = ?, updated_at = ? WHERE id = ?",
                        (manifest.name, manifest.description, timestamp, tool["id"]),
                    )
                    tool.update(
                        name=manifest.name, description=manifest.description, updated_at=timestamp
                    )
                else:
                    tool = {
                        "id": _id("tool"),
                        "slug": manifest.slug,
                        "name": manifest.name,
                        "description": manifest.description,
                        "active_version_id": None,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }
                    conn.execute(
                        "INSERT INTO tools VALUES (?, ?, ?, ?, NULL, ?, ?)",
                        (
                            tool["id"],
                            tool["slug"],
                            tool["name"],
                            tool["description"],
                            timestamp,
                            timestamp,
                        ),
                    )

                version_row = conn.execute(
                    "SELECT * FROM tool_versions WHERE tool_id = ? AND content_hash = ?",
                    (tool["id"], manifest.content_hash),
                ).fetchone()
                if version_row:
                    version = dict(version_row)
                else:
                    version = {
                        "id": _id("tver"),
                        "tool_id": tool["id"],
                        "version": manifest.version,
                        "state": ToolState.EVALUATED,
                        "content_hash": manifest.content_hash,
                        "manifest_json": _json(manifest.model_dump(mode="json")),
                        "bundle_path": bundle_path,
                        "eval_report_json": _json(eval_report.model_dump(mode="json")),
                        "source_run_id": source_run_id,
                        "created_at": timestamp,
                    }
                    conn.execute(
                        """INSERT INTO tool_versions
                        (id, tool_id, version, state, content_hash, manifest_json, bundle_path,
                         eval_report_json, source_run_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        tuple(version.values()),
                    )

                proposal_row = conn.execute(
                    """SELECT * FROM tool_proposals
                    WHERE source_run_id = ? AND tool_version_id = ?""",
                    (source_run_id, version["id"]),
                ).fetchone()
                if proposal_row:
                    proposal = dict(proposal_row)
                    proposal.pop("decision_reason", None)
                else:
                    proposal = {
                        "id": _id("tprop"),
                        "tool_id": tool["id"],
                        "tool_version_id": version["id"],
                        "source_run_id": source_run_id,
                        "status": ProposalStatus.PENDING,
                        "risk_level": manifest.risk_level,
                        "summary": f"Activate {manifest.name} {manifest.version}",
                        "created_at": timestamp,
                        "decided_at": None,
                    }
                    conn.execute(
                        """INSERT INTO tool_proposals
                        (id, tool_id, tool_version_id, source_run_id, status, risk_level,
                         summary, decision_reason, created_at, decided_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)""",
                        (
                            proposal["id"],
                            proposal["tool_id"],
                            proposal["tool_version_id"],
                            proposal["source_run_id"],
                            proposal["status"],
                            proposal["risk_level"],
                            proposal["summary"],
                            timestamp,
                        ),
                    )
                return tool, version, proposal

        tool_data, version_data, proposal_data = await self._call(operation)
        version_data = dict(version_data)
        version_data["manifest"] = _loads(version_data.pop("manifest_json"), {})
        version_data["eval_report"] = _loads(version_data.pop("eval_report_json"), None)
        version_data.pop("bundle_path", None)
        return (
            ToolV1.model_validate(tool_data),
            ToolVersionV1.model_validate(version_data),
            ToolProposalV1.model_validate(proposal_data),
        )

    async def get_tool_proposal(self, proposal_id: str) -> ToolProposalV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM tool_proposals WHERE id = ?", (proposal_id,)
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data.pop("decision_reason", None)
        return ToolProposalV1.model_validate(data)

    async def list_tool_proposals(
        self, status: str | None = None
    ) -> list[ToolProposalV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                if status:
                    return list(
                        self._connection()
                        .execute(
                            "SELECT * FROM tool_proposals WHERE status = ? ORDER BY created_at DESC",
                            (status,),
                        )
                        .fetchall()
                    )
                return list(
                    self._connection()
                    .execute("SELECT * FROM tool_proposals ORDER BY created_at DESC")
                    .fetchall()
                )

        result = []
        for row in await self._call(operation):
            data = dict(row)
            data.pop("decision_reason", None)
            result.append(ToolProposalV1.model_validate(data))
        return result

    async def create_approval(self, request: ApprovalRequestV1) -> ApprovalRequestV1:
        def operation() -> str:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO approvals
                    (id, run_id, proposal_id, action_id, kind, status, request_json,
                     decision_json, created_at, decided_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, NULL, ?, NULL)""",
                    (
                        request.id,
                        request.run_id,
                        request.proposal_id,
                        request.action_id,
                        request.kind,
                        _json(request.model_dump(mode="json")),
                        request.created_at.isoformat(),
                    ),
                )
                row = conn.execute(
                    "SELECT request_json FROM approvals WHERE action_id = ?",
                    (request.action_id,),
                ).fetchone()
                if not row:
                    raise RuntimeError("approval was not persisted")
                return str(row["request_json"])

        return ApprovalRequestV1.model_validate_json(await self._call(operation))

    async def get_pending_approval(self, run_id: str) -> ApprovalRequestV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT request_json FROM approvals
                    WHERE run_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1""",
                    (run_id,),
                ).fetchone()

        row = await self._call(operation)
        return ApprovalRequestV1.model_validate_json(row["request_json"]) if row else None

    async def get_latest_approval_record(self, run_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT status, request_json, decision_json, decided_at
                    FROM approvals WHERE run_id = ? ORDER BY created_at DESC LIMIT 1""",
                    (run_id,),
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["request"] = _loads(data.pop("request_json"), {})
        data["decision"] = _loads(data.pop("decision_json"), None)
        return data

    async def list_decided_unfinished_approvals(self) -> list[dict[str, Any]]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection().execute(
                        """SELECT a.status, a.request_json, a.decision_json,
                        r.id AS run_id, r.conversation_id
                        FROM approvals a JOIN runs r ON r.id = a.run_id
                        WHERE a.status IN ('approve','reject','draft')
                        AND r.status IN ('queued','running','awaiting_approval')
                        ORDER BY a.decided_at"""
                    )
                )

        result: list[dict[str, Any]] = []
        for row in await self._call(operation):
            data = dict(row)
            data["request"] = _loads(data.pop("request_json"), {})
            data["decision"] = _loads(data.pop("decision_json"), {})
            result.append(data)
        return result

    async def record_approval_decision(
        self, approval_id: str, decision: str, reason: str | None
    ) -> bool:
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE approvals SET status = ?, decision_json = ?, decided_at = ?
                    WHERE id = ? AND status = 'pending'""",
                    (decision, _json({"decision": decision, "reason": reason}), timestamp, approval_id),
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def decide_tool_proposal(
        self,
        proposal_id: str,
        decision: str,
        reason: str | None,
        action_id: str,
    ) -> dict[str, Any]:
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if existing:
                    return _loads(existing["result_json"], {})
                proposal_row = conn.execute(
                    "SELECT * FROM tool_proposals WHERE id = ?", (proposal_id,)
                ).fetchone()
                if not proposal_row:
                    raise KeyError("tool proposal not found")
                proposal = dict(proposal_row)
                if proposal["status"] not in {ProposalStatus.PENDING, decision}:
                    raise ValueError(f"proposal is already {proposal['status']}")
                if decision == ProposalStatus.APPROVED:
                    version = conn.execute(
                        "SELECT * FROM tool_versions WHERE id = ?",
                        (proposal["tool_version_id"],),
                    ).fetchone()
                    if not version or version["state"] not in {
                        ToolState.EVALUATED,
                        ToolState.APPROVED,
                        ToolState.ACTIVE,
                    }:
                        raise ValueError("only evaluated versions can be activated")
                    old_active = conn.execute(
                        "SELECT active_version_id FROM tools WHERE id = ?",
                        (proposal["tool_id"],),
                    ).fetchone()[0]
                    if old_active and old_active != proposal["tool_version_id"]:
                        conn.execute(
                            "UPDATE tool_versions SET state = ? WHERE id = ?",
                            (ToolState.APPROVED, old_active),
                        )
                    conn.execute(
                        "UPDATE tool_versions SET state = ? WHERE id = ?",
                        (ToolState.ACTIVE, proposal["tool_version_id"]),
                    )
                    conn.execute(
                        "UPDATE tools SET active_version_id = ?, updated_at = ? WHERE id = ?",
                        (proposal["tool_version_id"], timestamp, proposal["tool_id"]),
                    )
                conn.execute(
                    """UPDATE tool_proposals SET status = ?, decision_reason = ?, decided_at = ?
                    WHERE id = ?""",
                    (decision, reason, timestamp, proposal_id),
                )
                result = {"proposal_id": proposal_id, "status": decision, "applied": True}
                conn.execute(
                    "INSERT INTO idempotency_actions VALUES (?, ?, ?)",
                    (action_id, _json(result), timestamp),
                )
                return result

        return await self._call(operation)

    async def search_memories(self, query: str, limit: int = 5) -> list[str]:
        tokens = [token for token in query.replace('"', " ").split() if len(token) > 2][:12]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)

        def operation() -> list[str]:
            with self._lock:
                rows = self._connection().execute(
                    """SELECT m.content FROM memory_fts f
                    JOIN memory_items m ON m.rowid = f.rowid
                    WHERE memory_fts MATCH ? AND m.active = 1
                    ORDER BY bm25(memory_fts) LIMIT ?""",
                    (expression, limit),
                ).fetchall()
                return [row["content"] for row in rows]

        try:
            return await self._call(operation)
        except sqlite3.OperationalError:
            return []

    async def create_memory_proposal(
        self,
        kind: str,
        content: str,
        source_run_id: str | None,
        confidence: float,
    ) -> MemoryProposalV1:
        proposal_id, timestamp = _id("mprop"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO memory_proposals
                    (id, kind, content, source_run_id, confidence, status,
                     decision_reason, created_at, decided_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL)""",
                    (
                        proposal_id,
                        kind,
                        content,
                        source_run_id,
                        confidence,
                        ProposalStatus.PENDING,
                        timestamp,
                    ),
                )

        await self._call(operation)
        return MemoryProposalV1(
            id=proposal_id,
            kind=kind,
            content=content,
            source_run_id=source_run_id,
            confidence=confidence,
            status=ProposalStatus.PENDING,
            created_at=timestamp,
        )

    async def list_memory_proposals(
        self, status: str | None = ProposalStatus.PENDING
    ) -> list[MemoryProposalV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                if status:
                    return list(
                        self._connection()
                        .execute(
                            "SELECT * FROM memory_proposals WHERE status = ? ORDER BY created_at DESC",
                            (status,),
                        )
                        .fetchall()
                    )
                return list(
                    self._connection()
                    .execute("SELECT * FROM memory_proposals ORDER BY created_at DESC")
                    .fetchall()
                )

        result = []
        for row in await self._call(operation):
            data = dict(row)
            data.pop("decision_reason", None)
            result.append(MemoryProposalV1.model_validate(data))
        return result

    async def decide_memory_proposal(
        self, proposal_id: str, decision: str, reason: str | None
    ) -> MemoryProposalV1:
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM memory_proposals WHERE id = ?", (proposal_id,)
                ).fetchone()
                if not row:
                    raise KeyError("memory proposal not found")
                data = dict(row)
                if data["status"] != ProposalStatus.PENDING:
                    raise ValueError(f"proposal is already {data['status']}")
                conn.execute(
                    """UPDATE memory_proposals SET status = ?, decision_reason = ?, decided_at = ?
                    WHERE id = ?""",
                    (decision, reason, timestamp, proposal_id),
                )
                if decision == ProposalStatus.APPROVED:
                    conn.execute(
                        """INSERT INTO memory_items
                        (id, kind, content, source_run_id, active, created_at)
                        VALUES (?, ?, ?, ?, 1, ?)""",
                        (
                            _id("mem"),
                            data["kind"],
                            data["content"],
                            data["source_run_id"],
                            timestamp,
                        ),
                    )
                data.update(status=decision, decided_at=timestamp)
                data.pop("decision_reason", None)
                return data

        return MemoryProposalV1.model_validate(await self._call(operation))

    # ── Memory vectors (the same retrieval path the corpus uses) ─────────────

    async def get_memory_consent(self) -> tuple[bool, str | None]:
        def operation() -> tuple[bool, str | None]:
            with self._lock:
                row = self._connection().execute(
                    "SELECT cloud_consent, consent_reason FROM memory_settings WHERE id = 1"
                ).fetchone()
                if not row:
                    return False, None
                return bool(row["cloud_consent"]), row["consent_reason"]

        return await self._call(operation)

    async def set_memory_consent(
        self, consent: bool, reason: str | None
    ) -> tuple[bool, str | None]:
        """Grant or revoke cloud embedding for long-term memory.

        Revoking purges every stored vector, so nothing cloud-derived survives
        the withdrawal — the same contract corpus consent already offers.
        """
        timestamp = _now()

        def operation() -> tuple[bool, str | None]:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO memory_settings (id, cloud_consent, consent_reason, updated_at)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        cloud_consent = excluded.cloud_consent,
                        consent_reason = excluded.consent_reason,
                        updated_at = excluded.updated_at""",
                    (1 if consent else 0, reason, timestamp),
                )
                if not consent:
                    conn.execute("DELETE FROM memory_vectors")
                return consent, reason

        return await self._call(operation)

    async def memories_needing_vectors(self, limit: int = 500) -> list[dict[str, str]]:
        """Active memories with no current vector, newest first.

        `content_hash` guards edited content: a memory whose text changed no
        longer matches its stored vector, and must be embedded again rather than
        retrieved through a vector describing the old wording.
        """

        def operation() -> list[dict[str, str]]:
            with self._lock:
                rows = self._connection().execute(
                    """SELECT m.id, m.content, v.content_hash FROM memory_items m
                    LEFT JOIN memory_vectors v ON v.memory_id = m.id
                    WHERE m.active = 1 ORDER BY m.created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                return [
                    {"id": row["id"], "content": row["content"]}
                    for row in rows
                    if row["content_hash"] != memory_content_hash(row["content"])
                ]

        return await self._call(operation)

    async def store_memory_vectors(
        self, items: list[tuple[str, bytes, int]], embed_model: str
    ) -> int:
        if not items:
            return 0
        timestamp = _now()

        def operation() -> int:
            with self._transaction() as conn:
                written = 0
                for memory_id, embedding, dim in items:
                    row = conn.execute(
                        "SELECT content FROM memory_items WHERE id = ? AND active = 1",
                        (memory_id,),
                    ).fetchone()
                    if not row:
                        continue
                    conn.execute(
                        """INSERT INTO memory_vectors
                        (memory_id, embedding, dim, embed_model, content_hash, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(memory_id) DO UPDATE SET
                            embedding = excluded.embedding, dim = excluded.dim,
                            embed_model = excluded.embed_model,
                            content_hash = excluded.content_hash,
                            created_at = excluded.created_at""",
                        (
                            memory_id,
                            embedding,
                            dim,
                            embed_model,
                            memory_content_hash(row["content"]),
                            timestamp,
                        ),
                    )
                    written += 1
                return written

        return await self._call(operation)

    async def memory_search_vectors(self) -> list[sqlite3.Row]:
        """(id, embedding, dim) for every active memory that has a vector."""

        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        """SELECT v.memory_id AS id, v.embedding, v.dim
                        FROM memory_vectors v
                        JOIN memory_items m ON m.id = v.memory_id
                        WHERE m.active = 1"""
                    )
                    .fetchall()
                )

        return await self._call(operation)

    async def memories_by_ids(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)

        def operation() -> dict[str, str]:
            with self._lock:
                rows = self._connection().execute(
                    f"""SELECT id, content FROM memory_items
                    WHERE active = 1 AND id IN ({placeholders})""",
                    tuple(ids),
                ).fetchall()
                return {row["id"]: row["content"] for row in rows}

        return await self._call(operation)

    async def memory_vector_stats(self) -> dict[str, int]:
        def operation() -> dict[str, int]:
            with self._lock:
                conn = self._connection()
                active = conn.execute(
                    "SELECT COUNT(*) AS total FROM memory_items WHERE active = 1"
                ).fetchone()["total"]
                embedded = conn.execute(
                    """SELECT COUNT(*) AS total FROM memory_vectors v
                    JOIN memory_items m ON m.id = v.memory_id WHERE m.active = 1"""
                ).fetchone()["total"]
                return {"active": int(active), "embedded": int(embedded)}

        return await self._call(operation)

    # ── Personal knowledge corpus (Tier-1 RAG) ───────────────────────────────

    async def create_corpus_source(
        self, root_path: str, label: str, kind: str, provider: str = "local"
    ) -> CorpusSourceV1:
        source_id, timestamp = _id("src"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT id FROM corpus_sources WHERE root_path = ?", (root_path,)
                ).fetchone()
                if existing:
                    raise ValueError("a source with this path is already registered")
                conn.execute(
                    """INSERT INTO corpus_sources
                    (id, root_path, label, kind, consent, consent_reason, status,
                     file_count, chunk_count, embed_model, last_indexed_at,
                     last_error, created_at, updated_at, provider)
                    VALUES (?, ?, ?, ?, 0, NULL, 'pending', 0, 0, NULL, NULL, NULL, ?, ?, ?)""",
                    (source_id, root_path, label, kind, timestamp, timestamp, provider),
                )

        await self._call(operation)
        return CorpusSourceV1(
            id=source_id, root_path=root_path, label=label, kind=kind,
            provider=provider,
            consent=False, status="pending", file_count=0, chunk_count=0,
            last_indexed_at=None, last_error=None,
            created_at=timestamp, updated_at=timestamp,
        )

    async def list_corpus_sources(self) -> list[CorpusSourceV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute("SELECT * FROM corpus_sources ORDER BY created_at DESC")
                    .fetchall()
                )

        return [_corpus_source(row) for row in await self._call(operation)]

    async def get_corpus_source(self, source_id: str) -> CorpusSourceV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return (
                    self._connection()
                    .execute("SELECT * FROM corpus_sources WHERE id = ?", (source_id,))
                    .fetchone()
                )

        row = await self._call(operation)
        return _corpus_source(row) if row else None

    async def get_corpus_source_by_provider(
        self, provider: str
    ) -> CorpusSourceV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM corpus_sources WHERE provider = ? ORDER BY created_at LIMIT 1",
                    (provider,),
                ).fetchone()

        row = await self._call(operation)
        return _corpus_source(row) if row else None

    async def update_corpus_source_label(
        self, source_id: str, label: str
    ) -> CorpusSourceV1:
        timestamp = _now()

        def operation() -> sqlite3.Row:
            with self._transaction() as conn:
                conn.execute(
                    "UPDATE corpus_sources SET label = ?, updated_at = ? WHERE id = ?",
                    (label, timestamp, source_id),
                )
                row = conn.execute(
                    "SELECT * FROM corpus_sources WHERE id = ?", (source_id,)
                ).fetchone()
                if row is None:
                    raise KeyError("corpus source not found")
                return row

        return _corpus_source(await self._call(operation))

    async def set_corpus_consent(
        self, source_id: str, consent: bool, reason: str | None
    ) -> CorpusSourceV1:
        """Grant or revoke cloud-embedding consent. Revoking also purges the
        source's locally-stored embeddings, so nothing cloud-derived remains."""
        timestamp = _now()

        def operation() -> sqlite3.Row:
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT id FROM corpus_sources WHERE id = ?", (source_id,)
                ).fetchone()
                if not row:
                    raise KeyError("corpus source not found")
                if consent:
                    conn.execute(
                        """UPDATE corpus_sources SET consent = 1, consent_reason = ?,
                        status = CASE WHEN status IN ('revoked','error') THEN 'pending'
                                      ELSE status END,
                        updated_at = ? WHERE id = ?""",
                        (reason, timestamp, source_id),
                    )
                else:
                    conn.execute("DELETE FROM corpus_files WHERE source_id = ?", (source_id,))
                    conn.execute("DELETE FROM corpus_chunks WHERE source_id = ?", (source_id,))
                    conn.execute(
                        """UPDATE corpus_sources SET consent = 0, consent_reason = ?,
                        status = 'revoked', file_count = 0, chunk_count = 0,
                        last_indexed_at = NULL, updated_at = ? WHERE id = ?""",
                        (reason, timestamp, source_id),
                    )
                return conn.execute(
                    "SELECT * FROM corpus_sources WHERE id = ?", (source_id,)
                ).fetchone()

        return _corpus_source(await self._call(operation))

    async def delete_corpus_source(self, source_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM corpus_sources WHERE id = ?", (source_id,)
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def begin_corpus_indexing(self, source_id: str) -> None:
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """UPDATE corpus_sources SET status = 'indexing', last_error = NULL,
                    updated_at = ? WHERE id = ?""",
                    (timestamp, source_id),
                )

        await self._call(operation)

    async def get_corpus_file_index(self, source_id: str) -> dict[str, str]:
        """Map of rel_path -> content_hash for a source (drives incremental reindex)."""

        def operation() -> dict[str, str]:
            with self._lock:
                rows = self._connection().execute(
                    "SELECT rel_path, content_hash FROM corpus_files WHERE source_id = ?",
                    (source_id,),
                ).fetchall()
            return {row["rel_path"]: row["content_hash"] for row in rows}

        return await self._call(operation)

    async def upsert_corpus_file(
        self,
        source_id: str,
        rel_path: str,
        content_hash: str,
        lang: str,
        chunks: list[dict[str, Any]],
        graph_nodes: list[dict[str, Any]] | None = None,
        graph_edges: list[dict[str, Any]] | None = None,
        entity_nodes: list[dict[str, Any]] | None = None,
        entity_edges: list[dict[str, Any]] | None = None,
    ) -> None:
        """Replace one file's chunks (and optional code-graph / entity-graph rows)
        transactionally.

        Each chunk carries `text`, `embedding` (float32 bytes), `dim`, and optional
        `symbol`/`start_line`. `graph_*` are the deterministic code graph; `entity_*`
        are the cloud-extracted entity graph. All are keyed by the new `file_id`, so
        deleting the old file row cascades away every stale derivative before the new
        one lands — one file, one atomic write."""
        file_id, timestamp = _id("cf"), _now()
        graph_nodes = graph_nodes or []
        graph_edges = graph_edges or []
        entity_nodes = entity_nodes or []
        entity_edges = entity_edges or []

        def operation() -> None:
            with self._transaction() as conn:
                old = conn.execute(
                    "SELECT id FROM corpus_files WHERE source_id = ? AND rel_path = ?",
                    (source_id, rel_path),
                ).fetchone()
                if old:
                    conn.execute("DELETE FROM corpus_files WHERE id = ?", (old["id"],))
                conn.execute(
                    """INSERT INTO corpus_files
                    (id, source_id, rel_path, content_hash, lang, chunk_count, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (file_id, source_id, rel_path, content_hash, lang, len(chunks), timestamp),
                )
                for chunk in chunks:
                    conn.execute(
                        """INSERT INTO corpus_chunks
                        (id, source_id, file_id, rel_path, symbol, start_line, text,
                         embedding, dim, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _id("ck"), source_id, file_id, rel_path,
                            chunk.get("symbol"), chunk.get("start_line"),
                            chunk["text"], chunk["embedding"], chunk["dim"], timestamp,
                        ),
                    )
                for node in graph_nodes:
                    conn.execute(
                        """INSERT INTO code_graph_nodes
                        (id, source_id, file_id, rel_path, kind, name, qualname,
                         start_line, end_line, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _id("cgn"), source_id, file_id, rel_path,
                            node["kind"], node["name"], node["qualname"],
                            node["start_line"], node["end_line"], timestamp,
                        ),
                    )
                for edge in graph_edges:
                    conn.execute(
                        """INSERT INTO code_graph_edges
                        (id, source_id, file_id, rel_path, kind, src, dst_name,
                         dst_raw, line, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _id("cge"), source_id, file_id, rel_path,
                            edge["kind"], edge["src"], edge["dst_name"],
                            edge["dst_raw"], edge["line"], timestamp,
                        ),
                    )
                for node in entity_nodes:
                    conn.execute(
                        """INSERT INTO entity_nodes
                        (id, source_id, file_id, rel_path, name, kind, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _id("en"), source_id, file_id, rel_path,
                            node["name"], node["kind"], timestamp,
                        ),
                    )
                for edge in entity_edges:
                    conn.execute(
                        """INSERT INTO entity_edges
                        (id, source_id, file_id, rel_path, src_name, relation,
                         dst_name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _id("ee"), source_id, file_id, rel_path,
                            edge["src_name"], edge["relation"], edge["dst_name"],
                            timestamp,
                        ),
                    )

        await self._call(operation)

    async def remove_corpus_files(self, source_id: str, rel_paths: list[str]) -> int:
        if not rel_paths:
            return 0

        def operation() -> int:
            removed = 0
            with self._transaction() as conn:
                for rel_path in rel_paths:
                    cursor = conn.execute(
                        "DELETE FROM corpus_files WHERE source_id = ? AND rel_path = ?",
                        (source_id, rel_path),
                    )
                    removed += cursor.rowcount
            return removed

        return await self._call(operation)

    async def finish_corpus_indexing(
        self,
        source_id: str,
        status: str,
        embed_model: str | None = None,
        last_error: str | None = None,
    ) -> CorpusSourceV1:
        timestamp = _now()

        def operation() -> sqlite3.Row:
            with self._transaction() as conn:
                counts = conn.execute(
                    """SELECT COUNT(*) AS files, COALESCE(SUM(chunk_count), 0) AS chunks
                    FROM corpus_files WHERE source_id = ?""",
                    (source_id,),
                ).fetchone()
                indexed_at = timestamp if status == "indexed" else None
                conn.execute(
                    """UPDATE corpus_sources SET status = ?, file_count = ?, chunk_count = ?,
                    embed_model = ?, last_error = ?,
                    last_indexed_at = COALESCE(?, last_indexed_at), updated_at = ?
                    WHERE id = ?""",
                    (
                        status, counts["files"], counts["chunks"], embed_model,
                        last_error, indexed_at, timestamp, source_id,
                    ),
                )
                return conn.execute(
                    "SELECT * FROM corpus_sources WHERE id = ?", (source_id,)
                ).fetchone()

        return _corpus_source(await self._call(operation))

    async def corpus_search_vectors(
        self, provider: str | None = None
    ) -> list[sqlite3.Row]:
        """(id, embedding, dim) rows for every consented, indexed chunk."""

        def operation() -> list[sqlite3.Row]:
            with self._lock:
                provider_clause = " AND s.provider = ?" if provider else ""
                parameters = (provider,) if provider else ()
                return list(
                    self._connection()
                    .execute(
                        """SELECT c.id, c.embedding, c.dim FROM corpus_chunks c
                        JOIN corpus_sources s ON s.id = c.source_id
                        WHERE s.consent = 1 AND s.status = 'indexed'"""
                        + provider_clause,
                        parameters,
                    )
                    .fetchall()
                )

        return await self._call(operation)

    async def corpus_chunks_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)

        def operation() -> list[dict[str, Any]]:
            with self._lock:
                rows = self._connection().execute(
                    f"""SELECT c.id, c.source_id, c.rel_path, c.symbol,
                    c.start_line, c.text, s.label AS source_label,
                    s.provider AS source_provider
                    FROM corpus_chunks c
                    JOIN corpus_sources s ON s.id = c.source_id
                    WHERE c.id IN ({placeholders})""",
                    tuple(ids),
                ).fetchall()
            return [dict(row) for row in rows]

        return await self._call(operation)

    # ── Code graph (Graph-RAG Stage 1) ───────────────────────────────────────

    async def code_graph_stats(self) -> dict[str, Any]:
        """Node/edge counts (overall and by kind) over consented, indexed sources."""

        def operation() -> dict[str, Any]:
            with self._lock:
                conn = self._connection()
                nodes = conn.execute(
                    """SELECT n.kind AS kind, COUNT(*) AS count
                    FROM code_graph_nodes n JOIN corpus_sources s ON s.id = n.source_id
                    WHERE s.consent = 1 GROUP BY n.kind"""
                ).fetchall()
                edges = conn.execute(
                    """SELECT e.kind AS kind, COUNT(*) AS count
                    FROM code_graph_edges e JOIN corpus_sources s ON s.id = e.source_id
                    WHERE s.consent = 1 GROUP BY e.kind"""
                ).fetchall()
            node_counts = {row["kind"]: int(row["count"]) for row in nodes}
            edge_counts = {row["kind"]: int(row["count"]) for row in edges}
            return {
                "node_count": sum(node_counts.values()),
                "edge_count": sum(edge_counts.values()),
                "nodes_by_kind": node_counts,
                "edges_by_kind": edge_counts,
            }

        return await self._call(operation)

    async def code_graph_lookup(self, name: str, limit: int = 50) -> dict[str, Any]:
        """Resolve one symbol name to its definitions, callers, and callees.

        Matching is by simple name (the lightweight-graph trade-off): callers are
        `calls` edges whose target name is `name`; callees are the `calls` edges
        made *by* any definition named `name`. Restricted to consented sources."""
        name = (name or "").strip()
        if not name:
            return {"name": name, "definitions": [], "callers": [], "callees": [], "imports": []}

        def operation() -> dict[str, Any]:
            with self._lock:
                conn = self._connection()
                definitions = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT n.kind, n.name, n.qualname, n.rel_path,
                        n.start_line, n.end_line, s.label AS source_label
                        FROM code_graph_nodes n JOIN corpus_sources s ON s.id = n.source_id
                        WHERE n.name = ? AND n.kind != 'module' AND s.consent = 1
                        ORDER BY n.rel_path, n.start_line LIMIT ?""",
                        (name, limit),
                    ).fetchall()
                ]
                callers = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT DISTINCT e.src AS caller, e.rel_path, e.line,
                        e.dst_raw, s.label AS source_label
                        FROM code_graph_edges e JOIN corpus_sources s ON s.id = e.source_id
                        WHERE e.kind = 'calls' AND e.dst_name = ? AND s.consent = 1
                        ORDER BY e.rel_path, e.line LIMIT ?""",
                        (name, limit),
                    ).fetchall()
                ]
                callees = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT DISTINCT e.dst_name, e.dst_raw, e.rel_path, e.line,
                        s.label AS source_label
                        FROM code_graph_edges e JOIN corpus_sources s ON s.id = e.source_id
                        WHERE e.kind = 'calls' AND s.consent = 1 AND e.src IN (
                            SELECT qualname FROM code_graph_nodes WHERE name = ?
                        )
                        ORDER BY e.rel_path, e.line LIMIT ?""",
                        (name, limit),
                    ).fetchall()
                ]
                imports = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT DISTINCT e.rel_path, e.dst_raw, e.line,
                        s.label AS source_label
                        FROM code_graph_edges e JOIN corpus_sources s ON s.id = e.source_id
                        WHERE e.kind = 'imports' AND s.consent = 1
                        AND (e.dst_name = ? OR e.dst_raw = ?)
                        ORDER BY e.rel_path, e.line LIMIT ?""",
                        (name, name, limit),
                    ).fetchall()
                ]
            return {
                "name": name,
                "definitions": definitions,
                "callers": callers,
                "callees": callees,
                "imports": imports,
            }

        return await self._call(operation)

    async def code_graph_neighbor_names(self, symbols: list[str]) -> set[str]:
        """Names one hop away from `symbols` in the call graph (callers + callees).

        Used by hybrid retrieval to expand vector hits with structurally related
        code before reranking."""
        symbols = [s for s in {s.strip() for s in symbols} if s]
        if not symbols:
            return set()
        placeholders = ",".join("?" for _ in symbols)

        def operation() -> set[str]:
            with self._lock:
                conn = self._connection()
                callees = conn.execute(
                    f"""SELECT DISTINCT e.dst_name AS name FROM code_graph_edges e
                    JOIN corpus_sources s ON s.id = e.source_id
                    WHERE e.kind = 'calls' AND s.consent = 1 AND e.src IN (
                        SELECT qualname FROM code_graph_nodes WHERE name IN ({placeholders})
                    )""",
                    tuple(symbols),
                ).fetchall()
                callers = conn.execute(
                    f"""SELECT DISTINCT n.name AS name
                    FROM code_graph_edges e
                    JOIN code_graph_nodes n ON n.qualname = e.src
                    JOIN corpus_sources s ON s.id = e.source_id
                    WHERE e.kind = 'calls' AND s.consent = 1
                    AND e.dst_name IN ({placeholders})""",
                    tuple(symbols),
                ).fetchall()
            return {row["name"] for row in callees} | {row["name"] for row in callers}

        return await self._call(operation)

    async def corpus_chunks_by_symbols(
        self, symbols: list[str], exclude_ids: list[str], limit: int
    ) -> list[dict[str, Any]]:
        """Consented, indexed chunks whose `symbol` is one of `symbols`, excluding
        `exclude_ids`. Feeds graph-expanded candidates into rerank."""
        symbols = [s for s in {s.strip() for s in symbols} if s]
        if not symbols or limit <= 0:
            return []
        symbol_ph = ",".join("?" for _ in symbols)
        exclude_ph = ",".join("?" for _ in exclude_ids) if exclude_ids else ""
        clause = f"AND c.id NOT IN ({exclude_ph})" if exclude_ids else ""

        def operation() -> list[dict[str, Any]]:
            with self._lock:
                rows = self._connection().execute(
                    f"""SELECT c.id, c.source_id, c.rel_path, c.symbol,
                    c.start_line, c.text, s.label AS source_label,
                    s.provider AS source_provider FROM corpus_chunks c
                    JOIN corpus_sources s ON s.id = c.source_id
                    WHERE s.consent = 1 AND s.status = 'indexed'
                    AND c.symbol IN ({symbol_ph}) {clause}
                    LIMIT ?""",
                    (*symbols, *exclude_ids, limit),
                ).fetchall()
            return [dict(row) for row in rows]

        return await self._call(operation)

    async def corpus_chunks_by_paths(
        self,
        documents: list[tuple[str, str]],
        exclude_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Consented, indexed chunks belonging to the given (source_id, rel_path)
        documents, excluding `exclude_ids`, in document order. Feeds same-document
        expansion so a winning page can be answered from end to end."""
        documents = [pair for pair in documents if all(pair)]
        if not documents or limit <= 0:
            return []
        document_clause = " OR ".join(
            "(c.source_id = ? AND c.rel_path = ?)" for _ in documents
        )
        document_parameters = [value for pair in documents for value in pair]
        exclude_ph = ",".join("?" for _ in exclude_ids) if exclude_ids else ""
        clause = f"AND c.id NOT IN ({exclude_ph})" if exclude_ids else ""

        def operation() -> list[dict[str, Any]]:
            with self._lock:
                rows = self._connection().execute(
                    f"""SELECT c.id, c.source_id, c.rel_path, c.symbol,
                    c.start_line, c.text, s.label AS source_label,
                    s.provider AS source_provider FROM corpus_chunks c
                    JOIN corpus_sources s ON s.id = c.source_id
                    WHERE s.consent = 1 AND s.status = 'indexed'
                    AND ({document_clause}) {clause}
                    ORDER BY c.rel_path, c.start_line
                    LIMIT ?""",
                    (*document_parameters, *exclude_ids, limit),
                ).fetchall()
            return [dict(row) for row in rows]

        return await self._call(operation)

    # ── Entity graph (Graph-RAG Stage 2) ─────────────────────────────────────

    async def entity_graph_stats(self) -> dict[str, Any]:
        """Entity node/edge counts (overall and by kind) over consented sources."""

        def operation() -> dict[str, Any]:
            with self._lock:
                conn = self._connection()
                nodes = conn.execute(
                    """SELECT n.kind AS kind, COUNT(*) AS count
                    FROM entity_nodes n JOIN corpus_sources s ON s.id = n.source_id
                    WHERE s.consent = 1 GROUP BY n.kind"""
                ).fetchall()
                edges = conn.execute(
                    """SELECT COUNT(*) AS count FROM entity_edges e
                    JOIN corpus_sources s ON s.id = e.source_id WHERE s.consent = 1"""
                ).fetchone()
            kinds = {row["kind"]: int(row["count"]) for row in nodes}
            return {
                "node_count": sum(kinds.values()),
                "edge_count": int(edges["count"]) if edges else 0,
                "nodes_by_kind": kinds,
            }

        return await self._call(operation)

    async def entity_graph_lookup(self, name: str, limit: int = 50) -> dict[str, Any]:
        """Resolve an entity name to its kinds and its relationships (both
        directions). Matching is case-insensitive on the entity name."""
        name = (name or "").strip()
        if not name:
            return {"name": name, "kinds": [], "relations_out": [], "relations_in": []}

        def operation() -> dict[str, Any]:
            with self._lock:
                conn = self._connection()
                kinds = [
                    row["kind"]
                    for row in conn.execute(
                        """SELECT DISTINCT n.kind FROM entity_nodes n
                        JOIN corpus_sources s ON s.id = n.source_id
                        WHERE s.consent = 1 AND LOWER(n.name) = LOWER(?)""",
                        (name,),
                    ).fetchall()
                ]
                relations_out = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT DISTINCT e.relation, e.dst_name, e.rel_path,
                        s.label AS source_label FROM entity_edges e
                        JOIN corpus_sources s ON s.id = e.source_id
                        WHERE s.consent = 1 AND LOWER(e.src_name) = LOWER(?)
                        ORDER BY e.relation LIMIT ?""",
                        (name, limit),
                    ).fetchall()
                ]
                relations_in = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT DISTINCT e.relation, e.src_name, e.rel_path,
                        s.label AS source_label FROM entity_edges e
                        JOIN corpus_sources s ON s.id = e.source_id
                        WHERE s.consent = 1 AND LOWER(e.dst_name) = LOWER(?)
                        ORDER BY e.relation LIMIT ?""",
                        (name, limit),
                    ).fetchall()
                ]
            return {
                "name": name,
                "kinds": kinds,
                "relations_out": relations_out,
                "relations_in": relations_in,
            }

        return await self._call(operation)

    async def add_feedback(
        self, run_id: str, rating: str, correction: str | None
    ) -> str:
        feedback_id, timestamp = _id("feed"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO feedback VALUES (?, ?, ?, ?, ?)",
                    (feedback_id, run_id, rating, correction, timestamp),
                )

        await self._call(operation)
        return feedback_id

    async def create_tool_improvements_for_run(
        self, run_id: str, correction: str
    ) -> list[ToolImprovementProposalV1]:
        timestamp = _now()

        def operation() -> list[dict[str, Any]]:
            with self._transaction() as conn:
                run = conn.execute(
                    """SELECT r.tool_versions_json, m.content AS prompt
                    FROM runs r JOIN messages m ON m.id = r.user_message_id
                    WHERE r.id = ?""",
                    (run_id,),
                ).fetchone()
                if not run:
                    raise KeyError("run not found")
                pinned = _loads(run["tool_versions_json"], {})
                created: list[dict[str, Any]] = []
                for slug, version_pin in pinned.items():
                    version = conn.execute(
                        "SELECT tool_id FROM tool_versions WHERE id = ?",
                        (version_pin["version_id"],),
                    ).fetchone()
                    if not version:
                        continue
                    proposal_id = _id("timpr")
                    eval_case = {
                        "id": f"regression-{proposal_id}",
                        "name": f"Correction regression for {slug}",
                        "input": {
                            "source_run_id": run_id,
                            "original_request": run["prompt"],
                        },
                        "expected_properties": [correction],
                    }
                    data = {
                        "id": proposal_id,
                        "source_run_id": run_id,
                        "tool_id": version["tool_id"],
                        "tool_version_id": version_pin["version_id"],
                        "content_hash": version_pin["content_hash"],
                        "correction": correction,
                        "regression_eval_json": _json(eval_case),
                        "status": ProposalStatus.PENDING,
                        "created_at": timestamp,
                    }
                    conn.execute(
                        """INSERT INTO tool_improvement_proposals
                        (id, source_run_id, tool_id, tool_version_id, content_hash,
                         correction, regression_eval_json, status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        tuple(data.values()),
                    )
                    created.append(data)
                return created

        result = []
        for data in await self._call(operation):
            data["regression_eval"] = _loads(data.pop("regression_eval_json"), {})
            result.append(ToolImprovementProposalV1.model_validate(data))
        return result

    async def list_tool_improvements(
        self, status: str | None = ProposalStatus.PENDING
    ) -> list[ToolImprovementProposalV1]:
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                if status:
                    return list(
                        self._connection().execute(
                            """SELECT * FROM tool_improvement_proposals
                            WHERE status = ? ORDER BY created_at DESC""",
                            (status,),
                        )
                    )
                return list(
                    self._connection().execute(
                        "SELECT * FROM tool_improvement_proposals ORDER BY created_at DESC"
                    )
                )

        result = []
        for row in await self._call(operation):
            data = dict(row)
            data["regression_eval"] = _loads(data.pop("regression_eval_json"), {})
            result.append(ToolImprovementProposalV1.model_validate(data))
        return result

    async def get_tool_improvement(
        self, proposal_id: str
    ) -> ToolImprovementProposalV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM tool_improvement_proposals WHERE id = ?",
                    (proposal_id,),
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["regression_eval"] = _loads(data.pop("regression_eval_json"), {})
        return ToolImprovementProposalV1.model_validate(data)

    async def list_eligible_tool_revision_records(
        self, proposal_id: str
    ) -> list[dict[str, Any]]:
        """Return immutable, evaluated versions created after the correction proposal.

        This query deliberately does not infer that an arbitrary version is a
        correction. The caller must still name the exact target version in its
        approval decision.
        """

        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection().execute(
                        """SELECT v.* FROM tool_versions v
                        JOIN tool_improvement_proposals p ON p.tool_id = v.tool_id
                        WHERE p.id = ? AND v.id != p.tool_version_id
                        AND v.created_at >= p.created_at
                        AND v.state IN ('evaluated','approved','active')
                        ORDER BY v.created_at DESC""",
                        (proposal_id,),
                    )
                )

        result: list[dict[str, Any]] = []
        for row in await self._call(operation):
            data = dict(row)
            data["manifest"] = _loads(data.pop("manifest_json"), {})
            data["eval_report"] = _loads(data.pop("eval_report_json"), None)
            if data["eval_report"] and data["eval_report"].get("passed") is True:
                result.append(data)
        return result

    async def get_tool_revision_request(
        self, request_id: str
    ) -> ToolRevisionRequestV1 | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM tool_revision_requests WHERE id = ?", (request_id,)
                ).fetchone()

        row = await self._call(operation)
        if not row:
            return None
        data = dict(row)
        data["regression_eval"] = _loads(data.pop("regression_eval_json"), {})
        return ToolRevisionRequestV1.model_validate(data)

    async def decide_tool_improvement(
        self,
        proposal_id: str,
        decision: str,
        reason: str,
        action_id: str,
        *,
        target_version_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist one immutable improvement decision and its exact outcome.

        Approval without an evaluated target queues a governed revision request;
        it never changes the registry's active pointer. Supplying a target opts
        into one exact, evaluated, immutable bundle and activates it atomically.
        """

        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if existing:
                    return _loads(existing["result_json"], {})

                row = conn.execute(
                    "SELECT * FROM tool_improvement_proposals WHERE id = ?",
                    (proposal_id,),
                ).fetchone()
                if not row:
                    raise KeyError("tool improvement proposal not found")
                proposal = dict(row)
                if proposal["status"] != ProposalStatus.PENDING:
                    raise ValueError(f"proposal is already {proposal['status']}")
                if decision == "reject" and target_version_id is not None:
                    raise ValueError("a rejected proposal cannot name a target version")

                outcome = "rejected"
                status = ProposalStatus.REJECTED
                revision_request_id: str | None = None
                activated_version_id: str | None = None
                prior_version_id: str | None = None

                if decision == "approve" and target_version_id is None:
                    outcome = "revision_queued"
                    status = ProposalStatus.APPROVED
                    revision_request_id = _id("treq")
                    conn.execute(
                        """INSERT INTO tool_revision_requests
                        (id, proposal_id, tool_id, base_version_id, base_content_hash,
                         correction, regression_eval_json, status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)""",
                        (
                            revision_request_id,
                            proposal_id,
                            proposal["tool_id"],
                            proposal["tool_version_id"],
                            proposal["content_hash"],
                            proposal["correction"],
                            proposal["regression_eval_json"],
                            timestamp,
                        ),
                    )
                elif decision == "approve":
                    target = conn.execute(
                        """SELECT * FROM tool_versions
                        WHERE id = ? AND tool_id = ? AND id != ?
                        AND created_at >= ?
                        AND state IN ('evaluated','approved','active')""",
                        (
                            target_version_id,
                            proposal["tool_id"],
                            proposal["tool_version_id"],
                            proposal["created_at"],
                        ),
                    ).fetchone()
                    if not target:
                        raise ValueError(
                            "target must be an immutable evaluated revision created after the proposal"
                        )
                    report = _loads(target["eval_report_json"], None)
                    if not report or report.get("passed") is not True:
                        raise ValueError("target revision must have a passing evaluation report")
                    if target["content_hash"] == proposal["content_hash"]:
                        raise ValueError("target revision must differ from the pinned base version")

                    tool = conn.execute(
                        "SELECT active_version_id FROM tools WHERE id = ?",
                        (proposal["tool_id"],),
                    ).fetchone()
                    if not tool:
                        raise KeyError("tool not found")
                    prior_version_id = tool["active_version_id"]
                    if prior_version_id and prior_version_id != target_version_id:
                        conn.execute(
                            "UPDATE tool_versions SET state = ? WHERE id = ?",
                            (ToolState.APPROVED, prior_version_id),
                        )
                    conn.execute(
                        "UPDATE tool_versions SET state = ? WHERE id = ?",
                        (ToolState.ACTIVE, target_version_id),
                    )
                    conn.execute(
                        "UPDATE tools SET active_version_id = ?, updated_at = ? WHERE id = ?",
                        (target_version_id, timestamp, proposal["tool_id"]),
                    )
                    conn.execute(
                        """INSERT INTO tool_version_activation_log
                        (id, action_id, tool_id, target_version_id, prior_version_id,
                         reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _id("tvact"),
                            f"improvement-activation:{action_id}",
                            proposal["tool_id"],
                            target_version_id,
                            prior_version_id,
                            reason,
                            timestamp,
                        ),
                    )
                    outcome = "revision_activated"
                    status = ProposalStatus.APPROVED
                    activated_version_id = target_version_id

                conn.execute(
                    """UPDATE tool_improvement_proposals
                    SET status = ?, decision_reason = ?, decided_at = ?, outcome = ?,
                        revision_request_id = ?, target_version_id = ?
                    WHERE id = ?""",
                    (
                        status,
                        reason,
                        timestamp,
                        outcome,
                        revision_request_id,
                        target_version_id,
                        proposal_id,
                    ),
                )
                conn.execute(
                    """INSERT INTO tool_improvement_decisions
                    (id, proposal_id, action_id, decision, reason, outcome,
                     target_version_id, revision_request_id, prior_version_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _id("tidec"),
                        proposal_id,
                        action_id,
                        decision,
                        reason,
                        outcome,
                        target_version_id,
                        revision_request_id,
                        prior_version_id,
                        timestamp,
                    ),
                )
                result = {
                    "proposal_id": proposal_id,
                    "outcome": outcome,
                    "revision_request_id": revision_request_id,
                    "activated_version_id": activated_version_id,
                    "prior_version_id": prior_version_id,
                }
                conn.execute(
                    "INSERT INTO idempotency_actions VALUES (?, ?, ?)",
                    (action_id, _json(result), timestamp),
                )
                return result

        return await self._call(operation)

    # ── Tool Factory v2: tool definitions ────────────────────────────────────

    @staticmethod
    def _tool_definition_from_storage(
        payload: str, status: str
    ) -> ToolDefinitionV1:
        """Read immutable definition content with its current lifecycle status.

        The content-addressed JSON intentionally stays byte-stable after review;
        lifecycle promotion lives in the indexed ``status`` column. Returning
        that column here prevents an approved definition from continuing to look
        ``proposed`` to the registry and Tool Workshop.
        """
        definition = ToolDefinitionV1.model_validate_json(payload)
        return definition.model_copy(update={"status": status})

    async def list_tool_definitions(self) -> list[ToolDefinitionV1]:
        def operation() -> list[ToolDefinitionV1]:
            with self._lock:
                rows = self._connection().execute(
                    "SELECT definition_json, status FROM tool_definitions "
                    "ORDER BY slug, created_at"
                ).fetchall()
            return [
                self._tool_definition_from_storage(row["definition_json"], row["status"])
                for row in rows
            ]

        return await self._call(operation)

    async def get_active_tool_definition(
        self, slug: str
    ) -> ToolDefinitionV1 | None:
        def operation() -> ToolDefinitionV1 | None:
            with self._lock:
                row = self._connection().execute(
                    "SELECT definition_json, status FROM tool_definitions "
                    "WHERE slug = ? AND active = 1",
                    (slug,),
                ).fetchone()
            return (
                self._tool_definition_from_storage(
                    row["definition_json"], row["status"]
                )
                if row
                else None
            )

        return await self._call(operation)

    async def list_active_tool_definitions(self) -> list[ToolDefinitionV1]:
        def operation() -> list[ToolDefinitionV1]:
            with self._lock:
                rows = self._connection().execute(
                    "SELECT definition_json, status FROM tool_definitions "
                    "WHERE active = 1 ORDER BY slug"
                ).fetchall()
            return [
                self._tool_definition_from_storage(row["definition_json"], row["status"])
                for row in rows
            ]

        return await self._call(operation)

    async def upsert_tool_definition(
        self, definition: ToolDefinitionV1, *, activate: bool, source_run_id: str | None = None
    ) -> ToolDefinitionV1:
        """Insert a definition version (idempotent by slug+content_hash). When
        ``activate`` is set it becomes the sole active version for its slug."""
        payload = definition.model_dump_json()
        timestamp = _now()

        def operation() -> ToolDefinitionV1:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT id FROM tool_definitions WHERE slug = ? AND content_hash = ?",
                    (definition.slug, definition.content_hash),
                ).fetchone()
                if existing is None:
                    # A builtin whose content changed without a version bump would collide on
                    # UNIQUE(slug, version), so upgrade that row in place.
                    same_version = conn.execute(
                        "SELECT id FROM tool_definitions WHERE slug = ? AND version = ?",
                        (definition.slug, definition.version),
                    ).fetchone()
                    if same_version is not None:
                        conn.execute(
                            "UPDATE tool_definitions SET status = ?, content_hash = ?, "
                            "definition_json = ?, source_run_id = ?, created_at = ? "
                            "WHERE id = ?",
                            (
                                definition.status,
                                definition.content_hash,
                                payload,
                                source_run_id,
                                timestamp,
                                same_version["id"],
                            ),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO tool_definitions "
                            "(id, slug, version, status, content_hash, definition_json, "
                            "active, source_run_id, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
                            (
                                _id("tooldef"),
                                definition.slug,
                                definition.version,
                                definition.status,
                                definition.content_hash,
                                payload,
                                source_run_id,
                                timestamp,
                            ),
                        )
                if activate:
                    conn.execute(
                        "UPDATE tool_definitions SET active = 0 WHERE slug = ?",
                        (definition.slug,),
                    )
                    conn.execute(
                        "UPDATE tool_definitions SET active = 1 "
                        "WHERE slug = ? AND content_hash = ?",
                        (definition.slug, definition.content_hash),
                    )
            return definition

        return await self._call(operation)

    # Gate-1 definition proposals.

    def _definition_proposal_from_row(self, row: dict[str, Any]) -> ToolDefinitionProposalV1:
        return ToolDefinitionProposalV1(
            id=row["id"],
            definition_id=row["definition_id"],
            slug=row["slug"],
            version=row["version"],
            status=row["status"],
            risk_level=row["risk_level"],
            summary=row["summary"],
            source_run_id=row.get("source_run_id"),
            decision_reason=row.get("decision_reason"),
            created_at=row["created_at"],
            decided_at=row.get("decided_at"),
        )

    async def create_tool_definition_proposal(
        self,
        definition: ToolDefinitionV1,
        *,
        source_run_id: str,
        summary: str,
    ) -> tuple[ToolDefinitionV1, ToolDefinitionProposalV1]:
        """Gate-1: store a drafted definition (status 'proposed', not yet live) and
        a pending proposal over its capabilities. Idempotent by slug+content_hash so
        re-running the same toolify request reuses the same pending proposal."""
        payload = definition.model_dump_json()
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT id FROM tool_definitions WHERE slug = ? AND content_hash = ?",
                    (definition.slug, definition.content_hash),
                ).fetchone()
                if existing is None:
                    definition_id = _id("tooldef")
                    conn.execute(
                        "INSERT INTO tool_definitions "
                        "(id, slug, version, status, content_hash, definition_json, "
                        "active, source_run_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
                        (
                            definition_id,
                            definition.slug,
                            definition.version,
                            definition.status,
                            definition.content_hash,
                            payload,
                            source_run_id,
                            timestamp,
                        ),
                    )
                else:
                    definition_id = existing["id"]
                proposal_row = conn.execute(
                    "SELECT * FROM tool_definition_proposals "
                    "WHERE definition_id = ? AND status = ? ORDER BY created_at LIMIT 1",
                    (definition_id, ProposalStatus.PENDING.value),
                ).fetchone()
                if proposal_row:
                    proposal = dict(proposal_row)
                else:
                    proposal = {
                        "id": _id("tdprop"),
                        "definition_id": definition_id,
                        "source_run_id": source_run_id,
                        "status": ProposalStatus.PENDING.value,
                        "risk_level": RiskLevel.R3.value,
                        "summary": summary,
                        "decision_reason": None,
                        "created_at": timestamp,
                        "decided_at": None,
                    }
                    conn.execute(
                        "INSERT INTO tool_definition_proposals "
                        "(id, definition_id, source_run_id, status, risk_level, summary, "
                        "decision_reason, created_at, decided_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL)",
                        (
                            proposal["id"],
                            definition_id,
                            source_run_id,
                            proposal["status"],
                            proposal["risk_level"],
                            summary,
                            timestamp,
                        ),
                    )
                proposal["slug"] = definition.slug
                proposal["version"] = definition.version
                return proposal

        proposal_data = await self._call(operation)
        return definition, self._definition_proposal_from_row(proposal_data)

    async def list_tool_definition_proposals(
        self, status: str | None = None
    ) -> list[ToolDefinitionProposalV1]:
        def operation() -> list[dict[str, Any]]:
            query = (
                "SELECT p.*, d.slug AS slug, d.version AS version "
                "FROM tool_definition_proposals p "
                "JOIN tool_definitions d ON d.id = p.definition_id "
            )
            params: tuple[Any, ...] = ()
            if status:
                query += "WHERE p.status = ? "
                params = (status,)
            query += "ORDER BY p.created_at DESC"
            with self._lock:
                return [dict(row) for row in self._connection().execute(query, params).fetchall()]

        rows = await self._call(operation)
        return [self._definition_proposal_from_row(row) for row in rows]

    async def get_tool_definition_by_id(self, definition_id: str) -> ToolDefinitionV1 | None:
        def operation() -> dict[str, str] | None:
            with self._lock:
                row = self._connection().execute(
                    "SELECT definition_json, status FROM tool_definitions WHERE id = ?",
                    (definition_id,),
                ).fetchone()
            return dict(row) if row else None

        row = await self._call(operation)
        return (
            self._tool_definition_from_storage(row["definition_json"], row["status"])
            if row
            else None
        )

    async def decide_tool_definition_proposal(
        self, proposal_id: str, decision: str, reason: str | None, action_id: str
    ) -> dict[str, Any]:
        """Gate-1 apply: approve promotes the definition to the live 'defined'
        version of its slug (buildable, catalog-visible) unless a runnable version
        already exists (a pending upgrade stays inactive); reject tombstones it as
        'retired'. Idempotent via idempotency_actions."""
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if existing:
                    return _loads(existing["result_json"], {})
                prop = conn.execute(
                    "SELECT * FROM tool_definition_proposals WHERE id = ?", (proposal_id,)
                ).fetchone()
                if not prop:
                    raise KeyError("tool definition proposal not found")
                prop = dict(prop)
                if prop["status"] not in {ProposalStatus.PENDING.value, decision}:
                    raise ValueError(f"proposal is already {prop['status']}")
                definition_id = prop["definition_id"]
                defn = conn.execute(
                    "SELECT slug FROM tool_definitions WHERE id = ?", (definition_id,)
                ).fetchone()
                slug = defn["slug"] if defn else ""
                if decision == ProposalStatus.APPROVED.value:
                    has_runnable = conn.execute(
                        "SELECT 1 FROM tool_definition_builds "
                        "WHERE slug = ? AND status = 'active' LIMIT 1",
                        (slug,),
                    ).fetchone()
                    if has_runnable:
                        conn.execute(
                            "UPDATE tool_definitions SET status = 'defined' WHERE id = ?",
                            (definition_id,),
                        )
                    else:
                        conn.execute(
                            "UPDATE tool_definitions SET active = 0 WHERE slug = ?", (slug,)
                        )
                        conn.execute(
                            "UPDATE tool_definitions SET status = 'defined', active = 1 "
                            "WHERE id = ?",
                            (definition_id,),
                        )
                elif decision == ProposalStatus.REJECTED.value:
                    conn.execute(
                        "UPDATE tool_definitions SET status = 'retired', active = 0 WHERE id = ?",
                        (definition_id,),
                    )
                conn.execute(
                    "UPDATE tool_definition_proposals SET status = ?, decision_reason = ?, "
                    "decided_at = ? WHERE id = ?",
                    (decision, reason, timestamp, proposal_id),
                )
                result = {
                    "proposal_id": proposal_id,
                    "definition_id": definition_id,
                    "slug": slug,
                    "status": decision,
                    "applied": True,
                }
                conn.execute(
                    "INSERT INTO idempotency_actions VALUES (?, ?, ?)",
                    (action_id, _json(result), timestamp),
                )
                return result

        return await self._call(operation)

    # Declarative Gate-2 builds.

    def _definition_build_from_row(self, row: dict[str, Any]) -> ToolDefinitionBuildV1:
        return ToolDefinitionBuildV1(
            id=row["id"],
            definition_id=row["definition_id"],
            slug=row["slug"],
            version=row["version"],
            content_hash=row["content_hash"],
            status=row["status"],
            eval_report=_loads(row.get("eval_report_json"), None),
            implementation=row.get("implementation") or "",
            code_review=_loads(row.get("code_review_json"), None),
            source_run_id=row.get("source_run_id"),
            created_at=row["created_at"],
            decided_at=row.get("decided_at"),
        )

    async def get_buildable_definition(self, slug: str) -> ToolDefinitionV1 | None:
        """The newest 'defined' definition for a slug that still needs building
        (no active/evaluated build for its content hash). Serves `tool_factory`."""
        def operation() -> dict[str, str] | None:
            with self._lock:
                row = self._connection().execute(
                    "SELECT d.definition_json, d.status FROM tool_definitions d "
                    "WHERE d.slug = ? AND d.status = 'defined' AND NOT EXISTS ("
                    "  SELECT 1 FROM tool_definition_builds b "
                    "  WHERE b.definition_id = d.id "
                    "  AND b.status IN ('active','evaluated','superseded')"
                    ") ORDER BY d.created_at DESC LIMIT 1",
                    (slug,),
                ).fetchone()
            return dict(row) if row else None

        row = await self._call(operation)
        return (
            self._tool_definition_from_storage(row["definition_json"], row["status"])
            if row
            else None
        )

    async def get_runnable_definition(self, slug: str) -> ToolDefinitionV1 | None:
        """The definition tied to the slug's active build (runnable)."""
        def operation() -> dict[str, str] | None:
            with self._lock:
                row = self._connection().execute(
                    "SELECT d.definition_json, d.status FROM tool_definitions d "
                    "JOIN tool_definition_builds b ON b.definition_id = d.id "
                    "WHERE d.slug = ? AND b.status = 'active' LIMIT 1",
                    (slug,),
                ).fetchone()
            return dict(row) if row else None

        row = await self._call(operation)
        return (
            self._tool_definition_from_storage(row["definition_json"], row["status"])
            if row
            else None
        )

    async def get_runnable_build(self, slug: str) -> ToolDefinitionBuildV1 | None:
        """The slug's active build — carries the pinned authored implementation for
        code-authoring tools (empty for declarative)."""
        def operation() -> dict[str, Any] | None:
            with self._lock:
                row = self._connection().execute(
                    "SELECT * FROM tool_definition_builds WHERE slug = ? AND status = 'active' LIMIT 1",
                    (slug,),
                ).fetchone()
            return dict(row) if row else None

        row = await self._call(operation)
        return self._definition_build_from_row(row) if row else None

    async def declarative_build_index(self) -> dict[str, dict[str, bool]]:
        """Per-slug build presence: {slug: {"active": bool, "evaluated": bool}}.
        Drives declarative routing (runnable/buildable) and the browser."""
        def operation() -> list[sqlite3.Row]:
            with self._lock:
                return list(
                    self._connection()
                    .execute(
                        "SELECT slug, status, COUNT(*) AS n FROM tool_definition_builds "
                        "GROUP BY slug, status"
                    )
                    .fetchall()
                )

        index: dict[str, dict[str, bool]] = {}
        for row in await self._call(operation):
            entry = index.setdefault(row["slug"], {"active": False, "evaluated": False})
            if row["status"] in entry and row["n"]:
                entry[row["status"]] = True
        return index

    async def is_definition_hash_rejected(self, slug: str, content_hash: str) -> bool:
        def operation() -> bool:
            with self._lock:
                row = self._connection().execute(
                    "SELECT 1 FROM tool_definition_builds "
                    "WHERE slug = ? AND content_hash = ? AND status = 'rejected' "
                    "UNION SELECT 1 FROM tool_definitions "
                    "WHERE slug = ? AND content_hash = ? AND status = 'retired' LIMIT 1",
                    (slug, content_hash, slug, content_hash),
                ).fetchone()
                return row is not None

        return await self._call(operation)

    async def create_tool_definition_build(
        self,
        definition: ToolDefinitionV1,
        *,
        eval_report: EvalReportV1,
        source_run_id: str,
        implementation: str = "",
        code_review: dict[str, Any] | None = None,
    ) -> ToolDefinitionBuildV1:
        """Record an evaluated build (Gate-2 candidate) for a stored definition.
        Declarative tools carry no implementation (their behavior is a fixed host
        interpreter); code-authoring tools pin their AST-gated `run()` source here,
        and the build's content_hash folds that source in so different authored
        code is a distinct immutable build. Idempotent by (definition, code).
        Raises KeyError if the definition is not stored."""
        timestamp = _now()
        slug = definition.slug
        version = definition.version
        # The build's identity: the definition hash, plus the authored code when
        # present (declarative builds keep the bare definition hash — unchanged).
        build_hash = definition.content_hash
        if implementation:
            build_hash = hashlib.sha256(
                (definition.content_hash + "\x00" + implementation).encode("utf-8")
            ).hexdigest()
        review_json = _json(code_review) if code_review is not None else None

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                def_row = conn.execute(
                    "SELECT id FROM tool_definitions WHERE slug = ? AND content_hash = ?",
                    (slug, definition.content_hash),
                ).fetchone()
                if def_row is None:
                    raise KeyError("definition not found for build")
                definition_id = def_row["id"]
                row = conn.execute(
                    "SELECT * FROM tool_definition_builds "
                    "WHERE definition_id = ? AND content_hash = ?",
                    (definition_id, build_hash),
                ).fetchone()
                if row:
                    return dict(row)
                build = {
                    "id": _id("tdbuild"),
                    "definition_id": definition_id,
                    "slug": slug,
                    "version": version,
                    "content_hash": build_hash,
                    "status": "evaluated",
                    "eval_report_json": _json(eval_report.model_dump(mode="json")),
                    "source_run_id": source_run_id,
                    "created_at": timestamp,
                    "decided_at": None,
                    "implementation": implementation,
                    "code_review_json": review_json,
                }
                conn.execute(
                    "INSERT INTO tool_definition_builds "
                    "(id, definition_id, slug, version, content_hash, status, "
                    "eval_report_json, source_run_id, created_at, decided_at, "
                    "implementation, code_review_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                    (
                        build["id"],
                        definition_id,
                        slug,
                        version,
                        build_hash,
                        build["status"],
                        build["eval_report_json"],
                        source_run_id,
                        timestamp,
                        implementation,
                        review_json,
                    ),
                )
                return build

        row = await self._call(operation)
        return self._definition_build_from_row(row)

    async def list_tool_definition_builds(
        self, status: str | None = None
    ) -> list[ToolDefinitionBuildV1]:
        def operation() -> list[dict[str, Any]]:
            query = "SELECT * FROM tool_definition_builds "
            params: tuple[Any, ...] = ()
            if status:
                query += "WHERE status = ? "
                params = (status,)
            query += "ORDER BY created_at DESC"
            with self._lock:
                return [dict(r) for r in self._connection().execute(query, params).fetchall()]

        rows = await self._call(operation)
        return [self._definition_build_from_row(row) for row in rows]

    async def decide_tool_definition_build(
        self, build_id: str, decision: str, reason: str | None, action_id: str
    ) -> dict[str, Any]:
        """Gate-2 apply for a declarative tool: approve pins this build as the sole
        active (runnable) version — its definition becomes the live one, any prior
        active build is superseded; reject leaves the definition 'defined' (it can
        be rebuilt). Idempotent via idempotency_actions."""
        timestamp = _now()

        def operation() -> dict[str, Any]:
            with self._transaction() as conn:
                existing = conn.execute(
                    "SELECT result_json FROM idempotency_actions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if existing:
                    return _loads(existing["result_json"], {})
                build = conn.execute(
                    "SELECT * FROM tool_definition_builds WHERE id = ?", (build_id,)
                ).fetchone()
                if not build:
                    raise KeyError("tool definition build not found")
                build = dict(build)
                if build["status"] not in {"evaluated", decision}:
                    raise ValueError(f"build is already {build['status']}")
                slug = build["slug"]
                if decision == "active":
                    conn.execute(
                        "UPDATE tool_definition_builds SET status = 'superseded', "
                        "decided_at = ? WHERE slug = ? AND status = 'active'",
                        (timestamp, slug),
                    )
                    conn.execute(
                        "UPDATE tool_definition_builds SET status = 'active', decided_at = ? "
                        "WHERE id = ?",
                        (timestamp, build_id),
                    )
                    conn.execute(
                        "UPDATE tool_definitions SET active = 0 WHERE slug = ?", (slug,)
                    )
                    conn.execute(
                        "UPDATE tool_definitions SET active = 1 WHERE id = ?",
                        (build["definition_id"],),
                    )
                elif decision == "rejected":
                    conn.execute(
                        "UPDATE tool_definition_builds SET status = 'rejected', decided_at = ? "
                        "WHERE id = ?",
                        (timestamp, build_id),
                    )
                result = {
                    "build_id": build_id,
                    "definition_id": build["definition_id"],
                    "slug": slug,
                    "status": decision,
                    "applied": True,
                }
                conn.execute(
                    "INSERT INTO idempotency_actions VALUES (?, ?, ?)",
                    (action_id, _json(result), timestamp),
                )
                return result

        return await self._call(operation)

    # ── Customer intelligence (strictly account-scoped) ────────────────────

    async def create_customer_account(
        self, name: str, aliases: list[str], industry: str, region: str
    ) -> dict[str, Any]:
        account_id = f"cust_{uuid.uuid4().hex[:20]}"
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO customer_accounts
                    (id, name, aliases_json, industry, region, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
                    (
                        account_id, name.strip(), _json(aliases), industry.strip(),
                        region.strip(), timestamp, timestamp,
                    ),
                )

        await self._call(operation)
        return (await self.get_customer_account(account_id))  # type: ignore[return-value]

    async def update_customer_account(
        self, account_id: str, *, name: str, aliases: list[str],
        industry: str, region: str, status: str
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_accounts SET name = ?, aliases_json = ?,
                    industry = ?, region = ?, status = ?, updated_at = ? WHERE id = ?""",
                    (
                        name.strip(), _json(aliases), industry.strip(), region.strip(),
                        status, timestamp, account_id,
                    ),
                )
                return cursor.rowcount > 0

        return await self.get_customer_account(account_id) if await self._call(operation) else None

    async def delete_customer_account(self, account_id: str) -> bool:
        """Delete an account and its strictly account-scoped customer data."""

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM customer_accounts WHERE id = ?", (account_id,)
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def get_customer_account(self, account_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT a.*,
                    (SELECT COUNT(*) FROM customer_actions x
                     WHERE x.account_id = a.id AND x.status = 'open') AS open_actions,
                    (SELECT COUNT(*) FROM customer_sources s
                     WHERE s.account_id = a.id AND s.status = 'waiting') AS pending_notes,
                    (SELECT COUNT(*) FROM customer_wins w
                     WHERE w.account_id = a.id) AS wins,
                    (SELECT MAX(i.occurred_at) FROM customer_interactions i
                     WHERE i.account_id = a.id) AS last_interaction_at
                    FROM customer_accounts a WHERE a.id = ?""",
                    (account_id,),
                ).fetchone()

        row = await self._call(operation)
        if row is None:
            return None
        value = dict(row)
        value["aliases"] = _loads(value.pop("aliases_json"), [])
        return value

    async def list_customer_accounts(self) -> list[dict[str, Any]]:
        def operation() -> list[str]:
            with self._lock:
                return [
                    str(row["id"]) for row in self._connection().execute(
                        """SELECT id FROM customer_accounts
                        ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END,
                                 updated_at DESC"""
                    ).fetchall()
                ]

        values = await asyncio.gather(
            *(self.get_customer_account(item) for item in await self._call(operation))
        )
        return [item for item in values if item is not None]

    async def capture_customer_source(
        self, *, account_id: str, source_kind: str, title: str, content: str,
        source_ref: str, occurred_at: str | None
    ) -> tuple[dict[str, Any], bool]:
        source_id, timestamp = _id("csrc"), _now()
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

        def operation() -> tuple[sqlite3.Row, bool]:
            with self._transaction() as conn:
                existing = conn.execute(
                    """SELECT * FROM customer_sources
                    WHERE account_id = ? AND content_hash = ?""",
                    (account_id, digest),
                ).fetchone()
                if existing:
                    return existing, True
                conn.execute(
                    """INSERT INTO customer_sources
                    (id, account_id, source_kind, title, content, content_hash,
                     source_ref, occurred_at, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'waiting', ?, ?)""",
                    (
                        source_id, account_id, source_kind, title.strip(), content.strip(),
                        digest, source_ref.strip(), occurred_at, timestamp, timestamp,
                    ),
                )
                return conn.execute(
                    "SELECT * FROM customer_sources WHERE id = ?", (source_id,)
                ).fetchone(), False

        row, duplicate = await self._call(operation)
        return dict(row), duplicate

    async def get_customer_source(self, source_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM customer_sources WHERE id = ?", (source_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def create_customer_proposal(
        self, *, source_id: str, account_id: str, extraction: dict[str, Any],
        model: str, prompt_version: str
    ) -> dict[str, Any]:
        proposal_id, timestamp = _id("cup"), _now()

        def operation() -> sqlite3.Row:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO customer_update_proposals
                    (id, source_id, account_id, status, extraction_json, model,
                     prompt_version, created_at, decided_at)
                    VALUES (?, ?, ?, 'review', ?, ?, ?, ?, NULL)""",
                    (
                        proposal_id, source_id, account_id, _json(extraction),
                        model, prompt_version, timestamp,
                    ),
                )
                conn.execute(
                    "UPDATE customer_sources SET status = 'review', updated_at = ? WHERE id = ?",
                    (timestamp, source_id),
                )
                return conn.execute(
                    "SELECT * FROM customer_update_proposals WHERE id = ?",
                    (proposal_id,),
                ).fetchone()

        return dict(await self._call(operation))

    async def get_customer_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM customer_update_proposals WHERE id = ?",
                    (proposal_id,),
                ).fetchone()

        row = await self._call(operation)
        if row is None:
            return None
        value = dict(row)
        value["extraction"] = _loads(value.pop("extraction_json"), {})
        return value

    async def save_customer_proposal(
        self, proposal_id: str, extraction: dict[str, Any]
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                proposal = conn.execute(
                    "SELECT * FROM customer_update_proposals WHERE id = ?",
                    (proposal_id,),
                ).fetchone()
                if proposal is None or proposal["status"] != "review":
                    return False
                source = conn.execute(
                    "SELECT * FROM customer_sources WHERE id = ?", (proposal["source_id"],)
                ).fetchone()
                if source is None:
                    return False
                interaction_id = _id("cint")
                occurred_at = (
                    extraction.get("occurred_at") or source["occurred_at"] or timestamp
                )
                conn.execute(
                    """INSERT INTO customer_interactions
                    (id, account_id, source_id, title, occurred_at, summary, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        interaction_id, proposal["account_id"], source["id"],
                        source["title"], occurred_at, extraction.get("summary", ""), timestamp,
                    ),
                )
                for person in extraction.get("people", []):
                    conn.execute(
                        """INSERT INTO customer_people
                        (id, account_id, name, role, organization, evidence_json,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(account_id, name) DO UPDATE SET
                            role = CASE WHEN excluded.role != '' THEN excluded.role ELSE role END,
                            organization = CASE WHEN excluded.organization != ''
                                THEN excluded.organization ELSE organization END,
                            evidence_json = excluded.evidence_json, updated_at = excluded.updated_at""",
                        (
                            _id("cp"), proposal["account_id"], person["name"],
                            person.get("role", ""), person.get("organization", ""),
                            _json(person.get("evidence", {})), timestamp, timestamp,
                        ),
                    )
                for fact in extraction.get("facts", []):
                    conn.execute(
                        """INSERT INTO customer_facts
                        (id, account_id, interaction_id, kind, content, status,
                         confidence, evidence_json, created_at)
                        VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
                        (
                            _id("cfact"), proposal["account_id"], interaction_id,
                            fact["kind"], fact["content"], float(fact.get("confidence", 0.8)),
                            _json(fact.get("evidence", {})), timestamp,
                        ),
                    )
                for action in extraction.get("actions", []):
                    conn.execute(
                        """INSERT INTO customer_actions
                        (id, account_id, interaction_id, description, owner, due_at,
                         status, evidence_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                        (
                            _id("cact"), proposal["account_id"], interaction_id,
                            action["description"], action.get("owner", ""),
                            action.get("due_at"), _json(action.get("evidence", {})),
                            timestamp, timestamp,
                        ),
                    )
                conn.execute(
                    """UPDATE customer_update_proposals
                    SET status = 'approved', extraction_json = ?, decided_at = ?
                    WHERE id = ?""",
                    (_json(extraction), timestamp, proposal_id),
                )
                conn.execute(
                    "UPDATE customer_sources SET status = 'saved', updated_at = ? WHERE id = ?",
                    (timestamp, source["id"]),
                )
                conn.execute(
                    "UPDATE customer_accounts SET updated_at = ? WHERE id = ?",
                    (timestamp, proposal["account_id"]),
                )
                return True

        if not await self._call(operation):
            return None
        return await self.get_customer_proposal(proposal_id)

    async def customer_account_data(self, account_id: str) -> dict[str, Any] | None:
        account = await self.get_customer_account(account_id)
        if account is None:
            return None

        def operation() -> dict[str, list[dict[str, Any]]]:
            with self._lock:
                conn = self._connection()
                def rows(query: str) -> list[dict[str, Any]]:
                    return [dict(row) for row in conn.execute(query, (account_id,)).fetchall()]
                return {
                    "interactions": rows(
                        "SELECT * FROM customer_interactions WHERE account_id = ? "
                        "ORDER BY occurred_at DESC"
                    ),
                    "facts": rows(
                        "SELECT * FROM customer_facts WHERE account_id = ? "
                        "ORDER BY created_at DESC"
                    ),
                    "actions": rows(
                        "SELECT * FROM customer_actions WHERE account_id = ? "
                        "ORDER BY CASE status WHEN 'open' THEN 0 ELSE 1 END, due_at, created_at DESC"
                    ),
                    "people": rows(
                        "SELECT * FROM customer_people WHERE account_id = ? ORDER BY name"
                    ),
                    "sources": rows(
                        "SELECT * FROM customer_sources WHERE account_id = ? ORDER BY created_at DESC"
                    ),
                    "wins": [
                        self._win_row(item) for item in rows(
                            "SELECT * FROM customer_wins WHERE account_id = ? "
                            "ORDER BY COALESCE(won_at, created_at) DESC"
                        )
                    ],
                    "notes": [
                        self._note_row(item) for item in rows(
                            "SELECT * FROM customer_notes WHERE account_id = ? "
                            "ORDER BY pinned DESC, updated_at DESC"
                        )
                    ],
                }

        return {"account": account, **await self._call(operation)}

    async def update_customer_action(
        self, action_id: str, status: str
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                conn.execute(
                    "UPDATE customer_actions SET status = ?, updated_at = ? WHERE id = ?",
                    (status, timestamp, action_id),
                )
                return conn.execute(
                    "SELECT * FROM customer_actions WHERE id = ?", (action_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    # ── Hand edits ─────────────────────────────────────────────────────────
    # Everything an extraction can write, a person can correct. A model reading
    # a note is a first draft of the record, never the last word on it, so each
    # derived row is directly creatable, editable, and removable. Edits keep the
    # row's evidence untouched: the quote records where the claim came from, and
    # rewriting the claim does not rewrite its history.

    async def create_customer_fact(
        self, account_id: str, *, kind: str, content: str
    ) -> dict[str, Any] | None:
        fact_id, timestamp = _id("cfact"), _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM customer_accounts WHERE id = ?", (account_id,)
                ).fetchone() is None:
                    return None
                conn.execute(
                    """INSERT INTO customer_facts
                    (id, account_id, interaction_id, kind, content, status,
                     confidence, evidence_json, created_at)
                    VALUES (?, ?, NULL, ?, ?, 'active', 1.0, '{}', ?)""",
                    (fact_id, account_id, kind, content.strip(), timestamp),
                )
                conn.execute(
                    "UPDATE customer_accounts SET updated_at = ? WHERE id = ?",
                    (timestamp, account_id),
                )
                return conn.execute(
                    "SELECT * FROM customer_facts WHERE id = ?", (fact_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def update_customer_fact(
        self, fact_id: str, *, kind: str, content: str, status: str
    ) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                cursor = conn.execute(
                    "UPDATE customer_facts SET kind = ?, content = ?, status = ? WHERE id = ?",
                    (kind, content.strip(), status, fact_id),
                )
                if not cursor.rowcount:
                    return None
                return conn.execute(
                    "SELECT * FROM customer_facts WHERE id = ?", (fact_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def delete_customer_fact(self, fact_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                return conn.execute(
                    "DELETE FROM customer_facts WHERE id = ?", (fact_id,)
                ).rowcount > 0

        return await self._call(operation)

    async def create_customer_action(
        self, account_id: str, *, description: str, owner: str, due_at: str | None
    ) -> dict[str, Any] | None:
        action_id, timestamp = _id("cact"), _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM customer_accounts WHERE id = ?", (account_id,)
                ).fetchone() is None:
                    return None
                conn.execute(
                    """INSERT INTO customer_actions
                    (id, account_id, interaction_id, description, owner, due_at,
                     status, evidence_json, created_at, updated_at)
                    VALUES (?, ?, NULL, ?, ?, ?, 'open', '{}', ?, ?)""",
                    (
                        action_id, account_id, description.strip(), owner.strip(),
                        due_at, timestamp, timestamp,
                    ),
                )
                conn.execute(
                    "UPDATE customer_accounts SET updated_at = ? WHERE id = ?",
                    (timestamp, account_id),
                )
                return conn.execute(
                    "SELECT * FROM customer_actions WHERE id = ?", (action_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def edit_customer_action(
        self, action_id: str, *, description: str, owner: str,
        due_at: str | None, status: str,
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_actions SET description = ?, owner = ?,
                    due_at = ?, status = ?, updated_at = ? WHERE id = ?""",
                    (
                        description.strip(), owner.strip(), due_at, status,
                        timestamp, action_id,
                    ),
                )
                if not cursor.rowcount:
                    return None
                return conn.execute(
                    "SELECT * FROM customer_actions WHERE id = ?", (action_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def delete_customer_action(self, action_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                return conn.execute(
                    "DELETE FROM customer_actions WHERE id = ?", (action_id,)
                ).rowcount > 0

        return await self._call(operation)

    async def upsert_customer_person(
        self, account_id: str, *, name: str, role: str, organization: str
    ) -> dict[str, Any] | None:
        """Add or correct a contact. Name is unique per account, so re-adding an
        existing person is an edit of that person rather than a second card."""
        person_id, timestamp = _id("cp"), _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM customer_accounts WHERE id = ?", (account_id,)
                ).fetchone() is None:
                    return None
                conn.execute(
                    """INSERT INTO customer_people
                    (id, account_id, name, role, organization, evidence_json,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, '{}', ?, ?)
                    ON CONFLICT(account_id, name) DO UPDATE SET
                        role = excluded.role,
                        organization = excluded.organization,
                        updated_at = excluded.updated_at""",
                    (
                        person_id, account_id, name.strip(), role.strip(),
                        organization.strip(), timestamp, timestamp,
                    ),
                )
                return conn.execute(
                    "SELECT * FROM customer_people WHERE account_id = ? AND name = ?",
                    (account_id, name.strip()),
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def rename_customer_person(
        self, person_id: str, *, name: str, role: str, organization: str
    ) -> dict[str, Any] | None:
        """Edit a contact in place, including their name.

        A rename can collide with another contact on the same account; the
        UNIQUE constraint is left to say so rather than silently merging two
        people into one.
        """
        timestamp = _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_people SET name = ?, role = ?,
                    organization = ?, updated_at = ? WHERE id = ?""",
                    (
                        name.strip(), role.strip(), organization.strip(),
                        timestamp, person_id,
                    ),
                )
                if not cursor.rowcount:
                    return None
                return conn.execute(
                    "SELECT * FROM customer_people WHERE id = ?", (person_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def delete_customer_person(self, person_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                return conn.execute(
                    "DELETE FROM customer_people WHERE id = ?", (person_id,)
                ).rowcount > 0

        return await self._call(operation)

    async def update_customer_source(
        self, source_id: str, *, title: str, content: str, source_kind: str,
        occurred_at: str | None,
    ) -> dict[str, Any] | None:
        """Correct a captured note.

        Records already extracted from it are left alone: they were reviewed and
        saved as their own facts, and rewriting the note is not a decision to
        withdraw them. The content hash is recomputed so the edited text keeps
        de-duplicating future captures; colliding with another note on the same
        account raises, because two identical sources is exactly what the unique
        constraint exists to prevent.
        """
        timestamp = _now()
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_sources SET title = ?, content = ?,
                    content_hash = ?, source_kind = ?, occurred_at = ?,
                    updated_at = ? WHERE id = ?""",
                    (
                        title.strip(), content.strip(), digest, source_kind,
                        occurred_at, timestamp, source_id,
                    ),
                )
                if not cursor.rowcount:
                    return None
                return conn.execute(
                    "SELECT * FROM customer_sources WHERE id = ?", (source_id,)
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def delete_customer_source(self, source_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                return conn.execute(
                    "DELETE FROM customer_sources WHERE id = ?", (source_id,)
                ).rowcount > 0

        return await self._call(operation)

    # ── Direct notes ───────────────────────────────────────────────────────

    @staticmethod
    def _note_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        value = dict(row)
        value["pinned"] = bool(value["pinned"])
        return value

    async def create_customer_note(
        self, account_id: str, *, title: str, body: str, pinned: bool,
        origin: str, origin_ref: str,
    ) -> dict[str, Any] | None:
        note_id, timestamp = _id("cnote"), _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM customer_accounts WHERE id = ?", (account_id,)
                ).fetchone() is None:
                    return None
                conn.execute(
                    """INSERT INTO customer_notes
                    (id, account_id, title, body, pinned, origin, origin_ref,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        note_id, account_id, title.strip(), body.strip(),
                        int(pinned), origin, origin_ref.strip(), timestamp, timestamp,
                    ),
                )
                conn.execute(
                    "UPDATE customer_accounts SET updated_at = ? WHERE id = ?",
                    (timestamp, account_id),
                )
                return conn.execute(
                    "SELECT * FROM customer_notes WHERE id = ?", (note_id,)
                ).fetchone()

        row = await self._call(operation)
        return self._note_row(row) if row else None

    async def update_customer_note(
        self, note_id: str, *, title: str, body: str, pinned: bool
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> sqlite3.Row | None:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_notes SET title = ?, body = ?, pinned = ?,
                    updated_at = ? WHERE id = ?""",
                    (title.strip(), body.strip(), int(pinned), timestamp, note_id),
                )
                if not cursor.rowcount:
                    return None
                return conn.execute(
                    "SELECT * FROM customer_notes WHERE id = ?", (note_id,)
                ).fetchone()

        row = await self._call(operation)
        return self._note_row(row) if row else None

    async def delete_customer_note(self, note_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                return conn.execute(
                    "DELETE FROM customer_notes WHERE id = ?", (note_id,)
                ).rowcount > 0

        return await self._call(operation)

    # ── Cross-account search ───────────────────────────────────────────────

    async def search_customer_records(
        self, query: str, *, limit: int = 40
    ) -> tuple[list[dict[str, Any]], bool]:
        """Find anything customer-scoped that mentions `query`.

        A substring scan rather than an FTS index: this store holds a personal
        book of business — hundreds of accounts and thousands of rows — where
        one pass over six small tables is immediate, and an FTS index would mean
        six sets of sync triggers to keep honest for no felt gain. It also keeps
        partial words ("Coher") matching, which is what a search-as-you-type box
        is actually asked to do.
        """
        needle = query.strip().lower()
        if not needle:
            return [], False
        # LIKE's own wildcards have to be neutralised, or a stray "%" from the
        # user's query would silently match every row.
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        def operation() -> list[dict[str, Any]]:
            with self._lock:
                return [
                    dict(row) for row in self._connection().execute(
                        """SELECT * FROM (
                        SELECT 'account' AS kind, a.id AS id, a.id AS account_id,
                               a.name AS account_name, a.name AS title,
                               TRIM(a.industry || ' ' || a.region) AS snippet,
                               a.updated_at AS at
                        FROM customer_accounts a
                        WHERE LOWER(a.name) LIKE :q ESCAPE '\\'
                           OR LOWER(a.aliases_json) LIKE :q ESCAPE '\\'
                           OR LOWER(a.industry) LIKE :q ESCAPE '\\'
                           OR LOWER(a.region) LIKE :q ESCAPE '\\'
                        UNION ALL
                        SELECT 'note', n.id, n.account_id, a.name,
                               CASE WHEN n.title != '' THEN n.title ELSE 'Note' END,
                               n.body, n.updated_at
                        FROM customer_notes n JOIN customer_accounts a ON a.id = n.account_id
                        WHERE LOWER(n.title) LIKE :q ESCAPE '\\'
                           OR LOWER(n.body) LIKE :q ESCAPE '\\'
                        UNION ALL
                        SELECT 'win', w.id, w.account_id, a.name, w.title,
                               TRIM(w.brief || ' ' || w.dac_shape), COALESCE(w.won_at, w.created_at)
                        FROM customer_wins w JOIN customer_accounts a ON a.id = w.account_id
                        WHERE LOWER(w.title) LIKE :q ESCAPE '\\'
                           OR LOWER(w.brief) LIKE :q ESCAPE '\\'
                           OR LOWER(w.dac_shape) LIKE :q ESCAPE '\\'
                           OR LOWER(w.services_json) LIKE :q ESCAPE '\\'
                        UNION ALL
                        SELECT 'fact', f.id, f.account_id, a.name, f.kind,
                               f.content, f.created_at
                        FROM customer_facts f JOIN customer_accounts a ON a.id = f.account_id
                        WHERE LOWER(f.content) LIKE :q ESCAPE '\\'
                        UNION ALL
                        SELECT 'action', c.id, c.account_id, a.name,
                               c.description, c.owner, c.created_at
                        FROM customer_actions c JOIN customer_accounts a ON a.id = c.account_id
                        WHERE LOWER(c.description) LIKE :q ESCAPE '\\'
                           OR LOWER(c.owner) LIKE :q ESCAPE '\\'
                        UNION ALL
                        SELECT 'source', s.id, s.account_id, a.name, s.title,
                               s.content, COALESCE(s.occurred_at, s.created_at)
                        FROM customer_sources s JOIN customer_accounts a ON a.id = s.account_id
                        WHERE LOWER(s.title) LIKE :q ESCAPE '\\'
                           OR LOWER(s.content) LIKE :q ESCAPE '\\'
                        )
                        ORDER BY CASE kind
                            WHEN 'account' THEN 0 WHEN 'note' THEN 1 WHEN 'win' THEN 2
                            WHEN 'fact' THEN 3 WHEN 'action' THEN 4 ELSE 5 END,
                            at DESC
                        LIMIT :limit""",
                        {"q": pattern, "limit": limit + 1},
                    ).fetchall()
                ]

        rows = await self._call(operation)
        return rows[:limit], len(rows) > limit

    @staticmethod
    def _win_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        value = dict(row)
        value["services"] = _loads(value.pop("services_json"), [])
        return value

    async def create_customer_win(
        self, account_id: str, *, title: str, brief: str, services: list[str],
        dac_shape: str, yearly_arr: float | None, won_at: str | None,
        source_ref: str,
    ) -> dict[str, Any]:
        win_id, timestamp = _id("cwin"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO customer_wins
                    (id, account_id, title, brief, services_json, dac_shape,
                     yearly_arr, won_at, source_ref, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        win_id, account_id, title.strip(), brief.strip(),
                        _json(services), dac_shape.strip(), yearly_arr, won_at,
                        source_ref.strip(), timestamp, timestamp,
                    ),
                )
                conn.execute(
                    "UPDATE customer_accounts SET updated_at = ? WHERE id = ?",
                    (timestamp, account_id),
                )

        await self._call(operation)
        return (await self.get_customer_win(win_id))  # type: ignore[return-value]

    async def update_customer_win(
        self, win_id: str, *, title: str, brief: str, services: list[str],
        dac_shape: str, yearly_arr: float | None, won_at: str | None,
        source_ref: str,
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_wins SET title = ?, brief = ?,
                    services_json = ?, dac_shape = ?, yearly_arr = ?, won_at = ?,
                    source_ref = ?, updated_at = ? WHERE id = ?""",
                    (
                        title.strip(), brief.strip(), _json(services),
                        dac_shape.strip(), yearly_arr, won_at, source_ref.strip(),
                        timestamp, win_id,
                    ),
                )
                return cursor.rowcount > 0

        return await self.get_customer_win(win_id) if await self._call(operation) else None

    async def delete_customer_win(self, win_id: str) -> bool:
        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM customer_wins WHERE id = ?", (win_id,)
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def get_customer_win(self, win_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    """SELECT w.*, a.name AS account_name FROM customer_wins w
                    JOIN customer_accounts a ON a.id = w.account_id
                    WHERE w.id = ?""",
                    (win_id,),
                ).fetchone()

        row = await self._call(operation)
        return self._win_row(row) if row else None

    async def set_customer_win_arr(self, win_id: str, yearly_arr: float) -> bool:
        """Write an accepted estimate through to the win's own ARR figure."""
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    "UPDATE customer_wins SET yearly_arr = ?, updated_at = ? WHERE id = ?",
                    (yearly_arr, timestamp, win_id),
                )
                return cursor.rowcount > 0

        return await self._call(operation)

    async def upsert_win_valuation(
        self, win_id: str, *, estimated_yearly_arr: float | None, currency: str,
        lines: list[dict[str, Any]], explanation: str, confidence: str,
        unpriced: list[str], rates_verified: bool, model_used: str | None,
        prompt_version: str,
    ) -> dict[str, Any]:
        """Store the latest estimate for a win, replacing any earlier one.

        Re-estimating resets the status to 'proposed': a fresh reading of the
        notes has not been reviewed yet, whatever the user decided about the
        previous one.
        """
        valuation_id, timestamp = _id("cval"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO customer_win_valuations
                    (id, win_id, estimated_yearly_arr, currency, lines_json,
                     explanation, confidence, unpriced_json, rates_verified,
                     model_used, prompt_version, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)
                    ON CONFLICT(win_id) DO UPDATE SET
                        estimated_yearly_arr = excluded.estimated_yearly_arr,
                        currency = excluded.currency,
                        lines_json = excluded.lines_json,
                        explanation = excluded.explanation,
                        confidence = excluded.confidence,
                        unpriced_json = excluded.unpriced_json,
                        rates_verified = excluded.rates_verified,
                        model_used = excluded.model_used,
                        prompt_version = excluded.prompt_version,
                        status = 'proposed',
                        updated_at = excluded.updated_at""",
                    (
                        valuation_id, win_id, estimated_yearly_arr, currency,
                        _json(lines), explanation.strip(), confidence,
                        _json(unpriced), int(rates_verified), model_used,
                        prompt_version, timestamp, timestamp,
                    ),
                )

        await self._call(operation)
        return (await self.get_win_valuation(win_id))  # type: ignore[return-value]

    async def get_win_valuation(self, win_id: str) -> dict[str, Any] | None:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM customer_win_valuations WHERE win_id = ?",
                    (win_id,),
                ).fetchone()

        row = await self._call(operation)
        return dict(row) if row else None

    async def set_win_valuation_status(
        self, win_id: str, status: str
    ) -> dict[str, Any] | None:
        timestamp = _now()

        def operation() -> bool:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """UPDATE customer_win_valuations SET status = ?, updated_at = ?
                    WHERE win_id = ?""",
                    (status, timestamp, win_id),
                )
                return cursor.rowcount > 0

        return await self.get_win_valuation(win_id) if await self._call(operation) else None

    async def win_valuations_for(self, win_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Every stored estimate for the given wins, keyed by win id."""
        if not win_ids:
            return {}

        def operation() -> list[sqlite3.Row]:
            with self._lock:
                placeholders = ",".join("?" for _ in win_ids)
                return self._connection().execute(
                    f"SELECT * FROM customer_win_valuations WHERE win_id IN ({placeholders})",
                    tuple(win_ids),
                ).fetchall()

        return {str(row["win_id"]): dict(row) for row in await self._call(operation)}

    async def customer_dashboard_data(self) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            with self._lock:
                conn = self._connection()
                now = _now()
                counts = conn.execute(
                    """SELECT
                    (SELECT COUNT(*) FROM customer_accounts WHERE status = 'active') active_accounts,
                    (SELECT COUNT(*) FROM customer_actions WHERE status = 'open') open_actions,
                    (SELECT COUNT(*) FROM customer_actions
                     WHERE status = 'open' AND due_at IS NOT NULL AND due_at < ?) overdue_actions,
                    (SELECT COUNT(*) FROM customer_sources WHERE status = 'waiting') waiting_notes""",
                    (now,),
                ).fetchone()
                # The attention queue is read across accounts, so each action
                # arrives with the customer it belongs to; overdue first, then
                # due-soon, then the undated backlog.
                actions = [
                    dict(row) for row in conn.execute(
                        """SELECT c.*, a.name AS account_name
                        FROM customer_actions c
                        JOIN customer_accounts a ON a.id = c.account_id
                        WHERE c.status = 'open'
                        ORDER BY CASE
                            WHEN c.due_at IS NOT NULL AND c.due_at < ? THEN 0
                            WHEN c.due_at IS NOT NULL THEN 1 ELSE 2 END,
                            c.due_at, c.created_at
                        LIMIT 25""",
                        (now,),
                    ).fetchall()
                ]
                ids = [
                    str(row["id"]) for row in conn.execute(
                        "SELECT id FROM customer_accounts ORDER BY updated_at DESC LIMIT 5"
                    ).fetchall()
                ]
                win_totals = conn.execute(
                    """SELECT COUNT(*) AS total_wins,
                    COALESCE(SUM(yearly_arr), 0) AS total_yearly_arr
                    FROM customer_wins"""
                ).fetchone()
                win_services = [
                    (str(row["services_json"]), str(row["dac_shape"] or ""))
                    for row in conn.execute(
                        "SELECT services_json, dac_shape FROM customer_wins"
                    ).fetchall()
                ]
                recent_wins = [
                    dict(row) for row in conn.execute(
                        """SELECT w.*, a.name AS account_name FROM customer_wins w
                        JOIN customer_accounts a ON a.id = w.account_id
                        ORDER BY COALESCE(w.won_at, w.created_at) DESC LIMIT 6"""
                    ).fetchall()
                ]
                return {
                    **dict(counts), "priority_actions": actions, "ids": ids,
                    **dict(win_totals), "win_services": win_services,
                    "recent_wins": recent_wins,
                }

        value = await self._call(operation)
        accounts = await asyncio.gather(
            *(self.get_customer_account(item) for item in value.pop("ids"))
        )
        value["recent_accounts"] = [item for item in accounts if item]
        wins_by_service: dict[str, int] = {}
        dac_wins = 0
        for raw, dac_shape in value.pop("win_services"):
            services = [str(item) for item in _loads(raw, [])]
            # A recorded DAC shape counts on its own: the record-win form offers
            # it as the way to mark a DAC win, and a win can be a DAC deal before
            # anyone has tagged the service.
            if "DAC" in services or dac_shape.strip():
                dac_wins += 1
            for service in services:
                wins_by_service[service] = wins_by_service.get(service, 0) + 1
        value["wins_by_service"] = wins_by_service
        value["dac_wins"] = dac_wins
        value["recent_wins"] = [
            self._win_row(item) for item in value["recent_wins"]
        ]
        return value

    async def customer_settings(self) -> dict[str, Any]:
        def operation() -> sqlite3.Row | None:
            with self._lock:
                return self._connection().execute(
                    "SELECT * FROM customer_settings WHERE id = 1"
                ).fetchone()

        row = await self._call(operation)
        if row:
            value = dict(row)
            value.pop("id", None)
            return value
        return {
            "tracker_url": "", "activity_template": "", "updated_at": None
        }

    async def save_customer_settings(
        self, tracker_url: str, activity_template: str
    ) -> dict[str, Any]:
        timestamp = _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO customer_settings
                    (id, tracker_url, activity_template, updated_at)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET tracker_url = excluded.tracker_url,
                    activity_template = excluded.activity_template,
                    updated_at = excluded.updated_at""",
                    (tracker_url, activity_template, timestamp),
                )

        await self._call(operation)
        return await self.customer_settings()

    async def create_customer_output(
        self, account_id: str, interaction_id: str | None, kind: str, content: str
    ) -> dict[str, Any]:
        output_id, timestamp = _id("cout"), _now()

        def operation() -> None:
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO customer_outputs
                    (id, account_id, interaction_id, kind, content, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (output_id, account_id, interaction_id, kind, content, timestamp),
                )

        await self._call(operation)
        return {
            "id": output_id, "account_id": account_id, "kind": kind,
            "content": content, "created_at": timestamp,
        }
