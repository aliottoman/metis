"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  activateToolVersion,
  ApiError,
  decideToolImprovement,
  decideToolProposal,
  getToolImprovementEvidence,
  getToolVersionEvidence,
  listToolDefinitionBuilds,
  listToolDefinitionProposals,
  listToolDefinitions,
  listToolImprovementProposals,
  listTools,
  listToolVersions,
} from "@/lib/api";
import type {
  CapabilityProfile,
  ModelAccess,
  ToolDefinitionBuild,
  ToolDefinitionProposal,
  ToolDefinitionRecord,
  ToolImprovementEvidence,
  ToolImprovementProposal,
  ToolRecord,
  ToolVersion,
  ToolVersionEvidence,
} from "@/lib/types";

type Filter = "all" | "pending" | "active" | "drafts";

function toolGroup(tool: ToolRecord): Exclude<Filter, "all"> {
  if (tool.state === "active") return "active";
  if (["quarantined", "evaluated", "approved"].includes(tool.state)) return "pending";
  return "drafts";
}

function shortHash(value?: string): string {
  return value ? value.slice(0, 10) : "not recorded";
}

function modelAccessLabel(access: ModelAccess): string {
  if (!access.enabled) return "none";
  const roles = access.roles.length ? access.roles.join(", ") : "any";
  return `≤${access.max_calls_per_run} calls/run · roles: ${roles} · ≤${access.max_tokens_per_call} tokens/call`;
}

function runtimeAllowlistLabel(allowlists: CapabilityProfile["runtime_allowlists"]): string {
  const entries = Object.entries(allowlists);
  if (!entries.length) return "none";
  return entries.map(([key, value]) => (value ? `${key}: ${value}` : key)).join(", ");
}

export function ToolWorkshop() {
  const [tools, setTools] = useState<ToolRecord[]>([]);
  const [definitions, setDefinitions] = useState<ToolDefinitionRecord[]>([]);
  const [definitionProposals, setDefinitionProposals] = useState<ToolDefinitionProposal[]>([]);
  const [builds, setBuilds] = useState<ToolDefinitionBuild[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [versions, setVersions] = useState<{ tool: ToolRecord; items: ToolVersion[] } | null>(null);
  const [improvements, setImprovements] = useState<ToolImprovementProposal[]>([]);
  const [activationTarget, setActivationTarget] = useState<{
    tool: ToolRecord;
    version: ToolVersion;
    idempotencyKey: string;
    reason: string;
  } | null>(null);
  const [activationBusy, setActivationBusy] = useState(false);
  const [activationError, setActivationError] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<{
    title: string;
    base: ToolVersionEvidence;
    eligible: ToolVersionEvidence[];
    proposal?: ToolImprovementProposal;
  } | null>(null);
  const [evidenceBusyId, setEvidenceBusyId] = useState<string | null>(null);
  const [decisionTarget, setDecisionTarget] = useState<{
    proposal: ToolImprovementProposal;
    decision: "approve" | "reject";
    targetVersionId?: string;
    label: string;
    reason: string;
    idempotencyKey: string;
  } | null>(null);
  const [decisionBusy, setDecisionBusy] = useState(false);
  const [decisionNotice, setDecisionNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [registry, improvementInbox, definitionRecords, definitionInbox, definitionBuilds] = await Promise.all([
        listTools(),
        listToolImprovementProposals(),
        listToolDefinitions(),
        listToolDefinitionProposals("pending"),
        listToolDefinitionBuilds(),
      ]);
      const enriched = await Promise.all(registry.map(async (tool) => {
        try {
          const toolVersions = await listToolVersions(tool.id);
          const latest = toolVersions[0];
          if (!latest) return tool;
          return {
            ...tool,
            latest_version: latest.version,
            content_hash: latest.content_hash,
            risk_level: latest.risk_level ?? tool.risk_level,
            permissions: latest.permissions?.length ? latest.permissions : tool.permissions,
            evaluation: latest.evaluation,
          };
        } catch {
          return tool;
        }
      }));
      setTools(enriched);
      setImprovements(improvementInbox.filter((proposal) => proposal.status === "pending"));
      setDefinitions(definitionRecords);
      setDefinitionProposals(definitionInbox);
      setBuilds(definitionBuilds);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the tool registry.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  const counts = useMemo(() => ({
    all: tools.length,
    pending: tools.filter((tool) => toolGroup(tool) === "pending").length,
    active: tools.filter((tool) => toolGroup(tool) === "active").length,
    drafts: tools.filter((tool) => toolGroup(tool) === "drafts").length,
  }), [tools]);

  const visibleTools = filter === "all" ? tools : tools.filter((tool) => toolGroup(tool) === filter);
  const evaluatedBuilds = builds.filter((build) => build.status === "evaluated");

  async function decide(tool: ToolRecord, decision: "approve" | "reject") {
    const proposalId = tool.proposal_id ?? tool.id;
    setBusyId(tool.id);
    setError(null);
    try {
      await decideToolProposal(proposalId, decision);
      setTools((current) => current.map((item) => item.id === tool.id ? {
        ...item,
        state: decision === "approve" ? "approved" : "rejected",
      } : item));
      await load();
      window.setTimeout(() => void load(), 1200);
    } catch (actionError) {
      setError(actionError instanceof ApiError ? actionError.message : "The proposal could not be updated.");
    } finally {
      setBusyId(null);
    }
  }

  async function showVersions(tool: ToolRecord) {
    setBusyId(tool.id);
    try {
      setVersions({ tool, items: await listToolVersions(tool.id) });
    } catch (versionError) {
      setError(versionError instanceof Error ? versionError.message : "Could not load version history.");
    } finally {
      setBusyId(null);
    }
  }

  async function showVersionEvidence(tool: ToolRecord, version: ToolVersion) {
    setEvidenceBusyId(version.id);
    setError(null);
    try {
      const value = await getToolVersionEvidence(tool.id, version.id);
      setEvidence({ title: `${tool.name} ${version.version}`, base: value, eligible: [] });
    } catch (evidenceError) {
      setError(evidenceError instanceof Error ? evidenceError.message : "Could not verify tool evidence.");
    } finally {
      setEvidenceBusyId(null);
    }
  }

  async function showImprovementEvidence(proposal: ToolImprovementProposal) {
    setEvidenceBusyId(proposal.id);
    setError(null);
    try {
      const value: ToolImprovementEvidence = await getToolImprovementEvidence(proposal.id);
      const tool = tools.find((item) => item.id === proposal.tool_id);
      setEvidence({
        title: `${tool?.name ?? "Tool"} correction evidence`,
        base: value.base_version,
        eligible: value.eligible_revisions,
        proposal: value.proposal,
      });
    } catch (evidenceError) {
      setError(evidenceError instanceof Error ? evidenceError.message : "Could not verify correction evidence.");
    } finally {
      setEvidenceBusyId(null);
    }
  }

  function prepareImprovementDecision(
    proposal: ToolImprovementProposal,
    decision: "approve" | "reject",
    targetVersionId?: string,
  ) {
    setDecisionTarget({
      proposal,
      decision,
      targetVersionId,
      label: decision === "reject"
        ? "Reject correction"
        : targetVersionId
          ? "Approve exact evaluated revision"
          : "Approve and queue revision",
      reason: "",
      idempotencyKey: `improvement-${proposal.id}-${decision}-${crypto.randomUUID()}`,
    });
  }

  async function confirmImprovementDecision() {
    if (!decisionTarget || !decisionTarget.reason.trim() || decisionBusy) return;
    setDecisionBusy(true);
    setError(null);
    try {
      const result = await decideToolImprovement(
        decisionTarget.proposal.id,
        decisionTarget.decision,
        decisionTarget.idempotencyKey,
        decisionTarget.reason.trim(),
        decisionTarget.targetVersionId,
      );
      setDecisionNotice(
        result.outcome === "revision_queued"
          ? "Correction approved. A governed revision request was queued; the active version was not changed."
          : result.outcome === "revision_activated"
            ? "The exact evaluated immutable revision was activated."
            : "Correction rejected. The active version was not changed.",
      );
      setDecisionTarget(null);
      setEvidence(null);
      await load();
    } catch (decisionError) {
      setError(decisionError instanceof Error ? decisionError.message : "The correction decision could not be saved.");
    } finally {
      setDecisionBusy(false);
    }
  }

  async function confirmVersionActivation() {
    if (!activationTarget || !activationTarget.reason.trim() || activationBusy) return;
    setActivationBusy(true);
    setActivationError(null);
    setError(null);
    try {
      await activateToolVersion(
        activationTarget.tool.id,
        activationTarget.version.id,
        activationTarget.idempotencyKey,
        activationTarget.reason.trim(),
      );
      setActivationTarget(null);
      setVersions(null);
      await load();
    } catch (activationError) {
      const message = activationError instanceof ApiError ? activationError.message : "The prior version could not be activated.";
      setActivationError(message);
      setError(message);
    } finally {
      setActivationBusy(false);
    }
  }

  return (
    <div className="workspacePage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Governed capabilities</span>
          <h1>Tool Workshop</h1>
          <p>See what Metis can reuse, how each tool is contained, and the evidence behind every active version.</p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh registry"}
        </button>
      </header>

      <section className="statStrip" aria-label="Tool registry summary">
        <div><strong>{counts.active}</strong><span>Active tools</span></div>
        <div><strong>{counts.pending}</strong><span>Awaiting review</span></div>
        <div><strong>{counts.drafts}</strong><span>Drafts</span></div>
        <div><strong>{tools.reduce((sum, tool) => sum + (tool.evaluation?.total ?? 0), 0)}</strong><span>Evaluations run</span></div>
      </section>

      <section className="toolTrustBanner" aria-label="Trusted tool lifecycle">
        <span className="toolTrustMark" aria-hidden="true">✓</span>
        <div><strong>Explicit build requests go straight through</strong><p>Metis can define, evaluate, activate, and use a tool in one run when it remains local, contained, and network-free. Anything broader or inferred still pauses for review.</p></div>
        <span className="toolTrustTag">Trusted fast path</span>
      </section>

      <div className="filterTabs" role="tablist" aria-label="Tool status filter">
        {(["all", "pending", "active", "drafts"] as Filter[]).map((item) => (
          <button
            key={item}
            type="button"
            role="tab"
            aria-selected={filter === item}
            onClick={() => setFilter(item)}
          >
            {item === "all" ? "All" : item[0]!.toUpperCase() + item.slice(1)}
            <span>{counts[item]}</span>
          </button>
        ))}
      </div>

      {error ? <div className="notice errorNotice" role="alert"><strong>Workshop unavailable</strong><span>{error}</span></div> : null}
      {decisionNotice ? <div className="notice successNotice" role="status"><strong>Decision recorded</strong><span>{decisionNotice}</span></div> : null}

      <section className="improvementSection" aria-labelledby="improvement-title">
        <header>
          <div>
            <span className="eyebrow">Correction inbox</span>
            <h2 id="improvement-title">Proposed regressions</h2>
            <p>Corrections from real tool runs become reviewable regression cases. They do not modify the active version.</p>
          </div>
          <span className="sectionBadge">{improvements.length} pending</span>
        </header>
        {improvements.length ? (
          <div className="improvementGrid">
            {improvements.map((proposal) => {
              const tool = tools.find((item) => item.id === proposal.tool_id);
              return (
                <article className="improvementCard" key={proposal.id}>
                  <div className="improvementTopline">
                    <span>{tool?.name ?? "Tool correction"}</span>
                    <code>{shortHash(proposal.content_hash)}</code>
                  </div>
                  <blockquote>{proposal.correction}</blockquote>
                  <div className="regressionCase">
                    <strong>{proposal.regression_eval.name}</strong>
                    {proposal.regression_eval.expected_properties.map((property) => <span key={property}>✓ {property}</span>)}
                  </div>
                  <footer>
                    <span>Source run <code>{proposal.source_run_id.slice(0, 12)}</code></span>
                    <div className="improvementActions">
                      <button className="textButton" type="button" disabled={evidenceBusyId === proposal.id} onClick={() => void showImprovementEvidence(proposal)}>
                        {evidenceBusyId === proposal.id ? "Verifying…" : "Review evidence"}
                      </button>
                      <button className="dangerButton" type="button" onClick={() => prepareImprovementDecision(proposal, "reject")}>Reject</button>
                      <button className="primaryButton" type="button" onClick={() => prepareImprovementDecision(proposal, "approve")}>Queue revision</button>
                    </div>
                  </footer>
                </article>
              );
            })}
          </div>
        ) : (
          <p className="improvementEmpty">No pending correction regressions. New proposals appear here after corrective feedback on a tool-backed run.</p>
        )}
      </section>

      <section className="registrySection" aria-labelledby="registry-title">
        <header>
          <div>
            <span className="eyebrow">Tool Factory v2</span>
            <h2 id="registry-title">Tool Registry</h2>
            <p>Versioned definitions and their exact capability grants. “Defined” means the scope was approved; “Runnable” means the evaluated build is ready.</p>
          </div>
          <span className="sectionBadge">{definitions.length} registered</span>
        </header>

        {definitions.length ? (
          <div className="registryGrid">
            {definitions.map((record) => {
              const definition = record.definition;
              const capability = definition.capability_profile;
              return (
                <article className="toolCard registryCard" key={`${definition.slug}@${definition.version}`}>
                  <div className="toolCardTop">
                    <span className="toolIcon">{(definition.name || definition.slug || "?").slice(0, 1).toUpperCase()}</span>
                    <div className="toolIdentity">
                      <div>
                        <h2>{definition.name || definition.slug}</h2>
                        <span className={`statusPill status-${definition.status}`}>{definition.status}</span>
                      </div>
                      <p>{definition.description || "No description supplied."}</p>
                    </div>
                  </div>

                  <div className="registryPills" aria-label="Lifecycle state">
                    {record.runnable ? <span className="statusPill status-active">Runnable</span> : null}
                    {record.buildable ? <span className="statusPill status-evaluated">Buildable</span> : null}
                    {record.pending_build ? <span className="statusPill status-pending">Pending build</span> : null}
                    {record.disabled ? <span className="statusPill status-deprecated">Disabled</span> : null}
                    {!record.runnable && !record.buildable && !record.pending_build && !record.disabled ? (
                      <span className="statusPill">Idle</span>
                    ) : null}
                  </div>

                  <dl className="toolMeta">
                    <div><dt>Slug</dt><dd className="mono">{definition.slug || "—"}</dd></div>
                    <div><dt>Version</dt><dd>{definition.version || "—"}</dd></div>
                    <div><dt>Archetype</dt><dd>{definition.archetype || "—"}</dd></div>
                    <div><dt>Content hash</dt><dd className="mono">{shortHash(definition.content_hash)}</dd></div>
                  </dl>

                  <div className="capabilityPanel">
                    <span className="capabilityHead">Capability profile</span>
                    <dl className="toolMeta capabilityMeta">
                      <div><dt>Code allowlist</dt><dd>{capability.code_allowlist || "—"}</dd></div>
                      <div><dt>Runtime allowlists</dt><dd>{runtimeAllowlistLabel(capability.runtime_allowlists)}</dd></div>
                      <div><dt>Model access</dt><dd>{modelAccessLabel(capability.model_access)}</dd></div>
                      <div><dt>Network</dt><dd>{capability.network}</dd></div>
                      <div><dt>Filesystem</dt><dd>{capability.filesystem}</dd></div>
                    </dl>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <p className="registryEmpty">No tool definitions yet. When Metis hardens a repeatable process into a declarative tool, it appears here with its full capability grant.</p>
        )}

        <div className="definitionInbox">
          <header>
            <div>
              <span className="eyebrow">Capability review</span>
              <h3>Definition inbox</h3>
            </div>
            <span className="inboxNote">Approve in chat</span>
          </header>
          {definitionProposals.length ? (
            <ul className="definitionInboxList">
              {definitionProposals.map((proposal) => (
                <li className="definitionInboxRow" key={proposal.id}>
                  <div className="definitionInboxIdentity">
                    <strong className="mono">{proposal.slug || "definition"}</strong>
                    <span className="definitionVersion">v{proposal.version || "?"}</span>
                  </div>
                  <p>{proposal.summary || "A drafted definition is awaiting a capability decision."}</p>
                  <span className={`statusPill status-${proposal.status}`}>{proposal.status}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="registryEmpty">No definitions are waiting for a capability decision.</p>
          )}
        </div>

        {evaluatedBuilds.length ? (
          <div className="definitionInbox">
            <header>
              <div>
                <span className="eyebrow">Expanded-access review</span>
                <h3>Builds awaiting activation</h3>
              </div>
              <span className="inboxNote">Outside trusted fast path</span>
            </header>
            <ul className="definitionInboxList">
              {evaluatedBuilds.map((build) => (
                <li className="definitionInboxRow" key={build.id}>
                  <div className="definitionInboxIdentity">
                    <strong className="mono">{build.slug || "tool"}</strong>
                    <span className="definitionVersion">v{build.version || "?"}</span>
                  </div>
                  <p>{build.eval_report
                    ? `${build.eval_report.passed ? "Passed" : "Failed"} · score ${build.eval_report.score.toFixed(2)}`
                    : "Evaluated build awaiting activation."}</p>
                  <span className={`statusPill status-${build.status}`}>{build.status}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <div className="toolGrid" aria-live="polite">
        {loading && !tools.length ? Array.from({ length: 3 }).map((_, index) => <div className="skeletonCard" key={index} />) : null}
        {!loading && !visibleTools.length ? (
          <div className="emptyPanel">
            <span className="emptyGlyph">◇</span>
            <h2>No tools in this view</h2>
            <p>When Metis discovers a repeatable workflow, its tested proposal will appear here.</p>
          </div>
        ) : null}
        {visibleTools.map((tool) => {
          const canDecide = toolGroup(tool) === "pending";
          const score = tool.evaluation?.score ?? (tool.evaluation?.total ? tool.evaluation.passed / tool.evaluation.total : undefined);
          return (
            <article className="toolCard" key={tool.id}>
              <div className="toolCardTop">
                <span className="toolIcon">{tool.name.slice(0, 1).toUpperCase()}</span>
                <div className="toolIdentity">
                  <div>
                    <h2>{tool.name}</h2>
                    <span className={`statusPill status-${tool.state}`}>{tool.state}</span>
                  </div>
                  <p>{tool.description || "No description supplied."}</p>
                </div>
              </div>

              <dl className="toolMeta">
                <div><dt>Version</dt><dd>{tool.active_version ?? tool.latest_version ?? "Draft"}</dd></div>
                <div><dt>Risk</dt><dd>{tool.risk_level ?? "R0"}</dd></div>
                <div><dt>Content hash</dt><dd className="mono">{shortHash(tool.content_hash)}</dd></div>
                <div><dt>Eval score</dt><dd>{score == null ? "Not run" : `${Math.round(score <= 1 ? score * 100 : score)}%`}</dd></div>
              </dl>

              {tool.permissions?.length ? (
                <div className="permissionList" aria-label="Requested permissions">
                  {tool.permissions.map((permission) => <span key={permission}>{permission}</span>)}
                </div>
              ) : null}

              {tool.evaluation ? (
                <div className="evalBar">
                  <span style={{ width: `${Math.max(2, (tool.evaluation.passed / Math.max(1, tool.evaluation.total)) * 100)}%` }} />
                  <small>{tool.evaluation.passed} passed · {tool.evaluation.failed} failed</small>
                </div>
              ) : null}

              <footer className="cardActions">
                <button className="textButton" type="button" onClick={() => void showVersions(tool)} disabled={busyId === tool.id}>
                  Version history
                </button>
                {canDecide ? (
                  <div>
                    <button className="dangerButton" type="button" onClick={() => void decide(tool, "reject")} disabled={busyId === tool.id}>Reject</button>
                    <button className="primaryButton" type="button" onClick={() => void decide(tool, "approve")} disabled={busyId === tool.id}>Activate</button>
                  </div>
                ) : null}
              </footer>
            </article>
          );
        })}
      </div>

      {versions ? (
        <div className="modalBackdrop" role="presentation" onMouseDown={() => setVersions(null)}>
          <section className="modalCard" role="dialog" aria-modal="true" aria-labelledby="versions-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div><span className="eyebrow">Immutable history</span><h2 id="versions-title">{versions.tool.name}</h2></div>
              <button className="iconButton" type="button" aria-label="Close" onClick={() => setVersions(null)}>×</button>
            </header>
            <div className="versionList">
              {versions.items.map((version) => (
                <div className="versionRow" key={version.id || version.version}>
                  <span className="versionDot" />
                  <div><strong>{version.version}</strong><small>{version.created_at ? new Date(version.created_at).toLocaleString() : "Timestamp unavailable"}</small></div>
                  <span className={`statusPill status-${version.state}`}>{version.state}</span>
                  <code>{shortHash(version.content_hash)}</code>
                  <div className="versionRowActions">
                    <button className="textButton" type="button" disabled={evidenceBusyId === version.id} onClick={() => void showVersionEvidence(versions.tool, version)}>
                      {evidenceBusyId === version.id ? "Verifying…" : "Evidence"}
                    </button>
                    {version.state === "approved" && version.id !== versions.tool.active_version ? (
                      <button
                        className="secondaryButton versionActivateButton"
                        type="button"
                        onClick={() => {
                          setActivationError(null);
                          setActivationTarget({
                            tool: versions.tool,
                            version,
                            idempotencyKey: `restore-${versions.tool.id}-${version.id}-${crypto.randomUUID()}`,
                            reason: "",
                          });
                        }}
                      >
                        Activate
                      </button>
                    ) : null}
                  </div>
                </div>
              ))}
              {!versions.items.length ? <p className="muted">No immutable versions have been recorded yet.</p> : null}
            </div>
          </section>
        </div>
      ) : null}

      {activationTarget ? (
        <div className="modalBackdrop activationBackdrop" role="presentation" onMouseDown={() => !activationBusy && setActivationTarget(null)}>
          <section className="modalCard activationDialog" role="dialog" aria-modal="true" aria-labelledby="activation-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div><span className="eyebrow">Explicit version change</span><h2 id="activation-title">Activate {activationTarget.version.version}?</h2></div>
              <button className="iconButton" type="button" aria-label="Close" disabled={activationBusy} onClick={() => setActivationTarget(null)}>×</button>
            </header>
            <div className="activationBody">
              {activationError ? <div className="notice errorNotice activationError" role="alert"><strong>Activation blocked</strong><span>{activationError}</span></div> : null}
              <div className="activationWarning">
                <span>!</span>
                <p>This switches <strong>{activationTarget.tool.name}</strong> from its current version to an older immutable version. The current version remains available for rollback.</p>
              </div>
              <dl>
                <div><dt>Target version</dt><dd>{activationTarget.version.version}</dd></div>
                <div><dt>Content hash</dt><dd><code>{activationTarget.version.content_hash ?? "Unavailable"}</code></dd></div>
                <div><dt>Evaluation</dt><dd>{activationTarget.version.evaluation ? `${Math.round((activationTarget.version.evaluation.score ?? 0) * 100)}%` : "Previously approved"}</dd></div>
              </dl>
              <label htmlFor="activation-reason">Reason for restoring this version</label>
              <textarea
                id="activation-reason"
                value={activationTarget.reason}
                maxLength={2000}
                autoFocus
                placeholder="For example: Restore the last known-good output while the current regression is reviewed."
                onChange={(event) => setActivationTarget((current) => current ? { ...current, reason: event.target.value } : null)}
              />
            </div>
            <footer className="activationActions">
              <button className="secondaryButton" type="button" disabled={activationBusy} onClick={() => setActivationTarget(null)}>Keep current version</button>
              <button className="primaryButton" type="button" disabled={!activationTarget.reason.trim() || activationBusy} onClick={() => void confirmVersionActivation()}>{activationBusy ? "Activating…" : "Confirm activation"}</button>
            </footer>
          </section>
        </div>
      ) : null}

      {evidence ? (
        <div className="modalBackdrop evidenceBackdrop" role="presentation" onMouseDown={() => setEvidence(null)}>
          <section className="modalCard evidenceDialog" role="dialog" aria-modal="true" aria-labelledby="evidence-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div><span className="eyebrow">Verified immutable bundle</span><h2 id="evidence-title">{evidence.title}</h2></div>
              <button className="iconButton" type="button" aria-label="Close" onClick={() => setEvidence(null)}>×</button>
            </header>
            <div className="evidenceBody">
              <div className="evidenceIntegrity"><span>✓</span><p><strong>Content-addressed snapshot verified</strong><small>{evidence.base.content_hash}</small></p></div>
              {evidence.base.evidence_truncated ? <p className="evidenceWarning">Some oversized evidence was omitted. Activation remains hash-pinned, but inspect the bundle locally for a complete review.</p> : null}
              <details open>
                <summary>Manifest and permissions</summary>
                <pre>{JSON.stringify(evidence.base.manifest, null, 2)}</pre>
              </details>
              <details open>
                <summary>Evaluation report</summary>
                <pre>{JSON.stringify(evidence.base.eval_report ?? { passed: false, note: "No evaluation report" }, null, 2)}</pre>
              </details>
              {evidence.base.source_diff ? (
                <details open><summary>Source diff from active/base version</summary><pre>{evidence.base.source_diff}</pre></details>
              ) : null}
              <details>
                <summary>Reviewed source files ({evidence.base.files.length})</summary>
                <div className="evidenceFiles">
                  {evidence.base.files.map((file) => (
                    <details key={file.path}>
                      <summary><span>{file.path}</span><code>{shortHash(file.sha256)}</code></summary>
                      <pre>{file.content}</pre>
                    </details>
                  ))}
                </div>
              </details>
              {evidence.proposal ? (
                <section className="eligibleRevisionSection">
                  <h3>Evaluated revisions for this correction</h3>
                  {evidence.eligible.length ? evidence.eligible.map((revision) => (
                    <article key={revision.version_id}>
                      <div><strong>{String(revision.manifest.version ?? revision.version_id)}</strong><code>{shortHash(revision.content_hash)}</code></div>
                      <p>{revision.source_diff ? "A verified source diff is available." : "No source changes were reported."}</p>
                      <div className="eligibleRevisionActions">
                        <button className="textButton" type="button" onClick={() => setEvidence((current) => current ? { ...current, title: `Evaluated revision ${String(revision.manifest.version ?? revision.version_id)}`, base: revision } : null)}>Inspect evidence</button>
                        <button className="primaryButton" type="button" onClick={() => prepareImprovementDecision(evidence.proposal!, "approve", revision.version_id)}>Approve this exact revision</button>
                      </div>
                    </article>
                  )) : <p>No evaluated revision is attached yet. Approval will only queue a draft revision request and will not change the active tool.</p>}
                </section>
              ) : null}
            </div>
          </section>
        </div>
      ) : null}

      {decisionTarget ? (
        <div className="modalBackdrop activationBackdrop" role="presentation" onMouseDown={() => !decisionBusy && setDecisionTarget(null)}>
          <section className="modalCard activationDialog" role="dialog" aria-modal="true" aria-labelledby="improvement-decision-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div><span className="eyebrow">Immutable governance decision</span><h2 id="improvement-decision-title">{decisionTarget.label}</h2></div>
              <button className="iconButton" type="button" aria-label="Close" disabled={decisionBusy} onClick={() => setDecisionTarget(null)}>×</button>
            </header>
            <div className="activationBody">
              <div className="activationWarning">
                <span>!</span>
                <p>{decisionTarget.targetVersionId
                  ? "Only the selected evaluated, content-addressed version will be activated."
                  : decisionTarget.decision === "approve"
                    ? "This records the correction and queues a revision request. It does not change the active version."
                    : "This rejects the correction and leaves the active version unchanged."}</p>
              </div>
              <label htmlFor="improvement-decision-reason">Decision reason</label>
              <textarea id="improvement-decision-reason" value={decisionTarget.reason} maxLength={2000} autoFocus placeholder="Record why this decision is safe and appropriate." onChange={(event) => setDecisionTarget((current) => current ? { ...current, reason: event.target.value } : null)} />
            </div>
            <footer className="activationActions">
              <button className="secondaryButton" type="button" disabled={decisionBusy} onClick={() => setDecisionTarget(null)}>Cancel</button>
              <button className={decisionTarget.decision === "reject" ? "dangerButton" : "primaryButton"} type="button" disabled={!decisionTarget.reason.trim() || decisionBusy} onClick={() => void confirmImprovementDecision()}>{decisionBusy ? "Recording…" : decisionTarget.label}</button>
            </footer>
          </section>
        </div>
      ) : null}
    </div>
  );
}
