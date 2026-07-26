"use client";

import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";

import {
  createCorpusSource,
  deleteCorpusSource,
  getCodeGraphStats,
  getCorpusHealth,
  getEntityGraphStats,
  getNotionConnection,
  getProfile,
  listCorpusSources,
  lookupCodeGraphSymbol,
  lookupEntity,
  reindexCorpusSource,
  saveProfile,
  saveNotionConnection,
  searchCorpus,
  setCorpusConsent,
  syncNotion,
} from "@/lib/api";
import type {
  CodeGraphLookup,
  CodeGraphStats,
  CorpusHealth,
  CorpusSource,
  EntityGraphLookup,
  EntityGraphStats,
  KnowledgeSnippet,
  NotionConnection,
  PersonalProfile,
} from "@/lib/types";

const KINDS: CorpusSource["kind"][] = ["code", "docs", "notes", "mixed"];

function messageOf(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

export function KnowledgeCenter() {
  const [health, setHealth] = useState<CorpusHealth | null>(null);
  const [sources, setSources] = useState<CorpusSource[]>([]);
  const [profile, setProfile] = useState<PersonalProfile | null>(null);
  const [profileDraft, setProfileDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [savingProfile, setSavingProfile] = useState(false);
  const [notion, setNotion] = useState<NotionConnection | null>(null);
  const [notionToken, setNotionToken] = useState("");
  const [notionRoots, setNotionRoots] = useState("");
  const [notionLabel, setNotionLabel] = useState("Notion");
  const [notionBusy, setNotionBusy] = useState<"save" | "sync" | "consent" | null>(null);
  const [notionMessage, setNotionMessage] = useState<string | null>(null);

  const [path, setPath] = useState("");
  const [label, setLabel] = useState("");
  const [kind, setKind] = useState<CorpusSource["kind"]>("code");
  const [adding, setAdding] = useState(false);

  const [query, setQuery] = useState("");
  const [snippets, setSnippets] = useState<KnowledgeSnippet[] | null>(null);
  const [searching, setSearching] = useState(false);

  const [graphStats, setGraphStats] = useState<CodeGraphStats | null>(null);
  const [symbol, setSymbol] = useState("");
  const [lookup, setLookup] = useState<CodeGraphLookup | null>(null);
  const [lookingUp, setLookingUp] = useState(false);

  const [entityStats, setEntityStats] = useState<EntityGraphStats | null>(null);
  const [entityName, setEntityName] = useState("");
  const [entityLookup, setEntityLookup] = useState<EntityGraphLookup | null>(null);
  const [entityBusy, setEntityBusy] = useState(false);
  const [tab, setTab] = useState<"setup" | "explore">("setup");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextHealth, nextSources, nextProfile, nextGraph, nextEntities, nextNotion] =
        await Promise.all([
          getCorpusHealth(),
          listCorpusSources(),
          getProfile(),
          getCodeGraphStats().catch(() => null),
          getEntityGraphStats().catch(() => null),
          getNotionConnection(),
        ]);
      setHealth(nextHealth);
      setSources(nextSources);
      setProfile(nextProfile);
      setProfileDraft(nextProfile.content);
      setGraphStats(nextGraph);
      setEntityStats(nextEntities);
      setNotion(nextNotion);
      setNotionRoots(nextNotion.root_page_ids.join("\n"));
      setNotionLabel(nextNotion.label);
    } catch (loadError) {
      setError(messageOf(loadError, "Could not load your knowledge settings."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  async function addSource(event: FormEvent) {
    event.preventDefault();
    if (!path.trim()) return;
    setAdding(true);
    setError(null);
    try {
      const created = await createCorpusSource(path.trim(), label.trim(), kind);
      setSources((current) => [created, ...current]);
      setPath("");
      setLabel("");
    } catch (addError) {
      setError(messageOf(addError, "Could not add that source."));
    } finally {
      setAdding(false);
    }
  }

  async function toggleConsent(source: CorpusSource) {
    setBusyId(source.id);
    setError(null);
    try {
      const updated = await setCorpusConsent(
        source.id,
        !source.consent,
        source.consent ? "revoked from UI" : "granted from UI",
      );
      setSources((current) =>
        current.map((item) => (item.id === source.id ? updated : item)),
      );
    } catch (consentError) {
      setError(messageOf(consentError, "Could not update consent."));
    } finally {
      setBusyId(null);
    }
  }

  async function reindex(source: CorpusSource) {
    setBusyId(source.id);
    setError(null);
    try {
      await reindexCorpusSource(source.id);
      setSources(await listCorpusSources());
      setGraphStats(await getCodeGraphStats().catch(() => null));
      setEntityStats(await getEntityGraphStats().catch(() => null));
    } catch (reindexError) {
      setError(messageOf(reindexError, "Indexing did not complete."));
      setSources(await listCorpusSources());
    } finally {
      setBusyId(null);
    }
  }

  async function removeSource(source: CorpusSource) {
    setBusyId(source.id);
    setError(null);
    try {
      await deleteCorpusSource(source.id);
      setSources((current) => current.filter((item) => item.id !== source.id));
    } catch (deleteError) {
      setError(messageOf(deleteError, "Could not remove that source."));
    } finally {
      setBusyId(null);
    }
  }

  async function persistProfile() {
    setSavingProfile(true);
    setError(null);
    try {
      const saved = await saveProfile(profileDraft);
      setProfile(saved);
      setProfileDraft(saved.content);
    } catch (saveError) {
      setError(messageOf(saveError, "Could not save your profile."));
    } finally {
      setSavingProfile(false);
    }
  }

  async function persistNotion(event: FormEvent) {
    event.preventDefault();
    setNotionBusy("save");
    setNotionMessage(null);
    setError(null);
    try {
      const saved = await saveNotionConnection({
        accessToken: notionToken.trim() || undefined,
        rootPageIds: notionRoots.split(/[\n,]+/).map((item) => item.trim()).filter(Boolean),
        label: notionLabel.trim() || "Notion",
      });
      setNotion(saved);
      setNotionToken("");
      setNotionRoots(saved.root_page_ids.join("\n"));
      setSources(await listCorpusSources());
      setNotionMessage("Connection saved locally. Sync runs only when you ask it to.");
    } catch (saveError) {
      setError(messageOf(saveError, "Could not save the Notion connection."));
    } finally {
      setNotionBusy(null);
    }
  }

  async function runNotionSync() {
    setNotionBusy("sync");
    setNotionMessage(null);
    setError(null);
    try {
      const result = await syncNotion();
      const [connection, nextSources] = await Promise.all([
        getNotionConnection(),
        listCorpusSources(),
      ]);
      setNotion(connection);
      setSources(nextSources);
      setNotionMessage(result.message);
    } catch (syncError) {
      setError(messageOf(syncError, "Notion sync did not complete."));
      setNotion(await getNotionConnection().catch(() => notion));
    } finally {
      setNotionBusy(null);
    }
  }

  async function toggleNotionConsent() {
    if (!notion?.source) return;
    setNotionBusy("consent");
    setNotionMessage(null);
    setError(null);
    try {
      const updated = await setCorpusConsent(
        notion.source.id,
        !notion.source.consent,
        notion.source.consent ? "Notion RAG disabled from UI" : "Notion RAG enabled from UI",
      );
      setNotion((current) => current ? { ...current, source: updated } : current);
      setSources((current) => current.map((item) => item.id === updated.id ? updated : item));
      setNotionMessage(updated.consent
        ? "RAG indexing is enabled. Press Sync now to index the mirrored pages."
        : "RAG indexing is disabled; the local mirror remains on this Mac.");
    } catch (consentError) {
      setError(messageOf(consentError, "Could not update Notion indexing permission."));
    } finally {
      setNotionBusy(null);
    }
  }

  async function runSearch(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      setSnippets(await searchCorpus(query.trim()));
    } catch (searchError) {
      setError(messageOf(searchError, "Retrieval test failed."));
    } finally {
      setSearching(false);
    }
  }

  async function runLookup(event: FormEvent) {
    event.preventDefault();
    if (!symbol.trim()) return;
    setLookingUp(true);
    setError(null);
    try {
      setLookup(await lookupCodeGraphSymbol(symbol.trim()));
    } catch (lookupError) {
      setError(messageOf(lookupError, "Symbol lookup failed."));
    } finally {
      setLookingUp(false);
    }
  }

  async function runEntityLookup(event: FormEvent) {
    event.preventDefault();
    if (!entityName.trim()) return;
    setEntityBusy(true);
    setError(null);
    try {
      setEntityLookup(await lookupEntity(entityName.trim()));
    } catch (lookupError) {
      setError(messageOf(lookupError, "Entity lookup failed."));
    } finally {
      setEntityBusy(false);
    }
  }

  const cloudOff = health ? !health.available : false;
  const graphEmpty = !graphStats || graphStats.node_count === 0;
  const entityEnabled = health?.entity_graph_enabled ?? false;
  const entityEmpty = !entityStats || entityStats.node_count === 0;
  const profileDirty = profile ? profileDraft !== profile.content : profileDraft.length > 0;
  const localSources = sources.filter((source) => source.provider !== "notion");
  const indexedSources = sources.filter((source) => source.status === "indexed").length;
  const totalFiles = sources.reduce((total, source) => total + source.file_count, 0);
  const totalChunks = sources.reduce((total, source) => total + source.chunk_count, 0);

  return (
    <div className="workspacePage knowledgePage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Grounded in your own material</span>
          <h1>Knowledge</h1>
          <p>
            A small always-on profile plus a private, reranked index of your code and notes.
            Everything is local unless you grant a source cloud-embedding consent.
          </p>
        </div>
        <button className="secondaryButton" type="button" onClick={() => void load()} disabled={loading}>
          Refresh
        </button>
      </header>

      {error ? (
        <div className="notice errorNotice" role="alert">
          <strong>Something went wrong</strong>
          <span>{error}</span>
        </div>
      ) : null}

      {cloudOff ? (
        <div className="memoryPrinciple knowledgeBanner">
          <span className="principleMark">◐</span>
          <div>
            <strong>Cloud retrieval is off</strong>
            <p>
              Set <code>WAQIL_ALLOW_CLOUD_EMBEDDINGS=true</code> and configure OCI to embed and
              rerank. Until then, sources can be registered but not indexed, and answers use the
              local keyword path only — nothing leaves this Mac.
            </p>
          </div>
        </div>
      ) : null}

      <div className="knowledgeDashboardBar">
        <div className="knowledgeTabs" role="tablist" aria-label="Knowledge sections">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "setup"}
            className={`knowledgeTab ${tab === "setup" ? "isActive" : ""}`}
            onClick={() => setTab("setup")}
          >
            <span className="knowledgeTabGlyph" aria-hidden="true">⌁</span>
            <span><span className="knowledgeTabLabel">Library</span><span className="knowledgeTabHint">Connect &amp; organise</span></span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "explore"}
            className={`knowledgeTab ${tab === "explore" ? "isActive" : ""}`}
            onClick={() => setTab("explore")}
          >
            <span className="knowledgeTabGlyph" aria-hidden="true">⌕</span>
            <span><span className="knowledgeTabLabel">Explore</span><span className="knowledgeTabHint">Search &amp; trace</span></span>
          </button>
        </div>
        <div className="knowledgeOverview" aria-label="Knowledge overview">
          <div><span>Indexed</span><strong>{indexedSources}</strong><small>sources ready</small></div>
          <div><span>Material</span><strong>{totalFiles}</strong><small>files mirrored</small></div>
          <div><span>Recall</span><strong>{totalChunks}</strong><small>searchable chunks</small></div>
          <div className={notion?.configured ? "isHealthy" : ""}>
            <span>Notion</span><strong>{notion?.page_count ?? 0}</strong><small>{notion?.configured ? "pages connected" : "not connected"}</small>
          </div>
        </div>
      </div>

      {tab === "setup" ? (
      <div className="knowledgeSetupGrid">
      <section className="knowledgeSection profileSection">
        <div className="knowledgeSectionHead">
          <h2>Profile <span className="tierTag">Tier 0 · always on</span></h2>
          {profile?.updated_at ? (
            <span className="mutedMeta">Updated {new Date(profile.updated_at).toLocaleString()}</span>
          ) : null}
        </div>
        <p className="sectionLede">
          Stable facts the agent should never miss — who you are, your role, writing style, hard
          preferences. Injected verbatim every turn, so you don&apos;t need a long &ldquo;about me&rdquo; prompt.
        </p>
        <textarea
          className="knowledgeTextarea"
          value={profileDraft}
          onChange={(event) => setProfileDraft(event.target.value)}
          placeholder={"# About me\n- I'm …, I work at …\n- Writing style: concise, British English\n- Current projects: …"}
          rows={8}
          spellCheck={false}
        />
        <div className="cardActions">
          <span className="mutedMeta">{profileDraft.length} characters</span>
          <button
            className="primaryButton"
            type="button"
            onClick={() => void persistProfile()}
            disabled={savingProfile || !profileDirty}
          >
            {savingProfile ? "Saving…" : "Save profile"}
          </button>
        </div>
      </section>

      <section className="knowledgeSection notionSection notionWorkspace">
        <div className="notionHero">
          <div className="notionIdentity">
            <span className="notionMonogram" aria-hidden="true">N</span>
            <div>
              <span className="eyebrow">Connected workspace</span>
              <h2>Notion</h2>
              <p>Bring shared pages into Metis as a private, cited knowledge source.</p>
            </div>
          </div>
          <div className="notionHeroActions">
            <span className={`notionConnectionState ${notion?.configured ? "connected" : ""}`}>
              <i />{notion?.configured ? "Connected" : "Not connected"}
            </span>
            <button
              className="primaryButton notionSyncButton"
              type="button"
              disabled={!notion?.configured || notionBusy !== null}
              onClick={() => void runNotionSync()}
            >
              <span aria-hidden="true">↻</span>{notionBusy === "sync" ? "Syncing…" : "Sync now"}
            </button>
          </div>
        </div>

        <dl className="notionStats">
          <div><dt>Pages</dt><dd>{notion?.page_count ?? 0}</dd><small>mirrored locally</small></div>
          <div><dt>Search index</dt><dd>{notion?.source?.chunk_count ?? 0}</dd><small>retrievable chunks</small></div>
          <div><dt>Last refresh</dt><dd>{notion?.last_synced_at ? new Date(notion.last_synced_at).toLocaleDateString() : "Never"}</dd><small>manual only</small></div>
        </dl>

        {notionMessage ? <p className="notionMessage" role="status">{notionMessage}</p> : null}
        {notion?.last_error ? <p className="sourceError">Last sync: {notion.last_error}</p> : null}

        <form className="notionForm" onSubmit={(event) => void persistNotion(event)}>
          <div className="notionFormHeading">
            <div>
              <span className="notionFormKicker">Connection details</span>
              <strong>{notion?.configured ? "Update what Metis can read" : "Connect your workspace"}</strong>
            </div>
            <span className="notionSafety"><i />Token stays on this Mac</span>
          </div>
          <div className="notionFieldGrid">
            <label>
              <span>Integration token</span>
              <input
                className="knowledgeInput mono"
                type="password"
                value={notionToken}
                onChange={(event) => setNotionToken(event.target.value)}
                placeholder={notion?.token_configured ? "Stored · enter only to replace" : "ntn_… or secret_…"}
                autoComplete="off"
              />
            </label>
            <label>
              <span>Display name</span>
              <input
                className="knowledgeInput"
                value={notionLabel}
                onChange={(event) => setNotionLabel(event.target.value)}
                placeholder="Notion"
              />
            </label>
            <label className="notionRootsField">
              <span>Pages to include <em>optional</em></span>
              <textarea
                className="knowledgeTextarea mono"
                value={notionRoots}
                onChange={(event) => setNotionRoots(event.target.value)}
                placeholder={"Leave blank for everything shared, or paste one page URL per line."}
                rows={3}
                spellCheck={false}
              />
            </label>
          </div>
          <div className="notionFormFooter">
            <p><b>01</b> Create integration <span>→</span> <b>02</b> Share pages <span>→</span> <b>03</b> Sync here</p>
            <div className="notionActions">
              <button
                className="secondaryButton"
                type="button"
                disabled={!notion?.source || notionBusy !== null}
                onClick={() => void toggleNotionConsent()}
              >
                {notion?.source?.consent ? "RAG enabled" : "Enable RAG"}
              </button>
              <button
                className="primaryButton"
                type="submit"
                disabled={notionBusy !== null || (!notionToken.trim() && !notion?.token_configured)}
              >
                {notionBusy === "save" ? "Saving…" : notion?.configured ? "Save changes" : "Connect Notion"}
              </button>
            </div>
          </div>
        </form>
      </section>

      <section className="knowledgeSection localSourcesSection">
        <div className="knowledgeSectionHead">
          <h2>Sources <span className="tierTag">Tier 1 · retrieved</span></h2>
          {health ? <span className="mutedMeta mono">{health.embed_model} · {health.rerank_model}</span> : null}
        </div>
        <form className="sourceForm sourceAddForm" onSubmit={(event) => void addSource(event)}>
          <input
            className="knowledgeInput sourcePathInput"
            value={path}
            onChange={(event) => setPath(event.target.value)}
            placeholder="/absolute/path/to/a/repo/or/notes"
            spellCheck={false}
          />
          <input
            className="knowledgeInput"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="Label (optional)"
          />
          <select
            className="knowledgeInput"
            value={kind}
            onChange={(event) => setKind(event.target.value as CorpusSource["kind"])}
            aria-label="Source kind"
          >
            {KINDS.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
          <button className="primaryButton" type="submit" disabled={adding || !path.trim()}>
            {adding ? "Adding…" : "Add source"}
          </button>
        </form>

        <div className="sourceList sourceGrid" aria-live="polite">
          {loading && !localSources.length
            ? Array.from({ length: 2 }).map((_, index) => <div className="skeletonRow" key={index} />)
            : null}
          {!loading && !localSources.length ? (
            <div className="emptyPanel">
              <span className="emptyGlyph">◈</span>
              <h2>No sources yet</h2>
              <p>Add a folder of your code or notes above, then grant it consent to index.</p>
            </div>
          ) : null}
          {localSources.map((source) => (
            <article className="memoryCard sourceCard" key={source.id}>
              <div className="memoryCardHeader">
                <span className="sourceLabel">{source.label}<span className="memoryKind">{source.kind}</span></span>
                <span className={`statusPill status-${source.status}`}>{source.status}</span>
              </div>
              <p className="mono sourcePath">{source.root_path}</p>
              {source.last_error ? <p className="sourceError">{source.last_error}</p> : null}
              <dl className="memoryMeta">
                <div><dt>Consent</dt><dd>{source.consent ? "Granted" : "Not granted"}</dd></div>
                <div><dt>Files</dt><dd>{source.file_count}</dd></div>
                <div><dt>Chunks</dt><dd>{source.chunk_count}</dd></div>
                <div><dt>Indexed</dt><dd>{source.last_indexed_at ? new Date(source.last_indexed_at).toLocaleDateString() : "Never"}</dd></div>
              </dl>
              <footer className="cardActions">
                <button className="dangerButton" type="button" disabled={busyId === source.id} onClick={() => void removeSource(source)}>
                  Remove
                </button>
                <button className="secondaryButton" type="button" disabled={busyId === source.id} onClick={() => void toggleConsent(source)}>
                  {source.consent ? "Revoke consent" : "Grant consent"}
                </button>
                <button
                  className="primaryButton"
                  type="button"
                  disabled={busyId === source.id || !source.consent || cloudOff}
                  title={cloudOff ? "Enable cloud embeddings first" : !source.consent ? "Grant consent first" : ""}
                  onClick={() => void reindex(source)}
                >
                  {busyId === source.id ? "Indexing…" : "Index now"}
                </button>
              </footer>
            </article>
          ))}
        </div>
      </section>
      </div>
      ) : null}

      {tab === "explore" ? (
      <div className="knowledgeExploreGrid">
      <section className="knowledgeSection retrievalSection">
        <div className="knowledgeSectionHead">
          <h2>Test retrieval</h2>
        </div>
        <p className="sectionLede">Preview what the agent would retrieve for a question, after embed → cosine recall → rerank.</p>
        <form className="sourceForm" onSubmit={(event) => void runSearch(event)}>
          <input
            className="knowledgeInput sourcePathInput"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. how does login issue a session token?"
          />
          <button className="primaryButton" type="submit" disabled={searching || !query.trim() || cloudOff}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>
        {snippets ? (
          snippets.length ? (
            <div className="snippetList">
              {snippets.map((snippet, index) => (
                <article className="snippetCard" key={`${snippet.rel_path}-${index}`}>
                  <header>
                    <span className="mono snippetLoc">
                      {snippet.source_label} · {snippet.rel_path}
                      {snippet.symbol ? `::${snippet.symbol}` : ""}
                    </span>
                    <span className="scorePill">{snippet.score.toFixed(3)}</span>
                  </header>
                  <pre className="snippetText">{snippet.text}</pre>
                </article>
              ))}
            </div>
          ) : (
            <div className="emptyPanel"><span className="emptyGlyph">◈</span><h2>No matches</h2><p>Nothing indexed yet, or no consented source matched.</p></div>
          )
        ) : null}
      </section>

      <section className="knowledgeSection codeGraphSection">
        <div className="knowledgeSectionHead">
          <h2>Code graph <span className="tierTag">Graph-RAG · local</span></h2>
          {graphStats ? (
            <span className="mutedMeta mono">
              {graphStats.node_count} nodes · {graphStats.edge_count} edges
            </span>
          ) : null}
        </div>
        <p className="sectionLede">
          A deterministic call/def/import graph built locally from your Python sources — no cloud,
          no model. Retrieval uses it to pull in callers and callees of a vector hit; look a symbol
          up here to trace who calls it and what it calls.
        </p>
        {graphStats && !graphEmpty ? (
          <div className="graphStatRow">
            {Object.entries(graphStats.nodes_by_kind).map(([nodeKind, count]) => (
              <span className="graphStatPill" key={`n-${nodeKind}`}>
                {count} {nodeKind}
              </span>
            ))}
            {Object.entries(graphStats.edges_by_kind).map(([edgeKind, count]) => (
              <span className="graphStatPill graphStatEdge" key={`e-${edgeKind}`}>
                {count} {edgeKind}
              </span>
            ))}
          </div>
        ) : null}
        <form className="sourceForm" onSubmit={(event) => void runLookup(event)}>
          <input
            className="knowledgeInput sourcePathInput mono"
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
            placeholder="Symbol name, e.g. retrieve"
            spellCheck={false}
          />
          <button className="primaryButton" type="submit" disabled={lookingUp || !symbol.trim()}>
            {lookingUp ? "Tracing…" : "Trace symbol"}
          </button>
        </form>
        {graphEmpty ? (
          <p className="mutedMeta">
            The graph is empty. Index a consented source containing Python to populate it.
          </p>
        ) : null}
        {lookup ? (
          lookup.definitions.length || lookup.callers.length || lookup.callees.length ? (
            <div className="graphResult">
              <div className="graphColumn">
                <h3>Defined <span className="graphCount">{lookup.definitions.length}</span></h3>
                {lookup.definitions.length ? (
                  lookup.definitions.map((def, index) => (
                    <div className="graphEdgeRow" key={`d-${index}`}>
                      <span className="graphKind">{def.kind}</span>
                      <span className="mono graphLoc">
                        {def.rel_path}:{def.start_line}
                      </span>
                      <span className="mono graphQual">{def.qualname}</span>
                    </div>
                  ))
                ) : (
                  <p className="mutedMeta">No definition indexed.</p>
                )}
              </div>
              <div className="graphColumn">
                <h3>Callers <span className="graphCount">{lookup.callers.length}</span></h3>
                {lookup.callers.length ? (
                  lookup.callers.map((caller, index) => (
                    <div className="graphEdgeRow" key={`c-${index}`}>
                      <span className="mono graphQual">{caller.caller}</span>
                      <span className="mono graphLoc">
                        {caller.rel_path}:{caller.line}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="mutedMeta">Nothing calls this.</p>
                )}
              </div>
              <div className="graphColumn">
                <h3>Calls <span className="graphCount">{lookup.callees.length}</span></h3>
                {lookup.callees.length ? (
                  lookup.callees.map((callee, index) => (
                    <div className="graphEdgeRow" key={`e-${index}`}>
                      <span className="mono graphQual">{callee.dst_raw}</span>
                      <span className="mono graphLoc">
                        {callee.rel_path}:{callee.line}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="mutedMeta">Calls nothing indexed.</p>
                )}
              </div>
            </div>
          ) : (
            <div className="emptyPanel">
              <span className="emptyGlyph">◇</span>
              <h2>No graph entry for &ldquo;{lookup.name}&rdquo;</h2>
              <p>Try a function, method, or class name from an indexed Python source.</p>
            </div>
          )
        ) : null}
      </section>

      <section className="knowledgeSection entityGraphSection">
        <div className="knowledgeSectionHead">
          <h2>Entity graph <span className="tierTag">Graph-RAG · Stage 2</span></h2>
          {entityStats && !entityEmpty ? (
            <span className="mutedMeta mono">
              {entityStats.node_count} entities · {entityStats.edge_count} relations
            </span>
          ) : null}
        </div>
        <p className="sectionLede">
          For prose (notes and docs), Metis can use Cohere Command A to extract entities and how
          they relate, so you can trace connections across your writing. It costs a model call per
          file and sends text to the cloud, so it is off unless you enable{" "}
          <code>WAQIL_CORPUS_ENTITY_GRAPH=true</code>.
        </p>
        {!entityEnabled ? (
          <p className="mutedMeta">
            Entity extraction is disabled. The code graph above stays fully local and needs no
            model — this second stage is opt-in.
          </p>
        ) : entityEmpty ? (
          <p className="mutedMeta">
            Enabled, but no entities yet. Index a consented source containing notes or docs.
          </p>
        ) : (
          <div className="graphStatRow">
            {Object.entries(entityStats!.nodes_by_kind).map(([entityKind, count]) => (
              <span className="graphStatPill" key={`ent-${entityKind}`}>
                {count} {entityKind}
              </span>
            ))}
          </div>
        )}
        {entityEnabled && !entityEmpty ? (
          <>
            <form className="sourceForm" onSubmit={(event) => void runEntityLookup(event)}>
              <input
                className="knowledgeInput sourcePathInput"
                value={entityName}
                onChange={(event) => setEntityName(event.target.value)}
                placeholder="Entity name, e.g. Cohere"
                spellCheck={false}
              />
              <button className="primaryButton" type="submit" disabled={entityBusy || !entityName.trim()}>
                {entityBusy ? "Tracing…" : "Trace entity"}
              </button>
            </form>
            {entityLookup ? (
              entityLookup.kinds.length ||
              entityLookup.relations_out.length ||
              entityLookup.relations_in.length ? (
                <div className="graphResult">
                  <div className="graphColumn">
                    <h3>Is a <span className="graphCount">{entityLookup.kinds.length}</span></h3>
                    {entityLookup.kinds.length ? (
                      entityLookup.kinds.map((entityKind) => (
                        <div className="graphEdgeRow" key={`k-${entityKind}`}>
                          <span className="graphKind">{entityKind}</span>
                        </div>
                      ))
                    ) : (
                      <p className="mutedMeta">Kind unknown.</p>
                    )}
                  </div>
                  <div className="graphColumn">
                    <h3>Relates to <span className="graphCount">{entityLookup.relations_out.length}</span></h3>
                    {entityLookup.relations_out.length ? (
                      entityLookup.relations_out.map((rel, index) => (
                        <div className="graphEdgeRow" key={`ro-${index}`}>
                          <span className="mono graphQual">{rel.relation} → {rel.dst_name}</span>
                          <span className="mono graphLoc">{rel.rel_path}</span>
                        </div>
                      ))
                    ) : (
                      <p className="mutedMeta">No outgoing relations.</p>
                    )}
                  </div>
                  <div className="graphColumn">
                    <h3>Referenced by <span className="graphCount">{entityLookup.relations_in.length}</span></h3>
                    {entityLookup.relations_in.length ? (
                      entityLookup.relations_in.map((rel, index) => (
                        <div className="graphEdgeRow" key={`ri-${index}`}>
                          <span className="mono graphQual">{rel.src_name} → {rel.relation}</span>
                          <span className="mono graphLoc">{rel.rel_path}</span>
                        </div>
                      ))
                    ) : (
                      <p className="mutedMeta">Nothing references this.</p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="emptyPanel">
                  <span className="emptyGlyph">◇</span>
                  <h2>No entity &ldquo;{entityLookup.name}&rdquo;</h2>
                  <p>Try a name that appears in your indexed notes or docs.</p>
                </div>
              )
            ) : null}
          </>
        ) : null}
      </section>
      </div>
      ) : null}
    </div>
  );
}
