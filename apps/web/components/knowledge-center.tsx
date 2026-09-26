"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ArrowUpRight, BookOpen, FileText, FolderOpen, Link2, Pencil, Plus, RefreshCw, Search, ShieldCheck, UserRound } from "lucide-react";

import { MarkdownContent } from "@/components/markdown-content";
import { SelectMenu } from "@/components/select-menu";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";

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
const KNOWLEDGE_TABS = [
  { id: "setup", label: "Sources", icon: FolderOpen },
  { id: "profile", label: "Personal profile", icon: UserRound },
  { id: "notion", label: "Connections", icon: Link2 },
  { id: "explore", label: "Explore", icon: Search },
] as const;
type KnowledgeTab = typeof KNOWLEDGE_TABS[number]["id"];

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
  const [sourceAction, setSourceAction] = useState<"consent" | "index" | "remove" | null>(null);
  const [savingProfile, setSavingProfile] = useState(false);
  const [notion, setNotion] = useState<NotionConnection | null>(null);
  const [notionToken, setNotionToken] = useState("");
  const [notionRoots, setNotionRoots] = useState("");
  const [notionLabel, setNotionLabel] = useState("Notion");
  const [notionBusy, setNotionBusy] = useState<"save" | "sync" | "consent" | null>(null);
  const [notionMessage, setNotionMessage] = useState<string | null>(null);
  const [notionError, setNotionError] = useState<string | null>(null);

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
  const [tab, setTab] = useState<KnowledgeTab>("setup");
  const [sourceQuery, setSourceQuery] = useState("");
  const [sourceKind, setSourceKind] = useState("all");
  const [sourceFormOpen, setSourceFormOpen] = useState(false);
  const [profileEditing, setProfileEditing] = useState(false);
  const sourcePathInput = useRef<HTMLInputElement>(null);
  const draftsEdited = useRef({ profile: false, notion: false });
  const sourceBusy = useRef(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setNotionError(null);
    try {
      const [nextHealth, nextSources, nextProfile, nextGraph, nextEntities, nextNotion] =
        await Promise.all([
          getCorpusHealth(),
          listCorpusSources(),
          getProfile(),
          getCodeGraphStats().catch(() => null),
          getEntityGraphStats().catch(() => null),
          getNotionConnection().catch((problem) => {
            setNotionError(messageOf(problem, "Notion connection details could not be loaded."));
            return null;
          }),
        ]);
      setHealth(nextHealth);
      setSources(nextSources);
      setProfile(nextProfile);
      if (!draftsEdited.current.profile) setProfileDraft(nextProfile.content);
      setGraphStats(nextGraph);
      setEntityStats(nextEntities);
      setNotion(nextNotion);
      if (nextNotion && !draftsEdited.current.notion) {
        setNotionRoots(nextNotion.root_page_ids.join("\n"));
        setNotionLabel(nextNotion.label);
      }
    } catch (loadError) {
      setError(messageOf(loadError, "Could not load your knowledge settings."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  async function addSource(event: FormEvent) {
    event.preventDefault();
    if (!path.trim() || adding) return;
    setAdding(true);
    setError(null);
    try {
      const created = await createCorpusSource(path.trim(), label.trim(), kind);
      setSources((current) => [created, ...current]);
      setPath("");
      setLabel("");
      setSourceQuery("");
      setSourceKind("all");
      setSourceFormOpen(false);
    } catch (addError) {
      setError(messageOf(addError, "Could not add that source."));
    } finally {
      setAdding(false);
    }
  }

  async function toggleConsent(source: CorpusSource) {
    if (sourceBusy.current) return;
    sourceBusy.current = true;
    setBusyId(source.id);
    setSourceAction("consent");
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
      sourceBusy.current = false;
      setBusyId(null);
      setSourceAction(null);
    }
  }

  async function reindex(source: CorpusSource) {
    if (sourceBusy.current) return;
    sourceBusy.current = true;
    setBusyId(source.id);
    setSourceAction("index");
    setError(null);
    try {
      await reindexCorpusSource(source.id);
      setSources(await listCorpusSources());
      setGraphStats(await getCodeGraphStats().catch(() => null));
      setEntityStats(await getEntityGraphStats().catch(() => null));
    } catch (reindexError) {
      setError(messageOf(reindexError, "Indexing did not complete."));
      const latest = await listCorpusSources().catch(() => null);
      if (latest) setSources(latest);
    } finally {
      sourceBusy.current = false;
      setBusyId(null);
      setSourceAction(null);
    }
  }

  async function removeSource(source: CorpusSource) {
    if (sourceBusy.current) return;
    sourceBusy.current = true;
    setBusyId(source.id);
    setSourceAction("remove");
    setError(null);
    try {
      await deleteCorpusSource(source.id);
      setSources((current) => current.filter((item) => item.id !== source.id));
    } catch (deleteError) {
      setError(messageOf(deleteError, "Could not remove that source."));
    } finally {
      sourceBusy.current = false;
      setBusyId(null);
      setSourceAction(null);
    }
  }

  async function persistProfile() {
    if (savingProfile) return;
    setSavingProfile(true);
    setError(null);
    try {
      const saved = await saveProfile(profileDraft);
      setProfile(saved);
      setProfileDraft(saved.content);
      draftsEdited.current.profile = false;
      setProfileEditing(false);
    } catch (saveError) {
      setError(messageOf(saveError, "Could not save your profile."));
    } finally {
      setSavingProfile(false);
    }
  }

  async function persistNotion(event: FormEvent) {
    event.preventDefault();
    if (notionBusy) return;
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
      setNotionLabel(saved.label);
      draftsEdited.current.notion = false;
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
    if (!query.trim() || searching) return;
    setSearching(true);
    setSnippets(null);
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
    if (!symbol.trim() || lookingUp) return;
    setLookingUp(true);
    setLookup(null);
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
    if (!entityName.trim() || entityBusy) return;
    setEntityBusy(true);
    setEntityLookup(null);
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
  const visibleSources = localSources.filter((source) => (sourceKind === "all" || source.kind === sourceKind) && `${source.label} ${source.root_path}`.toLowerCase().includes(sourceQuery.trim().toLowerCase()));
  const indexedSources = sources.filter((source) => source.status === "indexed").length;
  const totalFiles = sources.reduce((total, source) => total + source.file_count, 0);
  const totalChunks = sources.reduce((total, source) => total + source.chunk_count, 0);

  return (
    <div className="workspacePage knowledgePage reference-workspace knowledge-workspace">
      <PageHeader
        icon={<BookOpen size={23} aria-hidden="true" />}
        eyebrow="Your reference library"
        title="Knowledge"
        lede="Keep your notes, sources, and personal context together. Give every answer a useful starting point."
        actions={<><button className="ui-btn" type="button" onClick={() => void load()} disabled={loading || savingProfile || notionBusy !== null || busyId !== null || adding}><RefreshCw size={14} aria-hidden="true" />{loading ? "Refreshing…" : "Refresh"}</button><button className="ui-btn is-primary" type="button" onClick={() => { setTab("setup"); setSourceFormOpen(true); window.setTimeout(() => sourcePathInput.current?.focus(), 0); }}><Plus size={15} aria-hidden="true" />Add source</button></>}
      />

      {error ? (
        <Notice kind="error" title={health ? "The request could not be completed" : "Knowledge unavailable"} action={!health ? "Try again" : undefined} onAction={() => void load()} onDismiss={health ? () => setError(null) : undefined}>
          <p>{error}</p>
        </Notice>
      ) : null}

      {cloudOff ? (
        <div className="memoryPrinciple knowledgeBanner reference-notice">
          <span className="principleMark"><ShieldCheck size={18} aria-hidden="true" /></span>
          <div>
            <strong>Your sources stay under your control</strong>
            <p>
              Cloud indexing is unavailable. You can organise sources now; answers use local keyword search until cloud retrieval is configured.
            </p>
          </div>
          <a className="ui-btn is-quiet is-sm" href="/settings">Settings<ArrowUpRight size={13} aria-hidden="true" /></a>
        </div>
      ) : null}

      <div className="reference-summary" aria-label="Knowledge overview">
        <div><FolderOpen size={17} aria-hidden="true" /><span><strong>{!health ? "—" : localSources.length}</strong> local sources</span></div>
        <div><ShieldCheck size={17} aria-hidden="true" /><span><strong>{!health ? "—" : indexedSources}</strong> indexed</span></div>
        <div><FileText size={17} aria-hidden="true" /><span><strong>{!health ? "—" : totalFiles.toLocaleString()}</strong> files</span></div>
        <div><Search size={17} aria-hidden="true" /><span><strong>{!health ? "—" : totalChunks.toLocaleString()}</strong> searchable passages</span></div>
      </div>
      <div className="knowledgeDashboardBar reference-navigation">
        <div className="knowledgeTabs" role="tablist" aria-label="Knowledge sections" onKeyDown={(event) => {
          const current = KNOWLEDGE_TABS.findIndex((item) => item.id === tab);
          const index = event.key === "Home" ? 0 : event.key === "End" ? KNOWLEDGE_TABS.length - 1 : event.key === "ArrowRight" ? (current + 1) % KNOWLEDGE_TABS.length : event.key === "ArrowLeft" ? (current + KNOWLEDGE_TABS.length - 1) % KNOWLEDGE_TABS.length : -1;
          if (index < 0) return;
          const next = KNOWLEDGE_TABS[index]!.id;
          event.preventDefault();
          setTab(next);
          event.currentTarget.querySelector<HTMLButtonElement>(`#knowledge-tab-${next}`)?.focus();
        }}>
          {KNOWLEDGE_TABS.map(({ id, label: tabLabel, icon: Icon }) => <button
            key={id}
            type="button"
            role="tab"
            id={`knowledge-tab-${id}`}
            aria-controls={`knowledge-panel-${id}`}
            tabIndex={tab === id ? 0 : -1}
            aria-selected={tab === id}
            className={`knowledgeTab ${tab === id ? "isActive" : ""}`}
            onClick={() => setTab(id)}
          >
            <Icon size={16} aria-hidden="true" /><span className="knowledgeTabLabel">{tabLabel}</span>{id === "profile" && profileDirty ? <span className="reference-unsaved" aria-label="Unsaved profile changes" /> : null}
          </button>)}
        </div>
      </div>

      {tab !== "explore" ? (
      <div id={`knowledge-panel-${tab}`} className={`knowledgeSetupGrid reference-panel is-${tab}`} role="tabpanel" aria-labelledby={`knowledge-tab-${tab}`} tabIndex={0}>
      {tab === "profile" ? (
      <section className="knowledgeSection profileSection">
        <div className="knowledgeSectionHead">
          <div><span className="reference-kicker">A note about you</span><h2>Personal profile</h2></div>
          <button className="ui-btn is-sm" type="button" disabled={savingProfile || (loading && !profile)} onClick={() => setProfileEditing((current) => !current)}><Pencil size={13} aria-hidden="true" />{profileEditing ? "Preview" : "Edit profile"}</button>
        </div>
        <p className="sectionLede">Your role, preferences, and working style. This note is included in every conversation, so keep it focused on what lasts.</p>
        <div className="reference-note-meta"><span className="ui-chip is-accent"><UserRound size={12} aria-hidden="true" />Always available</span>
          {profile?.updated_at ? (
            <span className="mutedMeta">Updated {new Date(profile.updated_at).toLocaleDateString()}</span>
          ) : null}
        </div>
        {profileEditing || !profileDraft.trim() ? (
        <textarea
          className="knowledgeTextarea"
          aria-label="Personal profile"
          disabled={savingProfile || (loading && !profile)}
          value={profileDraft}
          onChange={(event) => { draftsEdited.current.profile = true; setProfileEditing(true); setProfileDraft(event.target.value); }}
          placeholder={"# About me\n- I'm …, I work at …\n- Writing style: concise, British English\n- Current projects: …"}
          rows={8}
          spellCheck={false}
        />
        ) : <div className="reference-note-body"><MarkdownContent content={profileDraft} /></div>}
        <div className="cardActions">
          <span className="mutedMeta">{profileDraft.length} characters{profileDirty ? " · Unsaved changes" : profile ? " · Saved" : ""}</span>
          <button
            className="primaryButton"
            type="button"
            onClick={() => void persistProfile()}
            disabled={savingProfile || loading || !profileDirty}
          >
            {savingProfile ? "Saving…" : "Save profile"}
          </button>
        </div>
      </section>
      ) : null}

      {tab === "notion" ? (
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
              <i />{notion ? notion.configured ? "Connected" : "Not connected" : notionError ? "Unavailable" : "Checking…"}
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
        {notionError ? <Notice kind="error" title="Notion unavailable" action="Try again" onAction={() => void load()}>{notionError}</Notice> : null}
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
                disabled={notionBusy === "save"}
                value={notionToken}
                onChange={(event) => { draftsEdited.current.notion = true; setNotionToken(event.target.value); }}
                placeholder={notion?.token_configured ? "Stored · enter only to replace" : "ntn_… or secret_…"}
                autoComplete="off"
              />
            </label>
            <label>
              <span>Display name</span>
              <input
                className="knowledgeInput"
                disabled={notionBusy === "save"}
                value={notionLabel}
                onChange={(event) => { draftsEdited.current.notion = true; setNotionLabel(event.target.value); }}
                placeholder="Notion"
              />
            </label>
            <label className="notionRootsField">
              <span>Pages to include <em>optional</em></span>
              <textarea
                className="knowledgeTextarea mono"
                disabled={notionBusy === "save"}
                value={notionRoots}
                onChange={(event) => { draftsEdited.current.notion = true; setNotionRoots(event.target.value); }}
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
                {notionBusy === "consent" ? "Updating…" : notion?.source?.consent ? "Disable indexing" : "Enable indexing"}
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
      ) : null}

      {tab === "setup" ? (
      <section className="knowledgeSection localSourcesSection">
        <div className="knowledgeSectionHead">
          <div><span className="reference-kicker">Folders worth remembering</span><h2>Your sources</h2></div>
          <span className="mutedMeta">{visibleSources.length} of {localSources.length}</span>
        </div>
        <p className="sectionLede">Connect your code, documents, and notes. Each folder has its own permission to be indexed.</p>
        <div className="reference-tools"><label className="reference-search"><Search size={16} aria-hidden="true" /><input value={sourceQuery} onChange={(event) => setSourceQuery(event.target.value)} aria-label="Filter sources" placeholder="Find a source by name or folder…" type="search" /></label><SelectMenu className="reference-kind-filter" label="Source type" hideLabel value={sourceKind} onChange={setSourceKind} options={[{ value: "all", label: "All types" }, ...KINDS.map((item) => ({ value: item, label: item[0]!.toUpperCase() + item.slice(1) }))]} /></div>
        {sourceFormOpen ? (
        <form className="sourceForm sourceAddForm" onSubmit={(event) => void addSource(event)}>
          <div className="reference-form-heading"><strong>Add a local folder</strong><span>Register it now, then choose whether to allow indexing.</span></div>
          <input
            ref={sourcePathInput}
            className="knowledgeInput sourcePathInput"
            aria-label="Source folder path"
            disabled={adding}
            value={path}
            onChange={(event) => setPath(event.target.value)}
            placeholder="/absolute/path/to/a/repo/or/notes"
            spellCheck={false}
          />
          <input
            className="knowledgeInput"
            aria-label="Source label (optional)"
            disabled={adding}
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="Label (optional)"
          />
          <SelectMenu
            className="knowledgeSelect"
            hideLabel
            label="Source kind"
            value={kind}
            disabled={adding}
            onChange={(value) => setKind(value as CorpusSource["kind"])}
            options={KINDS.map((item) => ({ value: item, label: item }))}
          />
          <button className="primaryButton" type="submit" disabled={adding || !path.trim()}>
            {adding ? "Adding…" : "Add source"}
          </button>
          <button className="ui-btn is-quiet" type="button" disabled={adding} onClick={() => setSourceFormOpen(false)}>Cancel</button>
        </form>
        ) : null}

        <div className="sourceList sourceGrid ui-stagger" aria-live="polite">
          {loading && !localSources.length
            ? Array.from({ length: 2 }, (_, index) => <Skeleton key={index} rows={1} height={150} />)
            : null}
          {!loading && health && !localSources.length && !sourceFormOpen ? (
            <div className="reference-empty">
              <FolderOpen size={28} aria-hidden="true" />
              <h3>Build your reference library</h3>
              <p>Add a folder of code, documents, or notes. You decide when it is ready to be indexed.</p>
              <button className="ui-btn is-sm" type="button" onClick={() => { setSourceFormOpen(true); window.setTimeout(() => sourcePathInput.current?.focus(), 0); }}><Plus size={14} aria-hidden="true" />Add your first source</button>
            </div>
          ) : null}
          {!loading && localSources.length > 0 && !visibleSources.length ? <div className="reference-empty"><Search size={24} aria-hidden="true" /><h3>No matching sources</h3><p>Try a different name or include all source types.</p><button className="ui-btn is-sm" type="button" onClick={() => { setSourceQuery(""); setSourceKind("all"); }}>Clear filters</button></div> : null}
          {visibleSources.map((source) => (
            <article className="memoryCard sourceCard" key={source.id}>
              <div className="memoryCardHeader">
                <span className="reference-source-mark"><FolderOpen size={18} aria-hidden="true" /></span><span className="sourceLabel">{source.label || source.root_path.split("/").filter(Boolean).pop()}<span className="memoryKind">{source.kind}</span></span>
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
                <button className="dangerButton" type="button" disabled={busyId !== null} onClick={() => void removeSource(source)}>
                  {busyId === source.id && sourceAction === "remove" ? "Removing…" : "Remove"}
                </button>
                <button className="secondaryButton" type="button" disabled={busyId !== null} onClick={() => void toggleConsent(source)}>
                  {busyId === source.id && sourceAction === "consent" ? "Updating…" : source.consent ? "Revoke consent" : "Grant consent"}
                </button>
                <button
                  className="primaryButton"
                  type="button"
                  disabled={busyId !== null || !source.consent || cloudOff}
                  title={cloudOff ? "Enable cloud embeddings first" : !source.consent ? "Grant consent first" : ""}
                  onClick={() => void reindex(source)}
                >
                  {busyId === source.id && sourceAction === "index" ? "Indexing…" : "Index now"}
                </button>
              </footer>
            </article>
          ))}
        </div>
      </section>
      ) : null}
      </div>
      ) : null}

      {tab === "explore" ? (
      <div id="knowledge-panel-explore" className="knowledgeExploreGrid" role="tabpanel" aria-labelledby="knowledge-tab-explore" tabIndex={0}>
      <section className="knowledgeSection retrievalSection">
        <div className="knowledgeSectionHead">
          <div><span className="reference-kicker">Follow a question to its source</span><h2>Search your knowledge</h2></div>
        </div>
        <p className="sectionLede">Find relevant passages in your indexed sources and see the context Metis can use in an answer.</p>
        <form className="sourceForm" onSubmit={(event) => void runSearch(event)}>
          <input
            className="knowledgeInput sourcePathInput"
            value={query}
            aria-label="Search your knowledge"
            disabled={searching}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. how does login issue a session token?"
          />
          <button className="primaryButton" type="submit" disabled={searching || !query.trim() || cloudOff}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>
        {!snippets && !searching ? <div className="reference-search-prompt"><Search size={22} aria-hidden="true" /><div><strong>Start with what you want to know</strong><p>Ask a question or look for an idea across the sources you have indexed.</p></div></div> : null}
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
          <h2>Code connections <span className="tierTag">Local</span></h2>
          {graphStats ? (
            <span className="mutedMeta mono">
              {graphStats.node_count} nodes · {graphStats.edge_count} edges
            </span>
          ) : null}
        </div>
        <p className="sectionLede">
          Follow a function or class through your Python sources. See where it is defined,
          what calls it, and the code it depends on. This lookup stays on your Mac.
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
            aria-label="Code symbol to trace"
            disabled={lookingUp}
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
          <h2>Ideas &amp; connections <span className="tierTag">Optional</span></h2>
          {entityStats && !entityEmpty ? (
            <span className="mutedMeta mono">
              {entityStats.node_count} entities · {entityStats.edge_count} relations
            </span>
          ) : null}
        </div>
        <p className="sectionLede">
          Trace people, topics, and relationships across your notes and documents. Building these
          connections sends text to the cloud and uses a model call for each file, so it stays off
          until you enable it.
        </p>
        {!entityEnabled ? (
          <details className="reference-advanced"><summary>Enable connections across notes</summary><p className="mutedMeta">Set <code>WAQIL_CORPUS_ENTITY_GRAPH=true</code> in your service configuration, then index a consented source. Code connections remain fully local.</p></details>
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
                aria-label="Entity to find"
                disabled={entityBusy}
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
