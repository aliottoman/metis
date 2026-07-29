"use client";

import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  approveAsset,
  getAssetLogs,
  listAssets,
  revokeAssetApproval,
  saveAssetEnv,
  scanAssets,
  startAsset,
  stopAsset,
} from "@/lib/api";
import type { AssetV1 } from "@/lib/types";

const CATALOG_POLL_MS = 3_000;
const LOG_POLL_MS = 3_000;
const DEFAULT_ASSET_DRAWER_WIDTH = 640;
const MIN_ASSET_DRAWER_WIDTH = 480;
const MAX_ASSET_DRAWER_WIDTH = 880;
const TAG_TONES = ["mint", "coral", "violet", "blue", "gold"] as const;

type LifecycleFilter = "all" | "running" | "ready" | "review" | "setup";
type AssetAction = "start" | "stop" | "approve" | "revoke";
type PendingAssetAction = { assetId: string; action: AssetAction } | null;

function messageOf(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function normalizedStatus(asset: AssetV1): string {
  return asset.status.trim().toLowerCase();
}

function isRunning(asset: AssetV1): boolean {
  return normalizedStatus(asset) === "running";
}

function isActive(asset: AssetV1): boolean {
  return ["starting", "running", "stopping"].includes(normalizedStatus(asset));
}

function statusLabel(value: string): string {
  const readable = value.trim().replace(/[_-]+/g, " ");
  return readable
    ? readable.replace(/\b\w/g, (letter) => letter.toUpperCase())
    : "Discovered";
}

function statusClass(asset: AssetV1): string {
  const status = normalizedStatus(asset);
  if (status === "running") return "isRunning";
  if (["error", "failed", "crashed"].includes(status)) return "isError";
  if (["starting", "stopping"].includes(status)) return "isTransitioning";
  return asset.launchApproved ? "isReady" : "isReview";
}

function launchUrl(value: string | null): string | null {
  if (!value) return null;
  return /^(https?:\/\/|\/)/i.test(value) ? value : null;
}

function assetMatchesLifecycle(asset: AssetV1, filter: LifecycleFilter): boolean {
  if (filter === "running") return isActive(asset);
  if (filter === "ready") return asset.launchApproved && !isActive(asset);
  if (filter === "review") return asset.launchConfigured && !asset.launchApproved;
  if (filter === "setup") return !asset.launchConfigured;
  return true;
}

function mergeAsset(items: AssetV1[], updated: AssetV1): AssetV1[] {
  if (!items.some((asset) => asset.id === updated.id)) return [updated, ...items];
  return items.map((asset) => (asset.id === updated.id ? updated : asset));
}

function commandLabel(parts: string[]): string {
  return parts.map((part) => JSON.stringify(part)).join(" ");
}

function clampDrawerWidth(value: number): number {
  return Math.min(MAX_ASSET_DRAWER_WIDTH, Math.max(MIN_ASSET_DRAWER_WIDTH, value));
}

function displayTags(asset: AssetV1, limit = 5): string[] {
  return Array.from(new Set([asset.category, asset.framework, ...asset.tags]
    .map((value) => value.trim())
    .filter(Boolean)))
    .slice(0, limit);
}

function tagTone(tag: string): typeof TAG_TONES[number] {
  const score = Array.from(tag).reduce((sum, character) => sum + character.charCodeAt(0), 0);
  return TAG_TONES[score % TAG_TONES.length];
}

function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  return (words.length > 1 ? `${words[0]?.[0] ?? ""}${words[1]?.[0] ?? ""}` : name.slice(0, 2)).toUpperCase();
}

export function AssetLibrary() {
  const [assets, setAssets] = useState<AssetV1[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [lifecycle, setLifecycle] = useState<LifecycleFilter>("all");
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanNotice, setScanNotice] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [playerId, setPlayerId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState(DEFAULT_ASSET_DRAWER_WIDTH);
  const [drawerResizing, setDrawerResizing] = useState(false);
  const [envByAsset, setEnvByAsset] = useState<Record<string, Record<string, string>>>({});
  const [revealedEnv, setRevealedEnv] = useState<Record<string, boolean>>({});
  const [disclosureOpen, setDisclosureOpen] = useState<Record<string, boolean>>({});
  const [envSaving, setEnvSaving] = useState(false);
  const [envError, setEnvError] = useState<string | null>(null);
  const [envSaved, setEnvSaved] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<PendingAssetAction>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string | null>(null);
  const [logsBusy, setLogsBusy] = useState(false);
  const [logsError, setLogsError] = useState<string | null>(null);

  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const closeTimerRef = useRef<number | null>(null);
  const logRequestRef = useRef<string | null>(null);
  const drawerWidthRef = useRef(DEFAULT_ASSET_DRAWER_WIDTH);
  const drawerResizingRef = useRef(false);
  const drawerResizeOriginRef = useRef({ pointerX: 0, width: DEFAULT_ASSET_DRAWER_WIDTH });

  const loadCatalog = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const nextAssets = await listAssets();
      setAssets(nextAssets);
      setError(null);
    } catch (loadError) {
      setError(messageOf(loadError, "The local asset catalog could not be loaded."));
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    // This overlays live runtime state on the saved catalog. Project discovery remains manual.
    void loadCatalog();
    const poll = window.setInterval(() => void loadCatalog(true), CATALOG_POLL_MS);
    return () => window.clearInterval(poll);
  }, [loadCatalog]);

  useEffect(() => {
    const stored = Number(window.localStorage.getItem("metis.assetDrawerWidth"));
    if (!Number.isFinite(stored)) return;
    const next = clampDrawerWidth(stored);
    drawerWidthRef.current = next;
    setDrawerWidth(next);
  }, []);

  useEffect(() => () => {
    if (closeTimerRef.current != null) window.clearTimeout(closeTimerRef.current);
  }, []);

  const selected = assets.find((asset) => asset.id === selectedId) ?? null;
  const selectedActive = selected ? isActive(selected) : false;
  const player = assets.find((asset) => asset.id === playerId) ?? null;
  const playerUrl = player ? launchUrl(player.url) : null;

  const categories = useMemo(
    () => Array.from(new Set(assets.map((asset) => asset.category).filter(Boolean)))
      .sort((left, right) => left.localeCompare(right)),
    [assets],
  );

  const visibleAssets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return assets.filter((asset) => {
      if (category !== "all" && asset.category !== category) return false;
      if (!assetMatchesLifecycle(asset, lifecycle)) return false;
      if (!needle) return true;
      return [
        asset.name,
        asset.summary,
        asset.category,
        asset.framework,
        asset.entrypoint,
        ...asset.tags,
      ].some((value) => value.toLowerCase().includes(needle));
    });
  }, [assets, category, lifecycle, query]);

  const counts = useMemo(() => ({
    all: assets.length,
    running: assets.filter(isActive).length,
    ready: assets.filter((asset) => asset.launchApproved && !isActive(asset)).length,
    review: assets.filter((asset) => asset.launchConfigured && !asset.launchApproved).length,
    setup: assets.filter((asset) => !asset.launchConfigured).length,
  }), [assets]);

  const closeDrawer = useCallback(() => {
    drawerResizingRef.current = false;
    setDrawerResizing(false);
    setDrawerOpen(false);
    if (closeTimerRef.current != null) window.clearTimeout(closeTimerRef.current);
    closeTimerRef.current = window.setTimeout(() => {
      setSelectedId(null);
      setActionError(null);
      setLogsError(null);
      setEnvError(null);
      setEnvSaved(null);
      returnFocusRef.current?.focus();
      closeTimerRef.current = null;
    }, 300);
  }, []);

  function rememberDrawerWidth(width: number) {
    const next = clampDrawerWidth(width);
    drawerWidthRef.current = next;
    setDrawerWidth(next);
    window.localStorage.setItem("metis.assetDrawerWidth", String(next));
  }

  function handleDrawerResizeStart(event: ReactPointerEvent<HTMLDivElement>) {
    if (!drawerOpen || event.button !== 0) return;
    drawerResizeOriginRef.current = { pointerX: event.clientX, width: drawerWidthRef.current };
    drawerResizingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawerResizing(true);
  }

  function handleDrawerResizeMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!drawerResizingRef.current) return;
    const next = clampDrawerWidth(
      drawerResizeOriginRef.current.width + drawerResizeOriginRef.current.pointerX - event.clientX,
    );
    drawerWidthRef.current = next;
    setDrawerWidth(next);
  }

  function handleDrawerResizeEnd(event: ReactPointerEvent<HTMLDivElement>) {
    if (!drawerResizingRef.current) return;
    drawerResizingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    window.localStorage.setItem("metis.assetDrawerWidth", String(drawerWidthRef.current));
    setDrawerResizing(false);
  }

  function handleDrawerResizeKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? 1 : event.key === "ArrowRight" ? -1 : 0;
    if (direction) rememberDrawerWidth(drawerWidthRef.current + direction * 24);
  }

  useEffect(() => {
    if (drawerOpen && selectedId && !loading && !selected) closeDrawer();
  }, [closeDrawer, drawerOpen, loading, selected, selectedId]);

  useEffect(() => {
    if (playerId && !loading && !player) setPlayerId(null);
  }, [loading, player, playerId]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeDrawer();
    };
    document.addEventListener("keydown", onKeyDown);
    const frame = window.requestAnimationFrame(() => closeButtonRef.current?.focus());
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      window.cancelAnimationFrame(frame);
    };
  }, [closeDrawer, drawerOpen]);

  const loadLogs = useCallback(async (assetId: string, showBusy = true) => {
    logRequestRef.current = assetId;
    if (showBusy) setLogsBusy(true);
    try {
      const result = await getAssetLogs(assetId);
      if (logRequestRef.current !== assetId) return;
      setLogs(result.logs);
      setLogsError(null);
    } catch (logLoadError) {
      if (logRequestRef.current !== assetId) return;
      setLogsError(messageOf(logLoadError, "Runtime output is not available yet."));
    } finally {
      if (logRequestRef.current === assetId && showBusy) setLogsBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!drawerOpen || !selectedId) return;
    setLogs(null);
    setLogsError(null);
    void loadLogs(selectedId);
    if (!selectedActive) return;
    const poll = window.setInterval(() => void loadLogs(selectedId, false), LOG_POLL_MS);
    return () => window.clearInterval(poll);
  }, [drawerOpen, loadLogs, selectedActive, selectedId]);

  async function refreshCatalog() {
    setScanning(true);
    setError(null);
    setScanNotice(null);
    try {
      const nextAssets = await scanAssets();
      setAssets(nextAssets);
      setScanNotice(
        nextAssets.length === 1
          ? "Scan complete: 1 project is in the catalog."
          : `Scan complete: ${nextAssets.length} projects are in the catalog.`,
      );
    } catch (scanError) {
      setError(messageOf(scanError, "Metis could not scan the projects folder."));
    } finally {
      setScanning(false);
    }
  }

  function openSettings(assetId: string) {
    if (closeTimerRef.current != null) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    setSelectedId(assetId);
    setDrawerOpen(true);
    setActionError(null);
    setLogs(null);
    setLogsError(null);
    setEnvError(null);
    setEnvSaved(null);
  }

  function openAsset(asset: AssetV1) {
    if (isActive(asset)) {
      setPlayerId(asset.id);
      return;
    }
    openSettings(asset.id);
  }

  function updateEnv(assetId: string, key: string, value: string) {
    setEnvSaved(null);
    setEnvError(null);
    setEnvByAsset((current) => ({
      ...current,
      [assetId]: { ...current[assetId], [key]: value },
    }));
  }

  /** Only non-empty edits are sent: a blank field means "keep what is on disk". */
  function pendingEnv(asset: AssetV1): Record<string, string> {
    const edits = envByAsset[asset.id] ?? {};
    return Object.fromEntries(
      asset.envFile
        .map((variable) => [variable.key, edits[variable.key] ?? ""] as const)
        .filter(([, value]) => value.length > 0),
    );
  }

  async function saveEnv(asset: AssetV1) {
    const values = pendingEnv(asset);
    if (!Object.keys(values).length || envSaving) return;
    setEnvSaving(true);
    setEnvError(null);
    setEnvSaved(null);
    try {
      const updated = await saveAssetEnv(asset.id, values);
      setAssets((current) => mergeAsset(current, updated));
      // Clear the typed values once they live in .env, so nothing lingers in the page.
      setEnvByAsset((current) => ({ ...current, [asset.id]: {} }));
      setRevealedEnv({});
      const count = Object.keys(values).length;
      setEnvSaved(`Saved ${count} ${count === 1 ? "variable" : "variables"} to .env.`);
    } catch (saveError) {
      setEnvError(messageOf(saveError, "The project .env file could not be updated."));
    } finally {
      setEnvSaving(false);
    }
  }

  async function runAction(asset: AssetV1, action: AssetAction) {
    if (busyAction) return;
    setBusyAction({ assetId: asset.id, action });
    setActionError(null);
    try {
      // Runtime values now live in the project's .env, which the API loads into
      // the child process, so a launch carries no environment of its own.
      const updated = action === "start"
        ? await startAsset(asset.id, {})
        : action === "stop"
          ? await stopAsset(asset.id)
          : action === "approve"
            ? await approveAsset(asset.id)
            : await revokeAssetApproval(asset.id);
      setAssets((current) => mergeAsset(current, updated));
      if (action === "start") {
        setPlayerId(updated.id);
        closeDrawer();
      } else if (action === "stop" && playerId === updated.id) {
        setPlayerId(null);
      }
      void loadCatalog(true);
    } catch (assetError) {
      setActionError(messageOf(
        assetError,
        action === "start"
          ? "This asset could not be started."
          : action === "stop"
            ? "This asset could not be stopped."
            : "The launch trust decision could not be saved.",
      ));
    } finally {
      setBusyAction(null);
    }
  }

  function approveLaunchRecipe() {
    if (!selected || !selected.launchConfigured) return;
    const accepted = window.confirm(
      `Trust this exact launch recipe for ${selected.name}?\n\n${commandLabel(selected.launchCommand)}\n\nThis starts project code directly on your Mac with your user account's filesystem and network access. Recipes using uv may prepare an isolated dependency environment on first launch. Any recipe change will require approval again.`,
    );
    if (accepted) void runAction(selected, "approve");
  }

  const selectedBusyAction = selected && busyAction?.assetId === selected.id
    ? busyAction.action
    : null;

  return (
    <div className={`workspacePage assetsPage ${player ? "isPlayerOpen" : ""}`}>
      {player ? (
        <section className="assetPlayer" aria-labelledby="asset-player-title">
          <header className="assetPlayerBar">
            <button
              className="assetPlayerBack"
              type="button"
              onClick={() => {
                setPlayerId(null);
                setActionError(null);
              }}
            >
              <span aria-hidden="true">←</span>
              Back to assets
            </button>
            <div className="assetPlayerIdentity">
              <span>{player.category} · {player.framework}</span>
              <h1 id="asset-player-title">{player.name}</h1>
            </div>
            <span className={`assetStatus ${statusClass(player)}`} aria-live="polite">
              <i aria-hidden="true" />
              {statusLabel(player.status)}
            </span>
            <div className="assetPlayerActions">
              <button
                className="secondaryButton"
                type="button"
                onClick={() => openSettings(player.id)}
              >
                Settings
              </button>
              {isActive(player) ? (
                <button
                  className="dangerButton"
                  type="button"
                  disabled={busyAction != null || normalizedStatus(player) === "stopping"}
                  onClick={() => void runAction(player, "stop")}
                >
                  {busyAction?.assetId === player.id && busyAction.action === "stop"
                    || normalizedStatus(player) === "stopping"
                    ? "Stopping…"
                    : "Stop"}
                </button>
              ) : null}
              {isRunning(player) && playerUrl ? (
                <a href={playerUrl} target="_blank" rel="noreferrer">Open separately ↗</a>
              ) : null}
            </div>
          </header>

          {actionError ? <div className="assetActionError assetPlayerError" role="alert">{actionError}</div> : null}

          <div className="assetPlayerCanvas">
            {isRunning(player) && playerUrl ? (
              <div className="assetPlayerFrame">
                <iframe
                  src={playerUrl}
                  title={`${player.name} live preview`}
                  sandbox="allow-downloads allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts"
                  allow="autoplay; camera; clipboard-read; clipboard-write; fullscreen; microphone"
                  allowFullScreen
                />
              </div>
            ) : (
              <div
                className={`assetPlayerPlaceholder ${normalizedStatus(player) === "stopping"
                  ? "isStopping"
                  : isActive(player)
                    ? "isStarting"
                    : "isUnavailable"}`}
                role="status"
                aria-live="polite"
              >
                {isActive(player) ? <span className="assetPlayerSpinner" aria-hidden="true" /> : null}
                <h2>
                  {normalizedStatus(player) === "stopping"
                    ? `Stopping ${player.name}`
                    : normalizedStatus(player) === "running"
                      ? `Connecting to ${player.name}`
                    : isActive(player)
                      ? `Starting ${player.name}`
                      : `${player.name} is not running`}
                </h2>
                <p>
                  {normalizedStatus(player) === "stopping"
                    ? "Metis is closing the process and releasing its local port."
                    : isActive(player)
                      ? "The preview will appear here automatically as soon as the local app is ready."
                      : "Open Settings to review the process logs or start the asset again."}
                </p>
              </div>
            )}
          </div>
        </section>
      ) : (
        <>
          <header className="pageHeader assetsHeader">
            <div className="assetHeaderCopy">
              <span className="eyebrow">Local project collection</span>
              <h1>Assets</h1>
              <p className="assetCatalogSummary">
                <span><strong>{counts.all}</strong> assets</span>
                <span className={counts.running > 0 ? "isActive" : ""}><strong>{counts.running}</strong> active</span>
                <span>New projects appear only when you scan.</span>
              </p>
            </div>
            <button
              className="secondaryButton assetRefreshButton"
              type="button"
              onClick={() => void refreshCatalog()}
              disabled={scanning}
            >
              <span aria-hidden="true">↻</span>
              {scanning ? "Scanning for updates…" : "Scan for updates"}
            </button>
          </header>

          <section className="assetToolbar" aria-label="Find and filter assets">
            <label className="assetSearch">
              <span className="assetFieldLabel">Search</span>
              <span className="assetSearchControl">
                <span aria-hidden="true">⌕</span>
                <input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search assets"
                />
              </span>
            </label>

            <label className="assetSelectField">
              <span className="assetFieldLabel">Category</span>
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                <option value="all">All categories</option>
                {categories.map((item) => <option value={item} key={item}>{item}</option>)}
              </select>
            </label>

            <label className="assetSelectField assetStateField">
              <span className="assetFieldLabel">State</span>
              <select
                value={lifecycle}
                onChange={(event) => setLifecycle(event.target.value as LifecycleFilter)}
              >
                <option value="all">All assets ({counts.all})</option>
                <option value="running">Active ({counts.running})</option>
                <option value="ready">Ready ({counts.ready})</option>
                <option value="review">Needs review ({counts.review})</option>
                <option value="setup">Setup needed ({counts.setup})</option>
              </select>
            </label>
            <div className="assetToolbarSummary" aria-live="polite">
              <strong>{visibleAssets.length}</strong>
              <span>{visibleAssets.length === 1 ? "match" : "matches"}</span>
              {(query || category !== "all" || lifecycle !== "all") ? (
                <button type="button" onClick={() => { setQuery(""); setCategory("all"); setLifecycle("all"); }}>Clear</button>
              ) : null}
            </div>
          </section>

          {error ? (
            <div className="notice errorNotice" role="alert">
              <strong>Asset library unavailable</strong>
              <span>{error}</span>
            </div>
          ) : null}
          {scanNotice ? (
            <div className="assetScanNotice" role="status">
              <span aria-hidden="true">✓</span>
              {scanNotice}
            </div>
          ) : null}
          {actionError ? <div className="assetActionError" role="alert">{actionError}</div> : null}

          {loading ? (
            <div className="assetGrid" aria-label="Loading assets" aria-busy="true">
              {[0, 1, 2, 3, 4, 5].map((item) => <div className="assetSkeleton" key={item} />)}
            </div>
          ) : visibleAssets.length ? (
            <section className="assetGrid" aria-label="Asset catalog">
              {visibleAssets.map((asset) => {
                const active = isActive(asset);
                const stopping = normalizedStatus(asset) === "stopping";
                const stoppingRequest = busyAction?.assetId === asset.id && busyAction.action === "stop";
                return (
                  <article className={`assetCard ${active ? "isActive" : ""}`} key={asset.id}>
                    <button
                      className="assetCardOpen"
                      type="button"
                      onClick={() => openAsset(asset)}
                      aria-label={active ? `Open ${asset.name} live preview` : `Configure ${asset.name}`}
                    >
                      <span className="assetCardTopline">
                        <span className="assetCardIdentityWrap">
                          <span className="assetMonogram" data-tone={tagTone(asset.category)} aria-hidden="true">{initials(asset.name)}</span>
                          <span className="assetCardIdentity">
                            <strong>{asset.name}</strong>
                            <small>{asset.category}</small>
                          </span>
                        </span>
                        <span className={`assetStatus ${statusClass(asset)}`}>
                          <i aria-hidden="true" />
                          {asset.launchApproved
                            ? statusLabel(asset.status)
                            : asset.launchConfigured
                              ? "Trust required"
                              : "Setup needed"}
                        </span>
                      </span>
                      <span className="assetSummary">{asset.summary}</span>
                      <span className="assetTags" aria-label="Asset tags">
                        {displayTags(asset, 4).map((tag) => <span data-tone={tagTone(tag)} key={tag}>{tag}</span>)}
                      </span>
                      <span className="assetCardFooter"><b>{asset.entrypoint || "Project"}</b><span aria-hidden="true">Open ↗</span></span>
                    </button>
                    {active ? (
                      <button
                        className="assetCardStop"
                        type="button"
                        disabled={busyAction != null || stopping}
                        onClick={() => void runAction(asset, "stop")}
                        aria-label={`Stop ${asset.name}`}
                      >
                        {stoppingRequest || stopping ? "Stopping…" : "Stop"}
                      </button>
                    ) : null}
                  </article>
                );
              })}
            </section>
          ) : (
            <section className="assetEmpty" aria-live="polite">
              <span aria-hidden="true">◇</span>
              <h2>{assets.length ? "No assets match these filters" : "No projects discovered yet"}</h2>
              <p>
                {assets.length
                  ? "Try another search, category, or launch state."
                  : "Choose Scan for updates to inspect the projects folder. Launchable projects also need a reviewed .metis/asset.json recipe."}
              </p>
              {assets.length ? (
                <button
                  className="textButton"
                  type="button"
                  onClick={() => { setQuery(""); setCategory("all"); setLifecycle("all"); }}
                >
                  Clear filters
                </button>
              ) : (
                <button className="primaryButton" type="button" onClick={() => void refreshCatalog()} disabled={scanning}>
                  {scanning ? "Scanning for updates…" : "Scan for updates"}
                </button>
              )}
            </section>
          )}
        </>
      )}

      <div className={`assetDrawerLayer ${drawerOpen ? "isOpen" : ""} ${drawerResizing ? "isResizing" : ""}`} aria-hidden={!drawerOpen}>
        <button
          className="assetDrawerScrim"
          type="button"
          aria-label="Close asset details"
          tabIndex={drawerOpen ? 0 : -1}
          onClick={closeDrawer}
        />
        <aside
          className="assetDrawer"
          role="dialog"
          aria-modal="true"
          aria-labelledby={selected ? "asset-drawer-title" : undefined}
          style={{ "--asset-drawer-width": `${drawerWidth}px` } as CSSProperties}
        >
          <div
            className="assetDrawerResizeHandle"
            role="separator"
            aria-label="Resize asset sidebar"
            aria-orientation="vertical"
            aria-valuemin={MIN_ASSET_DRAWER_WIDTH}
            aria-valuemax={MAX_ASSET_DRAWER_WIDTH}
            aria-valuenow={drawerWidth}
            tabIndex={drawerOpen ? 0 : -1}
            onDoubleClick={() => rememberDrawerWidth(DEFAULT_ASSET_DRAWER_WIDTH)}
            onKeyDown={handleDrawerResizeKeyDown}
            onPointerDown={handleDrawerResizeStart}
            onPointerMove={handleDrawerResizeMove}
            onPointerUp={handleDrawerResizeEnd}
            onPointerCancel={handleDrawerResizeEnd}
            onLostPointerCapture={handleDrawerResizeEnd}
            title="Drag to resize · Double-click to reset"
          ><span /></div>
          {selected ? (
            <>
              <header className="assetDrawerHeader">
                <div>
                  <span className="assetDrawerKicker">{selected.category} · {selected.framework}</span>
                  <h2 id="asset-drawer-title">{selected.name}</h2>
                  <p>{selected.summary}</p>
                  <div className="assetDrawerTags" aria-label="Asset tags">
                    {displayTags(selected).map((tag) => <span data-tone={tagTone(tag)} key={tag}>{tag}</span>)}
                  </div>
                </div>
                <button
                  ref={closeButtonRef}
                  className="assetDrawerClose"
                  type="button"
                  aria-label="Close asset details"
                  onClick={closeDrawer}
                >
                  ×
                </button>
              </header>

              <div className="assetDrawerBody">
                <div className="assetRuntimeStrip">
                  <span className={`assetStatus ${statusClass(selected)}`}>
                    <i aria-hidden="true" />
                    {selected.launchApproved
                      ? statusLabel(selected.status)
                      : selected.launchConfigured
                        ? "Trust required"
                        : "Setup needed"}
                  </span>
                  <code>{selected.entrypoint || "Entrypoint not reported"}</code>
                </div>

                {!selected.launchConfigured ? (
                  <section className="assetRecipeNotice" aria-labelledby="asset-recipe-title">
                    <span aria-hidden="true">!</span>
                    <div>
                      <h3 id="asset-recipe-title">A reviewed launch recipe is required</h3>
                      <p>
                        Metis discovered this project, but will not guess how to run it. Add and
                        review <code>.metis/asset.json</code> before launch controls become available.
                      </p>
                    </div>
                  </section>
                ) : null}

                {selected.launchConfigured && !selected.launchApproved ? (
                  <section className="assetRecipeNotice assetTrustReview" aria-labelledby="asset-trust-title">
                    <span aria-hidden="true">!</span>
                    <div>
                      <h3 id="asset-trust-title">Review this host launch recipe</h3>
                      <p>
                        This command runs directly as your macOS user and can access your files and
                        network. Metis remembers approval only for this exact recipe.
                      </p>
                      <pre>{commandLabel(selected.launchCommand)}</pre>
                      <button
                        className="primaryButton"
                        type="button"
                        disabled={busyAction != null}
                        onClick={approveLaunchRecipe}
                      >
                        {selectedBusyAction === "approve" ? "Saving trust…" : "Trust this exact recipe"}
                      </button>
                    </div>
                  </section>
                ) : null}

                {selected.launchApproved ? (
                  <section className="assetTrustApproved" aria-label="Launch recipe trust">
                    <div>
                      <span aria-hidden="true">✓</span>
                      <p><strong>Exact recipe trusted</strong><small>A manifest change automatically requires review again.</small></p>
                    </div>
                    <button
                      type="button"
                      disabled={busyAction != null || selectedActive}
                      onClick={() => void runAction(selected, "revoke")}
                      title={selectedActive ? "Stop the asset before revoking trust" : "Require review before the next launch"}
                    >
                      {selectedBusyAction === "revoke" ? "Revoking…" : "Revoke"}
                    </button>
                  </section>
                ) : null}

                <details
                  className="assetDisclosure assetEnvSection"
                  open={disclosureOpen[`${selected.id}:environment`] ?? !selectedActive}
                  key={`environment-${selected.id}`}
                  onToggle={(event) => {
                    const open = event.currentTarget.open;
                    setDisclosureOpen((current) => ({
                      ...current,
                      [`${selected.id}:environment`]: open,
                    }));
                  }}
                >
                  <summary className="assetDisclosureSummary">
                    <span>
                      <span className="assetSectionLabel">Runtime configuration</span>
                      <strong id="asset-env-title">.env</strong>
                    </span>
                    <span>
                      {selected.envFilePresent
                        ? `${selected.envFile.length} ${selected.envFile.length === 1 ? "variable" : "variables"}`
                        : "No file"}
                    </span>
                  </summary>
                  <div className="assetDisclosureBody">
                    {selected.envFilePresent ? (
                      <>
                        <p>
                          These are the variables in this project&apos;s own <code>.env</code>.
                          Metis reports only whether a value is set — it never reads one back
                          into this page. Typing a value replaces it in the file.
                        </p>
                        <div className="assetEnvList">
                          {selected.envFile.map(({ key, isSet, sensitive }) => {
                            const fieldId = `asset-env-${selected.id}-${key}`.replace(/[^a-zA-Z0-9_-]/g, "-");
                            const revealId = `${selected.id}:${key}`;
                            const revealed = revealedEnv[revealId] === true;
                            const draft = envByAsset[selected.id]?.[key] ?? "";
                            return (
                              <label
                                className={`assetEnvField ${draft ? "isEdited" : ""}`}
                                htmlFor={fieldId}
                                key={key}
                              >
                                <span>
                                  <code>{key}</code>
                                  <small className={isSet ? "isSetBadge" : "isUnsetBadge"}>
                                    {isSet ? "Set" : "Empty"}
                                  </small>
                                  {sensitive ? <small>Sensitive</small> : null}
                                </span>
                                <span className="assetEnvControl">
                                  <input
                                    id={fieldId}
                                    type={sensitive && !revealed ? "password" : "text"}
                                    value={draft}
                                    onChange={(event) => updateEnv(selected.id, key, event.target.value)}
                                    autoComplete="off"
                                    spellCheck={false}
                                    placeholder={isSet ? "Value set — type to replace" : `Enter ${key}`}
                                  />
                                  {sensitive ? (
                                    <button
                                      type="button"
                                      aria-label={`${revealed ? "Hide" : "Show"} ${key}`}
                                      aria-pressed={revealed}
                                      onClick={() => setRevealedEnv((current) => ({
                                        ...current,
                                        [revealId]: !revealed,
                                      }))}
                                    >
                                      {revealed ? "Hide" : "Show"}
                                    </button>
                                  ) : null}
                                </span>
                              </label>
                            );
                          })}
                        </div>
                        {envError ? <p className="assetLogsError" role="alert">{envError}</p> : null}
                        {envSaved ? <p className="assetEnvSaved" role="status">{envSaved}</p> : null}
                        <div className="assetEnvActions">
                          <button
                            className="primaryButton"
                            type="button"
                            disabled={envSaving || !Object.keys(pendingEnv(selected)).length}
                            onClick={() => void saveEnv(selected)}
                          >
                            {envSaving ? "Saving…" : "Save to .env"}
                          </button>
                          {selectedActive ? (
                            <span className="assetEnvHint">
                              Saved values take effect the next time this asset starts.
                            </span>
                          ) : null}
                        </div>
                      </>
                    ) : (
                      <p>
                        This project has no <code>.env</code> file, so there is nothing to
                        configure. Add one in the project folder and it will appear here.
                      </p>
                    )}
                  </div>
                </details>

                {actionError ? <div className="assetActionError" role="alert">{actionError}</div> : null}

                {selected.launchApproved ? (
                  <div className="assetLaunchActions">
                    {selectedActive ? (
                      <button
                        className="dangerButton"
                        type="button"
                        disabled={busyAction != null || normalizedStatus(selected) === "stopping"}
                        onClick={() => void runAction(selected, "stop")}
                      >
                        {selectedBusyAction === "stop" || normalizedStatus(selected) === "stopping"
                          ? "Stopping…"
                          : "Stop asset"}
                      </button>
                    ) : (
                      <button
                        className="primaryButton"
                        type="button"
                        disabled={busyAction != null}
                        onClick={() => void runAction(selected, "start")}
                      >
                        {selectedBusyAction === "start"
                          ? "Starting…"
                          : "Start asset"}
                      </button>
                    )}
                  </div>
                ) : null}

                <details
                  className="assetDisclosure assetLogsSection"
                  open={disclosureOpen[`${selected.id}:logs`]
                    ?? ["error", "failed", "crashed"].includes(normalizedStatus(selected))}
                  key={`logs-${selected.id}`}
                  onToggle={(event) => {
                    const open = event.currentTarget.open;
                    setDisclosureOpen((current) => ({
                      ...current,
                      [`${selected.id}:logs`]: open,
                    }));
                  }}
                >
                  <summary className="assetDisclosureSummary">
                    <span>
                      <span className="assetSectionLabel">Process output</span>
                      <strong id="asset-logs-title">Logs</strong>
                    </span>
                    <span>{selectedActive ? "Live" : "View"}</span>
                  </summary>
                  <div className="assetDisclosureBody">
                    <button
                      className="textButton"
                      type="button"
                      onClick={() => void loadLogs(selected.id)}
                      disabled={logsBusy}
                    >
                      {logsBusy ? "Loading…" : "Refresh logs"}
                    </button>
                    {logsError ? <p className="assetLogsError">{logsError}</p> : null}
                    <pre aria-live="polite">{logsBusy && logs == null ? "Loading runtime output…" : logs || "No runtime output yet."}</pre>
                  </div>
                </details>
              </div>
            </>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
