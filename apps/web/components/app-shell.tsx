"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import { MetisCompanion } from "@/components/metis-companion";
import { deleteConversation, getRunRecord, listConversations } from "@/lib/api";
import { freshToken } from "@/lib/token";
import {
  CONVERSATIONS_CHANGED_EVENT,
  forgetConversation,
  readRecentConversations,
} from "@/lib/recent-conversations";
import type { ConversationSummary } from "@/lib/types";
import {
  RUN_INDICATORS_CHANGED_EVENT,
  acknowledgeConversationRun,
  readRunIndicators,
  updateConversationRun,
  type ConversationRunIndicator,
  type ConversationRunState,
} from "@/lib/run-indicators";

type NavIconName =
  | "chat"
  | "customers"
  | "assets"
  | "tools"
  | "knowledge"
  | "meetings"
  | "interviews"
  | "agents"
  | "memory"
  | "sizing"
  | "today"
  | "answers"
  | "settings";

const navigation: Array<{
  href: string;
  label: string;
  icon: NavIconName;
}> = [
  // Today is the front door: one ranked queue of everything waiting, so work
  // stops hiding behind eight equal destinations.
  { href: "/today", label: "Today", icon: "today" },
  { href: "/", label: "Chat", icon: "chat" },
  { href: "/customers", label: "Customers", icon: "customers" },
  { href: "/assets", label: "Assets", icon: "assets" },
  { href: "/tools", label: "Tool Workshop", icon: "tools" },
  { href: "/meetings", label: "Meetings", icon: "meetings" },
  { href: "/interviews", label: "Interviews", icon: "interviews" },
  { href: "/agents", label: "Agents", icon: "agents" },
  { href: "/knowledge", label: "Knowledge", icon: "knowledge" },
  { href: "/answers", label: "Answers", icon: "answers" },
  { href: "/memory", label: "Memory", icon: "memory" },
  { href: "/sizing", label: "Sizing", icon: "sizing" },
  { href: "/settings", label: "Settings", icon: "settings" },
];

type SidebarSectionId = "focus" | "conversations" | "library" | "capture" | "build" | "system";

const sidebarSections: Array<{
  id: SidebarSectionId;
  label: string;
  eyebrow: string;
  description: string;
  icon: NavIconName;
  hrefs: string[];
}> = [
  { id: "focus", label: "Focus", eyebrow: "Workspace", description: "What needs you now", icon: "today", hrefs: ["/today"] },
  { id: "conversations", label: "Chat", eyebrow: "Conversations", description: "Think, make, and decide", icon: "chat", hrefs: ["/"] },
  { id: "library", label: "Library", eyebrow: "Intelligence", description: "Your connected context", icon: "knowledge", hrefs: ["/customers", "/assets", "/knowledge", "/answers", "/memory", "/sizing"] },
  { id: "capture", label: "Capture", eyebrow: "Research", description: "Turn conversations into signal", icon: "meetings", hrefs: ["/meetings", "/interviews"] },
  { id: "build", label: "Build", eyebrow: "Capabilities", description: "Create reusable workflows", icon: "tools", hrefs: ["/tools"] },
  { id: "system", label: "Settings", eyebrow: "Metis", description: "Preferences and governance", icon: "settings", hrefs: ["/settings"] },
];

function sectionForPathname(pathname: string): SidebarSectionId {
  return sidebarSections.find((section) =>
    section.hrefs.some((href) => href === "/" ? pathname === "/" : pathname.startsWith(href)),
  )?.id ?? "conversations";
}

const DEFAULT_SIDEBAR_WIDTH = 272;
const MIN_SIDEBAR_WIDTH = 248;
const MAX_SIDEBAR_WIDTH = 304;
const COMPACT_HISTORY_LIMIT = 12;

function clampSidebarWidth(value: number): number {
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, value));
}

function NavIcon({ name }: { name: NavIconName }) {
  const common = {
    fill: "none",
    stroke: "currentColor",
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 1.55,
  };
  if (name === "today") {
    // A checklist: the page is a queue of decisions, not a destination.
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M4.2 5.6 5.4 6.8l2.2-2.4M4.2 10.4l1.2 1.2 2.2-2.4M4.2 15.2l1.2 1.2 2.2-2.4M10.6 5.3h5.2M10.6 10.1h5.2M10.6 14.9h5.2" /></svg>;
  }
  if (name === "answers") {
    // A speech mark: the bank holds things you have already said well.
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M7.4 8.6H5.2a1.4 1.4 0 0 1-1.4-1.4V5.8a1.4 1.4 0 0 1 1.4-1.4h2.2a1.4 1.4 0 0 1 1.4 1.4v3.9c0 2.1-1.2 3.6-3 4.1M16.2 8.6H14a1.4 1.4 0 0 1-1.4-1.4V5.8A1.4 1.4 0 0 1 14 4.4h2.2a1.4 1.4 0 0 1 1.4 1.4v3.9c0 2.1-1.2 3.6-3 4.1" /></svg>;
  }
  if (name === "chat") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M4 4.7h12v8.1H9l-3.7 2.8v-2.8H4z" /></svg>;
  }
  if (name === "customers") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M6.6 9.4a2.7 2.7 0 1 0 0-5.4 2.7 2.7 0 0 0 0 5.4ZM2.8 16c.2-3 1.4-4.5 3.8-4.5s3.6 1.5 3.8 4.5M13 9a2.2 2.2 0 1 0 0-4.4M11.8 11.7c3.1-.5 4.8.9 5 4.3" /></svg>;
  }
  if (name === "assets") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M3.7 5.3 10 2.8l6.3 2.5L10 7.8zM3.7 9.1 10 11.6l6.3-2.5M3.7 12.9 10 15.4l6.3-2.5" /></svg>;
  }
  if (name === "tools") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="m11.7 4.1 4.2 4.2-7.6 7.6-4.2-4.2zM10.1 5.7l4.2 4.2M3.5 16.5l2.1-.6-1.5-1.5z" /></svg>;
  }
  if (name === "knowledge") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M3.7 4.3h4.1c1.2 0 2.2.8 2.2 1.8v9.3c0-1-1-1.8-2.2-1.8H3.7zM16.3 4.3h-4.1c-1.2 0-2.2.8-2.2 1.8v9.3c0-1 1-1.8 2.2-1.8h4.1z" /></svg>;
  }
  if (name === "meetings") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M10 3.2a2 2 0 0 1 2 2v4.4a2 2 0 1 1-4 0V5.2a2 2 0 0 1 2-2ZM5.4 9.2a4.6 4.6 0 0 0 9.2 0M10 13.8v3" /></svg>;
  }
  if (name === "interviews") {
    // A question in a speech bubble: five of these, then the verdict.
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M4.6 3.9h10.8a1.6 1.6 0 0 1 1.6 1.6v6a1.6 1.6 0 0 1-1.6 1.6H9.6l-3.2 3v-3H4.6A1.6 1.6 0 0 1 3 11.5v-6a1.6 1.6 0 0 1 1.6-1.6ZM8.3 6.9a1.7 1.7 0 1 1 2.5 1.9c-.6.35-.8.7-.8 1.3M10 11.4v.05" /></svg>;
  }
  if (name === "agents") {
    // A microphone on a stand: an agent that speaks for someone else.
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M10 3.4a3.2 3.2 0 0 1 3.2 3.2v3a3.2 3.2 0 0 1-6.4 0v-3A3.2 3.2 0 0 1 10 3.4ZM5.2 9.6a4.8 4.8 0 0 0 9.6 0M10 14.4v2.2M7.4 16.6h5.2" /></svg>;
  }
  if (name === "memory") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M10 3.4a6.6 6.6 0 1 1-4.7 2M3.4 3.8v3h3M10 6.6v3.7l2.6 1.5" /></svg>;
  }
  if (name === "sizing") {
    return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M3.4 16.2V9.7h3.4v6.5zm5.1 0V4.5h3v11.7zm4.7 0v-8.4h3.4v8.4z" /></svg>;
  }
  return <svg viewBox="0 0 20 20" aria-hidden="true"><path {...common} d="M4 5.2h12M4 10h12M4 14.8h12M7 3.6v3.2M13 8.4v3.2M8.5 13.2v3.2" /></svg>;
}

function groupLabel(timestamp?: string): string {
  if (!timestamp) return "Earlier";
  const date = new Date(timestamp);
  const now = new Date();
  const delta = now.getTime() - date.getTime();
  if (delta < 86_400_000 && date.getDate() === now.getDate()) return "Today";
  if (delta < 7 * 86_400_000) return "Previous 7 days";
  return "Earlier";
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeConversation = searchParams.get("conversation");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [query, setQuery] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [activeSidebarSection, setActiveSidebarSection] = useState<SidebarSectionId>(() => sectionForPathname(pathname));
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [resizing, setResizing] = useState(false);
  const [animate, setAnimate] = useState(false);
  const [apiConnected, setApiConnected] = useState(true);
  const [deletingConversation, setDeletingConversation] = useState<string | null>(null);
  const [runIndicators, setRunIndicators] = useState<Record<string, ConversationRunIndicator>>({});
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const [mobileLayout, setMobileLayout] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const mobileMenuRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const sidebarPanelRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const drawerFocusRef = useRef<HTMLElement | null>(null);
  const resizingRef = useRef(false);
  const resizeOriginRef = useRef({ pointerX: 0, width: DEFAULT_SIDEBAR_WIDTH });
  const sidebarWidthRef = useRef(DEFAULT_SIDEBAR_WIDTH);
  const desktopPanelCollapsed = collapsed && !mobileLayout;

  useEffect(() => {
    // Apply the persisted collapse state on mount WITHOUT a transition (a
    // flex-basis transition fired during hydration sticks at its start value),
    // then enable transitions a frame later so user toggles animate smoothly.
    setCollapsed(window.localStorage.getItem("metis.sidebarCollapsed") === "1");
    // An absent key reads back as null, and Number(null) is 0 — which is finite,
    // so a plain isFinite check would clamp every fresh profile to the minimum
    // width instead of leaving it at the default.
    const savedWidth = Number(window.localStorage.getItem("metis.sidebarWidth"));
    if (Number.isFinite(savedWidth) && savedWidth > 0) {
      const nextWidth = clampSidebarWidth(savedWidth);
      sidebarWidthRef.current = nextWidth;
      setSidebarWidth(nextWidth);
    }
    const id = requestAnimationFrame(() =>
      requestAnimationFrame(() => setAnimate(true)),
    );
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    // A full rail plus detail panel leaves the workspace unusably narrow well
    // before phone width. Use the drawer until there is room for both.
    const query = window.matchMedia("(max-width: 1100px)");
    const sync = () => setMobileLayout(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  // CSS hides the detail panel on a collapsed desktop rail. Mirror that state
  // semantically so clipped controls cannot remain in the keyboard or screen-
  // reader order. Mobile always keeps the panel available inside its drawer.
  useEffect(() => {
    const panel = sidebarPanelRef.current;
    if (!panel) return;
    panel.inert = desktopPanelCollapsed;
    return () => {
      panel.inert = false;
    };
  }, [desktopPanelCollapsed]);

  // The off-canvas navigation is a real modal surface on small screens. Keep
  // closed links out of the tab order, trap focus while it is open, then put
  // focus back on the button that opened it.
  useEffect(() => {
    const sidebar = sidebarRef.current;
    const main = mainRef.current;
    if (!sidebar || !main) return;
    sidebar.inert = mobileLayout && !drawerOpen;
    main.inert = mobileLayout && drawerOpen;
    if (!mobileLayout || !drawerOpen) return;

    drawerFocusRef.current = document.activeElement as HTMLElement | null;
    const focusable = () => [
      mobileMenuRef.current,
      ...Array.from(
        sidebar.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ),
    ].filter((item): item is HTMLElement => Boolean(item));
    window.requestAnimationFrame(() => focusable()[1]?.focus());

    const trap = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setDrawerOpen(false);
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      document.removeEventListener("keydown", trap);
      drawerFocusRef.current?.focus();
      drawerFocusRef.current = null;
    };
  }, [drawerOpen, mobileLayout]);

  useEffect(() => {
    setActiveSidebarSection(sectionForPathname(pathname));
  }, [pathname]);

  useEffect(() => {
    const sync = () => setRunIndicators(readRunIndicators());
    sync();
    window.addEventListener(RUN_INDICATORS_CHANGED_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(RUN_INDICATORS_CHANGED_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  // Runs keep going when Chat is no longer the visible page. The shell stays
  // mounted across navigation, so it owns the small background status check and
  // turns completion into a durable, per-conversation unread signal.
  useEffect(() => {
    let stopped = false;
    const poll = async () => {
      const current = readRunIndicators();
      const pending = Object.values(current).filter((item) => item.state === "working" || item.state === "attention");
      await Promise.all(pending.map(async (item) => {
        try {
          const run = await getRunRecord(item.runId);
          if (stopped) return;
          const state: ConversationRunState =
            run.status === "completed" ? "done"
              : run.status === "failed" ? "failed"
                : run.status === "cancelled" ? "cancelled"
                  : run.status === "awaiting_approval" || run.status === "awaiting_input" ? "attention"
                    : "working";
          const visibleHere = pathname === "/" && activeConversation === item.conversationId && document.visibilityState === "visible";
          const unread = state === "working" ? false : !visibleHere;
          if (state !== item.state || unread !== item.unread) updateConversationRun(item.runId, state, unread);
        } catch {
          // The chat itself owns detailed connection recovery. A shell badge is
          // advisory and should not flash failure on a brief API interruption.
        }
      }));
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2500);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [activeConversation, pathname]);

  useEffect(() => {
    if (pathname !== "/" || !activeConversation) return;
    acknowledgeConversationRun(activeConversation);
  }, [activeConversation, pathname]);

  const toggleCollapsed = () => {
    if (mobileLayout) {
      setDrawerOpen(false);
      return;
    }
    resizingRef.current = false;
    setResizing(false);
    const next = !collapsed;
    setCollapsed(next);
    window.localStorage.setItem("metis.sidebarCollapsed", next ? "1" : "0");
    if (next && sidebarPanelRef.current?.contains(document.activeElement)) {
      window.requestAnimationFrame(() => {
        sidebarRef.current
          ?.querySelector<HTMLButtonElement>(
            `[data-sidebar-section="${activeSidebarSection}"]`,
          )
          ?.focus();
      });
    }
  };

  const openSidebarSection = (section: SidebarSectionId) => {
    if (mobileLayout) {
      setActiveSidebarSection(section);
      return;
    }
    if (activeSidebarSection === section && !collapsed) {
      toggleCollapsed();
      return;
    }
    setActiveSidebarSection(section);
    setCollapsed(false);
    window.localStorage.setItem("metis.sidebarCollapsed", "0");
  };

  const setAndRememberSidebarWidth = (width: number) => {
    const nextWidth = clampSidebarWidth(width);
    sidebarWidthRef.current = nextWidth;
    setSidebarWidth(nextWidth);
    window.localStorage.setItem("metis.sidebarWidth", String(nextWidth));
  };

  const handleResizeStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (collapsed || event.button !== 0) return;
    resizeOriginRef.current = { pointerX: event.clientX, width: sidebarWidthRef.current };
    resizingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setResizing(true);
  };

  const handleResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!resizingRef.current) return;
    const nextWidth = clampSidebarWidth(
      resizeOriginRef.current.width + event.clientX - resizeOriginRef.current.pointerX,
    );
    sidebarWidthRef.current = nextWidth;
    setSidebarWidth(nextWidth);
  };

  const handleResizeEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!resizingRef.current) return;
    resizingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    window.localStorage.setItem("metis.sidebarWidth", String(sidebarWidthRef.current));
    setResizing(false);
  };

  const handleResizeKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
    if (direction) setAndRememberSidebarWidth(sidebarWidthRef.current + direction * 16);
  };

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const items = await listConversations();
        if (!mounted) return;
        setConversations(items.length ? items : readRecentConversations());
        setApiConnected(true);
      } catch {
        if (!mounted) return;
        setConversations(readRecentConversations());
        setApiConnected(false);
      }
    };
    void load();
    const onChanged = () => void load();
    window.addEventListener(CONVERSATIONS_CHANGED_EVENT, onChanged);
    return () => {
      mounted = false;
      window.removeEventListener(CONVERSATIONS_CHANGED_EVENT, onChanged);
    };
  }, []);

  useEffect(() => setDrawerOpen(false), [pathname, activeConversation]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Shift is deliberately excluded: ⌘⇧K belongs to whichever page is open
      // (the Customers page searches customer records with it), and this
      // listener would otherwise steal the focus out from under it.
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setActiveSidebarSection("conversations");
        if (!mobileLayout) {
          setCollapsed(false);
          window.localStorage.setItem("metis.sidebarCollapsed", "0");
        }
        setDrawerOpen(true);
        window.setTimeout(() => searchRef.current?.focus(), 0);
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        router.push(`/?new=${freshToken()}`);
      } else if (event.key === "Escape") {
        resizingRef.current = false;
        setResizing(false);
        setDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileLayout, router]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return conversations;
    return conversations.filter((item) =>
      `${item.title} ${item.last_message ?? ""}`.toLowerCase().includes(normalized),
    );
  }, [conversations, query]);

  // The rail is for returning to current work, not for displaying the entire
  // database at once. Search always sees everything; the resting view shows a
  // compact recent set and offers the full history on request.
  const visibleHistory = useMemo(
    () => query.trim() || historyExpanded ? filtered : filtered.slice(0, COMPACT_HISTORY_LIMIT),
    [filtered, historyExpanded, query],
  );

  const grouped = useMemo(() => {
    return visibleHistory.reduce<Record<string, ConversationSummary[]>>((result, item) => {
      const label = groupLabel(item.updated_at ?? item.created_at);
      result[label] = [...(result[label] ?? []), item];
      return result;
    }, {});
  }, [visibleHistory]);

  const unreadRunCount = useMemo(
    () => Object.values(runIndicators).filter((item) => item.unread).length,
    [runIndicators],
  );

  const currentSidebarSection = sectionForPathname(pathname);
  const currentDestination = navigation.find((item) =>
    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href),
  );
  const activeSection = sidebarSections.find((section) => section.id === activeSidebarSection) ?? sidebarSections[1];
  const sectionNavigation = navigation.filter((item) => activeSection.hrefs.includes(item.href));

  const removeConversation = async (conversation: ConversationSummary) => {
    if (deletingConversation || !window.confirm(`Delete “${conversation.title}”? This permanently removes this chat and its messages.`)) return;
    setDeletingConversation(conversation.id);
    try {
      await deleteConversation(conversation.id);
      forgetConversation(conversation.id);
      setConversations((current) => current.filter((item) => item.id !== conversation.id));
      if (activeConversation === conversation.id) router.push("/");
    } catch {
      window.alert("This conversation could not be deleted. Please try again.");
    } finally {
      setDeletingConversation(null);
    }
  };

  return (
    <div
      className={`appShell ${collapsed ? "isCollapsed" : ""} ${animate ? "animate" : ""} ${resizing ? "isResizing" : ""}`}
      data-sidebar-state={desktopPanelCollapsed ? "collapsed" : "expanded"}
      data-sidebar-layout={mobileLayout ? "drawer" : "desktop"}
      data-drawer-state={drawerOpen ? "open" : "closed"}
      data-current-section={currentSidebarSection}
    >
      <div className="ambient" aria-hidden="true">
        <span className="bloom bloom-green" />
        <span className="bloom bloom-coral" />
        <span className="bloom bloom-lav" />
        <span className="bloom bloom-gold" />
      </div>
      <div className="grain" aria-hidden="true" />

      {/* Declared once for the whole app: the turbulence field that bends the
          companion's colour film, so it pools like liquid rather than sliding
          past as a rigid layer. */}
      <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
        <defs>
          <filter id="metisWarp" x="-35%" y="-35%" width="170%" height="170%" colorInterpolationFilters="sRGB">
            <feTurbulence type="fractalNoise" baseFrequency="0.019" numOctaves={2} seed={9} result="n" />
            <feDisplacementMap in="SourceGraphic" in2="n" scale={30} xChannelSelector="R" yChannelSelector="G" />
          </filter>
        </defs>
      </svg>

      <button
        ref={mobileMenuRef}
        className="mobileMenuButton"
        type="button"
        aria-label={drawerOpen ? "Close navigation" : "Open navigation"}
        aria-expanded={drawerOpen}
        aria-controls="metis-main-navigation"
        onClick={() => setDrawerOpen((current) => !current)}
      >
        <span />
        <span />
      </button>

      {drawerOpen ? (
        <button className="drawerScrim" tabIndex={-1} aria-label="Close navigation" onClick={() => setDrawerOpen(false)} />
      ) : null}

      <aside
        ref={sidebarRef}
        id="metis-main-navigation"
        className={`sidebar sidebarShell ${drawerOpen ? "sidebarOpen" : ""}`}
        aria-label="Main navigation"
        aria-hidden={mobileLayout && !drawerOpen ? true : undefined}
        role={mobileLayout ? "dialog" : undefined}
        aria-modal={mobileLayout && drawerOpen ? true : undefined}
        data-sidebar-state={desktopPanelCollapsed ? "collapsed" : "expanded"}
        data-active-section={activeSidebarSection}
        data-current-section={currentSidebarSection}
        data-current-destination={currentDestination?.href ?? ""}
        style={{ "--sidebar-width": `${sidebarWidth}px` } as CSSProperties}
      >
        <div className="sidebarRail sidebarShellRail" data-sidebar-rail>
          <Link href="/" className="railBrand" aria-label="Metis home" title="Metis">
            <span className="brandMark" aria-hidden="true">
              <MetisCompanion size={34} energy="expressive" />
            </span>
          </Link>

          <button
            className="railCompose"
            type="button"
            onClick={() => router.push(`/?new=${freshToken()}`)}
            aria-label="New conversation"
            title="New conversation · ⌘N"
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M10 4v12M4 10h12" />
            </svg>
          </button>

          <nav className="railNav" aria-label="Workspace sections">
            {sidebarSections.filter((section) => section.id !== "system").map((section) => {
              const panelActive = activeSidebarSection === section.id;
              const routeActive = currentSidebarSection === section.id;
              const routeLabel = routeActive && currentDestination
                ? currentDestination.label
                : section.label;
              const tooltip = routeActive && routeLabel !== section.label
                ? `${section.label} · Current: ${routeLabel}`
                : section.label;
              const tooltipId = `sidebar-${section.id}-tooltip`;
              return (
                <button
                  key={section.id}
                  type="button"
                  className={`railSectionButton ${panelActive ? "active isPanelActive" : ""} ${routeActive ? "isRouteActive" : ""}`}
                  data-sidebar-section={section.id}
                  data-panel-active={panelActive ? "true" : "false"}
                  data-route-active={routeActive ? "true" : "false"}
                  data-current-label={routeActive ? routeLabel : undefined}
                  onClick={() => openSidebarSection(section.id)}
                  aria-label={`${mobileLayout ? "Show" : panelActive && !desktopPanelCollapsed ? "Collapse" : "Open"} ${section.label} panel${routeActive ? `, current page ${routeLabel}` : ""}`}
                  aria-controls="metis-sidebar-panel"
                  aria-expanded={panelActive && !desktopPanelCollapsed}
                  aria-current={routeActive ? "location" : undefined}
                  aria-describedby={desktopPanelCollapsed ? tooltipId : undefined}
                  title={tooltip}
                >
                  <span className="railActivePill" aria-hidden="true" />
                  <span className="navGlyph" aria-hidden="true"><NavIcon name={section.icon} /></span>
                  {section.id === "conversations" && unreadRunCount ? <span className="railBadge" aria-hidden="true">{unreadRunCount}</span> : null}
                  <span id={tooltipId} className="railTooltip" role="tooltip">{tooltip}</span>
                </button>
              );
            })}
          </nav>

          <div className="railFooter">
            <span
              className={`railConnection ${apiConnected ? "connected" : "disconnected"}`}
              role="status"
              aria-label={apiConnected ? "Metis is connected" : "Metis is offline"}
              title={apiConnected ? "Metis is connected" : "Metis is offline"}
            />
            <button
              type="button"
              className={`railSectionButton ${activeSidebarSection === "system" ? "active isPanelActive" : ""} ${currentSidebarSection === "system" ? "isRouteActive" : ""}`}
              data-sidebar-section="system"
              data-panel-active={activeSidebarSection === "system" ? "true" : "false"}
              data-route-active={currentSidebarSection === "system" ? "true" : "false"}
              data-current-label={currentSidebarSection === "system" ? "Settings" : undefined}
              onClick={() => openSidebarSection("system")}
              aria-label={`${mobileLayout ? "Show" : activeSidebarSection === "system" && !desktopPanelCollapsed ? "Collapse" : "Open"} Settings panel${currentSidebarSection === "system" ? ", current page" : ""}`}
              aria-controls="metis-sidebar-panel"
              aria-expanded={activeSidebarSection === "system" && !desktopPanelCollapsed}
              aria-current={currentSidebarSection === "system" ? "location" : undefined}
              aria-describedby={desktopPanelCollapsed ? "sidebar-system-tooltip" : undefined}
              title="Settings"
            >
              <span className="railActivePill" aria-hidden="true" />
              <span className="navGlyph" aria-hidden="true"><NavIcon name="settings" /></span>
              <span id="sidebar-system-tooltip" className="railTooltip" role="tooltip">Settings</span>
            </button>
          </div>
        </div>

        <div
          ref={sidebarPanelRef}
          id="metis-sidebar-panel"
          className="sidebarPanel sidebarShellPanel"
          role="region"
          aria-labelledby="metis-sidebar-panel-title"
          aria-hidden={desktopPanelCollapsed ? true : undefined}
          data-panel-state={desktopPanelCollapsed ? "collapsed" : "expanded"}
          data-panel-section={activeSidebarSection}
        >
          <header className="sidebarPanelHeader">
            <div className="sidebarPanelHeading">
              <span className="sidebarPanelEyebrow">{activeSection.eyebrow}</span>
              <strong id="metis-sidebar-panel-title">{activeSection.label}</strong>
              <small className="sidebarPanelDescription">{activeSection.description}</small>
            </div>
            <button
              type="button"
              className="collapseToggle"
              onClick={toggleCollapsed}
              aria-label={mobileLayout ? "Close navigation" : "Collapse sidebar panel"}
              aria-controls="metis-sidebar-panel"
              aria-expanded={!desktopPanelCollapsed}
              title={mobileLayout ? "Close navigation" : "Collapse panel"}
            >
              <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                <path d="M10 3.5 5.5 8 10 12.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </header>

          {activeSidebarSection === "conversations" ? (
            <>
              <button className="newChatButton" type="button" onClick={() => router.push(`/?new=${freshToken()}`)} title="New conversation">
                <span aria-hidden="true">＋</span>
                <span className="navLabel">New conversation</span>
                <kbd>⌘ N</kbd>
              </button>

              <label className="sidebarSearch">
                <span aria-hidden="true">⌕</span>
                <input
                  ref={searchRef}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search conversations"
                  aria-label="Search conversations"
                />
                {query ? <button className="searchClear" type="button" aria-label="Clear search" onClick={() => { setQuery(""); searchRef.current?.focus(); }}>×</button> : null}
                <kbd><span>⌘</span><span>K</span></kbd>
              </label>

              <div className="historyHeader">
                <span>Recent</span>
                <span>{conversations.length}</span>
              </div>
              <div className="conversationHistory">
                {Object.entries(grouped).map(([label, items]) => (
                  <section key={label}>
                    <h2>{label}</h2>
                    {items.map((conversation) => (
                      <div key={conversation.id} className={`conversationHistoryItem run-${runIndicators[conversation.id]?.state ?? "idle"} ${runIndicators[conversation.id]?.unread ? "hasUnread" : ""}`}>
                        <Link href={`/?conversation=${encodeURIComponent(conversation.id)}`} onClick={() => acknowledgeConversationRun(conversation.id)} className={pathname === "/" && activeConversation === conversation.id ? "active" : ""} aria-current={pathname === "/" && activeConversation === conversation.id ? "page" : undefined} title={conversation.title}>
                          <span>{conversation.title}</span>
                          {runIndicators[conversation.id] && (
                            runIndicators[conversation.id].state !== "done" || runIndicators[conversation.id].unread
                          ) ? (
                            <small className="conversationRunState">
                              <i aria-hidden="true" />
                              {runIndicators[conversation.id].state === "working" ? "Working"
                                : runIndicators[conversation.id].state === "attention" ? "Needs you"
                                  : runIndicators[conversation.id].state === "failed" ? "Interrupted"
                                    : runIndicators[conversation.id].state === "cancelled" ? "Stopped"
                                      : runIndicators[conversation.id].unread ? "Done" : ""}
                            </small>
                          ) : null}
                        </Link>
                        <button type="button" className="conversationDeleteButton" aria-label={`Delete ${conversation.title}`} title="Delete conversation" disabled={deletingConversation === conversation.id} onClick={() => void removeConversation(conversation)}>×</button>
                      </div>
                    ))}
                  </section>
                ))}
                {!query.trim() && filtered.length > COMPACT_HISTORY_LIMIT ? (
                  <button
                    className="historyExpandButton"
                    type="button"
                    aria-expanded={historyExpanded}
                    onClick={() => setHistoryExpanded((value) => !value)}
                  >
                    <span>{historyExpanded ? "Show recent only" : `View all ${filtered.length} conversations`}</span>
                    <b aria-hidden="true">{historyExpanded ? "↑" : "↓"}</b>
                  </button>
                ) : null}
                {!filtered.length ? (
                  <p className="historyEmpty">{query ? "No matching conversations" : "Your local conversations will appear here."}</p>
                ) : null}
              </div>
            </>
          ) : (
            <nav className="primaryNav sectionNav" aria-label={activeSection.label}>
              {sectionNavigation.map((item) => {
                const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={active ? "active isCurrentDestination" : ""}
                    data-sidebar-destination={item.href}
                    data-route-active={active ? "true" : "false"}
                    aria-current={active ? "page" : undefined}
                    title={item.label}
                  >
                    <span className="navGlyph" aria-hidden="true"><NavIcon name={item.icon} /></span>
                    <span className="navLabel">{item.label}</span>
                    <svg className="navChevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3.5 4.5 4.5L6 12.5" /></svg>
                  </Link>
                );
              })}
            </nav>
          )}

          <div className="privacyBadge">
            <span className="privacyPulse" />
            <span>
              <strong>Private by default</strong>
              <small>Local memory · governed actions</small>
            </span>
          </div>
        </div>

        <div
          className="sidebarResizeHandle"
          role="separator"
          aria-label="Resize sidebar"
          aria-orientation="vertical"
          aria-valuemin={MIN_SIDEBAR_WIDTH}
          aria-valuemax={MAX_SIDEBAR_WIDTH}
          aria-valuenow={sidebarWidth}
          aria-hidden={mobileLayout || desktopPanelCollapsed ? true : undefined}
          tabIndex={!mobileLayout && !desktopPanelCollapsed ? 0 : -1}
          onDoubleClick={() => setAndRememberSidebarWidth(DEFAULT_SIDEBAR_WIDTH)}
          onKeyDown={handleResizeKeyDown}
          onPointerDown={handleResizeStart}
          onPointerMove={handleResizeMove}
          onPointerUp={handleResizeEnd}
          onPointerCancel={handleResizeEnd}
          onLostPointerCapture={handleResizeEnd}
          title="Drag to resize · Double-click to reset"
        ><span /></div>
      </aside>

      <main ref={mainRef} className="appMain">{children}</main>
    </div>
  );
}
