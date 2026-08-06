"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import { MetisMark, MetisWordmark } from "@/components/metis-mark";
import { deleteConversation, listConversations } from "@/lib/api";
import {
  CONVERSATIONS_CHANGED_EVENT,
  forgetConversation,
  readRecentConversations,
} from "@/lib/recent-conversations";
import type { ConversationSummary } from "@/lib/types";

type NavIconName =
  | "chat"
  | "customers"
  | "assets"
  | "tools"
  | "knowledge"
  | "memory"
  | "sizing"
  | "settings";

const navigation: Array<{ href: string; label: string; icon: NavIconName }> = [
  { href: "/", label: "Chat", icon: "chat" },
  { href: "/customers", label: "Customers", icon: "customers" },
  { href: "/assets", label: "Assets", icon: "assets" },
  { href: "/tools", label: "Tool Workshop", icon: "tools" },
  { href: "/knowledge", label: "Knowledge", icon: "knowledge" },
  { href: "/memory", label: "Memory", icon: "memory" },
  { href: "/sizing", label: "Sizing", icon: "sizing" },
  { href: "/settings", label: "Settings", icon: "settings" },
];

const DEFAULT_SIDEBAR_WIDTH = 254;
const MIN_SIDEBAR_WIDTH = 220;
const MAX_SIDEBAR_WIDTH = 390;

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
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [resizing, setResizing] = useState(false);
  const [animate, setAnimate] = useState(false);
  const [apiConnected, setApiConnected] = useState(true);
  const [deletingConversation, setDeletingConversation] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const resizingRef = useRef(false);
  const resizeOriginRef = useRef({ pointerX: 0, width: DEFAULT_SIDEBAR_WIDTH });
  const sidebarWidthRef = useRef(DEFAULT_SIDEBAR_WIDTH);

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

  const toggleCollapsed = () => {
    resizingRef.current = false;
    setResizing(false);
    setCollapsed((current) => {
      const next = !current;
      window.localStorage.setItem("metis.sidebarCollapsed", next ? "1" : "0");
      return next;
    });
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
        setCollapsed(false);
        window.localStorage.setItem("metis.sidebarCollapsed", "0");
        setDrawerOpen(true);
        window.setTimeout(() => searchRef.current?.focus(), 0);
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        router.push(`/?new=${crypto.randomUUID()}`);
      } else if (event.key === "Escape") {
        resizingRef.current = false;
        setResizing(false);
        setDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [router]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return conversations;
    return conversations.filter((item) =>
      `${item.title} ${item.last_message ?? ""}`.toLowerCase().includes(normalized),
    );
  }, [conversations, query]);

  const grouped = useMemo(() => {
    return filtered.reduce<Record<string, ConversationSummary[]>>((result, item) => {
      const label = groupLabel(item.updated_at ?? item.created_at);
      result[label] = [...(result[label] ?? []), item];
      return result;
    }, {});
  }, [filtered]);

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
    <div className={`appShell ${collapsed ? "isCollapsed" : ""} ${animate ? "animate" : ""} ${resizing ? "isResizing" : ""}`}>
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
            <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves={2} seed={9} result="n">
              <animate attributeName="baseFrequency" dur="26s" values="0.015;0.028;0.015" repeatCount="indefinite" />
            </feTurbulence>
            <feDisplacementMap in="SourceGraphic" in2="n" scale={30} xChannelSelector="R" yChannelSelector="G" />
          </filter>
        </defs>
      </svg>

      <button
        className="mobileMenuButton"
        type="button"
        aria-label="Open navigation"
        onClick={() => setDrawerOpen(true)}
      >
        <span />
        <span />
      </button>

      {drawerOpen ? (
        <button className="drawerScrim" aria-label="Close navigation" onClick={() => setDrawerOpen(false)} />
      ) : null}

      <aside
        className={`sidebar ${drawerOpen ? "sidebarOpen" : ""}`}
        aria-label="Main navigation"
        style={{ "--sidebar-width": `${sidebarWidth}px` } as CSSProperties}
      >
        <div className="brandRow">
          <Link href="/" className="brand" aria-label="Metis home">
            <span className="brandMark" aria-hidden="true">
              <MetisMark />
            </span>
            <span>
              <strong><MetisWordmark /></strong>
              <small>Private intelligence</small>
            </span>
          </Link>
          <span className={`connectionDot ${apiConnected ? "connected" : "disconnected"}`} title={apiConnected ? "Local API connected" : "Local API unavailable"} />
          <button
            type="button"
            className="collapseToggle"
            onClick={toggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
              <path d="M10 3.5 5.5 8 10 12.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        <button className="newChatButton" type="button" onClick={() => router.push(`/?new=${crypto.randomUUID()}`)} title="New conversation">
          <span aria-hidden="true">＋</span>
          <span className="navLabel">New conversation</span>
          <kbd>⌘ N</kbd>
        </button>

        <nav className="primaryNav" aria-label="Workspace">
          {navigation.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link key={item.href} href={item.href} className={active ? "active" : ""} title={item.label}>
                <span className="navGlyph" aria-hidden="true"><NavIcon name={item.icon} /></span>
                <span className="navLabel">{item.label}</span>
                <span className="navSignal" aria-hidden="true" />
              </Link>
            );
          })}
        </nav>

        <div className="sidebarDivider" />

        <div className="historyHeader">
          <span>Conversations</span>
          <span>{conversations.length}</span>
        </div>
        <label className="sidebarSearch">
          <span aria-hidden="true">⌕</span>
          <input
            ref={searchRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search"
            aria-label="Search conversations"
          />
          {query ? <button className="searchClear" type="button" aria-label="Clear search" onClick={() => { setQuery(""); searchRef.current?.focus(); }}>×</button> : null}
          <kbd><span>⌘</span><span>K</span></kbd>
        </label>

        <div className="conversationHistory">
          {Object.entries(grouped).map(([label, items]) => (
            <section key={label}>
              <h2>{label}</h2>
              {items.map((conversation) => (
                <div key={conversation.id} className="conversationHistoryItem">
                  <Link href={`/?conversation=${encodeURIComponent(conversation.id)}`} className={pathname === "/" && activeConversation === conversation.id ? "active" : ""} title={conversation.title}>
                    <span>{conversation.title}</span>
                  </Link>
                  <button type="button" className="conversationDeleteButton" aria-label={`Delete ${conversation.title}`} title="Delete conversation" disabled={deletingConversation === conversation.id} onClick={() => void removeConversation(conversation)}>×</button>
                </div>
              ))}
            </section>
          ))}
          {!filtered.length ? (
            <p className="historyEmpty">{query ? "No matching conversations" : "Your local conversations will appear here."}</p>
          ) : null}
        </div>

        <div className="privacyBadge">
          <span className="privacyPulse" />
          <span>
            <strong>Governed workspace</strong>
            <small>Local memory · explicit cloud choice</small>
          </span>
        </div>

        <div
          className="sidebarResizeHandle"
          role="separator"
          aria-label="Resize sidebar"
          aria-orientation="vertical"
          aria-valuemin={MIN_SIDEBAR_WIDTH}
          aria-valuemax={MAX_SIDEBAR_WIDTH}
          aria-valuenow={sidebarWidth}
          tabIndex={collapsed ? -1 : 0}
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

      <main className="appMain">{children}</main>
    </div>
  );
}
