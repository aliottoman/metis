"use client";

// The shell: one rail, one pane. The rail is the navigation — labels visible
// at rest, three groups by the job you are doing, Library folded away until
// you want it. Conversations live inside Chat, where they belong. On a phone
// the rail is a drawer; on a narrow desktop it folds to icons.

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  AudioLines,
  BookOpen,
  Bot,
  Brain,
  ChevronDown,
  Gauge,
  Menu,
  MessageCircleQuestion,
  MessageSquare,
  MessageSquareQuote,
  Package,
  PanelLeftClose,
  Plus,
  Settings,
  Sun,
  Users,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { MetisCompanion } from "@/components/metis-companion";
import { StatusDot } from "@/components/ui/status";
import { getRunRecord, listConversations } from "@/lib/api";
import { freshToken } from "@/lib/token";
import {
  RUN_INDICATORS_CHANGED_EVENT,
  acknowledgeConversationRun,
  readRunIndicators,
  updateConversationRun,
  type ConversationRunIndicator,
  type ConversationRunState,
} from "@/lib/run-indicators";

interface Destination {
  href: string;
  label: string;
  icon: LucideIcon;
}

interface Group {
  id: "work" | "voice" | "library";
  label: string;
  collapsible: boolean;
  items: Destination[];
}

const GROUPS: Group[] = [
  {
    id: "work",
    label: "Work",
    collapsible: false,
    items: [
      { href: "/today", label: "Today", icon: Sun },
      { href: "/", label: "Chat", icon: MessageSquare },
      { href: "/customers", label: "Customers", icon: Users },
    ],
  },
  {
    id: "voice",
    label: "Voice",
    collapsible: false,
    items: [
      { href: "/meetings", label: "Meetings", icon: AudioLines },
      { href: "/interviews", label: "Interviews", icon: MessageCircleQuestion },
      { href: "/agents", label: "Agents", icon: Bot },
    ],
  },
  {
    id: "library",
    label: "Library",
    collapsible: true,
    items: [
      { href: "/assets", label: "Assets", icon: Package },
      { href: "/knowledge", label: "Knowledge", icon: BookOpen },
      { href: "/answers", label: "Answers", icon: MessageSquareQuote },
      { href: "/memory", label: "Memory", icon: Brain },
      { href: "/tools", label: "Tool Workshop", icon: Wrench },
      { href: "/sizing", label: "Sizing", icon: Gauge },
    ],
  },
];

const SETTINGS: Destination = { href: "/settings", label: "Settings", icon: Settings };

export const FOCUS_SEARCH_EVENT = "metis:focus-search";

function isActive(href: string, pathname: string): boolean {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

function readFlag(key: string, fallback: boolean): boolean {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : raw === "1";
  } catch {
    return fallback;
  }
}

function writeFlag(key: string, value: boolean): void {
  try {
    window.localStorage.setItem(key, value ? "1" : "0");
  } catch {
    // A preference that cannot be saved still holds for this visit.
  }
}

function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const media = window.matchMedia(query);
    const sync = () => setMatches(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, [query]);
  return matches;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeConversation = searchParams.get("conversation");
  const narrow = useMedia("(max-width: 720px)");
  const medium = useMedia("(max-width: 1080px)");

  const [collapsedChoice, setCollapsedChoice] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [connected, setConnected] = useState(true);
  const [indicators, setIndicators] = useState<Record<string, ConversationRunIndicator>>({});
  const [pill, setPill] = useState<{ top: number; visible: boolean }>({ top: 0, visible: false });
  const navRef = useRef<HTMLElement>(null);
  const railRef = useRef<HTMLElement>(null);
  const menuRef = useRef<HTMLButtonElement>(null);

  // Folded by choice on a wide screen, by necessity on a medium one, and
  // never on a phone, where the rail is a drawer instead.
  const collapsed = !narrow && (medium || collapsedChoice);

  useEffect(() => {
    setCollapsedChoice(readFlag("metis.railCollapsed", false));
    setLibraryOpen(readFlag("metis.libraryOpen", false));
  }, []);

  // The active pill slides to whichever link is current. Measured after
  // layout so it follows the rail folding and the Library opening too.
  const placePill = useCallback(() => {
    const nav = navRef.current;
    const active = nav?.querySelector<HTMLElement>(".rail-link.is-active");
    if (!nav || !active) {
      setPill((current) => (current.visible ? { ...current, visible: false } : current));
      return;
    }
    setPill({ top: active.offsetTop, visible: true });
  }, []);
  useLayoutEffect(placePill, [placePill, pathname, collapsed, libraryOpen]);
  useEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    const observer = new ResizeObserver(() => placePill());
    observer.observe(nav);
    // The fold and the Library animate for a moment; settle the pill after.
    const settle = window.setTimeout(placePill, 360);
    return () => {
      observer.disconnect();
      window.clearTimeout(settle);
    };
  }, [placePill, collapsed, libraryOpen]);

  // A page inside the Library opens the group so the current page is visible.
  useEffect(() => {
    if (GROUPS[2].items.some((item) => isActive(item.href, pathname))) setLibraryOpen(true);
  }, [pathname]);

  useEffect(() => setDrawerOpen(false), [pathname, activeConversation]);

  // The drawer is a modal surface on a phone: focus goes in, Escape brings it out.
  useEffect(() => {
    const rail = railRef.current;
    if (!rail) return;
    rail.inert = narrow && !drawerOpen;
    if (!narrow || !drawerOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    window.requestAnimationFrame(() => rail.querySelector<HTMLElement>("a, button")?.focus());
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      (previous ?? menuRef.current)?.focus();
    };
  }, [drawerOpen, narrow]);

  // Runs keep going when Chat is not the visible page. The shell stays mounted
  // across navigation, so it owns the background check and turns completion
  // into the unread count on the Chat entry.
  useEffect(() => {
    const sync = () => setIndicators(readRunIndicators());
    sync();
    window.addEventListener(RUN_INDICATORS_CHANGED_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(RUN_INDICATORS_CHANGED_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  useEffect(() => {
    let stopped = false;
    const poll = async () => {
      const pending = Object.values(readRunIndicators()).filter(
        (item) => item.state === "working" || item.state === "attention",
      );
      await Promise.all(
        pending.map(async (item) => {
          try {
            const run = await getRunRecord(item.runId);
            if (stopped) return;
            const state: ConversationRunState =
              run.status === "completed" ? "done"
                : run.status === "failed" ? "failed"
                  : run.status === "cancelled" ? "cancelled"
                    : run.status === "awaiting_approval" || run.status === "awaiting_input" ? "attention"
                      : "working";
            const visibleHere =
              pathname === "/" && activeConversation === item.conversationId && document.visibilityState === "visible";
            const unread = state === "working" ? false : !visibleHere;
            if (state !== item.state || unread !== item.unread) updateConversationRun(item.runId, state, unread);
          } catch {
            // A badge is advisory; a brief API blip must not flash failure.
          }
        }),
      );
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [activeConversation, pathname]);
  useEffect(() => {
    if (pathname === "/" && activeConversation) acknowledgeConversationRun(activeConversation);
  }, [activeConversation, pathname]);

  // Connection status for the rail foot: one cheap call, then every half minute.
  useEffect(() => {
    let stopped = false;
    const check = () =>
      listConversations()
        .then(() => !stopped && setConnected(true))
        .catch(() => !stopped && setConnected(false));
    void check();
    const timer = window.setInterval(() => void check(), 30_000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey;
      // ⌘⇧K belongs to the page (Customers searches records with it).
      if (meta && !event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        if (pathname === "/") window.dispatchEvent(new Event(FOCUS_SEARCH_EVENT));
        else router.push("/?focus=search");
      } else if (meta && event.key.toLowerCase() === "n") {
        event.preventDefault();
        router.push(`/?new=${freshToken()}`);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pathname, router]);

  const unread = useMemo(() => Object.values(indicators).filter((item) => item.unread).length, [indicators]);

  const toggleCollapsed = () => {
    const next = !collapsedChoice;
    setCollapsedChoice(next);
    writeFlag("metis.railCollapsed", next);
  };
  const toggleLibrary = () => {
    const next = !libraryOpen;
    setLibraryOpen(next);
    writeFlag("metis.libraryOpen", next);
  };

  const link = (item: Destination) => {
    const active = isActive(item.href, pathname);
    const Icon = item.icon;
    return (
      <Link
        key={item.href}
        href={item.href}
        className={`rail-link${active ? " is-active" : ""}`}
        aria-current={active ? "page" : undefined}
        title={collapsed ? item.label : undefined}
      >
        <Icon size={18} strokeWidth={1.6} aria-hidden="true" />
        <span>{item.label}</span>
        {item.href === "/" && unread ? (
          <b className="rail-badge" aria-label={`${unread} finished runs to read`}>{unread}</b>
        ) : null}
      </Link>
    );
  };

  return (
    <div className={`shell${collapsed ? " is-collapsed" : ""}${drawerOpen ? " is-drawer-open" : ""}`}>
      <button
        ref={menuRef}
        className="shell-menu"
        type="button"
        aria-label={drawerOpen ? "Close navigation" : "Open navigation"}
        aria-expanded={drawerOpen}
        aria-controls="metis-rail"
        onClick={() => setDrawerOpen((current) => !current)}
      >
        <Menu size={18} aria-hidden="true" />
      </button>
      <button className="shell-scrim" type="button" tabIndex={-1} aria-hidden="true" onClick={() => setDrawerOpen(false)} />

      <aside ref={railRef} id="metis-rail" className="rail" aria-label="Main navigation">
        <Link href="/today" className="rail-brand" aria-label="Metis">
          <span className="brandMark" aria-hidden="true">
            <MetisCompanion size={30} energy="calm" />
          </span>
          <strong>Metis</strong>
        </Link>

        <button
          className="rail-new"
          type="button"
          onClick={() => router.push(`/?new=${freshToken()}`)}
          title="New chat · ⌘N"
        >
          <Plus size={16} strokeWidth={2} aria-hidden="true" />
          <span>New chat</span>
          <kbd aria-hidden="true">⌘N</kbd>
        </button>

        <nav ref={navRef} className="rail-nav" aria-label="Pages">
          <span className={`rail-pill${pill.visible ? " is-visible" : ""}`} style={{ top: pill.top }} aria-hidden="true" />
          {GROUPS.map((group) =>
            group.collapsible ? (
              <div className="rail-group" key={group.id}>
                <button
                  type="button"
                  className="rail-group-toggle"
                  aria-expanded={libraryOpen}
                  aria-controls={`rail-group-${group.id}`}
                  onClick={toggleLibrary}
                  title={collapsed ? group.label : undefined}
                >
                  <span>{group.label}</span>
                  <ChevronDown size={14} aria-hidden="true" />
                </button>
                <div id={`rail-group-${group.id}`} className={`rail-collapsible${libraryOpen ? " is-open" : ""}`}>
                  <div>{group.items.map(link)}</div>
                </div>
              </div>
            ) : (
              <div className="rail-group" key={group.id}>
                <span className="rail-group-label">{group.label}</span>
                {group.items.map(link)}
              </div>
            ),
          )}
        </nav>

        <div className="rail-foot">
          {link(SETTINGS)}
          <span
            className="rail-status"
            role="status"
            title={connected ? "Metis is connected · private by default" : "Metis is offline"}
          >
            <StatusDot state={connected ? "live" : "stopped"} />
            <span>{connected ? "Private by default" : "Offline"}</span>
          </span>
          {!narrow ? (
            <button
              type="button"
              className="rail-link rail-toggle"
              onClick={toggleCollapsed}
              aria-pressed={collapsedChoice}
              title={collapsed ? "Expand the rail" : "Fold the rail"}
            >
              <PanelLeftClose size={18} strokeWidth={1.6} aria-hidden="true" />
              <span>{collapsed ? "Expand" : "Fold rail"}</span>
            </button>
          ) : null}
        </div>
      </aside>

      <main className="appMain">
        <div key={pathname} className="pane-view">
          {children}
        </div>
      </main>
    </div>
  );
}
