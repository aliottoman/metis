"use client";

// The conversation list, beside the chat rather than in the navigation.
// Search sees everything; the resting view shows a compact recent set and
// offers the rest on request. Every row carries its run state, so a chat
// still working or waiting for you says so from here.

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { PanelLeftClose, PanelLeftOpen, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { FOCUS_SEARCH_EVENT } from "@/components/app-shell";
import { Skeleton } from "@/components/ui/skeleton";
import { deleteConversation, listConversations } from "@/lib/api";
import {
  CONVERSATIONS_CHANGED_EVENT,
  forgetConversation,
  readRecentConversations,
} from "@/lib/recent-conversations";
import {
  RUN_INDICATORS_CHANGED_EVENT,
  acknowledgeConversationRun,
  readRunIndicators,
  type ConversationRunIndicator,
} from "@/lib/run-indicators";
import type { ConversationSummary } from "@/lib/types";

const COMPACT_LIMIT = 14;

const RUN_WORDS: Record<string, string> = {
  working: "Working",
  attention: "Needs you",
  failed: "Interrupted",
  cancelled: "Stopped",
  done: "Done",
};

function groupLabel(timestamp?: string): string {
  if (!timestamp) return "Earlier";
  const date = new Date(timestamp);
  const now = new Date();
  const delta = now.getTime() - date.getTime();
  if (delta < 86_400_000 && date.getDate() === now.getDate()) return "Today";
  if (delta < 7 * 86_400_000) return "This week";
  return "Earlier";
}

export function ConversationList() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const active = searchParams.get("conversation");
  const [open, setOpen] = useState(true);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [indicators, setIndicators] = useState<Record<string, ConversationRunIndicator>>({});
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      setOpen(window.localStorage.getItem("metis.historyOpen") !== "0" && window.innerWidth > 720);
    } catch {
      setOpen(true);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    const load = () =>
      listConversations()
        .then((items) => mounted && setConversations(items.length ? items : readRecentConversations()))
        .catch(() => mounted && setConversations(readRecentConversations()))
        .finally(() => mounted && setLoaded(true));
    void load();
    const onChanged = () => void load();
    window.addEventListener(CONVERSATIONS_CHANGED_EVENT, onChanged);
    return () => {
      mounted = false;
      window.removeEventListener(CONVERSATIONS_CHANGED_EVENT, onChanged);
    };
  }, []);

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

  // ⌘K from anywhere lands here: the shell either fires the event on this
  // page or navigates with `focus=search` so the list mounts and focuses.
  useEffect(() => {
    const focus = () => {
      setOpen(true);
      window.setTimeout(() => searchRef.current?.focus(), 0);
    };
    window.addEventListener(FOCUS_SEARCH_EVENT, focus);
    if (searchParams.get("focus") === "search") focus();
    return () => window.removeEventListener(FOCUS_SEARCH_EVENT, focus);
  }, [searchParams]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    try {
      window.localStorage.setItem("metis.historyOpen", next ? "1" : "0");
    } catch {
      // The choice still holds for this visit.
    }
  };

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return conversations;
    return conversations.filter((item) => `${item.title} ${item.last_message ?? ""}`.toLowerCase().includes(needle));
  }, [conversations, query]);
  const visible = query.trim() || expanded ? filtered : filtered.slice(0, COMPACT_LIMIT);
  const grouped = useMemo(
    () =>
      visible.reduce<Record<string, ConversationSummary[]>>((result, item) => {
        const label = groupLabel(item.updated_at ?? item.created_at);
        result[label] = [...(result[label] ?? []), item];
        return result;
      }, {}),
    [visible],
  );

  const remove = async (conversation: ConversationSummary) => {
    if (deleting || !window.confirm(`Delete “${conversation.title}”? This removes the chat and its messages.`)) return;
    setDeleting(conversation.id);
    try {
      await deleteConversation(conversation.id);
      forgetConversation(conversation.id);
      setConversations((current) => current.filter((item) => item.id !== conversation.id));
      if (active === conversation.id) router.push("/");
    } catch {
      window.alert("This conversation could not be deleted. Try again.");
    } finally {
      setDeleting(null);
    }
  };

  return (
    <aside className={`chatHistory${open ? "" : " is-closed"}`} aria-label="Conversations">
      <button type="button" className="ui-btn is-quiet is-sm chatHistoryOpen" onClick={toggle} aria-label="Show conversations" title="Show conversations">
        <PanelLeftOpen size={16} aria-hidden="true" />
      </button>

      <header className="chatHistoryHead">
        <strong>Conversations</strong>
        <small>{conversations.length}</small>
        <button type="button" className="ui-btn is-quiet is-sm" onClick={toggle} aria-label="Hide conversations" title="Hide conversations">
          <PanelLeftClose size={16} aria-hidden="true" />
        </button>
      </header>

      <label className="chatHistorySearch">
        <Search size={14} aria-hidden="true" />
        <input
          ref={searchRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search conversations"
          aria-label="Search conversations"
        />
        {query ? (
          <button type="button" className="ui-btn is-quiet is-sm" aria-label="Clear search" onClick={() => { setQuery(""); searchRef.current?.focus(); }}>×</button>
        ) : (
          <kbd aria-hidden="true">⌘K</kbd>
        )}
      </label>

      <div className="chatHistoryList">
        {Object.entries(grouped).map(([label, items]) => (
          <section key={label} className="ui-stagger">
            <h2>{label}</h2>
            {items.map((conversation) => {
              const indicator = indicators[conversation.id];
              const current = pathname === "/" && active === conversation.id;
              const word = indicator && (indicator.state !== "done" || indicator.unread) ? RUN_WORDS[indicator.state] : "";
              return (
                <div
                  key={conversation.id}
                  className={`chatHistoryItem run-${indicator?.state ?? "idle"}${indicator?.unread ? " has-unread" : ""}`}
                >
                  <Link
                    href={`/?conversation=${encodeURIComponent(conversation.id)}`}
                    onClick={() => acknowledgeConversationRun(conversation.id)}
                    className={current ? "is-active" : ""}
                    aria-current={current ? "page" : undefined}
                    title={conversation.title}
                  >
                    <span>{conversation.title}</span>
                    {word ? <small><i aria-hidden="true">●</i>{word}</small> : null}
                  </Link>
                  <button
                    type="button"
                    className="chatHistoryDelete"
                    aria-label={`Delete ${conversation.title}`}
                    disabled={deleting === conversation.id}
                    onClick={() => void remove(conversation)}
                  >
                    ×
                  </button>
                </div>
              );
            })}
          </section>
        ))}
        {!query.trim() && filtered.length > COMPACT_LIMIT ? (
          <button type="button" className="ui-btn is-quiet is-sm chatHistoryMore" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>
            {expanded ? "Show recent only" : `Show all ${filtered.length}`}
          </button>
        ) : null}
        {!loaded && !filtered.length ? <Skeleton rows={6} height={30} /> : null}
        {loaded && !filtered.length ? (
          <p className="chatHistoryEmpty">{query ? "No matching conversations." : "Your conversations will appear here."}</p>
        ) : null}
      </div>
    </aside>
  );
}
