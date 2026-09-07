"use client";

// A new conversation opens on work, not on a wordmark: what is waiting on
// you as things to say, and where you were last.

import Link from "next/link";
import { useEffect, useState } from "react";

import type { Chat } from "@/hooks/use-chat";
import { getAttention, listConversations } from "@/lib/api";
import type { AttentionItem, ConversationSummary } from "@/lib/types";

/** A queue item phrased as something you would say to Metis. */
function promptFor(item: AttentionItem): string {
  switch (item.kind) {
    case "customer_action":
      return `Help me move this forward: ${item.title}`;
    case "customer_note":
      return `Analyze this note and propose what to record: ${item.title}`;
    case "memory":
      return `Is this worth remembering? ${item.title}`;
    case "tool_proposal":
      return `Walk me through this tool proposal: ${item.title}`;
    default:
      return item.title;
  }
}

export function Welcome({ chat }: { chat: Chat }) {
  const [waiting, setWaiting] = useState<AttentionItem[]>([]);
  const [recent, setRecent] = useState<ConversationSummary[]>([]);

  useEffect(() => {
    let mounted = true;
    void getAttention(3).then((feed) => mounted && setWaiting(feed.top)).catch(() => undefined);
    void listConversations().then((items) => mounted && setRecent(items.slice(0, 3))).catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  const start = (text: string) => {
    chat.setDraft(text);
    chat.focusComposer();
  };

  return (
    <div className="welcome">
      <span className="ui-eyebrow">{chat.selectedCustomer ? `Scoped to ${chat.selectedCustomer.name}` : "New conversation"}</span>
      <h1>What are we working on?</h1>
      {waiting.length ? (
        <section className="welcome-group" aria-label="Waiting on you">
          <h2>Waiting on you</h2>
          {waiting.map((item) => (
            <button key={item.key} type="button" className="welcome-row" onClick={() => start(promptFor(item))}>
              <span className={`ui-dot ${item.overdue ? "is-needs-review" : "is-waiting"}`} aria-hidden="true" />
              <span className="welcome-title">{item.title}</span>
              <small>{item.kind_label}{item.overdue ? " · overdue" : ""}</small>
            </button>
          ))}
        </section>
      ) : null}
      {recent.length ? (
        <section className="welcome-group" aria-label="Recent conversations">
          <h2>Pick up where you left off</h2>
          {recent.map((conversation) => (
            <Link key={conversation.id} className="welcome-row" href={`/?conversation=${encodeURIComponent(conversation.id)}`}>
              <span className="welcome-title">{conversation.title}</span>
              <small>{conversation.last_message ?? ""}</small>
            </Link>
          ))}
        </section>
      ) : null}
      {!waiting.length && !recent.length ? (
        <p className="welcome-empty">Ask about an account, drop in a document, or open a project. Everything stays on this machine.</p>
      ) : null}
    </div>
  );
}
