"use client";

// A new conversation opens on work, not on a wordmark: what is waiting on
// you as things to say, and where you were last.

import Link from "next/link";
import { FileText, Hammer, ListChecks } from "lucide-react";
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
      <p className="welcome-intro">Bring a question, a document, or an idea. We&rsquo;ll take it from here.</p>
      {!waiting.length ? (
        <div className="welcome-starters" aria-label="Start a conversation">
          <button type="button" className="welcome-starter" onClick={() => start("Help me analyze a document. Summarize the key points, highlight anything that needs attention, and suggest next steps.")}>
            <FileText size={20} aria-hidden="true" />
            <strong>Make sense of a document</strong>
            <span>Find the key points and what matters.</span>
          </button>
          <button type="button" className="welcome-starter" onClick={() => start("Help me plan the next step. Ask what I want to achieve, then turn it into a clear, practical plan.")}>
            <ListChecks size={20} aria-hidden="true" />
            <strong>Plan the next step</strong>
            <span>Turn an open question into a clear plan.</span>
          </button>
          <button type="button" className="welcome-starter" onClick={() => start("Help me build something. Start by understanding the idea, who it is for, and what a useful first version should do.")}>
            <Hammer size={20} aria-hidden="true" />
            <strong>Bring an idea to life</strong>
            <span>Shape a project and start making progress.</span>
          </button>
        </div>
      ) : null}
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
    </div>
  );
}
