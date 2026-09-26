"use client";

// A new conversation opens on work, not on a wordmark: what is waiting on
// you as things to say, and where you were last.

import { FileText, Hammer, ListChecks } from "lucide-react";
import { useEffect, useState } from "react";

import type { Chat } from "@/hooks/use-chat";
import { getAttention } from "@/lib/api";
import type { AttentionItem } from "@/lib/types";

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

  useEffect(() => {
    let mounted = true;
    void getAttention(3).then((feed) => mounted && setWaiting(feed.top)).catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  const start = (text: string) => {
    chat.setDraft(text);
    chat.focusComposer();
  };

  return (
    <div className="welcome">
      <div className="welcome-lead">
        {chat.selectedCustomer ? <span className="ui-eyebrow">Working with {chat.selectedCustomer.name}</span> : null}
        <h1>What can I help with?</h1>
        <p className="welcome-intro">Ask a question, add a file, or start making something.</p>
      </div>
      <div className="welcome-starters" aria-label="Start a conversation">
        <button type="button" className="welcome-starter" onClick={() => start("Help me analyze a document. Summarize the key points, highlight anything that needs attention, and suggest next steps.")}>
          <FileText size={17} aria-hidden="true" />
          <span>Understand a document</span>
        </button>
        <button type="button" className="welcome-starter" onClick={() => start("Help me plan the next step. Ask what I want to achieve, then turn it into a clear, practical plan.")}>
          <ListChecks size={17} aria-hidden="true" />
          <span>Plan a next step</span>
        </button>
        <button type="button" className="welcome-starter" onClick={() => start("Help me build something. Start by understanding the idea, who it is for, and what a useful first version should do.")}>
          <Hammer size={17} aria-hidden="true" />
          <span>Build something</span>
        </button>
      </div>
      {waiting.length ? (
        <details className="welcome-attention">
          <summary><span className="ui-dot is-waiting" aria-hidden="true" />{waiting.length} {waiting.length === 1 ? "thing needs" : "things need"} your attention</summary>
          <div className="welcome-attention-list">
            {waiting.map((item) => (
              <button key={item.key} type="button" className="welcome-row" onClick={() => start(promptFor(item))}>
                <span className="welcome-title">{item.title}</span>
                <small>{item.kind_label}{item.overdue ? " · overdue" : ""}</small>
              </button>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}
