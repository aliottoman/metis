"use client";

// The composer: a box to type in, Send, and one "+" menu for everything else
// (customer or project scope, files, sources, dictation). Stop sits beside
// Send while a run is live, when Send queues instead.

import { ArrowUp, Mic, Paperclip, Plus, Square, X } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";

import { CommandPicker, type PickerOption } from "@/components/command-picker";
import type { Chat } from "@/hooks/use-chat";
import type { useDictation } from "@/hooks/use-dictation";
import { attachmentBadge, CHAT_ATTACHMENT_ACCEPT } from "@/lib/attachments";
import type { KnowledgeScope } from "@/lib/types";

type Dictation = ReturnType<typeof useDictation>;
type Picker = "customer" | "project" | null;

const SLASH = /(^|\s)\/([a-zA-Z]*)$/;
const SOURCES: Array<{ value: KnowledgeScope; label: string; hint: string }> = [
  { value: "auto", label: "Auto", hint: "Everything relevant, Notion included" },
  { value: "notion", label: "Notion", hint: "Only synced Notion pages" },
  { value: "web", label: "Web", hint: "Search the web and cite it" },
];

export function Composer({ chat, dictation, voiceOpen }: { chat: Chat; dictation: Dictation; voiceOpen: boolean }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [picker, setPicker] = useState<Picker>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const caretRef = useRef<number | null>(null);
  const textarea = chat.composerRef;

  // The box grows with the draft and shrinks back; measuring from zero is
  // what lets it shrink.
  useEffect(() => {
    const box = textarea.current;
    if (!box) return;
    const frame = requestAnimationFrame(() => {
      box.style.height = "0";
      box.style.height = box.value ? `${Math.min(box.scrollHeight, 260)}px` : "";
    });
    return () => cancelAnimationFrame(frame);
  }, [chat.draft, textarea]);

  // After a programmatic edit (list continuation) put the caret back.
  useLayoutEffect(() => {
    const caret = caretRef.current;
    if (caret == null) return;
    caretRef.current = null;
    textarea.current?.setSelectionRange(caret, caret);
  }, [chat.draft, textarea]);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const key = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", key);
    };
  }, [menuOpen]);

  const customerOptions = useMemo<PickerOption[]>(() =>
    chat.customers.filter((customer) => customer.status === "active").map((customer, index) => ({
      id: customer.id,
      label: customer.name,
      meta: [customer.industry, customer.region].filter(Boolean).join(" · ") || "Customer account",
      badge: customer.wins > 0 ? `🏆 ${customer.wins}` : customer.open_actions > 0 ? `${customer.open_actions} open` : undefined,
      keywords: customer.aliases,
      group: index < 5 ? "Recent" : "All accounts",
    })), [chat.customers]);
  const projectOptions = useMemo<PickerOption[]>(() =>
    chat.projects.map((project) => ({
      id: project.id,
      label: project.name,
      glyph: project.initialized ? "◆" : "◇",
      meta: `${project.framework ? `${project.framework} · ` : ""}${project.initialized ? `${project.fileCount} files mapped` : "Project map not created yet"}`,
      keywords: project.framework ? [project.framework] : undefined,
      badge: chat.selectedProjectId === project.id ? "Active" : undefined,
    })), [chat.projects, chat.selectedProjectId]);

  const onDraftChange = (value: string) => {
    chat.setDraft(value);
    // A "/" at the start or after a space opens the menu, like "+".
    if (SLASH.test(value) && !chat.runActive) setMenuOpen(true);
  };

  const pick = (which: Exclude<Picker, null> | "attach" | "dictate") => {
    setMenuOpen(false);
    chat.setDraft((current) => current.replace(SLASH, "$1"));
    if (which === "attach") fileInput.current?.click();
    else if (which === "dictate") dictation.toggle();
    else setPicker(which);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    // Enter continues a markdown list; an empty item leaves it; anything else sends.
    const box = textarea.current;
    if (box && box.selectionStart === box.selectionEnd) {
      const value = box.value;
      const caret = box.selectionStart;
      const lineStart = value.lastIndexOf("\n", caret - 1) + 1;
      const lineEnd = value.indexOf("\n", caret);
      const line = value.slice(lineStart, lineEnd === -1 ? value.length : lineEnd);
      const marker = line.match(/^(\s*)([-*+]|\d+[.)])(\s+)(.*)$/);
      if (marker) {
        event.preventDefault();
        const [, indent, bullet, gap, content] = marker;
        if (content.trim() === "") {
          chat.setDraft(value.slice(0, lineStart) + value.slice(lineStart + indent.length + bullet.length + gap.length));
          caretRef.current = lineStart;
          return;
        }
        const next = /^\d/.test(bullet) ? `${Number.parseInt(bullet, 10) + 1}${bullet.replace(/^\d+/, "")}` : bullet;
        const insertion = `\n${indent}${next} `;
        chat.setDraft(value.slice(0, caret) + insertion + value.slice(caret));
        caretRef.current = caret + insertion.length;
        return;
      }
    }
    event.preventDefault();
    void chat.submit();
  };

  const placeholder = voiceOpen ? "Type while voice mode is open…"
    : dictation.state === "recording" ? "Listening…"
      : dictation.state === "transcribing" ? "Writing down what you said…"
        : chat.runActive ? "Type the next message — it sends when this run finishes"
          : chat.knowledgeScope === "notion" ? "Ask your Notion…"
            : chat.knowledgeScope === "web" ? "Ask the web, or paste a link…"
              : "Message Metis…";
  const canSend = (chat.draft.trim() || chat.attachments.length) && !chat.sending && !chat.uploading;
  const dictating = dictation.state === "recording" || dictation.state === "transcribing";

  return (
    <form className="composer" onSubmit={(event) => { event.preventDefault(); void chat.submit(); }}>
      {chat.attachments.length ? (
        <div className="composer-tray">
          {chat.attachments.map((attachment) => (
            <span className="composer-file" key={attachment.id}>
              <b>{attachmentBadge(attachment)}</b>
              <span>{attachment.name}</span>
              {!attachment.media_type?.startsWith("image/") ? (
                <button type="button" className="ui-btn is-quiet is-sm" disabled={chat.knowledgeAdds[attachment.id] === "adding" || chat.knowledgeAdds[attachment.id] === "added"} title="Keep this document in your knowledge base" onClick={() => void chat.addToKnowledge(attachment)}>
                  {chat.knowledgeAdds[attachment.id] === "added" ? "In knowledge" : chat.knowledgeAdds[attachment.id] === "adding" ? "Adding…" : "+ Knowledge"}
                </button>
              ) : null}
              <button type="button" className="composer-x" aria-label={`Remove ${attachment.name}`} onClick={() => chat.removeAttachment(attachment.id)}><X size={12} /></button>
            </span>
          ))}
        </div>
      ) : null}

      {chat.queued ? (
        <div className="composer-queued">
          <span className="ui-dot is-waiting" aria-hidden="true" />
          <p><strong>Queued</strong> {chat.queued.content}</p>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={chat.unqueue}>Undo</button>
        </div>
      ) : null}

      <textarea
        ref={textarea}
        rows={1}
        value={chat.draft}
        onChange={(event) => onDraftChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        aria-label="Message Metis"
        disabled={chat.sending}
      />

      <div className="composer-bar">
        <div className="composer-anchor" ref={menuRef}>
          <button type="button" className="ui-btn is-quiet is-sm composer-plus" aria-expanded={menuOpen} aria-haspopup="menu" onClick={() => setMenuOpen((value) => !value)} title="Add context, files or sources ( / )">
            <Plus size={16} aria-hidden="true" />
          </button>
          {menuOpen ? (
            <div className="composer-menu" role="menu" aria-label="Add to this message">
              <button type="button" role="menuitem" onClick={() => pick("customer")} disabled={chat.runActive}>
                <span>Customer</span><small>{chat.selectedCustomer ? chat.selectedCustomer.name : `${customerOptions.length} accounts`}</small>
              </button>
              <button type="button" role="menuitem" onClick={() => pick("project")} disabled={chat.runActive}>
                <span>Project</span><small>{chat.selectedProject ? chat.selectedProject.name : `${chat.projects.length} in the catalog`}</small>
              </button>
              <button type="button" role="menuitem" onClick={() => pick("attach")} disabled={chat.uploading}>
                <span>Attach files</span><small>Images, documents, code</small>
              </button>
              {dictation.state !== "unsupported" && !voiceOpen ? (
                <button type="button" role="menuitem" onClick={() => pick("dictate")}>
                  <span>Dictate</span><small>{dictation.provider === "elevenlabs" ? "ElevenLabs Scribe" : "Cohere Transcribe"}</small>
                </button>
              ) : null}
              <div className="composer-sources" role="group" aria-label="Answer sources">
                <span>Sources</span>
                {SOURCES.map((source) => (
                  <button key={source.value} type="button" className={chat.knowledgeScope === source.value ? "is-on" : ""} aria-pressed={chat.knowledgeScope === source.value} disabled={chat.runActive} title={source.hint} onClick={() => chat.setKnowledgeScope(source.value)}>
                    {source.label}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          {picker === "customer" ? (
            <CommandPicker
              label="Customer account scope"
              placeholder={`Search ${customerOptions.length} accounts…`}
              options={customerOptions}
              value={chat.selectedCustomerId}
              clearOption={{ id: "", label: "No customer", meta: "Ordinary chat routing" }}
              emptyMessage="No account matches that."
              onSelect={(id) => { chat.setSelectedCustomerId(id || null); setPicker(null); }}
              onDismiss={() => setPicker(null)}
              footer={<span>Only this account&rsquo;s reviewed facts and pinned notes enter the model context.</span>}
            />
          ) : null}
          {picker === "project" ? (
            <CommandPicker
              label="Project workspace"
              placeholder={`Search ${chat.projects.length} project${chat.projects.length === 1 ? "" : "s"}…`}
              options={projectOptions}
              value={chat.selectedProjectId}
              clearOption={{ id: "", label: "No project", meta: "Return to ordinary chat routing." }}
              emptyMessage={chat.projects.length ? "No project matches that." : "No projects in the catalog — type a name to create one."}
              busy={chat.projectOpening}
              onSelect={(id) => { void chat.chooseProject(id || null).then(() => setPicker(null)); }}
              onCreate={(name) => void chat.createProject(name).then(() => setPicker(null))}
              createMeta="New empty folder in your projects directory"
              onDismiss={() => setPicker(null)}
              footer={<span>{chat.projectOpening ? "Opening…" : "Writes always pause for approval"}</span>}
            />
          ) : null}
        </div>

        {chat.selectedCustomer ? (
          <span className="ui-chip is-accent composer-chip">
            {chat.selectedCustomer.name}
            <button type="button" className="ui-btn is-quiet is-sm" disabled={chat.trackerBusy || chat.runActive} title="Drop this account's activity-tracker update into the thread" onClick={() => void chat.trackerUpdate()}>
              {chat.trackerBusy ? "Building…" : "Tracker"}
            </button>
            <button type="button" className="composer-x" aria-label={`Remove ${chat.selectedCustomer.name}`} onClick={() => chat.setSelectedCustomerId(null)}><X size={12} /></button>
          </span>
        ) : null}
        {chat.selectedProject ? (
          <span className="ui-chip is-outline composer-chip">
            {chat.projectOpening ? "Mapping…" : chat.selectedProject.name}
            <button type="button" className="composer-x" aria-label={`Remove ${chat.selectedProject.name}`} disabled={chat.projectOpening} onClick={() => void chat.chooseProject(null)}><X size={12} /></button>
          </span>
        ) : null}
        {chat.knowledgeScope !== "auto" ? <span className="ui-chip composer-chip">{chat.knowledgeScope === "notion" ? "Notion only" : "Web"}</span> : null}

        <span className="composer-spacer" />

        {dictating ? (
          <button type="button" className="ui-btn is-sm" onClick={dictation.cancel}>Discard</button>
        ) : null}
        {dictation.state !== "unsupported" && !voiceOpen ? (
          <button
            type="button"
            className={`ui-btn is-quiet is-sm composer-mic is-${dictation.state}`}
            onClick={dictation.toggle}
            disabled={chat.sending || dictation.state === "transcribing"}
            aria-pressed={dictation.state === "recording"}
            aria-label={dictation.state === "recording" ? "Stop and transcribe" : "Dictate"}
            title={dictation.state === "recording" ? "Stop and transcribe" : "Dictate"}
            style={{ "--mic-level": dictation.level.toFixed(3) } as CSSProperties}
          >
            <Mic size={16} aria-hidden="true" />
          </button>
        ) : null}
        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => fileInput.current?.click()} disabled={chat.uploading} aria-label="Attach files" title="Attach files">
          <Paperclip size={16} aria-hidden="true" />
        </button>
        <input ref={fileInput} type="file" multiple hidden accept={CHAT_ATTACHMENT_ACCEPT} onChange={(event) => event.target.files && void chat.addFiles(event.target.files)} />
        {chat.runActive ? (
          <button type="button" className="ui-btn is-sm is-danger" onClick={() => void chat.stop()} aria-label="Stop the run"><Square size={12} aria-hidden="true" /> Stop</button>
        ) : null}
        <button type="submit" className="ui-btn is-primary composer-send" disabled={!canSend} aria-label={chat.runActive ? "Queue this message" : "Send"} title={chat.runActive ? "Queue for when the run finishes" : "Send · Enter"}>
          {chat.runActive ? <Plus size={16} aria-hidden="true" /> : <ArrowUp size={16} aria-hidden="true" />}
        </button>
      </div>
    </form>
  );
}
