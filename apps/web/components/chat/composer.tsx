"use client";

// The composer keeps the draft and its context together. Frequently used
// actions stay visible; the add menu holds the less common choices.

import { ArrowUp, BookOpen, Building2, ChevronDown, FolderOpen, Globe2, LoaderCircle, Mic, Paperclip, Plus, Search, Square, X } from "lucide-react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";

import { CommandPicker, type PickerOption } from "@/components/command-picker";
import type { Chat } from "@/hooks/use-chat";
import type { useDictation } from "@/hooks/use-dictation";
import { attachmentBadge, CHAT_ATTACHMENT_ACCEPT } from "@/lib/attachments";
import type { KnowledgeScope } from "@/lib/types";

type Dictation = ReturnType<typeof useDictation>;
type Picker = "customer" | "project" | null;

const SLASH = /(^|\s)\/([a-zA-Z]*)$/;
const SOURCES: Array<{ value: KnowledgeScope; label: string; hint: string }> = [
  { value: "auto", label: "Auto", hint: "Use the web and your sources when helpful" },
  { value: "notion", label: "Notion", hint: "Only synced Notion pages" },
  { value: "web", label: "Web", hint: "Search the web and cite it" },
];

export function Composer({ chat, dictation, voiceOpen }: { chat: Chat; dictation: Dictation; voiceOpen: boolean }) {
  const id = useId();
  const [menuOpen, setMenuOpen] = useState(false);
  const [picker, setPicker] = useState<Picker>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const sourceButtonRef = useRef<HTMLButtonElement>(null);
  const menuTriggerRef = useRef<HTMLButtonElement | HTMLTextAreaElement | null>(null);
  const textarea = chat.composerRef;

  // Measure before paint so each keystroke grows or shrinks the box without
  // a one-frame jump. The CSS max-height keeps long drafts scrollable.
  const resizeComposer = useCallback(() => {
    const box = textarea.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = box.value ? `${box.scrollHeight}px` : "";
    box.style.overflowY = box.scrollHeight > box.clientHeight + 1 ? "auto" : "hidden";
  }, [textarea]);

  useLayoutEffect(resizeComposer, [chat.draft, resizeComposer]);
  useEffect(() => {
    const box = textarea.current;
    const parent = box?.parentElement;
    if (!parent) return;
    let width = parent.clientWidth;
    const observer = new ResizeObserver(() => {
      const nextWidth = parent.clientWidth;
      if (nextWidth === width) return;
      width = nextWidth;
      resizeComposer();
    });
    observer.observe(parent);
    return () => observer.disconnect();
  }, [resizeComposer, textarea]);
  useEffect(() => {
    // Mobile keyboards can change the visual viewport without changing the
    // composer's width. Recalculate the textarea's scrollability at that size.
    window.addEventListener("resize", resizeComposer);
    window.visualViewport?.addEventListener("resize", resizeComposer);
    return () => {
      window.removeEventListener("resize", resizeComposer);
      window.visualViewport?.removeEventListener("resize", resizeComposer);
    };
  }, [resizeComposer]);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const key = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape" && !event.defaultPrevented) {
        event.preventDefault();
        setMenuOpen(false);
        if (menuRef.current?.contains(document.activeElement)) menuTriggerRef.current?.focus();
      }
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
    if (SLASH.test(value)) {
      menuTriggerRef.current = textarea.current;
      setMenuOpen(true);
    }
    else setMenuOpen(false);
  };

  const toggleMenu = (trigger: HTMLButtonElement | null, focusSource = false) => {
    setPicker(null);
    if (!menuOpen) {
      menuTriggerRef.current = trigger;
      setMenuOpen(true);
      requestAnimationFrame(() => menuRef.current?.querySelector<HTMLButtonElement>(focusSource
        ? '[role="menuitemradio"][aria-checked="true"]'
        : '[role="menuitem"]:not(:disabled)')?.focus());
    } else {
      setMenuOpen(false);
    }
  };

  const pick = (which: Exclude<Picker, null> | "attach" | "dictate") => {
    setMenuOpen(false);
    chat.setDraft((current) => current.replace(SLASH, "$1"));
    if (which === "attach") fileInput.current?.click();
    else if (which === "dictate") { dictation.toggle(); chat.focusComposer(); }
    else setPicker(which);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
    if (menuOpen && event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      setMenuOpen(false);
      return;
    }
    if (menuOpen && (event.key === "ArrowDown" || (event.key === "Enter" && SLASH.test(chat.draft)))) {
      event.preventDefault();
      menuRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')?.focus();
      return;
    }
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    if (canSend) void chat.submit();
  };

  const placeholder = voiceOpen ? "Type while voice mode is open…"
    : dictation.state === "recording" ? "Listening…"
      : dictation.state === "transcribing" ? "Writing down what you said…"
        : chat.runActive ? "Type the next message — it sends when this run finishes"
          : chat.knowledgeScope === "notion" ? "Ask your Notion…"
            : chat.knowledgeScope === "web" ? "Ask the web, or paste a link…"
              : "Message Metis…";
  const canSend = Boolean(chat.draft.trim() || chat.attachments.length) && !chat.sending && !chat.uploading && !chat.loadingConversation && !chat.projectOpening && !(chat.runActive && chat.queued);
  const dictating = dictation.state === "recording" || dictation.state === "transcribing";
  const selectedSource = SOURCES.find((source) => source.value === chat.knowledgeScope) ?? SOURCES[0];
  const SourceIcon = chat.knowledgeScope === "web" ? Search : chat.knowledgeScope === "notion" ? BookOpen : Globe2;

  return (
    <form className="composer" aria-label="Message composer" onSubmit={(event) => { event.preventDefault(); if (canSend) void chat.submit(); }}>
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
        <div className="composer-queued" role="status">
          <span className="ui-dot is-waiting" aria-hidden="true" />
          <p title={chat.queued.content}><strong>Queued</strong> {chat.queued.content || `${chat.queued.attachments.length} attached ${chat.queued.attachments.length === 1 ? "file" : "files"}`}</p>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={chat.unqueue}>Edit</button>
        </div>
      ) : null}

      {chat.selectedCustomer || chat.selectedProject ? (
        <div className="composer-context" aria-label="Conversation context">
          {chat.selectedCustomer ? (
            <span className="composer-context-chip">
              <Building2 size={13} aria-hidden="true" />
              <span className="composer-chip-label" title={chat.selectedCustomer.name}>{chat.selectedCustomer.name}</span>
              <button type="button" className="composer-context-action" disabled={chat.trackerBusy || chat.runActive} title="Add this account's activity tracker update to the conversation" onClick={() => void chat.trackerUpdate()}>
                {chat.trackerBusy ? "Building…" : "Tracker"}
              </button>
              <button type="button" className="composer-x" aria-label={`Remove ${chat.selectedCustomer.name}`} disabled={chat.runActive} onClick={() => chat.setSelectedCustomerId(null)}><X size={12} /></button>
            </span>
          ) : null}
          {chat.selectedProject ? (
            <span className="composer-context-chip">
              <FolderOpen size={13} aria-hidden="true" />
              <span className="composer-chip-label" title={chat.selectedProject.name}>{chat.projectOpening ? "Mapping…" : chat.selectedProject.name}</span>
              <button type="button" className="composer-x" aria-label={`Remove ${chat.selectedProject.name}`} disabled={chat.projectOpening || chat.runActive} onClick={() => void chat.chooseProject(null)}><X size={12} /></button>
            </span>
          ) : null}
        </div>
      ) : null}

      <textarea
        ref={textarea}
        rows={1}
        value={chat.draft}
        onChange={(event) => onDraftChange(event.target.value)}
        onKeyDown={onKeyDown}
        onPaste={(event) => {
          if (!event.clipboardData.files.length || chat.uploading) return;
          // Some clipboard entries carry both files and text. Keep the text
          // paste while attaching the files.
          if (!event.clipboardData.getData("text/plain") && !event.clipboardData.getData("text/html")) event.preventDefault();
          void chat.addFiles(Array.from(event.clipboardData.files));
        }}
        placeholder={placeholder}
        aria-label="Message Metis"
        aria-describedby={`${id}-hint`}
        disabled={chat.loadingConversation}
      />

      <div className="composer-bar">
        <div className="composer-left">
        <div className="composer-anchor" ref={menuRef} onBlur={(event) => {
          if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) setMenuOpen(false);
        }} onKeyDown={(event) => {
          if (!menuOpen || event.nativeEvent.isComposing) return;
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            setMenuOpen(false);
            menuTriggerRef.current?.focus();
            return;
          }
          if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const buttons = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)') ?? []);
          const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
          const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : current < 0 ? (event.key === "ArrowDown" ? 0 : buttons.length - 1)
            : (current + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
          buttons[next]?.focus();
        }}>
          <button ref={menuButtonRef} type="button" className="composer-tool composer-plus" aria-label="Add context or files" aria-expanded={menuOpen} aria-controls={menuOpen ? `${id}-menu` : undefined} aria-haspopup="menu" onClick={() => toggleMenu(menuButtonRef.current)} title="Add context or files ( / )">
            <Plus size={18} aria-hidden="true" />
          </button>
          <button ref={sourceButtonRef} type="button" className={`composer-source${chat.knowledgeScope !== "auto" ? " is-specific" : ""}`} aria-label={`Sources: ${selectedSource.label}. ${selectedSource.hint}`} aria-expanded={menuOpen} aria-controls={menuOpen ? `${id}-menu` : undefined} aria-haspopup="menu" disabled={chat.runActive} onClick={() => toggleMenu(sourceButtonRef.current, true)} title={chat.runActive ? "Sources are fixed while responding" : selectedSource.hint}>
            <SourceIcon size={15} aria-hidden="true" />
            <span>{selectedSource.label}</span>
            <ChevronDown size={13} aria-hidden="true" />
          </button>
          {menuOpen ? (
            <div className="composer-menu" id={`${id}-menu`} role="menu" aria-label="Add to this message">
              <button type="button" role="menuitem" onClick={() => pick("customer")} disabled={chat.runActive}>
                <Building2 size={16} aria-hidden="true" /><span><strong>Customer</strong><small>{chat.selectedCustomer ? chat.selectedCustomer.name : `${customerOptions.length} accounts`}</small></span>
              </button>
              <button type="button" role="menuitem" onClick={() => pick("project")} disabled={chat.runActive}>
                <FolderOpen size={16} aria-hidden="true" /><span><strong>Project</strong><small>{chat.selectedProject ? chat.selectedProject.name : `${chat.projects.length} in the catalog`}</small></span>
              </button>
              <button type="button" role="menuitem" onClick={() => pick("attach")} disabled={chat.uploading}>
                <Paperclip size={16} aria-hidden="true" /><span><strong>Attach files</strong><small>Images, documents, code</small></span>
              </button>
              {dictation.state !== "unsupported" && !voiceOpen ? (
                <button type="button" role="menuitem" onClick={() => pick("dictate")}>
                  <Mic size={16} aria-hidden="true" /><span><strong>Dictate</strong><small>{dictation.provider === "elevenlabs" ? "ElevenLabs Scribe" : "Cohere Transcribe"}</small></span>
                </button>
              ) : null}
              <div className="composer-sources" role="group" aria-label="Answer sources">
                <span>Search in</span>
                {SOURCES.map((source) => (
                  <button key={source.value} type="button" role="menuitemradio" className={chat.knowledgeScope === source.value ? "is-on" : ""} aria-checked={chat.knowledgeScope === source.value} disabled={chat.runActive} title={source.hint} onClick={() => {
                    chat.setKnowledgeScope(source.value);
                    chat.setDraft((current) => current.replace(SLASH, "$1"));
                    setMenuOpen(false);
                    chat.focusComposer();
                  }}>
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
              clearOption={{ id: "", label: "No customer", meta: "Continue without an account" }}
              emptyMessage="No account matches that."
              onSelect={(id) => { chat.setSelectedCustomerId(id || null); setPicker(null); chat.focusComposer(); }}
              onDismiss={() => setPicker(null)}
              footer={<span>Use this account&rsquo;s reviewed facts and pinned notes in the conversation.</span>}
            />
          ) : null}
          {picker === "project" ? (
            <CommandPicker
              label="Project workspace"
              placeholder={`Search ${chat.projects.length} project${chat.projects.length === 1 ? "" : "s"}…`}
              options={projectOptions}
              value={chat.selectedProjectId}
              clearOption={{ id: "", label: "No project", meta: "Continue without a project" }}
              emptyMessage={chat.projects.length ? "No project matches that." : "No projects yet. Type a name to create one."}
              busy={chat.projectOpening}
              onSelect={(id) => { void chat.chooseProject(id || null).then((opened) => { if (opened || !id) { setPicker(null); chat.focusComposer(); } }); }}
              onCreate={(name) => void chat.createProject(name).then((opened) => { if (opened) { setPicker(null); chat.focusComposer(); } })}
              createMeta="New empty folder in your projects directory"
              onDismiss={() => setPicker(null)}
              footer={<span>{chat.projectOpening ? "Opening…" : "Writes always pause for approval"}</span>}
            />
          ) : null}
        </div>
          <button type="button" className="composer-tool composer-attach" onClick={() => fileInput.current?.click()} disabled={chat.uploading} aria-label="Attach files" title="Attach files">
            <Paperclip size={17} aria-hidden="true" />
          </button>
        </div>

        <div className="composer-actions">
          {dictating ? (
            <button type="button" className="composer-discard" onClick={dictation.cancel}>Discard</button>
          ) : null}
          {dictation.state !== "unsupported" && !voiceOpen ? (
            <button
              type="button"
              className={`composer-tool composer-mic is-${dictation.state}`}
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
          {chat.runActive ? (
            <button type="button" className="composer-tool composer-stop" onClick={() => void chat.stop()} aria-label="Stop response" title="Stop response"><Square size={14} aria-hidden="true" /></button>
          ) : null}
          <button type="submit" className={`composer-send${chat.runActive ? " is-queue" : ""}${chat.sending || chat.uploading ? " is-loading" : ""}`} disabled={!canSend} aria-label={chat.runActive ? "Queue this message" : "Send message"} title={chat.queued ? "A message is already queued" : chat.runActive ? "Queue for when the response finishes" : "Send · Enter"}>
            {chat.sending || chat.uploading ? <LoaderCircle size={17} aria-hidden="true" /> : chat.runActive ? <Plus size={16} aria-hidden="true" /> : <ArrowUp size={17} aria-hidden="true" />}
            {chat.runActive ? <span>Queue</span> : null}
          </button>
        </div>
        <input ref={fileInput} type="file" multiple hidden accept={CHAT_ATTACHMENT_ACCEPT} onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length) void chat.addFiles(files);
        }} />
      </div>
      <div className={`composer-hint${chat.uploading || chat.projectOpening || chat.queued ? "" : " is-sr-only"}`} id={`${id}-hint`}>
        {chat.uploading ? <span role="status">Attaching files…</span> : chat.projectOpening ? <span role="status">Opening project…</span> : chat.queued ? <span role="status">Your queued message sends when this response finishes. You can keep drafting.</span> : <span>Enter to send. Shift Enter for a new line.</span>}
      </div>
    </form>
  );
}
