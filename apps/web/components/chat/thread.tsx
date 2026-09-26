"use client";

// The thread: your messages as quiet blocks, Metis's replies as the reading
// column, the run's working folded under each reply, and the one question
// waiting on you docked at the foot. Streaming follows only while you are
// already at the bottom.

import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { ArrowDown, Copy, Pencil, RotateCcw, ThumbsDown, ThumbsUp } from "lucide-react";

import { ApplyCard } from "@/components/apply-card";
import { ApprovalCard } from "@/components/approval-card";
import { ArtifactViewer } from "@/components/artifact-viewer";
import { FolderDropSheet } from "@/components/chat/folder-drop-sheet";
import { Welcome } from "@/components/chat/welcome";
import { ElicitationCard } from "@/components/elicitation-card";
import { MarkdownContent } from "@/components/markdown-content";
import { ProjectActivity } from "@/components/project-activity";
import { isPersisted, TOOL_BUILD_PROMPT, type Chat } from "@/hooks/use-chat";
import { attachmentBadge } from "@/lib/attachments";
import { droppedDirectories, looseFiles } from "@/lib/folder-drop";
import { nextThreadFollowing } from "@/lib/chat-scroll";
import { splitCitedSources } from "@/lib/markdown-links";
import { messageBelongsToRun } from "@/lib/run-history";
import type { ChatMessage } from "@/lib/types";

function clock(timestamp?: string): string {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function Thread({ chat }: { chat: Chat }) {
  const viewport = useRef<HTMLDivElement>(null);
  // Rows act through the latest chat, so a memoised row never holds a stale handler.
  const latest = useRef(chat);
  useEffect(() => {
    latest.current = chat;
  });
  const [atBottom, setAtBottom] = useState(true);
  const following = useRef(true);
  const lastScrollTop = useRef(0);
  const manualScrollUntil = useRef(0);
  const jumping = useRef(false);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const [droppedFolders, setDroppedFolders] = useState<FileSystemDirectoryEntry[]>([]);

  useLayoutEffect(() => {
    following.current = true;
    manualScrollUntil.current = 0;
    jumping.current = false;
    const element = viewport.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
      lastScrollTop.current = element.scrollTop;
    }
    setAtBottom(true);
  }, [chat.conversationId]);

  useEffect(() => {
    dragDepth.current = 0;
    setDragging(false);
    setDroppedFolders([]);
  }, [chat.conversationId]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const markManualScroll = () => { manualScrollUntil.current = performance.now() + 900; };
    const markScrollbarDrag = (event: PointerEvent) => {
      if (event.clientX >= element.getBoundingClientRect().right - 22) markManualScroll();
    };
    const markScrollbarMove = (event: PointerEvent) => {
      if (event.buttons === 1 && event.clientX >= element.getBoundingClientRect().right - 22) markManualScroll();
    };
    const markKeyboardScroll = (event: KeyboardEvent) => {
      if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) markManualScroll();
    };
    const sync = () => {
      let top = element.scrollTop;
      const manual = performance.now() < manualScrollUntil.current;
      const distance = element.scrollHeight - top - element.clientHeight;
      if (manual) jumping.current = false;
      following.current = nextThreadFollowing(following.current, lastScrollTop.current, top, distance, manual ? "scroll" : "resize");
      if (!manual && !jumping.current && following.current && distance > 1) {
        element.scrollTop = element.scrollHeight;
        top = element.scrollTop;
      }
      if (jumping.current && distance <= 1) jumping.current = false;
      lastScrollTop.current = top;
      setAtBottom(following.current);
    };
    const keepPosition = () => {
      if (following.current) {
        element.scrollTop = element.scrollHeight;
        lastScrollTop.current = element.scrollTop;
      }
      // A resize only changes geometry. It must not undo the reader's upward
      // scroll just because they paused close to the end.
      following.current = nextThreadFollowing(following.current, lastScrollTop.current, element.scrollTop, element.scrollHeight - element.scrollTop - element.clientHeight, "resize");
      setAtBottom(following.current);
    };
    const observer = new ResizeObserver(keepPosition);
    observer.observe(element);
    const list = element.querySelector(".thread-list");
    if (list) observer.observe(list);
    element.addEventListener("wheel", markManualScroll, { passive: true });
    element.addEventListener("touchmove", markManualScroll, { passive: true });
    element.addEventListener("pointerdown", markScrollbarDrag, { passive: true });
    element.addEventListener("pointermove", markScrollbarMove, { passive: true });
    element.addEventListener("keydown", markKeyboardScroll);
    element.addEventListener("scroll", sync, { passive: true });
    keepPosition();
    return () => {
      observer.disconnect();
      element.removeEventListener("wheel", markManualScroll);
      element.removeEventListener("touchmove", markManualScroll);
      element.removeEventListener("pointerdown", markScrollbarDrag);
      element.removeEventListener("pointermove", markScrollbarMove);
      element.removeEventListener("keydown", markKeyboardScroll);
      element.removeEventListener("scroll", sync);
    };
  }, [chat.conversationId, chat.hasMessages]);

  useLayoutEffect(() => {
    const element = viewport.current;
    if (!element || !following.current) return;
    element.scrollTop = element.scrollHeight;
    lastScrollTop.current = element.scrollTop;
  }, [chat.messages, chat.hasMessages]);

  const jump = useCallback(() => {
    const element = viewport.current;
    if (!element) return;
    following.current = true;
    manualScrollUntil.current = 0;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    jumping.current = !reduced && !latest.current.runActive;
    element.scrollTo({ top: element.scrollHeight, behavior: jumping.current ? "smooth" : "auto" });
    setAtBottom(true);
  }, []);

  return (
    <div className="thread-shell">
      <div
      ref={viewport}
      className={`thread${dragging ? " is-dragging" : ""}`}
      aria-label="Conversation messages"
      onDragEnter={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes("Files")) event.preventDefault();
      }}
      onDragLeave={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        // The entries are neutered once this handler returns; claim them first.
        const directories = droppedDirectories(event.dataTransfer.items);
        const files = looseFiles(event.dataTransfer.files, directories);
        if (files.length) void chat.addFiles(files);
        if (directories.length) setDroppedFolders(directories);
      }}
    >
      {dragging ? <div className="thread-drop"><strong>Drop files or a folder</strong><small>Files attach to this message. A folder can be indexed, opened as a project, or attached.</small></div> : null}
      {droppedFolders.length ? <FolderDropSheet chat={chat} directories={droppedFolders} onClose={() => setDroppedFolders([])} /> : null}

      {!chat.hasMessages ? (
        <Welcome chat={chat} />
      ) : (
        <div className="thread-list">
          {chat.loadingConversation ? <div className="thread-loading" role="status" aria-label="Opening conversation"><span /><span /><span /></div> : null}
          {chat.messages.map((message) => <Message key={message.id} message={message} chat={chat} latest={latest} />)}
          {chat.activeRunId && chat.events.length > 0 && !chat.messages.some((message) => messageBelongsToRun(message, chat.activeRunId)) ? (
            <ProjectActivity events={chat.events} live={chat.runActive && !chat.pendingApproval && !chat.pendingElicitation} attention={Boolean(chat.pendingApproval || chat.pendingElicitation)} stageLabel={chat.stageLabel} projectName={chat.selectedProject?.name} />
          ) : null}
          {chat.pendingApproval ? (
            <div className="thread-dock">
              <ApprovalCard approval={chat.pendingApproval} decided={chat.decidedApprovals.has(chat.pendingApproval.id)} decisionBusy={chat.decisionBusy} onDecision={chat.decide} approveLabel={chat.approveLabel} variant="inline" />
            </div>
          ) : null}
          {chat.pendingElicitation ? (
            <div className="thread-dock">
              <ElicitationCard elicitation={chat.pendingElicitation} answered={chat.answeredElicitations.has(chat.pendingElicitation.id)} answerBusy={chat.answerBusy} onAnswer={chat.answer} variant="inline" />
            </div>
          ) : null}
          {chat.pendingSuggestion ? (
            <div className="thread-dock">
              <ApplyCard suggestion={chat.pendingSuggestion} state={chat.suggestionState} onApply={chat.applyProposal} onDismiss={chat.dismissProposal} />
            </div>
          ) : null}
        </div>
      )}

      </div>
      {chat.hasMessages && !atBottom ? (
        <button type="button" className="ui-btn is-sm thread-jump" onClick={jump}>
          <ArrowDown size={14} aria-hidden="true" /> {chat.runActive ? "Follow answer" : "Jump to latest"}
        </button>
      ) : null}
    </div>
  );
}

interface MessageProps {
  message: ChatMessage;
  chat: Chat;
  latest: RefObject<Chat>;
}

// A row that is being edited, belongs to the active run, or is the newest
// answer reads live state; every other row shows only its own message.
function liveIn(chat: Chat, message: ChatMessage): boolean {
  return chat.editingMessageId === message.id
    || messageBelongsToRun(message, chat.activeRunId)
    || chat.latestAssistant?.id === message.id;
}

function sameRow(prev: MessageProps, next: MessageProps): boolean {
  const { message } = next;
  if (prev.message !== message || message.streaming || message.failed) return false;
  if (liveIn(prev.chat, message) || liveIn(next.chat, message)) return false;
  const p = prev.chat;
  const n = next.chat;
  return p.rewinding === n.rewinding
    && p.runActive === n.runActive
    && p.selectedCustomer === n.selectedCustomer
    && (p.copiedMessageId === message.id) === (n.copiedMessageId === message.id)
    && (p.savingToAccount === message.id) === (n.savingToAccount === message.id)
    && p.savedToAccount.has(message.id) === n.savedToAccount.has(message.id);
}

// Memoised on what the row shows: typing in the composer re-renders the
// thread, but a settled row outside the active run does not repaint.
const Message = memo(function Message({ message, chat, latest }: MessageProps) {
  const editing = chat.editingMessageId === message.id;
  const inRun = messageBelongsToRun(message, chat.activeRunId);
  const isLatest = message.id === chat.latestAssistant?.id;
  const settled = !message.streaming && !message.failed && Boolean(message.content) && !editing;
  const citedSources = settled && message.role === "assistant" ? splitCitedSources(message.content).sources.length : 0;
  const editBox = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (editing) editBox.current?.focus();
  }, [editing]);

  return (
    <article className={`msg is-${message.role}${message.failed ? " is-failed" : ""}${editing ? " is-editing" : ""}${message.streaming ? " is-streaming" : ""}`}>
      <header className="msg-head">
        <span className="msg-speaker">{message.role === "user" ? "You" : <><span className="msg-avatar" aria-hidden="true">✦</span>Metis</>}</span>
        <time dateTime={message.created_at}>{clock(message.created_at)}</time>
      </header>

      {inRun && (message.streaming || chat.events.length) ? (
        <ProjectActivity events={chat.events} reasoning={message.reasoning} live={Boolean(message.streaming) && !chat.pendingApproval && !chat.pendingElicitation} attention={Boolean(chat.pendingApproval || chat.pendingElicitation)} stageLabel={isLatest ? chat.stageLabel : null} projectName={chat.selectedProject?.name} />
      ) : message.reasoning ? (
        <details className="msg-thinking">
          <summary>How Metis worked</summary>
          <pre>{message.reasoning}</pre>
        </details>
      ) : null}

      {editing ? (
        <div className="msg-editor">
          <textarea
            ref={editBox}
            value={chat.editDraft}
            onChange={(event) => latest.current.setEditDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229 || chat.rewinding) return;
              if (event.key === "Escape") { event.preventDefault(); latest.current.cancelEditing(); }
              if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void latest.current.submitEdit(message); }
            }}
            aria-label="Edit your message"
            disabled={chat.rewinding}
            maxLength={20000}
          />
          <div className="msg-editor-bar">
            <small>Everything after this message is removed.</small>
            <button type="button" className="ui-btn is-quiet is-sm" onClick={latest.current.cancelEditing} disabled={chat.rewinding}>Cancel</button>
            <button type="button" className="ui-btn is-primary is-sm" disabled={chat.rewinding || !chat.editDraft.trim()} onClick={() => void latest.current.submitEdit(message)}>{chat.rewinding ? "Rewinding…" : "Send"}</button>
          </div>
        </div>
      ) : message.content ? (
        <div className="msg-body"><MarkdownContent content={message.content} /></div>
      ) : message.streaming && !inRun ? (
        <p className="msg-working">{(isLatest ? chat.stageLabel : null) ?? "Understanding the task and choosing a safe route…"}</p>
      ) : null}

      {message.attachments?.length ? (
        <div className="msg-files">{message.attachments.map((attachment) => <span key={attachment.id}><b>{attachmentBadge(attachment)}</b>{attachment.name}</span>)}</div>
      ) : null}
      {message.role === "assistant" && message.kind !== "tracker" && (inRun || (!chat.activeRunId && isLatest)) ? <ArtifactViewer artifacts={chat.artifacts} /> : null}

      {message.failed ? (
        <div className="msg-actions is-visible">
          <button type="button" className="ui-btn is-sm" onClick={() => latest.current.clearFailedResponse(message, true)}>Edit & retry</button>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => latest.current.clearFailedResponse(message, false)}>Clear</button>
        </div>
      ) : null}

      {settled ? (
        <footer className="msg-foot">
          {citedSources > 0 ? (
            <span className="msg-grounded">{citedSources} {citedSources === 1 ? "source" : "sources"} cited</span>
          ) : null}
          <div className="msg-actions">
            <button type="button" className="ui-btn is-quiet is-sm msg-action-icon" aria-label={chat.copiedMessageId === message.id ? "Copied" : "Copy message"} title={chat.copiedMessageId === message.id ? "Copied" : "Copy"} onClick={() => void latest.current.copyMessage(message)}><Copy size={14} aria-hidden="true" /></button>
            {message.role === "user" ? (
              <button type="button" className="ui-btn is-quiet is-sm msg-action-icon" aria-label="Edit message" onClick={() => latest.current.startEditing(message)} disabled={chat.runActive || chat.rewinding || !isPersisted(message)} title="Edit and rewind the conversation to here"><Pencil size={14} aria-hidden="true" /></button>
            ) : message.kind === "tracker" ? null : (
              <>
                <button type="button" className="ui-btn is-quiet is-sm msg-action-icon" aria-label="Regenerate response" onClick={() => void latest.current.retryAnswer(message)} disabled={chat.runActive || chat.rewinding} title="Regenerate response"><RotateCcw size={14} aria-hidden="true" /></button>
                {inRun ? (
                  <>
                    <button type="button" className="ui-btn is-quiet is-sm" disabled={chat.sending || chat.uploading || chat.runActive} title="Turn this repeatable process into a governed tool" onClick={() => void latest.current.submit(TOOL_BUILD_PROMPT)}>Make a tool</button>
                    {chat.selectedCustomer ? (
                      <button type="button" className="ui-btn is-quiet is-sm" disabled={chat.savingToAccount === message.id || chat.savedToAccount.has(message.id)} onClick={() => void latest.current.saveToAccount(message)}>
                        {chat.savedToAccount.has(message.id) ? `Saved to ${chat.selectedCustomer.name}` : chat.savingToAccount === message.id ? "Saving…" : `Save to ${chat.selectedCustomer.name}`}
                      </button>
                    ) : null}
                    {chat.feedback.mode === "sent" ? (
                      <span className="msg-rated">Feedback recorded</span>
                    ) : (
                      <>
                        <button type="button" className="ui-btn is-quiet is-sm msg-action-icon" onClick={() => void latest.current.rate("positive")} disabled={chat.feedback.busy} aria-label="This was useful" title="This was useful"><ThumbsUp size={14} aria-hidden="true" /></button>
                        <button type="button" className="ui-btn is-quiet is-sm msg-action-icon" onClick={() => void latest.current.rate("negative")} disabled={chat.feedback.busy} aria-label="Needs a correction" title="Needs a correction"><ThumbsDown size={14} aria-hidden="true" /></button>
                      </>
                    )}
                  </>
                ) : null}
              </>
            )}
          </div>
          {inRun && chat.feedback.mode === "correcting" ? (
            <div className="msg-correction">
              <textarea value={chat.feedback.correction} onChange={(event) => latest.current.setCorrection(event.target.value)} maxLength={20000} placeholder="What should Metis learn or correct? It becomes a memory proposal for you to review." aria-label="Correction" />
              <div className="msg-editor-bar">
                <button type="button" className="ui-btn is-quiet is-sm" onClick={latest.current.cancelCorrection}>Cancel</button>
                <button type="button" className="ui-btn is-primary is-sm" disabled={!chat.feedback.correction.trim() || chat.feedback.busy} onClick={() => void latest.current.rate("negative")}>{chat.feedback.busy ? "Recording…" : "Submit correction"}</button>
              </div>
            </div>
          ) : null}
        </footer>
      ) : null}
    </article>
  );
}, sameRow);
