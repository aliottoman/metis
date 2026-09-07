"use client";

// The thread: your messages as quiet blocks, Metis's replies as the reading
// column, the run's working folded under each reply, and the one question
// waiting on you docked at the foot. Streaming follows only while you are
// already at the bottom.

import { useCallback, useEffect, useRef, useState } from "react";

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
import { messageBelongsToRun } from "@/lib/run-history";
import type { ChatMessage } from "@/lib/types";

// Within this of the foot counts as reading the newest message.
const BOTTOM = 96;

function clock(timestamp?: string): string {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function Thread({ chat }: { chat: Chat }) {
  const viewport = useRef<HTMLDivElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [droppedFolders, setDroppedFolders] = useState<FileSystemDirectoryEntry[]>([]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const sync = () => setAtBottom(element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM);
    sync();
    element.addEventListener("scroll", sync, { passive: true });
    return () => element.removeEventListener("scroll", sync);
  }, [chat.hasMessages]);

  useEffect(() => {
    if (!atBottom) return;
    const frame = requestAnimationFrame(() => {
      const streaming = chat.messages.some((message) => message.streaming);
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      end.current?.scrollIntoView({ behavior: streaming || reduced ? "auto" : "smooth", block: "end" });
    });
    return () => cancelAnimationFrame(frame);
  }, [atBottom, chat.messages]);

  const jump = useCallback(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    setAtBottom(true);
  }, []);

  return (
    <div
      ref={viewport}
      className={`thread${dragging ? " is-dragging" : ""}`}
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false); }}
      onDrop={(event) => {
        event.preventDefault();
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
          {chat.loadingConversation ? <div className="thread-loading" aria-label="Opening"><span /><span /><span /></div> : null}
          {chat.messages.map((message) => <Message key={message.id} message={message} chat={chat} />)}
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
          <div ref={end} />
        </div>
      )}

      {chat.hasMessages && !atBottom ? (
        <button type="button" className="ui-btn is-sm thread-jump" onClick={jump}>
          ↓ {chat.runActive ? "Jump to the live answer" : "Jump to latest"}
        </button>
      ) : null}
    </div>
  );
}

function Message({ message, chat }: { message: ChatMessage; chat: Chat }) {
  const editing = chat.editingMessageId === message.id;
  const inRun = messageBelongsToRun(message, chat.activeRunId);
  const isLatest = message.id === chat.latestAssistant?.id;
  const settled = !message.streaming && !message.failed && Boolean(message.content) && !editing;
  const editBox = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (editing) editBox.current?.focus();
  }, [editing]);

  return (
    <article className={`msg is-${message.role}${message.failed ? " is-failed" : ""}${editing ? " is-editing" : ""}`}>
      <header className="msg-head">
        <span>{message.role === "user" ? "You" : "Metis"}</span>
        <time>{clock(message.created_at)}</time>
      </header>

      {inRun && (message.streaming || chat.events.length) ? (
        <ProjectActivity events={chat.events} reasoning={message.reasoning} live={Boolean(message.streaming) && !chat.pendingApproval && !chat.pendingElicitation} attention={Boolean(chat.pendingApproval || chat.pendingElicitation)} stageLabel={isLatest ? chat.stageLabel : null} projectName={chat.selectedProject?.name} />
      ) : message.reasoning ? (
        <details className="msg-thinking">
          <summary>Thought process · {message.reasoning.trim().split(/\s+/).length} words</summary>
          <pre>{message.reasoning}</pre>
        </details>
      ) : null}

      {editing ? (
        <div className="msg-editor">
          <textarea
            ref={editBox}
            value={chat.editDraft}
            onChange={(event) => chat.setEditDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") { event.preventDefault(); chat.cancelEditing(); }
              if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void chat.submitEdit(message); }
            }}
            aria-label="Edit your message"
            maxLength={20000}
          />
          <div className="msg-editor-bar">
            <small>Everything after this message is removed.</small>
            <button type="button" className="ui-btn is-quiet is-sm" onClick={chat.cancelEditing} disabled={chat.rewinding}>Cancel</button>
            <button type="button" className="ui-btn is-primary is-sm" disabled={chat.rewinding || !chat.editDraft.trim()} onClick={() => void chat.submitEdit(message)}>{chat.rewinding ? "Rewinding…" : "Send"}</button>
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
          <button type="button" className="ui-btn is-sm" onClick={() => chat.clearFailedResponse(message, true)}>Edit & retry</button>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => chat.clearFailedResponse(message, false)}>Clear</button>
        </div>
      ) : null}

      {settled ? (
        <footer className="msg-foot">
          {message.role === "assistant" && inRun && chat.groundedSources > 0 ? (
            <span className="msg-grounded">Grounded in {chat.groundedSources} {chat.groundedSources === 1 ? "source" : "sources"}</span>
          ) : null}
          <div className="msg-actions">
            <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void chat.copyMessage(message)}>{chat.copiedMessageId === message.id ? "Copied" : "Copy"}</button>
            {message.role === "user" ? (
              <button type="button" className="ui-btn is-quiet is-sm" onClick={() => chat.startEditing(message)} disabled={chat.runActive || chat.rewinding || !isPersisted(message)} title="Edit and rewind the conversation to here">Edit</button>
            ) : message.kind === "tracker" ? null : (
              <>
                <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void chat.retryAnswer(message)} disabled={chat.runActive || chat.rewinding} title="Ask again, for example after changing the model">{chat.rewinding ? "Retrying…" : "Retry"}</button>
                {inRun ? (
                  <>
                    <button type="button" className="ui-btn is-quiet is-sm" disabled={chat.sending || chat.uploading || chat.runActive} title="Turn this repeatable process into a governed tool" onClick={() => void chat.submit(TOOL_BUILD_PROMPT)}>Make a tool</button>
                    {chat.selectedCustomer ? (
                      <button type="button" className="ui-btn is-quiet is-sm" disabled={chat.savingToAccount === message.id || chat.savedToAccount.has(message.id)} onClick={() => void chat.saveToAccount(message)}>
                        {chat.savedToAccount.has(message.id) ? `Saved to ${chat.selectedCustomer.name}` : chat.savingToAccount === message.id ? "Saving…" : `Save to ${chat.selectedCustomer.name}`}
                      </button>
                    ) : null}
                    {chat.feedback.mode === "sent" ? (
                      <span className="msg-rated">Feedback recorded</span>
                    ) : (
                      <>
                        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void chat.rate("positive")} disabled={chat.feedback.busy} title="This was useful">👍</button>
                        <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void chat.rate("negative")} disabled={chat.feedback.busy} title="Needs a correction">👎</button>
                      </>
                    )}
                  </>
                ) : null}
              </>
            )}
          </div>
          {inRun && chat.feedback.mode === "correcting" ? (
            <div className="msg-correction">
              <textarea value={chat.feedback.correction} onChange={(event) => chat.setCorrection(event.target.value)} maxLength={20000} placeholder="What should Metis learn or correct? It becomes a memory proposal for you to review." aria-label="Correction" />
              <div className="msg-editor-bar">
                <button type="button" className="ui-btn is-quiet is-sm" onClick={chat.cancelCorrection}>Cancel</button>
                <button type="button" className="ui-btn is-primary is-sm" disabled={!chat.feedback.correction.trim() || chat.feedback.busy} onClick={() => void chat.rate("negative")}>{chat.feedback.busy ? "Recording…" : "Submit correction"}</button>
              </div>
            </div>
          ) : null}
        </footer>
      ) : null}
    </article>
  );
}
