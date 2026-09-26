"use client";

// The chat page: a header that says what is running, the thread, the
// composer, and the activity drawer beside them. State lives in useChat;
// this file only arranges the pieces.

import { Activity, Keyboard, Mic, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Composer } from "@/components/chat/composer";
import { Thread } from "@/components/chat/thread";
import { ModelControl } from "@/components/model-control";
import { TaskExecutionBar, TaskExecutionPanel } from "@/components/task-execution";
import { Notice } from "@/components/ui/notice";
import { VoicePanel } from "@/components/voice-panel";
import { useChat } from "@/hooks/use-chat";
import { useDictation } from "@/hooks/use-dictation";
import { useDialogFocus } from "@/hooks/use-dialog-focus";
import { isCloudActive } from "@/lib/model";
import { deriveTaskExecution } from "@/lib/task-execution";
import { mergeExecutionEvents } from "@/lib/execution-milestones";
import "@/components/task-execution.css";

export function ChatWorkspace() {
  const chat = useChat();
  const router = useRouter();
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [narrow, setNarrow] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [steering, setSteering] = useState(false);
  const activityButton = useRef<HTMLButtonElement>(null);
  const activityPanel = useRef<HTMLElement>(null);
  const workspace = useRef<HTMLDivElement>(null);
  const { setDraft, focusComposer } = chat;

  // Spoken words append, so half a typed sentence and half a spoken one both survive.
  const append = useCallback((text: string) => {
    const said = text.trim();
    if (!said) return;
    setDraft((current) => (current.trim() ? `${current.replace(/\s+$/, "")} ${said}` : said));
    focusComposer();
  }, [focusComposer, setDraft]);
  const dictation = useDictation(append);
  const openVoice = () => {
    // One microphone: drop any dictation before the voice room asks for it.
    dictation.cancel();
    setVoiceOpen(true);
  };

  const mode = chat.selectedProject ? "Project" : isCloudActive(chat.modelPreference) ? "Cloud" : "Local";
  const { timelineOpen, setTimelineOpen } = chat;
  const closeTask = useCallback(() => setTimelineOpen(false), [setTimelineOpen]);
  useDialogFocus(activityPanel, timelineOpen && narrow, closeTask);
  const execution = useMemo(() => deriveTaskExecution(mergeExecutionEvents(chat.executionEvents, chat.events, chat.activeRunId), {
    runId: chat.activeRunId,
    active: chat.runActive,
    connection: chat.connection,
    pendingDecision: Boolean(chat.pendingApproval || chat.pendingElicitation),
    project: Boolean(chat.selectedProject),
  }), [chat.activeRunId, chat.connection, chat.events, chat.executionEvents, chat.pendingApproval, chat.pendingElicitation, chat.runActive, chat.selectedProject]);
  const showTaskBar = execution.substantial || Boolean(chat.pendingApproval || chat.pendingElicitation);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 1080px)");
    const update = () => setNarrow(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => { setSteering(false); setStopping(false); }, [chat.activeRunId]);

  const steer = () => {
    setSteering(true);
    setVoiceOpen(false);
    if (narrow) setTimelineOpen(false);
    chat.focusComposer();
  };
  const stopTask = async () => {
    if (stopping || !chat.runActive) return;
    setStopping(true);
    try { await chat.stop(); } finally { setStopping(false); }
  };
  const showResponse = () => {
    setVoiceOpen(false);
    if (narrow) setTimelineOpen(false);
    window.setTimeout(() => {
      const thread = workspace.current?.querySelector<HTMLElement>(".thread");
      thread?.scrollTo({ top: thread.scrollHeight, behavior: chat.runActive || window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }, 0);
  };

  useEffect(() => {
    if (!timelineOpen) return;
    const close = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented || document.querySelector('[aria-modal="true"]')) return;
      event.preventDefault();
      setTimelineOpen(false);
      activityButton.current?.focus();
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [setTimelineOpen, timelineOpen]);

  return (
    <div ref={workspace} className={`chat has-execution${chat.timelineOpen ? " has-activity" : ""}`}>
      <section className="chat-main">
        <header className="chat-head">
          <div className="chat-title">
            <span className="ui-status"><span className={`ui-dot ${chat.runActive ? "is-waiting" : "is-live"}`} aria-hidden="true" />{mode}</span>
            <strong title={chat.conversationTitle}>{chat.conversationTitle}</strong>
          </div>
          <div className="chat-tools">
            <ModelControl
              preference={chat.modelPreference}
              onChooseProvider={(provider) => void chat.chooseProvider(provider)}
              onPreferenceChange={chat.setModelPreference}
              providerSaving={chat.providerSaving}
              project={chat.selectedProject}
              projectMode={chat.projectMode}
              onChooseProjectMode={(item) => void chat.chooseProjectMode(item)}
              projectBusy={chat.projectOpening}
              disabled={chat.runActive}
            />
            <div className="chat-mode" role="group" aria-label="Conversation mode">
              <button type="button" className={`ui-btn is-sm${voiceOpen ? " is-quiet" : ""}`} aria-pressed={!voiceOpen} onClick={() => setVoiceOpen(false)}><Keyboard size={14} aria-hidden="true" /> Chat</button>
              <button type="button" className={`ui-btn is-sm${voiceOpen ? "" : " is-quiet"}`} aria-pressed={voiceOpen} onClick={openVoice}><Mic size={14} aria-hidden="true" /> Voice</button>
            </div>
            <button ref={activityButton} type="button" className={`ui-btn is-sm chat-activity-toggle${chat.timelineOpen ? "" : " is-quiet"}`} aria-label="Task overview" aria-expanded={chat.timelineOpen} aria-controls="chat-activity" onClick={() => chat.setTimelineOpen(!chat.timelineOpen)} title="Plan, progress, outputs and decisions">
              <Activity size={14} aria-hidden="true" />
              <span>Task</span>
              {chat.events.length ? <small className="chat-count">{chat.events.length}</small> : null}
            </button>
          </div>
        </header>

        {chat.recoverableRuns.length || showTaskBar ? (
          <div className="chat-notice">
            {chat.recoverableRuns.length ? <Notice kind="info" title={chat.recoverableRuns.length === 1 ? "A run is waiting for you" : `${chat.recoverableRuns.length} runs are waiting for you`}>
              <div className="chat-recover">
                {chat.recoverableRuns.map((item) => (
                  <button key={item.run.id} type="button" className="ui-btn is-sm" disabled={item.run.id === chat.activeRunId} onClick={() => router.push(`/?conversation=${encodeURIComponent(item.run.conversation_id)}&run=${encodeURIComponent(item.run.id)}`)}>
                    {item.run.id === chat.activeRunId ? "Reviewing now" : item.approval?.title ?? "Review approval"}
                  </button>
                ))}
              </div>
            </Notice> : null}
            {showTaskBar ? <TaskExecutionBar execution={execution} stageLabel={chat.stageLabel} open={chat.timelineOpen} onOpen={() => chat.setTimelineOpen(true)} /> : null}
          </div>
        ) : null}

        {voiceOpen ? (
          <div className="chat-voice"><VoicePanel onHandoff={(item) => append(item.transcript)} onExitToChat={() => setVoiceOpen(false)} /></div>
        ) : (
          <Thread chat={chat} />
        )}

        <div className="chat-foot">
          {steering && (!chat.timelineOpen || narrow) ? <div className="task-composer-direction" role="status"><span>{chat.runActive ? "Add direction below. It sends after this run; stop the task to change course now." : "Continue this task with a follow-up below."}</span><button type="button" className="ui-btn is-quiet is-sm" aria-label="Dismiss direction hint" onClick={() => setSteering(false)}>×</button></div> : null}
          {chat.error ? (
            <Notice kind="error" title="Something interrupted this chat" action="Start fresh" onAction={chat.startFresh} onDismiss={() => chat.setError(null)}>
              {chat.error}
            </Notice>
          ) : null}
          {dictation.error ? <Notice kind="error" title="Dictation" onDismiss={dictation.dismissError}>{dictation.error}</Notice> : null}
          <Composer chat={chat} dictation={dictation} voiceOpen={voiceOpen} />
        </div>
      </section>

      {narrow && chat.timelineOpen ? <button type="button" className="task-panel-backdrop" tabIndex={-1} aria-label="Close task overview" onClick={closeTask} /> : null}
      <aside ref={activityPanel} className="chat-activity task-panel" id="chat-activity" aria-hidden={!chat.timelineOpen} inert={!chat.timelineOpen} aria-label="Task overview" role={narrow && chat.timelineOpen ? "dialog" : undefined} aria-modal={narrow && chat.timelineOpen || undefined}>
        <button type="button" className="ui-btn is-quiet is-sm chat-activity-close" aria-label="Close task overview" onClick={() => { chat.setTimelineOpen(false); activityButton.current?.focus(); }}><X size={16} /></button>
        <TaskExecutionPanel chat={chat} execution={execution} stopping={stopping} steering={steering} onSteer={steer} onStop={() => void stopTask()} onShowResponse={showResponse} />
      </aside>
    </div>
  );
}
