"use client";

// The chat page: a header that says what is running, the thread, the
// composer, and the activity drawer beside them. State lives in useChat;
// this file only arranges the pieces.

import { Activity, Keyboard, Mic, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";

import { Composer } from "@/components/chat/composer";
import { Thread } from "@/components/chat/thread";
import { ModelControl } from "@/components/model-control";
import { RunTimeline } from "@/components/run-timeline";
import { Notice } from "@/components/ui/notice";
import { VoicePanel } from "@/components/voice-panel";
import { useChat } from "@/hooks/use-chat";
import { useDictation } from "@/hooks/use-dictation";
import { isCloudActive } from "@/lib/model";

export function ChatWorkspace() {
  const chat = useChat();
  const router = useRouter();
  const [voiceOpen, setVoiceOpen] = useState(false);
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

  return (
    <div className={`chat${chat.timelineOpen ? " has-activity" : ""}`}>
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
            <button type="button" className={`ui-btn is-sm${chat.timelineOpen ? "" : " is-quiet"}`} aria-pressed={chat.timelineOpen} onClick={() => chat.setTimelineOpen(!chat.timelineOpen)} title="Show what Metis did">
              <Activity size={14} aria-hidden="true" />
              {chat.runActive ? "Live" : "Activity"}
              {chat.events.length ? <small className="chat-count">{chat.events.length}</small> : null}
            </button>
          </div>
        </header>

        {chat.recoverableRuns.length ? (
          <div className="chat-notice">
            <Notice kind="info" title={chat.recoverableRuns.length === 1 ? "A run is waiting for you" : `${chat.recoverableRuns.length} runs are waiting for you`}>
              <div className="chat-recover">
                {chat.recoverableRuns.map((item) => (
                  <button key={item.run.id} type="button" className="ui-btn is-sm" disabled={item.run.id === chat.activeRunId} onClick={() => router.push(`/?conversation=${encodeURIComponent(item.run.conversation_id)}&run=${encodeURIComponent(item.run.id)}`)}>
                    {item.run.id === chat.activeRunId ? "Reviewing now" : item.approval?.title ?? "Review approval"}
                  </button>
                ))}
              </div>
            </Notice>
          </div>
        ) : null}

        {voiceOpen ? (
          <div className="chat-voice"><VoicePanel onHandoff={(item) => append(item.transcript)} onExitToChat={() => setVoiceOpen(false)} /></div>
        ) : (
          <Thread chat={chat} />
        )}

        <div className="chat-foot">
          {chat.error ? (
            <Notice kind="error" title="Something interrupted this chat" action="Start fresh" onAction={chat.startFresh} onDismiss={() => chat.setError(null)}>
              {chat.error}
            </Notice>
          ) : null}
          {dictation.error ? <Notice kind="error" title="Dictation" onDismiss={dictation.dismissError}>{dictation.error}</Notice> : null}
          <Composer chat={chat} dictation={dictation} voiceOpen={voiceOpen} />
        </div>
      </section>

      <aside className="chat-activity" aria-hidden={!chat.timelineOpen} aria-label="Activity">
        <button type="button" className="ui-btn is-quiet is-sm chat-activity-close" aria-label="Close activity" onClick={() => chat.setTimelineOpen(false)}><X size={16} /></button>
        <RunTimeline events={chat.events} connection={chat.connection} streamError={chat.streamError} onDecision={chat.decide} decidedApprovals={chat.decidedApprovals} decisionBusy={chat.decisionBusy} approveLabel={chat.approveLabel} />
      </aside>
    </div>
  );
}
