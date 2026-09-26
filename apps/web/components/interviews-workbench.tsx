"use client";

// The interview room. Setup, then a stage to talk to, then a scorecard that
// holds a number the API calculated, never one this file made up.

import { useCallback, useEffect, useMemo, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";
import { Check, ClipboardList, MessageCircle, Mic, Sparkles } from "lucide-react";

import { Transcript } from "@/components/audio/stage";
import { InterviewLive } from "@/components/interviews/live";
import { Delivery, Scorecard } from "@/components/interviews/scorecard";
import { InterviewSetup } from "@/components/interviews/setup";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { useInterviewSession } from "@/hooks/use-interview-session";
import { getInterviewAvailability } from "@/lib/api";
import { safeFilePart } from "@/lib/audio";
import { clearInterviewDraft, draftToContext, EMPTY_DRAFT, INTERVIEW_TYPES, loadInterviewDraft, parseFocusAreas, saveInterviewDraft, type InterviewDraft } from "@/lib/interviews";
import type { InterviewAvailability } from "@/lib/types";

export function InterviewsWorkbench() {
  return (
    <ConversationProvider>
      <Body />
    </ConversationProvider>
  );
}

function Body() {
  const [availability, setAvailability] = useState<InterviewAvailability | null>(null);
  const [availabilityError, setAvailabilityError] = useState<string | null>(null);
  const [draft, setDraft] = useState<InterviewDraft>(EMPTY_DRAFT);
  const [draftLoaded, setDraftLoaded] = useState(false);
  const context = useMemo(() => draftToContext(draft), [draft]);
  const interview = useInterviewSession({ context });
  const checkAvailability = useCallback(async () => {
    setAvailabilityError(null);
    try {
      setAvailability(await getInterviewAvailability());
    } catch (error) {
      setAvailabilityError(error instanceof Error ? error.message : "Interview readiness could not be checked.");
    }
  }, []);

  // The draft loads after mount so server and first paint agree, then every
  // edit is kept: an accidental navigation must not cost a pasted description.
  useEffect(() => {
    setDraft(loadInterviewDraft());
    setDraftLoaded(true);
    void checkAvailability();
  }, [checkAvailability]);
  useEffect(() => { if (draftLoaded) saveInterviewDraft(draft); }, [draft, draftLoaded]);

  const retryRound = useCallback(() => { interview.reset(); void interview.start(); }, [interview]);
  const tryAnotherType = useCallback(() => { setDraft((current) => ({ ...current, interview_type: "" })); interview.reset(); }, [interview]);
  const newRole = useCallback(() => { clearInterviewDraft(); setDraft({ ...EMPTY_DRAFT }); interview.reset(); }, [interview]);

  const jobTitle = interview.session?.job_title ?? draft.job_title;
  const companyName = interview.session?.company_name ?? draft.company_name;
  const round = INTERVIEW_TYPES.find((item) => item.value === (interview.session?.interview_type ?? draft.interview_type))?.label ?? "";
  const focus = interview.session?.focus_areas ?? parseFocusAreas(draft.focus_areas);
  const transcript = interview.turns.map((turn) => ({ id: String(turn.ordinal), speaker: turn.role === "agent" ? "Chiron" : "You", text: turn.text, agent: turn.role === "agent" }));
  const preparing = interview.phase === "setup" || interview.phase === "ready";
  const step = preparing ? 0 : interview.live ? 1 : 2;

  return (
    <div className={`interviews conversations-interviews${interview.live ? " is-interview-live" : ""}`}>
      <PageHeader eyebrow="Your practice room" title="Interviews" icon={<Mic size={22} aria-hidden="true" />} lede={preparing ? "Practice the conversation before it matters. A focused round, thoughtful follow-ups, and feedback you can use." : interview.live ? "One conversation at a time. Take a breath, make your point, and keep going." : "Return to your answers, understand your delivery, and choose what to practice next."} />
      <ol className="interview-journey" aria-label="Interview progress">{[{ label: "Prepare your round", hint: "Role and focus", Icon: ClipboardList }, { label: "Have the conversation", hint: "Five questions, live follow-ups", Icon: MessageCircle }, { label: "Reflect and improve", hint: "Your answers and delivery", Icon: Sparkles }].map(({ label, hint, Icon }, index) => <li key={label} className={index === step ? "is-current" : index < step ? "is-done" : ""} aria-current={index === step ? "step" : undefined}><span>{index < step ? <Check size={17} aria-hidden="true" /> : <Icon size={17} aria-hidden="true" />}</span><div><strong>{label}</strong><small>{hint}</small></div></li>)}</ol>
      {interview.error ? <Notice kind="error" onDismiss={interview.dismissError}>{interview.error}</Notice> : null}
      {availabilityError ? <Notice kind="error" title="Interview service unavailable" action="Try again" onAction={() => void checkAvailability()}>{availabilityError}</Notice> : !availability ? <Notice kind="info">Checking interview readiness…</Notice> : null}

      {preparing ? (
        <InterviewSetup draft={draft} onChange={setDraft} availability={availability} canBegin={context !== null && availability?.available === true} onBegin={() => void interview.start()} />
      ) : null}

      {interview.live ? <InterviewLive interview={interview} jobTitle={jobTitle} companyName={companyName} round={round} focus={focus} /> : null}

      {interview.phase === "complete" && interview.scorecard ? (
        <Scorecard scorecard={interview.scorecard} turns={interview.turns} delivery={interview.delivery} deliveryStage={interview.deliveryStage} deliveryBusy={interview.deliveryBusy} onRetryDelivery={() => void interview.retryDelivery()} onRetryRound={retryRound} onTryAnotherType={tryAnotherType} onNewRole={newRole} />
      ) : null}

      {interview.phase === "failed" || interview.phase === "ended_early" ? (
        <section className="interview-scorecard" aria-label="Interview interrupted">
          <h2>{interview.phase === "failed" ? "The interview did not finish" : "Ended early, not scored"}</h2>
          <p>{interview.phase === "failed" ? "The connection ended before Chiron submitted an evaluation. Nothing is invented in that case: there is no score, and your transcript is kept below." : "The round stopped before the fifth answer, so there is no score. Next time, tell Chiron you want to stop; after three answers it can still evaluate."}</p>
          <div className="interview-next">
            <button type="button" className="ui-btn is-primary" onClick={retryRound}>Try again</button>
            <button type="button" className="ui-btn" onClick={() => interview.reset()}>Back to setup</button>
          </div>
          {interview.phase === "ended_early" ? <Delivery delivery={interview.delivery} stage={interview.deliveryStage} busy={interview.deliveryBusy} onRetry={() => void interview.retryDelivery()} /> : null}
          <Transcript turns={transcript} filename={`${safeFilePart(jobTitle)}-interview-transcript.txt`} />
        </section>
      ) : null}
    </div>
  );
}
