"use client";

// The interview room. Setup, then a stage to talk to, then a scorecard that
// holds a number the API calculated, never one this file made up.

import { useCallback, useEffect, useMemo, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";

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
  const [draft, setDraft] = useState<InterviewDraft>(EMPTY_DRAFT);
  const [draftLoaded, setDraftLoaded] = useState(false);
  const context = useMemo(() => draftToContext(draft), [draft]);
  const interview = useInterviewSession({ context });

  // The draft loads after mount so server and first paint agree, then every
  // edit is kept: an accidental navigation must not cost a pasted description.
  useEffect(() => {
    setDraft(loadInterviewDraft());
    setDraftLoaded(true);
    void getInterviewAvailability().then(setAvailability).catch(() => setAvailability(null));
  }, []);
  useEffect(() => { if (draftLoaded) saveInterviewDraft(draft); }, [draft, draftLoaded]);

  const retryRound = useCallback(() => { interview.reset(); void interview.start(); }, [interview]);
  const tryAnotherType = useCallback(() => { setDraft((current) => ({ ...current, interview_type: "" })); interview.reset(); }, [interview]);
  const newRole = useCallback(() => { clearInterviewDraft(); setDraft({ ...EMPTY_DRAFT }); interview.reset(); }, [interview]);

  const jobTitle = interview.session?.job_title ?? draft.job_title;
  const companyName = interview.session?.company_name ?? draft.company_name;
  const round = INTERVIEW_TYPES.find((item) => item.value === (interview.session?.interview_type ?? draft.interview_type))?.label ?? "";
  const focus = interview.session?.focus_areas ?? parseFocusAreas(draft.focus_areas);
  const transcript = interview.turns.map((turn) => ({ id: String(turn.ordinal), speaker: turn.role === "agent" ? "Chiron" : "You", text: turn.text, agent: turn.role === "agent" }));

  return (
    <div className="interviews">
      <PageHeader eyebrow="Mock rounds" title="Interviews" lede="Five questions, real follow-ups, and a delivery read from the recording. No rehearsed praise." />
      {interview.error ? <Notice kind="error" onDismiss={interview.dismissError}>{interview.error}</Notice> : null}

      {interview.phase === "setup" || interview.phase === "ready" ? (
        <InterviewSetup draft={draft} onChange={setDraft} availability={availability} canBegin={context !== null && availability?.available !== false} onBegin={() => void interview.start()} />
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
