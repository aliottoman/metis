"use client";

// The interview room. The setup form is the last time the user types context;
// after Start the page is an orb to talk to, and the scorecard that replaces
// it holds a number the API calculated — never one the model, or this file,
// made up. No confetti lives here on purpose: a verdict is a verdict.

import { useCallback, useEffect, useMemo, useState } from "react";
import { ConversationProvider } from "@elevenlabs/react";

import { ElevenLabsOrb, type ElevenLabsOrbState } from "@/components/elevenlabs-orb";
import { getInterviewAvailability } from "@/lib/api";
import {
  EMPTY_DRAFT,
  INTERVIEW_TYPES,
  clearInterviewDraft,
  draftToContext,
  elapsedLabel,
  loadInterviewDraft,
  recommendationLabel,
  saveInterviewDraft,
  validateInterviewDraft,
  type InterviewDraft,
} from "@/lib/interviews";
import type { InterviewAvailability, InterviewScorecard } from "@/lib/types";
import {
  useInterviewSession,
  type InterviewPhase,
  type LiveInterviewTurn,
} from "@/hooks/use-interview-session";

const ORB_COLORS: [string, string] = ["#72528a", "#ff7759"];

const PHASE_LABEL: Record<InterviewPhase, string> = {
  setup: "Fill in the role to begin",
  ready: "Ready when you are",
  connecting: "Connecting",
  listening: "Listening",
  agent_speaking: "Chiron is speaking",
  evaluating: "Scoring the interview",
  complete: "Interview complete",
  failed: "Connection failed",
  ended_early: "Ended early",
};

const RUBRIC_ROWS: Array<{ key: keyof RubricScores; label: string }> = [
  { key: "specific_evidence", label: "Specific evidence" },
  { key: "role_depth", label: "Role depth" },
  { key: "relevance", label: "Relevance" },
  { key: "structure", label: "Structure" },
  { key: "communication", label: "Communication" },
];

interface RubricScores {
  specific_evidence: number;
  role_depth: number;
  relevance: number;
  structure: number;
  communication: number;
}

function orbState(phase: InterviewPhase): ElevenLabsOrbState {
  if (phase === "listening") return "listening";
  if (phase === "agent_speaking") return "talking";
  if (phase === "connecting" || phase === "evaluating") return "thinking";
  return null;
}

function roundLabel(value: string): string {
  return INTERVIEW_TYPES.find((round) => round.value === value)?.label ?? value;
}

/** Reduced motion is honoured live: the WebGL orb is replaced by a still one. */
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

export function InterviewsWorkbench() {
  return (
    <ConversationProvider>
      <InterviewsWorkbenchBody />
    </ConversationProvider>
  );
}

function InterviewsWorkbenchBody() {
  const [availability, setAvailability] = useState<InterviewAvailability | null>(null);
  const [draft, setDraft] = useState<InterviewDraft>(EMPTY_DRAFT);
  const [draftLoaded, setDraftLoaded] = useState(false);
  const [touched, setTouched] = useState<Partial<Record<keyof InterviewDraft, boolean>>>({});
  const [confirmingEnd, setConfirmingEnd] = useState(false);
  const [typedText, setTypedText] = useState("");
  const reducedMotion = useReducedMotion();

  const context = useMemo(() => draftToContext(draft), [draft]);
  const interview = useInterviewSession({ context });

  // The draft loads after mount (not during render) so server and first client
  // paint agree, then every edit is preserved — an accidental navigation must
  // not cost a pasted job description.
  useEffect(() => {
    setDraft(loadInterviewDraft());
    setDraftLoaded(true);
    void getInterviewAvailability()
      .then(setAvailability)
      .catch(() => setAvailability(null));
  }, []);
  useEffect(() => {
    if (draftLoaded) saveInterviewDraft(draft);
  }, [draft, draftLoaded]);

  const problems = validateInterviewDraft(draft);
  const showSetup = interview.phase === "setup" || interview.phase === "ready";
  const showLive = interview.live;
  const showScorecard = interview.phase === "complete" && interview.scorecard !== null;
  const showInterrupted =
    interview.phase === "failed" || interview.phase === "ended_early";

  const update = useCallback((field: keyof InterviewDraft, value: string) => {
    setDraft((current) => ({ ...current, [field]: value }));
  }, []);
  const touch = useCallback((field: keyof InterviewDraft) => {
    setTouched((current) => ({ ...current, [field]: true }));
  }, []);

  const begin = useCallback(() => {
    setConfirmingEnd(false);
    void interview.start();
  }, [interview]);

  const retryRound = useCallback(() => {
    interview.reset();
    void interview.start();
  }, [interview]);

  const tryAnotherType = useCallback(() => {
    setDraft((current) => ({ ...current, interview_type: "" }));
    setTouched((current) => ({ ...current, interview_type: false }));
    interview.reset();
  }, [interview]);

  const newRole = useCallback(() => {
    clearInterviewDraft();
    setDraft({ ...EMPTY_DRAFT });
    setTouched({});
    interview.reset();
  }, [interview]);

  const sendTyped = useCallback(() => {
    const text = typedText.trim();
    if (!text) return;
    interview.sendText(text);
    setTypedText("");
  }, [interview, typedText]);

  const jobTitle = interview.session?.job_title ?? draft.job_title;
  const companyName = interview.session?.company_name ?? draft.company_name;
  const interviewType = interview.session?.interview_type ?? draft.interview_type;

  return (
    <div className="workspacePage interviewsPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Mock rounds</span>
          <h1>Interviews</h1>
          <p>Five questions. Direct feedback. No rehearsed praise.</p>
        </div>
      </header>

      {interview.error ? (
        <div className="composerError" role="alert">
          <span>!</span>
          <p>{interview.error}</p>
          <button
            className="errorDismiss"
            type="button"
            aria-label="Dismiss"
            onClick={interview.dismissError}
          >
            ×
          </button>
        </div>
      ) : null}

      {showSetup ? (
        <section className="interviewSetup" aria-label="Interview setup">
          {availability && !availability.available ? (
            <div className="notice interviewUnavailable" role="status">
              <strong>Interviews aren&apos;t configured yet</strong>
              <ul>
                {(availability.missing.length
                  ? availability.missing
                  : [availability.reason]
                ).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="interviewFormGrid">
            <label className="interviewField">
              <span>Job title</span>
              <input
                type="text"
                value={draft.job_title}
                maxLength={200}
                placeholder="Senior Data Engineer"
                onChange={(event) => update("job_title", event.target.value)}
                onBlur={() => touch("job_title")}
              />
              {touched.job_title && problems.job_title ? (
                <small className="interviewFieldProblem" role="alert">
                  {problems.job_title}
                </small>
              ) : null}
            </label>
            <label className="interviewField">
              <span>Company</span>
              <input
                type="text"
                value={draft.company_name}
                maxLength={200}
                placeholder="Batelco"
                onChange={(event) => update("company_name", event.target.value)}
                onBlur={() => touch("company_name")}
              />
              {touched.company_name && problems.company_name ? (
                <small className="interviewFieldProblem" role="alert">
                  {problems.company_name}
                </small>
              ) : null}
            </label>
          </div>

          <label className="interviewField interviewDescriptionField">
            <span>Job description</span>
            <textarea
              value={draft.job_description}
              rows={12}
              placeholder="Paste the whole job description. Chiron draws every question from it."
              onChange={(event) => update("job_description", event.target.value)}
              onBlur={() => touch("job_description")}
            />
            {touched.job_description && problems.job_description ? (
              <small className="interviewFieldProblem" role="alert">
                {problems.job_description}
              </small>
            ) : null}
          </label>

          <div
            className="interviewRounds"
            role="radiogroup"
            aria-label="Interview round"
          >
            {INTERVIEW_TYPES.map((round) => (
              <button
                key={round.value}
                type="button"
                role="radio"
                aria-checked={draft.interview_type === round.value}
                className={`interviewRoundCard ${
                  draft.interview_type === round.value ? "selected" : ""
                }`}
                onClick={() => {
                  update("interview_type", round.value);
                  touch("interview_type");
                }}
              >
                <strong>{round.label}</strong>
                <span>{round.hint}</span>
              </button>
            ))}
          </div>
          {touched.interview_type && problems.interview_type ? (
            <small className="interviewFieldProblem" role="alert">
              {problems.interview_type}
            </small>
          ) : null}

          <div className="interviewStartRow">
            <button
              className="primaryButton interviewStartButton"
              type="button"
              disabled={
                context === null ||
                availability?.available === false ||
                interview.phase === "connecting"
              }
              onClick={begin}
            >
              Begin 5-question interview
            </button>
            <span className="mutedMeta">
              Five questions, one at a time. Chiron evaluates at the end, not
              along the way.
            </span>
          </div>
        </section>
      ) : null}

      {showLive ? (
        <section
          className={`interviewLive is-${interview.phase}`}
          aria-label="Live interview"
        >
          <div className="atmos" aria-hidden="true">
            <span className="chromeForm cf1" />
            <span className="chromeForm cf2" />
            <span className="chromeForm cf3" />
            <span className="chromeForm cf4" />
          </div>
          <div className="interviewLiveInner">
            <header className="interviewLiveHeader">
              <div className="interviewLiveRole">
                <span className="interviewLiveEyebrow">
                  {roundLabel(interviewType)} round
                </span>
                <strong>
                  {jobTitle} · {companyName}
                </strong>
              </div>
              <div className="interviewLiveMeta">
                <span
                  className="interviewProgress"
                  aria-live="polite"
                  aria-label={
                    interview.questionNumber > 0
                      ? `Question ${interview.questionNumber} of ${interview.questionLimit}`
                      : "Interview starting"
                  }
                >
                  {interview.questionNumber > 0
                    ? `Question ${interview.questionNumber} of ${interview.questionLimit}`
                    : "Opening"}
                </span>
                <span className="interviewElapsed">
                  {elapsedLabel(interview.elapsed)}
                </span>
              </div>
            </header>

            <div className="interviewOrbStage" role="status" aria-live="polite">
              {reducedMotion ? (
                <div
                  className={`interviewOrbStill is-${interview.phase}`}
                  aria-hidden="true"
                >
                  <i />
                </div>
              ) : (
                <div className="interviewOrbShell">
                  <ElevenLabsOrb
                    className="interviewOrbRenderer"
                    colors={ORB_COLORS}
                    agentState={orbState(interview.phase)}
                    getInputVolume={interview.getInputVolume}
                    getOutputVolume={interview.getOutputVolume}
                  />
                </div>
              )}
              <div className="interviewStageCopy">
                <strong>{PHASE_LABEL[interview.phase]}</strong>
                <p>
                  {interview.phase === "evaluating"
                    ? "Chiron is scoring the round. The scorecard lands here."
                    : interview.phase === "connecting"
                      ? "Setting up the room."
                      : "Answer out loud. Chiron won't coach you mid-round."}
                </p>
              </div>
            </div>

            <div className="interviewControls">
              <button
                className="interviewActionButton"
                type="button"
                onClick={() => interview.setMuted(!interview.isMuted)}
                aria-pressed={interview.isMuted}
              >
                <span aria-hidden="true">{interview.isMuted ? "◌" : "◉"}</span>
                {interview.isMuted ? "Unmute" : "Mute"}
              </button>
              {confirmingEnd ? (
                <span className="interviewEndConfirm" role="group" aria-label="Confirm ending the interview">
                  <span>
                    End before question five? It won&apos;t get a full score.
                  </span>
                  <button
                    className="interviewActionButton isStop"
                    type="button"
                    onClick={() => {
                      setConfirmingEnd(false);
                      interview.endEarly();
                    }}
                  >
                    End now
                  </button>
                  <button
                    className="interviewActionButton"
                    type="button"
                    onClick={() => setConfirmingEnd(false)}
                  >
                    Keep going
                  </button>
                </span>
              ) : (
                <button
                  className="interviewActionButton isStop"
                  type="button"
                  onClick={() => setConfirmingEnd(true)}
                >
                  <span aria-hidden="true">■</span> End interview
                </button>
              )}
            </div>

            <form
              className="interviewTypedRow"
              onSubmit={(event) => {
                event.preventDefault();
                sendTyped();
              }}
            >
              <input
                type="text"
                value={typedText}
                placeholder="Can't speak right now? Type your reply."
                aria-label="Type a reply to Chiron"
                onChange={(event) => setTypedText(event.target.value)}
              />
              <button
                className="interviewActionButton"
                type="submit"
                disabled={!typedText.trim()}
              >
                Send
              </button>
            </form>

            {interview.turns.length ? (
              <details className="interviewTranscriptPanel">
                <summary>Transcript</summary>
                <TranscriptList turns={interview.turns} />
              </details>
            ) : null}
          </div>
        </section>
      ) : null}

      {showScorecard && interview.scorecard ? (
        <ScorecardView
          scorecard={interview.scorecard}
          turns={interview.turns}
          onRetryRound={retryRound}
          onTryAnotherType={tryAnotherType}
          onNewRole={newRole}
        />
      ) : null}

      {showInterrupted ? (
        <section className="interviewInterrupted" aria-label="Interview interrupted">
          <h2>
            {interview.phase === "failed"
              ? "The interview didn’t finish"
              : "Ended early — not scored"}
          </h2>
          <p>
            {interview.phase === "failed"
              ? "The connection ended before Chiron submitted an evaluation. Nothing is invented in that case: there is no score, and your transcript is kept below."
              : "The round stopped before the fifth answer, so there is no score. To get a provisional read next time, tell Chiron you want to stop instead — after three answers it can still evaluate."}
          </p>
          <div className="interviewNextActions">
            <button className="primaryButton" type="button" onClick={retryRound}>
              Try again
            </button>
            <button className="secondaryButton" type="button" onClick={() => interview.reset()}>
              Back to setup
            </button>
          </div>
          {interview.turns.length ? (
            <details className="interviewTranscriptPanel" open>
              <summary>Transcript</summary>
              <TranscriptList turns={interview.turns} />
            </details>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}

function TranscriptList({ turns }: { turns: LiveInterviewTurn[] }) {
  return (
    <ol className="interviewTranscript">
      {turns.map((turn) => (
        <li key={turn.ordinal} className={`is-${turn.role}`}>
          <span>{turn.role === "agent" ? "Chiron" : "You"}</span>
          <p>{turn.text}</p>
        </li>
      ))}
    </ol>
  );
}

function ScorecardView({
  scorecard,
  turns,
  onRetryRound,
  onTryAnotherType,
  onNewRole,
}: {
  scorecard: InterviewScorecard;
  turns: LiveInterviewTurn[];
  onRetryRound: () => void;
  onTryAnotherType: () => void;
  onNewRole: () => void;
}) {
  const evaluation = scorecard.evaluation;
  const scored = scorecard.overall_score !== null;
  return (
    <section className="interviewScorecard" aria-label="Interview scorecard">
      <header className="interviewScoreHead">
        <div className="interviewScoreDial" aria-label={
          scored ? `Overall score ${scorecard.overall_score?.toFixed(1)} out of 10` : "Not scored"
        }>
          <strong>{scored ? scorecard.overall_score?.toFixed(1) : "—"}</strong>
          <span>/ 10</span>
        </div>
        <div className="interviewScoreSummary">
          <div className="interviewScoreBadges">
            <span
              className={`interviewRecommendation is-${
                scorecard.recommendation ?? "unscored"
              }`}
            >
              {recommendationLabel(scorecard.recommendation)}
            </span>
            {scorecard.provisional ? (
              <span className="interviewProvisional">
                Provisional — ended after {evaluation.completed_question_count} of 5
                questions
              </span>
            ) : null}
          </div>
          <p className="interviewVerdict">{evaluation.verdict}</p>
        </div>
      </header>

      {!scored ? (
        <p className="interviewUnscored">
          Fewer than three questions were answered, so this round gets no number.
          The notes below are qualitative only.
        </p>
      ) : null}

      <div className="interviewRubric">
        {RUBRIC_ROWS.map((row) => (
          <div className="interviewRubricRow" key={row.key}>
            <span>{row.label}</span>
            <div className="interviewRubricTrack" aria-hidden="true">
              <i style={{ width: `${evaluation[row.key] * 10}%` }} />
            </div>
            <strong>{evaluation[row.key]}</strong>
          </div>
        ))}
      </div>

      <div className="interviewScoreBlock">
        <h3>Strongest answer</h3>
        <blockquote>&ldquo;{evaluation.strongest_answer_quote}&rdquo;</blockquote>
        <p>{evaluation.strongest_answer_reason}</p>
      </div>

      <div className="interviewScoreBlock">
        <h3>The three problems that cost the most</h3>
        <ol className="interviewProblems">
          {evaluation.improvements.map((item, index) => (
            <li key={index}>
              <strong>{item.what_happened}</strong>
              <p className="interviewEvidence">&ldquo;{item.evidence}&rdquo;</p>
              <p>{item.why_it_hurt}</p>
              <p className="interviewBetter">{item.better_approach}</p>
            </li>
          ))}
        </ol>
      </div>

      <div className="interviewScoreBlock">
        <h3>Ten-minute drill</h3>
        <p>{evaluation.drill}</p>
      </div>

      <div className="interviewNextActions">
        <button className="primaryButton" type="button" onClick={onRetryRound}>
          Retry this round
        </button>
        <button className="secondaryButton" type="button" onClick={onTryAnotherType}>
          Try another interview type
        </button>
        <button className="secondaryButton" type="button" onClick={onNewRole}>
          New role
        </button>
      </div>

      {turns.length ? (
        <details className="interviewTranscriptPanel">
          <summary>Full transcript</summary>
          <TranscriptList turns={turns} />
        </details>
      ) : null}
    </section>
  );
}
