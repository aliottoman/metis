"use client";

// The interview room. The setup form is the last time the user types context;
// after Start the page is an orb to talk to, and the scorecard that replaces
// it holds a number the API calculated — never one the model, or this file,
// made up. No confetti lives here on purpose: a verdict is a verdict.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
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
  parseFocusAreas,
  recommendationLabel,
  saveInterviewDraft,
  validateInterviewDraft,
  type InterviewDraft,
} from "@/lib/interviews";
import type {
  InterviewAvailability,
  InterviewDelivery,
  InterviewDeliveryStage,
  InterviewScorecard,
} from "@/lib/types";
import {
  useInterviewSession,
  type InterviewPhase,
  type LiveInterviewTurn,
} from "@/hooks/use-interview-session";

const ORB_COLORS: [string, string] = ["#72528a", "#ff7759"];

const QUICK_FOCUS_AREAS = [
  "Specific evidence",
  "Architecture trade-offs",
  "Leadership judgment",
  "Failure recovery",
  "First 90 days",
  "Stakeholder communication",
] as const;

const PHASE_LABEL: Record<InterviewPhase, string> = {
  setup: "Fill in the role to begin",
  ready: "Ready when you are",
  connecting: "Connecting",
  listening: "Listening",
  agent_speaking: "Chiron is speaking",
  evaluating: "Scoring the interview",
  debriefing: "Your verdict is ready",
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

function orbState(phase: InterviewPhase, isSpeaking: boolean): ElevenLabsOrbState {
  if (phase === "listening") return "listening";
  if (phase === "agent_speaking") return "talking";
  if (phase === "debriefing") return isSpeaking ? "talking" : "listening";
  if (phase === "connecting" || phase === "evaluating") return "thinking";
  return null;
}

function roundLabel(value: string): string {
  return INTERVIEW_TYPES.find((round) => round.value === value)?.label ?? value;
}

function transcriptText(turns: LiveInterviewTurn[]): string {
  return turns
    .map((turn) => `${turn.role === "agent" ? "Chiron" : "You"}: ${turn.text}`)
    .join("\n\n");
}

function scorecardText(
  scorecard: InterviewScorecard,
  turns: LiveInterviewTurn[],
  delivery: InterviewDelivery | null,
): string {
  const evaluation = scorecard.evaluation;
  const lines = [
    "METIS INTERVIEW SCORECARD",
    "",
    `Overall: ${scorecard.overall_score === null ? "Not scored" : `${scorecard.overall_score.toFixed(1)} / 10`}`,
    `Recommendation: ${recommendationLabel(scorecard.recommendation)}`,
    scorecard.provisional
      ? `Provisional: ${evaluation.completed_question_count} of 5 questions completed`
      : "Round: Complete",
    "",
    "VERDICT",
    evaluation.verdict,
    "",
    "RUBRIC",
    ...RUBRIC_ROWS.map((row) => `${row.label}: ${evaluation[row.key]} / 10`),
    "",
    "STRONGEST ANSWER",
    `“${evaluation.strongest_answer_quote}”`,
    evaluation.strongest_answer_reason,
    "",
    "IMPROVEMENTS",
    ...evaluation.improvements.flatMap((item, index) => [
      `${index + 1}. ${item.what_happened}`,
      `Evidence: “${item.evidence}”`,
      `Why it hurt: ${item.why_it_hurt}`,
      `Better approach: ${item.better_approach}`,
      "",
    ]),
    "TEN-MINUTE DRILL",
    evaluation.drill,
  ];

  if (delivery) {
    lines.push(
      "",
      "DELIVERY",
      `Pace: ${Math.round(delivery.words_per_minute)} words per minute`,
      `Fillers: ${delivery.filler_count}`,
      `Long pauses: ${delivery.long_pause_count}`,
      `Hedges: ${delivery.hedging_count}`,
    );
    if (delivery.note) lines.push(delivery.note);
  }

  if (turns.length) {
    lines.push("", "TRANSCRIPT", transcriptText(turns));
  }
  return lines.join("\n");
}

function safeFilePart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "interview";
}

function downloadText(filename: string, contents: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function copyText(contents: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(contents);
    return;
  }
  const field = document.createElement("textarea");
  field.value = contents;
  field.readOnly = true;
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.append(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("Clipboard access is unavailable.");
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
  const endInterviewButtonRef = useRef<HTMLButtonElement>(null);
  const keepGoingButtonRef = useRef<HTMLButtonElement>(null);
  const restoreEndFocusRef = useRef(false);
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

  useLayoutEffect(() => {
    if (confirmingEnd) {
      // Cancellation is the safe default when this destructive choice opens.
      keepGoingButtonRef.current?.focus();
      return;
    }
    if (restoreEndFocusRef.current) {
      restoreEndFocusRef.current = false;
      endInterviewButtonRef.current?.focus();
    }
  }, [confirmingEnd]);

  const keepInterviewGoing = useCallback(() => {
    restoreEndFocusRef.current = true;
    setConfirmingEnd(false);
  }, []);

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
  const toggleQuickFocus = useCallback((area: string) => {
    setDraft((current) => {
      const selected = parseFocusAreas(current.focus_areas);
      const exists = selected.some(
        (item) => item.toLocaleLowerCase() === area.toLocaleLowerCase(),
      );
      const next = exists
        ? selected.filter(
            (item) => item.toLocaleLowerCase() !== area.toLocaleLowerCase(),
          )
        : selected.length < 6
          ? [...selected, area]
          : selected;
      return { ...current, focus_areas: next.join(", ") };
    });
    touch("focus_areas");
  }, [touch]);

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
  const focusAreas =
    interview.session?.focus_areas ?? parseFocusAreas(draft.focus_areas);
  const latestTurn = interview.turns.at(-1) ?? null;

  return (
    <div className="workspacePage interviewsPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Mock rounds</span>
          <h1>Interviews</h1>
          <p>
            Five questions, real follow-ups, and a delivery read from the
            recording. No rehearsed praise.
          </p>
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
        <form
          className="interviewSetup interviewLaunchpad"
          aria-label="Interview setup"
          onSubmit={(event) => {
            event.preventDefault();
            if (context && availability?.available !== false) begin();
          }}
        >
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

          <div className="interviewLaunchGrid">
            <div className="interviewLaunchEditor">
              <section className="interviewLaunchSection" aria-labelledby="interview-role-heading">
                <header className="interviewLaunchSectionHead">
                  <span className="interviewLaunchStep">01</span>
                  <div>
                    <h2 id="interview-role-heading">Build the room</h2>
                    <p>Give Chiron the exact role and source material to interview against.</p>
                  </div>
                </header>

                <div className="interviewFormGrid">
            <label className="interviewField">
              <span>Job title</span>
              <input
                id="interview-job-title"
                type="text"
                value={draft.job_title}
                maxLength={200}
                placeholder="Senior Data Engineer"
                onChange={(event) => update("job_title", event.target.value)}
                onBlur={() => touch("job_title")}
                aria-invalid={touched.job_title && Boolean(problems.job_title)}
                aria-describedby={touched.job_title && problems.job_title ? "interview-job-title-problem" : undefined}
              />
              {touched.job_title && problems.job_title ? (
                <small id="interview-job-title-problem" className="interviewFieldProblem" role="alert">
                  {problems.job_title}
                </small>
              ) : null}
            </label>
            <label className="interviewField">
              <span>Company</span>
              <input
                id="interview-company"
                type="text"
                value={draft.company_name}
                maxLength={200}
                placeholder="Batelco"
                onChange={(event) => update("company_name", event.target.value)}
                onBlur={() => touch("company_name")}
                aria-invalid={touched.company_name && Boolean(problems.company_name)}
                aria-describedby={touched.company_name && problems.company_name ? "interview-company-problem" : undefined}
              />
              {touched.company_name && problems.company_name ? (
                <small id="interview-company-problem" className="interviewFieldProblem" role="alert">
                  {problems.company_name}
                </small>
              ) : null}
            </label>
                </div>

          <label className="interviewField interviewDescriptionField">
            <span>Job description</span>
            <textarea
              id="interview-job-description"
              value={draft.job_description}
              rows={12}
              placeholder="Paste the whole job description. Chiron draws every question from it."
              onChange={(event) => update("job_description", event.target.value)}
              onBlur={() => touch("job_description")}
              aria-invalid={touched.job_description && Boolean(problems.job_description)}
              aria-describedby={touched.job_description && problems.job_description ? "interview-job-description-problem" : undefined}
            />
            {touched.job_description && problems.job_description ? (
              <small id="interview-job-description-problem" className="interviewFieldProblem" role="alert">
                {problems.job_description}
              </small>
            ) : null}
                </label>
              </section>

              <section className="interviewLaunchSection" aria-labelledby="interview-round-heading">
                <header className="interviewLaunchSectionHead">
                  <span className="interviewLaunchStep">02</span>
                  <div>
                    <h2 id="interview-round-heading">Choose the pressure</h2>
                    <p>Pick the interviewer lens Chiron should hold for all five questions.</p>
                  </div>
                </header>

                <div
                  className="interviewRounds"
                  role="radiogroup"
                  aria-label="Interview round"
                  aria-describedby={touched.interview_type && problems.interview_type ? "interview-type-problem" : undefined}
                >
            {INTERVIEW_TYPES.map((round) => (
              <label
                key={round.value}
                className={`interviewRoundCard ${
                  draft.interview_type === round.value ? "selected" : ""
                }`}
              >
                <input
                  className="visuallyHidden"
                  type="radio"
                  name="interview_type"
                  value={round.value}
                  checked={draft.interview_type === round.value}
                  onChange={() => {
                    update("interview_type", round.value);
                    touch("interview_type");
                  }}
                />
                <strong>{round.label}</strong>
                <span>{round.hint}</span>
              </label>
            ))}
                </div>
                {touched.interview_type && problems.interview_type ? (
                  <small id="interview-type-problem" className="interviewFieldProblem" role="alert">
                    {problems.interview_type}
                  </small>
                ) : null}
              </section>

              <section className="interviewLaunchSection" aria-labelledby="interview-focus-heading">
                <header className="interviewLaunchSectionHead">
                  <span className="interviewLaunchStep">03</span>
                  <div>
                    <h2 id="interview-focus-heading">Direct the follow-ups</h2>
                    <p>Add an objective or tap the areas where you want Chiron to push hardest.</p>
                  </div>
                </header>

                <div className="interviewSteering">
            <span className="interviewSteeringEyebrow">
              Make it specific · optional
            </span>
            <label className="interviewField interviewObjectiveField">
              <span>What should this round probe?</span>
              <textarea
                id="interview-objective"
                value={draft.interview_objective}
                rows={3}
                placeholder="e.g. Grill me on why I chose ElevenLabs over building my own voice stack, and press on every architecture trade-off."
                onChange={(event) => update("interview_objective", event.target.value)}
                onBlur={() => touch("interview_objective")}
                aria-invalid={touched.interview_objective && Boolean(problems.interview_objective)}
                aria-describedby={touched.interview_objective && problems.interview_objective ? "interview-objective-problem" : undefined}
              />
              {touched.interview_objective && problems.interview_objective ? (
                <small id="interview-objective-problem" className="interviewFieldProblem" role="alert">
                  {problems.interview_objective}
                </small>
              ) : null}
            </label>
            <div className="interviewQuickFocus" aria-label="Quick focus areas">
              <span>Quick add</span>
              <div className="interviewQuickFocusList">
                {QUICK_FOCUS_AREAS.map((area) => {
                  const selected = parseFocusAreas(draft.focus_areas).some(
                    (item) => item.toLocaleLowerCase() === area.toLocaleLowerCase(),
                  );
                  return (
                    <button
                      key={area}
                      className={`interviewQuickFocusChip ${selected ? "isSelected" : ""}`}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => toggleQuickFocus(area)}
                    >
                      <span aria-hidden="true">{selected ? "✓" : "+"}</span>
                      {area}
                    </button>
                  );
                })}
              </div>
            </div>
            <label className="interviewField">
              <span>Focus areas</span>
              <input
                id="interview-focus-areas"
                type="text"
                value={draft.focus_areas}
                placeholder="architecture trade-offs, why ElevenLabs, scaling — up to six, comma-separated"
                onChange={(event) => update("focus_areas", event.target.value)}
                onBlur={() => touch("focus_areas")}
                aria-invalid={touched.focus_areas && Boolean(problems.focus_areas)}
                aria-describedby={touched.focus_areas && problems.focus_areas ? "interview-focus-problem" : undefined}
              />
              {touched.focus_areas && problems.focus_areas ? (
                <small id="interview-focus-problem" className="interviewFieldProblem" role="alert">
                  {problems.focus_areas}
                </small>
              ) : null}
              {parseFocusAreas(draft.focus_areas).length ? (
                <span className="interviewFocusChips" aria-hidden="true">
                  {parseFocusAreas(draft.focus_areas).map((area, index) => (
                    <i key={`${index}-${area}`}>{area}</i>
                  ))}
                </span>
              ) : null}
                  </label>
                </div>
              </section>
            </div>

            <aside className="interviewSessionBrief" aria-label="Session brief">
              <span className="interviewBriefEyebrow">Session brief</span>
              <div className="interviewBriefRole">
                <strong>{draft.job_title.trim() || "Your target role"}</strong>
                <span>
                  {draft.company_name.trim() || "Company"} · {draft.interview_type
                    ? `${roundLabel(draft.interview_type)} round`
                    : "Choose a round"}
                </span>
              </div>
              <dl className="interviewBriefFacts">
                <div>
                  <dt>Format</dt>
                  <dd>5 questions + live follow-ups</dd>
                </div>
                <div>
                  <dt>Score</dt>
                  <dd>Full at 5 · provisional from 3</dd>
                </div>
                <div>
                  <dt>Delivery</dt>
                  <dd>Pace, fillers, hedging and pauses</dd>
                </div>
              </dl>
              {parseFocusAreas(draft.focus_areas).length ? (
                <div className="interviewBriefFocus">
                  <span>Focus</span>
                  <div>
                    {parseFocusAreas(draft.focus_areas).map((area) => (
                      <i key={area}>{area}</i>
                    ))}
                  </div>
                </div>
              ) : null}
              {draft.interview_objective.trim() ? (
                <blockquote className="interviewBriefObjective">
                  {draft.interview_objective.trim()}
                </blockquote>
              ) : null}
              <div className="interviewStartRow interviewBriefStart">
                <button
                  className="primaryButton interviewStartButton"
                  type="submit"
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
                  Chiron stays in character and scores only after the round.
                </span>
              </div>
            </aside>
          </div>
        </form>
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
                {focusAreas.length ? (
                  <span className="interviewLiveFocus" aria-label="Focus areas">
                    {focusAreas.map((area, index) => (
                      <i key={`${index}-${area}`}>{area}</i>
                    ))}
                  </span>
                ) : null}
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
                    agentState={orbState(interview.phase, interview.isSpeaking)}
                    getInputVolume={interview.getInputVolume}
                    getOutputVolume={interview.getOutputVolume}
                  />
                </div>
              )}
              <div className="interviewStageCopy">
                <strong>{PHASE_LABEL[interview.phase]}</strong>
                <p>
                  {interview.audioPaused
                    ? "Audio is paused locally. The live room remains open."
                    : interview.phase === "debriefing"
                      ? "Your score is ready. Stay for Chiron’s spoken debrief, or view it now."
                      : interview.phase === "evaluating"
                        ? "Chiron is scoring the round and preparing your verdict."
                        : interview.phase === "connecting"
                          ? "Setting up the room."
                          : "Answer out loud. Chiron won't coach you mid-round."}
                </p>
              </div>
            </div>

            <div className="interviewLiveContextGrid">
              <section className="interviewCurrentQuestion" aria-label="Current question">
                <span>Current question</span>
                <p>
                  {interview.latestQuestion ||
                    "Chiron’s opening question will stay pinned here once the round begins."}
                </p>
              </section>
              <section
                className={`interviewCaptionPanel ${latestTurn ? `is-${latestTurn.role}` : ""}`}
                aria-label="Live captions"
                aria-live="polite"
                aria-atomic="true"
              >
                <span>
                  {latestTurn
                    ? latestTurn.role === "agent"
                      ? "Chiron · live caption"
                      : "You · live caption"
                    : "Live captions"}
                </span>
                <p>{latestTurn?.text ?? "Conversation captions will appear here."}</p>
              </section>
            </div>

            <div className="interviewLiveUtilityBar">
              <div className="interviewAudioControls" aria-label="Audio controls">
                <button
                  className="interviewActionButton"
                  type="button"
                  onClick={interview.toggleAudioPaused}
                  aria-pressed={interview.audioPaused}
                >
                  <span aria-hidden="true">{interview.audioPaused ? "▶" : "Ⅱ"}</span>
                  {interview.audioPaused ? "Resume audio" : "Pause audio"}
                </button>
                <button
                  className="interviewActionButton"
                  type="button"
                  disabled={interview.audioPaused}
                  onClick={() => interview.setMuted(!interview.isMuted)}
                  aria-pressed={interview.isMuted}
                >
                  <span aria-hidden="true">{interview.isMuted ? "◌" : "◉"}</span>
                  {interview.isMuted ? "Unmute" : "Mute mic"}
                </button>
              </div>
              <label className="interviewVolumeControl">
                <span>Output volume</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={Math.round(interview.outputVolume * 100)}
                  aria-label={`Output volume ${Math.round(interview.outputVolume * 100)} percent`}
                  onChange={(event) =>
                    interview.setOutputVolume(Number(event.target.value) / 100)
                  }
                />
                <output>{Math.round(interview.outputVolume * 100)}%</output>
              </label>
            </div>

            <div className="interviewControls">
              {interview.phase === "debriefing" ? (
                <button
                  className="primaryButton interviewViewResultsButton"
                  type="button"
                  onClick={interview.viewResults}
                >
                  View results
                </button>
              ) : confirmingEnd ? (
                <div
                  className="interviewEndConfirm"
                  role="group"
                  aria-label="Choose how to finish the interview"
                  onKeyDown={(event) => {
                    if (event.key === "Escape") {
                      event.preventDefault();
                      keepInterviewGoing();
                    }
                  }}
                >
                  <span className="interviewEndConfirmCopy">
                    {interview.finishRequested
                      ? "Chiron already has your finish request. Confirm it in the conversation, or leave now without a score."
                      : interview.candidateAnswerCount >= 3
                        ? "You have enough answers for a provisional score. Finish through Chiron, or leave without one."
                        : "A score needs at least three answers. You can leave now without one, or keep going."}
                  </span>
                  {interview.canFinishAndScore ? (
                    <button
                      className="interviewActionButton isFinish"
                      type="button"
                      onClick={() => {
                        setConfirmingEnd(false);
                        interview.requestFinishAndScore();
                      }}
                    >
                      Finish &amp; score
                    </button>
                  ) : null}
                  <button
                    className="interviewActionButton isStop"
                    type="button"
                    onClick={() => {
                      setConfirmingEnd(false);
                      interview.endEarly();
                    }}
                  >
                    End without score
                  </button>
                  <button
                    ref={keepGoingButtonRef}
                    className="interviewActionButton"
                    type="button"
                    onClick={keepInterviewGoing}
                  >
                    Keep going
                  </button>
                </div>
              ) : (
                <button
                  ref={endInterviewButtonRef}
                  className="interviewActionButton isStop"
                  type="button"
                  onClick={() => setConfirmingEnd(true)}
                >
                  <span aria-hidden="true">■</span> End interview
                </button>
              )}
            </div>

            {interview.finishRequested ? (
              <p className="interviewFinishStatus" role="status" aria-live="polite">
                Finish requested. Chiron will ask for one spoken confirmation,
                then score the answers you completed.
              </p>
            ) : null}

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
                <TranscriptToolbar
                  turns={interview.turns}
                  filename={`${safeFilePart(jobTitle)}-interview-transcript.txt`}
                />
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
          delivery={interview.delivery}
          deliveryStage={interview.deliveryStage}
          deliveryBusy={interview.deliveryBusy}
          onRetryDelivery={() => void interview.retryDelivery()}
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
          {interview.phase === "ended_early" ? (
            // The backend still measures delivery for an early end — the
            // round wasn't scored, but how it sounded is still worth having.
            <DeliverySection
              delivery={interview.delivery}
              stage={interview.deliveryStage}
              busy={interview.deliveryBusy}
              onRetry={() => void interview.retryDelivery()}
            />
          ) : null}
          {interview.turns.length ? (
            <details className="interviewTranscriptPanel" open>
              <summary>Transcript</summary>
              <TranscriptToolbar
                turns={interview.turns}
                filename={`${safeFilePart(jobTitle)}-interview-transcript.txt`}
              />
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

function TranscriptToolbar({
  turns,
  filename,
}: {
  turns: LiveInterviewTurn[];
  filename: string;
}) {
  const [notice, setNotice] = useState("");
  const contents = transcriptText(turns);
  return (
    <div className="interviewTranscriptToolbar">
      <div className="interviewExportActions">
        <button
          className="secondaryButton interviewExportButton"
          type="button"
          onClick={() => {
            void copyText(contents)
              .then(() => setNotice("Transcript copied"))
              .catch(() => setNotice("Clipboard unavailable — use Export instead"));
          }}
        >
          Copy transcript
        </button>
        <button
          className="secondaryButton interviewExportButton"
          type="button"
          onClick={() => {
            downloadText(filename, contents);
            setNotice("Transcript exported");
          }}
        >
          Export .txt
        </button>
      </div>
      <span className="interviewExportStatus" role="status" aria-live="polite">
        {notice}
      </span>
    </div>
  );
}

function ScorecardView({
  scorecard,
  turns,
  delivery,
  deliveryStage,
  deliveryBusy,
  onRetryDelivery,
  onRetryRound,
  onTryAnotherType,
  onNewRole,
}: {
  scorecard: InterviewScorecard;
  turns: LiveInterviewTurn[];
  delivery: InterviewDelivery | null;
  deliveryStage: InterviewDeliveryStage;
  deliveryBusy: boolean;
  onRetryDelivery: () => void;
  onRetryRound: () => void;
  onTryAnotherType: () => void;
  onNewRole: () => void;
}) {
  const evaluation = scorecard.evaluation;
  const scored = scorecard.overall_score !== null;
  const [exportNotice, setExportNotice] = useState("");
  const report = scorecardText(scorecard, turns, delivery);
  return (
    <section className="interviewScorecard" aria-label="Interview scorecard">
      <div className="interviewScorecardToolbar">
        <span>Shareable report</span>
        <div className="interviewExportActions">
          <button
            className="secondaryButton interviewExportButton"
            type="button"
            onClick={() => {
              void copyText(report)
                .then(() => setExportNotice("Scorecard copied"))
                .catch(() => setExportNotice("Clipboard unavailable — use Export instead"));
            }}
          >
            Copy scorecard
          </button>
          <button
            className="secondaryButton interviewExportButton"
            type="button"
            onClick={() => {
              downloadText("metis-interview-scorecard.txt", report);
              setExportNotice("Scorecard exported");
            }}
          >
            Export report
          </button>
        </div>
        <span className="interviewExportStatus" role="status" aria-live="polite">
          {exportNotice}
        </span>
      </div>
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

      <DeliverySection
        delivery={delivery}
        stage={deliveryStage}
        busy={deliveryBusy}
        onRetry={onRetryDelivery}
      />

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
          <TranscriptToolbar
            turns={turns}
            filename="metis-interview-transcript.txt"
          />
          <TranscriptList turns={turns} />
        </details>
      ) : null}
    </section>
  );
}

/** Chips like "um ×9", loudest habit first. */
function breakdownChips(breakdown: Record<string, number>): string[] {
  return Object.entries(breakdown)
    .sort(([, a], [, b]) => b - a)
    .map(([token, count]) => `${token} ×${count}`);
}

function DeliverySection({
  delivery,
  stage,
  busy,
  onRetry,
}: {
  delivery: InterviewDelivery | null;
  stage: InterviewDeliveryStage;
  busy: boolean;
  onRetry: () => void;
}) {
  return (
    <div className="interviewScoreBlock interviewDelivery" aria-label="Delivery analysis">
      <h3>
        Delivery
        <span className="interviewDeliveryTag">measured from the recording</span>
      </h3>

      {stage === "ready" && delivery ? (
        <>
          <div className="interviewDeliveryGrid">
            <div className="interviewDeliveryStat">
              <strong>{delivery.filler_count}</strong>
              <span>
                fillers
                {delivery.candidate_word_count > 0
                  ? ` · ${delivery.filler_rate_per_100_words.toFixed(1)} per 100 words`
                  : ""}
              </span>
              {delivery.filler_count > 0 ? (
                <span className="interviewDeliveryChips">
                  {breakdownChips(delivery.filler_breakdown).map((chip) => (
                    <i key={chip}>{chip}</i>
                  ))}
                </span>
              ) : null}
            </div>
            <div className="interviewDeliveryStat">
              <strong>{Math.round(delivery.words_per_minute)}</strong>
              <span>
                words a minute · {elapsedLabel(delivery.candidate_talk_seconds)} of
                talking
              </span>
            </div>
            <div className="interviewDeliveryStat">
              <strong>{delivery.long_pause_count}</strong>
              <span>
                long pauses
                {delivery.long_pause_count > 0
                  ? ` · longest ${delivery.longest_pause_seconds.toFixed(1)}s`
                  : " inside your answers"}
              </span>
            </div>
            <div className="interviewDeliveryStat">
              <strong>{delivery.hedging_count}</strong>
              <span>hedges</span>
              {delivery.hedging_count > 0 ? (
                <span className="interviewDeliveryChips">
                  {breakdownChips(delivery.hedging_breakdown).map((chip) => (
                    <i key={chip}>{chip}</i>
                  ))}
                </span>
              ) : null}
            </div>
          </div>
          {delivery.note ? (
            <p className="interviewDeliveryNote">{delivery.note}</p>
          ) : null}
          <p className="interviewDeliveryFoot">
            Counted by verbatim speech-to-text over the call audio. These numbers
            never change the overall score you already heard.
          </p>
        </>
      ) : null}

      {stage === "" || stage === "pending" ? (
        <p className="interviewDeliveryPending" role="status">
          Measuring your delivery from the recording — fillers, pace, pauses.
          This lands a moment after the call ends.
        </p>
      ) : null}

      {stage === "unavailable" ? (
        <p className="interviewDeliveryNote">
          No recording was available for this round, so there are no delivery
          metrics. The scorecard above is unaffected.
        </p>
      ) : null}

      {stage === "failed" ? (
        <div className="interviewDeliveryFailed">
          <p>The delivery analysis didn&apos;t finish.</p>
          <button
            className="secondaryButton"
            type="button"
            disabled={busy}
            onClick={onRetry}
          >
            {busy ? "Analyzing…" : "Analyze again"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
