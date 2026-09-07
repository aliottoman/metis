"use client";

// The verdict: a number the backend computed, the rubric, the three
// problems that cost the most, a drill, and how it sounded.

import { Transcript } from "@/components/audio/stage";
import { useToast } from "@/components/ui/toast";
import type { LiveInterviewTurn } from "@/hooks/use-interview-session";
import { downloadText, elapsedLabel } from "@/lib/audio";
import { recommendationLabel } from "@/lib/interviews";
import type { InterviewDelivery, InterviewDeliveryStage, InterviewScorecard } from "@/lib/types";

type RubricKey = "specific_evidence" | "role_depth" | "relevance" | "structure" | "communication";
const RUBRIC: Array<[RubricKey, string]> = [["specific_evidence", "Specific evidence"], ["role_depth", "Role depth"], ["relevance", "Relevance"], ["structure", "Structure"], ["communication", "Communication"]];

function report(scorecard: InterviewScorecard, delivery: InterviewDelivery | null): string {
  const e = scorecard.evaluation;
  const lines = [
    `Metis interview scorecard`,
    `Overall: ${scorecard.overall_score === null ? "not scored" : `${scorecard.overall_score.toFixed(1)} / 10`} · ${recommendationLabel(scorecard.recommendation)}${scorecard.provisional ? " (provisional)" : ""}`,
    ``, e.verdict, ``,
    ...RUBRIC.map(([key, label]) => `${label}: ${e[key]}`),
    ``, `Strongest answer: "${e.strongest_answer_quote}"`, e.strongest_answer_reason, ``,
    ...e.improvements.flatMap((item, index) => [`${index + 1}. ${item.what_happened}`, `   "${item.evidence}"`, `   ${item.why_it_hurt}`, `   Better: ${item.better_approach}`]),
    ``, `Drill: ${e.drill}`,
  ];
  if (delivery) lines.push(``, `Delivery: ${delivery.filler_count} fillers, ${Math.round(delivery.words_per_minute)} wpm, ${delivery.long_pause_count} long pauses, ${delivery.hedging_count} hedges.`);
  return lines.join("\n");
}

function chips(breakdown: Record<string, number>): string[] {
  return Object.entries(breakdown).sort(([, a], [, b]) => b - a).map(([token, count]) => `${token} ×${count}`);
}

export function Delivery({ delivery, stage, busy, onRetry }: { delivery: InterviewDelivery | null; stage: InterviewDeliveryStage; busy: boolean; onRetry: () => void }) {
  return (
    <section className="interview-block" aria-label="Delivery">
      <h3>Delivery <small>measured from the recording</small></h3>
      {stage === "ready" && delivery ? (
        <>
          <div className="interview-delivery">
            <div><strong>{delivery.filler_count}</strong><span>fillers{delivery.candidate_word_count > 0 ? ` · ${delivery.filler_rate_per_100_words.toFixed(1)} per 100 words` : ""}</span>{delivery.filler_count > 0 ? <small>{chips(delivery.filler_breakdown).join(" · ")}</small> : null}</div>
            <div><strong>{Math.round(delivery.words_per_minute)}</strong><span>words a minute · {elapsedLabel(delivery.candidate_talk_seconds)} talking</span></div>
            <div><strong>{delivery.long_pause_count}</strong><span>long pauses{delivery.long_pause_count > 0 ? ` · longest ${delivery.longest_pause_seconds.toFixed(1)}s` : ""}</span></div>
            <div><strong>{delivery.hedging_count}</strong><span>hedges</span>{delivery.hedging_count > 0 ? <small>{chips(delivery.hedging_breakdown).join(" · ")}</small> : null}</div>
          </div>
          {delivery.note ? <p>{delivery.note}</p> : null}
          <small>Counted by verbatim speech-to-text over the call audio. These never change the score.</small>
        </>
      ) : stage === "failed" ? (
        <p>The delivery analysis did not finish. <button type="button" className="ui-btn is-sm" disabled={busy} onClick={onRetry}>{busy ? "Analyzing…" : "Analyze again"}</button></p>
      ) : stage === "unavailable" ? (
        <p>No recording was available for this round, so there are no delivery figures.</p>
      ) : (
        <p role="status">Measuring fillers, pace and pauses from the recording. This lands a moment after the call ends.</p>
      )}
    </section>
  );
}

export function Scorecard({ scorecard, turns, delivery, deliveryStage, deliveryBusy, onRetryDelivery, onRetryRound, onTryAnotherType, onNewRole }: {
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
  const toast = useToast();
  const e = scorecard.evaluation;
  const scored = scorecard.overall_score !== null;
  const text = report(scorecard, delivery);
  return (
    <section className="interview-scorecard" aria-label="Interview scorecard">
      <header className="interview-score-head">
        <div className="interview-dial" aria-label={scored ? `Overall score ${scorecard.overall_score?.toFixed(1)} out of 10` : "Not scored"}><strong>{scored ? scorecard.overall_score?.toFixed(1) : "—"}</strong><span>/ 10</span></div>
        <div>
          <span className={`ui-chip is-${scorecard.recommendation ?? "unscored"}`}>{recommendationLabel(scorecard.recommendation)}</span>
          {scorecard.provisional ? <span className="ui-chip">provisional · {e.completed_question_count} of 5</span> : null}
          <p className="interview-verdict">{e.verdict}</p>
          {!scored ? <small>Fewer than three questions were answered, so this round gets no number.</small> : null}
        </div>
        <span className="interview-score-actions">
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => void navigator.clipboard.writeText(text).then(() => toast("Scorecard copied"))}>Copy</button>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={() => { downloadText("metis-interview-scorecard.txt", text); toast("Scorecard exported"); }}>Export</button>
        </span>
      </header>

      <div className="interview-rubric">
        {RUBRIC.map(([key, label]) => <div key={key}><span>{label}</span><i aria-hidden="true"><b style={{ width: `${Number(e[key]) * 10}%` }} /></i><strong>{Number(e[key])}</strong></div>)}
      </div>

      <section className="interview-block"><h3>Strongest answer</h3><blockquote>“{e.strongest_answer_quote}”</blockquote><p>{e.strongest_answer_reason}</p></section>
      <section className="interview-block">
        <h3>The three problems that cost the most</h3>
        <ol className="interview-problems">
          {e.improvements.map((item, index) => <li key={index}><strong>{item.what_happened}</strong><blockquote>“{item.evidence}”</blockquote><p>{item.why_it_hurt}</p><p className="interview-better">{item.better_approach}</p></li>)}
        </ol>
      </section>
      <section className="interview-block"><h3>Ten-minute drill</h3><p>{e.drill}</p></section>
      <Delivery delivery={delivery} stage={deliveryStage} busy={deliveryBusy} onRetry={onRetryDelivery} />

      <div className="interview-next">
        <button type="button" className="ui-btn is-primary" onClick={onRetryRound}>Retry this round</button>
        <button type="button" className="ui-btn" onClick={onTryAnotherType}>Try another round</button>
        <button type="button" className="ui-btn" onClick={onNewRole}>New role</button>
      </div>

      <Transcript open={false} title="Full transcript" turns={turns.map((turn) => ({ id: String(turn.ordinal), speaker: turn.role === "agent" ? "Chiron" : "You", text: turn.text, agent: turn.role === "agent" }))} filename="metis-interview-transcript.txt" />
    </section>
  );
}
