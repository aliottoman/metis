"use client";

// What the meeting produced: the readout, what it suggests (each with the
// seconds that produced it), who spoke, and the log of how it was made.

import { Notice } from "@/components/ui/notice";
import { clock } from "@/lib/audio";
import type { MeetingDetail, MeetingProposal, MeetingStage } from "@/lib/types";

export const STAGE_LABEL: Record<MeetingStage, string> = {
  uploaded: "Queued",
  isolating: "Cleaning the audio",
  transcribing: "Transcribing",
  analyzing: "Reading it through",
  ready: "Ready",
  failed: "Failed",
};
const PIPELINE: Array<[MeetingStage, string]> = [["uploaded", "Stored"], ["isolating", "Cleaned"], ["transcribing", "Transcribed"], ["analyzing", "Analyzed"], ["ready", "Ready"]];
const PROPOSAL_LABEL: Record<MeetingProposal["kind"], string> = { account_link: "Account match", action: "Action item", decision: "Decision" };
const KEEP_LABEL: Record<MeetingProposal["kind"], string> = { account_link: "Link", action: "Keep action", decision: "Confirm" };

function proposalText(proposal: MeetingProposal): string {
  return proposal.payload.account_name ?? proposal.payload.description ?? proposal.payload.summary ?? "";
}

export function MeetingOverview({ detail, busy, onSeek, onDecide, onName, onRetry }: {
  detail: MeetingDetail;
  busy: string | null;
  onSeek: (seconds: number | null) => void;
  onDecide: (proposal: MeetingProposal, status: "accepted" | "rejected") => void;
  onName: (speakerId: string, name: string) => void;
  onRetry: () => void;
}) {
  const meeting = detail.meeting;
  const open = detail.proposals.filter((item) => item.status === "proposed");
  const reviewed = detail.proposals.filter((item) => item.status !== "proposed");
  const stageAt = Math.max(0, PIPELINE.findIndex(([stage]) => stage === meeting.stage));

  return (
    <div className="meeting-overview">
      {meeting.stage !== "ready" ? (
        <section className="meeting-card">
          <h3>{meeting.stage === "failed" ? "This recording needs attention" : STAGE_LABEL[meeting.stage]}</h3>
          <ol className="meeting-pipeline">
            {PIPELINE.map(([stage, label], index) => <li key={stage} className={index < stageAt ? "is-done" : index === stageAt ? (meeting.stage === "failed" ? "is-failed" : "is-active") : ""}>{label}</li>)}
          </ol>
          {meeting.error ? <Notice kind="error" action="Retry analysis" onAction={onRetry}>{meeting.error}</Notice> : <small>You can leave this page; it keeps going.</small>}
        </section>
      ) : null}

      {meeting.summary ? <section className="meeting-card"><h3>Readout</h3><p className="meeting-summary">{meeting.summary}</p></section> : null}

      <section className="meeting-card">
        <h3>Suggestions <small>{open.length ? `${open.length} to review` : "nothing open"}</small></h3>
        {!meeting.account_id && open.some((item) => item.kind === "action") ? <small className="meeting-note">Link a customer before keeping actions.</small> : null}
        {open.map((proposal) => (
          <article key={proposal.id} className="meeting-proposal">
            <span className="ui-chip">{PROPOSAL_LABEL[proposal.kind]}</span>
            <div>
              <p>{proposalText(proposal)}</p>
              {proposal.evidence_start !== null ? <button type="button" className="meeting-time" onClick={() => onSeek(proposal.evidence_start)}>▶ Evidence at {clock(proposal.evidence_start)}</button> : null}
            </div>
            <span>
              <button type="button" className="ui-btn is-primary is-sm" disabled={busy === proposal.id || (proposal.kind === "action" && !meeting.account_id)} onClick={() => onDecide(proposal, "accepted")}>{KEEP_LABEL[proposal.kind]}</button>
              <button type="button" className="ui-btn is-quiet is-sm" disabled={busy === proposal.id} onClick={() => onDecide(proposal, "rejected")} aria-label="Dismiss">Dismiss</button>
            </span>
          </article>
        ))}
        {reviewed.length ? (
          <details className="meeting-reviewed">
            <summary>{reviewed.length} decided</summary>
            {reviewed.map((proposal) => <p key={proposal.id}><span className="ui-chip">{proposal.status === "accepted" ? "kept" : "dismissed"}</span> {PROPOSAL_LABEL[proposal.kind]} · {proposalText(proposal)}</p>)}
          </details>
        ) : null}
      </section>

      <section className="meeting-card">
        <h3>Voices <small>name them once; every turn follows</small></h3>
        <div className="meeting-speakers">
          {detail.speakers.map((speaker, index) => (
            <label key={speaker.speaker_id} className={`ui-field tone-${index % 4}`}>
              <span>Voice {index + 1}</span>
              <input list="meeting-speaker-names" defaultValue={speaker.display_name || speaker.speaker_id} aria-label={`Name for ${speaker.speaker_id}`} onBlur={(event) => { const value = event.target.value.trim(); if (value && value !== speaker.display_name) onName(speaker.speaker_id, value); }} />
            </label>
          ))}
        </div>
        <datalist id="meeting-speaker-names">{detail.suggested_names.map((name) => <option key={name} value={name} />)}</datalist>
      </section>

      {detail.events.length ? (
        <details className="meeting-log">
          <summary>How this record was made</summary>
          <ol>{detail.events.map((event, index) => <li key={index} className={event.stage === "failed" ? "is-failed" : ""}><strong>{STAGE_LABEL[event.stage as MeetingStage] ?? event.stage}</strong><span>{event.message || "Stage completed"}</span><time>{new Date(event.created_at).toLocaleString()}</time></li>)}</ol>
        </details>
      ) : null}
    </div>
  );
}
