"use client";

// What the meeting produced: the readout, what it suggests (each with the
// seconds that produced it), who spoke, and the log of how it was made.

import { useState } from "react";
import { CheckCircle2, ListChecks, MessageSquareText, Play, Users } from "lucide-react";

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

function SpeakerName({ speakerId, name, index, busy, onName }: { speakerId: string; name: string; index: number; busy: boolean; onName: (id: string, name: string) => Promise<boolean> }) {
  const [draft, setDraft] = useState<string | null>(null);
  const value = draft ?? name;
  const changed = draft !== null && draft.trim() !== name;
  return <form className={`meeting-speaker-editor tone-${index % 4}`} onSubmit={(event) => { event.preventDefault(); if (!busy && value.trim() && changed) void onName(speakerId, value.trim()).then((saved) => { if (saved) setDraft(null); }); }}>
    <label className="ui-field"><span>Speaker {index + 1}</span><input list="meeting-speaker-names" value={value} placeholder={`Name speaker ${index + 1}`} disabled={busy} onChange={(event) => setDraft(event.target.value)} aria-label={`Name speaker ${index + 1}`} /></label>
    {changed ? <div><button type="button" className="ui-btn is-quiet is-sm" disabled={busy} onClick={() => setDraft(null)}>Cancel</button><button type="submit" className="ui-btn is-sm" disabled={busy || !value.trim()}>{busy ? "Saving…" : "Save name"}</button></div> : null}
  </form>;
}

export function MeetingOverview({ detail, busy, onSeek, onDecide, onName, onRetry }: {
  detail: MeetingDetail;
  busy: string | null;
  onSeek: (seconds: number | null) => void;
  onDecide: (proposal: MeetingProposal, status: "accepted" | "rejected") => void;
  onName: (speakerId: string, name: string) => Promise<boolean>;
  onRetry: () => void;
}) {
  const meeting = detail.meeting;
  const open = detail.proposals.filter((item) => item.status === "proposed");
  const reviewed = detail.proposals.filter((item) => item.status !== "proposed");
  const stageAt = Math.max(0, PIPELINE.findIndex(([stage]) => stage === (meeting.stage === "failed" ? meeting.failed_stage : meeting.stage)));

  return (
    <div className="meeting-overview">
      {meeting.stage !== "ready" ? (
        <section className="meeting-card">
          <h3>{meeting.stage === "failed" ? "This recording needs attention" : STAGE_LABEL[meeting.stage]}</h3>
          <ol className="meeting-pipeline">
            {PIPELINE.map(([stage, label], index) => <li key={stage} className={index < stageAt ? "is-done" : index === stageAt ? (meeting.stage === "failed" ? "is-failed" : "is-active") : ""}>{label}</li>)}
          </ol>
          {meeting.error ? <Notice kind="error" action={busy === "retry" ? "Retrying…" : "Try again"} onAction={onRetry}>{meeting.error}</Notice> : <small>You can leave this page; it keeps going.</small>}
        </section>
      ) : null}

      {meeting.summary ? <section className="meeting-card meeting-readout"><h3><MessageSquareText size={16} aria-hidden="true" />At a glance</h3><p className="meeting-summary">{meeting.summary}</p></section> : null}

      <section className="meeting-card">
        <h3><ListChecks size={16} aria-hidden="true" />Next steps <small>{open.length ? `${open.length} to review` : meeting.stage === "ready" ? "up to date" : "preparing"}</small></h3>
        <p className="meeting-section-description">Review the suggestions before they become part of a customer's record.</p>
        {!meeting.account_id && open.some((item) => item.kind === "action") ? <small className="meeting-note">Link a customer before keeping actions.</small> : null}
        {open.map((proposal) => (
          <article key={proposal.id} className="meeting-proposal">
            <span className="ui-chip">{PROPOSAL_LABEL[proposal.kind]}</span>
            <div>
              <p>{proposalText(proposal)}</p>
              {proposal.evidence_start !== null ? <button type="button" className="meeting-time" onClick={() => onSeek(proposal.evidence_start)}><Play size={11} aria-hidden="true" />Listen at {clock(proposal.evidence_start)}</button> : null}
            </div>
            <span>
              <button type="button" className="ui-btn is-primary is-sm" disabled={busy !== null || (proposal.kind === "action" && !meeting.account_id)} onClick={() => onDecide(proposal, "accepted")}>{busy === proposal.id ? "Saving…" : KEEP_LABEL[proposal.kind]}</button>
              <button type="button" className="ui-btn is-quiet is-sm" disabled={busy !== null} onClick={() => onDecide(proposal, "rejected")} aria-label={`Dismiss ${PROPOSAL_LABEL[proposal.kind].toLowerCase()}: ${proposalText(proposal)}`}>Dismiss</button>
            </span>
          </article>
        ))}
        {!open.length ? <p className="meeting-review-clear"><CheckCircle2 size={16} aria-hidden="true" />{meeting.stage === "ready" ? reviewed.length ? "Every suggestion has been reviewed." : "No suggested next steps in this recording." : "Suggestions will appear here when the recording is ready."}</p> : null}
        {reviewed.length ? (
          <details className="meeting-reviewed">
            <summary>{reviewed.length} decided</summary>
            {reviewed.map((proposal) => <p key={proposal.id}><span className="ui-chip">{proposal.status === "accepted" ? "kept" : "dismissed"}</span> {PROPOSAL_LABEL[proposal.kind]} · {proposalText(proposal)}</p>)}
          </details>
        ) : null}
      </section>

      <section className="meeting-card">
        <h3><Users size={16} aria-hidden="true" />Speakers</h3><p className="meeting-section-description">Add a name once. It follows the speaker throughout the transcript.</p>
        <div className="meeting-speakers">
          {detail.speakers.map((speaker, index) => (
            <SpeakerName key={speaker.speaker_id} speakerId={speaker.speaker_id} name={speaker.display_name} index={index} busy={busy !== null} onName={onName} />
          ))}
        </div>
        {!detail.speakers.length ? <p className="meeting-section-description">Speakers appear after transcription.</p> : null}
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
