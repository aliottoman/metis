"use client";

// The Meetings workbench — a list of recordings on the left, one transcript on
// the right, and the audio underneath both.
//
// The transcript is the point, so it gets the space. Words highlight as the
// audio plays and clicking one seeks to it, which is what makes a two-hour
// recording navigable: you scan the text, you click the sentence you half
// remember, and you hear it. Everything the meeting suggests stays a proposal
// with the seconds that produced it, so accepting one is a four-second check
// rather than an act of faith.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  correctMeetingTurn,
  decideMeetingProposal,
  getMeeting,
  listMeetings,
  meetingAudioUrl,
  nameMeetingSpeaker,
  retryMeeting,
  uploadMeeting,
} from "@/lib/api";
import type {
  Meeting,
  MeetingDetail,
  MeetingProposal,
  MeetingStage,
  MeetingTurn,
} from "@/lib/types";

/** What each stage is called while you are waiting for it. */
const STAGE_LABEL: Record<MeetingStage, string> = {
  uploaded: "Queued",
  isolating: "Cleaning the audio",
  transcribing: "Transcribing",
  analyzing: "Reading it through",
  ready: "Ready",
  failed: "Failed",
};

/** Stages that are still moving, and so worth polling for. */
const WORKING: ReadonlySet<MeetingStage> = new Set([
  "uploaded",
  "isolating",
  "transcribing",
  "analyzing",
]);

function clock(seconds: number | null | undefined): string {
  const whole = Math.max(0, Math.floor(seconds ?? 0));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function meetingDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Recently"
    : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

function fileSize(bytes: number): string {
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
}

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "S";
}

const PROPOSAL_LABEL: Record<MeetingProposal["kind"], string> = {
  account_link: "Account match",
  action: "Action item",
  decision: "Decision",
};

export function MeetingsWorkbench() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [position, setPosition] = useState(0);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const playerRef = useRef<HTMLAudioElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refreshList = useCallback(async () => {
    try {
      const rows = await listMeetings();
      setMeetings(rows);
      setSelected((current) => current ?? rows[0]?.id ?? null);
    } catch (listError) {
      setError(listError instanceof Error ? listError.message : "Could not read meetings.");
    }
  }, []);

  const refreshDetail = useCallback(async (meetingId: string) => {
    try {
      setDetail(await getMeeting(meetingId));
    } catch {
      setDetail(null);
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  useEffect(() => {
    if (selected) void refreshDetail(selected);
  }, [selected, refreshDetail]);

  // Poll only while something is actually moving. A ready meeting is a static
  // document and polling it forever is a spinner that means nothing.
  const stage = detail?.meeting.stage;
  useEffect(() => {
    if (!selected || !stage || !WORKING.has(stage)) return;
    const timer = window.setInterval(() => {
      void refreshDetail(selected);
      void refreshList();
    }, 3_000);
    return () => window.clearInterval(timer);
  }, [selected, stage, refreshDetail, refreshList]);

  async function upload(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const meeting = await uploadMeeting(file);
      await refreshList();
      setSelected(meeting.id);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "That upload failed.");
    } finally {
      setUploading(false);
    }
  }

  const speakerNames = useMemo(() => {
    const named = new Map<string, string>();
    detail?.speakers.forEach((speaker) =>
      named.set(speaker.speaker_id, speaker.display_name),
    );
    return named;
  }, [detail]);

  function seek(seconds: number | null) {
    const player = playerRef.current;
    if (!player || seconds === null) return;
    player.currentTime = Math.max(0, seconds);
    void player.play().catch(() => undefined);
  }

  async function saveCorrection(turn: MeetingTurn) {
    if (!selected || !draft.trim() || draft.trim() === turn.text) {
      setEditing(null);
      return;
    }
    try {
      await correctMeetingTurn(selected, turn.id, draft.trim());
      await refreshDetail(selected);
    } catch (correctionError) {
      setError(
        correctionError instanceof Error ? correctionError.message : "Could not save that.",
      );
    } finally {
      setEditing(null);
    }
  }

  async function decide(proposal: MeetingProposal, status: "accepted" | "rejected") {
    if (!selected) return;
    try {
      await decideMeetingProposal(selected, proposal.id, status);
      await refreshDetail(selected);
    } catch (decideError) {
      setError(decideError instanceof Error ? decideError.message : "Could not save that.");
    }
  }

  const open = detail?.proposals.filter((item) => item.status === "proposed") ?? [];
  const readyCount = meetings.filter((meeting) => meeting.stage === "ready").length;
  const totalSeconds = meetings.reduce((sum, meeting) => sum + (meeting.duration_seconds ?? 0), 0);
  const totalHours = totalSeconds >= 3600
    ? `${(totalSeconds / 3600).toFixed(totalSeconds >= 36_000 ? 0 : 1)}h`
    : clock(totalSeconds);

  return (
    <div className="workspacePage meetingsPage">
      <header className="pageHeader meetingHero">
        <div className="meetingHeroCopy">
          <span className="eyebrow">Conversation intelligence</span>
          <h1>Every meeting,<br /><em>made useful.</em></h1>
          <p>Metis turns recordings into a navigable transcript, named speakers, and evidence-linked actions. You decide what becomes part of the record.</p>
        </div>
        <div className="meetingHeroAside">
          <div className="meetingHeroOrb" aria-hidden="true"><i /><i /><i /></div>
          <button
            className="primaryButton"
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
          >
            <span aria-hidden="true">＋</span>{uploading ? "Uploading…" : "Add recording"}
          </button>
        </div>
        <div className="meetingHeroStats" aria-label="Meeting library summary">
          <span><strong>{meetings.length}</strong><small>recordings</small></span>
          <span><strong>{readyCount}</strong><small>ready to review</small></span>
          <span><strong>{totalHours}</strong><small>captured</small></span>
        </div>
      </header>

      {error ? (
        <div className="composerError" role="alert">
          <span>!</span>
          <p>{error}</p>
          <button className="errorDismiss" type="button" aria-label="Dismiss" onClick={() => setError(null)}>×</button>
        </div>
      ) : null}

      <input
        ref={fileRef}
        type="file"
        hidden
        accept="audio/*,video/*"
        onChange={(event) => void upload(event.target.files)}
      />

      <div className="meetingsLayout meetingStudio">
        <aside
          className={`meetingsList ${dragging ? "isDragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            void upload(event.dataTransfer.files);
          }}
        >
          <div className="meetingLibraryHead">
            <div><span className="eyebrow">Library</span><strong>{meetings.length} recording{meetings.length === 1 ? "" : "s"}</strong></div>
            <button type="button" onClick={() => fileRef.current?.click()} aria-label="Add a recording">＋</button>
          </div>
          <div className="meetingLibraryRows">
            {meetings.length ? (
              meetings.map((meeting, meetingIndex) => (
                <button
                  key={meeting.id}
                  type="button"
                  className={`meetingRow ${selected === meeting.id ? "selected" : ""}`}
                  onClick={() => setSelected(meeting.id)}
                >
                  <span className={`meetingRowGlyph tone-${meetingIndex % 4}`} aria-hidden="true">
                    {[3, 7, 11, 6, 13, 9, 4].map((height, index) => <i key={index} style={{ height }} />)}
                  </span>
                  <span className="meetingRowCopy">
                    <strong>{meeting.title || meeting.audio_filename}</strong>
                    <small>{meetingDate(meeting.created_at)} · {fileSize(meeting.audio_bytes)}</small>
                  </span>
                  <span className={`meetingStage is-${meeting.stage}`}>
                    {meeting.stage === "ready" ? clock(meeting.duration_seconds) : STAGE_LABEL[meeting.stage]}
                  </span>
                </button>
              ))
            ) : (
              <button className="meetingDropEmpty" type="button" onClick={() => fileRef.current?.click()}>
                <span aria-hidden="true">↥</span>
                <strong>Drop a recording</strong>
                <small>Audio or video · Metis handles the rest</small>
              </button>
            )}
          </div>
          {meetings.length ? <p className="meetingDropHint">Drop audio anywhere in this column to add it.</p> : null}
        </aside>

        <section className="meetingDetail">
          {!detail ? (
            <div className="meetingDetailEmpty">
              <span aria-hidden="true">◌</span>
              <strong>Select a recording</strong>
              <p>Audio, insights, and the full transcript will appear here.</p>
            </div>
          ) : (
            <>
              <div className="meetingHead">
                <div className="meetingTitleBlock">
                  <span className="eyebrow">Now reviewing</span>
                  <h2>{detail.meeting.title || detail.meeting.audio_filename}</h2>
                  <p>
                    {meetingDate(detail.meeting.created_at)}
                    {detail.meeting.duration_seconds ? ` · ${clock(detail.meeting.duration_seconds)}` : ""}
                    {detail.meeting.language ? ` · ${detail.meeting.language.toUpperCase()}` : ""}
                    {` · ${detail.speakers.length} speaker${detail.speakers.length === 1 ? "" : "s"}`}
                  </p>
                </div>
                <span className={`meetingStage meetingStagePill is-${detail.meeting.stage}`}>
                  <i aria-hidden="true" />{STAGE_LABEL[detail.meeting.stage]}
                </span>
                {detail.meeting.stage === "failed" ? (
                  <button
                    className="secondaryButton"
                    type="button"
                    onClick={() => void retryMeeting(detail.meeting.id).then(() => refreshDetail(detail.meeting.id))}
                  >
                    Retry
                  </button>
                ) : null}
              </div>

              {detail.meeting.stage !== "ready" ? (
                <ol className="meetingProgress">
                  {detail.events.map((event, index) => (
                    <li key={`${event.created_at}-${index}`}>
                      <b>{STAGE_LABEL[event.stage as MeetingStage] ?? event.stage}</b>
                      {event.message ? <span>{event.message}</span> : null}
                    </li>
                  ))}
                </ol>
              ) : null}

              {detail.meeting.error ? (
                <p className="mutedMeta" role="alert">{detail.meeting.error}</p>
              ) : null}

              <div className="meetingPlayerDock">
                <span className="meetingPlayGlyph" aria-hidden="true">▶</span>
                <div className="meetingWave" aria-hidden="true">
                  {[8, 14, 22, 13, 27, 18, 31, 17, 24, 11, 29, 20, 12, 25, 16, 30, 19, 9, 21, 13, 27, 16, 8].map((height, index) => (
                    <i key={index} style={{ height }} />
                  ))}
                </div>
                <audio
                  ref={playerRef}
                  className="meetingPlayer"
                  controls
                  preload="metadata"
                  src={meetingAudioUrl(detail.meeting.id)}
                  onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
                />
              </div>

              {open.length ? (
                <section className="meetingInsights">
                  <header><div><span className="eyebrow">Metis noticed</span><h3>{open.length} item{open.length === 1 ? "" : "s"} worth reviewing</h3></div><small>Nothing is filed automatically</small></header>
                  <ul className="meetingProposals">
                    {open.map((proposal) => (
                      <li key={proposal.id} className={`is-${proposal.kind}`}>
                        <span className="meetingProposalIcon" aria-hidden="true">
                          {proposal.kind === "action" ? "↗" : proposal.kind === "decision" ? "◆" : "◎"}
                        </span>
                        <div>
                          <span className="meetingProposalKind">{PROPOSAL_LABEL[proposal.kind]}</span>
                          <p>{proposal.payload.account_name ?? proposal.payload.description ?? proposal.payload.summary}</p>
                          {proposal.evidence_start !== null ? (
                            <button className="meetingEvidence" type="button" onClick={() => seek(proposal.evidence_start)}>
                              <span aria-hidden="true">▶</span> Hear the evidence · {clock(proposal.evidence_start)}
                            </button>
                          ) : null}
                        </div>
                        <div className="meetingProposalActions">
                          <button className="proposalAccept" type="button" onClick={() => void decide(proposal, "accepted")}>Accept</button>
                          <button className="proposalDismiss" type="button" onClick={() => void decide(proposal, "rejected")} aria-label="Dismiss proposal">×</button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </section>
              ) : null}

              <section className="meetingTranscriptSection">
                <header>
                  <div><span className="eyebrow">Transcript</span><h3>The conversation</h3></div>
                  <small>Click a word to hear it · double-click text to correct</small>
                </header>
                <ol className="meetingTranscript">
                  {detail.turns.map((turn) => {
                    const speakerName = speakerNames.get(turn.speaker_id) || turn.speaker_id;
                    return (
                      <li key={turn.id} className={`${turn.corrected_at ? "isCorrected " : ""}speakerTone-${turn.ordinal % 4}`}>
                        <span className="meetingSpeakerAvatar" aria-hidden="true">{initials(speakerName)}</span>
                        <div className="meetingTurnBody">
                          <div className="meetingTurnHead">
                            <input
                              className="meetingSpeaker"
                              list="meeting-speaker-names"
                              defaultValue={speakerName}
                              aria-label={`Name for ${turn.speaker_id}`}
                              onBlur={(event) => {
                                const value = event.target.value.trim();
                                if (!selected || !value || value === speakerNames.get(turn.speaker_id)) return;
                                void nameMeetingSpeaker(selected, turn.speaker_id, value).then(setDetail);
                              }}
                            />
                            <button className="meetingTimestamp" type="button" onClick={() => seek(turn.start_seconds)}>
                              {clock(turn.start_seconds)}
                            </button>
                          </div>
                          {editing === turn.id ? (
                            <textarea
                              className="meetingEditor"
                              autoFocus
                              value={draft}
                              onChange={(event) => setDraft(event.target.value)}
                              onBlur={() => void saveCorrection(turn)}
                              onKeyDown={(event) => {
                                if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void saveCorrection(turn); }
                                if (event.key === "Escape") setEditing(null);
                              }}
                            />
                          ) : (
                            <p
                              className="meetingTurnText"
                              onDoubleClick={() => { setEditing(turn.id); setDraft(turn.text); }}
                              title="Double-click to correct"
                            >
                              {turn.words.length ? turn.words.map((word, index) => (
                                <span
                                  key={`${turn.id}-${index}`}
                                  className={word.start !== null && word.end !== null && position >= word.start && position <= word.end ? "isSpoken" : ""}
                                  onClick={() => seek(word.start)}
                                >{word.text}{" "}</span>
                              )) : turn.text}
                            </p>
                          )}
                          {turn.corrected_at ? <span className="mutedMeta">Corrected · originally “{turn.original_text}”</span> : null}
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </section>

              <datalist id="meeting-speaker-names">
                {detail.suggested_names.map((name) => (
                  <option key={name} value={name} />
                ))}
              </datalist>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
