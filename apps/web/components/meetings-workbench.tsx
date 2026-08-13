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

  return (
    <div className="workspacePage meetingsPage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Recordings</span>
          <h1>Meetings</h1>
          <p>
            Drop a recording in. Metis transcribes it with speakers and word timings,
            then suggests what it might mean — nothing is filed until you say so.
          </p>
        </div>
        <button
          className="primaryButton"
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? "Uploading…" : "Add a recording"}
        </button>
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

      <div className="meetingsLayout">
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
          {meetings.length ? (
            meetings.map((meeting) => (
              <button
                key={meeting.id}
                type="button"
                className={`meetingRow ${selected === meeting.id ? "selected" : ""}`}
                onClick={() => setSelected(meeting.id)}
              >
                <strong>{meeting.title || meeting.audio_filename}</strong>
                <span className={`meetingStage is-${meeting.stage}`}>
                  {STAGE_LABEL[meeting.stage]}
                  {meeting.duration_seconds ? ` · ${clock(meeting.duration_seconds)}` : ""}
                </span>
              </button>
            ))
          ) : (
            <p className="mutedMeta">Drop a recording here, or use the button above.</p>
          )}
        </aside>

        <section className="meetingDetail">
          {!detail ? (
            <p className="mutedMeta">Select a recording.</p>
          ) : (
            <>
              <div className="meetingHead">
                <h2>{detail.meeting.title || detail.meeting.audio_filename}</h2>
                <span className={`meetingStage is-${detail.meeting.stage}`}>
                  {STAGE_LABEL[detail.meeting.stage]}
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

              {/* Every stage it passed through, so a slow import is legible
                  rather than a spinner with nothing behind it. */}
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

              <audio
                ref={playerRef}
                className="meetingPlayer"
                controls
                preload="metadata"
                src={meetingAudioUrl(detail.meeting.id)}
                onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
              />

              {open.length ? (
                <ul className="meetingProposals">
                  {open.map((proposal) => (
                    <li key={proposal.id}>
                      <span className="meetingProposalKind">{proposal.kind.replace("_", " ")}</span>
                      <p>
                        {proposal.payload.account_name ??
                          proposal.payload.description ??
                          proposal.payload.summary}
                      </p>
                      {proposal.evidence_start !== null ? (
                        <button className="textButton" type="button" onClick={() => seek(proposal.evidence_start)}>
                          Hear it ({clock(proposal.evidence_start)})
                        </button>
                      ) : null}
                      <button className="textButton" type="button" onClick={() => void decide(proposal, "accepted")}>Accept</button>
                      <button className="textButton" type="button" onClick={() => void decide(proposal, "rejected")}>Dismiss</button>
                    </li>
                  ))}
                </ul>
              ) : null}

              <ol className="meetingTranscript">
                {detail.turns.map((turn) => (
                  <li key={turn.id} className={turn.corrected_at ? "isCorrected" : ""}>
                    <div className="meetingTurnHead">
                      <input
                        className="meetingSpeaker"
                        list="meeting-speaker-names"
                        defaultValue={speakerNames.get(turn.speaker_id) || turn.speaker_id}
                        aria-label={`Name for ${turn.speaker_id}`}
                        onBlur={(event) => {
                          const value = event.target.value.trim();
                          if (!selected || !value || value === speakerNames.get(turn.speaker_id)) return;
                          void nameMeetingSpeaker(selected, turn.speaker_id, value).then(setDetail);
                        }}
                      />
                      <button className="textButton" type="button" onClick={() => seek(turn.start_seconds)}>
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
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            void saveCorrection(turn);
                          }
                          if (event.key === "Escape") setEditing(null);
                        }}
                      />
                    ) : (
                      <p
                        className="meetingTurnText"
                        onDoubleClick={() => {
                          setEditing(turn.id);
                          setDraft(turn.text);
                        }}
                        title="Double-click to correct"
                      >
                        {turn.words.length ? (
                          turn.words.map((word, index) => (
                            <span
                              key={`${turn.id}-${index}`}
                              className={
                                word.start !== null &&
                                word.end !== null &&
                                position >= word.start &&
                                position <= word.end
                                  ? "isSpoken"
                                  : ""
                              }
                              onClick={() => seek(word.start)}
                            >
                              {word.text}{" "}
                            </span>
                          ))
                        ) : (
                          turn.text
                        )}
                      </p>
                    )}
                    {turn.corrected_at ? (
                      <span className="mutedMeta">Corrected · originally “{turn.original_text}”</span>
                    ) : null}
                  </li>
                ))}
              </ol>

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
