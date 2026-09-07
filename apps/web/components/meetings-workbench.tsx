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
  deleteMeeting,
  decideMeetingProposal,
  getMeeting,
  listMeetings,
  meetingAudioUrl,
  nameMeetingSpeaker,
  renameMeeting,
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

const PIPELINE: Array<{ stage: MeetingStage; label: string }> = [
  { stage: "uploaded", label: "Stored" },
  { stage: "isolating", label: "Cleaned" },
  { stage: "transcribing", label: "Transcribed" },
  { stage: "analyzing", label: "Analyzed" },
  { stage: "ready", label: "Ready" },
];

type MeetingView = "overview" | "transcript" | "activity";
type MeetingFilter = "all" | "ready" | "working" | "failed";

const PLAYER_SPEEDS = [1, 1.25, 1.5, 2] as const;

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

function transcriptText(detail: MeetingDetail): string {
  const names = new Map(
    detail.speakers.map((speaker) => [speaker.speaker_id, speaker.display_name]),
  );
  return detail.turns
    .map((turn) => `[${clock(turn.start_seconds)}] ${names.get(turn.speaker_id) || turn.speaker_id}\n${turn.text}`)
    .join("\n\n");
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
  const [uploadTitle, setUploadTitle] = useState("");
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [meetingQuery, setMeetingQuery] = useState("");
  const [meetingFilter, setMeetingFilter] = useState<MeetingFilter>("all");
  const [transcriptQuery, setTranscriptQuery] = useState("");
  const [view, setView] = useState<MeetingView>("overview");
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [savingTitle, setSavingTitle] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [proposalBusy, setProposalBusy] = useState<string | null>(null);
  const playerRef = useRef<HTMLAudioElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const detailRequestRef = useRef(0);

  const refreshList = useCallback(async () => {
    try {
      const rows = await listMeetings();
      setMeetings(rows);
      setSelected((current) =>
        current && rows.some((meeting) => meeting.id === current)
          ? current
          : rows[0]?.id ?? null,
      );
    } catch (listError) {
      setError(listError instanceof Error ? listError.message : "Could not read meetings.");
    }
  }, []);

  const refreshDetail = useCallback(async (meetingId: string) => {
    const requestId = ++detailRequestRef.current;
    try {
      const next = await getMeeting(meetingId);
      if (requestId === detailRequestRef.current) setDetail(next);
    } catch {
      if (requestId === detailRequestRef.current) setDetail(null);
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  useEffect(() => {
    if (selected) {
      void refreshDetail(selected);
    } else {
      detailRequestRef.current += 1;
      setDetail(null);
    }
  }, [selected, refreshDetail]);

  useEffect(() => {
    setPosition(0);
    setDuration(0);
    setPlaying(false);
    setView("overview");
    setTranscriptQuery("");
    setEditingTitle(false);
    playerRef.current?.pause();
  }, [selected]);

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
    const queued = Array.from(files ?? []);
    if (!queued.length) return;
    setUploading(true);
    setError(null);
    let latest: Meeting | null = null;
    const failures: string[] = [];
    try {
      for (const file of queued) {
        try {
          latest = await uploadMeeting(
            file,
            queued.length === 1 ? uploadTitle.trim() : "",
          );
          // Make partial progress visible even if a later file fails.
          await refreshList();
        } catch {
          failures.push(file.name);
        }
      }
      await refreshList();
      if (latest) setSelected(latest.id);
      if (!failures.length) setUploadTitle("");
      if (failures.length) {
        setError(
          failures.length === queued.length
            ? "Those recordings could not be uploaded. Please check the file format and try again."
            : `${queued.length - failures.length} added. Could not add ${failures.join(", ")}.`,
        );
      }
    } finally {
      if (fileRef.current) fileRef.current.value = "";
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
    setProposalBusy(proposal.id);
    try {
      await decideMeetingProposal(selected, proposal.id, status);
      await refreshDetail(selected);
    } catch (decideError) {
      setError(decideError instanceof Error ? decideError.message : "Could not save that.");
    } finally {
      setProposalBusy(null);
    }
  }

  async function saveTitle() {
    if (!selected || !detail) return;
    const title = titleDraft.trim();
    if (!title || title === detail.meeting.title) {
      setEditingTitle(false);
      return;
    }
    setSavingTitle(true);
    try {
      await renameMeeting(selected, title);
      await Promise.all([refreshDetail(selected), refreshList()]);
      setEditingTitle(false);
    } catch (renameError) {
      setError(renameError instanceof Error ? renameError.message : "Could not rename that recording.");
    } finally {
      setSavingTitle(false);
    }
  }

  async function removeSelectedMeeting() {
    if (!selected || !detail || deleting) return;
    const title = detail.meeting.title || detail.meeting.audio_filename;
    if (!window.confirm(`Remove “${title}” from the meeting library?`)) return;
    setDeleting(true);
    setError(null);
    try {
      await deleteMeeting(selected);
      detailRequestRef.current += 1;
      setDetail(null);
      setSelected(null);
      await refreshList();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Could not remove that meeting.");
    } finally {
      setDeleting(false);
    }
  }

  function togglePlayback() {
    const player = playerRef.current;
    if (!player) return;
    if (player.paused) void player.play().catch(() => setError("That recording could not be played."));
    else player.pause();
  }

  function movePlayback(by: number) {
    const player = playerRef.current;
    if (!player) return;
    player.currentTime = Math.max(0, Math.min(player.duration || 0, player.currentTime + by));
  }

  function changePlaybackRate() {
    const index = PLAYER_SPEEDS.indexOf(playbackRate as (typeof PLAYER_SPEEDS)[number]);
    const next = PLAYER_SPEEDS[(index + 1) % PLAYER_SPEEDS.length];
    setPlaybackRate(next);
    if (playerRef.current) playerRef.current.playbackRate = next;
  }

  async function copyTranscript() {
    if (!detail) return;
    try {
      await navigator.clipboard.writeText(transcriptText(detail));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_600);
    } catch {
      setError("The transcript could not be copied.");
    }
  }

  function exportTranscript() {
    if (!detail) return;
    const blob = new Blob([transcriptText(detail)], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${(detail.meeting.title || "meeting").replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "meeting"}-transcript.txt`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const open = detail?.proposals.filter((item) => item.status === "proposed") ?? [];
  const reviewed = detail?.proposals.filter((item) => item.status !== "proposed") ?? [];
  const filteredMeetings = meetings.filter((meeting) => {
    const queryMatch = !meetingQuery.trim()
      || `${meeting.title} ${meeting.audio_filename}`.toLowerCase().includes(meetingQuery.trim().toLowerCase());
    const filterMatch = meetingFilter === "all"
      || (meetingFilter === "ready" && meeting.stage === "ready")
      || (meetingFilter === "working" && WORKING.has(meeting.stage))
      || (meetingFilter === "failed" && meeting.stage === "failed");
    return queryMatch && filterMatch;
  });
  const filteredTurns = detail?.turns.filter((turn) => {
    if (!transcriptQuery.trim()) return true;
    const speaker = speakerNames.get(turn.speaker_id) || turn.speaker_id;
    return `${speaker} ${turn.text}`.toLowerCase().includes(transcriptQuery.trim().toLowerCase());
  }) ?? [];
  const readyCount = meetings.filter((meeting) => meeting.stage === "ready").length;
  const totalSeconds = meetings.reduce((sum, meeting) => sum + (meeting.duration_seconds ?? 0), 0);
  const totalHours = totalSeconds >= 3600
    ? `${(totalSeconds / 3600).toFixed(totalSeconds >= 36_000 ? 0 : 1)}h`
    : clock(totalSeconds);
  const activePipelineIndex = detail
    ? Math.max(0, PIPELINE.findIndex((item) => item.stage === detail.meeting.stage))
    : 0;

  return (
    <div className="workspacePage meetingsPage audioWorkspacePage">
      <header className="pageHeader">
        <div>
          <span className="eyebrow">Conversation intelligence</span>
          <h1>Meetings</h1>
          <p>
            <span><strong>{meetings.length}</strong> recordings</span>
            <span><strong>{readyCount}</strong> ready to review</span>
            <span><strong>{totalHours}</strong> captured</span>
          </p>
        </div>
        <button className="primaryButton" type="button" onClick={() => fileRef.current?.click()} disabled={uploading}>
          <span aria-hidden="true">＋</span>{uploading ? "Uploading…" : "Choose recordings"}
        </button>
      </header>

      <section className="meetingCapture" aria-label="New recording">
        <div className="meetingCaptureCard">
          <span className={`meetingCaptureOrb ${uploading ? "isBusy" : ""}`} aria-hidden="true"><i /><i /><i /></span>
          <div><span className="eyebrow">New recording</span><strong>{uploading ? "Adding it to your studio…" : "Drop in audio or video"}</strong><small>Speaker-aware transcript · evidence-linked insights</small></div>
          <label className="meetingUploadTitle">
            <span>Optional title</span>
            <input value={uploadTitle} onChange={(event) => setUploadTitle(event.target.value)} placeholder="Monday product sync" maxLength={200} />
          </label>
          <button className="primaryButton" type="button" onClick={() => fileRef.current?.click()} disabled={uploading}>
            <span aria-hidden="true">＋</span>{uploading ? "Uploading…" : "Choose recordings"}
          </button>
          <small>MP3, WAV, M4A, MP4, or WebM</small>
        </div>
      </section>

      {error ? (
        <div className="composerError" role="alert">
          <span>!</span><p>{error}</p>
          <button className="errorDismiss" type="button" aria-label="Dismiss" onClick={() => setError(null)}>×</button>
        </div>
      ) : null}

      <input ref={fileRef} type="file" hidden multiple accept="audio/*,video/*" onChange={(event) => void upload(event.target.files)} />

      <div className="meetingsLayout meetingStudio meetingStudioShell">
        <aside
          className={`meetingsList meetingLibrary ${dragging ? "isDragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false); }}
          onDrop={(event) => { event.preventDefault(); setDragging(false); void upload(event.dataTransfer.files); }}
        >
          <div className="meetingLibraryHead">
            <div><span className="eyebrow">Recording library</span><strong>{meetings.length} conversation{meetings.length === 1 ? "" : "s"}</strong></div>
            <button type="button" onClick={() => fileRef.current?.click()} aria-label="Add recordings">＋</button>
          </div>
          <label className="meetingLibrarySearch">
            <span aria-hidden="true">⌕</span>
            <input value={meetingQuery} onChange={(event) => setMeetingQuery(event.target.value)} placeholder="Search recordings" aria-label="Search recordings" />
            {meetingQuery ? <button type="button" onClick={() => setMeetingQuery("")} aria-label="Clear recording search">×</button> : null}
          </label>
          <div className="meetingLibraryFilters" role="group" aria-label="Filter recordings">
            {(["all", "ready", "working", "failed"] as MeetingFilter[]).map((filter) => (
              <button key={filter} type="button" className={meetingFilter === filter ? "selected" : ""} aria-pressed={meetingFilter === filter} onClick={() => setMeetingFilter(filter)}>
                {filter === "all" ? "All" : filter === "working" ? "Processing" : filter[0].toUpperCase() + filter.slice(1)}
              </button>
            ))}
          </div>
          <div className="meetingLibraryRows">
            {filteredMeetings.length ? filteredMeetings.map((meeting, meetingIndex) => (
              <button key={meeting.id} type="button" className={`meetingRow ${selected === meeting.id ? "selected" : ""}`} aria-pressed={selected === meeting.id} onClick={() => setSelected(meeting.id)}>
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
            )) : meetings.length ? (
              <div className="meetingLibraryEmpty"><span>⌕</span><strong>No recordings match</strong><small>Try another search or status.</small></div>
            ) : (
              <button className="meetingDropEmpty" type="button" onClick={() => fileRef.current?.click()}>
                <span aria-hidden="true">↥</span><strong>Drop a recording</strong><small>Audio or video · Metis handles the rest</small>
              </button>
            )}
          </div>
          <p className="meetingDropHint">Drop multiple recordings anywhere in this column.</p>
        </aside>

        <section className="meetingDetail meetingReviewStudio">
          {!detail ? (
            <div className="meetingDetailEmpty"><span aria-hidden="true">◌</span><strong>Choose a conversation</strong><p>Its audio, decisions, speakers, and transcript will open here.</p></div>
          ) : (
            <>
              <header className="meetingHead meetingReviewHeader">
                <div className="meetingTitleBlock">
                  <span className="eyebrow">Conversation studio</span>
                  {editingTitle ? (
                    <form className="meetingTitleEditor" onSubmit={(event) => { event.preventDefault(); void saveTitle(); }}>
                      <input autoFocus value={titleDraft} maxLength={200} onChange={(event) => setTitleDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Escape") setEditingTitle(false); }} aria-label="Meeting title" />
                      <button type="submit" disabled={savingTitle}>{savingTitle ? "Saving…" : "Save"}</button>
                      <button type="button" onClick={() => setEditingTitle(false)}>Cancel</button>
                    </form>
                  ) : (
                    <button className="meetingEditableTitle" type="button" onClick={() => { setTitleDraft(detail.meeting.title || detail.meeting.audio_filename); setEditingTitle(true); }} title="Rename meeting">
                      <h2>{detail.meeting.title || detail.meeting.audio_filename}</h2><span aria-hidden="true">✎</span>
                    </button>
                  )}
                  <p>{meetingDate(detail.meeting.created_at)}{detail.meeting.duration_seconds ? ` · ${clock(detail.meeting.duration_seconds)}` : ""}{detail.meeting.language ? ` · ${detail.meeting.language.toUpperCase()}` : ""}{` · ${detail.speakers.length} voice${detail.speakers.length === 1 ? "" : "s"}`}</p>
                </div>
                <div className="meetingHeaderActions">
                  <span className={`meetingStage meetingStagePill is-${detail.meeting.stage}`}><i aria-hidden="true" />{STAGE_LABEL[detail.meeting.stage]}</span>
                  {detail.meeting.stage === "failed" ? <button className="secondaryButton" type="button" onClick={() => void retryMeeting(detail.meeting.id).then(() => refreshDetail(detail.meeting.id))}>Retry analysis</button> : null}
                  <button className="meetingUtilityButton" type="button" onClick={() => void copyTranscript()} disabled={!detail.turns.length}>{copied ? "Copied" : "Copy transcript"}</button>
                  <span className="visuallyHidden" role="status" aria-live="polite">{copied ? "Transcript copied" : ""}</span>
                  <button className="meetingUtilityButton" type="button" onClick={exportTranscript} disabled={!detail.turns.length}>Export .txt</button>
                  <button className="meetingUtilityButton isDanger" type="button" onClick={() => void removeSelectedMeeting()} disabled={deleting}>{deleting ? "Removing…" : "Remove"}</button>
                </div>
              </header>

              <div className="meetingPlayerDock meetingPlayerStudio">
                <button className="meetingPlayGlyph" type="button" onClick={togglePlayback} aria-label={playing ? "Pause recording" : "Play recording"}><span aria-hidden="true">{playing ? "Ⅱ" : "▶"}</span></button>
                <button className="meetingSkipButton" type="button" onClick={() => movePlayback(-10)} aria-label="Go back 10 seconds">−10</button>
                <div className="meetingPlayerTimeline">
                  <input type="range" min={0} max={Math.max(duration, 1)} step={0.05} value={Math.min(position, Math.max(duration, 1))} onChange={(event) => { const next = Number(event.target.value); if (playerRef.current) playerRef.current.currentTime = next; setPosition(next); }} aria-label="Recording position" />
                  <div className="meetingWave" aria-hidden="true">{[8, 14, 22, 13, 27, 18, 31, 17, 24, 11, 29, 20, 12, 25, 16, 30, 19, 9, 21, 13, 27, 16, 8, 17, 25, 13, 19, 9, 24, 15, 28].map((height, index) => <i key={index} className={duration && position / duration > index / 31 ? "isPast" : ""} style={{ height }} />)}</div>
                  <span>{clock(position)} <i>/</i> {duration ? clock(duration) : clock(detail.meeting.duration_seconds)}</span>
                </div>
                <button className="meetingSkipButton" type="button" onClick={() => movePlayback(10)} aria-label="Go forward 10 seconds">+10</button>
                <button className="meetingRateButton" type="button" onClick={changePlaybackRate} aria-label={`Playback speed ${playbackRate} times`}>{playbackRate}×</button>
                <audio ref={playerRef} preload="metadata" src={meetingAudioUrl(detail.meeting.id)} onLoadedMetadata={(event) => { setDuration(event.currentTarget.duration || detail.meeting.duration_seconds || 0); event.currentTarget.playbackRate = playbackRate; }} onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
              </div>

              <nav className="meetingDetailTabs" aria-label="Meeting views" role="tablist">
                {(["overview", "transcript", "activity"] as MeetingView[]).map((item) => (
                  <button key={item} id={`meeting-tab-${item}`} type="button" role="tab" aria-selected={view === item} aria-controls={`meeting-panel-${item}`} className={view === item ? "selected" : ""} onClick={() => setView(item)}>
                    {item[0].toUpperCase() + item.slice(1)}
                    {item === "overview" && open.length ? <span>{open.length}</span> : item === "transcript" ? <span>{detail.turns.length}</span> : null}
                  </button>
                ))}
              </nav>

              {view === "overview" ? (
                <div className="meetingOverview" id="meeting-panel-overview" role="tabpanel" aria-labelledby="meeting-tab-overview">
                  {detail.meeting.stage !== "ready" ? (
                    <section className="meetingPipeline" aria-label="Processing progress">
                      <header><div><span className="eyebrow">Processing</span><h3>{detail.meeting.stage === "failed" ? "This recording needs attention" : STAGE_LABEL[detail.meeting.stage]}</h3></div><small>{detail.meeting.stage === "failed" ? "Your original recording is safe" : "You can leave this page"}</small></header>
                      <ol>{PIPELINE.map((item, index) => <li key={item.stage} className={index < activePipelineIndex || detail.meeting.stage === "ready" ? "complete" : index === activePipelineIndex ? detail.meeting.stage === "failed" ? "failed" : "active" : ""}><span>{index < activePipelineIndex ? "✓" : index + 1}</span><strong>{item.label}</strong></li>)}</ol>
                      {detail.meeting.error ? <p role="alert">{detail.meeting.error}</p> : null}
                    </section>
                  ) : null}

                  {detail.meeting.summary ? (
                    <section className="meetingReadout">
                      <span className="meetingReadoutMark" aria-hidden="true">✦</span>
                      <div>
                        <header><span className="eyebrow">Meeting readout</span><small>Grounded in the transcript</small></header>
                        <p>{detail.meeting.summary}</p>
                      </div>
                      <button type="button" onClick={() => setView("transcript")}>Check the record →</button>
                    </section>
                  ) : null}

                  <div className="meetingOverviewGrid">
                    <section className="meetingInsights">
                      <header><div><span className="eyebrow">Review queue</span><h3>{open.length ? `${open.length} grounded item${open.length === 1 ? "" : "s"}` : "Everything has been reviewed"}</h3></div><small>{!detail.meeting.account_id && open.some((proposal) => proposal.kind === "action") ? "Link a customer before keeping actions" : "Hear the evidence before deciding"}</small></header>
                      {open.length ? <ul className="meetingProposals">{open.map((proposal) => (
                        <li key={proposal.id} className={`is-${proposal.kind}`}>
                          <span className="meetingProposalIcon" aria-hidden="true">{proposal.kind === "action" ? "↗" : proposal.kind === "decision" ? "◆" : "◎"}</span>
                          <div><span className="meetingProposalKind">{PROPOSAL_LABEL[proposal.kind]}</span><p>{proposal.payload.account_name ?? proposal.payload.description ?? proposal.payload.summary}</p>{proposal.evidence_start !== null ? <button className="meetingEvidence" type="button" onClick={() => seek(proposal.evidence_start)}><span aria-hidden="true">▶</span> Evidence at {clock(proposal.evidence_start)}</button> : null}</div>
                          <div className="meetingProposalActions">
                            <button className="proposalAccept" type="button" disabled={proposalBusy === proposal.id || (proposal.kind === "action" && !detail.meeting.account_id)} title={proposal.kind === "action" && !detail.meeting.account_id ? "Link this meeting to a customer first" : undefined} onClick={() => void decide(proposal, "accepted")}>{proposal.kind === "action" ? "Keep action" : proposal.kind === "decision" ? "Confirm" : "Link"}</button>
                            <button className="proposalDismiss" type="button" disabled={proposalBusy === proposal.id} onClick={() => void decide(proposal, "rejected")} aria-label={`Dismiss ${PROPOSAL_LABEL[proposal.kind].toLowerCase()}`}>×</button>
                          </div>
                        </li>
                      ))}</ul> : <div className="meetingReviewEmpty"><span aria-hidden="true">✓</span><div><strong>No open suggestions</strong><p>Anything previously kept or dismissed is in Activity.</p></div></div>}
                    </section>

                    <aside className="meetingParticipants">
                      <header><div><span className="eyebrow">People</span><h3>Name the voices once</h3></div><span>{detail.speakers.length}</span></header>
                      <div className="meetingParticipantList">
                        {detail.speakers.map((speaker, index) => {
                          const speakerName = speaker.display_name || speaker.speaker_id;
                          return <label key={speaker.speaker_id} className={`speakerTone-${index % 4}`}><span className="meetingSpeakerAvatar" aria-hidden="true">{initials(speakerName)}</span><span><small>Voice {index + 1}</small><input list="meeting-speaker-names" defaultValue={speakerName} aria-label={`Name for ${speaker.speaker_id}`} onBlur={(event) => { const value = event.target.value.trim(); if (!selected || !value || value === speakerNames.get(speaker.speaker_id)) return; void nameMeetingSpeaker(selected, speaker.speaker_id, value).then(() => refreshDetail(selected)); }} /></span></label>;
                        })}
                      </div>
                      <p>Names update every turn from that voice. Suggested people come from the linked account only.</p>
                    </aside>
                  </div>

                  {detail.turns.length ? <section className="meetingConversationPreview"><header><div><span className="eyebrow">Jump back in</span><h3>Recent moments</h3></div><button type="button" className="textButton" onClick={() => setView("transcript")}>Open full transcript →</button></header><ol>{detail.turns.slice(0, 3).map((turn) => <li key={turn.id}><button type="button" onClick={() => seek(turn.start_seconds)}>{clock(turn.start_seconds)}</button><div><strong>{speakerNames.get(turn.speaker_id) || turn.speaker_id}</strong><p>{turn.text}</p></div></li>)}</ol></section> : null}
                </div>
              ) : null}

              {view === "transcript" ? (
                <section className="meetingTranscriptSection meetingTranscriptWorkspace" id="meeting-panel-transcript" role="tabpanel" aria-labelledby="meeting-tab-transcript">
                  <header><div><span className="eyebrow">Verbatim record</span><h3>The conversation</h3><small>Click words to play · double-click a paragraph to correct</small></div><label className="meetingTranscriptSearch"><span aria-hidden="true">⌕</span><input value={transcriptQuery} onChange={(event) => setTranscriptQuery(event.target.value)} placeholder="Search transcript or speaker" aria-label="Search transcript" />{transcriptQuery ? <button type="button" onClick={() => setTranscriptQuery("")} aria-label="Clear transcript search">×</button> : null}</label></header>
                  <ol className="meetingTranscript">
                    {filteredTurns.map((turn) => {
                      const speakerName = speakerNames.get(turn.speaker_id) || turn.speaker_id;
                      return <li key={turn.id} className={`${turn.corrected_at ? "isCorrected " : ""}speakerTone-${turn.ordinal % 4}`}><span className="meetingSpeakerAvatar" aria-hidden="true">{initials(speakerName)}</span><div className="meetingTurnBody"><div className="meetingTurnHead"><strong className="meetingSpeakerName">{speakerName}</strong><button className="meetingTimestamp" type="button" onClick={() => seek(turn.start_seconds)}>{clock(turn.start_seconds)}</button>{turn.corrected_at ? <span className="meetingCorrectedBadge">Corrected</span> : null}<button className="meetingEditTurn" type="button" onClick={() => { setEditing(turn.id); setDraft(turn.text); }}>Correct text</button></div>{editing === turn.id ? <div className="meetingEditorWrap"><textarea className="meetingEditor" autoFocus value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void saveCorrection(turn); } if (event.key === "Escape") setEditing(null); }} /><div><button type="button" onClick={() => void saveCorrection(turn)}>Save correction</button><button type="button" onClick={() => setEditing(null)}>Cancel</button></div></div> : <p className="meetingTurnText">{turn.words.length ? turn.words.map((word, index) => <button type="button" key={`${turn.id}-${index}`} className={`meetingTranscriptWord ${word.start !== null && word.end !== null && position >= word.start && position <= word.end ? "isSpoken" : ""}`} onClick={() => seek(word.start)} aria-label={`Play from ${word.text} at ${clock(word.start)}`}>{word.text}{" "}</button>) : turn.text}</p>}{turn.corrected_at ? <details className="meetingOriginalText"><summary>See original transcript</summary><p>{turn.original_text}</p></details> : null}</div></li>;
                    })}
                  </ol>
                  {!filteredTurns.length ? <div className="meetingTranscriptEmpty"><span>⌕</span><strong>No matching moment</strong><p>Try a speaker name or a phrase you remember.</p></div> : null}
                </section>
              ) : null}

              {view === "activity" ? (
                <div className="meetingActivityView" id="meeting-panel-activity" role="tabpanel" aria-labelledby="meeting-tab-activity">
                  <section className="meetingReviewed"><header><div><span className="eyebrow">Decision log</span><h3>Reviewed suggestions</h3></div><small>{reviewed.length} decided</small></header>{reviewed.length ? <ul>{reviewed.map((proposal) => <li key={proposal.id} className={`is-${proposal.status}`}><span aria-hidden="true">{proposal.status === "accepted" ? "✓" : "×"}</span><div><strong>{PROPOSAL_LABEL[proposal.kind]}</strong><p>{proposal.payload.account_name ?? proposal.payload.description ?? proposal.payload.summary}</p></div><small>{proposal.status === "accepted" ? "Kept" : "Dismissed"}</small></li>)}</ul> : <div className="meetingReviewEmpty"><span aria-hidden="true">◌</span><div><strong>No decisions yet</strong><p>Suggestions you review will stay visible here.</p></div></div>}</section>
                  <section className="meetingEventTimeline"><header><div><span className="eyebrow">Processing log</span><h3>How this record was made</h3></div></header><ol>{detail.events.map((event, index) => <li key={`${event.created_at}-${index}`}><span className={event.stage === "failed" ? "failed" : ""}>{event.stage === "failed" ? "!" : index + 1}</span><div><strong>{STAGE_LABEL[event.stage as MeetingStage] ?? event.stage}</strong><p>{event.message || "Stage completed"}</p><small>{new Date(event.created_at).toLocaleString()}</small></div></li>)}</ol></section>
                </div>
              ) : null}

              <datalist id="meeting-speaker-names">{detail.suggested_names.map((name) => <option key={name} value={name} />)}</datalist>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
