"use client";

// Meetings: recordings on the left, one recording on the right with the
// shared player above its overview and transcript.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Upload } from "lucide-react";

import { MeetingOverview, STAGE_LABEL } from "@/components/meetings/overview";
import { MeetingTranscript } from "@/components/meetings/transcript";
import { AudioPlayer, type AudioPlayerHandle } from "@/components/ui/audio-player";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Status, type StatusState } from "@/components/ui/status";
import { useToast } from "@/components/ui/toast";
import { correctMeetingTurn, decideMeetingProposal, deleteMeeting, getMeeting, listMeetings, meetingAudioUrl, nameMeetingSpeaker, renameMeeting, retryMeeting, uploadMeeting } from "@/lib/api";
import { clock, downloadText, safeFilePart, transcriptText } from "@/lib/audio";
import type { Meeting, MeetingDetail, MeetingProposal, MeetingStage, MeetingTurn } from "@/lib/types";

const WORKING = new Set<MeetingStage>(["uploaded", "isolating", "transcribing", "analyzing"]);
const POLL_MS = 3_000;
type Filter = "all" | "ready" | "working" | "failed";
const FILTERS: Array<[Filter, string]> = [["all", "All"], ["ready", "Ready"], ["working", "Processing"], ["failed", "Failed"]];

function stageStatus(stage: MeetingStage): StatusState {
  return stage === "ready" ? "ready" : stage === "failed" ? "needs-review" : "waiting";
}

function meetingDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Recently" : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

function plainTranscript(detail: MeetingDetail): string {
  const names = new Map(detail.speakers.map((speaker) => [speaker.speaker_id, speaker.display_name]));
  return transcriptText(detail.turns.map((turn) => ({ speaker: names.get(turn.speaker_id) || turn.speaker_id, text: turn.text, at: clock(turn.start_seconds) })));
}

export function MeetingsWorkbench() {
  const toast = useToast();
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [view, setView] = useState<"overview" | "transcript">("overview");
  const [position, setPosition] = useState(0);
  const [title, setTitle] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const player = useRef<AudioPlayerHandle>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const detailRequest = useRef(0);

  const refreshList = useCallback(async () => {
    try {
      const rows = await listMeetings();
      setMeetings(rows);
      setSelected((current) => (current && rows.some((meeting) => meeting.id === current) ? current : rows[0]?.id ?? null));
    } catch (listError) {
      setError(listError instanceof Error ? listError.message : "Could not read meetings.");
    }
  }, []);
  const refreshDetail = useCallback(async (id: string) => {
    const request = ++detailRequest.current;
    try {
      const next = await getMeeting(id);
      if (request === detailRequest.current) setDetail(next);
    } catch {
      if (request === detailRequest.current) setDetail(null);
    }
  }, []);

  useEffect(() => { void refreshList(); }, [refreshList]);
  useEffect(() => {
    setPosition(0);
    setView("overview");
    setTitle(null);
    if (selected) void refreshDetail(selected);
    else { detailRequest.current += 1; setDetail(null); }
  }, [refreshDetail, selected]);

  // Poll only while something is moving; a ready meeting is a document.
  const stage = detail?.meeting.stage;
  useEffect(() => {
    if (!selected || !stage || !WORKING.has(stage)) return;
    const timer = window.setInterval(() => { void refreshDetail(selected); void refreshList(); }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [refreshDetail, refreshList, selected, stage]);

  const names = useMemo(() => new Map(detail?.speakers.map((speaker) => [speaker.speaker_id, speaker.display_name]) ?? []), [detail]);

  const upload = async (files: FileList | null) => {
    const queued = Array.from(files ?? []);
    if (!queued.length) return;
    setUploading(true);
    setError(null);
    const failed: string[] = [];
    let last: Meeting | null = null;
    for (const file of queued) {
      try { last = await uploadMeeting(file); await refreshList(); } catch { failed.push(file.name); }
    }
    if (last) setSelected(last.id);
    if (failed.length) setError(failed.length === queued.length ? "Those recordings could not be uploaded. Check the format and try again." : `Could not add ${failed.join(", ")}.`);
    if (fileInput.current) fileInput.current.value = "";
    setUploading(false);
  };

  // One shape for every edit: mark busy, do it, re-read.
  const edit = async (key: string, work: () => Promise<unknown>, fallback: string) => {
    if (!selected) return;
    setBusy(key);
    setError(null);
    try {
      await work();
      await Promise.all([refreshDetail(selected), refreshList()]);
    } catch (editError) {
      setError(editError instanceof Error ? editError.message : fallback);
    } finally {
      setBusy(null);
    }
  };
  const seek = (seconds: number | null) => { if (seconds !== null) player.current?.seek(seconds); };
  const remove = () => {
    if (!detail || !window.confirm(`Remove “${detail.meeting.title || detail.meeting.audio_filename}” from the library?`)) return;
    void edit("delete", async () => { await deleteMeeting(detail.meeting.id); setSelected(null); }, "Could not remove that meeting.");
  };
  const saveTitle = () => {
    const next = title?.trim();
    if (!detail || !next || next === detail.meeting.title) { setTitle(null); return; }
    void edit("title", () => renameMeeting(detail.meeting.id, next), "Could not rename that recording.").then(() => setTitle(null));
  };

  const visible = meetings.filter((meeting) => {
    const matchesQuery = !query.trim() || `${meeting.title} ${meeting.audio_filename}`.toLowerCase().includes(query.trim().toLowerCase());
    const matchesFilter = filter === "all" || (filter === "working" ? WORKING.has(meeting.stage) : meeting.stage === filter);
    return matchesQuery && matchesFilter;
  });
  const ready = meetings.filter((meeting) => meeting.stage === "ready").length;
  const meeting = detail?.meeting;
  const openCount = detail?.proposals.filter((item) => item.status === "proposed").length ?? 0;

  return (
    <div className="meetings" onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false); }} onDrop={(event) => { event.preventDefault(); setDragging(false); void upload(event.dataTransfer.files); }}>
      <PageHeader
        eyebrow="Conversation intelligence"
        title="Meetings"
        lede={`${meetings.length} ${meetings.length === 1 ? "recording" : "recordings"} · ${ready} ready to review`}
        actions={<button type="button" className="ui-btn is-primary" onClick={() => fileInput.current?.click()} disabled={uploading}><Upload size={14} aria-hidden="true" /> {uploading ? "Uploading…" : "Add recordings"}</button>}
      />
      <input ref={fileInput} type="file" hidden multiple accept="audio/*,video/*" onChange={(event) => void upload(event.target.files)} />
      {error ? <Notice kind="error" onDismiss={() => setError(null)}>{error}</Notice> : null}
      {dragging ? <div className="meetings-drop">Drop audio or video to add it</div> : null}

      <div className="meetings-body">
        <aside className="accounts" aria-label="Recordings">
          <input type="search" className="accounts-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search recordings" aria-label="Search recordings" />
          <div className="accounts-filters" role="group" aria-label="Filter recordings">
            {FILTERS.map(([value, label]) => <button key={value} type="button" className={`ui-chip${filter === value ? " is-accent" : ""}`} aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
          </div>
          <div className="accounts-list">
            {visible.map((item) => (
              <button key={item.id} type="button" className={`accounts-row${selected === item.id ? " is-selected" : ""}`} aria-current={selected === item.id ? "page" : undefined} onClick={() => setSelected(item.id)}>
                <span><strong>{item.title || item.audio_filename}</strong><small>{meetingDate(item.created_at)} · {item.stage === "ready" ? clock(item.duration_seconds) : STAGE_LABEL[item.stage]}</small></span>
                <i className={`ui-dot is-${stageStatus(item.stage)}`} aria-hidden="true" />
              </button>
            ))}
            {!visible.length ? <p className="accounts-empty">{meetings.length ? "No recordings match." : "Drop a recording anywhere on this page, or add one above."}</p> : null}
          </div>
        </aside>

        <main className="meeting">
          {!meeting || !detail ? (
            <p className="records-empty">{meetings.length ? "Choose a recording." : "MP3, WAV, M4A, MP4 or WebM. Metis cleans the audio, transcribes it with speakers, and reads it through."}</p>
          ) : (
            <>
              <header className="meeting-head">
                {title !== null ? (
                  <form className="meeting-title" onSubmit={(event) => { event.preventDefault(); saveTitle(); }}>
                    <input autoFocus value={title} maxLength={200} onChange={(event) => setTitle(event.target.value)} onKeyDown={(event) => { if (event.key === "Escape") setTitle(null); }} aria-label="Meeting title" />
                    <button type="submit" className="ui-btn is-primary is-sm" disabled={busy === "title"}>Save</button>
                    <button type="button" className="ui-btn is-quiet is-sm" onClick={() => setTitle(null)}>Cancel</button>
                  </form>
                ) : (
                  <div>
                    <h2><button type="button" className="meeting-rename" title="Rename" onClick={() => setTitle(meeting.title || meeting.audio_filename)}>{meeting.title || meeting.audio_filename}</button></h2>
                    <p>{meetingDate(meeting.created_at)}{meeting.duration_seconds ? ` · ${clock(meeting.duration_seconds)}` : ""}{meeting.language ? ` · ${meeting.language.toUpperCase()}` : ""} · {detail.speakers.length} {detail.speakers.length === 1 ? "voice" : "voices"}</p>
                  </div>
                )}
                <Status state={stageStatus(meeting.stage)} label={STAGE_LABEL[meeting.stage]} />
                <span className="meeting-actions">
                  <button type="button" className="ui-btn is-quiet is-sm" disabled={!detail.turns.length} onClick={() => void navigator.clipboard.writeText(plainTranscript(detail)).then(() => toast("Transcript copied"))}>Copy</button>
                  <button type="button" className="ui-btn is-quiet is-sm" disabled={!detail.turns.length} onClick={() => downloadText(`${safeFilePart(meeting.title || "meeting")}-transcript.txt`, plainTranscript(detail))}>Export</button>
                  <button type="button" className="ui-btn is-quiet is-sm is-danger-text" disabled={busy === "delete"} onClick={remove}>Remove</button>
                </span>
              </header>

              <AudioPlayer ref={player} src={meetingAudioUrl(meeting.id)} label="Recording" onTime={setPosition} onError={() => setError("That recording could not be played.")} />

              <nav className="ui-tabs" aria-label="Meeting views">
                <button type="button" className={view === "overview" ? "is-active" : ""} aria-current={view === "overview" ? "page" : undefined} onClick={() => setView("overview")}>Overview{openCount ? <small>{openCount}</small> : null}</button>
                <button type="button" className={view === "transcript" ? "is-active" : ""} aria-current={view === "transcript" ? "page" : undefined} onClick={() => setView("transcript")}>Transcript<small>{detail.turns.length}</small></button>
              </nav>

              {view === "overview" ? (
                <MeetingOverview detail={detail} busy={busy} onSeek={seek}
                  onDecide={(proposal: MeetingProposal, status) => void edit(proposal.id, () => decideMeetingProposal(meeting.id, proposal.id, status), "Could not save that.")}
                  onName={(speakerId, name) => void edit(speakerId, () => nameMeetingSpeaker(meeting.id, speakerId, name), "Could not name that voice.")}
                  onRetry={() => void edit("retry", () => retryMeeting(meeting.id), "Could not retry.")} />
              ) : (
                <MeetingTranscript turns={detail.turns} names={names} position={position} onSeek={seek} onCorrect={(turn: MeetingTurn, text) => edit(turn.id, () => correctMeetingTurn(meeting.id, turn.id, text), "Could not save that correction.")} />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
