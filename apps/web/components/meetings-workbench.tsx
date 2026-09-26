"use client";

// A recording library opens into one focused workspace. The URL owns the
// selected recording; background refreshes only update its stored data.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, ArrowUpRight, AudioLines, CheckCircle2, ChevronRight, Clock3, Copy, Download, FileAudio, ListChecks, LoaderCircle, Search, Upload } from "lucide-react";

import { MeetingOverview, STAGE_LABEL } from "@/components/meetings/overview";
import { MeetingTranscript } from "@/components/meetings/transcript";
import { AudioPlayer, type AudioPlayerHandle } from "@/components/ui/audio-player";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { Status, type StatusState } from "@/components/ui/status";
import { useToast } from "@/components/ui/toast";
import { correctMeetingTurn, decideMeetingProposal, deleteMeeting, getMeeting, listMeetings, meetingAudioUrl, nameMeetingSpeaker, renameMeeting, retryMeeting, uploadMeeting } from "@/lib/api";
import { clock, downloadText, safeFilePart, transcriptText } from "@/lib/audio";
import type { Meeting, MeetingDetail, MeetingProposal, MeetingStage, MeetingTurn } from "@/lib/types";
import { usePoll } from "@/hooks/use-poll";

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
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedMeeting = searchParams.get("meeting");
  const requestedTurn = searchParams.get("turn");
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [loaded, setLoaded] = useState(false);
  const selected = requestedMeeting;
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
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
  const listRequest = useRef(0);
  const selectedRef = useRef<string | null>(null);
  const openedSource = useRef<string | null>(null);
  const editBusy = useRef(false);
  const uploadBusy = useRef(false);
  const dragDepth = useRef(0);
  const libraryHeading = useRef<HTMLHeadingElement>(null);
  const workspaceHeading = useRef<HTMLHeadingElement>(null);
  const focusedRecording = useRef<string | null>(null);
  const lastOpened = useRef<string | null>(null);
  const openRecording = (id: string) => { lastOpened.current = id; router.push(`/meetings?meeting=${encodeURIComponent(id)}`, { scroll: false }); };
  const backToLibrary = () => { lastOpened.current = selected; router.push("/meetings", { scroll: false }); };

  const refreshList = useCallback(async () => {
    const request = ++listRequest.current;
    try {
      const rows = await listMeetings();
      if (request === listRequest.current) { setMeetings(rows); setListError(null); }
    } catch (problem) {
      if (request === listRequest.current) setListError(problem instanceof Error ? problem.message : "Could not read meetings.");
    } finally {
      if (request === listRequest.current) setLoaded(true);
    }
  }, []);
  const refreshDetail = useCallback(async (id: string) => {
    const request = ++detailRequest.current;
    if (selectedRef.current === id) setDetailError(null);
    try {
      const next = await getMeeting(id);
      if (request === detailRequest.current && selectedRef.current === id) { setDetail(next); return true; }
      return false;
    } catch (problem) {
      if (request === detailRequest.current && selectedRef.current === id) setDetailError(problem instanceof Error ? problem.message : "This recording could not be loaded.");
      return false;
    }
  }, []);

  useEffect(() => { void refreshList(); }, [refreshList]);
  useEffect(() => {
    selectedRef.current = selected;
    setPosition(0);
    setView("overview");
    setTitle(null);
    openedSource.current = null;
    if (selected) void refreshDetail(selected);
    else {
      focusedRecording.current = null;
      detailRequest.current += 1; setDetail(null); setDetailError(null);
      const previous = lastOpened.current;
      if (previous) requestAnimationFrame(() => { const card = document.getElementById(`recording-${previous}`); if (card) card.focus(); else libraryHeading.current?.focus(); });
    }
  }, [refreshDetail, selected]);

  useEffect(() => {
    if (!selected || detail?.meeting.id !== selected || focusedRecording.current === selected) return;
    focusedRecording.current = selected;
    if (requestedTurn && detail.turns.some((turn) => turn.id === requestedTurn)) return;
    const frame = requestAnimationFrame(() => { workspaceHeading.current?.scrollIntoView({ block: "start", behavior: "auto" }); workspaceHeading.current?.focus({ preventScroll: true }); });
    return () => cancelAnimationFrame(frame);
  }, [detail, requestedTurn, selected]);

  useEffect(() => {
    const sourceKey = requestedMeeting && requestedTurn ? `${requestedMeeting}:${requestedTurn}` : null;
    if (!sourceKey) { openedSource.current = null; return; }
    if (openedSource.current !== sourceKey && selected === requestedMeeting && detail?.meeting.id === requestedMeeting && detail.turns.some((turn) => turn.id === requestedTurn)) {
      openedSource.current = sourceKey;
      setView("transcript");
    }
  }, [detail, requestedMeeting, requestedTurn, selected]);

  // Poll only while something is moving; a ready meeting is a document.
  const moving = meetings.some((meeting) => WORKING.has(meeting.stage)) || Boolean(detail?.meeting.id === selected && WORKING.has(detail.meeting.stage));
  const refreshBoth = useCallback(async () => {
    await Promise.all([selected ? refreshDetail(selected) : Promise.resolve(), refreshList()]);
  }, [refreshDetail, refreshList, selected]);
  usePoll(refreshBoth, POLL_MS, moving);

  const names = useMemo(() => new Map(detail?.speakers.map((speaker, index) => [speaker.speaker_id, speaker.display_name || `Speaker ${index + 1}`]) ?? []), [detail]);

  const upload = async (files: FileList | null) => {
    const queued = Array.from(files ?? []);
    if (!queued.length || uploadBusy.current) return;
    uploadBusy.current = true;
    setUploading(true);
    setError(null);
    const failed: string[] = [];
    let last: Meeting | null = null;
    for (const file of queued) {
      try { last = await uploadMeeting(file); await refreshList(); } catch { failed.push(file.name); }
    }
    if (last) openRecording(last.id);
    if (failed.length) setError(failed.length === queued.length ? "Those recordings could not be uploaded. Check the format and try again." : `Could not add ${failed.join(", ")}.`);
    if (fileInput.current) fileInput.current.value = "";
    setUploading(false);
    uploadBusy.current = false;
  };

  // One shape for every edit: mark busy, do it, re-read.
  const edit = async (key: string, work: () => Promise<unknown>, fallback: string): Promise<boolean> => {
    if (!selected || editBusy.current) return false;
    editBusy.current = true;
    setBusy(key);
    setError(null);
    try {
      await work();
      const current = selectedRef.current;
      const [refreshed] = await Promise.all([current ? refreshDetail(current) : Promise.resolve(true), refreshList()]);
      return refreshed;
    } catch (editError) {
      setError(editError instanceof Error ? editError.message : fallback);
      return false;
    } finally {
      editBusy.current = false;
      setBusy(null);
    }
  };
  const seek = (seconds: number | null) => { if (seconds !== null) player.current?.seek(seconds); };
  const remove = () => {
    if (!detail || !window.confirm(`Remove “${detail.meeting.title || detail.meeting.audio_filename}” from the library?`)) return;
    void edit("delete", async () => { await deleteMeeting(detail.meeting.id); selectedRef.current = null; backToLibrary(); }, "Could not remove that meeting.");
  };
  const saveTitle = () => {
    const next = title?.trim();
    if (!detail || !next || next === detail.meeting.title) { setTitle(null); return; }
    void edit("title", () => renameMeeting(detail.meeting.id, next), "Could not rename that recording.").then((saved) => { if (saved) setTitle(null); });
  };

  const visible = meetings.filter((meeting) => {
    const matchesQuery = !query.trim() || `${meeting.title} ${meeting.audio_filename}`.toLowerCase().includes(query.trim().toLowerCase());
    const matchesFilter = filter === "all" || (filter === "working" ? WORKING.has(meeting.stage) : meeting.stage === filter);
    return matchesQuery && matchesFilter;
  });
  const ready = meetings.filter((meeting) => meeting.stage === "ready").length;
  const meeting = detail?.meeting.id === selected ? detail.meeting : undefined;
  const openCount = detail?.proposals.filter((item) => item.status === "proposed").length ?? 0;

  const counts = { all: meetings.length, ready, working: meetings.filter((item) => WORKING.has(item.stage)).length, failed: meetings.filter((item) => item.stage === "failed").length };
  const filtered = Boolean(query.trim()) || filter !== "all";

  return (
    <div className={`meetings conversations-meetings${selected ? " is-recording-open" : ""}`}
      onDragEnter={(event) => { if (Array.from(event.dataTransfer.types).includes("Files")) { event.preventDefault(); dragDepth.current += 1; setDragging(true); } }}
      onDragOver={(event) => { if (Array.from(event.dataTransfer.types).includes("Files")) event.preventDefault(); }}
      onDragLeave={() => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDragging(false); }}
      onDrop={(event) => { if (Array.from(event.dataTransfer.types).includes("Files")) { event.preventDefault(); dragDepth.current = 0; setDragging(false); void upload(event.dataTransfer.files); } }}>
      <input ref={fileInput} type="file" hidden multiple accept="audio/*,video/*" onChange={(event) => void upload(event.target.files)} />
      {dragging ? <div className="meetings-drop"><Upload size={28} aria-hidden="true" /><strong>Drop your recordings here</strong><span>Audio or video · add several at once</span></div> : null}

      {!selected ? <>
        <PageHeader eyebrow="Conversation intelligence" title="Meetings" icon={<AudioLines size={22} aria-hidden="true" />} lede="Every conversation, ready to return to. Find the words, decisions, and next steps."
          actions={<button type="button" className="ui-btn is-primary" onClick={() => fileInput.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle size={15} className="conversation-spin" aria-hidden="true" /> : <Upload size={15} aria-hidden="true" />}{uploading ? "Adding recordings…" : "Add recordings"}</button>} />
        {error || listError ? <Notice kind="error" action={listError ? "Try again" : undefined} onAction={() => void refreshList()} onDismiss={error ? () => setError(null) : undefined}>{error || listError}</Notice> : null}
        <div className="recording-library-stats" aria-label="Recording library summary">
          <div><span className="conversation-mark"><AudioLines size={19} aria-hidden="true" /></span><span><strong>{loaded ? counts.all : "—"}</strong><small>Recordings in your library</small></span></div>
          <div><span className="conversation-mark"><CheckCircle2 size={19} aria-hidden="true" /></span><span><strong>{loaded ? ready : "—"}</strong><small>Ready to revisit</small></span></div>
          <div><span className="conversation-mark"><Clock3 size={19} aria-hidden="true" /></span><span><strong>{loaded ? counts.working : "—"}</strong><small>Being prepared</small></span></div>
        </div>
        <section className="recording-catalog" aria-labelledby="recordings-title">
          <div className="recording-catalog-head"><div><h2 id="recordings-title" ref={libraryHeading} tabIndex={-1}>Your recordings</h2><p>Open a conversation to listen, review, and keep what matters.</p></div><label className="conversation-search"><Search size={17} aria-hidden="true" /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a recording…" aria-label="Search recordings" /></label></div>
          <div className="recording-filter-row"><div className="recording-filters" role="group" aria-label="Filter recordings">{FILTERS.map(([value, label]) => <button key={value} type="button" className={`ui-chip${filter === value ? " is-accent" : ""}`} aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}<small>{loaded ? counts[value] : "—"}</small></button>)}</div><span className="recording-result-count" role="status">{loaded ? `${visible.length} ${visible.length === 1 ? "recording" : "recordings"}` : "Loading library…"}</span></div>
          {!loaded ? <Skeleton rows={4} height={96} /> : visible.length ? <div className="recording-grid">{visible.map((item) => <button id={`recording-${item.id}`} key={item.id} type="button" className={`recording-card is-${item.stage}`} onClick={() => openRecording(item.id)}>
            <span className="recording-card-top"><span className="conversation-mark"><FileAudio size={20} aria-hidden="true" /></span><Status state={stageStatus(item.stage)} label={STAGE_LABEL[item.stage]} /></span>
            <span className="recording-card-title">{item.title || item.audio_filename}</span>
            <span className="recording-card-summary">{item.summary || (item.stage === "ready" ? "Your transcript and reviewed next steps are ready to explore." : item.stage === "failed" ? "This recording needs a little attention. Open it to review and retry." : "Preparing the transcript and finding the moments that matter.")}</span>
            <span className="recording-card-meta"><span><Clock3 size={12} aria-hidden="true" />{item.duration_seconds ? clock(item.duration_seconds) : "Duration pending"}</span><span>{meetingDate(item.created_at)}</span></span>
            <span className="recording-card-foot"><span>{item.account_id ? "Linked to a customer" : "Open recording"}</span><ArrowUpRight size={16} aria-hidden="true" /></span>
          </button>)}</div> : !listError ? <div className="conversation-empty"><span className="conversation-mark"><AudioLines size={30} aria-hidden="true" /></span><h3>{meetings.length ? "No recordings match" : "Good conversations deserve a place to land"}</h3><p>{meetings.length ? "Try another search or show the whole library." : "Add a meeting recording to get a speaker transcript and suggested next steps you can review."}</p><button type="button" className="ui-btn is-primary" disabled={uploading} onClick={() => { if (filtered) { setQuery(""); setFilter("all"); } else fileInput.current?.click(); }}>{filtered ? "Show all recordings" : "Add your first recording"}</button><small>MP3, WAV, M4A, MP4, or WebM</small></div> : null}
        </section>
      </> : <section className="recording-workspace" aria-label="Recording workspace">
        <div className="recording-breadcrumb"><button type="button" className="ui-btn is-quiet is-sm" onClick={backToLibrary}><ArrowLeft size={15} aria-hidden="true" />All recordings</button><ChevronRight size={13} aria-hidden="true" /><span>Recording workspace</span></div>
        {error ? <Notice kind="error" onDismiss={() => setError(null)}>{error}</Notice> : null}
        {detailError ? <Notice kind="error" title={meeting ? "Recording could not be refreshed" : "Recording unavailable"} action="Try again" onAction={() => void refreshDetail(selected)}>{detailError}</Notice> : null}
        {!meeting || !detail ? detailError ? null : <Skeleton rows={6} height={42} /> : <>
          <header className="recording-workspace-head">
            <span className="conversation-mark"><AudioLines size={25} aria-hidden="true" /></span>
            {title !== null ? <form className="meeting-title" onSubmit={(event) => { event.preventDefault(); saveTitle(); }}><input autoFocus value={title} maxLength={200} disabled={busy === "title"} onChange={(event) => setTitle(event.target.value)} onKeyDown={(event) => { if (event.key === "Escape") setTitle(null); }} aria-label="Meeting title" /><button type="submit" className="ui-btn is-primary is-sm" disabled={busy !== null || !title.trim()}>{busy === "title" ? "Saving…" : "Save"}</button><button type="button" className="ui-btn is-quiet is-sm" disabled={busy === "title"} onClick={() => setTitle(null)}>Cancel</button></form> : <div className="recording-workspace-name"><h1 ref={workspaceHeading} tabIndex={-1}>{meeting.title || meeting.audio_filename}</h1><p>{meetingDate(meeting.created_at)}{meeting.duration_seconds ? ` · ${clock(meeting.duration_seconds)}` : ""}{meeting.language ? ` · ${meeting.language.toUpperCase()}` : ""} · {detail.speakers.length} {detail.speakers.length === 1 ? "speaker" : "speakers"}</p></div>}
            <Status state={stageStatus(meeting.stage)} label={STAGE_LABEL[meeting.stage]} />
          </header>
          <div className="recording-workspace-actions"><button type="button" className="ui-btn is-sm" disabled={busy !== null} onClick={() => setTitle(meeting.title || meeting.audio_filename)}>Rename recording</button><button type="button" className="ui-btn is-quiet is-sm" disabled={!detail.turns.length} onClick={() => void navigator.clipboard.writeText(plainTranscript(detail)).then(() => toast("Transcript copied")).catch(() => setError("The transcript could not be copied. Export it to save a text file."))}><Copy size={14} aria-hidden="true" />Copy transcript</button><button type="button" className="ui-btn is-quiet is-sm" disabled={!detail.turns.length} onClick={() => downloadText(`${safeFilePart(meeting.title || "meeting")}-transcript.txt`, plainTranscript(detail))}><Download size={14} aria-hidden="true" />Export</button><button type="button" className="ui-btn is-quiet is-sm is-danger-text" disabled={busy !== null} onClick={remove}>{busy === "delete" ? "Removing…" : "Remove"}</button></div>
          <div className="recording-player"><AudioPlayer key={meeting.id} ref={player} src={meetingAudioUrl(meeting.id)} label="Listen to the recording" onTime={setPosition} onError={() => setError("That recording could not be played.")} /></div>
          <div className="recording-workspace-tabs" role="tablist" aria-label="Meeting views" onKeyDown={(event) => { if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) { event.preventDefault(); const next = event.key === "Home" ? "overview" : event.key === "End" ? "transcript" : view === "overview" ? "transcript" : "overview"; setView(next); document.getElementById(`meeting-tab-${next}`)?.focus(); } }}><button id="meeting-tab-overview" type="button" role="tab" aria-controls="meeting-panel-overview" aria-selected={view === "overview"} tabIndex={view === "overview" ? 0 : -1} onClick={() => setView("overview")}><ListChecks size={16} aria-hidden="true" />Overview{openCount > 0 ? <small>{openCount} to review</small> : null}</button><button id="meeting-tab-transcript" type="button" role="tab" aria-controls="meeting-panel-transcript" aria-selected={view === "transcript"} tabIndex={view === "transcript" ? 0 : -1} onClick={() => setView("transcript")}><AudioLines size={16} aria-hidden="true" />Transcript<small>{detail.turns.length}</small></button></div>
          <div id="meeting-panel-overview" role="tabpanel" aria-labelledby="meeting-tab-overview" hidden={view !== "overview"}><MeetingOverview key={meeting.id} detail={detail} busy={busy} onSeek={seek} onDecide={(proposal: MeetingProposal, status) => void edit(proposal.id, () => decideMeetingProposal(meeting.id, proposal.id, status), "Could not save that.")} onName={(speakerId, name) => edit(speakerId, () => nameMeetingSpeaker(meeting.id, speakerId, name), "Could not name that speaker.")} onRetry={() => void edit("retry", () => retryMeeting(meeting.id), "Could not retry.")} /></div>
          <div id="meeting-panel-transcript" role="tabpanel" aria-labelledby="meeting-tab-transcript" hidden={view !== "transcript"}><MeetingTranscript key={meeting.id} turns={detail.turns} names={names} position={position} sourceTurnId={view === "transcript" && selected === requestedMeeting ? requestedTurn : null} onSeek={seek} onCorrect={(turn: MeetingTurn, text) => edit(turn.id, () => correctMeetingTurn(meeting.id, turn.id, text), "Could not save that correction.")} /></div>
        </>}
      </section>}
    </div>
  );
}
