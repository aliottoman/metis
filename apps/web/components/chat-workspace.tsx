"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { CSSProperties, FormEvent, KeyboardEvent, PointerEvent as ReactPointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ArtifactViewer } from "@/components/artifact-viewer";
import { CommandPicker, type PickerOption } from "@/components/command-picker";
import { CustomerDashboardSnippet } from "@/components/customer-dashboard-snippet";
import { MarkdownContent } from "@/components/markdown-content";
import { RunTimeline } from "@/components/run-timeline";
import { MetisCompanion } from "@/components/metis-companion";
import { MetisMark, MetisWordmark } from "@/components/metis-mark";
import { ModelControl } from "@/components/model-control";
import {
  ApiError,
  cancelRun,
  createConversation,
  createCorpusSource,
  createCustomerNote,
  decideRun,
  getConversation,
  getConversationProject,
  getModelPreference,
  getLocalModelSession,
  listCustomers,
  listCorpusSources,
  listProjectWorkspaces,
  listRecoverableRuns,
  openProjectWorkspace,
  reindexCorpusSource,
  scanAssets,
  sendMessage,
  setCorpusConsent,
  setModelPreference,
  submitFeedback,
  uploadFile,
} from "@/lib/api";
import { rememberConversation } from "@/lib/recent-conversations";
import { mergeAssistantReasoning, mergeAssistantRunEvent, messageBelongsToRun } from "@/lib/run-history";
import { attachmentBadge, CHAT_ATTACHMENT_ACCEPT } from "@/lib/attachments";
import {
  ATTACHMENT_TEXT_BUDGET_BYTES,
  attachableFiles,
  droppedDirectories,
  findProjectForFolder,
  findSourceForFolder,
  formatByteSize,
  guessRootPath,
  looseFiles,
  MAX_ATTACHABLE_FILES,
  scanFolderEntry,
  suggestCorpusKind,
  totalBytes,
  type FolderScan,
} from "@/lib/folder-drop";
import type { ArtifactRef, AttachmentRef, ChatMessage, CorpusSource, CustomerAccount, KnowledgeScope, LocalModelSession, ModelPreference, ProjectMode, ProjectWorkspace, RecoverableRun, RunEventV1 } from "@/lib/types";
import { useRunEvents } from "@/hooks/use-run-events";

function UserAvatar() {
  return (
    <span className="userAvatarGlyph" aria-hidden="true">
      <i className="userAvatarHead" />
      <i className="userAvatarBody" />
    </span>
  );
}

/** The last complete-looking line of thinking, for the collapsed preview. */
function reasoningTail(reasoning: string): string {
  const lines = reasoning.split("\n").map((line) => line.trim()).filter(Boolean);
  return lines[lines.length - 1] ?? "";
}

/**
 * The model's thinking, collapsed by default. While the run is live the header
 * shows the newest line so there is a sense of progress without a wall of text;
 * expanding shows everything received so far.
 */
function ReasoningPanel({ reasoning, live }: { reasoning: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Follow the newest thinking only while expanded and still streaming.
  useEffect(() => {
    if (!open || !live) return;
    const body = bodyRef.current;
    if (body) body.scrollTop = body.scrollHeight;
  }, [live, open, reasoning]);

  const preview = live ? reasoningTail(reasoning) : `${reasoning.trim().split(/\s+/).length} words of thinking`;
  return (
    <section className={`reasoningPanel ${live ? "isLive" : ""} ${open ? "isOpen" : ""}`}>
      <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span className="reasoningGlyph" aria-hidden="true"><i /><i /><i /></span>
        <span className="reasoningLabel">{live ? "Thinking" : "Thought process"}</span>
        <span className="reasoningPreview">{preview}</span>
        <span className="reasoningChevron" aria-hidden="true">⌄</span>
      </button>
      {open ? <div className="reasoningBody" ref={bodyRef}>{reasoning}</div> : null}
    </section>
  );
}

// Sent verbatim through the composer's send path when the user asks Metis to
// distill a finished answer into a governed, reusable tool definition.
const TOOL_BUILD_PROMPT = "Turn this repeatable process into a reusable tool.";
const DEFAULT_ACTIVITY_WIDTH = 350;
const MIN_ACTIVITY_WIDTH = 300;
const MAX_ACTIVITY_WIDTH = 620;

function clampActivityWidth(value: number): number {
  return Math.min(MAX_ACTIVITY_WIDTH, Math.max(MIN_ACTIVITY_WIDTH, value));
}

function stringFrom(payload: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "string" && value) return value;
    if (value && typeof value === "object") {
      const nested = value as Record<string, unknown>;
      if (typeof nested.content === "string") return nested.content;
      if (typeof nested.text === "string") return nested.text;
    }
  }
  return undefined;
}

function artifactsFrom(payload: Record<string, unknown>): ArtifactRef[] {
  const raw = Array.isArray(payload.artifacts)
    ? payload.artifacts
    : payload.artifact
      ? [payload.artifact]
      : (payload.id ?? payload.artifact_id) && (payload.filename ?? payload.name)
        ? [payload]
        : [];
  return raw.flatMap((value) => {
    if (!value || typeof value !== "object") return [];
    const item = value as Record<string, unknown>;
    const id = item.id ?? item.artifact_id;
    if (!id) return [];
    return [{
      id: String(id),
      name: String(item.name ?? item.filename ?? "artifact"),
      media_type: item.media_type ? String(item.media_type) : item.content_type ? String(item.content_type) : undefined,
      size: Number.isFinite(Number(item.size ?? item.size_bytes)) ? Number(item.size ?? item.size_bytes) : undefined,
      sha256: item.sha256 ? String(item.sha256) : undefined,
      download_url: item.download_url ? String(item.download_url) : undefined,
    }];
  });
}

export function ChatWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedConversationId = searchParams.get("conversation");
  const requestedRunId = searchParams.get("run");
  const newRequestToken = searchParams.get("new");
  const [conversationId, setConversationId] = useState<string | null>(requestedConversationId);
  const [conversationTitle, setConversationTitle] = useState("New conversation");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<AttachmentRef[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactRef[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(requestedRunId);
  const [recoverableRuns, setRecoverableRuns] = useState<RecoverableRun[]>([]);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [timelineOpen, setTimelineOpen] = useState(false);
  const [timelineWidth, setTimelineWidth] = useState(DEFAULT_ACTIVITY_WIDTH);
  const [timelineResizing, setTimelineResizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decidedApprovals, setDecidedApprovals] = useState<Set<string>>(new Set());
  const [decisionBusy, setDecisionBusy] = useState<string | null>(null);
  const [feedbackMode, setFeedbackMode] = useState<"idle" | "correcting" | "sent">("idle");
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [correction, setCorrection] = useState("");
  const [stageLabel, setStageLabel] = useState<string | null>(null);
  const [modelPreference, setModelPreferenceState] = useState<ModelPreference | null>(null);
  const [localModelSession, setLocalModelSession] = useState<LocalModelSession | null>(null);
  const [providerSaving, setProviderSaving] = useState(false);
  const [projects, setProjects] = useState<ProjectWorkspace[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectMode, setProjectMode] = useState<ProjectMode>("grok_bootstrap_local");
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [projectOpening, setProjectOpening] = useState(false);
  const [knowledgeScope, setKnowledgeScope] = useState<KnowledgeScope>("auto");
  const [customers, setCustomers] = useState<CustomerAccount[]>([]);
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);
  const [customerPickerOpen, setCustomerPickerOpen] = useState(false);
  // The slash menu. A "/" that starts the message or follows a space opens it;
  // the text after the slash filters it, and picking a command strips that
  // fragment back out of the draft so it never gets sent.
  const [slashOpen, setSlashOpen] = useState(false);
  const [slashQuery, setSlashQuery] = useState("");
  // Answers already written back to the account, so the button can say so
  // rather than silently accepting the same answer twice.
  const [savedToAccount, setSavedToAccount] = useState<Set<string>>(new Set());
  const [savingToAccount, setSavingToAccount] = useState<string | null>(null);
  // A dropped folder is routed, never uploaded: the scan below is local, and
  // only the option the user picks does anything.
  const [folderDrop, setFolderDrop] = useState<{ scan: FolderScan; ignored: string[] } | null>(null);
  const [folderScanning, setFolderScanning] = useState(false);
  const [folderBusy, setFolderBusy] = useState<"project" | "knowledge" | "attach" | "rescan" | null>(null);
  const [folderPath, setFolderPath] = useState("");
  const [folderKind, setFolderKind] = useState<CorpusSource["kind"]>("code");
  const [folderConsent, setFolderConsent] = useState(false);
  const [folderNotice, setFolderNotice] = useState<string | null>(null);
  const [corpusSources, setCorpusSources] = useState<CorpusSource[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composerWidthRef = useRef(0);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const loadedConversationRef = useRef<string | null>(null);
  const latestRunRef = useRef<string | null>(requestedRunId);
  const handledNewRequestRef = useRef<string | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const timelineWidthRef = useRef(DEFAULT_ACTIVITY_WIDTH);
  const timelineResizingRef = useRef(false);
  const timelineResizeOriginRef = useRef({ pointerX: 0, width: DEFAULT_ACTIVITY_WIDTH });
  const workspaceGenerationRef = useRef(0);

  const resetConversationState = useCallback(() => {
    workspaceGenerationRef.current += 1;
    setConversationId(null);
    setConversationTitle("New conversation");
    loadedConversationRef.current = null;
    latestRunRef.current = null;
    setMessages([]);
    setDraft("");
    setAttachments([]);
    setArtifacts([]);
    setActiveRunId(null);
    setFeedbackMode("idle");
    setFeedbackBusy(false);
    setCorrection("");
    setDecidedApprovals(new Set());
    setDecisionBusy(null);
    setStageLabel(null);
    setTimelineOpen(false);
    timelineResizingRef.current = false;
    setTimelineResizing(false);
    setLoadingConversation(false);
    setSending(false);
    setUploading(false);
    setDragActive(false);
    setFolderDrop(null);
    setFolderScanning(false);
    setFolderBusy(null);
    setFolderNotice(null);
    setProjectPickerOpen(false);
    setProjectOpening(false);
    setSelectedCustomerId(null);
    setCustomerPickerOpen(false);
    setSavedToAccount(new Set());
    setSavingToAccount(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (textareaRef.current) textareaRef.current.style.height = "";
  }, []);

  // The composer's height follows the draft rather than the keystroke, so
  // sending, restoring a failed send, and "Edit & retry" all resize it too.
  // Measuring from zero is what lets it shrink again, not only grow.
  const resizeComposer = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0";
    textarea.style.height = textarea.value ? `${Math.min(textarea.scrollHeight, 180)}px` : "";
    composerWidthRef.current = textarea.clientWidth;
  }, []);

  // A frame late, because scrollHeight measured mid-layout — while the welcome
  // screen is still collapsing into the message list — reports the wrapping of
  // a near-zero-width box and pins the composer to its 180px maximum.
  useEffect(() => {
    const frame = requestAnimationFrame(resizeComposer);
    return () => cancelAnimationFrame(frame);
  }, [draft, resizeComposer]);

  // Width changes rewrap the text, so the measured height is stale: the sidebar
  // collapsing, the activity drawer opening, or the window resizing all count.
  // Height-only changes are this effect's own doing and must not re-trigger it.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (textareaRef.current?.clientWidth !== composerWidthRef.current) resizeComposer();
    });
    observer.observe(textarea);
    return () => observer.disconnect();
  }, [resizeComposer]);

  useEffect(() => {
    const stored = window.localStorage.getItem("metis.knowledgeScope");
    if (stored === "auto" || stored === "notion") setKnowledgeScope(stored);
  }, []);

  useEffect(() => {
    let mounted = true;
    void getLocalModelSession()
      .then((value) => mounted && setLocalModelSession(value))
      .catch(() => undefined);
    const listener = (event: Event) => {
      const value = (event as CustomEvent<LocalModelSession>).detail;
      if (value) setLocalModelSession(value);
    };
    window.addEventListener("metis:model-session", listener);
    return () => {
      mounted = false;
      window.removeEventListener("metis:model-session", listener);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    void listCustomers()
      .then((items) => mounted && setCustomers(items))
      .catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  function chooseKnowledgeScope(scope: KnowledgeScope) {
    setKnowledgeScope(scope);
    window.localStorage.setItem("metis.knowledgeScope", scope);
  }

  useEffect(() => {
    let mounted = true;
    const loadProvider = () => {
      void getModelPreference()
        .then((preference) => mounted && setModelPreferenceState(preference))
        .catch(() => undefined);
    };
    loadProvider();
    window.addEventListener("focus", loadProvider);
    return () => {
      mounted = false;
      window.removeEventListener("focus", loadProvider);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const loadProjects = () => {
      void listProjectWorkspaces()
        .then((items) => mounted && setProjects(items))
        .catch(() => undefined);
    };
    loadProjects();
    window.addEventListener("focus", loadProjects);
    return () => {
      mounted = false;
      window.removeEventListener("focus", loadProjects);
    };
  }, []);

  /** Resolves true only when a project actually opened, so callers that own
   *  surrounding UI (the folder-drop sheet) know whether to close it. */
  async function chooseProject(projectId: string | null): Promise<boolean> {
    if (projectOpening || runActive) return false;
    if (!projectId) {
      setSelectedProjectId(null);
      setProjectPickerOpen(false);
      return false;
    }
    const target = projects.find((project) => project.id === projectId);
    if (!modelPreference?.oci_available && (!target?.initialized || projectMode === "grok_continuous")) {
      setError("Project mode needs Grok through OCI for the initial repository map. Configure OCI in Settings first.");
      return false;
    }
    setProjectOpening(true);
    setError(null);
    try {
      const opened = await openProjectWorkspace(projectId, projectMode);
      setProjects((current) => current.map((item) => item.id === opened.id ? opened : item));
      setSelectedProjectId(opened.id);
      setProjectPickerOpen(false);
      return true;
    } catch (openError) {
      setError(openError instanceof Error ? openError.message : "Metis could not open that project.");
      return false;
    } finally {
      setProjectOpening(false);
    }
  }

  async function chooseProjectMode(mode: ProjectMode) {
    if (projectOpening || runActive || projectMode === mode) return;
    if (mode === "grok_continuous" && !modelPreference?.oci_available) {
      setError("Keep Grok needs OCI Responses to be available.");
      return;
    }
    const previousMode = projectMode;
    setProjectMode(mode);
    if (!selectedProjectId) return;
    setProjectOpening(true);
    setError(null);
    try {
      const opened = await openProjectWorkspace(selectedProjectId, mode);
      setProjects((current) => current.map((item) => item.id === opened.id ? opened : item));
    } catch (openError) {
      setProjectMode(previousMode);
      setError(openError instanceof Error ? openError.message : "The project mode could not be changed.");
    } finally {
      setProjectOpening(false);
    }
  }

  async function chooseChatProvider(provider: "local" | "oci") {
    if (providerSaving || modelPreference?.provider === provider) return;
    if (provider === "oci" && !modelPreference?.oci_available) {
      setError("Cloud reasoning is not configured yet. Add the OCI project settings before selecting it.");
      return;
    }
    setProviderSaving(true);
    setError(null);
    try {
      const current = modelPreference ?? {
        mode: "split" as const,
        model: null,
        provider: "local" as const,
        oci_tools: ["code_interpreter" as const],
        oci_available: false,
      };
      setModelPreferenceState(await setModelPreference(
        current.mode,
        current.model,
        provider,
        current.oci_tools,
      ));
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "The model route could not be changed.");
    } finally {
    setProviderSaving(false);
    }
  }

  useEffect(() => {
    const stored = Number(window.localStorage.getItem("metis.activityWidth"));
    if (!Number.isFinite(stored)) return;
    const next = clampActivityWidth(stored);
    timelineWidthRef.current = next;
    setTimelineWidth(next);
  }, []);

  useEffect(() => {
    let mounted = true;
    const load = () => {
      void listRecoverableRuns()
        .then((runs) => {
          if (mounted) setRecoverableRuns(runs);
        })
        .catch(() => {
          // Recovery discovery is additive; ordinary chat remains available offline.
        });
    };
    load();
    window.addEventListener("focus", load);
    return () => {
      mounted = false;
      window.removeEventListener("focus", load);
    };
  }, []);

  useEffect(() => {
    if (newRequestToken) {
      if (handledNewRequestRef.current !== newRequestToken) {
        handledNewRequestRef.current = newRequestToken;
        resetConversationState();
      }
      router.replace("/");
      window.setTimeout(() => textareaRef.current?.focus(), 0);
      return;
    }
    handledNewRequestRef.current = null;
    if (!requestedConversationId || loadedConversationRef.current === requestedConversationId) return;
    let mounted = true;
    setConversationId(requestedConversationId);
    setConversationTitle("Opening conversation…");
    setMessages([]);
    setArtifacts([]);
    setActiveRunId(requestedRunId);
    if (requestedRunId) setTimelineOpen(true);
    setFeedbackMode("idle");
    setCorrection("");
    setDecidedApprovals(new Set());
    setLoadingConversation(true);
    setError(null);
    void Promise.all([
      getConversation(requestedConversationId),
      getConversationProject(requestedConversationId),
    ])
      .then(([conversation, projectSession]) => {
        if (!mounted) return;
        setMessages(conversation.messages);
        setConversationTitle(conversation.title || "Conversation");
        const restoredRunId = requestedRunId ?? conversation.latest_run_id ?? null;
        latestRunRef.current = conversation.latest_run_id ?? null;
        setActiveRunId(restoredRunId);
        loadedConversationRef.current = requestedConversationId;
        rememberConversation(conversation);
        if (projectSession) {
          setSelectedProjectId(projectSession.projectId);
          setProjectMode(projectSession.mode);
        } else {
          setSelectedProjectId(null);
        }
      })
      .catch((loadError) => {
        if (!mounted) return;
        // The conversation named in the URL is gone — an interrupted run, a
        // restored tab, a database that moved on. Drop the dead ids instead of
        // keeping them: the run stream would otherwise poll a 404 forever,
        // leaving the activity drawer stuck on "Reconnecting" behind a Stop
        // button for a run that already ended. The reset clears the error, so
        // the explanation is restored after it.
        resetConversationState();
        setError(loadError instanceof Error ? loadError.message : "Could not open this conversation.");
        // `router.replace("/")` does not strip an existing query string here, so
        // a restored tab would surface this banner on every single reload.
        window.history.replaceState(null, "", "/");
      })
      .finally(() => mounted && setLoadingConversation(false));
    return () => { mounted = false; };
  }, [newRequestToken, requestedConversationId, requestedRunId, resetConversationState, router]);

  useEffect(() => {
    if (!requestedConversationId) return;
    if (!requestedRunId) {
      if (loadedConversationRef.current === requestedConversationId) {
        setArtifacts([]);
        setActiveRunId(latestRunRef.current);
      }
      return;
    }
    setActiveRunId(requestedRunId);
    setArtifacts([]);
    setDecidedApprovals(new Set());
    setStageLabel(null);
    setTimelineOpen(true);
  }, [requestedConversationId, requestedRunId]);

  const handleRunEvent = useCallback((event: RunEventV1) => {
    const type = event.type.toLowerCase();
    const text = stringFrom(event.payload, "delta", "text_delta", "content_delta", "content", "message", "final", "response");
    const eventArtifacts = artifactsFrom(event.payload);
    if (eventArtifacts.length) {
      setArtifacts((current) => {
        const ids = new Set(current.map((item) => item.id));
        return [...current, ...eventArtifacts.filter((item) => !ids.has(item.id))];
      });
    }

    // Descriptive stage narration for the in-flight indicator. Once the answer
    // starts streaming (or the run ends) the stage line is no longer shown, so
    // clear it to avoid a stale label leaking into the next run.
    // Thinking is its own channel: it neither becomes answer text nor ends the
    // stage narration, since the answer has not started yet while it streams.
    if (type === "message.reasoning") {
      setMessages((current) => mergeAssistantReasoning(current, event.run_id, text ?? ""));
      return;
    }

    if (type === "stage.entered") {
      setStageLabel(stringFrom(event.payload, "label") ?? null);
    } else if (type.includes("delta") || type.includes("failed") || type.includes("completed") || type.includes("cancelled")) {
      setStageLabel(null);
    }

    if (type.includes("delta") || type.includes("failed") || ["assistant.message", "message.completed", "run.completed", "completed"].includes(type)) {
      setMessages((current) => mergeAssistantRunEvent(current, event, text));
    }
  }, []);

  const { events, connection, error: streamError, reconnect } = useRunEvents(activeRunId, handleRunEvent);

  useEffect(() => {
    if (scrollFrameRef.current !== null) cancelAnimationFrame(scrollFrameRef.current);
    scrollFrameRef.current = requestAnimationFrame(() => {
      const streaming = messages.some((message) => message.streaming);
      messageEndRef.current?.scrollIntoView({ behavior: streaming ? "auto" : "smooth", block: "end" });
      scrollFrameRef.current = null;
    });
    return () => {
      if (scrollFrameRef.current !== null) cancelAnimationFrame(scrollFrameRef.current);
    };
  }, [messages]);

  const hasMessages = messages.length > 0 || loadingConversation;
  const runActive = Boolean(activeRunId) && !["closed", "error"].includes(connection);

  const droppedName = folderDrop?.scan.name ?? "";
  const droppedFiles = folderDrop?.scan.files;
  const matchedProject = useMemo(
    () => droppedName ? findProjectForFolder(droppedName, projects) : undefined,
    [droppedName, projects],
  );
  const matchedSource = useMemo(
    () => droppedName ? findSourceForFolder(droppedName, corpusSources) : undefined,
    [corpusSources, droppedName],
  );
  const droppedAttachable = useMemo(
    () => droppedFiles ? attachableFiles(droppedFiles, CHAT_ATTACHMENT_ACCEPT) : [],
    [droppedFiles],
  );
  const droppedAttachableBytes = totalBytes(droppedAttachable);
  // The composer uploads one request per file and the host caps the aggregate
  // text, so a folder past either bound is pointed at indexing instead.
  const attachBlockedReason = !droppedAttachable.length
    ? "Nothing in this folder is a format the composer can attach."
    : droppedAttachable.length > MAX_ATTACHABLE_FILES
      ? `${droppedAttachable.length} attachable files is past the ${MAX_ATTACHABLE_FILES} this composer uploads at once — index it instead.`
      : droppedAttachableBytes > ATTACHMENT_TEXT_BUDGET_BYTES
        ? `${formatByteSize(droppedAttachableBytes)} is past the ${formatByteSize(ATTACHMENT_TEXT_BUDGET_BYTES)} context budget — index it instead.`
        : null;
  // Indexing sends the folder's text to the cloud embedder, so an unconsented
  // source has no runnable action until the box is ticked.
  const knowledgeLabel = folderBusy === "knowledge"
    ? "Working…"
    : matchedSource
      ? matchedSource.consent ? "Reindex source" : folderConsent ? "Grant consent & index" : "Consent required"
      : folderConsent ? "Add & index" : "Add source";
  const knowledgeDisabled = Boolean(folderBusy)
    || (!matchedSource && !folderPath.trim())
    || (Boolean(matchedSource) && !matchedSource?.consent && !folderConsent);

  async function addFiles(files: FileList | File[]) {
    const items = Array.from(files);
    if (!items.length) return;
    const generation = workspaceGenerationRef.current;
    setUploading(true);
    setError(null);
    try {
      const results = await Promise.allSettled(items.map(uploadFile));
      const uploaded = results.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
      if (generation !== workspaceGenerationRef.current) return;
      if (uploaded.length) setAttachments((current) => [...current, ...uploaded]);
      const failed = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
      if (failed.length) {
        const detail = failed[0]?.reason instanceof Error ? failed[0].reason.message : "Unsupported or unreadable file.";
        setError(`${failed.length} ${failed.length === 1 ? "file" : "files"} could not be attached. ${detail}`);
      }
    } catch (uploadError) {
      if (generation !== workspaceGenerationRef.current) return;
      setError(uploadError instanceof Error ? uploadError.message : "Attachments could not be uploaded.");
    } finally {
      if (generation === workspaceGenerationRef.current) {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    }
  }

  function closeFolderDrop() {
    setFolderDrop(null);
    setFolderNotice(null);
    setFolderBusy(null);
  }

  /**
   * A drop tells us a folder's name and contents but never its absolute path,
   * so the sheet resolves what it can against the catalogs already loaded and
   * asks only for what is genuinely unknowable in a browser.
   */
  async function routeFolderDrop(directories: FileSystemDirectoryEntry[]) {
    const [dropped, ...ignored] = directories;
    if (!dropped) return;
    const generation = workspaceGenerationRef.current;
    setFolderScanning(true);
    setFolderNotice(null);
    setError(null);
    setProjectPickerOpen(false);
    try {
      const [scan, sources] = await Promise.all([
        scanFolderEntry(dropped),
        // A missing corpus only costs the path guess; it must not block routing.
        listCorpusSources().catch(() => [] as CorpusSource[]),
      ]);
      if (generation !== workspaceGenerationRef.current) return;
      const existing = findSourceForFolder(scan.name, sources);
      setCorpusSources(sources);
      setFolderPath(
        existing?.root_path
        ?? guessRootPath(scan.name, sources.map((item) => item.root_path))
        ?? "",
      );
      setFolderKind(existing?.kind ?? suggestCorpusKind(scan.files));
      setFolderConsent(false);
      setFolderDrop({ scan, ignored: ignored.map((entry) => entry.name) });
    } catch (scanError) {
      if (generation !== workspaceGenerationRef.current) return;
      setError(scanError instanceof Error ? scanError.message : "That folder could not be read.");
    } finally {
      if (generation === workspaceGenerationRef.current) setFolderScanning(false);
    }
  }

  async function openDroppedProject() {
    if (!matchedProject) return;
    setFolderBusy("project");
    const opened = await chooseProject(matchedProject.id);
    setFolderBusy(null);
    if (opened) closeFolderDrop();
  }

  async function rescanForDroppedProject() {
    setFolderBusy("rescan");
    setFolderNotice(null);
    try {
      await scanAssets();
      const found = await listProjectWorkspaces();
      setProjects(found);
      if (!findProjectForFolder(droppedName, found)) {
        setFolderNotice(`Still no project named “${droppedName}”. Projects are only discovered under the folders configured in Settings.`);
      }
    } catch (rescanError) {
      setFolderNotice(rescanError instanceof Error ? rescanError.message : "The project catalog could not be refreshed.");
    } finally {
      setFolderBusy(null);
    }
  }

  async function indexDroppedFolder() {
    if (!folderDrop) return;
    const path = folderPath.trim();
    if (!matchedSource && !path) {
      setFolderNotice("Metis needs the folder's full path on disk before it can index it.");
      return;
    }
    setFolderBusy("knowledge");
    setFolderNotice(null);
    try {
      const source = matchedSource ?? await createCorpusSource(path, folderDrop.scan.name, folderKind);
      const decided = source.consent || !folderConsent
        ? source
        : await setCorpusConsent(source.id, true, "granted from a composer folder drop");
      setCorpusSources((current) => current.some((item) => item.id === decided.id)
        ? current.map((item) => item.id === decided.id ? decided : item)
        : [...current, decided]);
      if (!decided.consent) {
        setFolderNotice("Added as a source. It stays unindexed until you allow cloud embedding for it.");
        return;
      }
      const result = await reindexCorpusSource(decided.id);
      setFolderNotice(`Indexed ${result.files_indexed} file(s) into ${result.chunks} chunk(s). Future answers can cite this folder.`);
    } catch (indexError) {
      setFolderNotice(indexError instanceof Error ? indexError.message : "That folder could not be indexed.");
    } finally {
      setFolderBusy(null);
    }
  }

  async function attachDroppedFiles() {
    if (!droppedAttachable.length || attachBlockedReason) return;
    setFolderBusy("attach");
    await addFiles(droppedAttachable.map((item) => item.file));
    setFolderBusy(null);
    closeFolderDrop();
  }

  async function submit(event?: FormEvent, overrideContent?: string) {
    event?.preventDefault();
    // An override lets in-thread affordances (e.g. "Build a tool from this")
    // reuse this exact send path without disturbing the composer draft.
    const usingOverride = typeof overrideContent === "string";
    const content = (overrideContent ?? draft).trim();
    const outgoingAttachments = usingOverride ? [] : attachments;
    if ((!content && !outgoingAttachments.length) || sending || uploading || runActive) return;
    const generation = workspaceGenerationRef.current;
    setSending(true);
    setError(null);

    const submittedContent = content || "Please review the attached file or files.";
    const userMessage: ChatMessage = {
      id: `optimistic-${crypto.randomUUID()}`,
      role: "user",
      content: submittedContent,
      attachments: outgoingAttachments,
      created_at: new Date().toISOString(),
    };
    setMessages((current) => [...current, userMessage]);
    if (!usingOverride) {
      setDraft("");
      setAttachments([]);
    }
    setArtifacts([]);

    try {
      let targetConversationId = conversationId;
      if (!targetConversationId) {
        const title = content.slice(0, 54) || outgoingAttachments[0]?.name || "New conversation";
        const created = await createConversation(title);
        if (generation !== workspaceGenerationRef.current) return;
        if (!created.id) throw new Error("The API did not return a conversation ID.");
        targetConversationId = created.id;
        setConversationId(created.id);
        setConversationTitle(title);
        loadedConversationRef.current = created.id;
        rememberConversation({ ...created, title });
        router.replace(`/?conversation=${encodeURIComponent(created.id)}`);
      }
      const run = await sendMessage(
        targetConversationId,
        submittedContent,
        userMessage.attachments ?? [],
        selectedProjectId ? { id: selectedProjectId, mode: projectMode } : null,
        knowledgeScope,
        selectedCustomerId,
      );
      if (generation !== workspaceGenerationRef.current) return;
      if (!run.run_id) throw new Error("The API did not return a run ID.");
      setDecidedApprovals(new Set());
      setActiveRunId(run.run_id);
      latestRunRef.current = run.run_id;
      setFeedbackMode("idle");
      setCorrection("");
      setTimelineOpen(true);
      router.replace(`/?conversation=${encodeURIComponent(targetConversationId)}&run=${encodeURIComponent(run.run_id)}`);
      setMessages((current) => [...current, { id: `assistant-${run.run_id}`, run_id: run.run_id, role: "assistant", content: "", streaming: true }]);
    } catch (sendError) {
      if (generation !== workspaceGenerationRef.current) return;
      setMessages((current) => current.filter((message) => message.id !== userMessage.id));
      if (!usingOverride) {
        setDraft(content);
        setAttachments(userMessage.attachments ?? []);
      }
      setError(sendError instanceof Error ? sendError.message : "The message could not be sent.");
    } finally {
      if (generation === workspaceGenerationRef.current) setSending(false);
    }
  }

  async function handleDecision(approvalId: string, decision: "approve" | "reject") {
    if (!activeRunId) return;
    setDecisionBusy(approvalId);
    setError(null);
    try {
      await decideRun(activeRunId, approvalId, decision);
      setDecidedApprovals((current) => new Set(current).add(approvalId));
      setRecoverableRuns((current) => current.filter((item) => item.run.id !== activeRunId));
      if (connection === "closed" || connection === "error") reconnect();
    } catch (decisionError) {
      setError(decisionError instanceof ApiError ? decisionError.message : "The decision could not be recorded.");
    } finally {
      setDecisionBusy(null);
    }
  }

  async function handleCancel() {
    if (!activeRunId) return;
    try {
      await cancelRun(activeRunId);
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : "The run could not be cancelled.");
    }
  }

  async function handleFeedback(rating: "positive" | "negative") {
    if (!activeRunId || feedbackBusy) return;
    if (rating === "negative" && feedbackMode !== "correcting") {
      setFeedbackMode("correcting");
      return;
    }
    setFeedbackBusy(true);
    setError(null);
    try {
      await submitFeedback(activeRunId, rating, rating === "negative" ? correction.trim() : undefined);
      setFeedbackMode("sent");
      setCorrection("");
    } catch (feedbackError) {
      setError(feedbackError instanceof Error ? feedbackError.message : "Feedback could not be recorded.");
    } finally {
      setFeedbackBusy(false);
    }
  }

  const SLASH_TRIGGER = /(^|\s)\/([a-zA-Z]*)$/;

  function onDraftChange(value: string) {
    setDraft(value);
    const match = SLASH_TRIGGER.exec(value);
    setSlashOpen(Boolean(match) && !runActive);
    setSlashQuery(match?.[2] ?? "");
  }

  /** Remove the "/fragment" the user was typing once a command is chosen. */
  function clearSlashFragment() {
    setDraft((current) => current.replace(SLASH_TRIGGER, "$1"));
    setSlashOpen(false);
    setSlashQuery("");
    textareaRef.current?.focus();
  }

  function runSlashCommand(id: string) {
    clearSlashFragment();
    if (id === "customer") { setCustomerPickerOpen(true); setProjectPickerOpen(false); }
    if (id === "project") { setProjectPickerOpen(true); setCustomerPickerOpen(false); }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  function clearFailedResponse(messageId: string, runId?: string) {
    setMessages((current) => current.filter((message) => message.id !== messageId));
    setError(null);
    if (runId && runId === activeRunId) {
      setActiveRunId(null);
      setArtifacts([]);
      setStageLabel(null);
      setTimelineOpen(false);
      router.replace(conversationId ? `/?conversation=${encodeURIComponent(conversationId)}` : "/");
    }
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  }

  function startFreshConversation() {
    const token = crypto.randomUUID();
    handledNewRequestRef.current = token;
    resetConversationState();
    router.push(`/?new=${token}`);
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  }

  function rememberTimelineWidth(width: number) {
    const next = clampActivityWidth(width);
    timelineWidthRef.current = next;
    setTimelineWidth(next);
    window.localStorage.setItem("metis.activityWidth", String(next));
  }

  function handleTimelineResizeStart(event: ReactPointerEvent<HTMLDivElement>) {
    if (!timelineOpen || event.button !== 0) return;
    timelineResizeOriginRef.current = { pointerX: event.clientX, width: timelineWidthRef.current };
    timelineResizingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    setTimelineResizing(true);
  }

  function handleTimelineResizeMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!timelineResizingRef.current) return;
    const next = clampActivityWidth(
      timelineResizeOriginRef.current.width + timelineResizeOriginRef.current.pointerX - event.clientX,
    );
    timelineWidthRef.current = next;
    setTimelineWidth(next);
  }

  function handleTimelineResizeEnd(event: ReactPointerEvent<HTMLDivElement>) {
    if (!timelineResizingRef.current) return;
    timelineResizingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    window.localStorage.setItem("metis.activityWidth", String(timelineWidthRef.current));
    setTimelineResizing(false);
  }

  function handleTimelineResizeKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? 1 : event.key === "ArrowRight" ? -1 : 0;
    if (direction) rememberTimelineWidth(timelineWidthRef.current + direction * 16);
  }

  const latestAssistant = useMemo(() => [...messages].reverse().find((message) => message.role === "assistant"), [messages]);
  const latestUser = useMemo(() => [...messages].reverse().find((message) => message.role === "user"), [messages]);
  const latestRunEvent = useMemo(
    () => events.reduce<RunEventV1 | null>(
      (latest, event) => !latest || event.sequence > latest.sequence ? event : latest,
      null,
    ),
    [events],
  );
  const compactActivity = useMemo(() => {
    const type = latestRunEvent?.type ?? "";
    const countLabel = `${events.length} ${events.length === 1 ? "step" : "steps"}`;
    if (type.includes("approval") || type.includes("interrupt")) {
      return { label: "Needs approval", detail: countLabel, tone: "attention", live: false };
    }
    if (type.includes("fail") || type.includes("error")) {
      return { label: "Run interrupted", detail: countLabel, tone: "danger", live: false };
    }
    if (type.includes("cancel")) {
      return { label: "Run stopped", detail: countLabel, tone: "neutral", live: false };
    }
    if (runActive) {
      return {
        label: stageLabel ?? "Working on your request",
        detail: events.length ? `${countLabel} · live` : "Starting securely",
        tone: "live",
        live: true,
      };
    }
    return {
      label: type.includes("complete") ? "Completed" : "Activity",
      detail: countLabel,
      tone: "success",
      live: false,
    };
  }, [events.length, latestRunEvent, runActive, stageLabel]);

  // What the companion is feeling. Derived from the same signals the run
  // timeline uses, so the creature can never contradict the activity panel.
  const companionMood = useMemo<"idle" | "listening" | "thinking" | "done" | "trouble">(() => {
    if (error) return "trouble";
    if (runActive) return "thinking";
    const type = latestRunEvent?.type ?? "";
    if (type.includes("fail") || type.includes("error")) return "trouble";
    if (type.includes("complete")) return "done";
    if (draft.trim()) return "listening";
    return "idle";
  }, [draft, error, latestRunEvent, runActive]);

  const companionLabel = useMemo(() => {
    if (companionMood === "trouble") return "Something interrupted this";
    if (companionMood === "thinking") return stageLabel ?? "Working on it";
    if (companionMood === "listening") return "Listening";
    if (companionMood === "done") return "Done";
    return "Ready";
  }, [companionMood, stageLabel]);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );
  const selectedCustomer = useMemo(
    () => customers.find((customer) => customer.id === selectedCustomerId) ?? null,
    [customers, selectedCustomerId],
  );

  // The API already returns accounts most-recently-touched first, so the first
  // few are the ones a scoped conversation is most likely about.
  const customerOptions = useMemo<PickerOption[]>(() => {
    const active = customers.filter((customer) => customer.status === "active");
    return active.map((customer, index) => ({
      id: customer.id,
      label: customer.name,
      meta: [customer.industry, customer.region].filter(Boolean).join(" · ") || "Customer account",
      badge: customer.wins > 0
        ? `🏆 ${customer.wins}`
        : customer.open_actions > 0
          ? `${customer.open_actions} open`
          : undefined,
      keywords: customer.aliases,
      group: index < 5 ? "Recent" : "All accounts",
    }));
  }, [customers]);

  const projectOptions = useMemo<PickerOption[]>(
    () => projects.map((project) => ({
      id: project.id,
      label: project.name,
      glyph: project.initialized ? "◆" : "◇",
      meta: `${project.framework ? `${project.framework} · ` : ""}${project.initialized ? `${project.fileCount} files mapped` : "Grok map not created yet"}`,
      keywords: project.framework ? [project.framework] : undefined,
      badge: selectedProjectId === project.id ? "Active" : undefined,
    })),
    [projects, selectedProjectId],
  );

  /** Write an answer back onto the account this conversation is scoped to.
   *
   *  Scoping a chat to a customer used to be read-only: the account's facts
   *  went in and nothing came out, so anything worth keeping had to be
   *  retyped in the workbench. It lands as a note — the user's own record —
   *  never as an extracted fact, which stays a reviewed decision. */
  async function saveAnswerToAccount(message: ChatMessage) {
    if (!selectedCustomerId || !message.content.trim() || savingToAccount) return;
    setSavingToAccount(message.id);
    setError(null);
    try {
      await createCustomerNote(selectedCustomerId, {
        title: conversationTitle.slice(0, 200),
        body: message.content,
        origin: "chat",
        origin_ref: conversationId ?? "",
      });
      setSavedToAccount((current) => new Set(current).add(message.id));
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "That answer could not be saved to the account.");
    } finally {
      setSavingToAccount(null);
    }
  }

  return (
    <div
      className={`chatWorkspace ${!hasMessages ? "isEmpty" : ""} ${timelineOpen ? "timelineVisible" : ""} ${timelineResizing ? "timelineResizing" : ""}`}
      style={{ "--activity-width": `${timelineWidth}px` } as CSSProperties}
    >
      <section className="conversationPane">
        <header className="chatHeader">
          <div className="chatTitle">
            <span className="localStatus"><i />{selectedProject ? "Project" : modelPreference?.provider === "oci" ? "Cloud" : "Local"}</span>
            <strong title={conversationTitle}>{conversationTitle}</strong>
          </div>
          <div className="chatHeaderActions">
            {/* Customer and project scope live on the composer, next to the
                message they actually scope — see .composerScope below. What
                RUNS the message lives here, top-right: a provider for plain
                chat, or the project's own two modes when one is scoped. */}
            <ModelControl
              preference={modelPreference}
              onChooseProvider={(provider) => void chooseChatProvider(provider)}
              providerSaving={providerSaving}
              project={selectedProject}
              projectMode={projectMode}
              onChooseProjectMode={(mode) => void chooseProjectMode(mode)}
              projectBusy={projectOpening}
              disabled={runActive}
            />
            <button className="headerNewChat" type="button" onClick={startFreshConversation} disabled={runActive} title={runActive ? "Stop the active run first" : "Start a new conversation"}>
              <span aria-hidden="true">＋</span><span>New chat</span>
            </button>
            <button className={`timelineToggle ${runActive ? "isLive" : ""} ${timelineOpen ? "active" : ""}`} type="button" onClick={() => setTimelineOpen((value) => !value)} aria-pressed={timelineOpen}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 11.8 6.2 8.6l2.2 2.1L13 5.8M9.7 5.8H13v3.3" /></svg>
              <span>{runActive ? "Live activity" : "Activity"}</span>{events.length ? <b>{events.length}</b> : null}
            </button>
          </div>
        </header>

        {recoverableRuns.length ? (
          <section className="recoveryBanner" aria-label="Runs awaiting approval">
            <div className="recoverySummary">
              <span className="recoveryIcon" aria-hidden="true">↻</span>
              <div>
                <strong>{recoverableRuns.length === 1 ? "A run is waiting for you" : `${recoverableRuns.length} runs are waiting for you`}</strong>
                <small>Reviewing the saved approval resumes from its durable checkpoint.</small>
              </div>
            </div>
            <div className="recoveryActions">
              {recoverableRuns.map((item) => {
                const selected = item.run.id === activeRunId;
                return (
                  <button
                    key={item.run.id}
                    type="button"
                    className={selected ? "selected" : ""}
                    disabled={selected}
                    onClick={() => router.push(`/?conversation=${encodeURIComponent(item.run.conversation_id)}&run=${encodeURIComponent(item.run.id)}`)}
                    title={item.approval?.summary ?? "Open the saved run"}
                  >
                    <span>{item.approval?.risk_level ?? "R3"}</span>
                    {selected ? "Reviewing now" : item.approval?.title ?? "Review approval"}
                  </button>
                );
              })}
            </div>
          </section>
        ) : null}

        {folderScanning || folderDrop ? (
          <section className="folderRoutePopover" aria-label="Route this folder">
            <div className="projectWorkspaceIntro">
              <span className="eyebrow">Folder dropped</span>
              <strong>{folderScanning ? "Reading the folder…" : folderDrop?.scan.name}</strong>
              {folderScanning ? (
                <p>Counting locally what Metis could index or attach. Build and vendor directories are passed over.</p>
              ) : folderDrop ? (
                <p>
                  {`${folderDrop.scan.files.length}${folderDrop.scan.truncated ? "+" : ""} file${folderDrop.scan.files.length === 1 ? "" : "s"}`}
                  {` · ${formatByteSize(totalBytes(folderDrop.scan.files))}`}
                  {folderDrop.scan.skippedDirectories ? ` · ${folderDrop.scan.skippedDirectories} build folder${folderDrop.scan.skippedDirectories === 1 ? "" : "s"} passed over` : ""}
                  {folderDrop.ignored.length ? ` · one folder at a time, so ${folderDrop.ignored.join(", ")} was ignored` : ""}
                </p>
              ) : null}
            </div>

            {folderDrop ? (
              <div className="folderRouteList">
                <article className="folderRoute">
                  <header>
                    <i aria-hidden="true">◇</i>
                    <div><strong>Open as project</strong><small>Metis reads, searches, and proposes exact edits under approval. Nothing leaves this machine.</small></div>
                  </header>
                  {matchedProject ? (
                    <button type="button" className="primaryButton" disabled={Boolean(folderBusy) || projectOpening || runActive} onClick={() => void openDroppedProject()}>
                      {folderBusy === "project" ? "Opening…" : `Open ${matchedProject.name}`}
                    </button>
                  ) : (
                    <div className="folderRouteAside">
                      <p>Not in the projects catalog. Projects are discovered by scanning the folders configured in Settings.</p>
                      <button type="button" className="secondaryButton" disabled={Boolean(folderBusy)} onClick={() => void rescanForDroppedProject()}>
                        {folderBusy === "rescan" ? "Rescanning…" : "Rescan projects"}
                      </button>
                    </div>
                  )}
                </article>

                <article className="folderRoute">
                  <header>
                    <i aria-hidden="true">⌗</i>
                    <div><strong>Index as knowledge</strong><small>Chunked and embedded once, then cited in this conversation and every one after it.</small></div>
                  </header>
                  <div className="folderRouteForm">
                    {matchedSource ? (
                      <p className="folderRoutePath mono">{matchedSource.root_path}</p>
                    ) : (
                      <>
                        {/* A browser never reveals a dropped folder's real path; the
                            guess comes from where existing sources already live. */}
                        <input
                          className="knowledgeInput"
                          value={folderPath}
                          onChange={(event) => setFolderPath(event.target.value)}
                          placeholder="/absolute/path/to/the/folder"
                          aria-label="Folder path on disk"
                          spellCheck={false}
                          disabled={Boolean(folderBusy)}
                        />
                        <select
                          className="knowledgeInput"
                          value={folderKind}
                          onChange={(event) => setFolderKind(event.target.value as CorpusSource["kind"])}
                          aria-label="Source kind"
                          disabled={Boolean(folderBusy)}
                        >
                          {["code", "docs", "notes", "mixed"].map((item) => <option key={item} value={item}>{item}</option>)}
                        </select>
                      </>
                    )}
                    {matchedSource?.consent ? null : (
                      <label className="folderRouteConsent">
                        <input type="checkbox" checked={folderConsent} onChange={(event) => setFolderConsent(event.target.checked)} disabled={Boolean(folderBusy)} />
                        <span>Allow cloud embedding for this folder. Its text is sent to Cohere to build the index; the vectors stay in local SQLite.</span>
                      </label>
                    )}
                    <button type="button" className="primaryButton" disabled={knowledgeDisabled} onClick={() => void indexDroppedFolder()}>
                      {knowledgeLabel}
                    </button>
                    {matchedSource ? <small className="folderRouteHint">Already a source · {matchedSource.file_count} file(s) · {matchedSource.chunk_count} chunk(s)</small> : null}
                  </div>
                </article>

                <article className="folderRoute">
                  <header>
                    <i aria-hidden="true">＋</i>
                    <div><strong>Attach the files</strong><small>Text goes into this one message and counts against the {formatByteSize(ATTACHMENT_TEXT_BUDGET_BYTES)} context budget.</small></div>
                  </header>
                  {attachBlockedReason ? (
                    <div className="folderRouteAside"><p>{attachBlockedReason}</p></div>
                  ) : (
                    <button type="button" className="secondaryButton" disabled={Boolean(folderBusy) || uploading || runActive} onClick={() => void attachDroppedFiles()}>
                      {folderBusy === "attach" ? "Uploading…" : `Attach ${droppedAttachable.length} file${droppedAttachable.length === 1 ? "" : "s"} · ${formatByteSize(droppedAttachableBytes)}`}
                    </button>
                  )}
                </article>
              </div>
            ) : null}

            <footer className="folderRouteFooter">
              {folderNotice
                ? <p role="status">{folderNotice}</p>
                : <span>Reading a folder is local. Only the route you choose acts on it.</span>}
              <button type="button" className="textButton" onClick={closeFolderDrop} disabled={Boolean(folderBusy)}>Dismiss</button>
            </footer>
          </section>
        ) : null}

        <div
          className={`messageViewport ${dragActive ? "dragActive" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { if (event.currentTarget === event.target) setDragActive(false); }}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            // DataTransfer entries are neutered once this handler returns, so
            // the directories have to be claimed before anything awaits.
            const directories = droppedDirectories(event.dataTransfer.items);
            const files = looseFiles(event.dataTransfer.files, directories);
            if (files.length) void addFiles(files);
            if (directories.length) void routeFolderDrop(directories);
          }}
        >
          {dragActive ? <div className="dropOverlay"><span>＋</span><strong>Drop files or a folder</strong><small>Files become context · a folder can be indexed, opened as a project, or attached</small></div> : null}
          {!hasMessages ? (
            <div className="welcomeState">
              <MetisCompanion mood={companionMood} />
              <span className="eyebrow">Your private thinking partner</span>
              <h1><MetisWordmark className="heroWordmark" /></h1>
              <p>Ask Metis to understand a project, create an artifact, or turn a workflow into a tested local capability.</p>
              <CustomerDashboardSnippet />
              <div className="trustLine"><span><i />On-device models</span><span><i />Sandboxed tools</span><span><i />Approved memory</span></div>
            </div>
          ) : (
            <div className="messageList">
              {loadingConversation ? <div className="messageLoading"><span /><span /><span /></div> : null}
              {messages.map((message, messageIndex) => (
                <article className={`chatMessage message-${message.role} ${message.failed ? "message-failed" : ""}`} key={message.id}>
                  <div className={`messageAvatar ${message.role === "assistant" ? "metisAvatar" : "userAvatar"}`}>
                    {message.role === "user" ? <UserAvatar /> : <MetisMark animated={message.streaming} />}
                  </div>
                  <div className="messageBody">
                    <div className="messageAuthor"><strong>{message.role === "user" ? "You" : "Metis"}</strong>{message.streaming ? <span className="thinkingPulse"><i /><i /><i /></span> : null}</div>
                    {message.reasoning ? <ReasoningPanel reasoning={message.reasoning} live={Boolean(message.streaming) && !message.content} /> : null}
                    {message.content ? <MarkdownContent content={message.content} /> : message.streaming && !message.reasoning ? <p className="workingText">{(message.id === latestAssistant?.id ? stageLabel : null) ?? "Understanding the task and choosing a safe route…"}</p> : null}
                    {message.failed ? (
                      <div className="failedActions">
                        <button className="retryResponse" type="button" onClick={() => {
                          const previous = [...messages.slice(0, messageIndex)].reverse().find((item) => item.role === "user");
                          if (previous) setDraft(previous.content);
                          clearFailedResponse(message.id, message.run_id);
                        }}>Edit & retry</button>
                        <button className="clearFailedResponse" type="button" onClick={() => clearFailedResponse(message.id, message.run_id)}>Clear failed response</button>
                      </div>
                    ) : null}
                    {message.attachments?.length ? <div className="inlineAttachments">{message.attachments.map((attachment) => <span key={attachment.id}><b>{attachmentBadge(attachment)}</b>{attachment.name}</span>)}</div> : null}
                    {message.role === "assistant" && (messageBelongsToRun(message, activeRunId) || (!activeRunId && message.id === latestAssistant?.id)) ? <ArtifactViewer artifacts={artifacts} /> : null}
                    {messageBelongsToRun(message, activeRunId) && !message.streaming && !message.failed ? (
                      <section className="feedbackControls" aria-label="Response feedback">
                        {feedbackMode === "sent" ? (
                          <span className="feedbackRecorded">✓ Feedback recorded; any durable learning remains pending review.</span>
                        ) : feedbackMode === "correcting" ? (
                          <div className="correctionForm">
                            <label htmlFor="metis-correction">What should Metis learn or correct?</label>
                            <textarea id="metis-correction" value={correction} onChange={(event) => setCorrection(event.target.value)} maxLength={20000} placeholder="Describe the correction. It will become a governed memory proposal, not active knowledge." />
                            <div><button type="button" className="textButton" onClick={() => { setFeedbackMode("idle"); setCorrection(""); }}>Cancel</button><button type="button" className="primaryButton" disabled={!correction.trim() || feedbackBusy} onClick={() => void handleFeedback("negative")}>{feedbackBusy ? "Recording…" : "Submit correction"}</button></div>
                          </div>
                        ) : (
                          <div className="feedbackPrompt"><span>Was this useful?</span><button type="button" onClick={() => void handleFeedback("positive")} disabled={feedbackBusy}>Yes</button><button type="button" onClick={() => void handleFeedback("negative")} disabled={feedbackBusy}>Needs correction</button></div>
                        )}
                        {feedbackMode !== "correcting" ? (
                          <div className="toolBuildRow">
                            <button
                              type="button"
                              className="textButton toolBuildButton"
                              disabled={sending || uploading || runActive}
                              title="Ask Metis to turn this repeatable process into a governed, reusable tool"
                              onClick={() => void submit(undefined, TOOL_BUILD_PROMPT)}
                            >
                              Build a tool from this
                            </button>
                            {selectedCustomer && message.content.trim() ? (
                              <button
                                type="button"
                                className="textButton saveToAccountButton"
                                disabled={savingToAccount === message.id || savedToAccount.has(message.id)}
                                title={`Keep this answer on ${selectedCustomer.name} as an account note`}
                                onClick={() => void saveAnswerToAccount(message)}
                              >
                                {savedToAccount.has(message.id)
                                  ? `✓ Saved to ${selectedCustomer.name}`
                                  : savingToAccount === message.id
                                    ? "Saving…"
                                    : `Save to ${selectedCustomer.name}`}
                              </button>
                            ) : null}
                          </div>
                        ) : null}
                      </section>
                    ) : null}
                  </div>
                  {message.role === "user" && message.id === latestUser?.id && activeRunId ? (
                    <button
                      className={`compactRunActivity tone-${compactActivity.tone} ${compactActivity.live ? "isLive" : ""}`}
                      type="button"
                      onClick={() => setTimelineOpen((value) => !value)}
                      aria-expanded={timelineOpen}
                      aria-label={`${compactActivity.label}. ${compactActivity.detail}. ${timelineOpen ? "Close" : "Open"} full activity.`}
                    >
                      <span className="compactActivityCore" aria-hidden="true"><i /><i /><i /><i /></span>
                      <span className="compactActivityCopy"><strong key={compactActivity.label}>{compactActivity.label}</strong><small>{compactActivity.detail}</small></span>
                      <span className="compactActivityChevron" aria-hidden="true">⌄</span>
                    </button>
                  ) : null}
                </article>
              ))}
              <div ref={messageEndRef} />
            </div>
          )}
        </div>

        <div className="composerDock">
          {error ? <div className="composerError" role="alert"><span>!</span><p><strong>Something interrupted this chat</strong>{error}</p><button className="errorFreshStart" type="button" onClick={startFreshConversation}>Start fresh</button><button className="errorDismiss" type="button" aria-label="Dismiss error" onClick={() => setError(null)}>×</button></div> : null}
          {attachments.length ? (
            <div className="attachmentTray">
              {attachments.map((attachment) => (
                <span key={attachment.id}><b>{attachmentBadge(attachment)}</b><span><strong>{attachment.name}</strong><small>{attachment.size ? `${Math.ceil(attachment.size / 1024)} KB` : "Ready"}</small></span><button type="button" aria-label={`Remove ${attachment.name}`} onClick={() => setAttachments((current) => current.filter((item) => item.id !== attachment.id))}>×</button></span>
              ))}
            </div>
          ) : null}
          {selectedCustomer || selectedProject ? (
            <div className="composerChips" aria-label="Context added to this message">
              {selectedCustomer ? (
                <span className="contextChip" data-kind="customer">
                  <i aria-hidden="true">◎</i>
                  <span>{selectedCustomer.name}</span>
                  <button type="button" aria-label={`Remove ${selectedCustomer.name}`} onClick={() => setSelectedCustomerId(null)}>×</button>
                </span>
              ) : null}
              {selectedProject ? (
                <span className="contextChip" data-kind="project">
                  <i aria-hidden="true">◇</i>
                  <span>{projectOpening ? "Mapping…" : selectedProject.name}</span>
                  <button type="button" aria-label={`Remove ${selectedProject.name}`} disabled={projectOpening} onClick={() => void chooseProject(null)}>×</button>
                </span>
              ) : null}
            </div>
          ) : null}
          {hasMessages ? (
            <div className="companionDock" data-mood={companionMood}>
              <MetisCompanion mood={companionMood} />
              <span className="companionDockLabel">{companionLabel}</span>
            </div>
          ) : null}
          <form className="composer" onSubmit={(event) => void submit(event)}>
            <textarea
              ref={textareaRef}
              rows={1}
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={knowledgeScope === "notion" ? "Ask only from your synced Notion…" : "Message Metis…"}
              aria-label="Message Metis"
              disabled={sending}
            />
            <div className="composerToolbar">
              <div>
                {/* The composer carries per-message scope: what this message is
                    about. What RUNS it (the model/provider) lives top-right in
                    the header. Customer and project are mutually exclusive. */}
                <div className="composerScope">
                  <div className="headerPickerAnchor">
                    <button
                      className="composerAddContext"
                      type="button"
                      aria-expanded={slashOpen}
                      aria-haspopup="listbox"
                      onClick={() => setSlashOpen((value) => !value)}
                      disabled={runActive}
                      title="Add a customer or project to this message  ( / )"
                    >
                      <i aria-hidden="true">＋</i>
                      <span>Context</span>
                      <kbd>/</kbd>
                    </button>
                    {slashOpen ? (
                      <CommandPicker
                        label="Add to this message"
                        placeholder="Type a command…"
                        options={[
                          { id: "customer", label: "Customer", meta: `Scope to one of ${customerOptions.length} accounts` },
                          { id: "project", label: "Project", meta: "Open a project workspace" },
                        ].filter((option) =>
                          !slashQuery || option.id.startsWith(slashQuery.toLowerCase()),
                        )}
                        value={null}
                        emptyMessage="No command matches that."
                        onSelect={(id) => runSlashCommand(id)}
                        onDismiss={() => setSlashOpen(false)}
                        footer={<span>Type <code>/</code> in the message box to reach this any time.</span>}
                      />
                    ) : null}
                    {customerPickerOpen ? (
                      <CommandPicker
                        label="Customer account scope"
                        placeholder={`Search ${customerOptions.length} accounts…`}
                        options={customerOptions}
                        value={selectedCustomerId}
                        clearOption={{ id: "", label: "No customer", meta: "Ordinary chat routing" }}
                        emptyMessage="No account matches that."
                        onSelect={(id) => { setSelectedCustomerId(id || null); setCustomerPickerOpen(false); }}
                        onDismiss={() => setCustomerPickerOpen(false)}
                        footer={<span>Only this account&rsquo;s reviewed facts and pinned notes enter the model context.</span>}
                      />
                    ) : null}
                    {projectPickerOpen ? (
                      <CommandPicker
                        label="Project workspace"
                        placeholder={`Search ${projects.length} project${projects.length === 1 ? "" : "s"}…`}
                        options={projectOptions}
                        value={selectedProjectId}
                        clearOption={{ id: "", label: "No project", meta: "Return to ordinary chat routing." }}
                        emptyMessage={projects.length ? "No project matches that." : "No projects in the catalog — use Assets → Scan for updates first."}
                        busy={projectOpening}
                        onSelect={(id) => void chooseProject(id || null)}
                        onDismiss={() => setProjectPickerOpen(false)}
                        header={
                          <div className="projectWorkspaceIntro">
                            <span className="eyebrow">Whole-project mode</span>
                            <p>Grok maps once, then Metis reads, searches, and proposes exact edits under approval. Pick who leads each step from the model control, top-right.</p>
                          </div>
                        }
                        footer={<span>{projectOpening ? "Opening…" : "Writes always pause for approval"}</span>}
                      />
                    ) : null}
                  </div>
                </div>
                <button className="attachButton" type="button" onClick={() => fileInputRef.current?.click()} disabled={uploading || runActive} aria-label="Attach images or files">＋ <span>{uploading ? "Uploading…" : "Attach"}</span></button>
                <input ref={fileInputRef} type="file" multiple hidden accept={CHAT_ATTACHMENT_ACCEPT} onChange={(event) => event.target.files && void addFiles(event.target.files)} />
                <div className="knowledgeScopeSwitch" role="group" aria-label="Answer sources">
                  <span>Sources</span>
                  <button
                    type="button"
                    className={knowledgeScope === "auto" ? "selected" : ""}
                    aria-pressed={knowledgeScope === "auto"}
                    disabled={runActive}
                    onClick={() => chooseKnowledgeScope("auto")}
                    title="Use all relevant Metis context, including Notion when it helps"
                  >Auto</button>
                  <button
                    type="button"
                    className={knowledgeScope === "notion" ? "selected notionSelected" : ""}
                    aria-pressed={knowledgeScope === "notion"}
                    disabled={runActive}
                    onClick={() => chooseKnowledgeScope("notion")}
                    title="Answer only from synced Notion pages; refuse when there is no support"
                  >Notion</button>
                </div>
                <span className="composerHint">Enter to send · Shift Enter for a new line</span>
              </div>
              {runActive ? (
                <button className="stopButton" type="button" onClick={() => void handleCancel()} aria-label="Stop run"><span /> Stop</button>
              ) : (
                <button className="sendButton" type="submit" disabled={(!draft.trim() && !attachments.length) || sending || uploading} aria-label="Send message">↗</button>
              )}
            </div>
          </form>
          <p className="composerDisclaimer">Metis can make mistakes. Review generated code and approve persistent changes deliberately.</p>
        </div>
      </section>

      <aside className={`runDrawer ${timelineOpen ? "open" : ""}`} aria-hidden={!timelineOpen}>
        <div
          className="runDrawerResizeHandle"
          role="separator"
          aria-label="Resize activity sidebar"
          aria-orientation="vertical"
          aria-valuemin={MIN_ACTIVITY_WIDTH}
          aria-valuemax={MAX_ACTIVITY_WIDTH}
          aria-valuenow={timelineWidth}
          tabIndex={timelineOpen ? 0 : -1}
          onDoubleClick={() => rememberTimelineWidth(DEFAULT_ACTIVITY_WIDTH)}
          onKeyDown={handleTimelineResizeKeyDown}
          onPointerDown={handleTimelineResizeStart}
          onPointerMove={handleTimelineResizeMove}
          onPointerUp={handleTimelineResizeEnd}
          onPointerCancel={handleTimelineResizeEnd}
          onLostPointerCapture={handleTimelineResizeEnd}
          title="Drag to resize · Double-click to reset"
        ><span /></div>
        <button className="runDrawerClose" type="button" aria-label="Close activity" onClick={() => setTimelineOpen(false)}>×</button>
        <RunTimeline
          events={events}
          connection={connection}
          streamError={streamError}
          onDecision={handleDecision}
          decidedApprovals={decidedApprovals}
          decisionBusy={decisionBusy}
          approveLabel={
            modelPreference?.provider !== "oci"
            && localModelSession
            && localModelSession.state === "off"
            && localModelSession.selected_model
              ? `Approve & relaunch ${localModelSession.selected_model}`
              : "Approve once"
          }
        />
      </aside>
    </div>
  );
}
