"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { CSSProperties, FormEvent, KeyboardEvent, PointerEvent as ReactPointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ArtifactViewer } from "@/components/artifact-viewer";
import { MarkdownContent } from "@/components/markdown-content";
import { RunTimeline } from "@/components/run-timeline";
import { MetisCompanion, MetisMark } from "@/components/metis-mark";
import {
  ApiError,
  cancelRun,
  createConversation,
  decideRun,
  getConversation,
  getConversationProject,
  getModelPreference,
  listProjectWorkspaces,
  listRecoverableRuns,
  openProjectWorkspace,
  sendMessage,
  setModelPreference,
  submitFeedback,
  uploadFile,
} from "@/lib/api";
import { rememberConversation } from "@/lib/recent-conversations";
import { mergeAssistantRunEvent, messageBelongsToRun } from "@/lib/run-history";
import { attachmentBadge, CHAT_ATTACHMENT_ACCEPT } from "@/lib/attachments";
import type { ArtifactRef, AttachmentRef, ChatMessage, KnowledgeScope, ModelPreference, ProjectMode, ProjectWorkspace, RecoverableRun, RunEventV1 } from "@/lib/types";
import { useRunEvents } from "@/hooks/use-run-events";

const suggestions = [
  { label: "Map a codebase", prompt: "Review the attached README and create a reference architecture with Python diagrams." },
  { label: "Build a reusable tool", prompt: "Turn this repeatable workflow into a tested local tool and show me the proposal before activation." },
  { label: "Explain a repository", prompt: "Read the attached project files and explain the architecture, risks, and a sensible next step." },
];

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
  const [providerSaving, setProviderSaving] = useState(false);
  const [projects, setProjects] = useState<ProjectWorkspace[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectMode, setProjectMode] = useState<ProjectMode>("grok_bootstrap_local");
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [projectOpening, setProjectOpening] = useState(false);
  const [knowledgeScope, setKnowledgeScope] = useState<KnowledgeScope>("auto");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
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
    setProjectPickerOpen(false);
    setProjectOpening(false);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (textareaRef.current) textareaRef.current.style.height = "";
  }, []);

  useEffect(() => {
    const stored = window.localStorage.getItem("metis.knowledgeScope");
    if (stored === "auto" || stored === "notion") setKnowledgeScope(stored);
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

  async function chooseProject(projectId: string | null) {
    if (projectOpening || runActive) return;
    if (!projectId) {
      setSelectedProjectId(null);
      setProjectPickerOpen(false);
      return;
    }
    const target = projects.find((project) => project.id === projectId);
    if (!modelPreference?.oci_available && (!target?.initialized || projectMode === "grok_continuous")) {
      setError("Project mode needs Grok through OCI for the initial repository map. Configure OCI in Settings first.");
      return;
    }
    setProjectOpening(true);
    setError(null);
    try {
      const opened = await openProjectWorkspace(projectId, projectMode);
      setProjects((current) => current.map((item) => item.id === opened.id ? opened : item));
      setSelectedProjectId(opened.id);
      setProjectPickerOpen(false);
    } catch (openError) {
      setError(openError instanceof Error ? openError.message : "Metis could not open that project.");
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
        if (mounted) setError(loadError instanceof Error ? loadError.message : "Could not open this conversation.");
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
  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  return (
    <div
      className={`chatWorkspace ${timelineOpen ? "timelineVisible" : ""} ${timelineResizing ? "timelineResizing" : ""}`}
      style={{ "--activity-width": `${timelineWidth}px` } as CSSProperties}
    >
      <section className="conversationPane">
        <header className="chatHeader">
          <div className="chatTitle">
            <span className="localStatus"><i />{selectedProject ? "Project" : modelPreference?.provider === "oci" ? "Cloud" : "Local"}</span>
            <strong title={conversationTitle}>{conversationTitle}</strong>
          </div>
          <div className="chatHeaderActions">
            <button
              className={`projectQuickSwitch ${selectedProject ? "selected" : ""}`}
              type="button"
              aria-expanded={projectPickerOpen}
              onClick={() => setProjectPickerOpen((value) => !value)}
              disabled={projectOpening || runActive}
              title="Open a whole project as the current coding workspace"
            >
              <i aria-hidden="true">◇</i>
              <span>{projectOpening ? "Mapping…" : selectedProject?.name ?? "Project"}</span>
              <b aria-hidden="true">⌄</b>
            </button>
            <div className="modelQuickSwitch" role="group" aria-label="Reasoning model">
              <span>Model</span>
              <button
                type="button"
                className={modelPreference?.provider !== "oci" ? "selected" : ""}
                aria-pressed={modelPreference?.provider !== "oci"}
                disabled={providerSaving || Boolean(selectedProject)}
                onClick={() => void chooseChatProvider("local")}
                title="Use on-device Ollama models"
              ><i aria-hidden="true" />Local</button>
              <button
                type="button"
                className={modelPreference?.provider === "oci" ? "selected" : ""}
                aria-pressed={modelPreference?.provider === "oci"}
                disabled={providerSaving || modelPreference?.oci_available !== true || Boolean(selectedProject)}
                onClick={() => void chooseChatProvider("oci")}
                title={modelPreference?.oci_available ? "Use Grok 4.3 through OCI" : "Configure OCI in Settings first"}
              ><i aria-hidden="true" />Cloud</button>
              {providerSaving ? <small role="status">Saving…</small> : null}
            </div>
            <button className="headerNewChat" type="button" onClick={startFreshConversation} disabled={runActive} title={runActive ? "Stop the active run first" : "Start a new conversation"}>
              <span aria-hidden="true">＋</span><span>New chat</span>
            </button>
            <button className={`timelineToggle ${timelineOpen ? "active" : ""}`} type="button" onClick={() => setTimelineOpen((value) => !value)} aria-pressed={timelineOpen}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 11.8 6.2 8.6l2.2 2.1L13 5.8M9.7 5.8H13v3.3" /></svg>
              Activity {events.length ? <b>{events.length}</b> : null}
            </button>
          </div>
        </header>

        {projectPickerOpen ? (
          <section className="projectWorkspacePopover" aria-label="Project workspace">
            <div className="projectWorkspaceIntro">
              <span className="eyebrow">Whole-project mode</span>
              <strong>{selectedProject ? selectedProject.name : "Choose a project"}</strong>
              <p>Grok creates the first local map. Metis then reads, searches, and proposes exact edits with approval.</p>
            </div>
            <div className="projectModeSwitch" role="group" aria-label="Project reasoning mode">
              <button
                type="button"
                className={projectMode === "grok_bootstrap_local" ? "selected" : ""}
                aria-pressed={projectMode === "grok_bootstrap_local"}
                onClick={() => void chooseProjectMode("grok_bootstrap_local")}
              ><span>Fast &amp; private</span><strong>Grok → Local</strong><small>Grok maps once; North handles project turns.</small></button>
              <button
                type="button"
                className={projectMode === "grok_continuous" ? "selected" : ""}
                aria-pressed={projectMode === "grok_continuous"}
                onClick={() => void chooseProjectMode("grok_continuous")}
              ><span>Largest context</span><strong>Keep Grok</strong><small>Grok leads every bounded project step.</small></button>
            </div>
            <div className="projectWorkspaceList">
              <button type="button" className={!selectedProject ? "selected" : ""} onClick={() => void chooseProject(null)}>
                <span className="projectGlyph">—</span><span><strong>No project</strong><small>Return to ordinary chat routing.</small></span>
              </button>
              {projects.map((project) => (
                <button key={project.id} type="button" className={selectedProjectId === project.id ? "selected" : ""} onClick={() => void chooseProject(project.id)}>
                  <span className="projectGlyph">{project.initialized ? "◆" : "◇"}</span>
                  <span><strong>{project.name}</strong><small>{project.framework ? `${project.framework} · ` : ""}{project.initialized ? `${project.fileCount} files mapped` : "Grok map not created yet"}</small></span>
                  {selectedProjectId === project.id ? <b>Active</b> : null}
                </button>
              ))}
              {!projects.length ? <div className="projectWorkspaceEmpty"><strong>No projects in the catalog</strong><small>Use Assets → Scan for updates first.</small></div> : null}
            </div>
            <footer><span><i />Writes always pause for approval</span><button type="button" onClick={() => setProjectPickerOpen(false)}>Done</button></footer>
          </section>
        ) : null}

        {selectedProject ? (
          <div className="projectContextStrip">
            <span><i />{selectedProject.name}</span>
            <small>{projectMode === "grok_continuous" ? "Grok leads · local project tools" : "Grok mapped · North leads locally"}</small>
            <code>{selectedProject.metisMdPath}</code>
            <button type="button" onClick={() => setProjectPickerOpen(true)}>Change</button>
          </div>
        ) : null}

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

        <div
          className={`messageViewport ${dragActive ? "dragActive" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { if (event.currentTarget === event.target) setDragActive(false); }}
          onDrop={(event) => { event.preventDefault(); setDragActive(false); void addFiles(event.dataTransfer.files); }}
        >
          {dragActive ? <div className="dropOverlay"><span>＋</span><strong>Drop files to add context</strong><small>Images, documents, and source files · no archives</small></div> : null}
          {!hasMessages ? (
            <div className="welcomeState">
              <MetisCompanion />
              <span className="eyebrow">Your private thinking partner</span>
              <h1>What are we building?</h1>
              <p>Ask Metis to understand a project, create an artifact, or turn a workflow into a tested local capability.</p>
              <div className="suggestionGrid">
                {suggestions.map((suggestion, index) => (
                  <button key={suggestion.label} type="button" onClick={() => { setDraft(suggestion.prompt); textareaRef.current?.focus(); }}>
                    <span>0{index + 1}</span><strong>{suggestion.label}</strong><small>{suggestion.prompt}</small><i>↗</i>
                  </button>
                ))}
              </div>
              <div className="trustLine"><span><i />On-device models</span><span><i />Sandboxed tools</span><span><i />Approved memory</span></div>
            </div>
          ) : (
            <div className="messageList">
              {loadingConversation ? <div className="messageLoading"><span /><span /><span /></div> : null}
              {messages.map((message, messageIndex) => (
                <article className={`chatMessage message-${message.role} ${message.failed ? "message-failed" : ""}`} key={message.id}>
                  <div className={`messageAvatar ${message.role === "assistant" ? "metisAvatar" : ""}`}>
                    {message.role === "user" ? "You" : <MetisMark animated={message.streaming} />}
                  </div>
                  <div className="messageBody">
                    <div className="messageAuthor"><strong>{message.role === "user" ? "You" : "Metis"}</strong>{message.streaming ? <span className="thinkingPulse"><i /><i /><i /></span> : null}</div>
                    {message.content ? <MarkdownContent content={message.content} /> : message.streaming ? <p className="workingText">{(message.id === latestAssistant?.id ? stageLabel : null) ?? "Understanding the task and choosing a safe route…"}</p> : null}
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
                          </div>
                        ) : null}
                      </section>
                    ) : null}
                  </div>
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
          <form className="composer" onSubmit={(event) => void submit(event)}>
            <textarea
              ref={textareaRef}
              rows={1}
              value={draft}
              onChange={(event) => { setDraft(event.target.value); event.target.style.height = "0"; event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`; }}
              onKeyDown={onComposerKeyDown}
              placeholder={knowledgeScope === "notion" ? "Ask only from your synced Notion…" : "Message Metis…"}
              aria-label="Message Metis"
              disabled={sending}
            />
            <div className="composerToolbar">
              <div>
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
        <RunTimeline events={events} connection={connection} streamError={streamError} onDecision={handleDecision} decidedApprovals={decidedApprovals} decisionBusy={decisionBusy} />
      </aside>
    </div>
  );
}
