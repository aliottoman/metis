"use client";

// The chat engine: one conversation, its messages, and the run in flight.
// Everything that talks to the API or to the run stream lives here; the
// composer and the thread only render what this returns and call back into
// it. Kept as one hook because the pieces genuinely share state — a send
// creates the conversation, adopts the message id, starts the run and
// subscribes to its events, and a rewind undoes all four.

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useRunEvents } from "@/hooks/use-run-events";
import {
  ApiError,
  answerElicitation,
  applyCustomerProposal,
  cancelRun,
  createConversation,
  createCustomerNote,
  createCustomerOutput,
  createProjectAsset,
  decideRun,
  getConversation,
  getConversationProject,
  getLocalModelSession,
  getModelPreference,
  getRunResult,
  listCustomers,
  listProjectWorkspaces,
  listRecoverableRuns,
  openProjectWorkspace,
  rewindConversation,
  sendMessage,
  setModelPreference as saveModelPreference,
  submitFeedback,
  uploadFile,
  addDocumentToKnowledge,
} from "@/lib/api";
import { latestPendingApproval } from "@/lib/approvals";
import { latestPendingElicitation } from "@/lib/elicitations";
import { isCloudActive } from "@/lib/model";
import { clinePassReady, projectMappingReady } from "@/lib/model-route";
import { rememberConversation } from "@/lib/recent-conversations";
import { mergeAssistantReasoning, mergeAssistantRunEvent, messageBelongsToRun } from "@/lib/run-history";
import { trackConversationRun, updateConversationRun } from "@/lib/run-indicators";
import { latestActionSuggestion } from "@/lib/suggestions";
import { freshToken } from "@/lib/token";
import type {
  ArtifactRef,
  AttachmentRef,
  ChatMessage,
  CustomerAccount,
  KnowledgeScope,
  LocalModelSession,
  ModelPreference,
  ProjectMode,
  ProjectWorkspace,
  RecoverableRun,
  RunEventV1,
} from "@/lib/types";

export type Chat = ReturnType<typeof useChat>;

/** True for a message the server knows by the id we hold. Optimistic and
 *  streaming messages carry client ids that cannot anchor a rewind. */
export function isPersisted(message: ChatMessage): boolean {
  return !message.id.startsWith("optimistic-") && !message.id.startsWith("assistant-");
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

function mergeArtifacts(current: ArtifactRef[], incoming: ArtifactRef[]): ArtifactRef[] {
  const ids = new Set(current.map((item) => item.id));
  return [...current, ...incoming.filter((item) => !ids.has(item.id))];
}

/** Why a project mode cannot run, or null when it can. */
function projectModeBlocked(mode: ProjectMode, preference: ModelPreference | null): string | null {
  if (mode === "grok_continuous" && !preference?.oci_available) {
    return "Grok mode needs the Grok lane enabled and OCI Responses configured in Settings first.";
  }
  if (mode === "cohere_continuous" && !preference?.cohere_available) {
    return "Command A+ mode needs a Cohere API key. Add WAQIL_COHERE_API_KEY in Settings first.";
  }
  if (mode === "grok_bootstrap_local" && preference?.provider === "cline"
    && !clinePassReady(preference.cline_available, preference.cline_models)) {
    return "The selected ClinePass role ladder needs WAQIL_CLINE_API_KEY in Settings first.";
  }
  return null;
}

const TERMINAL = ["run.completed", "run.failed", "run.cancelled", "completed", "failed", "cancelled"];

// Sent verbatim when the user asks Metis to distil an answer into a tool.
export const TOOL_BUILD_PROMPT = "Turn this repeatable process into a reusable tool.";

export function useChat() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedConversationId = searchParams.get("conversation");
  const requestedRunId = searchParams.get("run");
  const newRequestToken = searchParams.get("new");

  // -- the conversation ------------------------------------------------------
  const [conversationId, setConversationId] = useState<string | null>(requestedConversationId);
  const [conversationTitle, setConversationTitle] = useState("New conversation");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // -- the composer's contents -----------------------------------------------
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<AttachmentRef[]>([]);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [queued, setQueued] = useState<{ content: string; attachments: AttachmentRef[] } | null>(null);
  const [knowledgeAdds, setKnowledgeAdds] = useState<Record<string, "adding" | "added" | "error">>({});
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const focusComposer = useCallback(() => window.setTimeout(() => composerRef.current?.focus(), 0), []);

  // -- the run in flight -------------------------------------------------------
  const [activeRunId, setActiveRunId] = useState<string | null>(requestedRunId);
  const [artifacts, setArtifacts] = useState<ArtifactRef[]>([]);
  const [stageLabel, setStageLabel] = useState<string | null>(null);
  const [recoverableRuns, setRecoverableRuns] = useState<RecoverableRun[]>([]);
  const [timelineOpen, setTimelineOpen] = useState(false);
  const [decidedApprovals, setDecidedApprovals] = useState<Set<string>>(new Set());
  const [answeredElicitations, setAnsweredElicitations] = useState<Set<string>>(new Set());
  const [appliedProposals, setAppliedProposals] = useState<Set<string>>(new Set());
  const [dismissedProposals, setDismissedProposals] = useState<Set<string>>(new Set());
  const [decisionBusy, setDecisionBusy] = useState<string | null>(null);
  const [answerBusy, setAnswerBusy] = useState<string | null>(null);
  const [applyingProposal, setApplyingProposal] = useState<string | null>(null);

  // -- message-level actions -----------------------------------------------
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [rewinding, setRewinding] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ mode: "idle" | "correcting" | "sent"; busy: boolean; correction: string }>({ mode: "idle", busy: false, correction: "" });
  const [savedToAccount, setSavedToAccount] = useState<Set<string>>(new Set());
  const [savingToAccount, setSavingToAccount] = useState<string | null>(null);
  const [trackerBusy, setTrackerBusy] = useState(false);

  // -- scope: model, project, customer, sources ---------------------------------
  const [modelPreference, setModelPreference] = useState<ModelPreference | null>(null);
  const [localModelSession, setLocalModelSession] = useState<LocalModelSession | null>(null);
  const [providerSaving, setProviderSaving] = useState(false);
  const [projects, setProjects] = useState<ProjectWorkspace[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectMode, setProjectMode] = useState<ProjectMode>("grok_bootstrap_local");
  const [projectOpening, setProjectOpening] = useState(false);
  const [customers, setCustomers] = useState<CustomerAccount[]>([]);
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);
  const [knowledgeScope, setKnowledgeScopeState] = useState<KnowledgeScope>("auto");

  // Runs this workspace started itself, so arriving at their URL does not
  // open the activity drawer the way a deep link into a run does.
  const selfStartedRunsRef = useRef<Set<string>>(new Set());
  // Conversation id -> its in-flight run, so switching away and back
  // re-subscribes to a run whose streaming message is not persisted yet.
  const liveRunsRef = useRef<Map<string, string>>(new Map());
  const loadedConversationRef = useRef<string | null>(null);
  const latestRunRef = useRef<string | null>(requestedRunId);
  const handledNewRequestRef = useRef<string | null>(null);
  // Bumped on every reset so a stale async result cannot land in the next
  // conversation.
  const generationRef = useRef(0);

  const reset = useCallback(() => {
    generationRef.current += 1;
    loadedConversationRef.current = null;
    latestRunRef.current = null;
    setConversationId(null);
    setConversationTitle("New conversation");
    setMessages([]);
    setDraft("");
    setAttachments([]);
    setArtifacts([]);
    setActiveRunId(null);
    setStageLabel(null);
    setTimelineOpen(false);
    setDecidedApprovals(new Set());
    setAnsweredElicitations(new Set());
    setDecisionBusy(null);
    setAnswerBusy(null);
    setLoadingConversation(false);
    setSending(false);
    setUploading(false);
    setQueued(null);
    setEditingMessageId(null);
    setEditDraft("");
    setRewinding(false);
    setCopiedMessageId(null);
    setFeedback({ mode: "idle", busy: false, correction: "" });
    setSavedToAccount(new Set());
    setSavingToAccount(null);
    setSelectedCustomerId(null);
    setProjectOpening(false);
    setError(null);
  }, []);

  // -- scope loading -------------------------------------------------------------
  useEffect(() => {
    const stored = window.localStorage.getItem("metis.knowledgeScope");
    if (stored === "auto" || stored === "notion" || stored === "web") setKnowledgeScopeState(stored);
  }, []);
  useEffect(() => {
    let mounted = true;
    void getLocalModelSession().then((value) => mounted && setLocalModelSession(value)).catch(() => undefined);
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
    void listCustomers().then((items) => mounted && setCustomers(items)).catch(() => undefined);
    return () => { mounted = false; };
  }, []);
  // The provider, the project catalog and the recoverable runs all refresh
  // when the window regains focus: they change in Settings or on disk.
  useEffect(() => {
    let mounted = true;
    const load = () => {
      void getModelPreference().then((value) => mounted && setModelPreference(value)).catch(() => undefined);
      void listProjectWorkspaces().then((items) => mounted && setProjects(items)).catch(() => undefined);
      void listRecoverableRuns().then((runs) => mounted && setRecoverableRuns(runs)).catch(() => undefined);
    };
    load();
    window.addEventListener("focus", load);
    return () => {
      mounted = false;
      window.removeEventListener("focus", load);
    };
  }, []);

  const setKnowledgeScope = (scope: KnowledgeScope) => {
    setKnowledgeScopeState(scope);
    window.localStorage.setItem("metis.knowledgeScope", scope);
  };

  // -- the run stream ---------------------------------------------------------
  const handleRunEvent = useCallback((event: RunEventV1) => {
    const type = event.type.toLowerCase();
    const text = stringFrom(event.payload, "delta", "text_delta", "content_delta", "content", "message", "final", "response");
    const eventArtifacts = artifactsFrom(event.payload);
    if (eventArtifacts.length) setArtifacts((current) => mergeArtifacts(current, eventArtifacts));

    if (type === "message.reasoning") {
      setMessages((current) => mergeAssistantReasoning(current, event.run_id, text ?? ""));
      return;
    }
    // The agent resolved an account from the message itself; show it as the scope.
    if (type === "customer.scoped") {
      const accountId = stringFrom(event.payload, "account_id");
      if (accountId) setSelectedCustomerId(accountId);
      return;
    }
    if (type === "stage.entered") setStageLabel(stringFrom(event.payload, "label") ?? null);
    else if (type.includes("delta") || type.includes("failed") || type.includes("completed") || type.includes("cancelled")) setStageLabel(null);

    if (TERMINAL.includes(type)) {
      liveRunsRef.current.forEach((runId, convId) => {
        if (runId === event.run_id) liveRunsRef.current.delete(convId);
      });
      updateConversationRun(
        event.run_id,
        type.includes("fail") ? "failed" : type.includes("cancel") ? "cancelled" : "done",
        document.visibilityState !== "visible",
      );
    } else if (type === "run.awaiting_approval" || type === "run.interrupted" || type === "elicitation.requested") {
      updateConversationRun(event.run_id, "attention", true);
    }
    if (type.includes("delta") || type.includes("failed") || ["assistant.message", "message.completed", "run.completed", "completed"].includes(type)) {
      setMessages((current) => mergeAssistantRunEvent(current, event, text));
    }
  }, []);

  const { events, connection, error: streamError, reconnect } = useRunEvents(activeRunId, handleRunEvent);
  const runActive = Boolean(activeRunId) && !["closed", "error"].includes(connection);
  const hasMessages = messages.length > 0 || loadingConversation;

  // -- opening a conversation, or a fresh one --------------------------------------
  useEffect(() => {
    if (newRequestToken) {
      if (handledNewRequestRef.current !== newRequestToken) {
        handledNewRequestRef.current = newRequestToken;
        reset();
      }
      router.replace("/");
      focusComposer();
      return;
    }
    handledNewRequestRef.current = null;
    if (!requestedConversationId || loadedConversationRef.current === requestedConversationId) return;
    let mounted = true;
    setConversationId(requestedConversationId);
    setConversationTitle("Opening conversation…");
    setMessages([]);
    setArtifacts([]);
    setActiveRunId(requestedRunId ?? liveRunsRef.current.get(requestedConversationId) ?? null);
    if (requestedRunId && !selfStartedRunsRef.current.has(requestedRunId)) setTimelineOpen(true);
    setFeedback({ mode: "idle", busy: false, correction: "" });
    setDecidedApprovals(new Set());
    setAnsweredElicitations(new Set());
    setLoadingConversation(true);
    setError(null);
    void Promise.all([getConversation(requestedConversationId), getConversationProject(requestedConversationId)])
      .then(([conversation, projectSession]) => {
        if (!mounted) return;
        setMessages(conversation.messages);
        setConversationTitle(conversation.title || "Conversation");
        latestRunRef.current = conversation.latest_run_id ?? null;
        setActiveRunId(requestedRunId ?? liveRunsRef.current.get(requestedConversationId) ?? conversation.latest_run_id ?? null);
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
        // The conversation in the URL is gone. Drop its ids so the run
        // stream does not poll a 404 forever, then explain.
        reset();
        setError(loadError instanceof Error ? loadError.message : "Could not open this conversation.");
        window.history.replaceState(null, "", "/");
      })
      .finally(() => mounted && setLoadingConversation(false));
    return () => { mounted = false; };
  }, [focusComposer, newRequestToken, requestedConversationId, requestedRunId, reset, router]);

  useEffect(() => {
    if (!requestedConversationId) return;
    if (!requestedRunId) {
      if (loadedConversationRef.current === requestedConversationId) {
        setArtifacts([]);
        setActiveRunId(liveRunsRef.current.get(requestedConversationId) ?? latestRunRef.current);
      }
      return;
    }
    setActiveRunId(requestedRunId);
    setArtifacts([]);
    setDecidedApprovals(new Set());
    setAnsweredElicitations(new Set());
    setStageLabel(null);
    if (!selfStartedRunsRef.current.has(requestedRunId)) setTimelineOpen(true);
  }, [requestedConversationId, requestedRunId]);

  // A reopened conversation restores its artifacts from the run they came from.
  const latestAssistant = useMemo(() => [...messages].reverse().find((message) => message.role === "assistant"), [messages]);
  useEffect(() => {
    const runId = activeRunId ?? latestAssistant?.run_id;
    if (!runId) return;
    let mounted = true;
    void getRunResult(runId)
      .then((result) => {
        const restored = artifactsFrom(result);
        if (mounted && restored.length) setArtifacts((current) => mergeArtifacts(current, restored));
      })
      .catch(() => undefined);
    return () => { mounted = false; };
  }, [activeRunId, latestAssistant?.run_id]);

  // -- sending -------------------------------------------------------------------
  const submit = useCallback(async (overrideContent?: string) => {
    const usingOverride = typeof overrideContent === "string";
    const content = (overrideContent ?? draft).trim();
    const outgoing = usingOverride ? [] : attachments;
    if ((!content && !outgoing.length) || sending || uploading) return;
    // A run is in flight: this becomes the next message. The composer clears
    // as if it had sent, because from the user's side it has.
    if (runActive) {
      setQueued({ content, attachments: outgoing });
      if (!usingOverride) {
        setDraft("");
        setAttachments([]);
      }
      return;
    }
    const generation = generationRef.current;
    setSending(true);
    setError(null);
    const sent = content || "Please review the attached file or files.";
    const userMessage: ChatMessage = { id: `optimistic-${freshToken()}`, role: "user", content: sent, attachments: outgoing, created_at: new Date().toISOString() };
    setMessages((current) => [...current, userMessage]);
    if (!usingOverride) {
      setDraft("");
      setAttachments([]);
    }
    setArtifacts([]);
    try {
      let target = conversationId;
      let title = conversationTitle;
      if (!target) {
        title = content.slice(0, 54) || outgoing[0]?.name || "New conversation";
        const created = await createConversation(title);
        if (generation !== generationRef.current) return;
        if (!created.id) throw new Error("The API did not return a conversation ID.");
        target = created.id;
        setConversationId(created.id);
        setConversationTitle(title);
        loadedConversationRef.current = created.id;
        rememberConversation({ ...created, title });
        router.replace(`/?conversation=${encodeURIComponent(created.id)}`);
      }
      const run = await sendMessage(target, sent, outgoing, selectedProjectId ? { id: selectedProjectId, mode: projectMode } : null, knowledgeScope, selectedCustomerId);
      if (generation !== generationRef.current) return;
      if (!run.run_id) throw new Error("The API did not return a run ID.");
      // Adopt the stored id, so editing this message can rewind to it.
      if (run.message_id) setMessages((current) => current.map((item) => item.id === userMessage.id ? { ...item, id: run.message_id! } : item));
      setDecidedApprovals(new Set());
      setAnsweredElicitations(new Set());
      selfStartedRunsRef.current.add(run.run_id);
      liveRunsRef.current.set(target, run.run_id);
      trackConversationRun({ conversationId: target, runId: run.run_id, title });
      setActiveRunId(run.run_id);
      latestRunRef.current = run.run_id;
      setFeedback({ mode: "idle", busy: false, correction: "" });
      router.replace(`/?conversation=${encodeURIComponent(target)}&run=${encodeURIComponent(run.run_id)}`);
      setMessages((current) => [...current, { id: `assistant-${run.run_id}`, run_id: run.run_id, role: "assistant", content: "", streaming: true }]);
    } catch (sendError) {
      if (generation !== generationRef.current) return;
      setMessages((current) => current.filter((message) => message.id !== userMessage.id));
      if (!usingOverride) {
        setDraft(content);
        setAttachments(userMessage.attachments ?? []);
      }
      setError(sendError instanceof Error ? sendError.message : "The message could not be sent.");
    } finally {
      if (generation === generationRef.current) setSending(false);
    }
  }, [attachments, conversationId, conversationTitle, draft, knowledgeScope, projectMode, router, runActive, selectedCustomerId, selectedProjectId, sending, uploading]);

  // The queued message fires when the run that blocked it ends, however it ends.
  const submitRef = useRef(submit);
  submitRef.current = submit;
  useEffect(() => {
    if (!queued || runActive || sending || uploading) return;
    const pending = queued;
    setQueued(null);
    setAttachments(pending.attachments);
    void submitRef.current(pending.content);
  }, [queued, runActive, sending, uploading]);

  const unqueue = () => {
    if (!queued) return;
    setDraft(queued.content);
    setAttachments(queued.attachments);
    setQueued(null);
  };

  const addFiles = async (files: FileList | File[]) => {
    const items = Array.from(files);
    if (!items.length) return;
    const generation = generationRef.current;
    setUploading(true);
    setError(null);
    const results = await Promise.allSettled(items.map(uploadFile));
    if (generation !== generationRef.current) return;
    const uploaded = results.flatMap((result) => (result.status === "fulfilled" ? [result.value] : []));
    if (uploaded.length) setAttachments((current) => [...current, ...uploaded]);
    const failed = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
    if (failed.length) {
      const detail = failed[0]?.reason instanceof Error ? failed[0].reason.message : "Unsupported or unreadable file.";
      setError(`${failed.length} ${failed.length === 1 ? "file" : "files"} could not be attached. ${detail}`);
    }
    setUploading(false);
  };

  const removeAttachment = (id: string) => setAttachments((current) => current.filter((item) => item.id !== id));

  /** Keep an attached document in the knowledge base, retrievable later. */
  const addToKnowledge = async (attachment: AttachmentRef) => {
    const state = knowledgeAdds[attachment.id];
    if (state === "adding" || state === "added") return;
    setKnowledgeAdds((current) => ({ ...current, [attachment.id]: "adding" }));
    try {
      await addDocumentToKnowledge(attachment.id);
      setKnowledgeAdds((current) => ({ ...current, [attachment.id]: "added" }));
    } catch (addError) {
      setKnowledgeAdds((current) => ({ ...current, [attachment.id]: "error" }));
      setError(addError instanceof ApiError ? addError.message : "That document could not be added to the knowledge base.");
    }
  };

  // -- editing, retrying, copying ------------------------------------------------
  const startEditing = (message: ChatMessage) => {
    if (runActive || rewinding || !isPersisted(message)) return;
    setEditingMessageId(message.id);
    setEditDraft(message.content);
  };
  const cancelEditing = () => {
    setEditingMessageId(null);
    setEditDraft("");
  };

  /** Rewind to a message, then send new text as a fresh turn. */
  const rewindAndSend = async (messageId: string, content: string, failure: string) => {
    if (!conversationId || rewinding) return;
    setRewinding(true);
    setError(null);
    const generation = generationRef.current;
    try {
      const rewound = await rewindConversation(conversationId, messageId);
      if (generation !== generationRef.current) return;
      setMessages(rewound.messages);
      setArtifacts([]);
      setActiveRunId(null);
      latestRunRef.current = null;
      setStageLabel(null);
      cancelEditing();
      await submitRef.current(content);
    } catch (rewindError) {
      if (generation !== generationRef.current) return;
      setError(rewindError instanceof Error ? rewindError.message : failure);
    } finally {
      if (generation === generationRef.current) setRewinding(false);
    }
  };

  const submitEdit = async (message: ChatMessage) => {
    const content = editDraft.trim();
    if (!content) return;
    if (content === message.content) {
      cancelEditing();
      return;
    }
    // The original attachments belong to a turn that no longer exists.
    await rewindAndSend(message.id, content, "That message could not be edited.");
  };

  /** Ask the same question again, usually after switching the model. Rewinds
   *  to the question so the model does not see it twice. */
  const retryAnswer = async (assistant: ChatMessage) => {
    if (runActive || rewinding) return;
    const index = messages.findIndex((item) => item.id === assistant.id);
    const question = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");
    if (!question || !isPersisted(question)) return;
    await rewindAndSend(question.id, question.content, "That answer could not be retried.");
  };

  const copyMessage = async (message: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedMessageId(message.id);
      window.setTimeout(() => setCopiedMessageId((current) => (current === message.id ? null : current)), 1600);
    } catch {
      setError("This browser would not give Metis access to the clipboard.");
    }
  };

  /** Put a failed reply's question back in the composer and drop the reply. */
  const clearFailedResponse = (message: ChatMessage, restoreQuestion: boolean) => {
    if (restoreQuestion) {
      const index = messages.findIndex((item) => item.id === message.id);
      const previous = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");
      if (previous) setDraft(previous.content);
    }
    setMessages((current) => current.filter((item) => item.id !== message.id));
    setError(null);
    if (message.run_id && message.run_id === activeRunId) {
      setActiveRunId(null);
      setArtifacts([]);
      setStageLabel(null);
      setTimelineOpen(false);
      router.replace(conversationId ? `/?conversation=${encodeURIComponent(conversationId)}` : "/");
    }
    focusComposer();
  };

  const startFresh = () => {
    const token = freshToken();
    handledNewRequestRef.current = token;
    reset();
    router.push(`/?new=${token}`);
    focusComposer();
  };

  // -- decisions, answers, proposals, feedback, cancel ----------------------------
  const pendingApproval = latestPendingApproval(events, decidedApprovals);
  const pendingElicitation = latestPendingElicitation(events, answeredElicitations);
  const pendingSuggestion = latestActionSuggestion(events, dismissedProposals);
  const suggestionState: "idle" | "applying" | "applied" = !pendingSuggestion
    ? "idle"
    : appliedProposals.has(pendingSuggestion.proposal_id) ? "applied"
      : applyingProposal === pendingSuggestion.proposal_id ? "applying" : "idle";

  // Only a genuine on-device model can be relaunched by an approval.
  const approveLabel = !isCloudActive(modelPreference) && localModelSession?.state === "off" && localModelSession.selected_model
    ? `Approve & relaunch ${localModelSession.selected_model}`
    : "Approve once";

  const decide = async (approvalId: string, decision: "approve" | "reject") => {
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
  };

  const answer = async (elicitationId: string, reply: { option?: string; text?: string }) => {
    if (!activeRunId) return;
    setAnswerBusy(elicitationId);
    setError(null);
    try {
      await answerElicitation(activeRunId, { ...reply, elicitationId });
      setAnsweredElicitations((current) => new Set(current).add(elicitationId));
      if (connection === "closed" || connection === "error") reconnect();
    } catch (answerError) {
      setError(answerError instanceof ApiError ? answerError.message : "The answer could not be sent.");
    } finally {
      setAnswerBusy(null);
    }
  };

  const applyProposal = async (proposalId: string) => {
    setApplyingProposal(proposalId);
    setError(null);
    try {
      await applyCustomerProposal(proposalId);
      setAppliedProposals((current) => new Set(current).add(proposalId));
    } catch (applyError) {
      setError(applyError instanceof Error ? applyError.message : "That extraction could not be applied.");
    } finally {
      setApplyingProposal(null);
    }
  };
  const dismissProposal = (proposalId: string) => setDismissedProposals((current) => new Set(current).add(proposalId));

  const stop = async () => {
    if (!activeRunId) return;
    try {
      await cancelRun(activeRunId);
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : "The run could not be cancelled.");
    }
  };

  const rate = async (rating: "positive" | "negative") => {
    if (!activeRunId || feedback.busy) return;
    if (rating === "negative" && feedback.mode !== "correcting") {
      setFeedback((current) => ({ ...current, mode: "correcting" }));
      return;
    }
    setFeedback((current) => ({ ...current, busy: true }));
    setError(null);
    try {
      await submitFeedback(activeRunId, rating, rating === "negative" ? feedback.correction.trim() : undefined);
      setFeedback({ mode: "sent", busy: false, correction: "" });
    } catch (feedbackError) {
      setError(feedbackError instanceof Error ? feedbackError.message : "Feedback could not be recorded.");
      setFeedback((current) => ({ ...current, busy: false }));
    }
  };
  const setCorrection = (correction: string) => setFeedback((current) => ({ ...current, correction }));
  const cancelCorrection = () => setFeedback({ mode: "idle", busy: false, correction: "" });

  // -- scope actions -----------------------------------------------------------------
  const selectedProject = useMemo(() => projects.find((project) => project.id === selectedProjectId) ?? null, [projects, selectedProjectId]);
  const selectedCustomer = useMemo(() => customers.find((customer) => customer.id === selectedCustomerId) ?? null, [customers, selectedCustomerId]);

  /** True only when a project actually opened. `known` is a freshly listed
   *  catalog, for a caller that just created the project. */
  const chooseProject = async (projectId: string | null, known: ProjectWorkspace[] = projects): Promise<boolean> => {
    if (projectOpening || runActive) return false;
    if (!projectId) {
      setSelectedProjectId(null);
      return false;
    }
    const target = known.find((project) => project.id === projectId);
    const blocked = projectModeBlocked(projectMode, modelPreference)
      ?? (target?.initialized || projectMappingReady(modelPreference) ? null : "Mapping a project for the first time needs ClinePass, Grok, or Command A+ configured in Settings.");
    if (blocked) {
      setError(blocked);
      return false;
    }
    setProjectOpening(true);
    setError(null);
    try {
      const opened = await openProjectWorkspace(projectId, projectMode);
      setProjects((current) => current.map((item) => (item.id === opened.id ? opened : item)));
      setSelectedProjectId(opened.id);
      return true;
    } catch (openError) {
      setError(openError instanceof Error ? openError.message : "Metis could not open that project.");
      return false;
    } finally {
      setProjectOpening(false);
    }
  };

  const createProject = async (name: string) => {
    if (projectOpening || runActive) return;
    setProjectOpening(true);
    setError(null);
    try {
      const created = await createProjectAsset(name);
      const found = await listProjectWorkspaces();
      setProjects(found);
      setProjectOpening(false);
      await chooseProject(created.id, found);
    } catch (createError) {
      setProjectOpening(false);
      setError(createError instanceof Error ? createError.message : "Metis could not create that project.");
    }
  };

  const chooseProjectMode = async (mode: ProjectMode) => {
    if (projectOpening || runActive || projectMode === mode) return;
    const blocked = projectModeBlocked(mode, modelPreference);
    if (blocked) {
      setError(blocked);
      return;
    }
    const previous = projectMode;
    setProjectMode(mode);
    if (!selectedProjectId) return;
    setProjectOpening(true);
    setError(null);
    try {
      const opened = await openProjectWorkspace(selectedProjectId, mode);
      setProjects((current) => current.map((item) => (item.id === opened.id ? opened : item)));
    } catch (openError) {
      setProjectMode(previous);
      setError(openError instanceof Error ? openError.message : "The project mode could not be changed.");
    } finally {
      setProjectOpening(false);
    }
  };

  const chooseProvider = async (provider: "local" | "oci" | "cohere" | "cline") => {
    if (providerSaving || modelPreference?.provider === provider) return;
    const missing =
      provider === "oci" && !modelPreference?.oci_available ? "Cloud reasoning is not configured yet. Add the OCI project settings before selecting it."
        : provider === "cohere" && !modelPreference?.cohere_available ? "Cohere is not configured yet. Add WAQIL_COHERE_API_KEY before selecting it."
          : provider === "cline" && !clinePassReady(modelPreference?.cline_available === true, modelPreference?.cline_models ?? []) ? "ClinePass is not configured yet. Add WAQIL_CLINE_API_KEY before selecting it."
            : null;
    if (missing) {
      setError(missing);
      return;
    }
    setProviderSaving(true);
    setError(null);
    try {
      const current = modelPreference ?? { mode: "split" as const, model: null, provider: "local" as const, oci_tools: ["code_interpreter" as const], oci_available: false, cohere_available: false };
      setModelPreference(await saveModelPreference(current.mode, current.model, provider, current.oci_tools));
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "The model route could not be changed.");
    } finally {
      setProviderSaving(false);
    }
  };

  /** Keep an answer on the scoped account as a note in your own words. */
  const saveToAccount = async (message: ChatMessage) => {
    if (!selectedCustomerId || !message.content.trim() || savingToAccount) return;
    setSavingToAccount(message.id);
    setError(null);
    try {
      await createCustomerNote(selectedCustomerId, { title: conversationTitle.slice(0, 200), body: message.content, origin: "chat", origin_ref: conversationId ?? "" });
      setSavedToAccount((current) => new Set(current).add(message.id));
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "That answer could not be saved to the account.");
    } finally {
      setSavingToAccount(null);
    }
  };

  /** Drop the account's activity-tracker update into the thread. It is
   *  generated server-side and persisted on the account; the chat copy is
   *  a client-side echo. */
  const trackerUpdate = async () => {
    if (!selectedCustomerId || trackerBusy || runActive) return;
    setTrackerBusy(true);
    setError(null);
    try {
      const output = await createCustomerOutput(selectedCustomerId);
      const body = output.content.trim();
      if (!body) {
        setError("The tracker update came back empty — this account has no interactions to summarise yet.");
        return;
      }
      setMessages((current) => [...current, { id: `tracker-${freshToken()}`, role: "assistant", content: body, kind: "tracker", created_at: new Date().toISOString() }]);
    } catch (trackerError) {
      setError(trackerError instanceof Error ? trackerError.message : "The tracker update could not be generated.");
    } finally {
      setTrackerBusy(false);
    }
  };

  // How many sources grounded the current run's answer, for the reply's footer.
  const groundedSources = useMemo(() => {
    let count = 0;
    for (const event of events) {
      if (event.type === "context.retrieved") {
        const value = Number(event.payload.knowledge_snippet_count ?? 0);
        if (Number.isFinite(value)) count = Math.max(count, value);
      }
    }
    return count;
  }, [events]);

  return {
    // conversation
    conversationId, conversationTitle, messages, loadingConversation, hasMessages, error, setError, startFresh,
    // composer
    draft, setDraft, attachments, addFiles, removeAttachment, uploading, sending, submit, queued, unqueue, knowledgeAdds, addToKnowledge, composerRef, focusComposer,
    // run
    activeRunId, runActive, events, connection, streamError, stageLabel, artifacts, stop, recoverableRuns, timelineOpen, setTimelineOpen,
    pendingApproval, pendingElicitation, pendingSuggestion, suggestionState, approveLabel, decide, decidedApprovals, decisionBusy, answer, answeredElicitations, answerBusy, applyProposal, dismissProposal, groundedSources,
    // message actions
    editingMessageId, editDraft, setEditDraft, startEditing, cancelEditing, submitEdit, retryAnswer, rewinding, copyMessage, copiedMessageId, clearFailedResponse,
    feedback, rate, setCorrection, cancelCorrection, saveToAccount, savedToAccount, savingToAccount, trackerUpdate, trackerBusy, latestAssistant,
    // scope
    modelPreference, setModelPreference, providerSaving, chooseProvider, localModelSession,
    projects, selectedProject, selectedProjectId, chooseProject, createProject, projectMode, chooseProjectMode, projectOpening, setProjects,
    customers, selectedCustomer, selectedCustomerId, setSelectedCustomerId, knowledgeScope, setKnowledgeScope,
  };
}
