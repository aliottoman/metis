// Wires the form and the talk button to one ElevenLabs WebRTC conversation.
// The token comes from our own server; the API key never reaches this file.

import { Conversation } from "https://esm.sh/@elevenlabs/client";

const els = {
  orb: document.getElementById("orb"),
  stateLabel: document.getElementById("state-label"),
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
  error: document.getElementById("error"),
  log: document.getElementById("log"),
  panelStatus: document.getElementById("panel-status"),
  jobTitle: document.getElementById("job-title"),
  company: document.getElementById("company"),
  jd: document.getElementById("jd"),
};

let conversation = null;

// Returns the three context fields, trimmed.
function readForm() {
  return {
    job_title: els.jobTitle.value.trim(),
    company_name: els.company.value.trim(),
    job_description: els.jd.value.trim(),
  };
}

// Enables the start button only once all three fields have something in them.
function refreshStartButton() {
  const form = readForm();
  const ready = form.job_title && form.company_name && form.job_description;
  els.start.disabled = !ready || conversation !== null;
  if (!conversation) {
    els.stateLabel.textContent = ready
      ? "Ready. Press start and it will brief you."
      : "Fill in the role, then start.";
  }
}

// Paints the orb and the label for the current conversation state.
function setState(state, label) {
  els.orb.dataset.state = state;
  els.stateLabel.textContent = label;
}

// Shows an error under the controls, or clears it.
function setError(message) {
  els.error.textContent = message || "";
  els.error.hidden = !message;
}

// Appends one line to the running transcript.
function appendLog(source, text) {
  if (!text) return;
  const item = document.createElement("li");
  item.className = source === "user" ? "line line-user" : "line line-agent";
  item.textContent = text;
  els.log.appendChild(item);
  els.log.scrollTop = els.log.scrollHeight;
}

// Asks our server for a one-conversation token and the variables to send with it.
async function openSession(form) {
  const response = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(form),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Could not start the session (HTTP ${response.status}).`);
  }
  return response.json();
}

// Starts the interview: mic permission, token, then the WebRTC conversation.
async function start() {
  setError("");
  els.start.disabled = true;
  setState("connecting", "Connecting…");

  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setState("idle", "Microphone access is needed to run an interview.");
    setError("The browser blocked the microphone. Allow it and try again.");
    refreshStartButton();
    return;
  }

  try {
    const opened = await openSession(readForm());

    conversation = await Conversation.startSession({
      conversationToken: opened.conversation_token,
      connectionType: "webrtc",
      dynamicVariables: opened.dynamic_variables,
      onConnect: () => setState("listening", "Listening."),
      onDisconnect: () => finish(),
      onError: (message) => {
        setError(typeof message === "string" ? message : "The voice connection failed.");
        finish();
      },
      onModeChange: ({ mode }) =>
        mode === "speaking"
          ? setState("speaking", "Speaking…")
          : setState("listening", "Listening."),
      onMessage: ({ message, source }) => appendLog(source, message),
    });

    els.stop.hidden = false;
    lockForm(true);
  } catch (error) {
    setError(error.message || "Could not start the interview.");
    setState("idle", "Fill in the role, then start.");
    conversation = null;
    refreshStartButton();
  }
}

// Ends the conversation from this side.
async function stop() {
  if (!conversation) return;
  const active = conversation;
  conversation = null;
  await active.endSession().catch(() => {});
  finish();
}

// Returns the page to its idle state after a conversation ends for any reason.
function finish() {
  conversation = null;
  els.stop.hidden = true;
  lockForm(false);
  setState("idle", "Interview ended. The debrief is in the transcript above.");
  refreshStartButton();
}

// Freezes the context fields while a conversation is live.
function lockForm(locked) {
  els.jobTitle.disabled = locked;
  els.company.disabled = locked;
  els.jd.disabled = locked;
}

// Warns on load if the server is missing its ElevenLabs configuration.
async function checkHealth() {
  try {
    const health = await (await fetch("/api/health")).json();
    if (!health.api_key_set || !health.agent_id_set) {
      els.panelStatus.textContent =
        "Server is missing INTERVIEW_ELEVENLABS_API_KEY or INTERVIEW_ELEVENLABS_AGENT_ID.";
      els.panelStatus.classList.add("bad");
    }
  } catch {
    // A failed health check is not worth blocking the page over.
  }
}

for (const field of [els.jobTitle, els.company, els.jd]) {
  field.addEventListener("input", refreshStartButton);
}
els.start.addEventListener("click", start);
els.stop.addEventListener("click", stop);
window.addEventListener("pagehide", () => {
  if (conversation) conversation.endSession().catch(() => {});
});

refreshStartButton();
checkHealth();
