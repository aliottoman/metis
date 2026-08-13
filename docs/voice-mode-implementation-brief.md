# Implementation brief — Voice Mode for Metis

**For:** Claude Code or Codex working in `/Users/aliottoman/Developer/metis`
**Status:** Ready for implementation after the three owner confirmations in §3
**Primary deliverable:** A restricted voice conversation mode inside the existing Metis chat surface. ElevenLabs provides speech recognition, turn-taking, interruption handling, and speech synthesis; Metis remains the only reasoner and the only component allowed to retrieve data or mutate local records.

This is an extension to a mature local-first application, not a separate voice app. Match the existing architecture, naming, tests, comment density, and visual language.

---

## 1. Read first

Read these files before changing code:

1. `README.md`
2. `docs/architecture.md`
3. `apps/api/src/waqil_api/api.py`, especially `POST /transcribe`
4. `apps/api/src/waqil_api/control_plane.py`, especially retrieval, synthesis, grounding, publication, and customer execution
5. `apps/api/src/waqil_api/model_provider.py`
6. `apps/api/src/waqil_api/policy.py`
7. `apps/api/src/waqil_api/customer_tools.py`
8. `apps/api/src/waqil_api/customer_intelligence.py`
9. `apps/api/src/waqil_api/database.py`, especially customer tables and write methods
10. `apps/web/hooks/use-dictation.ts`
11. `apps/web/components/chat-workspace.tsx`
12. `apps/web/components/settings-panel.tsx`
13. `apps/mac/Sources/Metis/WebWindow.swift`

### Invariants that must remain true

- The trusted FastAPI process on `127.0.0.1:8000` remains the only component that decides route, risk, permissions, retrieval, and mutation.
- Port 8000 is never exposed through a tunnel.
- Microphone transcripts, meeting transcripts, webhook payloads, retrieved documents, and caller-supplied chat history are untrusted evidence. None may approve a governed action, widen a capability, enable networking, activate a tool, or enter durable memory automatically.
- Enabling a voice session pre-authorizes only the narrow voice capability defined in this brief. It does not count as either of Metis's tool approval steps.
- Cloud use remains explicit. The UI must say plainly that voice audio is sent to ElevenLabs and that retrieved evidence may be sent to the selected cloud model.
- Local vectors and indexes never leave the machine.
- Voice cannot approve, build, delete, overwrite, execute code, use a sandbox, access arbitrary files, or author or activate tools.
- Existing behavior remains unchanged when voice is disabled and when dictation continues to use Cohere.

---

## 2. Product boundaries

This program contains four independently reviewable capabilities:

1. ElevenLabs as an optional provider for existing composer dictation.
2. Interactive voice conversation in the existing chat surface.
3. Narrow, append-only customer writes initiated by explicit voice commands.
4. Meeting recording ingestion and a spoken morning brief.

They must not land as one indivisible change. Follow the delivery phases in §13.

### Voice may read

- Customer accounts, people, facts, open actions, wins, and recent notes
- The Notion corpus and general corpus, with natural spoken source attribution
- The answer bank
- A specific attention item when asked about it
- Meeting transcripts created by the Meetings surface
- Bounded recent conversation context
- Ordinary clarification and follow-up turns

The morning brief is not a conversational voice capability. It remains a Today-tab playback action.

### Voice may write

Only create or append operations on:

- customer notes
- customer facts
- customer actions
- customer people
- customer wins

Every voice write must use a dedicated voice-domain service over the existing customer store. Do not expose the general `/customers/*` mutation API or the normal customer tool catalog to the voice model.

### Voice may never

- Delete or overwrite an existing customer record
- Upsert a person over an existing person
- Change an action's status
- Apply an extraction proposal
- Create, edit, build, activate, or run a tool
- Build or modify a project or application
- Write a file or artifact
- Execute in Podman or any other sandbox
- Approve an interrupted run
- Access secrets or arbitrary network destinations
- Select its own model or permissions

---

## 3. Owner confirmations required before implementation

Confirm these three items. Do not infer them:

1. **Customer auto-link threshold.** Proposed default: auto-link only at confidence `>= 0.92` and only when the best match leads the second-best match by at least `0.10`. Exact normalized account names and unique aliases may be handled as deterministic matches. Everything else is proposed for review.
2. **Stock ElevenLabs voice ID.** Add the setting now; an empty value keeps voice unavailable until the owner supplies it.
3. **Corporate-device tunnel permission.** If this Mac is Oracle-managed, the owner must confirm that a named Cloudflare Tunnel exposing the isolated ingress process is permitted. Do not start tunnel work until confirmed.

Already decided; do not re-ask:

- Transport is ElevenLabs Agents Platform with Custom LLM over a named Cloudflare Tunnel.
- English only.
- Stock voice.
- Initial voice reasoning model is `deepseek-v4-flash:cloud`, held in a separate voice preference.
- Voice may perform only the append-only customer writes defined here.
- Visual receipts are the default safety mechanism. Spoken confirmation defaults off.
- Meetings are in scope as a later delivery phase.
- The morning brief is a Today-tab button.

---

## 4. Architecture

### 4.1 Runtime shape

```text
Browser / Metis WKWebView
  ├─ loopback HTTP + SSE ─────────────► Metis API :8000
  │                                      trusted policy, retrieval,
  │                                      voice graph, customer writes,
  │                                      events and persistence
  │
  └─ @elevenlabs/react ── WebRTC ─────► ElevenLabs Agents Platform
                                           Scribe ASR
                                           turn-taking and interruption
                                           Flash v2.5 TTS
                                                │
                                                │ Custom LLM request
                                                ▼
                                         named Cloudflare Tunnel
                                                │
                                                ▼
                                      isolated voice ingress process
                                      separate port and ASGI app
                                                │
                                                │ loopback HTTP only
                                                ▼
                                           Metis API :8000
```

ElevenLabs supplies ears, mouth, and conversational timing. Metis supplies every thought, source, decision, and mutation.

Do not configure ElevenLabs Knowledge Base, RAG, server-side domain tools, batch calling, music, or sound effects.

### 4.2 Isolated ingress process

Create a separate FastAPI application and process. It must not import the main Metis ASGI app, mount a Metis router, reuse the main middleware stack, or hold a database, blob-store, corpus, customer-service, or model-broker handle.

It is a protocol adapter with exactly these public routes:

- `GET /health`
- `POST /v1/chat/completions`
- `POST /v1/elevenlabs/post-call`

The isolation test must compare the registered route set to this allowlist and inspect application state for forbidden handles. Any future route addition must fail that test until deliberately reviewed.

`POST /v1/chat/completions`:

- Accept the OpenAI Chat Completions request shape required by ElevenLabs.
- Require `Authorization: Bearer <WAQIL_VOICE_SHARED_SECRET>` before reading or forwarding the body.
- Require streaming and return valid SSE chunks ending in `[DONE]`.
- Accept the configured public model alias in the required `model` field, but never use it to choose Metis's internal model. Reject any other alias with `422`.
- Discard caller-supplied system messages as authority. Only the latest user turn and bounded conversational messages are forwarded as untrusted conversation evidence.
- Reject unexpected tools. Only the specifically configured ElevenLabs system tools may appear.
- Apply body-size limits, concurrency limits, request timeouts, and per-session rate limits.
- Forward only to the dedicated internal voice endpoint over loopback.

`POST /v1/elevenlabs/post-call`:

- Verify the raw-body ElevenLabs HMAC signature and timestamp before parsing.
- Be idempotent on the ElevenLabs conversation ID and event type.
- Forward the verified payload to a dedicated loopback ingestion endpoint.
- Return promptly; long processing occurs as a background job in Metis.
- Never turn extracted content into customer facts, actions, memory, or account links automatically.

Cloudflare Access protects the Custom LLM route with a service-token policy. The webhook route uses ElevenLabs HMAC verification and, where available, an egress-IP allowlist. Document the intentional Access policy exception for the webhook path.

### 4.3 Session and tunnel lifecycle

The trusted Metis API owns session lifecycle:

1. The browser requests a voice session over loopback.
2. Metis verifies configuration and corporate-tunnel permission.
3. Metis starts the isolated ingress process if necessary.
4. Metis starts the named `cloudflared` connector.
5. Metis mints a short-lived ElevenLabs signed conversation URL server-side.
6. The browser begins WebRTC and renews a short Metis session lease.
7. The session ends on explicit stop, idle timeout, page hide, navigation, WebRTC failure, or expired server lease.
8. When no active voice lease remains, Metis stops `cloudflared` and the ingress process.

Client cleanup is helpful but not trusted. A browser crash must still cause the server lease to expire and close the tunnel.

The API key never reaches the browser. The bearer secret never appears in logs, SSE payloads, browser state, or exceptions.

### 4.4 Session identity and two-channel output

Define stable identifiers:

- `voice_session_id`: one open voice session in Metis
- `provider_conversation_id`: the ElevenLabs conversation ID
- `turn_id`: one finalized user utterance and its response
- `run_id`: the corresponding Metis run/event stream identity

Every request, event, write receipt, post-call record, and log entry must carry the relevant identifiers.

There are two output channels:

1. The Custom LLM SSE response contains only the short spoken rendition that ElevenLabs will synthesize.
2. A loopback Metis SSE stream carries the full written answer, citations, state changes, write receipts, errors, and build-deferred events to the browser.

Do not try to smuggle the written answer through speech text or provider metadata.

---

## 5. Restricted voice graph

Build a separate, explicitly compiled voice graph or coordinator. Reuse extracted services and contracts from the normal graph; do not call private graph nodes as an ad hoc pipeline and do not route through the normal planner.

The read path is:

```text
validate session
  → classify fixed refusals
  → resolve bounded customer context
  → retrieve permitted evidence
  → synthesize written + spoken output
  → persist and publish voice events
```

The narrow write path is:

```text
validate session
  → classify fixed refusals
  → detect explicit append command
  → resolve exactly one account
  → validate create-only payload
  → enforce CUSTOMER_APPEND policy
  → idempotent transaction
  → emit visual receipt + timeline event
  → synthesize a short acknowledgement
```

The voice graph receives no registry, sandbox, project engine, build coordinator, approval-resume service, arbitrary customer mutation client, or filesystem grant.

### Policy contract

Add a canonical `PolicyPermission.CUSTOMER_APPEND` at `RiskLevel.R2`. Do not represent customer mutation as `WRITE_RUN_ARTIFACTS`; it is a different domain capability.

The voice ceiling is:

- `CONVERSATION_RESPONSE`
- permitted read capabilities needed for bounded retrieval
- `MODEL_BROKER`
- `CUSTOMER_APPEND`
- no execution boundary
- no general network permission

Policy is evaluated again after every model-authored structured response. The model cannot add a permission by returning a new action name.

### Retrieval

Reuse the existing customer evidence, answer-bank, Notion, corpus, and attention services through public service methods. Add an explicit voice retrieval profile that excludes:

- active tool catalogs
- project context
- attachments not deliberately provided to this voice session
- long-term memory harvesting
- web research unless a later brief explicitly adds it

Cloud consent remains source-specific. Voice mode must not silently widen which corpus material may leave the machine.

### Rendition

The synthesis result must have separate fields:

```json
{
  "written": "Full grounded answer with normal Metis citations.",
  "spoken": "A natural two-to-four sentence rendition.",
  "citations": [],
  "intent": "read | customer_append | refuse_build | clarify_account | navigation",
  "write": null
}
```

The default spoken response is two to four sentences. Longer spoken output is allowed only when the user explicitly asks for detail.

The spoken rendition must not vocalize Markdown syntax, URLs, tables, fenced code, Mermaid, or bracketed citation markers. It should attribute evidence naturally, for example: “I found three relevant sources; the strongest is the Batelco service request.”

If the model omits or returns an invalid `spoken` field, apply a deterministic Markdown-to-speech normalizer. It must:

- strip code fences and inline-code markers
- replace a table with a short description of its dimensions
- remove URLs while retaining useful link labels
- remove bracketed citation markers
- flatten headings and lists into ordinary sentences
- apply a hard character and sentence limit

The written answer remains complete on screen. Skipping the normal `ground_review` revision is allowed only for the spoken rendition. The written answer must still be defensibly grounded by deterministic citation validation or another bounded voice-specific check.

---

## 6. Build refusal and protected intents

Build refusal is enforced before retrieval or model invocation.

Reuse these existing classifiers exactly:

```python
from .model_provider import (
    describes_a_new_tool,
    is_explicit_build_request,
    is_explicit_toolify_request,
    is_new_application_request,
    is_project_build_instruction,
)
```

If any classifier returns true:

1. Do not enter the voice graph.
2. Do not call a model.
3. Do not invoke any tool or write service.
4. Return the fixed spoken refusal.
5. Emit `voice.build_deferred` with the verbatim finalized transcript, `voice_session_id`, `turn_id`, and `run_id`.
6. Prefill the normal chat composer and show a small handoff card.

Store this exact refusal once as a module constant:

> I can't build from voice — that needs the chat window. I've dropped it in your composer, so it's ready when you switch over.

Apply the same host-side pattern to approval attempts, deletes, overwrites, filesystem requests, tool activation, sandbox execution, and model-selection requests. Each gets a fixed safe response and no graph entry.

---

## 7. Voice customer writes

### 7.1 Authorization semantics

There is no mandatory spoken read-back in the default mode.

A write may occur only when the current finalized utterance contains an explicit imperative to create or append a supported record, for example:

- “Add a note to Batelco that the workshop moved to Thursday.”
- “Create an action for me to send the sizing by Friday.”
- “Record this as a win for BAPCO.”

Ordinary statements, inferred takeaways, earlier turns, retrieved evidence, post-call analysis, and model suggestions cannot trigger a write.

Starting voice mode pre-authorizes this narrow R2 append capability for the session. A transcript still cannot approve any governed action or expand the voice capability.

When `spoken_confirmation` is enabled, use a two-turn flow: read back the exact proposed record, obtain an unambiguous yes in the immediately following turn, and then commit. Confirmation expires on interruption, topic change, timeout, account change, or session end. It defaults off.

### 7.2 Account resolution

- Exactly one account must resolve above the configured threshold.
- An explicitly scoped customer page may supply the default account, but the receipt must still name it.
- If more than one account is plausible, do not write. Ask one short question or show a picker.
- Never guess from the currently visible account when the utterance explicitly names a different or ambiguous account.

### 7.3 Create-only service

Implement dedicated create-only operations for all five record types.

In particular, do not reuse `upsert_customer_person` for voice. A matching person must produce a clarification or a refusal to overwrite, never an update.

The model may return only a typed candidate payload. The trusted host copies the relevant transcript excerpt verbatim and validates lengths, dates, account identity, and supported record type before committing.

### 7.4 Idempotency and transactionality

Every write requires an idempotency key derived from `voice_session_id`, `turn_id`, and the single selected write intent. A retry of the Custom LLM request must return the original result without creating another row.

The customer row, provenance, write receipt, outbox/timeline event, and idempotency record must commit in one database transaction.

### 7.5 Provenance and receipt

Persist an immutable voice-write receipt containing:

- receipt ID
- record type and record ID
- account ID and display name
- `source: "voice"`
- `voice_session_id`
- `provider_conversation_id`
- `turn_id`
- `run_id`
- verbatim transcript excerpt
- normalized payload that was committed
- creation timestamp
- undo timestamp and actor, when applicable

The browser immediately displays a silent confirmation card such as:

> Added note to Batelco: “Workshop moved to Thursday.”

The card includes one obvious **Undo** control.

### 7.6 Undo

Undo is a UI-only compensating action, not a voice delete capability.

- The write response returns a short-lived, single-use undo token bound to the exact receipt and browser session.
- The loopback UI calls a dedicated undo endpoint.
- The endpoint may reverse only the record created by that receipt.
- It cannot accept a caller-supplied record ID or record type.
- It fails if the record has subsequently been edited or if the token was used or expired.
- The reversal and receipt update are atomic.
- The immutable receipt remains after reversal for audit.
- No undo or delete operation is present in the Custom LLM schema or voice service interface.

---

## 8. ElevenLabs provider and dictation

Add an ElevenLabs provider beside Cohere in `model_provider.py`. Its `transcribe()` method must preserve the existing call signature.

Settings:

- `WAQIL_ELEVENLABS_API_KEY`
- `WAQIL_ELEVENLABS_AGENT_ID`
- `WAQIL_ELEVENLABS_STT_MODEL`
- `WAQIL_ELEVENLABS_TTS_MODEL`
- `WAQIL_ELEVENLABS_VOICE_ID`
- `WAQIL_ELEVENLABS_TRANSCRIBE_MAX_BYTES`
- `WAQIL_VOICE_SHARED_SECRET`
- `WAQIL_VOICE_PUBLIC_MODEL_ALIAS`
- `WAQIL_VOICE_MODEL`
- named-tunnel configuration values

Add empty secret values to `.env.example` using the existing Cohere-key pattern.

Runtime preferences belong in a new persisted `GET|PUT /settings/speech` contract:

- `stt_provider: cohere | elevenlabs`
- `spoken_confirmation: boolean`
- voice model choice from a server-owned allowlist
- availability and configuration diagnostics

Environment values provide startup defaults; the settings endpoint persists owner choices. Do not imply that a browser radio edits environment variables.

Make audio-container acceptance provider-aware. Preserve the Cohere path byte-for-byte when Cohere is selected. Do not globally widen Cohere's documented allowlist because ElevenLabs supports more containers.

Update dictation UI copy so it names the selected provider instead of hard-coding Cohere.

---

## 9. Voice surface

Voice mode lives inside the existing chat workspace, not on a new page.

Use `@elevenlabs/react` and the existing Metis microphone visual vocabulary. Show:

- start and stop controls
- listening, thinking, speaking, interrupted, reconnecting, and failed states
- interim transcript that visibly settles when finalized
- an obvious interrupt control
- the full written answer and citations
- write receipts and undo
- build-deferred handoff cards
- elapsed billable session time
- a clear idle-timeout countdown near expiry
- a one-time cloud-data disclosure before first use

State must be honest. Silence during retrieval or a slow model call must appear as a named state, not resemble a dropped connection.

Implement teardown on explicit stop, idle timeout, route change, component unmount, `pagehide`, failed WebRTC, and expired server lease. Add native WKWebView coverage for WebRTC audio capture and page-lifecycle behavior; browser-only success is insufficient for Metis.

Client tools may perform navigation only, such as opening a customer or the attention feed. They may not mutate data or call arbitrary URLs.

---

## 10. Post-call ingestion

The verified ElevenLabs post-call webhook creates or updates one idempotent ingestion job.

The transcript may be stored as conversation evidence and indexed only under the existing corpus-consent rules. Structured extraction produces proposals, not committed customer records, durable memories, actions, or account links.

Record:

- provider conversation ID
- source voice session and Metis conversation IDs
- raw verified event hash
- transcript turns
- analysis payload as untrusted provider output
- processing status and errors

Never treat provider analysis or data-collection fields as trusted host decisions.

---

## 11. Meetings

Meetings are a separate delivery phase and the only new main route.

### 11.1 Ingestion pipeline

```text
upload
  → content-addressed blob
  → optional ElevenLabs audio isolation
  → Scribe batch transcription
  → diarized speaker turns + word timestamps
  → persisted meeting record
  → summary, decision, action and customer-link proposals
```

Use a durable background job with observable stages and replayable events. A failed stage must preserve enough state to retry without orphaning or duplicating the blob.

Persist original audio, isolated-audio reference when used, transcript text, speaker turns, word timings, provider request identity, processing state, and errors.

### 11.2 Meetings UI

Add:

- `apps/web/app/meetings/`
- `apps/web/components/meetings-workbench.tsx`

Follow existing workbench structure and Metis design tokens. Support:

- drag-and-drop upload
- durable progress states
- audio playback
- synchronized word highlighting
- click-to-seek
- visually separated speaker turns
- persistent speaker naming
- suggested names from people on an accepted linked account
- transcript correction
- proposed customer links, decisions, and actions

### 11.3 Customer linking

Use deterministic normalized names and aliases first, then the existing entity graph and customer search.

- Auto-link only when the confirmed threshold and winner-margin rules pass.
- Below threshold, create a proposal requiring one click.
- Every accepted or automatic link records the transcript span and timestamps that produced it.
- Derived customer actions remain proposals until accepted.

### 11.4 Transcript correction and alignment

Scribe word timestamps are sufficient during normal ingestion. Do not run forced alignment then.

When the owner corrects a line:

1. Identify the smallest affected speaker segment and bounded audio interval.
2. Send only the corrected plain text and bounded audio to forced alignment.
3. Do not send diarized multi-speaker text; forced alignment does not support it.
4. Validate monotonic timings and splice the corrected interval into the existing timeline.
5. Preserve the original transcript and an edit audit record.

---

## 12. Spoken morning brief

Add a **Listen** button to the existing Today surface. It is independent of interactive voice mode and should ship earlier.

Requirements:

- Fetch the existing structured morning brief.
- Generate a natural spoken rendition rather than reading Markdown or list punctuation.
- Render TTS once and cache by date, voice ID, TTS model, and brief-content hash.
- Reuse cached audio on repeated playback.
- Provide play, pause, replay, loading, and error states.
- Do not start an Agents Platform conversation or tunnel.

---

## 13. Delivery phases

Each phase lands as reviewable changes with its own tests. Do not begin a later phase to make an earlier phase appear complete.

### Phase 0 — feasibility spike

- Confirm corporate tunnel permission.
- Prove named tunnel → isolated ingress → loopback Metis → streamed Custom LLM response.
- Prove `@elevenlabs/react` WebRTC works in both a normal browser and Metis WKWebView.
- Measure speech-finalization, retrieval, first-token, and first-audio latency.
- Exercise interruption and cleanup after browser crash or forced app exit.
- Record measured p50/p95 timings and agree on an acceptable latency budget before building the polished UI.

This spike may use fixed read-only responses. No customer writes.

### Phase 1 — read-only voice

- Isolated ingress and auth
- Server lease and tunnel lifecycle
- Dedicated restricted voice graph
- Read-only retrieval and dual rendition
- Voice surface and honest state machine
- Build and protected-intent refusal
- Composer handoff
- Post-call transcript ingestion as evidence only

### Phase 2 — customer appends

- `CUSTOMER_APPEND` policy permission
- Typed create-only voice service
- Account ambiguity handling
- Idempotency and atomic receipts
- Visual cards and UI-only undo
- Optional spoken-confirmation mode

### Phase 3 — quick independent value

- ElevenLabs dictation provider selection
- Spoken morning brief with cache

These may ship before Phase 1 if their contracts remain independent.

### Phase 4 — Meetings

- Durable ingestion jobs
- Scribe diarization and word timings
- Synchronized player and transcript
- Speaker naming
- Customer-link and action proposals
- Segment-scoped correction and forced alignment

---

## 14. Testing and acceptance

### Provider and settings

- ElevenLabs provider unit tests mirroring Cohere provider tests
- `stt_provider=cohere` regression proving existing behavior is unchanged
- Provider-aware media-container tests
- Settings contract, persistence, availability, and secret-redaction tests

### Ingress isolation and security

- Exact public route allowlist
- No forbidden runtime handles or imported Metis app/router
- Missing or incorrect bearer token → `401` before body processing
- Wrong public model alias → `422`; configured alias cannot alter internal routing
- Caller system messages cannot alter policy or prompt authority
- Unknown tools are rejected
- Rate, size, timeout, and concurrency limits
- Valid streaming chunks and terminal `[DONE]`
- Webhook HMAC, timestamp, replay, and idempotency tests
- Secret canary tests for logs and returned errors

### Voice graph and refusals

- Table-driven build/toolify/application phrases
- Each fixed refusal proves retrieval, model invocation, and graph entry never occurred
- Approval, delete, overwrite, filesystem, sandbox, and model-selection refusal tests
- Voice graph has no registry, project engine, sandbox, approval-resume, or arbitrary mutation handle
- Policy is recomputed after structured model output
- Written and spoken rendition contract tests
- Markdown-to-speech fallback tests for tables, code, links, citations, and Mermaid

### Sessions and UI

- Signed URL is minted server-side and the API key never reaches the browser
- Idle timeout, page hide, navigation, unmount, WebRTC failure, and lease expiry end billing and stop the connector
- Multiple tabs do not prematurely stop a connector needed by another active lease
- Elapsed time and every state transition render honestly
- Browser and WKWebView microphone/WebRTC coverage
- Build deferral pre-fills the composer verbatim

### Customer writes

- Ordinary statements and inferred facts never write
- Explicit supported imperatives may create exactly one record
- Optional spoken confirmation requires a fresh, unambiguous second turn
- Ambiguous account resolution never commits
- A Custom LLM retry returns the original receipt without duplicating a row
- Every record type carries complete voice provenance
- Person creation cannot update an existing person
- Customer row, receipt, event, and idempotency key are atomic
- Voice interfaces expose no delete, update, status-change, or general customer endpoint
- Undo is single-use, receipt-bound, UI-only, and fails after later edits
- Undo preserves its immutable audit receipt

### Meetings and brief

- Failed meeting stages retry without orphaned or duplicated blobs
- Word timings round-trip and remain monotonic
- Low-confidence links and actions remain proposals
- Forced alignment is segment-scoped and never receives diarized text
- Original transcript and edit history survive correction
- Morning brief cache keys change with content, voice, or model and prevent duplicate renders otherwise

Run the full backend and web suites before declaring a phase complete. Report exact commands, pass/fail counts, skipped tests, and pre-existing failures. Do not edit unrelated tests to make the suite green.

---

## 15. Operational documentation

Document:

- ElevenLabs agent configuration and public model alias
- Named Cloudflare Tunnel creation and hostname
- Cloudflare Access service-token policy and webhook exception
- secret generation and rotation
- how the Mac app starts and stops the ingress and connector
- idle lease and rate-limit defaults
- data sent to ElevenLabs and to the selected reasoning provider
- how to disable voice immediately
- how to inspect voice receipts and post-call jobs
- failure recovery when `cloudflared`, ElevenLabs, Ollama, or the browser disconnects

Do not automate creation of external Cloudflare or ElevenLabs resources without explicit owner authorization. Repository code may validate and consume their configuration.

---

## 16. Non-goals

- No re-theme of Metis
- No separate voice page
- No replacement of existing dictation
- No ElevenLabs Knowledge Base or domain tool registry
- No direct ElevenLabs-to-Ollama path
- No audio-native reasoning model
- No phone-call route
- No multilingual mode
- No automatic meeting actions, facts, memories, or uncertain customer links
- No widening of voice permissions while implementing adjacent work

If a future phone route is added, visual receipts are insufficient on their own. Spoken confirmation becomes mandatory and requires a separate threat model and brief.

---

## 17. Reporting back

For every phase, report:

- what landed
- what remains
- which owner decisions were confirmed
- any assumptions made
- measured voice latency where applicable
- exact test results, including failures and skips
- external configuration still required
- any contract or test changed and why

Do not describe a phase as complete while a required acceptance test, lifecycle path, or security boundary remains unimplemented.
