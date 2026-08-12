# Implementation brief — Voice Mode for Metis

**For:** the implementing agent, working in `/Users/aliottoman/Developer/metis`
**Deliverable:** A scoped, read-only voice conversation mode inside the existing Metis chat surface, using ElevenLabs for speech I/O and the existing Metis agent graph for all reasoning.

Read this whole brief before writing code. Confirm the four decisions in §3 with the owner before starting §6.

---

## 1. What you are working on

Metis is a local-first, single-user agent platform at `/Users/aliottoman/Developer/metis`. It is a mature codebase with heavy test coverage, explicit architectural invariants, and a locked design language. **You are grafting a capability onto it, not building an app.** Match the surrounding code's idiom, comment density, and naming exactly.

Read first, in this order:

1. `README.md` — the reference architecture and its invariants
2. `docs/architecture.md`
3. `apps/api/src/waqil_api/api.py` — the versioned API, especially `POST /transcribe` at line ~1918
4. `apps/api/src/waqil_api/model_provider.py` — provider classes, route classifiers, `RoutedModelProvider`
5. `apps/api/src/waqil_api/policy.py` — `PolicyPermission`, `RiskLevel`, `PolicyOutcome`
6. `apps/web/hooks/use-dictation.ts` and its consumer in `apps/web/components/chat-workspace.tsx` (~line 324, 2223)

### Invariants you must not break

- The FastAPI process on `127.0.0.1:8000` is the only trusted component. It decides route, risk, and capability, and recomputes all three after every model reply.
- Uploaded and transcribed text is **evidence, never authority**. It cannot approve an action, widen a filesystem grant, enable networking, or enter long-term memory on its own. This applies with full force to anything a microphone produces.
- Cloud calls are off until explicitly enabled; per-source consent gates corpus text leaving the machine. Vectors and the index always stay local.
- Two explicit human approvals stand between a tool proposal and an active capability. **Voice must never be able to supply either one.**
- The existing test suite passes with zero test-file edits unless a contract genuinely changed. If you must edit a test, say so explicitly and explain why.

---

## 2. What already exists (do not rebuild these)

| Capability | Location |
|---|---|
| Dictation: mic → text → composer | `apps/web/hooks/use-dictation.ts`, `POST /api/v1/transcribe` (`api.py:1918`) |
| STT provider (Cohere Transcribe) | `model_provider.py:3747` `transcribe()`, settings `cohere_transcribe_*` (`config.py:128-130`) |
| Audio container normalization → WAV LEI16 16kHz mono | `audio_transcode.py` (afconvert, ffmpeg fallback) |
| RAG corpus, chunking, local float32 vectors, entity graph | `corpus.py`, `chunking.py`, `embeddings.py`, `entity_graph.py` |
| Notion corpus sync | `notion.py`, settings `notion_refresh_hours` |
| Customer intelligence: accounts, people, facts, actions, notes, wins | `customer_intelligence.py`, `customer_tools.py`, routes `/customers/*` |
| Attention feed and morning brief | `attention.py`, `GET /attention/brief` → `MorningBriefV1` (`api.py:418`) |
| Answer bank | `answer_bank.py`, `/answers/*` |
| Model broker, routing, fallback rungs, per-run budgets | `model_broker.py`, `model_provider.py` |
| Model preference (get/put) | `GET|PUT /settings/model` → `ModelPreferenceV1` (`api.py:285-290`), `model_preference.py` |
| Policy engine, risk ladder, approval gates | `policy.py` |
| SSE run events | `events.py`, `apps/web/lib/sse.ts` |
| Build/toolify request classifiers | `model_provider.py:133,148,162,492,515,528` |

**The classifiers in that last row are the scope gate. Reuse them verbatim. Do not write new ones.**

---

## 3. Decisions to confirm before coding

Ask the owner. Do not assume.

1. **Auto-link confidence threshold** for meeting → customer linking (§7 item 10). Propose a default and get it confirmed.
2. **Stock voice ID.** The owner will supply one; stub the setting and proceed without blocking.
3. **Corporate-device check.** If this Mac is Oracle-managed, the owner must confirm an outbound tunnel exposing a local service is permitted before §7 item 4 is built. Everything else can proceed regardless.

**Already decided — do not re-ask:**

- **Transport: ElevenLabs Agents Platform with Custom LLM, over a tunnel to an isolated shim process (§4.1–4.3).** The owner weighed this against the browser-orchestrated alternative (§4.4) and chose it deliberately: turn-taking, interruption handling, and latency are judged essential, and hand-building them is not viable. **Do not revisit this or propose §4.4 as a simplification.**
- **Language: English only.** Do not configure `language_detection` or multilingual models. Use an English TTS voice.
- **Voice identity: stock**, supplied as a setting (`WAQIL_ELEVENLABS_VOICE_ID`). No Voice Design in this build.
- **Voice model: `deepseek-v4-flash:cloud`** as the starting default, exposed as a setting so it can be dropped to something cheaper after listening. Metis does the retrieval; the model only phrases 2–4 grounded sentences, so capability is not the constraint — time-to-first-token is.
- **Voice may write, narrowly.** Customer records and notes only — create and append, never delete, never code. See §5 and §7 item 6.
- **No spoken echo before writes.** Confirmation is visual (§5). ElevenLabs quota is a real constraint; spending TTS characters on read-backs is the wrong place to spend them.
- **Meetings tab is in scope.** See §7 items 8–11.
- **Morning brief is a Today-tab button, not a voice-mode capability.** See §7 item 12.

---

## 4. Architecture

> **Decided: §4.1–4.3, the Agents Platform path.** Read all of §4 anyway — §4.3 is the part that keeps this safe, and it is easy to build wrong.
>
> The owner considered and rejected the browser-orchestrated alternative (§4.4, retained below for context only). Turn-taking, interruption handling, and latency are treated as essential to whether this is worth using at all, and those are precisely what the Agents Platform supplies.
>
> A third shape — pointing ElevenLabs' Custom LLM directly at Ollama Cloud — **is rejected and must not be built.** It removes the tunnel by removing Metis: no corpus, no customers, no Notion, no citations, no policy gate, no build refusal. It is a generic chatbot with a good voice, and it is not this product.

### 4.1 The shape

```
Browser (Metis web, loopback)
  └─ @elevenlabs/react  ── WebRTC ──►  ElevenLabs Agents Platform
                                         │  Scribe ASR
                                         │  turn-taking + interruption
                                         │  Flash v2.5 TTS
                                         │
                                         └── Custom LLM ──► HTTPS tunnel
                                                              │
                                                              ▼
                                         POST /api/v1/voice/chat/completions
                                         (OpenAI-compatible shim, bearer secret)
                                                              │
                                                              ▼
                                              Metis restricted voice route
                                              retrieve → synthesize → publish
                                                              │
                                                              ▼
                                                     Ollama (cloud/local)
```

ElevenLabs supplies ears, mouth, and timing. Metis supplies every thought. **Do not use the ElevenLabs knowledge base, its RAG, or its tool registry** — Metis owns retrieval and capability. The agent is configured as thin as it can be made.

### 4.2 Why Custom LLM rather than a webhook tool

The owner's Ollama models must do the reasoning. A hosted ElevenLabs LLM calling a `ask_metis` webhook would split the brain and put a foreign model in the conversational loop. Custom LLM keeps Metis as the only reasoner.

This is viable because **voice mode never triggers a LangGraph approval interrupt.** No tool builds, no tool activation, no sandbox — so no `PolicyApprovalRequired`, and therefore no state a chat-completions stream cannot express. The narrow customer/note writes in §5 do *not* change this: they are ordinary domain writes to SQLite at `RiskLevel.R2`, confirmed conversationally within a turn, not graph interrupts. If scope ever widens past that line, revisit this decision rather than working around it.

### 4.3 Tunnel the shim, not Metis

**Do not expose port 8000.** Mount the voice shim as a **separate FastAPI process on its own port**, carrying that one route and nothing else, and point the tunnel at that process. Port 8000 remains loopback-only exactly as it is today, so a defect in any other Metis route stays unreachable from the internet — it is not mounted on the exposed process at all.

**The single way to build this wrong** is to mount the voice route on the existing app "for now" — sharing the ASGI app, including a Metis router, or importing the main FastAPI instance. Any of those silently exposes the entire API and destroys the whole protection. The shim is a **separate ASGI application** that:

- shares no router, no app instance, and no middleware stack with the Metis API;
- has **no database handle, no blob-store handle, and no model-broker handle** of its own;
- reaches Metis only by HTTP over loopback, exactly as any other local client would;
- carries one route and a health check, and nothing else.

Treat it as a proxy that happens to speak OpenAI's dialect. **Assert the isolation in a test** (§10) rather than trusting it to survive future edits.

The shim process talks to Metis over loopback like any other local client. Layer the following:

- Bearer secret (`WAQIL_VOICE_SHARED_SECRET`), checked before any work.
- A Cloudflare Access service-token policy, so unauthenticated requests are rejected at Cloudflare's edge and never reach the Mac.
- Rate limiting on the route.
- **Connector lifecycle tied to the session.** Metis starts `cloudflared` when voice mode opens and stops it when the session ends. Use a named tunnel so the hostname the ElevenLabs agent config points at stays stable; when the connector is down the hostname simply fails. The door exists only while a conversation is actually happening.

### 4.4 Browser-orchestrated — considered and not chosen

**Retained for context only. Do not build this.** It was rejected because it trades away turn-taking, interruption handling, and a tuned latency budget — the things that decide whether voice mode is usable at all. Recorded here so the trade-off is legible to whoever reads this later, and so the alternative is not rediscovered as if it were new.

The browser is already on the owner's Mac. It can reach `127.0.0.1:8000` **and** it can reach ElevenLabs. So it bridges the two, and nothing ever needs to reach inward.

```
Browser (Metis web, loopback)
  ├─ mic ──────────────► ElevenLabs streaming STT (WebSocket, direct)
  │                            │ transcript
  │                            ▼
  ├─ POST /api/v1/... ──► Metis on 127.0.0.1:8000  (SSE, exactly as chat works today)
  │                            │ restricted voice route → Ollama
  │                            ▼ answer text, sentence-chunked
  └─ ────────────────────► ElevenLabs streaming TTS (WebSocket, direct) ──► speaker
```

**Audio never passes through Metis.** The browser holds both ElevenLabs sockets directly, which is simpler and lower-latency than proxying frames through FastAPI. Metis sees text in and text out, over loopback, on the transport it already uses.

What this buys:

- **No tunnel.** Port 8000 would stay sealed — no `cloudflared`, no shared secret, no Cloudflare Access policy, no connector lifecycle, no exposed route to defend.
- **No OpenAI-compatible shim.** Metis would be called by its own client on its own contract, and §7 item 4 would not exist.
- **Lower ElevenLabs spend.** The Agents Platform bills by conversation minute, which includes silence, thinking time, and retrieval latency. Streaming STT and TTS bill by audio processed and characters synthesized. On a work assistant with pauses, that gap is significant and runs the wrong way for Agents Platform. Confirm current rates before committing, but the shape is not in doubt.
- **The API key stays server-side.** The browser must never hold `WAQIL_ELEVENLABS_API_KEY`. Mint short-lived ElevenLabs session tokens from a loopback Metis route and hand those to the client.

What it costs, stated honestly:

- **You build turn-taking.** Voice-activity detection and endpointing in the browser, using the signals streaming STT provides. Expect real tuning before a mid-sentence pause stops being read as end-of-turn.
- **You build barge-in.** Cut TTS playback the moment speech is detected, and discard the rest of the queued audio. Not conceptually hard; fiddly to make feel natural.
- **Latency is yours to own.** Nobody has tuned this budget for you. Measure time-to-first-audio and keep it visible during development, not as an afterthought.

Do not treat these as small. They are the specific things the Agents Platform is genuinely good at, and this path trades them away deliberately, for a sealed port and a smaller bill.

---

## 5. Scope

### Voice mode may read

- Customers: accounts, people, facts, open actions, wins, recent notes
- The Notion corpus and the general corpus, with citations spoken naturally ("three sources — the strongest is the Batelco SR")
- The answer bank: "what did we tell them about X"
- The attention feed, when asked about a specific item. **Not the morning brief** — that is a Today-tab button (§7 item 12), deliberately kept out of the conversational surface.
- Meeting transcripts produced by the Meetings tab
- Ordinary conversation, clarification, follow-ups

### Voice mode may write — customers and notes only

Permitted: **create and append** on customer notes, facts, actions, people, and wins, through the existing `customer_tools.py` paths and `/customers/*` routes.

Writes commit immediately — there is **no spoken read-back**. ElevenLabs quota is a real constraint and read-backs are an expensive way to buy safety. The safety property (a mis-transcribed fact must never land silently) is bought visually instead, at zero TTS cost, because the owner is looking at Metis while talking to it.

Every voice-originated write must, without exception:

1. **Surface a confirmation card on write.** The moment it commits, a card appears in the voice surface — *"Added note to Batelco: '…'"* — carrying an **undo** control. Visible, immediate, and silent.
2. **Undo in one action.** Not a multi-step reversal, not a delete flow. One control.
3. **Carry provenance.** Persist `source: "voice"`, the conversation id, and the transcript excerpt that produced it, so every voice-written record is auditable and traceable back to the words that caused it.
4. **Appear in the run timeline** like any other write.
5. **Refuse to guess an account.** If entity resolution is ambiguous, do not write. Ask once, in one short sentence, or surface an on-screen picker — implementer's choice, but writing to the wrong customer is the worst available failure and must be unreachable by accident.

**Build a setting for spoken confirmation, default off.** If transcription accuracy proves poor in practice, the owner turns it on; the read-back path must already exist rather than needing a rebuild. And note the condition: **if a phone-call route is ever added, there is no screen, and spoken confirmation becomes mandatory in that context.** Write the visual path so this substitution is possible.

### Explicitly out of scope for voice mode

- **Any delete or overwrite of an existing record.** Voice creates and appends; it never destroys. The `DELETE /customers/*` routes are unreachable from voice, full stop.
- Any project build, app build, or scaffold
- Any tool definition, tool authoring, or tool activation
- Any approval of anything
- Any filesystem write, sandbox, or Podman execution
- Anything that would raise `PolicyPermission` above `WRITE_RUN_ARTIFACTS` / `MODEL_BROKER` (`RiskLevel.R2`)

Note the tension with the "evidence, never authority" invariant (§1): a transcript is evidence, and a write is an act of authority. What reconciles them here is that the write stays **visible and instantly reversible** — the owner sees every one and can undo it without leaving the conversation. That is the load-bearing property. Do not weaken it for fluency.

---

## 6. The build refusal — three layers

A build request arriving by voice must fail **server-side**, not because a system prompt asked nicely.

**Layer 1 — capability set.** The voice route is constructed with a restricted capability profile: no tool registry handle, no sandbox, no project engine. Permissions ceiling `MODEL_BROKER` (`RiskLevel.R2`). A build cannot be attempted because the machinery is absent.

**Layer 2 — classifier gate.** Before the graph runs, the transcribed turn passes through the existing classifiers:

```python
from .model_provider import (
    describes_a_new_tool,
    is_explicit_build_request,
    is_explicit_toolify_request,
    is_new_application_request,
    is_project_build_instruction,
)
```

If any returns `True`, short-circuit and return the fixed refusal below. Never reach the graph.

**Layer 3 — agent prompt.** The ElevenLabs agent's system prompt states it cannot build. Softest layer, present only so the refusal arrives conversationally rather than as an abrupt canned line.

### The fixed refusal

Store as a single module constant. Exact copy:

> "I can't build from voice — that needs the chat window. I've dropped it in your composer, so it's ready when you switch over."

**And make that true.** When the gate fires, emit an SSE event (`voice.build_deferred`) carrying the verbatim transcribed request. The web client pre-fills the chat composer with it and surfaces a small card in the Metis glass idiom. The request is never lost, and the handoff costs one keystroke. This is the difference between a refusal that feels like a wall and one that feels like a colleague.

---

## 7. Work items, in dependency order

Each item should land as its own reviewable change with tests.

**1. ElevenLabs provider class.** In `model_provider.py`, alongside the Cohere provider. Implements `transcribe()` against Scribe with the same signature the existing call site uses (`api.py:1969`). Settings: `WAQIL_ELEVENLABS_API_KEY`, `WAQIL_STT_PROVIDER=cohere|elevenlabs`, `WAQIL_ELEVENLABS_STT_MODEL`, `WAQIL_ELEVENLABS_TTS_MODEL`, `WAQIL_ELEVENLABS_VOICE_ID`, `WAQIL_ELEVENLABS_TRANSCRIBE_MAX_BYTES`. Follow the `cohere_*` naming and `Field(...)` bounds idiom in `config.py`. Add empty keys to `.env.example` matching the `WAQIL_COHERE_API_KEY` pattern.

Note Scribe accepts more containers than Cohere. `audio_transcode.py`'s allowlist is Cohere-specific and documented as verbatim from Cohere's error message — make the allowlist provider-aware rather than widening it globally.

**2. STT provider selection.** `/transcribe` dispatches on `settings.stt_provider`. Existing dictation behaviour must be byte-identical when the setting is `cohere`. Add a provider radio to `settings-panel.tsx` in the existing `settingsSection` idiom (see the OCI tools section, ~line 190). Update the mic button `title` in `chat-workspace.tsx:2231` — it currently hardcodes "Dictate with Cohere Transcribe".

**3. The restricted voice route.** A voice-scoped path through the existing graph: `retrieve → synthesize → publish`. **Skip `ground_review`** — its revision pass doubles latency and voice answers are short. Cap output length hard (2–4 sentences unless asked to go long). Separate model preference from chat, defaulting per the decided list in §3.

**The rendition problem — solve this properly, it is the difference between usable and unbearable.** Metis answers are long, markdown-heavy, and citation-dense, with tables, fenced code, and mermaid. Reading that aloud verbatim is intolerable: bullet characters, backticks, and bracketed citations all get vocalized.

Preferred fix: the `synthesize` node emits a short **`spoken`** field alongside the written answer — a model-authored spoken lede that handles citations naturally ("three sources, the strongest is the Batelco SR") rather than reciting reference markers. Fall back to a deterministic markdown→speech normalizer (strip fences, flatten a table to "a table with six rows", expand or drop citation markers) when no `spoken` field is present.

The written answer still renders in full on screen. Voice gets the lede; the eye gets everything.

**4. The OpenAI-compatible shim.** `POST /voice/chat/completions`, on the **isolated shim process** described in §4.3 — never on port 8000, never on the main app. Accepts the OpenAI chat-completions request shape, streams SSE deltas back. Authenticated with a bearer secret (`WAQIL_VOICE_SHARED_SECRET`); reject unauthenticated calls with 401 and log the attempt. As the only off-loopback route in the entire system it must be conservative, thoroughly tested, and must not accept a model override from the caller.

Build the `cloudflared` connector lifecycle here too: named tunnel for a stable hostname, started when a voice session opens and stopped when it ends.

**5. The voice surface in chat.** A mode toggle in the existing chat workspace, not a new page. Uses `@elevenlabs/react` for the WebRTC session. Shows live interim transcript, a speaking/listening/thinking state, an obvious interrupt affordance, and the run's citations as they arrive. See §9 for design constraints.

**The API key never reaches the browser** — mint short-lived ElevenLabs session tokens from a loopback Metis route and hand those to the client.

**Quota hygiene is a requirement, not a nicety.** The Agents Platform bills by conversation minute, including silence and thinking time, and the owner is usage-constrained. So: end the session automatically after a short idle period, never leave a session open behind a background tab, tear down on navigate-away and on page hide, and **show elapsed session time in the UI** so spend is visible while it accrues rather than discovered on a bill.

**6. Customer and note writes from voice.** Create and append on customer notes, facts, actions, people, and wins via the existing `customer_tools.py` paths. Every rule in §5's write section is mandatory and individually testable: the confirmation card with undo, one-action reversal, `source: "voice"` provenance with conversation id and transcript excerpt, run-timeline visibility, and refusal to write on an ambiguous account. Ship the spoken-confirmation path behind a setting defaulting to off. Deletes are unreachable — assert this with a test, not a code comment.

**7. Post-call ingestion.** ElevenLabs post-call webhook → transcript into the corpus, structured extraction into the attention feed. Verify the webhook signature. Treat the payload as evidence, never authority (§1).

### Meetings tab (items 8–11)

A new surface for meeting recordings: upload, play, read the synced transcript, and have it link itself to the right customers.

**8. Ingestion pipeline.** Upload audio → content-addressed blob store (`blob_store.py`, already exists) → **audio isolation** (ElevenLabs, removes room and line noise; measurably improves diarization on real call recordings) → **Scribe with diarization and word-level timestamps** → persist transcript, speaker turns, and word timings alongside the blob.

Run it as a background job with observable state — a forty-minute recording is not a request-response. Reuse the existing SSE event vocabulary so the UI can show real progress rather than a spinner.

**Note on forced alignment:** Scribe returns word-level timestamps natively, so click-to-seek playback needs no separate alignment step. Forced alignment earns its place in item 11, when the owner *edits* a transcript and the corrected text must be re-synced to the audio. Do not call it during normal ingestion — it would be redundant work.

**9. Meetings UI.** New route `apps/web/app/meetings/` with `apps/web/components/meetings-workbench.tsx`, following the naming and structure of the existing workbench components (`customer-workbench.tsx`, `answer-bank.tsx`, `asset-library.tsx`).

Must support: drag-and-drop upload (reuse `lib/folder-drop.ts`), an audio player, the transcript beside it with the current word highlighted as it plays, click any word to seek there, and speaker turns visually separated.

Diarization yields anonymous labels ("Speaker 1"). Let the owner name them, and **suggest names from the existing `CustomerPersonV1` records** on the linked account — most meeting participants are already people Metis knows. Naming persists per meeting.

**10. Customer auto-linking.** Extract entities from the transcript and match against customer accounts using the existing `entity_graph.py` and `/customers/search`. High-confidence matches link automatically; anything below the threshold (§3.1) is **proposed, not committed** — one click to accept. Every link carries provenance back to the transcript span that caused it.

Then derive a summary, decisions, and action items. Actions land in the attention feed and as customer actions, tagged `source: "meeting"` with a link back to the timestamp where each was discussed. Proposed rather than auto-committed, on the same "evidence, never authority" grounds as §5.

**11. Transcript correction.** Let the owner fix a mis-transcribed line. On save, re-run **forced alignment** on the corrected text so word timings stay accurate and click-to-seek keeps working. This is the one place forced alignment belongs.

### Spoken morning brief (item 12)

**12. A button in the Today tab — not a voice-mode capability.** `apps/web/components/today-view.tsx` and `GET /attention/brief` (`api.py:418`, `MorningBriefV1`) both already exist. Press it, hear the brief.

This is deliberately the simplest surface that delivers the value: a button, a TTS render, and a player. It carries none of the conversational machinery, works whether or not the transport decision in §4 has landed, and can ship before any of items 3–5.

Two requirements:

- **Spoken rendition, not markdown read aloud.** The brief is structured data; narrate it as speech ("four things need you today — the loudest is…"), not as a list with punctuation read out. See §7 item 3's rendition note.
- **Cache per day.** Render once, key the audio by brief content hash, replay for free. The owner will press this repeatedly on the same morning and should be billed for it once.

---

## 8. ElevenLabs features worth configuring

Configure these; they cost little and earn their place:

- **Turn-taking model + interruption.** The reason this architecture exists. Tune endpointing so a mid-thought pause is not treated as end-of-turn.
- **Client tools.** Run in the browser, need no tunnel. Wire voice navigation: "open Batelco", "show me the attention feed" → client-side router push. Cheap and genuinely delightful.
- **System tools.** `end_call` and `skip_turn`. **Not** `language_detection` — the owner has decided English only (§3).
- **Dynamic variables / conversation initiation data.** Inject the currently open customer, today's date, and the attention-feed summary at session start. Pressing talk on the Batelco page should mean the agent already knows.
- **Evaluation criteria + data collection.** Post-call structured extraction from the transcript, feeding item 7.

Do **not** configure: the ElevenLabs knowledge base, its RAG, its tool registry, batch calling, music, or sound effects.

---

## 9. Design constraints

Metis owns the design language. Do not import ElevenLabs' visual style — import its **interaction patterns**, rendered in Metis's existing tokens: pale translucent navigation, cool foggy canvas, soft white glass surfaces, green for interaction and live posture, charcoal for primary actions, the blue/lavender/coral/ink quartet reserved for identity.

Specifically: the orb/level indicator, the interim transcript that settles as it finalizes, a visible and honest state machine (listening / thinking / speaking / interrupted), and an interrupt affordance that is obviously clickable. Reuse the existing `micButton` / `micRing` level-ring vocabulary from `chat-workspace.tsx` rather than inventing a second visual language for the microphone.

State must be honest about latency. If a model turn is slow, say so visually — never let silence read as a dropped call.

---

## 10. Testing

The repo has a strong test culture; match it.

- Provider unit tests mirroring `apps/api/tests/test_cohere_provider.py`
- Gate tests: a table of build-request phrasings, each asserting the fixed refusal and that the graph was never entered
- Auth tests on the shim: unauthenticated → 401, model override attempt → rejected
- **Shim isolation test (§4.3):** the shim app exposes only its one route plus a health check, and holds no database, blob-store, or model-broker handle. This is the test that stops a future edit from quietly widening what the tunnel reaches — write it so it fails loudly if someone mounts a Metis router on the shim.
- Session teardown: idle timeout fires, and navigating away or hiding the page ends the conversation rather than leaving it billing
- Contract tests for the new settings and events, mirroring `test_contracts.py`
- A regression test that `stt_provider=cohere` leaves existing dictation behaviour unchanged
- **Write-path tests:** no write commits without explicit spoken confirmation; every voice-written record carries `source: "voice"` and its transcript excerpt; delete routes are unreachable from the voice capability set
- **Meetings tests:** ingestion survives a recording that fails mid-pipeline without orphaning the blob; word timings round-trip; low-confidence customer links are proposed and not committed

Run the full suite before declaring done. Report the actual result, including failures.

---

## 11. Non-goals

- No re-theme of Metis
- **Voice mode** adds no new page — it lives inside the chat surface. The **Meetings tab** is the one new route, and it follows the existing workbench pattern rather than inventing a layout.
- No replacement of the existing dictation flow; it stays as-is with a swappable provider
- **No wiring ElevenLabs directly to Ollama Cloud.** It removes the tunnel by removing Metis (§4). Rejected.
- No ElevenLabs speech-to-speech (voice changer) — it does not understand language and has no role here
- No audio-native realtime model (GPT Realtime, Gemini Live) — those replace the Ollama brain, which defeats the premise
- No widening of voice scope "while you're in there"

---

## 12. Reporting back

When done, state plainly: what landed, what the test suite actually reported, which of §3's decisions were confirmed versus assumed, and anything in §7 you did not complete and why.