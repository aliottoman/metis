# Voice mode — operations

What is running, what leaves the machine, and how to turn it off. The design
and its reasoning are in `metis-voice-mode-brief.md`; this is the page you
want when something is not working at nine in the morning.

## What the three features are, and what each needs

| Feature | Needs | Cloud call |
|---|---|---|
| Dictation | `WAQIL_COHERE_API_KEY` **or** `WAQIL_ELEVENLABS_API_KEY` | The clip, to the selected transcriber |
| Spoken morning brief | `WAQIL_ELEVENLABS_API_KEY` | The brief's spoken text, to ElevenLabs |
| Voice conversation | The key, an agent, a named tunnel, and the go-ahead below | Audio both ways; the question and its evidence to the voice model |

The first two work on their own. Nothing in the third is required for them,
and turning the third off leaves both untouched.

## Turning voice off, immediately

```bash
WAQIL_VOICE_ENABLED=false
```

Restart Metis. Every open session ends, both child processes stop, and
`GET /api/v1/voice` reports why. Dictation and the brief keep working.

Without restarting: close the voice panel in every window. The lease sweep
stops the connector within about a minute. Killing the Metis process does the
same — shutdown ends every session before the loop goes away.

## One-time setup

### The three parts, briefly

- **ElevenLabs Agent:** the live audio layer. It listens, detects turns and
  interruptions, and speaks Metis's answer. It does not retrieve records or
  decide what may be written.
- **Custom LLM:** the Agent's connection to the Metis brain. ElevenLabs sends
  finalized words to this OpenAI-compatible endpoint; Metis retrieves,
  reasons, applies policy, and returns only the short text to speak.
- **Cloudflare Tunnel:** the temporary, outbound bridge that lets ElevenLabs
  reach the isolated Custom LLM endpoint. It carries finalized transcript and
  answer text, not microphone audio. It never exposes Metis port 8000 and runs
  while Voice is briefly pre-warmed or a session has a live browser lease.

Create the tunnel first, because its hostname is required when configuring the
ElevenLabs Agent. Then create the Agent and paste its id into Metis.

### 1. The named tunnel

Install `cloudflared`, authorize it once, create the named tunnel and route a
hostname as described below. Set `WAQIL_VOICE_TUNNEL_NAME` and
`WAQIL_VOICE_TUNNEL_HOSTNAME` before creating the Agent.

### 2. The ElevenLabs agent

Create it in the ElevenLabs console. Metis never creates external resources on
your behalf; it validates and consumes what you configure.

Configure the agent with:

- **Custom LLM** → `https://<your tunnel hostname>/v1/chat/completions`
- **Model name** → whatever `WAQIL_VOICE_PUBLIC_MODEL_ALIAS` is set to
  (default `metis-voice`). The ingress compares this string and refuses
  anything else with a 422. It never routes on it: which model reasons is the
  preference stored in Metis.
- **Authorization** → `Bearer <WAQIL_VOICE_SHARED_SECRET>`
- **Post-call webhook** → `https://<your tunnel hostname>/v1/elevenlabs/post-call`

Leave Knowledge Base, RAG, server-side tools, batch calling, music and sound
effects **off**. Metis is the only component that retrieves anything.

`GET /api/v1/voice` prints the Custom LLM URL and the alias back to you, so
you can copy them rather than reconstruct them.

Use ElevenLabs → **Agents** → **New agent**. Open the saved agent and copy its
Agent ID from its details menu or from the `agent_...` value in the page URL.
Set `WAQIL_ELEVENLABS_AGENT_ID` to that value. The agent must be published.

### 3. Cloudflare commands

```bash
cloudflared tunnel login
cloudflared tunnel create metis-voice
cloudflared tunnel route dns metis-voice voice.example.com
```

Then set `WAQIL_VOICE_TUNNEL_NAME` and `WAQIL_VOICE_TUNNEL_HOSTNAME`.

Metis starts and stops the connector itself, pointed at
`http://127.0.0.1:<WAQIL_VOICE_INGRESS_PORT>` — the isolated ingress, never
port 8000. There is no code path that can put 8000 on that command line.

**This Mac is Oracle-managed.** The brief's Appendix A.3 records the owner's
authorization for a connector in principle, and asks for one further explicit
go-ahead at the moment one is first started here. That go-ahead is:

```bash
WAQIL_VOICE_TUNNEL_AUTHORIZED=true
```

Until it is set, voice refuses to start and says exactly this.

### 4. Cloudflare Access

Protect the Custom LLM route with a service-token policy:

- Application path: `/v1/chat/completions`
- Policy: Service Auth, the token you issue to ElevenLabs

**Document the exception:** the webhook path `/v1/elevenlabs/post-call` must
be excluded from Access. ElevenLabs signs it with HMAC rather than presenting
a service token, and the ingress verifies that signature over the raw body
before parsing. Add an egress-IP allowlist there too where available.

### 5. Secrets

`WAQIL_VOICE_SHARED_SECRET` is the bearer the ingress checks. Leave it empty
and Metis mints one into `.data/voice-secret` (mode 0600) during startup, so it
already exists when you configure the ElevenLabs Custom LLM API key. The
alternative was a default, and a default here is an open door.

To rotate: set a new value (32+ characters), restart Metis, update the agent's
Authorization header. To rotate the minted one instead, delete
`.data/voice-secret` and restart.

`WAQIL_ELEVENLABS_WEBHOOK_SECRET` is ElevenLabs' own signing secret from the
webhook configuration. Without it **no webhook is accepted** — an unverified
payload is not evidence, it is input from the internet.

## What runs while a session is open

```
Metis API :8000            always running
  ├─ voice ingress :8788   pre-warmed on entry to Voice; kept for a live session
  └─ cloudflared           same lifecycle
```

Both are child processes of the API. Entering Voice prepares them and a
single-use ElevenLabs conversation token for at most two minutes so Start does
not have to wait on tunnel and token setup. Pre-warming does not open an audio
connection or start a billed voice conversation. If Start is not pressed,
both processes stop after the window expires. Once a call begins, they remain
while a live lease exists. Two windows can each hold a session; closing one
does not take the connector out from under the other.

The Mac app inherits this: `open` starts the API, ⌘Q stops it, and shutdown
ends every voice session before the loop closes.

## Leases

The browser renews a short lease (`WAQIL_VOICE_LEASE_SECONDS`, default 45s) at
a third of its length. The server sweeps expired leases every ten seconds.

This is the only teardown path that survives a browser crash. Everything else
— the stop button, unmount, navigation, `pagehide` — is a convenience that
makes the common case fast. A tab that dies mid-sentence cannot tell anyone,
so the lease it stopped renewing is what closes the tunnel.

`WAQIL_VOICE_SESSION_MAX_SECONDS` (default 30 minutes) is the hard ceiling, so
a forgotten open tab cannot bill all afternoon.

## Rate and size limits on the ingress

| Setting | Default | Why |
|---|---|---|
| `WAQIL_VOICE_MAX_BODY_BYTES` | 128 KB | A spoken turn is a sentence and a few bounded prior ones |
| `WAQIL_VOICE_MAX_CONCURRENCY` | 4 | One person is talking |
| `WAQIL_VOICE_RATE_PER_MINUTE` | 60 | Per provider conversation, sliding window |
| `WAQIL_VOICE_REQUEST_TIMEOUT_SECONDS` | 60 | The loopback call to :8000 |

## How a turn streams

The voice model writes plain speech, and Metis does not wait for it to
finish. Each sentence is split off as it arrives, normalized for the ear,
checked against the retrieved evidence, and only then handed to the ingress,
which relays it to ElevenLabs as one streaming chunk. The voice starts on the
first sentence while the model is still writing the rest.

The claim gate runs before a sentence leaves, not after: a sentence stating a
figure the records do not contain is withheld, and Metis says so at the end
("I left out a figure your records don't contain"). A listener cannot unhear
a number, so nothing is spoken and then corrected. The screen shows exactly
the words that were spoken, with the citations beside them.

## What leaves the machine

**To ElevenLabs:** your microphone audio, sent directly from the browser over
WebRTC, and the spoken sentences of each answer. Not its citations, not any
record id. The Cloudflare tunnel carries finalized transcript and spoken text
between the ElevenLabs Agent and Metis; the browser gets the answer with its
sources over loopback.

**To the voice model** (`WAQIL_VOICE_MODEL`, a hosted Ollama model by default):
the question, the bounded conversation history, and the evidence retrieved to
answer it. Corpus consent still applies per source; voice does not widen it.

**Never:** vectors, indexes, credentials, or anything from the tool catalog,
project workspaces, or attachments.

## What voice cannot do

Enforced before retrieval and before any model call, in `voice_intents.py`:
build, toolify, approve, delete, overwrite, change an action's status, read
files, activate a tool, run code, or select a model. Each gets a fixed spoken
refusal, and the build and approval refusals prefill your composer with the
verbatim utterance so the chat window can finish the job.

The voice graph is constructed with four read-only services plus one
create-only write service, and nothing else. It holds no registry, sandbox,
project engine, coding session or approval path — not disabled, never passed
in. The write service exposes exactly two methods, `commit` and `undo`; there
is no update, no status change and no delete reachable through it, and
`upsert_customer_person` — which would turn a duplicate contact into a silent
edit of the existing one — is deliberately not among them.

## What voice can add

Five record types, create-only: a note, a fact, an action, a person, or a win.

A write happens **only** when the utterance in front of Metis is an explicit
imperative to file one:

| Said | Result |
|---|---|
| "Add a note to Batelco that the workshop moved to Thursday" | writes |
| "Create an action for me to send the sizing by Friday" | writes |
| "The workshop moved to Thursday" | answers, writes nothing |
| "We should probably add a note about that" | answers, writes nothing |
| "Add a note and an action for Batelco" | asks which one |
| "Add a note about Bahrain" (two matching accounts) | asks which account |

Detection is deterministic and the model is never asked *whether* to write. It
is asked only to shape the text of a write the host already decided on, and
even then the host copies the transcript verbatim, parses any due date itself,
and bounds every field before committing.

Starting a voice session pre-authorizes this one narrow R2 capability
(`customer:append`). It is not either of Metis's two tool approvals, and no
transcript can widen it.

**Undo** is a browser action on the receipt card, never a spoken one — "undo
that" is refused, because deleting a customer record on a misheard word is
exactly what the refusal list exists for. The token is single-use, expires in
fifteen minutes, and is bound to one receipt; it cannot name a record type or
a record id at all. Undo also refuses a record someone has edited since.

The receipt survives an undo. "This was added and then taken back" is a
different fact from "this never happened".

**Spoken confirmation** (Settings → Speech) turns every write into two turns:
Metis reads the exact record back and waits for an unambiguous yes in the very
next turn. Anything else — a correction, a new subject, forty-five seconds of
silence — drops it. Off by default, because the receipt card and its Undo are
the standing safety mechanism and a read-back on every append taxes the
ordinary case to catch the rare one.

## Inspecting what happened

Post-call webhooks land in the `voice_post_calls` table, idempotent on the
provider's conversation id and event type:

```bash
sqlite3 .data/waqil.db \
  "SELECT created_at, provider_conversation_id, status FROM voice_post_calls ORDER BY created_at DESC LIMIT 20;"
```

`provider_analysis_json` is the provider's own analysis, stored under a name
that says whose opinion it is. Nothing in that table becomes a customer fact,
an action, a memory or an account link on its own.

The first thing said creates the conversation and titles it; written
answers land in it, so a spoken
conversation can be read back in the chat window afterwards.

Everything voice added, including what was later undone:

```bash
sqlite3 .data/waqil.db "SELECT created_at, record_type, account_name, undone_at, transcript_excerpt FROM voice_write_receipts ORDER BY created_at DESC LIMIT 20;"
```

## When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| "Voice isn't set up yet" with a named variable | Exactly that variable | Set it, restart |
| Agent connects, says nothing | Custom LLM URL or bearer wrong | Compare against `GET /api/v1/voice` |
| Agent returns an error immediately | Model name is not the alias | The ingress 422s anything else, on purpose |
| Webhook never arrives | Access is covering the webhook path | Exclude `/v1/elevenlabs/post-call` |
| Webhook 401s | Missing or wrong `WAQIL_ELEVENLABS_WEBHOOK_SECRET` | Copy it from the webhook configuration |
| Session ends after ~45s | Browser stopped renewing | Check the console; the lease is doing its job |
| `cloudflared` dies | Tunnel credentials or DNS | `cloudflared tunnel run <name>` by hand to see the error |
| Voice answers "I held that answer back" | The claim gate found a figure your records do not contain | Ask in the chat window, where the working is visible |
| Ollama unreachable | The voice model is a hosted Ollama model | Check `GET /api/v1/health` |

The ingress writes no access log. Every line would carry a conversation
identifier, and that process has no reason to keep a record of who spoke when.

## Meetings

A separate surface (`/meetings`) and the only new main route. Upload a
recording, get a diarized transcript with word timings, and a set of proposals
nobody has agreed to yet.

### The pipeline

```
uploaded → isolating → transcribing → analyzing → ready
```

Each stage is committed **before** the work it names begins, which is why the
stages exist as a column rather than as local variables: a resumed job needs
to know which step was interrupted, and a stage written only on success cannot
say. The blob is content-addressed and stored before any provider is called,
so a failure at any later stage is retryable without re-uploading an hour of
audio — `POST /meetings/{id}/retry` resumes from the failed stage, never from
the upload. After three failures it stops offering, because a recording that
fails three times is failing for a reason retrying will not fix.

Audio isolation is off by default (`WAQIL_MEETING_AUDIO_ISOLATION`). It costs a
second call over the whole file, it helps a noisy recording and is pure spend
on a clean one, and which is which is your call. When it is on and it fails,
the job transcribes the original instead — a noisy transcript beats no
transcript. The isolated track never replaces the original: the recording is
evidence, the isolated version is a derived artifact.

### What a meeting produces

Proposals, and only proposals. An action stays a proposal until you accept it,
and accepting it marks the proposal — it does not silently create a customer
action. A transcript is a machine's best guess at what a room said.

Customer linking uses the same deterministic name-match scale as a spoken
write (Appendix A.1): auto-link only at `>= 0.90` with the runner-up trailing
by `>= 0.15`, scored against the title and the transcript's opening — where a
meeting names who it is with — rather than the whole hour, so one passing
mention of a competitor cannot outweigh the actual host. Anything below that
is a one-click proposal. Every proposal carries the turn and the seconds that
produced it, so accepting one is a four-second check against the audio.

### Correcting a line

Double-click any line. The correction saves, and `original_text` keeps what the
provider actually heard so the two can be read side by side; every change is
recorded in `meeting_edits`.

Realignment is deliberately narrow. Forced alignment is a **single-speaker**
service, so the corrected line is realigned against its own speaker's seconds
of audio and never a diarized span — handing it a multi-speaker passage
produces timings that look right and are not. Spliced timings are checked for
monotonicity and discarded if they go backwards, because a player that jumps
is indistinguishable from a transcript that is wrong.

If realignment is impossible — no ffmpeg, a provider failure, non-monotonic
output — the correction still lands with its original timings. A correct line
with slightly stale timings beats a correct line that will not save.

Segment extraction needs `ffmpeg` (`brew install ffmpeg`); afconvert cannot
trim. Without it, corrections save and keep their original timings.
