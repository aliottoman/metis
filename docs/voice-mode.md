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

### 1. The ElevenLabs agent

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

Set `WAQIL_ELEVENLABS_AGENT_ID` to the agent's id.

### 2. The named tunnel

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

### 3. Cloudflare Access

Protect the Custom LLM route with a service-token policy:

- Application path: `/v1/chat/completions`
- Policy: Service Auth, the token you issue to ElevenLabs

**Document the exception:** the webhook path `/v1/elevenlabs/post-call` must
be excluded from Access. ElevenLabs signs it with HMAC rather than presenting
a service token, and the ingress verifies that signature over the raw body
before parsing. Add an egress-IP allowlist there too where available.

### 4. Secrets

`WAQIL_VOICE_SHARED_SECRET` is the bearer the ingress checks. Leave it empty
and Metis mints one into `.data/voice-secret` (mode 0600) on first use — the
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
  ├─ voice ingress :8788   started on the first session, stopped when the last one ends
  └─ cloudflared           same lifecycle
```

Both are child processes of the API and both stop when no live lease remains.
Two windows can each hold a session; closing one does not take the connector
out from under the other.

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

## What leaves the machine

**To ElevenLabs:** your microphone audio, and the short spoken rendition of
each answer. Not the written answer, not its citations, not any record id —
the tunnel carries speech, and the browser gets everything else over loopback.

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

The voice graph is constructed with four read-only services and nothing else.
It holds no registry, sandbox, project engine, coding session, approval path
or customer mutation client — not disabled, never passed in.

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

Written answers land in the voice session's conversation, so a spoken
conversation can be read back in the chat window afterwards.

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
