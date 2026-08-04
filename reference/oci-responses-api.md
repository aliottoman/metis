# OCI Generative AI — the OpenAI-compatible Responses API

Verified on 2026-08-04 against `openai` 2.46.0 and `oci-genai-auth` 1.0.0 as
installed in this repo's `.venv`, against the working client in
`apps/api/src/waqil_api/model_provider.py`, and against live requests to the
Chicago endpoint (the project-header requirement and the vision input shape
were both exercised for real). Signatures below were read from the installed
SDK, not recalled. Re-verify after upgrading either package.

## Building the client — the mistake everyone makes

`AsyncOpenAI` **has no `auth` parameter.** Its accepted keywords are:

```
api_key, admin_api_key, workload_identity, organization, project,
webhook_secret, provider, base_url, websocket_base_url, timeout,
max_retries, default_headers, default_query, http_client
```

`OciUserPrincipalAuth` is an **httpx auth flow**, not an OpenAI credential. It
signs the outgoing request, so it belongs on the httpx client that carries it:

```python
import httpx
from oci_genai_auth import OciUserPrincipalAuth
from openai import AsyncOpenAI

http_client = httpx.AsyncClient(
    auth=OciUserPrincipalAuth(profile_name=OCI_PROFILE),
    timeout=TIMEOUT_SECONDS,
)
client = AsyncOpenAI(
    api_key="not-used",          # required by the constructor, unused by OCI
    base_url=OCI_RESPONSES_BASE_URL,
    project=OCI_RESPONSES_PROJECT_ID,   # REQUIRED for Grok — see below
    http_client=http_client,
    max_retries=MAX_RETRIES,
    timeout=TIMEOUT_SECONDS,
)
```

### The Responses project — required for Grok, and it is not the compartment

Non-OpenAI models on this surface (every `xai.*` and `meta.*` id) reject any
request that does not carry a Responses project:

```
HTTP 400: Non-OpenAI models require 'OpenAI-Project' or 'opc-conversation-store-id' header
```

`project=` on the constructor is the fix — the SDK sends it as the
`OpenAI-Project` header on every request. The value reaches the app as
`OCI_RESPONSES_PROJECT_ID`. It is **not** the compartment OCID: the compartment
scopes classic GenAI calls (embeddings, dedicated clusters) and is never sent
on a Responses request; the Responses project is what scopes requests here. An
app holding a compartment id but no project id cannot call Grok on this
surface, no matter how correct its auth is.

Four ways this goes wrong, all seen in real builds:

| Wrong | What happens |
|---|---|
| `AsyncOpenAI(base_url=..., auth=OciUserPrincipalAuth(...))` | `TypeError: unexpected keyword argument 'auth'` — the constructor has no such parameter |
| `AsyncOpenAI(api_key="", default_headers={"ocid-compartment-id": ...})` | Requests are unsigned. OCI rejects them. The header is not a real auth mechanism |
| `default_headers={"x": None}` | httpx raises on a `None` header value — a required env var that was never set fails here rather than at startup |
| Omitting `project=` with a Grok/Llama model | HTTP 400 `Non-OpenAI models require 'OpenAI-Project'…` on every request — pass `project=OCI_RESPONSES_PROJECT_ID` |

`api_key` must be a non-empty string because the constructor validates it; the
value is never used once an httpx auth flow is signing the request.

### Package and import names

- Distribution: `oci-genai-auth` (hyphens) — this is what goes in `requirements.txt`
- Import: `oci_genai_auth` (underscores) — `from oci_genai_auth import OciUserPrincipalAuth`

Exported auth flows: `OciUserPrincipalAuth`, `OciSessionAuth`,
`OciInstancePrincipalAuth`, `OciResourcePrincipalAuth`, `HttpxOciAuth`.

`OciUserPrincipalAuth(config_file=..., profile_name="DEFAULT", ...)` reads the
standard `~/.oci/config`. It re-signs on an interval, so a long-lived client is
fine — do not rebuild it per request.

## Responses, not Chat Completions

Background mode and resumable streaming exist **only on the Responses API**.
`client.chat.completions.create(...)` has no `background` parameter, no response
id to poll, and no sequence numbers. Passing `extra_body={"background": True}`
to Chat Completions does not enable anything; it sends an ignored field.

If the requirement mentions background jobs, resuming a stream, or cancelling
work in flight, the answer is `client.responses`.

## The three calls

Read from `openai.resources.responses.AsyncResponses` as installed.

```python
# 1. Synchronous — one signed request in, the completed Response out.
response = await client.responses.create(model=OCI_MODEL, input=prompt)
text = response.output_text

# 2. Background + stream — events ride the ORIGINAL signed connection.
stream = await client.responses.create(
    model=OCI_MODEL, input=prompt, background=True, stream=True
)

# 3. Cancel a streamed background job that has not finished.
await client.responses.cancel(response_id)
```

## Background retrieval and signed auth do not mix

`background=True` without `stream=True` looks right — create returns a queued
`Response` with an id to poll — and then every such job ends `failed` with:

```
No or an invalid authentication header was provided on the request.
```

OCI request-signing covers exactly one HTTP request. A background job is
re-executed later by a service worker, and that worker has no API key to
present, so under `OciUserPrincipalAuth` the deferred execution can never
authenticate. Verified live against the Chicago endpoint (2026-08-04): the
same vision request failed backgrounded and completed synchronously.

The two shapes that work under signed auth:

- **Synchronous create** — the pattern for request/response work such as
  document extraction. Application-level deferral belongs to the app (FastAPI
  `BackgroundTasks` wrapping the call), never to the provider call.
- **`background=True, stream=True`, consumed on the original connection** —
  for long jobs with live output and resume (next section). The signed
  request stays open; nothing is re-executed.

Two more mistakes seen in real builds:

- Treating the stream object as a `Response` — `create(..., stream=True)`
  returns an event iterator; `.id` and `.output_text` live on responses, not
  on streams.
- Reading fields with `.get(...)` — responses are typed SDK objects, not dicts.
  The assembled text of a completed response is `response.output_text`.

Accepted parameters, as verified:

- `create(...)`: `model`, `input`, `background`, `stream`, `store`,
  `include`, `previous_response_id`, …
- `retrieve(...)`: `response_id`, `stream`, `starting_after`, `include`
- `cancel(...)`: `response_id`

## Resumable streaming — `starting_after`

This is the whole mechanism, and it is one parameter.

Every streamed event carries a `sequence_number`. Record the last one you
successfully handed to the client. To resume after a dropped connection, call
`retrieve` on the same response id with `stream=True` and `starting_after` set
to that number; the server replays from the next event onward.

```python
# Live stream — remember the last sequence number as you go.
stream = await client.responses.create(
    model=OCI_MODEL, input=prompt, background=True, stream=True
)
async for event in stream:
    last_sequence_number = event.sequence_number
    if event.type == "response.output_text.delta":
        await send(event.delta)

# Reconnect — no tokens lost, no work repeated.
stream = await client.responses.retrieve(
    response_id,
    stream=True,
    starting_after=last_sequence_number,
)
async for event in stream:
    last_sequence_number = event.sequence_number
    ...
```

`sequence_number` is present on every event type, including lifecycle events
(`ResponseQueuedEvent`, `ResponseCompletedEvent`) and not only text deltas — so
resume works from any point, not just mid-sentence.

Do **not** invent your own counter by counting deltas. A locally incremented
counter is not the server's sequence number, and `starting_after` will resume
from the wrong place.

## Event types

Text arrives as `response.output_text.delta` with the chunk in `.delta`:

```python
if event.type == "response.output_text.delta":
    buffer.append(event.delta)
```

Lifecycle events include `ResponseQueuedEvent`, `ResponseCompletedEvent`,
`ResponseFailedEvent`, `ResponseIncompleteEvent`. For a non-streamed response,
`response.output_text` is the assembled text.

## Status values

`Response.status` is a literal, verified as:

```
"queued" | "in_progress" | "completed" | "failed" | "cancelled" | "incomplete"
```

Use these exact strings for job state. Note `"cancelled"` (two l's) and that
`"failed"` and `"incomplete"` are distinct terminal states — a UI that only
models `completed`/`cancelled` will silently mislabel both.

A background job is finished when the status is anything other than `queued` or
`in_progress`.

## Vision input — images go inside a user message

Multimodal input is a message list. One user message carries an `input_text`
part and one `input_image` part per image; `image_url` is a plain string, and
a local file travels as a data URL with its **real** MIME type:

```python
data = path.read_bytes()
mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
response = await client.responses.create(
    model=OCI_MODEL,
    background=True,
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": extraction_prompt},
            {"type": "input_image",
             "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}"},
        ],
    }],
)
```

Wrong shapes seen in real builds: `{"type": "text"}` and
`{"type": "image_url", "image_url": {"url": ...}}` are **Chat Completions**
part types — the Responses API does not accept them. And labelling every
upload `image/jpeg` corrupts PNG decoding; sniff the bytes.

## Configuration

Every value from the environment, no secrets in code:

| Setting | Default |
|---|---|
| base URL | `https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1` |
| model | `xai.grok-4.3` (on-demand; no dedicated AI cluster required) |
| profile | `DEFAULT` |
| Responses project id | **required for Grok, no default** — `OCI_RESPONSES_PROJECT_ID` |
| compartment OCID | used by classic GenAI surfaces (embeddings); **never sent** on Responses calls |

Read configuration lazily. Importing the app and booting its local routes must
succeed with **no** OCI environment at all — construct the client on first use,
and when a required setting is missing, fail the *feature* (a clear 503/500
from the extraction endpoint naming the variable), never the import. An app
that raises at import time cannot even serve its health route.

Two base-URL shapes exist in this estate and are not interchangeable:

- `…/openai/v1` — the OpenAI-compatible surface these clients use
- `…/20231130/actions/v1` — an older path seen with plain API-key clients

## Not verified here

The following are OCI-side behaviours this machine cannot confirm; treat them
as requiring a check against the tenancy rather than as settled:

- which models support `background=true` on the Chicago endpoint
- whether `store` must be enabled for a background response to be retrievable
- guardrail configuration exposed by the service itself, as opposed to a
  guardrail step written in application code

The client-side contract above — parameter names, event fields, status
literals, auth construction — is verified.
