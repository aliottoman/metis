# Single-runtime FastAPI app with a static frontend

The shape used for demos here: one Python process serving both the JSON API and
a no-build frontend. No node toolchain, no second server, no proxy.

Verified on 2026-08-03 against `fastapi` / `starlette` / `uvicorn` as installed
in this repo's `.venv`, and against this repo's own API layer.

## Layout

```
app/
  __init__.py        # present, so `app.` imports resolve as a real package
  config.py          # every setting, read from the environment, validated here
  main.py            # FastAPI instance, routes, static mount
  <domain>.py        # one module per concern
  static/
    index.html
    style.css
    app.js
requirements.txt     # pinned
.env.example         # every variable named, no real values
README.md            # env vars, run command, what the demo proves
```

Run with `uvicorn app.main:app --reload`. If `uvicorn app.main:app` fails with
`ModuleNotFoundError: app`, it is being run from the wrong directory — the
project root must be the working directory.

## Serving static files

Mount the directory once rather than writing a passthrough route. A hand-rolled
`FileResponse` handler re-implements — usually incorrectly — content types,
range requests, and path traversal defence.

```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="…")

@app.get("/")
async def index() -> FileResponse:
    # Serve the shell explicitly so "/" is not a directory listing.
    return FileResponse(STATIC / "index.html")

app.mount("/static", StaticFiles(directory=STATIC), name="static")
```

Resolve paths from `__file__`, never from the process working directory —
`FileResponse("app/static/index.html")` only works when uvicorn happens to have
been started from the project root.

## Request bodies — the 422 that looks like a frontend bug

**A bare scalar parameter is a query parameter, not a body field.**

```python
@app.post("/api/jobs")
async def submit(prompt: str):          # ← reads ?prompt=… from the URL
    ...
```

A frontend posting `JSON.stringify({prompt})` against that endpoint gets `422
Unprocessable Entity` on every single request, and the error names `prompt` as
missing — which reads like the frontend forgot to send it.

Declare a model:

```python
from pydantic import BaseModel

class JobRequest(BaseModel):
    prompt: str

@app.post("/api/jobs", status_code=202)
async def submit(body: JobRequest) -> JobSubmitted:
    ...
```

Rule: if the frontend sends a JSON body, the endpoint takes a Pydantic model.

## Server-sent events

```python
from fastapi import Request
from fastapi.responses import StreamingResponse

@app.get("/api/jobs/{job_id}/stream")
async def stream(job_id: str, request: Request, after: int | None = None):
    async def events():
        async for item in source(job_id, after=after):
            if await request.is_disconnected():
                break
            yield f"id: {item.sequence}\nevent: {item.type}\ndata: {json.dumps(item.payload)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Details that matter:

- each frame ends with a **blank line** — `\n\n`. Without it the browser buffers
  and nothing appears
- an `id:` line lets `EventSource` send `Last-Event-ID` on auto-reconnect
- check `request.is_disconnected()` or the generator runs on after the client
  is gone
- take the resume point as a query parameter so a client can reconnect
  explicitly rather than only via `EventSource`'s own retry

Client side:

```js
const source = new EventSource(`/api/jobs/${id}/stream?after=${lastSeq}`);
source.addEventListener("delta", (e) => append(JSON.parse(e.data)));
source.onerror = () => source.close();   // EventSource retries by default
```

## In-memory state

Fine for a demo, with the constraints stated rather than discovered:

```python
from dataclasses import dataclass, field
from datetime import UTC, datetime

@dataclass
class Job:
    id: str
    prompt: str
    status: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

JOBS: dict[str, Job] = {}      # dict, not list — lookup by id is the common path
```

- **timezone-aware** timestamps (`datetime.now(UTC)`). A naive datetime
  serialises without an offset and every browser reads it as local time, so
  elapsed counters are wrong by the UTC offset
- a `dict` keyed by id, not a list scanned linearly
- state dies with the process — say so in the README instead of implying
  otherwise

## Background work

`asyncio.create_task` returns a task that is garbage-collected if nothing holds
it, and swallows exceptions if nothing ever awaits it. Keep a reference and
attach a done-callback:

```python
_tasks: set[asyncio.Task] = set()

def spawn(coro) -> None:
    # Hold a reference until completion, and surface failures instead of
    # letting a dead task look like a job that is still running.
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
```

Never mark work complete in a bare `finally:` — that reports success for a job
that raised. Set a terminal status on each path explicitly.

## Output that must not reach the user

When a check can withhold a response, the endpoint must not serialise the body
at all. Returning the text alongside a `"blocked"` verdict and expecting the
frontend to hide it is not withholding it:

```python
@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> JobView:
    job = JOBS[job_id]
    return JobView(
        id=job.id,
        status=job.status,
        verdict=job.verdict,
        reason=job.reason,
        # The body is present only when the check allowed it. A client cannot
        # reveal what was never sent.
        output=job.output if job.verdict != "block" else None,
    )
```

The same applies to the stream: if the check runs on the finished output, do
not stream deltas straight through unchecked and then decide afterwards.

## Frontend without a framework

- build DOM nodes with `textContent` for anything model-generated.
  `innerHTML` with model output is an injection sink, and the output is exactly
  the untrusted part
- one poll interval, cleared when the page hides
- state pills driven by the API's own status strings, so the UI cannot drift
  from the backend's vocabulary

## Requirements

Pin what is actually imported, nothing more:

```
fastapi==<version>
uvicorn==<version>
openai==<version>
```

Do not invent version numbers. Read them from the environment the code will run
in (`pip freeze`, or the installed `dist-info`); a plausible-looking pin that
does not exist fails at install time, after the review has already passed.
