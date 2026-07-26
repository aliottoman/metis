# Metis web

The local Next.js client for Metis. It is intentionally dependency-light and talks directly to the loopback FastAPI service.

## Run locally

From the repository root:

```bash
pnpm install --frozen-lockfile
pnpm dev:web
```

The web app listens on `http://127.0.0.1:3000`. Set `NEXT_PUBLIC_API_URL` when the API is not at its default `http://127.0.0.1:8000` location.

## API behavior used by the client

- All application endpoints are below `/api/v1`; health falls back from `/api/v1/health` to `/health` for compatibility.
- A posted message returns a `run_id`; run updates are consumed from replayable SSE using `?after=<last-sequence>`.
- SSE delivery is treated as at-least-once. Events are deduplicated by event ID and sequence before rendering.
- Runs paused at an approval checkpoint are discovered through `/runs?status=awaiting_approval`; recovery links pin both the conversation and run in the URL before replaying SSE.
- Artifact previews are fetched as blobs. SVG, JSON, and source previews are placed in an empty-sandbox iframe; downloads retain the API's attachment headers.
- Asset launches use a full-size in-Metis iframe player. The client polls only the saved catalog for live process state; new project discovery still requires the explicit **Scan for updates** action.
- Conversation metadata and messages are separate requests. Recent metadata is cached in local storage only as a sidebar fallback when the API is offline.
- The chat picker and drag-and-drop surface accept PNG, JPEG, WebP, and GIF images alongside supported documents and source files.
- Tool and memory changes always use explicit decision endpoints. The UI never treats a generated proposal as active by itself.
- Activating a previously approved tool version requires a reason, a unique idempotency key, and an explicit confirmation dialog. Tool correction proposals remain read-only evidence in the Workshop.

## Checks

```bash
pnpm --dir apps/web typecheck
pnpm --dir apps/web test
pnpm --dir apps/web build
```
