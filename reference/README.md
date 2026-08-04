# Coding reference library

Ground truth for the APIs and shapes Metis builds against. Indexed as a corpus
source and retrieved into project build turns, so the model writing a file has
the real signature in front of it instead of a recollection of one.

This exists because of a specific failure. A build asked for the OCI Responses
API with background mode produced `client.chat.completions.create(...)`, an
`api_key=""` with a fabricated auth header, and a `501 Not Implemented` where
the resumable stream should have been. None of that was a reasoning error — the
model had never seen the API and had no way to look it up. Everything it needed
was installed in a virtualenv three directories away.

## What belongs here

Facts a model cannot derive from the project it is editing:

- external API surfaces — exact signatures, parameter names, auth construction
- structural patterns for the kinds of app built here repeatedly
- the difference between two things that look interchangeable and are not

## What does not

Anything already discoverable from the code being edited. Project-specific
conventions belong in that project's `.metis/METIS.md`. This library is for
things no amount of reading the local tree would reveal.

## The rule for writing entries

**Every claim is verified against something on this machine before it is
written down.** The installed package, the SDK's own signature, working code
that runs. A reference that is confidently wrong is worse than no reference:
the model believed its own invention last time, and it will believe this file
too. Where something could not be verified, the entry says so in place.

Each file states its verification basis at the top, with versions. When a
dependency is upgraded, re-verify rather than assuming the entry survived.

## Files

| File | Covers | Verified against |
|---|---|---|
| `oci-responses-api.md` | OCI Generative AI via the OpenAI-compatible Responses API: auth, background mode, resumable streaming, cancellation | `openai` 2.46.0, `oci-genai-auth` 1.0.0, `apps/api/src/waqil_api/model_provider.py` |
| `fastapi-static-app.md` | Single-runtime FastAPI app serving an API and a static frontend: layout, SSE, job state, entrypoints | `fastapi` / `starlette` / `uvicorn` as installed; this repo's own API |
