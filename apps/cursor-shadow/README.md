# Metis Cursor shadow benchmark

This package is an evaluation-only adapter for comparing Cursor's local SDK
with the production ClineCore path. It is deliberately not a selectable Metis
coding engine.

`--describe` is free and makes no model request. A live run requires both
`--live` and `CURSOR_API_KEY`, plus a `.metis-cursor-shadow` file containing
`metis-cursor-shadow-v1` in its disposable workspace. It enables
the Cursor sandbox, loads no ambient settings or MCP servers, disallows web and
subagents, sends one benchmark prompt, enforces a timeout and observed token
ceiling, and offers no approval. Its final diff must still pass Metis's
independent overlay and acceptance verification before it is comparable.
