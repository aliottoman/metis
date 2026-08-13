"""The voice ingress: a protocol adapter, and deliberately nothing else.

This is the only Metis process reachable from outside the machine, and it is
its own distribution package rather than a module of `waqil_api` so that the
separation is structural instead of a promise. It imports nothing from the
main application — no ASGI app, no router, no database, no blob store, no
corpus, no customer service, no model broker — and it does not construct
`waqil_api.config.Settings` either, which is the subtler half of the same
rule: that object holds every cloud credential the machine has, and a process
on the far end of a tunnel has no business holding any of them.

What it does hold is six values read from the environment, one HTTP client
pointed at loopback, and a semaphore. Everything it is asked to do, it asks
port 8000 to decide.
"""
