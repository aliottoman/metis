"""Run the ingress as its own process.

Its own process, not a thread or a mounted sub-application, because the point
of the isolation is that a compromise of the internet-facing adapter does not
land inside the process that holds the database handle and every credential.
Metis starts and stops this on demand; nothing keeps it running when no voice
session is open.
"""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import IngressConfig, IngressConfigError

logger = logging.getLogger("waqil.voice.ingress")


def run() -> None:
    try:
        config = IngressConfig.from_environment()
    except IngressConfigError as error:
        # Named plainly and refused: an ingress that starts with a weak or
        # missing secret is worse than one that does not start at all.
        raise SystemExit(f"voice ingress cannot start: {error}") from error
    logger.info("voice ingress listening on %s:%s", config.host, config.port)
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
        # No access log: every line would carry a conversation identifier, and
        # this process has no reason to keep a record of who spoke when.
        access_log=False,
    )


if __name__ == "__main__":
    run()
