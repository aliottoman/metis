from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from .contracts import RunEventV1, RunStatus
from .database import Database


TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}


class EventBus:
    """Durable event outbox with in-process wakeups for SSE consumers."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._conditions: defaultdict[str, asyncio.Condition] = defaultdict(asyncio.Condition)

    async def emit(
        self,
        run_id: str,
        thread_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> RunEventV1:
        event = await self.database.append_event(
            run_id, thread_id, event_type, payload, checkpoint_id
        )
        condition = self._conditions[run_id]
        async with condition:
            condition.notify_all()
        return event

    async def stream(
        self,
        run_id: str,
        *,
        after: int = 0,
        heartbeat_seconds: float = 15,
    ) -> AsyncIterator[RunEventV1 | None]:
        cursor = after
        condition = self._conditions[run_id]
        while True:
            events = await self.database.list_events(run_id, cursor)
            for event in events:
                cursor = event.sequence
                yield event
            run = await self.database.get_run(run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                return
            try:
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=heartbeat_seconds)
            except TimeoutError:
                yield None

