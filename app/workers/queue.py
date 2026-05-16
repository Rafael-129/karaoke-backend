from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

# Minimal async queue for future background processing.


@dataclass
class QueueItem:
    job_id: str
    handler: Callable[[], Awaitable[None]]


class JobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    async def enqueue(self, item: QueueItem) -> None:
        await self._queue.put(item)

    async def run_once(self) -> None:
        item = await self._queue.get()
        try:
            await item.handler()
        finally:
            self._queue.task_done()
