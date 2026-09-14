"""Generic async task polling."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional

from ..exceptions import TaskFailed, TaskTimeout
from ..models import TaskStatus


async def poll_until_done(
    get_status: Callable[[str], Awaitable[TaskStatus]],
    task_id: str,
    interval: float = 5.0,
    timeout: float = 600.0,
    on_update: Optional[Callable[[TaskStatus], None]] = None,
) -> TaskStatus:
    start = time.monotonic()
    last_state = None

    while True:
        status = await get_status(task_id)

        if on_update and status.state != last_state:
            on_update(status)
            last_state = status.state

        if status.state == "succeeded":
            return status

        if status.state in ("failed", "cancelled"):
            raise TaskFailed(
                f"Task {task_id} ended with state={status.state}: {status.error}"
            )

        if time.monotonic() - start >= timeout:
            raise TaskTimeout(f"Task {task_id} timed out after {timeout}s")

        await asyncio.sleep(max(0.0, interval))
