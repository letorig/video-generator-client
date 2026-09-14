"""Main SDK client."""

from __future__ import annotations

from typing import Callable, Optional

import httpx

from .config import Config
from .models import TaskRef, TaskStatus, VideoRequest
from .providers.registry import get_provider_class
from .utils.polling import poll_until_done
from .utils.video import download_video


class Task:
    """Handle for a remote generation task."""

    def __init__(self, client: "VideoClient", ref: TaskRef):
        self._client = client
        self.ref = ref
        self.task_id = ref.task_id
        self.provider = ref.provider

    async def status(self) -> TaskStatus:
        return await self._client.status(self.provider, self.task_id)

    async def wait(
        self,
        interval: Optional[float] = None,
        timeout: Optional[float] = None,
        on_update: Optional[Callable[[TaskStatus], None]] = None,
    ) -> TaskStatus:
        return await self._client.wait(
            self.provider,
            self.task_id,
            interval=interval,
            timeout=timeout,
            on_update=on_update,
        )

    async def cancel(self) -> bool:
        return await self._client.cancel(self.provider, self.task_id)

    async def download(self, dest: str) -> str:
        result = await self.wait()
        if not result.video_url:
            raise RuntimeError("Task finished but returned no video URL")
        return await download_video(result.video_url, dest)


class VideoClient:
    """Single async entry point for all registered providers."""

    def __init__(
        self,
        config: Optional[Config] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.config = config or Config()
        self._http = http_client
        self._owns_http = http_client is None
        self._providers = {}

    def _get_provider(self, name: str):
        key = name.lower()
        if key not in self._providers:
            cls = get_provider_class(key)
            self._providers[key] = cls(self.config, client=self._http)
        return self._providers[key]

    async def generate(self, provider: str, **kwargs) -> Task:
        request = VideoRequest(**kwargs)
        ref = await self._get_provider(provider).submit(request)
        return Task(self, ref)

    async def status(self, provider: str, task_id: str) -> TaskStatus:
        return await self._get_provider(provider).get_status(task_id)

    async def wait(
        self,
        provider: str,
        task_id: str,
        interval: Optional[float] = None,
        timeout: Optional[float] = None,
        on_update: Optional[Callable[[TaskStatus], None]] = None,
    ) -> TaskStatus:
        return await poll_until_done(
            self._get_provider(provider).get_status,
            task_id,
            interval=self.config.poll_interval if interval is None else interval,
            timeout=self.config.poll_timeout if timeout is None else timeout,
            on_update=on_update,
        )

    async def cancel(self, provider: str, task_id: str) -> bool:
        return await self._get_provider(provider).cancel(task_id)

    async def generate_and_wait(self, provider: str, **kwargs) -> TaskStatus:
        task = await self.generate(provider, **kwargs)
        return await task.wait()

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> "VideoClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
