"""Provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import httpx

from ..config import Config
from ..models import TaskRef, TaskStatus, VideoRequest


class BaseProvider(ABC):
    name = "base"

    def __init__(self, config: Config, client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
            self._owns_client = True
        return self._client

    @abstractmethod
    def is_configured(self) -> bool:
        """Return whether the provider has enough credentials."""

    @abstractmethod
    async def submit(self, request: VideoRequest) -> TaskRef:
        """Submit a generation task."""

    @abstractmethod
    async def get_status(self, task_id: str) -> TaskStatus:
        """Return normalized task status."""

    async def cancel(self, task_id: str) -> bool:
        return False

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
