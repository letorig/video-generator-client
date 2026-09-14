"""Kling provider adapter."""

from __future__ import annotations

import time

import jwt

from ..exceptions import APIError, ProviderNotConfigured
from ..models import TaskRef, TaskStatus, VideoRequest
from .base import BaseProvider

KLING_MODELS = {
    "1.6": "kling-v1-6",
    "2.0": "kling-v2",
    "2.1": "kling-v2-1",
}


class KlingProvider(BaseProvider):
    name = "kling"

    def is_configured(self) -> bool:
        return bool(self.config.kling_access_key and self.config.kling_secret_key)

    def _token(self) -> str:
        if not self.is_configured():
            raise ProviderNotConfigured("Set KLING_ACCESS_KEY and KLING_SECRET_KEY")
        now = int(time.time())
        payload = {
            "iss": self.config.kling_access_key,
            "exp": now + 1800,
            "nbf": now - 5,
        }
        return jwt.encode(payload, self.config.kling_secret_key, algorithm="HS256")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/json",
        }

    async def submit(self, request: VideoRequest) -> TaskRef:
        model = KLING_MODELS.get(request.model or "1.6", request.model or "1.6")
        endpoint = "text2video" if not request.image_url else "image2video"
        payload = {
            "model_name": model,
            "prompt": request.prompt,
            "duration": str(request.duration),
            "aspect_ratio": request.aspect_ratio,
            "mode": "std",
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.image_url:
            payload["image"] = request.image_url
        payload.update(request.extra)

        url = f"{self.config.kling_base_url.rstrip('/')}/v1/videos/{endpoint}"
        response = await self.client.post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise APIError("Kling submit failed", response.status_code, response.text)

        data = response.json()
        task_id = data.get("data", {}).get("task_id") or data.get("task_id")
        if not task_id:
            raise APIError("Kling returned no task id", payload=data)
        return TaskRef(task_id=task_id, provider=self.name, raw=data)

    async def get_status(self, task_id: str) -> TaskStatus:
        for endpoint in ("text2video", "image2video"):
            url = f"{self.config.kling_base_url.rstrip('/')}/v1/videos/{endpoint}/{task_id}"
            response = await self.client.get(url, headers=self._headers())
            if response.status_code == 200:
                return _parse_status(task_id, response.json())
        raise APIError(f"Kling status failed for task {task_id}")


def _parse_status(task_id: str, data: dict) -> TaskStatus:
    inner = data.get("data", {})
    raw = str(inner.get("task_status") or "").lower()
    state = {
        "submitted": "pending",
        "processing": "running",
        "succeed": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(raw, "pending")

    video_url = None
    if state == "succeeded":
        videos = inner.get("task_result", {}).get("videos") or []
        if videos:
            video_url = videos[0].get("url")

    return TaskStatus(
        task_id=task_id,
        provider="kling",
        state=state,
        progress=inner.get("progress"),
        video_url=video_url,
        error=inner.get("task_status_msg"),
        raw=data,
    )
