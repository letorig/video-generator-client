"""Seedance provider adapter.

The exact Seedance endpoint/model names depend on the API gateway you use.
Set SEEDANCE_BASE_URL to your gateway rather than relying on a guessed URL.
"""

from __future__ import annotations

from ..exceptions import APIError, ProviderNotConfigured
from ..models import TaskRef, TaskStatus, VideoRequest
from .base import BaseProvider

SEEDANCE_MODELS = {
    "2.5": "seedance-2.5",
    "2.0-mini": "seedance-2.0-mini",
    "fast": "seedance-fast",
}


class SeedanceProvider(BaseProvider):
    name = "seedance"

    def is_configured(self) -> bool:
        return bool(self.config.seedance_api_key and self.config.seedance_base_url)

    def _headers(self) -> dict[str, str]:
        if not self.is_configured():
            raise ProviderNotConfigured("Set SEEDANCE_API_KEY and SEEDANCE_BASE_URL")
        return {
            "Authorization": f"Bearer {self.config.seedance_api_key}",
            "Content-Type": "application/json",
        }

    async def submit(self, request: VideoRequest) -> TaskRef:
        model = SEEDANCE_MODELS.get(request.model or "2.5", request.model or "2.5")
        payload = {
            "model": model,
            "prompt": request.prompt,
            "duration": request.duration,
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.image_url:
            payload["image_url"] = request.image_url
        if request.seed is not None:
            payload["seed"] = request.seed
        payload.update(request.extra)

        url = f"{self.config.seedance_base_url.rstrip('/')}/v1/videos/generations"
        response = await self.client.post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise APIError("Seedance submit failed", response.status_code, response.text)

        data = response.json()
        task_id = data.get("id") or data.get("task_id") or data.get("data", {}).get("id")
        if not task_id:
            raise APIError("Seedance returned no task id", payload=data)
        return TaskRef(task_id=task_id, provider=self.name, raw=data)

    async def get_status(self, task_id: str) -> TaskStatus:
        url = f"{self.config.seedance_base_url.rstrip('/')}/v1/videos/generations/{task_id}"
        response = await self.client.get(url, headers=self._headers())
        if response.status_code >= 400:
            raise APIError("Seedance status failed", response.status_code, response.text)
        return _parse_status(task_id, response.json())


def _parse_status(task_id: str, data: dict) -> TaskStatus:
    raw = str(data.get("status") or data.get("state") or "").lower()
    state = {
        "queued": "pending",
        "pending": "pending",
        "processing": "running",
        "running": "running",
        "succeeded": "succeeded",
        "success": "succeeded",
        "completed": "succeeded",
        "failed": "failed",
        "error": "failed",
        "cancelled": "cancelled",
    }.get(raw, "pending")

    video_url = None
    if state == "succeeded":
        outputs = data.get("outputs") or []
        video_url = (
            data.get("video_url")
            or data.get("output", {}).get("video_url")
            or (outputs[0].get("url") if outputs and isinstance(outputs[0], dict) else None)
        )

    return TaskStatus(
        task_id=task_id,
        provider="seedance",
        state=state,
        progress=data.get("progress"),
        video_url=video_url,
        error=data.get("error"),
        raw=data,
    )
