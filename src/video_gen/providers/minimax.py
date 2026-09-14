"""MiniMax / Hailuo provider adapter."""

from __future__ import annotations

from ..exceptions import APIError, ProviderNotConfigured
from ..models import TaskRef, TaskStatus, VideoRequest
from .base import BaseProvider

MINIMAX_MODELS = {
    "hailuo": "MiniMax-Hailuo-02",
    "hailuo-02": "MiniMax-Hailuo-02",
    "t2v-01": "T2V-01",
    "i2v-01": "I2V-01",
}


class MiniMaxProvider(BaseProvider):
    name = "minimax"

    def is_configured(self) -> bool:
        return bool(self.config.minimax_api_key and self.config.minimax_base_url)

    def _headers(self) -> dict[str, str]:
        if not self.is_configured():
            raise ProviderNotConfigured("Set MINIMAX_API_KEY and MINIMAX_BASE_URL")
        return {
            "Authorization": f"Bearer {self.config.minimax_api_key}",
            "Content-Type": "application/json",
        }

    async def submit(self, request: VideoRequest) -> TaskRef:
        model = MINIMAX_MODELS.get(request.model or "hailuo", request.model or "hailuo")
        endpoint = "video_generation" if not request.image_url else "image_to_video"
        payload = {
            "model": model,
            "prompt": request.prompt,
            "duration": request.duration,
            "resolution": request.resolution,
        }
        if request.image_url:
            payload["first_frame_image"] = request.image_url
        payload.update(request.extra)

        params = {}
        if self.config.minimax_group_id:
            params["GroupId"] = self.config.minimax_group_id

        url = f"{self.config.minimax_base_url.rstrip('/')}/v1/{endpoint}"
        response = await self.client.post(
            url, json=payload, headers=self._headers(), params=params
        )
        if response.status_code >= 400:
            raise APIError("MiniMax submit failed", response.status_code, response.text)

        data = response.json()
        task_id = data.get("task_id")
        if not task_id:
            raise APIError("MiniMax returned no task id", payload=data)
        return TaskRef(task_id=task_id, provider=self.name, raw=data)

    async def get_status(self, task_id: str) -> TaskStatus:
        params = {"task_id": task_id}
        if self.config.minimax_group_id:
            params["GroupId"] = self.config.minimax_group_id

        url = f"{self.config.minimax_base_url.rstrip('/')}/v1/query/video_generation"
        response = await self.client.get(url, headers=self._headers(), params=params)
        if response.status_code >= 400:
            raise APIError("MiniMax status failed", response.status_code, response.text)
        return _parse_status(task_id, response.json())


def _parse_status(task_id: str, data: dict) -> TaskStatus:
    raw = str(data.get("status") or "").lower()
    state = {
        "preparing": "pending",
        "queueing": "pending",
        "processing": "running",
        "success": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(raw, "pending")

    video_url = data.get("video_url")
    return TaskStatus(
        task_id=task_id,
        provider="minimax",
        state=state,
        video_url=video_url,
        error=data.get("error_message"),
        raw=data,
    )
