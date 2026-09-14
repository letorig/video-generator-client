"""Wan / Alibaba DashScope provider adapter.

The default DashScope hostname remains supported, but DashScope now recommends
region/workspace-specific endpoints. Set DASHSCOPE_BASE_URL accordingly.
"""

from __future__ import annotations

from ..exceptions import APIError, ProviderNotConfigured
from ..models import TaskRef, TaskStatus, VideoRequest
from .base import BaseProvider

WAN_MODELS = {
    "2.1": "wanx2.1-t2v-turbo",
    "2.1-plus": "wanx2.1-t2v-plus",
    "2.2": "wanx2.2-t2v-plus",
    "2.6": "wan2.6-t2v",
    "i2v": "wanx2.1-i2v-turbo",
}


class WanProvider(BaseProvider):
    name = "wan"

    def is_configured(self) -> bool:
        return bool(self.config.dashscope_api_key and self.config.dashscope_base_url)

    def _headers(self) -> dict[str, str]:
        if not self.is_configured():
            raise ProviderNotConfigured("Set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL")
        return {
            "Authorization": f"Bearer {self.config.dashscope_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    async def submit(self, request: VideoRequest) -> TaskRef:
        model = WAN_MODELS.get(request.model or "2.1", request.model or "2.1")
        payload = {
            "model": model,
            "input": {"prompt": request.prompt},
            "parameters": {
                "duration": request.duration,
                "size": _size_from_aspect(request.aspect_ratio, request.resolution),
            },
        }
        if request.negative_prompt:
            payload["input"]["negative_prompt"] = request.negative_prompt
        if request.image_url:
            payload["input"]["img_url"] = request.image_url
        if request.seed is not None:
            payload["parameters"]["seed"] = request.seed
        extra = dict(request.extra)
        payload["parameters"].update(extra.pop("parameters", {}))
        payload["input"].update(extra.pop("input", {}))
        payload["parameters"].update(extra.pop("parameters_extra", {}))
        payload["input"].update(extra.pop("input_extra", {}))
        payload.update(extra)

        url = (
            f"{self.config.dashscope_base_url.rstrip('/')}"
            "/api/v1/services/aigc/video-generation/video-synthesis"
        )
        response = await self.client.post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise APIError("Wan submit failed", response.status_code, response.text)

        data = response.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise APIError("Wan returned no task id", payload=data)
        return TaskRef(task_id=task_id, provider=self.name, raw=data)

    async def get_status(self, task_id: str) -> TaskStatus:
        url = f"{self.config.dashscope_base_url.rstrip('/')}/api/v1/tasks/{task_id}"
        response = await self.client.get(url, headers=self._headers())
        if response.status_code >= 400:
            raise APIError("Wan status failed", response.status_code, response.text)
        return _parse_status(task_id, response.json())


def _size_from_aspect(ratio: str, resolution: str) -> str:
    table = {
        ("16:9", "720p"): "1280*720",
        ("16:9", "1080p"): "1920*1080",
        ("9:16", "720p"): "720*1280",
        ("9:16", "1080p"): "1080*1920",
        ("1:1", "720p"): "960*960",
        ("1:1", "1080p"): "1080*1080",
    }
    return table.get((ratio, resolution), "1280*720")


def _parse_status(task_id: str, data: dict) -> TaskStatus:
    out = data.get("output", {})
    raw = str(out.get("task_status") or "").lower()
    state = {
        "pending": "pending",
        "queued": "pending",
        "running": "running",
        "processing": "running",
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(raw, "pending")

    return TaskStatus(
        task_id=task_id,
        provider="wan",
        state=state,
        video_url=out.get("video_url") if state == "succeeded" else None,
        error=out.get("message"),
        raw=data,
    )
