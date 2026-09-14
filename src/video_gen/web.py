"""FastAPI backend for the local web interface."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .catalog import ASPECT_RATIOS, PROVIDER_INFO, RESOLUTIONS
from .client import VideoClient
from .config import Config
from .exceptions import VideoGenError
from .utils.video import download_video

STATIC_DIR = Path(__file__).parent / "static"
TASKS: dict[str, dict[str, Any]] = {}
CLIENT: Optional[VideoClient] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global CLIENT
    CLIENT = VideoClient()
    try:
        yield
    finally:
        if CLIENT is not None:
            await CLIENT.aclose()
            CLIENT = None
        TASKS.clear()


app = FastAPI(title="Unified Video Gen", version="0.1.0", lifespan=lifespan)


class GenerateBody(BaseModel):
    provider: str
    prompt: str = Field(min_length=1)
    model: Optional[str] = None
    negative_prompt: Optional[str] = None
    image_url: Optional[str] = None
    duration: int = Field(default=5, ge=1, le=60)
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    seed: Optional[int] = None


def _client() -> VideoClient:
    if CLIENT is None:
        raise HTTPException(503, "Server is still starting")
    return CLIENT


def _configured() -> dict[str, bool]:
    cfg = Config()
    return {
        "seedance": bool(cfg.seedance_api_key),
        "kling": bool(cfg.kling_access_key and cfg.kling_secret_key),
        "minimax": bool(cfg.minimax_api_key),
        "wan": bool(cfg.dashscope_api_key),
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/providers")
async def api_providers() -> dict[str, Any]:
    configured = _configured()
    return {
        "providers": [
            {
                "name": name,
                "label": info["label"],
                "models": list(info["models"].keys()),
                "supports_image": info["supports_image"],
                "configured": configured[name],
            }
            for name, info in PROVIDER_INFO.items()
        ],
        "aspect_ratios": ASPECT_RATIOS,
        "resolutions": RESOLUTIONS,
    }


@app.post("/api/generate")
async def api_generate(body: GenerateBody) -> dict[str, str]:
    provider = body.provider.lower()
    info = PROVIDER_INFO.get(provider)
    if info is None:
        raise HTTPException(400, f"Unknown provider: {body.provider}")
    if not _configured().get(provider, False):
        raise HTTPException(400, f"Provider '{provider}' is not configured")
    if body.image_url and not info["supports_image"]:
        raise HTTPException(400, f"Provider '{provider}' does not support image-to-video")

    try:
        task = await _client().generate(
            provider,
            prompt=body.prompt,
            model=body.model,
            negative_prompt=body.negative_prompt,
            image_url=body.image_url,
            duration=body.duration,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            seed=body.seed,
        )
    except VideoGenError as exc:
        raise HTTPException(400, str(exc)) from exc

    local_id = uuid.uuid4().hex[:12]
    TASKS[local_id] = {
        "local_id": local_id,
        "task_id": task.task_id,
        "provider": provider,
        "prompt": body.prompt,
        "state": "pending",
        "video_url": None,
        "error": None,
    }
    asyncio.create_task(_poll(local_id, task))
    return {"local_id": local_id, "task_id": task.task_id, "provider": provider}


async def _poll(local_id: str, task: Any) -> None:
    entry = TASKS.get(local_id)
    if entry is None:
        return

    def on_update(status: Any) -> None:
        entry["state"] = status.state
        entry["video_url"] = status.video_url
        entry["error"] = status.error

    try:
        result = await task.wait(on_update=on_update)
        entry["state"] = "succeeded"
        entry["video_url"] = result.video_url
    except Exception as exc:
        entry["state"] = "failed"
        entry["error"] = str(exc)


@app.get("/api/tasks")
async def api_tasks() -> dict[str, list[dict[str, Any]]]:
    return {"tasks": list(reversed(list(TASKS.values())))}


@app.get("/api/tasks/{local_id}")
async def api_task(local_id: str) -> dict[str, Any]:
    entry = TASKS.get(local_id)
    if entry is None:
        raise HTTPException(404, "No such task")
    return entry


@app.get("/api/tasks/{local_id}/download")
async def api_download(local_id: str) -> FileResponse:
    entry = TASKS.get(local_id)
    if entry is None or not entry.get("video_url"):
        raise HTTPException(404, "No video for this task yet")

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{local_id}.mp4"
    if not dest.exists():
        try:
            await download_video(entry["video_url"], str(dest))
        except Exception as exc:
            raise HTTPException(502, f"Video download failed: {exc}") from exc
    return FileResponse(dest, media_type="video/mp4", filename=f"{local_id}.mp4")
