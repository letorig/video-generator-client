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

from .catalog import (
    ASPECT_RATIOS,
    DISCOVERED_PROVIDERS,
    PROVIDER_INFO,
    RESOLUTIONS,
    static_model_options,
)
from .client import VideoClient
from .config import Config
from .exceptions import VideoGenError
from .orcarouter import (
    ApiKeyAdapter,
    ConnectError,
    CredentialStore,
    InvalidApiKey,
    LoginManager,
    select,
)
from .providers.orcarouter import OrcaRouterProvider
from .utils.video import download_video

STATIC_DIR = Path(__file__).parent / "static"
TASKS: dict[str, dict[str, Any]] = {}
CLIENT: Optional[VideoClient] = None
#: One server-side login lock. The API key never leaves this process: the
#: browser only ever receives the masked form and the catalog metadata.
LOGIN: Optional[LoginManager] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global CLIENT, LOGIN
    CLIENT = VideoClient()
    LOGIN = LoginManager(CredentialStore())
    try:
        yield
    finally:
        if LOGIN is not None:
            LOGIN.cancel()
        LOGIN = None
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
        "orcarouter": bool(CredentialStore().current()),
    }


def _login() -> LoginManager:
    if LOGIN is None:
        raise HTTPException(503, "Server is still starting")
    return LOGIN


def _model_options(provider: str, requires_image: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The exact option list for a selector, plus where it came from.

    Capability filtering happens here, once, for every entry point — the browser
    never receives an option the current input cannot use, and never falls back
    to free text. Only minimal model metadata crosses to the browser; the API key
    stays in this process.
    """
    if provider not in DISCOVERED_PROVIDERS:
        return (
            [{"id": short, "api_id": api_id} for short, api_id in PROVIDER_INFO[provider]["models"].items()],
            {"source": "static", "degraded": False},
        )

    catalog = OrcaRouterProvider(Config()).catalog()
    filtered = select(catalog.models, "video", ("image",) if requires_image else ())
    options = [
        {
            "id": m.id,
            "context_length": m.context_length,
            "input_modalities": list(m.input_modalities),
            "endpoints": list(m.endpoint_types),
        }
        for m in filtered
    ]
    meta: dict[str, Any] = {
        "source": catalog.source,
        "degraded": catalog.degraded,
        "detail": catalog.detail,
        "catalog_model_count": len(catalog.models),
    }
    if not options and catalog.degraded:
        # Discovery failed outright, so the verified seed keeps a fresh install
        # usable — clearly labelled as degraded.
        options = [{"id": m, "context_length": None, "input_modalities": [], "endpoints": []}
                   for m in static_model_options(provider, requires_image)]
        meta["source"] = f"{catalog.source}+static"
    elif not options:
        # Discovery succeeded and authoritatively lists no such model for this
        # key. The seed must *not* be substituted here: it would advertise models
        # the key cannot call. Report the gap instead.
        kind = "image-to-video" if requires_image else "video"
        meta["detail"] = (
            f"This OrcaRouter key is not entitled to any {kind} model. "
            "Check the key's model access in the OrcaRouter console."
        )
    return options, meta


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/providers")
async def api_providers(
    image: bool = False, provider: Optional[str] = None
) -> dict[str, Any]:
    configured = _configured()
    names = [provider.lower()] if provider else list(PROVIDER_INFO)
    entries = []
    for name in names:
        info = PROVIDER_INFO.get(name)
        if info is None:
            raise HTTPException(400, f"Unknown provider: {name}")
        options, meta = _model_options(name, image and bool(info["supports_image"]))
        entries.append(
            {
                "name": name,
                "label": info["label"],
                "models": options,
                "supports_image": info["supports_image"],
                "configured": configured[name],
                "discovered": name in DISCOVERED_PROVIDERS,
                "catalog": meta,
            }
        )
    return {
        "providers": entries,
        "aspect_ratios": ASPECT_RATIOS,
        "resolutions": RESOLUTIONS,
    }


@app.get("/api/models")
async def api_models(provider: str, image: bool = False) -> dict[str, Any]:
    """Recompute the selector options for the current provider and input type."""
    name = provider.lower()
    info = PROVIDER_INFO.get(name)
    if info is None:
        raise HTTPException(400, f"Unknown provider: {name}")
    options, meta = _model_options(name, image and bool(info["supports_image"]))
    return {"provider": name, "models": options, "catalog": meta}


# --------------------------------------------------------------------------- #
# OrcaRouter authentication (both choices, server-side key handling)
# --------------------------------------------------------------------------- #


class ApiKeyBody(BaseModel):
    api_key: str = Field(min_length=1)


class CompleteBody(BaseModel):
    attempt: int
    code: str = Field(min_length=1)


class CancelBody(BaseModel):
    attempt: Optional[int] = None


@app.get("/api/orcarouter/auth")
async def orcarouter_auth_state() -> dict[str, Any]:
    return _login().snapshot()


@app.post("/api/orcarouter/auth/pkce/start")
async def orcarouter_auth_start() -> dict[str, Any]:
    """Begin Flow B. The authorize URL carries only the S256 challenge."""
    try:
        started = _login().start()
    except ConnectError as exc:
        raise HTTPException(400, str(exc)) from exc
    return started


@app.post("/api/orcarouter/auth/pkce/complete")
async def orcarouter_auth_complete(body: CompleteBody) -> dict[str, Any]:
    """Redeem the pasted code — ignored if the attempt was superseded."""
    login = _login()
    try:
        credential = await login.complete(body.attempt, body.code)
    except InvalidApiKey as exc:
        raise HTTPException(400, str(exc)) from exc
    except ConnectError as exc:
        raise HTTPException(400, str(exc)) from exc
    if credential is None:
        raise HTTPException(409, "This sign-in attempt is no longer current.")
    return login.snapshot()


@app.post("/api/orcarouter/auth/api-key")
async def orcarouter_auth_api_key(body: ApiKeyBody) -> dict[str, Any]:
    """The other choice, on the same credential seam."""
    login = _login()
    try:
        ApiKeyAdapter(CredentialStore()).save(body.api_key)
    except InvalidApiKey as exc:
        raise HTTPException(400, str(exc)) from exc
    return login.snapshot()


@app.post("/api/orcarouter/auth/cancel")
async def orcarouter_auth_cancel(body: CancelBody) -> dict[str, Any]:
    """Release the server lock. Also the target of the pagehide keepalive call."""
    login = _login()
    login.cancel(body.attempt)
    return login.snapshot()


@app.post("/api/orcarouter/auth/logout")
async def orcarouter_auth_logout() -> dict[str, Any]:
    _login().clear()
    return _login().snapshot()


@app.post("/api/generate")
async def api_generate(body: GenerateBody) -> dict[str, str]:
    provider = body.provider.lower()
    info = PROVIDER_INFO.get(provider)
    if info is None:
        raise HTTPException(400, f"Unknown provider: {body.provider}")
    if not _configured().get(provider, False):
        raise HTTPException(400, f"Provider '{provider}' is not configured")
    if provider == "orcarouter":
        credential = CredentialStore().current()
        if credential is not None and credential.needs_reauth:
            raise HTTPException(
                400,
                "The stored OrcaRouter credential was rejected. Sign in again or "
                "save a new API key.",
            )
    if body.image_url and not info["supports_image"]:
        raise HTTPException(400, f"Provider '{provider}' does not support image-to-video")
    if body.image_url and provider in DISCOVERED_PROVIDERS:
        # Second layer of protection only: the selector already excludes models
        # that do not declare image input.
        allowed = [option["id"] for option in _model_options(provider, True)[0]]
        if allowed and body.model and body.model not in allowed:
            raise HTTPException(
                400,
                f"Model '{body.model}' cannot accept an image input. "
                f"Choose one of: {', '.join(allowed[:8])}",
            )

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
