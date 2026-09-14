"""Shared Pydantic models."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ProviderName = Literal["seedance", "kling", "minimax", "wan"]
TaskState = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    image_url: Optional[str] = None
    duration: int = 5
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    seed: Optional[int] = None
    model: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class TaskRef(BaseModel):
    task_id: str
    provider: str
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(BaseModel):
    task_id: str
    provider: str
    state: TaskState
    progress: Optional[float] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class VideoResult(BaseModel):
    task_id: str
    provider: str
    video_url: str
    duration: Optional[int] = None
    raw: dict[str, Any] = Field(default_factory=dict)
