"""Unified Video Gen."""

from .client import Task, VideoClient
from .config import Config
from .models import TaskRef, TaskStatus, VideoRequest, VideoResult

__version__ = "0.1.0"

__all__ = [
    "VideoClient",
    "Task",
    "Config",
    "VideoRequest",
    "TaskRef",
    "TaskStatus",
    "VideoResult",
]
