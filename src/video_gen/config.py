"""Environment-based configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Config:
    seedance_api_key: Optional[str] = field(default_factory=lambda: _get("SEEDANCE_API_KEY"))
    seedance_base_url: Optional[str] = field(default_factory=lambda: _get("SEEDANCE_BASE_URL"))

    kling_access_key: Optional[str] = field(default_factory=lambda: _get("KLING_ACCESS_KEY"))
    kling_secret_key: Optional[str] = field(default_factory=lambda: _get("KLING_SECRET_KEY"))
    kling_base_url: str = field(
        default_factory=lambda: _get("KLING_BASE_URL", "https://api.klingai.com")  # type: ignore[arg-type]
    )

    minimax_api_key: Optional[str] = field(default_factory=lambda: _get("MINIMAX_API_KEY"))
    minimax_group_id: Optional[str] = field(default_factory=lambda: _get("MINIMAX_GROUP_ID"))
    minimax_base_url: Optional[str] = field(default_factory=lambda: _get("MINIMAX_BASE_URL"))

    dashscope_api_key: Optional[str] = field(default_factory=lambda: _get("DASHSCOPE_API_KEY"))
    dashscope_base_url: str = field(
        default_factory=lambda: _get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com")  # type: ignore[arg-type]
    )

    poll_interval: float = field(default_factory=lambda: float(_get("POLL_INTERVAL", "5") or 5))
    poll_timeout: float = field(default_factory=lambda: float(_get("POLL_TIMEOUT", "600") or 600))
