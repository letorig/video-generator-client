"""Provider/model metadata shared by the CLI and web UI."""

from __future__ import annotations

from typing import TypedDict


class ProviderInfo(TypedDict):
    label: str
    models: dict[str, str]
    supports_image: bool


PROVIDER_INFO: dict[str, ProviderInfo] = {
    "seedance": {
        "label": "Seedance (ByteDance)",
        "models": {"2.5": "seedance-2.5", "2.0-mini": "seedance-2.0-mini", "fast": "seedance-fast"},
        "supports_image": True,
    },
    "kling": {
        "label": "Kling (Kuaishou)",
        "models": {"1.6": "kling-v1-6", "2.0": "kling-v2", "2.1": "kling-v2-1"},
        "supports_image": True,
    },
    "minimax": {
        "label": "MiniMax / Hailuo",
        "models": {"hailuo": "MiniMax-Hailuo-02", "t2v-01": "T2V-01", "i2v-01": "I2V-01"},
        "supports_image": True,
    },
    "wan": {
        "label": "Wan (Alibaba DashScope)",
        "models": {
            "2.1": "wanx2.1-t2v-turbo",
            "2.1-plus": "wanx2.1-t2v-plus",
            "2.2": "wanx2.2-t2v-plus",
            "i2v": "wanx2.1-i2v-turbo",
        },
        "supports_image": True,
    },
}

ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
RESOLUTIONS = ["720p", "1080p"]
