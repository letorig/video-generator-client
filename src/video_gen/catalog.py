"""Provider/model metadata shared by the CLI and web UI.

Most providers publish a small fixed set of short model names, so their options
are a static map. OrcaRouter's catalog is live and per-workspace, so its options
are resolved from the catalog at request time (see
:mod:`video_gen.orcarouter.catalog`); the entry below is the verified cold-start
fallback used only when discovery fails.
"""

from __future__ import annotations

from typing import TypedDict

from .orcarouter.catalog import SEED_MODELS, select


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
    "orcarouter": {
        "label": "OrcaRouter",
        # Verified cold-start fallback only. The live catalog replaces this
        # whenever discovery succeeds; it is never merged into a live result.
        "models": {m.id: m.id for m in SEED_MODELS if "openai-video" in m.endpoint_types},
        "supports_image": True,
    },
}

#: Providers whose model list comes from live discovery rather than this map.
DISCOVERED_PROVIDERS = frozenset({"orcarouter"})

ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
RESOLUTIONS = ["720p", "1080p"]


def static_model_options(provider: str, requires_image: bool = False) -> list[str]:
    """Fallback option list for a discovered provider, filtered by capability.

    Mirrors the live filter exactly, including the fail-closed rule for image
    input, so a degraded catalog never offers a model the live one would hide.
    """
    if provider not in DISCOVERED_PROVIDERS:
        return list(PROVIDER_INFO[provider]["models"])

    modalities = ("image",) if requires_image else ()
    return [m.id for m in select(SEED_MODELS, "video", modalities)]
