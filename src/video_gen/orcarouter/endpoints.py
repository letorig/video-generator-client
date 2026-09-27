"""OrcaRouter origin and endpoint resolution.

Authentication and inference live on two *different* public origins. They are
never derived from one another: the auth origin is not the API origin with a
path swapped, and the API origin is not the auth origin with ``/v1`` appended.
Getting this wrong is the single most common OrcaRouter integration mistake,
because ``https://api.orcarouter.ai/v1/auth/keys`` looks plausible and 404s.

Resolution order, highest priority first:

1. ``ORCA_AUTH_BASE_URL`` / ``ORCA_API_BASE_URL`` (explicit, separate overrides)
2. ``ORCA_BASE_URL`` (one shared origin, for self-hosted deployments)
3. the public defaults below
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

DEFAULT_AUTH_BASE_URL = "https://www.orcarouter.ai"
DEFAULT_API_BASE_URL = "https://api.orcarouter.ai/v1"

#: The consent screen. This is a browser page, not an API.
AUTHORIZE_PATH = "/auth"
#: Code -> key exchange. Note the ``/api/v1/auth`` prefix, *not* ``/v1/auth``.
EXCHANGE_PATH = "/api/v1/auth/keys"
#: Device-grant endpoints (RFC 8628). Not used by this client, published here so
#: the origin policy stays the single place that knows OrcaRouter paths.
DEVICE_CODE_PATH = "/api/v1/auth/device/code"
DEVICE_TOKEN_PATH = "/api/v1/auth/device/token"
#: Inference-side paths, relative to the API base.
MODELS_PATH = "/models"
VIDEO_GENERATIONS_PATH = "/video/generations"

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class OriginError(ValueError):
    """An OrcaRouter base URL is unusable."""


def _get(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def _clean(url: str, *, setting: str) -> str:
    """Validate one configured base URL and normalise trailing slashes."""
    candidate = url.strip()
    parts = urlsplit(candidate)
    if parts.scheme not in ("http", "https"):
        raise OriginError(f"{setting} must be an absolute http(s) URL, got {candidate!r}")
    if not parts.hostname:
        raise OriginError(f"{setting} has no host: {candidate!r}")
    if parts.scheme == "http" and parts.hostname.lower() not in _LOOPBACK_HOSTS:
        raise OriginError(
            f"{setting} uses plain HTTP for non-loopback host {parts.hostname!r}; "
            "HTTPS is required except for loopback development"
        )
    return candidate.rstrip("/")


@dataclass(frozen=True)
class Origins:
    """The two OrcaRouter origins a client needs. Never equal by accident."""

    auth_base: str
    api_base: str

    def authorize_url(self) -> str:
        return f"{self.auth_base}{AUTHORIZE_PATH}"

    def exchange_url(self) -> str:
        return f"{self.auth_base}{EXCHANGE_PATH}"

    def device_code_url(self) -> str:
        return f"{self.auth_base}{DEVICE_CODE_PATH}"

    def device_token_url(self) -> str:
        return f"{self.auth_base}{DEVICE_TOKEN_PATH}"

    def models_url(self) -> str:
        return f"{self.api_base}{MODELS_PATH}"

    def video_generations_url(self) -> str:
        return f"{self.api_base}{VIDEO_GENERATIONS_PATH}"


def resolve_origins(
    auth_base_url: Optional[str] = None,
    api_base_url: Optional[str] = None,
    shared_base_url: Optional[str] = None,
) -> Origins:
    """Resolve the auth and inference origins from arguments, then environment."""
    shared = shared_base_url or _get("ORCA_BASE_URL")

    auth = auth_base_url or _get("ORCA_AUTH_BASE_URL")
    if auth is None:
        auth = shared or DEFAULT_AUTH_BASE_URL
    # The shared self-hosted base carries the public API prefix; an explicit
    # ORCA_API_BASE_URL is taken exactly as written.
    api = api_base_url or _get("ORCA_API_BASE_URL")
    if api is None:
        api = f"{shared.rstrip('/')}/v1" if shared else DEFAULT_API_BASE_URL

    return Origins(
        auth_base=_clean(auth, setting="ORCA_AUTH_BASE_URL"),
        api_base=_clean(api, setting="ORCA_API_BASE_URL"),
    )
