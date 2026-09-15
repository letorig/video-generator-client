"""The OrcaRouter credential seam.

Both ways of getting an OrcaRouter key — pasting an existing ``sk-orca-…`` key,
or signing in with an account through OAuth 2.0 + PKCE — are adapters on one
small interface and both produce the *same* :class:`Credential`. Nothing
downstream (the provider adapter, model discovery, the CLI, the web UI) knows or
cares which path was used.

The key is a normal, durable OrcaRouter API key. It is not a refresh token:
there is no refresh grant to call, so nothing here ever refreshes. An upstream
``401`` is a terminal reauthentication requirement, handled generation-safely by
:meth:`CredentialStore.mark_needs_reauth`.

Secrets live in the project's existing store — ``.env`` / environment variables,
the same place every other provider's key lives. No second credential store is
introduced.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

#: Managed by this module. Comments and unrelated provider keys are preserved.
ENV_KEY = "ORCAROUTER_API_KEY"
ENV_METHOD = "ORCAROUTER_AUTH_METHOD"
ENV_USER_ID = "ORCAROUTER_USER_ID"
ENV_SCOPE = "ORCAROUTER_SCOPE"
ENV_NEEDS_REAUTH = "ORCAROUTER_NEEDS_REAUTH"

_MANAGED = (ENV_KEY, ENV_METHOD, ENV_USER_ID, ENV_SCOPE, ENV_NEEDS_REAUTH)

#: Provider IDs used in the UI and the registry. API-key and Auth stay distinct
#: everywhere both can appear, so support, logout and reauthentication are never
#: ambiguous.
METHOD_API_KEY = "api_key"
METHOD_PKCE = "pkce"
METHOD_ENV = "env"

#: Lightweight shape check only. A prefix is not proof that a key is valid, and
#: this deliberately does not claim to have verified anything.
KEY_PREFIX = "sk-orca-"


class CredentialError(Exception):
    """Base class for credential problems."""


class InvalidApiKey(CredentialError):
    """The pasted value is obviously not an OrcaRouter key."""


class NeedsReauth(CredentialError):
    """The stored credential was rejected upstream and must be replaced."""


def looks_like_api_key(value: str) -> bool:
    """Cheap format check to catch obvious input mistakes."""
    candidate = (value or "").strip()
    return candidate.startswith(KEY_PREFIX) and len(candidate) > len(KEY_PREFIX) + 4


def mask_secret(value: Optional[str]) -> str:
    """Render a secret for display. Never returns the whole key."""
    if not value:
        return ""
    if len(value) <= 12:
        return "…"
    return f"{value[:8]}…{value[-4:]}"


def redact(text: str, *secrets_: Optional[str]) -> str:
    """Remove any of ``secrets_`` from ``text`` before it is logged or raised."""
    out = text
    for secret in secrets_:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "sk-orca-***")
    return out


def _get(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def default_env_path() -> Path:
    """The file ``python-dotenv`` already loads: ``.env`` in the working directory."""
    override = _get("ORCAROUTER_ENV_FILE")
    return Path(override) if override else Path(".env")


@dataclass(frozen=True)
class Credential:
    """An OrcaRouter API key plus the provenance the UI needs to show it.

    ``generation`` is captured when a request is issued and compared when a
    ``401`` comes back, so a late failure from an old request can never mark a
    freshly reauthorized credential as broken.
    """

    api_key: str = field(repr=False)
    method: str = METHOD_API_KEY
    user_id: Optional[str] = None
    scope: Optional[str] = None
    generation: int = 0
    needs_reauth: bool = False

    def __repr__(self) -> str:  # pragma: no cover - defensive, asserted in tests
        return (
            f"Credential(method={self.method!r}, masked={mask_secret(self.api_key)!r}, "
            f"user_id={self.user_id!r}, scope={self.scope!r}, "
            f"generation={self.generation}, needs_reauth={self.needs_reauth})"
        )

    __str__ = __repr__

    @property
    def masked(self) -> str:
        return mask_secret(self.api_key)

    @property
    def usable(self) -> bool:
        return bool(self.api_key) and not self.needs_reauth


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        values[name.strip()] = raw.strip()
    return values


def _write_env_file(path: Path, updates: dict[str, Optional[str]]) -> None:
    """Update managed keys in place, preserving every other line verbatim.

    ``None`` removes the key. No lockfile, no second store, no reformatting of
    lines this module does not own.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    remaining = dict(updates)
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        name = stripped.partition("=")[0].strip() if "=" in stripped else None
        if name in remaining and not stripped.startswith("#"):
            value = remaining.pop(name)
            if value is not None:
                out.append(f"{name}={value}")
            continue
        out.append(line)

    pending = [(k, v) for k, v in remaining.items() if v is not None]
    if pending:
        if out and out[-1].strip():
            out.append("")
        out.append("# OrcaRouter (written by video-gen orcarouter-login)")
        out.extend(f"{name}={value}" for name, value in pending)

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - platform without POSIX modes
        pass


class CredentialStore:
    """Reads and writes the OrcaRouter credential in the project's secret store."""

    def __init__(self, env_path: Optional[Path] = None):
        self._env_path = env_path
        self._generation = 0

    @property
    def env_path(self) -> Path:
        return self._env_path or default_env_path()

    @property
    def generation(self) -> int:
        return self._generation

    def _saved(self) -> dict[str, str]:
        return _read_env_file(self.env_path)

    def _value(self, name: str) -> Optional[str]:
        """Environment wins over the file, matching every other provider."""
        return _get(name) or self._saved().get(name)

    def current(self) -> Optional[Credential]:
        api_key = self._value(ENV_KEY)
        if not api_key:
            return None
        method = self._value(ENV_METHOD) or METHOD_ENV
        needs_reauth = (self._value(ENV_NEEDS_REAUTH) or "").lower() in ("1", "true", "yes")
        return Credential(
            api_key=api_key,
            method=method,
            user_id=self._value(ENV_USER_ID),
            scope=self._value(ENV_SCOPE),
            generation=self._generation,
            needs_reauth=needs_reauth,
        )

    def save(
        self,
        api_key: str,
        *,
        method: str,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Credential:
        """Persist a key and clear any previous reauthentication flag.

        The old secret is only replaced here, on success — never deleted ahead of
        a login that might fail, which would turn a transient error into
        irreversible account loss.
        """
        self._generation += 1
        _write_env_file(
            self.env_path,
            {
                ENV_KEY: api_key,
                ENV_METHOD: method,
                ENV_USER_ID: user_id,
                ENV_SCOPE: scope,
                ENV_NEEDS_REAUTH: None,
            },
        )
        os.environ[ENV_KEY] = api_key
        os.environ[ENV_METHOD] = method
        for name, value in ((ENV_USER_ID, user_id), (ENV_SCOPE, scope)):
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        os.environ.pop(ENV_NEEDS_REAUTH, None)
        return Credential(
            api_key=api_key,
            method=method,
            user_id=user_id,
            scope=scope,
            generation=self._generation,
        )

    def clear(self) -> None:
        """Remove the stored credential and its metadata."""
        self._generation += 1
        _write_env_file(self.env_path, {name: None for name in _MANAGED})
        for name in _MANAGED:
            os.environ.pop(name, None)

    def mark_needs_reauth(self, generation: int) -> bool:
        """Flag the credential that issued a rejected request.

        Returns whether this call changed anything. A ``401`` that arrives for an
        older generation is ignored: it belongs to a credential that has already
        been replaced, and must not poison the new one.
        """
        if generation != self._generation:
            return False
        _write_env_file(self.env_path, {ENV_NEEDS_REAUTH: "1"})
        os.environ[ENV_NEEDS_REAUTH] = "1"
        return True


class CredentialAdapter(ABC):
    """How a credential is *obtained*. Both paths end at the same result."""

    method: ClassVar[str]

    @abstractmethod
    def obtain(self) -> Credential:
        """Return a usable credential, or raise :class:`CredentialError`."""


class ApiKeyAdapter(CredentialAdapter):
    """The manual path: the user pastes an ``sk-orca-…`` key they already have."""

    method = METHOD_API_KEY

    def __init__(self, store: CredentialStore):
        self._store = store

    def obtain(self) -> Credential:
        credential = self._store.current()
        if credential is None:
            raise InvalidApiKey("No OrcaRouter API key is configured.")
        return credential

    def save(self, api_key: str) -> Credential:
        candidate = (api_key or "").strip()
        if not looks_like_api_key(candidate):
            raise InvalidApiKey(
                "That does not look like an OrcaRouter key. "
                f"Keys start with '{KEY_PREFIX}'. Create one at "
                "https://www.orcarouter.ai/console/token"
            )
        return self._store.save(candidate, method=METHOD_API_KEY)

    def clear(self) -> None:
        self._store.clear()
