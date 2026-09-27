"""OrcaRouter OAuth 2.0 + PKCE connect flows.

Two flows, chosen by where the client runs:

* **Flow A — loopback redirect.** The CLI runs on the user's own machine, so it
  binds ``127.0.0.1:<ephemeral>`` *before* opening the browser and receives the
  code automatically. One click, no copy-paste.
* **Flow B — out-of-band code.** The web UI is served by a local server that may
  be bound to ``0.0.0.0`` and reached from another device, so a loopback
  ``callback_url`` would name the wrong machine. Flow B needs no predictable
  address; the consent screen shows a code and the user pastes it back.

Both always send ``code_challenge_method=S256``. Flow B mandates it, and Flow A
sends it unconditionally because the user can pick "show me a code" on the
consent screen even when a real ``callback_url`` was supplied.

The verifier never leaves this process before the exchange: it is not in any
URL, log line, error message, or telemetry event.
"""

from __future__ import annotations

import asyncio
import webbrowser
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode

import httpx

from .credentials import (
    METHOD_PKCE,
    ApiKeyAdapter,
    Credential,
    CredentialStore,
)
from .endpoints import Origins, resolve_origins
from .pkce import challenge_for, generate_state, generate_verifier, state_matches

#: The default scope. ``connector`` is the only other value the endpoint accepts.
DEFAULT_SCOPE = "api"

#: How long a loopback listener waits for the browser before giving up.
LOOPBACK_TIMEOUT = 180.0

#: The consent screen caps PKCE-issued keys at 10 per user per 24 hours.
QUOTA_HINT = (
    "OrcaRouter limits how many keys an app may issue per account per day. "
    "Reuse the stored key instead of signing in again, or try later."
)

#: Default browser-launch command per platform, used only by the CLI flow.
_BROWSER_COMMANDS = {"darwin": "open", "win32": "start"}


class ConnectError(Exception):
    """The connect flow could not produce a credential."""

    #: Whether the user can usefully retry without changing anything.
    retryable = True


class AuthorizationDenied(ConnectError):
    """The user declined on the consent screen. Terminal for this attempt."""

    retryable = False


class ConnectTimeout(ConnectError):
    """No callback arrived before the deadline."""


class StateMismatch(ConnectError):
    """The callback state did not match. Treated as hostile; nothing is redeemed."""

    retryable = False


class ExchangeError(ConnectError):
    """The code -> key exchange failed."""


class ScopeInsufficient(ConnectError):
    """The granted scope does not cover this client's use of the key."""

    retryable = False


@dataclass(frozen=True)
class ExchangeResult:
    """A successful exchange. ``api_key`` never appears in a repr."""

    api_key: str = field(repr=False)
    user_id: Optional[str] = None
    scope: Optional[str] = None
    requested_scope: str = DEFAULT_SCOPE

    def __repr__(self) -> str:  # pragma: no cover - defensive, asserted in tests
        from .credentials import mask_secret

        return (
            f"ExchangeResult(masked={mask_secret(self.api_key)!r}, "
            f"user_id={self.user_id!r}, scope={self.scope!r})"
        )

    __str__ = __repr__


@dataclass
class PendingAuthorization:
    """One in-flight authorization attempt. Holds the secret verifier."""

    attempt: int
    flow: str
    authorize_url: str
    verifier: str = field(repr=False)
    state: str = field(repr=False)
    scope: str = DEFAULT_SCOPE
    callback_url: str = ""


# --------------------------------------------------------------------------- #
# Authorize URL
# --------------------------------------------------------------------------- #


def build_authorize_url(
    origins: Origins,
    pending: PendingAuthorization,
    *,
    app_name: str,
    login_hint: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Compose the consent-screen URL.

    ``callback_url`` is either the literal ``oob`` or a loopback redirect; the
    host is ``www.orcarouter.ai`` and is never derived from the inference origin.
    """
    params = {
        "callback_url": pending.callback_url,
        "code_challenge": challenge_for(pending.verifier),
        "code_challenge_method": "S256",
        "state": pending.state,
        "app_name": app_name,
        "scope": pending.scope,
    }
    if login_hint:
        params["login_hint"] = login_hint
    if prompt:
        params["prompt"] = prompt
    return f"{origins.authorize_url()}?{urlencode(params)}"


def new_pending(
    *,
    flow: str,
    attempt: int = 1,
    callback_url: str = "oob",
    scope: str = DEFAULT_SCOPE,
) -> PendingAuthorization:
    """A fresh verifier and state for every attempt — never reused."""
    return PendingAuthorization(
        attempt=attempt,
        flow=flow,
        authorize_url="",
        verifier=generate_verifier(),
        state=generate_state(),
        scope=scope,
        callback_url=callback_url,
    )


# --------------------------------------------------------------------------- #
# Exchange
# --------------------------------------------------------------------------- #


async def exchange_code(
    origins: Origins,
    *,
    code: str,
    verifier: str,
    scope: str = DEFAULT_SCOPE,
    client: Optional[httpx.AsyncClient] = None,
) -> ExchangeResult:
    """Redeem an auth code for an OrcaRouter API key.

    Posts to the *auth* origin at ``/api/v1/auth/keys`` — never to
    ``api.orcarouter.ai/v1/auth/keys``, which is a 404.
    """
    body = {
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    }
    owns = client is None
    http = client or httpx.AsyncClient(timeout=30.0)
    try:
        try:
            response = await http.post(origins.exchange_url(), json=body)
        except httpx.HTTPError as exc:
            raise ExchangeError(
                f"Could not reach the OrcaRouter authorization service at "
                f"{origins.auth_base}: {type(exc).__name__}"
            ) from exc
    finally:
        if owns:
            await http.aclose()

    if response.status_code == 400:
        raise ExchangeError(
            "OrcaRouter rejected the exchange: the code challenge method did not "
            "match the one sent at authorize time."
        )
    if response.status_code == 403:
        raise ExchangeError(
            "OrcaRouter rejected the code. It is unknown, expired, already used, "
            "or the verifier did not match. Start a new sign-in."
        )
    if response.status_code == 429:
        raise ExchangeError(f"OrcaRouter rate-limited key issuance. {QUOTA_HINT}")
    if response.status_code >= 400:
        raise ExchangeError(f"OrcaRouter exchange failed with HTTP {response.status_code}.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExchangeError("OrcaRouter returned a non-JSON exchange response.") from exc

    api_key = payload.get("key")
    if not isinstance(api_key, str) or not api_key:
        raise ExchangeError("OrcaRouter returned no key in the exchange response.")

    # Read the *granted* scope back. It is not necessarily what was asked for.
    granted = payload.get("scope")
    granted = granted if isinstance(granted, str) else None
    if granted != scope:
        raise ScopeInsufficient(
            f"OrcaRouter granted scope {granted!r}, which does not cover this "
            f"client's use of the key (requires {scope!r})."
        )

    user_id = payload.get("user_id")
    return ExchangeResult(
        api_key=api_key,
        user_id=str(user_id) if user_id is not None else None,
        scope=granted,
        requested_scope=scope,
    )


# --------------------------------------------------------------------------- #
# Flow A — loopback redirect
# --------------------------------------------------------------------------- #

_CLOSE_PAGE = (
    "<!doctype html><meta charset='utf-8'><title>OrcaRouter</title>"
    "<body style='font:16px system-ui;padding:48px;text-align:center'>"
    "<h2>Connected to OrcaRouter</h2><p>You can close this tab and return to "
    "your terminal.</p></body>"
)


async def connect_loopback(
    store: CredentialStore,
    *,
    app_name: str = "Video Gen Client",
    origins: Optional[Origins] = None,
    scope: str = DEFAULT_SCOPE,
    timeout: float = LOOPBACK_TIMEOUT,
    open_browser: bool = True,
    on_authorize_url=None,
    client: Optional[httpx.AsyncClient] = None,
) -> Credential:
    """Flow A: bind loopback first, open the browser, receive the code, exchange."""
    resolved = origins or resolve_origins()
    loop = asyncio.get_running_loop()
    outcome: asyncio.Future = loop.create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            # Drain headers so the browser sees a complete response.
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            parts = request_line.decode("latin-1").split(" ")
            target = parts[1] if len(parts) > 1 else "/"
            path, _, query = target.partition("?")
            params = dict(
                pair.split("=", 1) if "=" in pair else (pair, "")
                for pair in query.split("&")
                if pair
            )
            from urllib.parse import unquote_plus

            params = {k: unquote_plus(v) for k, v in params.items()}

            if path != "/cb":
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return

            body = _CLOSE_PAGE.encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()

            if not outcome.done():
                # Compare state before doing anything else with the payload.
                if not state_matches(pending.state, params.get("state")):
                    outcome.set_exception(
                        StateMismatch(
                            "The callback state did not match this sign-in attempt; "
                            "the code was not redeemed."
                        )
                    )
                elif params.get("error"):
                    outcome.set_exception(
                        AuthorizationDenied(
                            f"Authorization was not granted ({params['error']})."
                        )
                    )
                elif params.get("code"):
                    outcome.set_result(params["code"])
                else:
                    outcome.set_exception(
                        ExchangeError("The callback carried neither a code nor an error.")
                    )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):  # pragma: no cover - client vanished
                pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    pending = new_pending(
        flow="loopback",
        callback_url=f"http://127.0.0.1:{port}/cb",
        scope=scope,
    )
    pending.authorize_url = build_authorize_url(resolved, pending, app_name=app_name)

    if on_authorize_url is not None:
        on_authorize_url(pending.authorize_url)
    if open_browser:
        _open_browser(pending.authorize_url)

    try:
        try:
            code = await asyncio.wait_for(outcome, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ConnectTimeout(
                f"No authorization callback arrived within {int(timeout)}s. "
                "Nothing was redeemed; run the command again."
            ) from exc
    finally:
        server.close()
        try:
            await server.wait_closed()
        except (ConnectionError, OSError):  # pragma: no cover
            pass

    result = await exchange_code(
        resolved, code=code, verifier=pending.verifier, scope=scope, client=client
    )
    return store.save(
        result.api_key, method=METHOD_PKCE, user_id=result.user_id, scope=result.scope
    )


def _open_browser(url: str) -> bool:
    """Best effort. The URL is always printed, so a failure is not fatal."""
    try:
        return webbrowser.open(url)
    except (webbrowser.Error, OSError, RuntimeError):  # pragma: no cover - headless
        return False


# --------------------------------------------------------------------------- #
# Flow B — out-of-band code
# --------------------------------------------------------------------------- #


def start_oob(
    *,
    app_name: str = "Video Gen Client",
    origins: Optional[Origins] = None,
    scope: str = DEFAULT_SCOPE,
    attempt: int = 1,
) -> PendingAuthorization:
    """Build a Flow B authorization. ``callback_url`` is the literal ``oob``."""
    resolved = origins or resolve_origins()
    pending = new_pending(flow="oob", attempt=attempt, callback_url="oob", scope=scope)
    pending.authorize_url = build_authorize_url(resolved, pending, app_name=app_name)
    return pending


async def complete_oob(
    store: CredentialStore,
    pending: PendingAuthorization,
    code: str,
    *,
    origins: Optional[Origins] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Credential:
    """Redeem the code the user pasted back."""
    resolved = origins or resolve_origins()
    result = await exchange_code(
        resolved, code=code.strip(), verifier=pending.verifier, scope=pending.scope,
        client=client,
    )
    return store.save(
        result.api_key, method=METHOD_PKCE, user_id=result.user_id, scope=result.scope
    )


__all__ = [
    "AuthorizationDenied",
    "ConnectError",
    "ConnectTimeout",
    "ExchangeError",
    "ExchangeResult",
    "LoginManager",
    "PendingAuthorization",
    "ScopeInsufficient",
    "StateMismatch",
    "build_authorize_url",
    "complete_oob",
    "connect_loopback",
    "exchange_code",
    "new_pending",
    "start_oob",
]


# --------------------------------------------------------------------------- #
# Server-side login lifecycle
# --------------------------------------------------------------------------- #


class LoginManager:
    """A generation-safe "login in progress" lock for the web UI.

    Every terminal path — success, denial, exchange error, timeout, explicit
    cancel, switching authentication method, closing the modal, unmount,
    reload/close, and ``pagehide`` — must release both the server lock and the UI
    state. The monotonic ``attempt`` id makes that safe: a late response from an
    attempt that is no longer current can never write a credential or mutate UI
    state.
    """

    def __init__(self, store: CredentialStore, *, app_name: str = "Video Gen Client"):
        self._store = store
        self._app_name = app_name
        self._attempt = 0
        self._pending: Optional[PendingAuthorization] = None
        self._status = "idle"
        self._detail = ""
        #: Set when a browser user pastes an API key instead; keeps the two
        #: authentication choices independently usable.
        self._api_keys = ApiKeyAdapter(store)

    # -- state ------------------------------------------------------------- #

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def snapshot(self) -> dict:
        credential = self._store.current()
        return {
            "attempt": self._attempt,
            "status": self._status,
            "busy": self.busy,
            "detail": self._detail,
            "has_credential": credential is not None,
            "needs_reauth": bool(credential and credential.needs_reauth),
            "method": credential.method if credential else None,
            "masked": credential.masked if credential else None,
            "user_id": credential.user_id if credential else None,
            "scope": credential.scope if credential else None,
        }

    # -- lifecycle --------------------------------------------------------- #

    def start(self, *, scope: str = DEFAULT_SCOPE, origins: Optional[Origins] = None) -> dict:
        """Begin a Flow B attempt, replacing any previous one."""
        self._attempt += 1
        self._pending = start_oob(
            app_name=self._app_name, origins=origins, scope=scope, attempt=self._attempt
        )
        self._status = "pending"
        self._detail = ""
        return {
            "attempt": self._attempt,
            "authorize_url": self._pending.authorize_url,
            "flow": self._pending.flow,
        }

    async def complete(
        self,
        attempt: int,
        code: str,
        *,
        scope: str = DEFAULT_SCOPE,
        origins: Optional[Origins] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[Credential]:
        """Redeem a pasted code, but only if this attempt is still current.

        Returns ``None`` when the attempt was superseded or cancelled — the key
        is then discarded rather than persisted.
        """
        if attempt != self._attempt or self._pending is None:
            return None
        pending = self._pending
        try:
            credential = await complete_oob(
                self._store, pending, code, origins=origins, client=client
            )
        except ConnectError as exc:
            if attempt == self._attempt:
                self._status = "error"
                self._detail = str(exc)
                self._pending = None
            raise

        if attempt != self._attempt:
            # Cancelled or superseded while the exchange was in flight. Do not
            # let a late success become the stored credential.
            return None

        self._status = "connected"
        self._detail = ""
        self._pending = None
        return credential

    def cancel(self, attempt: Optional[int] = None) -> bool:
        """Release the lock. Called by Cancel, unmount, and ``pagehide``."""
        if attempt is not None and attempt != self._attempt:
            return False
        was_busy = self._pending is not None
        self._pending = None
        if self._status == "pending":
            self._status = "cancelled"
        self._detail = ""
        return was_busy

    def set_api_key(self, value: str) -> Credential:
        """The other authentication choice, on the same credential seam."""
        return self._api_keys.save(value)

    def clear(self) -> None:
        self.cancel()
        self._store.clear()
        self._status = "idle"
        self._detail = ""
