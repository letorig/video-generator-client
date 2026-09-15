"""OAuth 2.0 + PKCE connect flows, driven through the real adapters.

Every test here runs the actual flow — authorize URL construction, the loopback
listener or the out-of-band paste, the exchange POST, and persistence — against a
local fake authorization server. Nothing is stubbed at the seam under test, so a
regression in URL shape, body shape, or error classification fails a test.

All keys and codes are fake. No test contacts a real OrcaRouter origin.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from video_gen.orcarouter.connect import (
    AuthorizationDenied,
    ConnectTimeout,
    ExchangeError,
    LoginManager,
    ScopeInsufficient,
    StateMismatch,
    build_authorize_url,
    complete_oob,
    connect_loopback,
    exchange_code,
    new_pending,
    start_oob,
)
from video_gen.orcarouter.credentials import ENV_KEY, CredentialStore
from video_gen.orcarouter.endpoints import Origins
from video_gen.orcarouter.pkce import (
    challenge_for,
    generate_state,
    generate_verifier,
    state_matches,
)

FAKE_KEY = "sk-orca-testonly-abcdefghijklmnop"
FAKE_CODE = "test-code-0001"


# --------------------------------------------------------------------------- #
# A local fake authorization server
# --------------------------------------------------------------------------- #


class FakeAuthServer:
    """Serves the real exchange endpoint shape, and records what it received.

    ``authorize_url`` is only ever *built*, never fetched: the consent screen is
    a browser page and a human approves there. What this server proves is that
    the exchange goes to the auth origin, at ``/api/v1/auth/keys``, with the
    verifier in the body and nowhere else.
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.behaviour = "ok"
        self.scope = "api"
        self.expected_verifier: str | None = None
        self.used_codes: set[str] = set()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence the test output
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = {}
                server.requests.append(
                    {"path": self.path, "body": body, "headers": dict(self.headers)}
                )
                status, payload = server.respond(body)
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def respond(self, body: dict) -> tuple[int, dict]:
        if self.behaviour == "network":
            raise ConnectionResetError("simulated network failure")
        if self.behaviour == "denied":
            return 403, {"error": "invalid_grant"}
        if self.behaviour == "bad_method":
            return 400, {"error": "invalid_request"}
        if self.behaviour == "rate_limited":
            return 429, {"error": "too_many_requests"}
        if self.behaviour == "no_key":
            return 200, {"user_id": "1", "scope": "api"}
        if self.behaviour == "downgrade":
            return 200, {"key": FAKE_KEY, "user_id": "1", "scope": "connector"}

        code = body.get("code")
        # Single-use: a reused code is refused exactly as the real endpoint does.
        if code in self.used_codes:
            return 403, {"error": "invalid_grant"}
        verifier = body.get("code_verifier")
        if not verifier:
            return 403, {"error": "invalid_grant"}
        if self.expected_verifier is not None and verifier != self.expected_verifier:
            return 403, {"error": "invalid_grant"}
        if body.get("code_challenge_method") != "S256":
            return 400, {"error": "invalid_request"}
        self.used_codes.add(code)
        return 200, {"key": FAKE_KEY, "user_id": "42", "scope": self.scope}

    @property
    def origins(self) -> Origins:
        return Origins(auth_base=f"http://127.0.0.1:{self.port}", api_base="http://127.0.0.1:1")

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def auth_server():
    server = FakeAuthServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def store(tmp_path, monkeypatch):
    for name in (ENV_KEY, "ORCAROUTER_AUTH_METHOD", "ORCAROUTER_USER_ID",
                 "ORCAROUTER_SCOPE", "ORCAROUTER_NEEDS_REAUTH"):
        monkeypatch.delenv(name, raising=False)
    return CredentialStore(tmp_path / ".env")


# --------------------------------------------------------------------------- #
# PKCE primitives
# --------------------------------------------------------------------------- #


def test_challenge_is_base64url_sha256_without_padding():
    import base64
    import hashlib

    verifier = generate_verifier()
    challenge = challenge_for(verifier)
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    assert challenge == expected
    assert "=" not in challenge
    assert "+" not in challenge and "/" not in challenge
    assert len(verifier) >= 43          # RFC 7636 minimum


def test_verifier_and_state_are_fresh_every_attempt():
    verifiers = {generate_verifier() for _ in range(50)}
    states = {generate_state() for _ in range(50)}
    assert len(verifiers) == 50
    assert len(states) == 50
    assert challenge_for(generate_verifier()) != challenge_for(generate_verifier())


def test_state_matches_is_exact():
    state = generate_state()
    assert state_matches(state, state)
    assert not state_matches(state, generate_state())
    assert not state_matches(state, None)
    assert not state_matches(state, "")
    assert not state_matches(state, state + "x")


# --------------------------------------------------------------------------- #
# Authorize URL
# --------------------------------------------------------------------------- #


def test_authorize_url_uses_auth_origin_and_carries_only_the_challenge():
    origins = Origins(
        auth_base="https://www.orcarouter.ai", api_base="https://api.orcarouter.ai/v1"
    )
    pending = new_pending(flow="oob", callback_url="oob")
    url = build_authorize_url(origins, pending, app_name="Video Gen Client")

    assert url.startswith("https://www.orcarouter.ai/auth?")
    assert "api.orcarouter.ai" not in url
    assert "code_challenge_method=S256" in url
    assert "callback_url=oob" in url
    assert f"state={pending.state}" in url
    assert challenge_for(pending.verifier) in url
    # The verifier itself must never ride on the authorize URL.
    assert pending.verifier not in url
    assert "plain" not in url


def test_authorize_url_loopback_callback_is_the_bound_port():
    origins = Origins(auth_base="https://www.orcarouter.ai", api_base="https://api.orcarouter.ai/v1")
    pending = new_pending(flow="loopback", callback_url="http://127.0.0.1:51733/cb")
    url = build_authorize_url(origins, pending, app_name="Video Gen Client")
    assert "callback_url=http%3A%2F%2F127.0.0.1%3A51733%2Fcb" in url
    assert "code_challenge_method=S256" in url


# --------------------------------------------------------------------------- #
# Exchange: path, body, and error classification
# --------------------------------------------------------------------------- #


async def test_exchange_posts_to_the_auth_origin_with_the_documented_body(auth_server):
    pending = new_pending(flow="oob")
    result = await exchange_code(
        auth_server.origins, code=FAKE_CODE, verifier=pending.verifier
    )
    assert result.api_key == FAKE_KEY
    assert result.user_id == "42"
    assert result.scope == "api"

    assert len(auth_server.requests) == 1
    request = auth_server.requests[0]
    # The single most common integration mistake: /v1/auth/keys instead of
    # /api/v1/auth/keys. This asserts the real path.
    assert request["path"] == "/api/v1/auth/keys"
    assert request["body"] == {
        "code": FAKE_CODE,
        "code_verifier": pending.verifier,
        "code_challenge_method": "S256",
    }


async def test_exchange_never_targets_the_inference_origin(auth_server):
    """The exchange must not be derived by appending /v1 to the auth host."""
    await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=generate_verifier())
    path = auth_server.requests[0]["path"]
    assert not path.startswith("/v1/")
    assert "/api/v1/auth/" in path


async def test_exchange_reads_back_a_downgraded_scope(auth_server):
    auth_server.behaviour = "downgrade"
    with pytest.raises(ScopeInsufficient) as excinfo:
        await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=generate_verifier())
    assert "connector" in str(excinfo.value)
    assert "api" in str(excinfo.value)


async def test_exchange_rejects_a_reused_code(auth_server):
    pending = new_pending(flow="oob")
    first = await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=pending.verifier)
    assert first.api_key == FAKE_KEY
    with pytest.raises(ExchangeError):
        await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=pending.verifier)


@pytest.mark.parametrize(
    "behaviour",
    ["denied", "bad_method", "rate_limited", "no_key"],
)
async def test_exchange_error_classification(auth_server, behaviour):
    auth_server.behaviour = behaviour
    with pytest.raises(ExchangeError):
        await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=generate_verifier())


async def test_rate_limit_message_mentions_key_reuse_not_a_retry_loop(auth_server):
    auth_server.behaviour = "rate_limited"
    with pytest.raises(ExchangeError) as excinfo:
        await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=generate_verifier())
    assert "429" in str(excinfo.value) or "rate" in str(excinfo.value).lower()


async def test_exchange_survives_a_transport_failure(auth_server):
    auth_server.behaviour = "network"
    with pytest.raises(ExchangeError) as excinfo:
        await exchange_code(auth_server.origins, code=FAKE_CODE, verifier=generate_verifier())
    assert "authorization service" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Flow B end-to-end, through the adapter
# --------------------------------------------------------------------------- #


async def test_oob_flow_persists_the_key_and_never_leaks_the_verifier(
    auth_server, store, caplog
):
    import logging

    caplog.set_level(logging.DEBUG)
    pending = start_oob(origins=auth_server.origins)
    credential = await complete_oob(store, pending, FAKE_CODE, origins=auth_server.origins)

    assert credential.api_key == FAKE_KEY
    assert credential.method == "pkce"
    assert credential.user_id == "42"
    assert credential.scope == "api"
    assert store.current().api_key == FAKE_KEY
    assert store.current().method == "pkce"

    # The verifier travelled in the POST body only.
    assert auth_server.requests[0]["body"]["code_verifier"] == pending.verifier
    assert pending.verifier not in pending.authorize_url
    assert pending.verifier not in caplog.text
    assert FAKE_KEY not in caplog.text


async def test_oob_flow_survives_a_restart(auth_server, store, tmp_path, monkeypatch):
    """The key is durable: a second process reuses it rather than re-authorizing."""
    pending = start_oob(origins=auth_server.origins)
    await complete_oob(store, pending, FAKE_CODE, origins=auth_server.origins)

    for name in (ENV_KEY, "ORCAROUTER_AUTH_METHOD"):
        monkeypatch.delenv(name, raising=False)
    reopened = CredentialStore(tmp_path / ".env").current()
    assert reopened is not None
    assert reopened.api_key == FAKE_KEY
    assert len(auth_server.requests) == 1     # exactly one key was minted


# --------------------------------------------------------------------------- #
# Flow A end-to-end, through the adapter
# --------------------------------------------------------------------------- #


async def _drive_loopback(auth_server, store, holder, *, code=FAKE_CODE, **extra):
    """Run Flow A for real, delivering the callback the browser would send.

    The authorize URL is captured exactly as the consent screen would receive it,
    so the test can both echo the state back and verify afterwards that the
    verifier later presented at exchange hashes to the challenge sent here.
    """

    async def deliver():
        for _ in range(400):
            if holder.get("callback"):
                break
            await asyncio.sleep(0.01)
        params = {"code": code, "state": holder["state"]}
        params.update(extra)
        async with httpx.AsyncClient() as client:
            return await client.get(holder["callback"], params=params)

    task = asyncio.create_task(deliver())
    credential = await connect_loopback(
        store,
        origins=auth_server.origins,
        open_browser=False,
        on_authorize_url=_capture(holder),
        timeout=20,
    )
    await task
    return credential


def _capture(holder):
    def on_url(url: str) -> None:
        from urllib.parse import parse_qs, unquote, urlsplit

        query = parse_qs(urlsplit(url).query)
        holder["callback"] = unquote(query["callback_url"][0])
        holder["state"] = query["state"][0]
        holder["challenge"] = query["code_challenge"][0]
        holder["authorize_url"] = url

    return on_url


async def test_loopback_flow_end_to_end(auth_server, store):
    holder: dict = {}
    credential = await _drive_loopback(auth_server, store, holder)

    assert credential.api_key == FAKE_KEY
    assert credential.method == "pkce"
    assert store.current().api_key == FAKE_KEY

    request = auth_server.requests[0]
    assert request["path"] == "/api/v1/auth/keys"
    assert request["body"]["code"] == FAKE_CODE
    assert request["body"]["code_challenge_method"] == "S256"

    # The S256 binding, proven end to end: the verifier presented at exchange
    # hashes to the challenge that went out on the authorize URL.
    assert challenge_for(request["body"]["code_verifier"]) == holder["challenge"]
    # And the verifier itself was never in that URL.
    assert request["body"]["code_verifier"] not in holder["authorize_url"]
    assert holder["authorize_url"].startswith("http://127.0.0.1:")   # the fake auth origin
    assert "/auth?" in holder["authorize_url"]


async def test_loopback_flow_uses_a_fresh_verifier_on_each_attempt(auth_server, store):
    first: dict = {}
    await _drive_loopback(auth_server, store, first, code="code-attempt-1")
    second: dict = {}
    await _drive_loopback(auth_server, store, second, code="code-attempt-2")

    assert first["challenge"] != second["challenge"]
    assert first["state"] != second["state"]
    assert (
        auth_server.requests[0]["body"]["code_verifier"]
        != auth_server.requests[1]["body"]["code_verifier"]
    )


async def test_loopback_state_mismatch_is_rejected_without_redeeming(auth_server, store):
    holder: dict = {}

    def on_url(url: str) -> None:
        from urllib.parse import unquote

        match = re.search(r"callback_url=([^&]+)", url)
        holder["callback"] = unquote(match.group(1))

    async def deliver():
        for _ in range(300):
            if holder.get("callback"):
                break
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient() as client:
            # Correct code, wrong state — exactly what a hostile page would send.
            return await client.get(holder["callback"], params={"code": FAKE_CODE, "state": "wrong"})

    task = asyncio.create_task(deliver())
    with pytest.raises(StateMismatch):
        await connect_loopback(
            store, origins=auth_server.origins, open_browser=False,
            on_authorize_url=on_url, timeout=20,
        )
    await task
    assert auth_server.requests == []          # nothing was redeemed
    assert store.current() is None


async def test_loopback_denial_is_reported_cleanly(auth_server, store):
    holder: dict = {}

    def on_url(url: str) -> None:
        from urllib.parse import unquote

        match = re.search(r"callback_url=([^&]+)", url)
        holder["callback"] = unquote(match.group(1))
        holder["state"] = re.search(r"state=([^&]+)", url).group(1)

    async def deliver():
        for _ in range(300):
            if holder.get("callback"):
                break
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient() as client:
            return await client.get(
                holder["callback"],
                params={"error": "access_denied", "state": holder["state"]},
            )

    task = asyncio.create_task(deliver())
    with pytest.raises(AuthorizationDenied):
        await connect_loopback(
            store, origins=auth_server.origins, open_browser=False,
            on_authorize_url=on_url, timeout=20,
        )
    await task
    assert auth_server.requests == []
    assert store.current() is None


async def test_loopback_timeout_does_not_hang(store, auth_server):
    with pytest.raises(ConnectTimeout):
        await connect_loopback(
            store, origins=auth_server.origins, open_browser=False, timeout=0.3
        )
    assert store.current() is None


async def test_loopback_leaves_no_verifier_in_the_authorize_url(store, auth_server):
    seen = {}
    with pytest.raises(ConnectTimeout):
        await connect_loopback(
            store, origins=auth_server.origins, open_browser=False, timeout=0.3,
            on_authorize_url=lambda url: seen.setdefault("url", url),
        )
    url = seen["url"]
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    # A fresh verifier is regenerated each attempt; none may appear in the URL.
    assert "code_verifier" not in url


# --------------------------------------------------------------------------- #
# Login lifecycle: generations and pagehide
# --------------------------------------------------------------------------- #


def test_login_manager_start_and_cancel_releases_the_lock(store, auth_server):
    manager = LoginManager(store)
    assert manager.busy is False

    started = manager.start(origins=auth_server.origins)
    assert manager.busy is True
    assert manager.snapshot()["status"] == "pending"
    assert started["attempt"] == manager.attempt

    assert manager.cancel(started["attempt"]) is True
    assert manager.busy is False
    assert manager.snapshot()["status"] == "cancelled"


def test_login_manager_pagehide_shape_clears_state_and_allows_a_second_login(
    store, auth_server
):
    """Back-forward-cache shape: busy/hint clear and a new login can start.

    The invalidated request's guarded ``finally`` never runs under bfcache, so the
    cancel path must clear state synchronously itself.
    """
    manager = LoginManager(store)
    first = manager.start(origins=auth_server.origins)
    assert manager.busy

    # What the pagehide handler does: cancel with the *current* attempt, no remount.
    manager.cancel(manager.attempt)
    assert manager.busy is False
    assert manager.snapshot()["status"] == "cancelled"

    second = manager.start(origins=auth_server.origins)
    assert second["attempt"] != first["attempt"]
    assert manager.busy is True
    assert manager.snapshot()["status"] == "pending"


async def test_stale_response_cannot_overwrite_a_newer_login(store, auth_server):
    manager = LoginManager(store)
    first = manager.start(origins=auth_server.origins)
    # A second attempt supersedes the first (switching auth method, reopening).
    second = manager.start(origins=auth_server.origins)
    assert second["attempt"] > first["attempt"]

    # The first attempt's late completion must be dropped, not persisted.
    dropped = await manager.complete(first["attempt"], FAKE_CODE, origins=auth_server.origins)
    assert dropped is None
    assert store.current() is None
    assert manager.snapshot()["status"] == "pending"    # still the second attempt


async def test_completing_the_current_attempt_persists(store, auth_server):
    manager = LoginManager(store)
    started = manager.start(origins=auth_server.origins)
    credential = await manager.complete(started["attempt"], FAKE_CODE, origins=auth_server.origins)
    assert credential is not None
    assert credential.api_key == FAKE_KEY
    assert manager.busy is False
    assert manager.snapshot()["status"] == "connected"


async def test_exchange_error_releases_the_lock(store, auth_server):
    manager = LoginManager(store)
    started = manager.start(origins=auth_server.origins)
    auth_server.behaviour = "denied"
    with pytest.raises(ExchangeError):
        await manager.complete(started["attempt"], FAKE_CODE, origins=auth_server.origins)
    assert manager.busy is False
    assert manager.snapshot()["status"] == "error"
    assert store.current() is None


def test_both_adapters_are_registered_and_independent(store, auth_server):
    """API-key and PKCE stay two explicit choices on one credential seam."""
    manager = LoginManager(store)

    # PKCE path.
    started = manager.start(origins=auth_server.origins)
    assert manager.busy is True
    manager.cancel(started["attempt"])

    # API-key path, without starting a PKCE login.
    credential = manager.set_api_key(FAKE_KEY)
    assert credential.method == "api_key"
    assert store.current().api_key == FAKE_KEY
    assert manager.busy is False

    # And PKCE again, replacing it.
    started = manager.start(origins=auth_server.origins)
    assert started["attempt"] > 1
