"""OrcaRouter credential seam: API-key adapter, persistence, and lifecycle."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from video_gen.orcarouter import credentials as creds
from video_gen.orcarouter.credentials import (
    ENV_KEY,
    ENV_NEEDS_REAUTH,
    ApiKeyAdapter,
    CredentialStore,
    InvalidApiKey,
    looks_like_api_key,
    mask_secret,
    redact,
)

FAKE_KEY = "sk-orca-testonly-0000000000000000"
FAKE_KEY_2 = "sk-orca-testonly-1111111111111111"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store backed by a throwaway .env, with the real environment cleared."""
    for name in (ENV_KEY, "ORCAROUTER_AUTH_METHOD", "ORCAROUTER_USER_ID",
                 "ORCAROUTER_SCOPE", ENV_NEEDS_REAUTH):
        monkeypatch.delenv(name, raising=False)
    return CredentialStore(tmp_path / ".env")


# -- format helpers --------------------------------------------------------- #


def test_looks_like_api_key_accepts_orca_prefix():
    assert looks_like_api_key(FAKE_KEY)
    assert looks_like_api_key("  " + FAKE_KEY + "  ")


@pytest.mark.parametrize("value", ["", "sk-other-x", "orca-1234567890", "sk-orca-", "sk-orca-12"])
def test_looks_like_api_key_rejects_obvious_mistakes(value):
    assert not looks_like_api_key(value)


def test_mask_secret_never_returns_the_whole_key():
    masked = mask_secret(FAKE_KEY)
    assert masked != FAKE_KEY
    assert FAKE_KEY not in masked
    assert masked.startswith("sk-orca-")
    assert mask_secret("") == ""
    assert mask_secret("short") == "…"


def test_redact_removes_every_secret_from_text():
    text = f"failed with {FAKE_KEY} and {FAKE_KEY_2}"
    cleaned = redact(text, FAKE_KEY, FAKE_KEY_2)
    assert FAKE_KEY not in cleaned
    assert FAKE_KEY_2 not in cleaned


# -- API-key adapter: save / read / clear / mask ---------------------------- #


def test_api_key_adapter_save_read_clear(store):
    adapter = ApiKeyAdapter(store)
    assert store.current() is None
    with pytest.raises(InvalidApiKey):
        adapter.obtain()

    saved = adapter.save(FAKE_KEY)
    assert saved.method == "api_key"
    assert saved.masked == mask_secret(FAKE_KEY)
    assert saved.api_key not in repr(saved)

    reread = store.current()
    assert reread is not None
    assert reread.api_key == FAKE_KEY    # read back from the .env file
    assert adapter.obtain().api_key == FAKE_KEY

    adapter.clear()
    assert store.current() is None


def test_api_key_adapter_rejects_bad_input_without_storing(store):
    adapter = ApiKeyAdapter(store)
    with pytest.raises(InvalidApiKey):
        adapter.save("not-a-key")
    assert store.current() is None


def test_saving_preserves_unrelated_env_lines(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "# my notes\nWAN_API_KEY=abc\nDASHSCOPE_API_KEY=xyz\n\n# trailing comment\n",
        encoding="utf-8",
    )
    store = CredentialStore(path)
    store.save(FAKE_KEY, method="api_key")

    text = path.read_text(encoding="utf-8")
    assert "WAN_API_KEY=abc" in text
    assert "DASHSCOPE_API_KEY=xyz" in text
    assert "# my notes" in text
    assert f"{ENV_KEY}={FAKE_KEY}" in text

    store.clear()
    text = path.read_text(encoding="utf-8")
    assert ENV_KEY not in text
    assert "WAN_API_KEY=abc" in text


def test_credential_repr_and_str_are_masked(store):
    credential = ApiKeyAdapter(store).save(FAKE_KEY)
    assert FAKE_KEY not in repr(credential)
    assert FAKE_KEY not in str(credential)
    assert mask_secret(FAKE_KEY) in repr(credential)


# -- terminal 401 handling -------------------------------------------------- #


def test_mark_needs_reauth_marks_only_the_current_generation(store):
    first = ApiKeyAdapter(store).save(FAKE_KEY)
    assert store.mark_needs_reauth(first.generation) is True
    assert store.current().needs_reauth is True

    # Signing in again replaces the credential and clears the flag.
    second = ApiKeyAdapter(store).save(FAKE_KEY_2)
    assert second.generation != first.generation
    assert store.current().needs_reauth is False

    # A late 401 from the *old* generation must not poison the new credential.
    assert store.mark_needs_reauth(first.generation) is False
    assert store.current().needs_reauth is False
    assert store.current().api_key == FAKE_KEY_2


def test_clear_bumps_generation_so_stale_401_is_ignored(store):
    credential = ApiKeyAdapter(store).save(FAKE_KEY)
    generation = credential.generation
    store.clear()
    assert store.mark_needs_reauth(generation) is False


def test_needs_reauth_credential_is_not_usable(store):
    ApiKeyAdapter(store).save(FAKE_KEY)
    store.mark_needs_reauth(store.generation)
    credential = store.current()
    assert credential is not None
    assert credential.usable is False


# -- no fake refresh -------------------------------------------------------- #


def test_no_refresh_grant_machinery_exists():
    """A durable OrcaRouter key has no refresh token and no refresh endpoint.

    Only *code* is inspected, not prose: the docstrings deliberately explain that
    there is no refresh grant, and that is the behaviour we want documented.
    """
    import ast

    for module in (creds, __import__("video_gen.orcarouter.connect", fromlist=["x"])):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not {n for n in names if "refresh" in n.lower()}
        # No string literal may name a refresh grant type or endpoint.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                assert "refresh_token" not in lowered
                assert "grant_type" not in lowered


def test_env_variable_wins_over_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    CredentialStore(path).save(FAKE_KEY, method="api_key")
    monkeypatch.setenv(ENV_KEY, FAKE_KEY_2)
    assert CredentialStore(path).current().api_key == FAKE_KEY_2


# -- secrets stay out of logs ----------------------------------------------- #


def test_secrets_do_not_reach_logs_or_exceptions(store, caplog):
    logging.getLogger("video_gen").setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        adapter = ApiKeyAdapter(store)
        credential = adapter.save(FAKE_KEY)
        try:
            raise InvalidApiKey(f"could not use {mask_secret(credential.api_key)}")
        except InvalidApiKey as exc:
            assert FAKE_KEY not in str(exc)
            assert FAKE_KEY not in repr(exc)
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in os.environ.get("PYTEST_CURRENT_TEST", "")
