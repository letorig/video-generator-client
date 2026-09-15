"""Origin policy and the two-adapter equivalence of the credential seam."""

from __future__ import annotations

import httpx
import pytest

from video_gen.config import Config
from video_gen.models import VideoRequest
from video_gen.orcarouter.catalog import Catalog, parse_models
from video_gen.orcarouter.credentials import (
    ENV_KEY,
    METHOD_API_KEY,
    METHOD_PKCE,
    ApiKeyAdapter,
    Credential,
    CredentialStore,
)
from video_gen.orcarouter.endpoints import (
    DEFAULT_API_BASE_URL,
    DEFAULT_AUTH_BASE_URL,
    OriginError,
    resolve_origins,
)
from video_gen.providers.orcarouter import OrcaRouterProvider

FAKE_KEY_API = "sk-orca-testonly-apikey0000000000"
FAKE_KEY_PKCE = "sk-orca-testonly-pkce000000000000"

CATALOG = Catalog(
    models=parse_models(
        {"data": [{"id": "kling/kling-v3", "supported_endpoint_types": ["openai-video"]}]}
    ),
    source="live",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (ENV_KEY, "ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL",
                 "ORCAROUTER_AUTH_METHOD", "ORCAROUTER_NEEDS_REAUTH"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Origins
# --------------------------------------------------------------------------- #


def test_public_defaults_are_two_distinct_origins():
    origins = resolve_origins()
    assert origins.auth_base == "https://www.orcarouter.ai"
    assert origins.api_base == "https://api.orcarouter.ai/v1"
    assert origins.auth_base != origins.api_base


def test_authorize_and_exchange_paths_are_on_the_auth_origin():
    origins = resolve_origins()
    assert origins.authorize_url() == "https://www.orcarouter.ai/auth"
    # The single most common integration mistake.
    assert origins.exchange_url() == "https://www.orcarouter.ai/api/v1/auth/keys"
    assert "/v1/auth/keys" not in origins.exchange_url().replace("/api/v1/auth/keys", "")


def test_the_inference_origin_never_carries_auth_paths():
    origins = resolve_origins()
    assert origins.models_url() == "https://api.orcarouter.ai/v1/models"
    for url in (origins.exchange_url(), origins.authorize_url()):
        assert "api.orcarouter.ai" not in url


def test_defaults_match_the_documented_constants():
    assert DEFAULT_AUTH_BASE_URL == "https://www.orcarouter.ai"
    assert DEFAULT_API_BASE_URL == "https://api.orcarouter.ai/v1"


def test_shared_base_is_a_fallback_for_both_origins(monkeypatch):
    monkeypatch.setenv("ORCA_BASE_URL", "https://self-hosted.example")
    origins = resolve_origins()
    assert origins.auth_base == "https://self-hosted.example"
    assert origins.api_base == "https://self-hosted.example/v1"


def test_explicit_overrides_beat_the_shared_base(monkeypatch):
    monkeypatch.setenv("ORCA_BASE_URL", "https://shared.example")
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://auth.example")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api.example/v1")
    origins = resolve_origins()
    assert origins.auth_base == "https://auth.example"
    assert origins.api_base == "https://api.example/v1"
    assert origins.exchange_url() == "https://auth.example/api/v1/auth/keys"


def test_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://env.example")
    origins = resolve_origins(auth_base_url="https://arg.example")
    assert origins.auth_base == "https://arg.example"


def test_one_origin_is_never_derived_from_the_other_by_hostname_swap(monkeypatch):
    """Overriding only the auth origin must not move inference, or vice versa."""
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://auth-only.example")
    assert resolve_origins().api_base == DEFAULT_API_BASE_URL

    monkeypatch.delenv("ORCA_AUTH_BASE_URL")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api-only.example/v1")
    assert resolve_origins().auth_base == DEFAULT_AUTH_BASE_URL


@pytest.mark.parametrize(
    "value",
    ["http://evil.example", "http://orcarouter.ai", "ftp://x.example", "not-a-url", "https://"],
)
def test_remote_origins_must_be_https(monkeypatch, value):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", value)
    with pytest.raises(OriginError):
        resolve_origins()


@pytest.mark.parametrize("value", ["http://127.0.0.1:8080", "http://localhost:9000", "http://[::1]:1234"])
def test_loopback_http_is_permitted_for_development(monkeypatch, value):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", value)
    assert resolve_origins().auth_base == value


def test_trailing_slashes_are_normalised():
    origins = resolve_origins(auth_base_url="https://a.example/", api_base_url="https://b.example/v1/")
    assert origins.auth_base == "https://a.example"
    assert origins.api_base == "https://b.example/v1"
    assert origins.models_url() == "https://b.example/v1/models"


def test_provider_reads_its_origins_from_config(monkeypatch):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://auth.example")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api.example/v1")
    origins = OrcaRouterProvider(Config()).origins()
    assert origins.authorize_url() == "https://auth.example/auth"
    assert origins.video_generations_url() == "https://api.example/v1/video/generations"


# --------------------------------------------------------------------------- #
# Two adapters, one credential result
# --------------------------------------------------------------------------- #


def test_api_key_and_pkce_adapters_yield_the_same_credential_type(tmp_path, monkeypatch):
    """The seam requirement: same result type, same downstream, different method."""
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: CATALOG)

    api_store = CredentialStore(tmp_path / "api.env")
    via_key = ApiKeyAdapter(api_store).save(FAKE_KEY_API)

    pkce_store = CredentialStore(tmp_path / "pkce.env")
    via_pkce = pkce_store.save(FAKE_KEY_PKCE, method=METHOD_PKCE, user_id="9", scope="api")

    assert isinstance(via_key, Credential) and isinstance(via_pkce, Credential)
    assert via_key.method == METHOD_API_KEY
    assert via_pkce.method == METHOD_PKCE
    # Everything except provenance is identical in shape.
    assert set(via_key.__dataclass_fields__) == set(via_pkce.__dataclass_fields__)
    assert via_key.usable and via_pkce.usable


async def test_downstream_does_not_care_which_adapter_produced_the_credential(
    tmp_path, monkeypatch
):
    """The provider sends the same request either way."""
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: CATALOG)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("Authorization")))
        return httpx.Response(200, json={"task_id": "t"})

    api_store = CredentialStore(tmp_path / "api.env")
    ApiKeyAdapter(api_store).save(FAKE_KEY_API)
    pkce_store = CredentialStore(tmp_path / "pkce.env")
    pkce_store.save(FAKE_KEY_PKCE, method=METHOD_PKCE, scope="api")

    for store, expected_key in ((api_store, FAKE_KEY_API), (pkce_store, FAKE_KEY_PKCE)):
        # Two separate processes: each reads only its own stored credential.
        monkeypatch.delenv(ENV_KEY, raising=False)
        provider = OrcaRouterProvider(
            Config(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            store=store,
        )
        await provider.submit(VideoRequest(prompt="x", model="kling/kling-v3"))
        assert store.current().api_key == expected_key

    assert len(seen) == 2
    assert seen[0][0] == seen[1][0] == "https://api.orcarouter.ai/v1/video/generations"
    assert seen[0][1] == f"Bearer {FAKE_KEY_API}"
    assert seen[1][1] == f"Bearer {FAKE_KEY_PKCE}"


def test_both_adapters_register_distinct_methods_for_the_ui(tmp_path, monkeypatch):
    """The two choices stay distinguishable wherever both can appear."""
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: CATALOG)
    store = CredentialStore(tmp_path / ".env")
    assert METHOD_API_KEY != METHOD_PKCE

    ApiKeyAdapter(store).save(FAKE_KEY_API)
    assert store.current().method == METHOD_API_KEY
    store.save(FAKE_KEY_PKCE, method=METHOD_PKCE)
    assert store.current().method == METHOD_PKCE


def test_losing_one_choice_does_not_affect_the_other(tmp_path, monkeypatch):
    """A user without a browser keeps the API-key path working."""
    store = CredentialStore(tmp_path / ".env")
    store.save(FAKE_KEY_PKCE, method=METHOD_PKCE)
    store.clear()
    assert store.current() is None
    # The API-key adapter is still fully functional after a PKCE credential is gone.
    assert ApiKeyAdapter(store).save(FAKE_KEY_API).api_key == FAKE_KEY_API
