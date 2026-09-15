"""The FastAPI surface: dropdown options, both auth choices, and key containment.

The web layer is where the "do not expose the key to the browser" rule is easiest
to break, so the central assertion here is that no response body ever contains a
credential — only its masked form.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from video_gen import web
from video_gen.orcarouter.catalog import Catalog, parse_models
from video_gen.orcarouter.connect import LoginManager
from video_gen.orcarouter.credentials import ENV_KEY, CredentialStore

FAKE_KEY = "sk-orca-testonly-abcdefghijklmnop"

CATALOG = Catalog(
    models=parse_models(
        {
            "data": [
                {"id": "kling/kling-v3", "supported_endpoint_types": ["openai-video"]},
                {"id": "kling/kling-v2-6", "supported_endpoint_types": ["openai-video"]},
                {
                    "id": "minimax/minimax-h3",
                    "supported_endpoint_types": ["openai-video"],
                    "architecture": {
                        "input_modalities": ["text", "image", "video", "audio"],
                        "output_modalities": ["video"],
                    },
                },
                {
                    "id": "vendor/chat-vision",
                    "supported_endpoint_types": ["openai"],
                    "architecture": {"input_modalities": ["text", "image"]},
                },
            ]
        }
    ),
    source="live",
)

DEGRADED = Catalog(models=(), source="seed", degraded=True, detail="catalog HTTP 503")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    for name in ("ORCAROUTER_AUTH_METHOD", "ORCAROUTER_USER_ID", "ORCAROUTER_SCOPE",
                 "ORCAROUTER_NEEDS_REAUTH"):
        monkeypatch.delenv(name, raising=False)

    store = CredentialStore(tmp_path / ".env")
    monkeypatch.setattr(web, "CredentialStore", lambda *a, **k: store)
    monkeypatch.setattr(web, "LOGIN", LoginManager(store))
    monkeypatch.setattr(web, "CLIENT", None)
    monkeypatch.setattr(
        "video_gen.providers.orcarouter.discover",
        lambda **kwargs: CATALOG,
    )
    with TestClient(web.app) as test_client:
        web.LOGIN = LoginManager(store)
        yield test_client, store


# --------------------------------------------------------------------------- #
# Model dropdown options
# --------------------------------------------------------------------------- #


def test_providers_lists_orcarouter_as_a_named_first_class_provider(client):
    test_client, _ = client
    body = test_client.get("/api/providers").json()
    by_name = {p["name"]: p for p in body["providers"]}
    assert "orcarouter" in by_name
    entry = by_name["orcarouter"]
    assert entry["label"] == "OrcaRouter"
    assert entry["discovered"] is True
    assert entry["catalog"]["source"] == "live"


def test_video_options_for_a_text_request_come_from_the_catalog(client):
    test_client, _ = client
    body = test_client.get("/api/models", params={"provider": "orcarouter"}).json()
    assert sorted(o["id"] for o in body["models"]) == ["kling/kling-v2-6", "kling/kling-v3", "minimax/minimax-h3"]
    # The vision chat model must not appear in a video dropdown.
    assert "vendor/chat-vision" not in [o["id"] for o in body["models"]]


def test_options_become_a_strict_subset_once_an_image_is_attached(client):
    test_client, _ = client
    text_options = test_client.get(
        "/api/models", params={"provider": "orcarouter", "image": "false"}
    ).json()["models"]
    image_options = test_client.get(
        "/api/models", params={"provider": "orcarouter", "image": "true"}
    ).json()["models"]

    assert {o["id"] for o in image_options} < {o["id"] for o in text_options}
    # Only the model that explicitly declares image input survives.
    assert [o["id"] for o in image_options] == ["minimax/minimax-h3"]


def test_options_are_never_free_text_and_are_never_empty_for_a_live_catalog(client):
    test_client, _ = client
    for image in ("false", "true"):
        options = test_client.get(
            "/api/models", params={"provider": "orcarouter", "image": image}
        ).json()["models"]
        assert options, f"empty dropdown for image={image}"
        for option in options:
            assert "/" in option["id"]           # vendor namespace preserved
            assert set(option) >= {"id", "input_modalities", "endpoints"}


def test_other_providers_keep_their_existing_static_options(client):
    test_client, _ = client
    body = test_client.get("/api/models", params={"provider": "wan"}).json()
    assert [o["id"] for o in body["models"]] == ["2.1", "2.1-plus", "2.2", "i2v"]
    assert body["catalog"]["source"] == "static"


def test_degraded_catalog_is_labelled_and_falls_back_to_verified_seed(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: DEGRADED)

    body = test_client.get("/api/models", params={"provider": "orcarouter"}).json()
    assert body["catalog"]["degraded"] is True
    assert body["catalog"]["source"].startswith("seed")
    # Still a real, filtered option list — never free text, never empty.
    assert body["models"]
    assert "minimax/minimax-h3" in [o["id"] for o in body["models"]]


def test_degraded_image_options_still_fail_closed(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: DEGRADED)
    options = test_client.get(
        "/api/models", params={"provider": "orcarouter", "image": "true"}
    ).json()["models"]
    assert [o["id"] for o in options] == ["minimax/minimax-h3"]   # the image-declaring seed entry


def test_a_live_catalog_with_no_video_models_does_not_substitute_the_seed(client, monkeypatch):
    """Live discovery is authoritative: an empty entitlement is reported, not faked.

    Substituting the seed here would advertise models the key cannot call, which
    is exactly what the authenticated catalog exists to prevent.
    """
    test_client, _ = client
    empty_live = Catalog(models=(), source="live", degraded=False)
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: empty_live)

    body = test_client.get("/api/models", params={"provider": "orcarouter"}).json()
    assert body["models"] == []
    assert body["catalog"]["degraded"] is False
    assert "not entitled" in body["catalog"]["detail"]


def test_a_live_catalog_with_chat_only_models_offers_no_video_options(client, monkeypatch):
    """The real shape of this key's entitlement: chat models, zero video models."""
    test_client, _ = client
    chat_only = Catalog(
        models=parse_models(
            {
                "data": [
                    {
                        "id": "vendor/chat-only",
                        "supported_endpoint_types": ["openai", "openai-response"],
                        "architecture": {"input_modalities": ["text", "image"]},
                    }
                ]
            }
        ),
        source="live",
    )
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: chat_only)

    body = test_client.get("/api/models", params={"provider": "orcarouter"}).json()
    assert body["models"] == []
    assert body["catalog"]["catalog_model_count"] == 1
    assert "not entitled" in body["catalog"]["detail"]


def test_unknown_provider_is_rejected(client):
    test_client, _ = client
    assert test_client.get("/api/models", params={"provider": "nope"}).status_code == 400


# --------------------------------------------------------------------------- #
# Generation guard on the server side too
# --------------------------------------------------------------------------- #


def test_generate_rejects_an_image_with_an_incompatible_model(client):
    test_client, store = client
    store.save(FAKE_KEY, method="api_key")
    response = test_client.post(
        "/api/generate",
        json={
            "provider": "orcarouter",
            "prompt": "hello",
            "model": "kling/kling-v3",
            "image_url": "https://img/a.png",
        },
    )
    assert response.status_code == 400
    assert "cannot accept an image input" in response.json()["detail"]


def test_generate_rejects_an_unconfigured_provider(client):
    test_client, _ = client
    response = test_client.post(
        "/api/generate", json={"provider": "orcarouter", "prompt": "hello"}
    )
    assert response.status_code == 400
    assert "not configured" in response.json()["detail"]


def test_generate_reports_needs_reauth_instead_of_calling_a_dead_key(client):
    test_client, store = client
    store.save(FAKE_KEY, method="pkce")
    store.mark_needs_reauth(store.generation)
    response = test_client.post(
        "/api/generate", json={"provider": "orcarouter", "prompt": "hello"}
    )
    assert response.status_code == 400
    assert "rejected" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Both authentication choices
# --------------------------------------------------------------------------- #


def test_auth_state_exposes_only_masked_credential_metadata(client):
    test_client, store = client
    assert test_client.get("/api/orcarouter/auth").json()["has_credential"] is False

    store.save(FAKE_KEY, method="pkce", user_id="42", scope="api")
    body = test_client.get("/api/orcarouter/auth").json()
    assert body["has_credential"] is True
    assert body["method"] == "pkce"
    assert body["user_id"] == "42"
    assert body["scope"] == "api"
    assert FAKE_KEY not in json.dumps(body)
    assert body["masked"] != FAKE_KEY


def test_api_key_choice_saves_through_the_web_endpoint(client):
    test_client, store = client
    response = test_client.post("/api/orcarouter/auth/api-key", json={"api_key": FAKE_KEY})
    assert response.status_code == 200
    assert response.json()["method"] == "api_key"
    assert store.current().api_key == FAKE_KEY
    assert FAKE_KEY not in response.text          # only the masked form is returned


def test_api_key_choice_rejects_an_obviously_wrong_value(client):
    test_client, store = client
    response = test_client.post("/api/orcarouter/auth/api-key", json={"api_key": "nope"})
    assert response.status_code == 400
    assert store.current() is None


def test_pkce_choice_returns_an_authorize_url_with_only_the_challenge(client):
    test_client, _ = client
    body = test_client.post("/api/orcarouter/auth/pkce/start").json()

    url = body["authorize_url"]
    assert url.startswith("https://www.orcarouter.ai/auth?")
    assert "code_challenge_method=S256" in url
    assert "callback_url=oob" in url            # Flow B: no predictable address
    assert "code_verifier" not in url
    assert body["flow"] == "oob"
    assert body["attempt"] >= 1
    assert web.LOGIN.busy is True


def test_cancel_releases_the_server_lock(client):
    test_client, _ = client
    started = test_client.post("/api/orcarouter/auth/pkce/start").json()
    assert web.LOGIN.busy is True

    body = test_client.post("/api/orcarouter/auth/cancel", json={"attempt": started["attempt"]}).json()
    assert body["busy"] is False
    assert web.LOGIN.busy is False


def test_pagehide_cancel_with_a_null_attempt_clears_the_lock(client):
    """What the browser sends on pagehide: no attempt id, must still release."""
    test_client, _ = client
    test_client.post("/api/orcarouter/auth/pkce/start")
    assert web.LOGIN.busy is True

    body = test_client.post("/api/orcarouter/auth/cancel", json={"attempt": None}).json()
    assert body["busy"] is False
    assert body["status"] == "cancelled"


def test_a_second_login_can_start_after_a_pagehide_cancel(client):
    """The bfcache shape: no remount, but a new attempt is accepted immediately."""
    test_client, _ = client
    first = test_client.post("/api/orcarouter/auth/pkce/start").json()
    test_client.post("/api/orcarouter/auth/cancel", json={"attempt": None})

    second = test_client.post("/api/orcarouter/auth/pkce/start").json()
    assert second["attempt"] > first["attempt"]
    assert web.LOGIN.busy is True
    assert second["authorize_url"] != first["authorize_url"]


def test_a_superseded_attempt_cannot_complete(client):
    test_client, store = client
    first = test_client.post("/api/orcarouter/auth/pkce/start").json()
    test_client.post("/api/orcarouter/auth/pkce/start")      # supersedes

    response = test_client.post(
        "/api/orcarouter/auth/pkce/complete", json={"attempt": first["attempt"], "code": "c"}
    )
    assert response.status_code == 409
    assert store.current() is None


def test_logout_clears_the_credential(client):
    test_client, store = client
    store.save(FAKE_KEY, method="api_key")
    body = test_client.post("/api/orcarouter/auth/logout").json()
    assert body["has_credential"] is False
    assert store.current() is None


def test_switching_auth_method_keeps_both_available(client):
    """Neither choice may disable the other."""
    test_client, store = client
    test_client.post("/api/orcarouter/auth/api-key", json={"api_key": FAKE_KEY})
    assert web.LOGIN.snapshot()["method"] == "api_key"

    started = test_client.post("/api/orcarouter/auth/pkce/start").json()
    assert started["attempt"] >= 1
    # The API key is untouched while a PKCE login is in flight.
    assert store.current().api_key == FAKE_KEY
    assert store.current().method == "api_key"


# --------------------------------------------------------------------------- #
# The key must never reach the browser
# --------------------------------------------------------------------------- #


def test_no_endpoint_ever_returns_the_raw_key(client):
    test_client, store = client
    store.save(FAKE_KEY, method="pkce", user_id="7", scope="api")

    for path in (
        "/",
        "/api/providers",
        "/api/models?provider=orcarouter",
        "/api/orcarouter/auth",
        "/api/tasks",
    ):
        response = test_client.get(path)
        assert FAKE_KEY not in response.text, path

    for path, payload in (
        ("/api/orcarouter/auth/cancel", {"attempt": None}),
        ("/api/orcarouter/auth/logout", None),
    ):
        response = test_client.post(path, json=payload) if payload else test_client.post(path)
        assert FAKE_KEY not in response.text, path


def test_index_page_offers_both_auth_choices_and_no_secret(client):
    test_client, _ = client
    html = test_client.get("/").text
    # Two distinct, labelled entries on the same credential seam.
    assert "OrcaRouter - API" in html
    assert "OrcaRouter - Auth" in html
    assert "Connect with OrcaRouter" in html
    assert "console/token" in html
    # The pagehide contract is wired in the client.
    assert "pagehide" in html
    assert "keepalive" in html
    assert "sk-orca-testonly" not in html
