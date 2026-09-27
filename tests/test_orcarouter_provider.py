"""The OrcaRouter provider adapter, including the terminal-401 path."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from video_gen.config import Config
from video_gen.exceptions import APIError, ProviderNotConfigured, ProviderReauthRequired
from video_gen.models import VideoRequest
from video_gen.orcarouter.catalog import Catalog, parse_models
from video_gen.orcarouter.credentials import ENV_KEY, CredentialStore
from video_gen.providers.orcarouter import (
    DEFAULT_VIDEO_MODEL,
    OrcaRouterProvider,
    build_body,
    namespace_of,
)
from video_gen.providers.registry import PROVIDERS, get_provider_class

FAKE_KEY = "sk-orca-testonly-abcdefghijklmnop"

CATALOG = Catalog(
    models=parse_models(
        {
            "data": [
                {"id": "kling/kling-v3", "supported_endpoint_types": ["openai-video"]},
                {
                    "id": "minimax/minimax-h3",
                    "supported_endpoint_types": ["openai-video"],
                    "architecture": {
                        "input_modalities": ["text", "image", "video", "audio"],
                        "output_modalities": ["video"],
                    },
                },
            ]
        }
    ),
    source="live",
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.delenv("ORCAROUTER_NEEDS_REAUTH", raising=False)
    monkeypatch.delenv("ORCAROUTER_AUTH_METHOD", raising=False)
    s = CredentialStore(tmp_path / ".env")
    s.save(FAKE_KEY, method="api_key", scope="api")
    return s


def make_provider(store, handler, monkeypatch, catalog=CATALOG):
    """A provider whose catalog and HTTP transport are both deterministic."""
    monkeypatch.setattr(
        "video_gen.providers.orcarouter.discover", lambda **kwargs: catalog
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OrcaRouterProvider(Config(), client=client, store=store)


# --------------------------------------------------------------------------- #
# Registry and configuration
# --------------------------------------------------------------------------- #


def test_orcarouter_is_a_first_class_registered_provider():
    assert "orcarouter" in PROVIDERS
    assert get_provider_class("orcarouter").name == "orcarouter"
    assert get_provider_class("OrcaRouter").name == "orcarouter"


def test_is_configured_follows_the_credential(store, monkeypatch):
    monkeypatch.setattr("video_gen.providers.orcarouter.discover", lambda **k: CATALOG)

    empty = CredentialStore(Path("/nonexistent/.env"))
    monkeypatch.delenv(ENV_KEY, raising=False)
    assert OrcaRouterProvider(Config(), store=empty).is_configured() is False

    assert OrcaRouterProvider(Config(), store=store).is_configured() is True
    store.mark_needs_reauth(store.generation)
    assert OrcaRouterProvider(Config(), store=store).is_configured() is False


def test_missing_credential_names_both_ways_to_get_one(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    provider = OrcaRouterProvider(Config(), store=CredentialStore(Path("/nope/.env")))
    with pytest.raises(ProviderNotConfigured) as excinfo:
        provider._auth_headers()
    message = str(excinfo.value)
    assert "orcarouter-login" in message
    assert "console/token" in message


def test_needs_reauth_is_terminal_and_makes_no_request(store, monkeypatch):
    """A revoked durable key must not be retried and has no refresh path."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={})

    store.mark_needs_reauth(store.generation)
    provider = make_provider(store, handler, monkeypatch)

    with pytest.raises(ProviderReauthRequired):
        provider._auth_headers()
    assert calls == []           # nothing was attempted, so nothing to refresh


def test_config_origin_overrides_are_honoured(monkeypatch):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://self-hosted.test")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://self-hosted.test/v1")
    monkeypatch.delenv(ENV_KEY, raising=False)
    origins = OrcaRouterProvider(Config()).origins()
    assert origins.auth_base == "https://self-hosted.test"
    assert origins.api_base == "https://self-hosted.test/v1"


# --------------------------------------------------------------------------- #
# Submit / poll wire format
# --------------------------------------------------------------------------- #


async def test_submit_posts_to_the_video_endpoint_on_the_api_origin(store, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "task_abc",
                "task_id": "task_abc",
                "object": "video",
                "model": "kling/kling-v3",
                "status": "queued",
                "progress": 0,
            },
        )

    provider = make_provider(store, handler, monkeypatch)
    ref = await provider.submit(VideoRequest(prompt="a cat", model="kling/kling-v3"))

    assert ref.task_id == "task_abc"
    assert ref.provider == "orcarouter"
    assert seen["url"] == "https://api.orcarouter.ai/v1/video/generations"
    assert "www.orcarouter.ai" not in seen["url"]
    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    assert seen["body"]["model"] == "kling/kling-v3"      # namespace preserved
    assert seen["body"]["prompt"] == "a cat"
    assert seen["body"]["metadata"]["duration"] == "5"


async def test_status_parses_the_wrapped_uppercase_envelope(store, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/video/generations/task_abc")
        return httpx.Response(
            200,
            json={
                "code": "success",
                "message": "",
                "data": {
                    "task_id": "task_abc",
                    "status": "SUCCESS",
                    "progress": "100%",
                    "result_url": "https://cdn.example/video.mp4",
                    "action": "omniVideo",
                    "fail_reason": "",
                },
            },
        )

    provider = make_provider(store, handler, monkeypatch)
    status = await provider.get_status("task_abc")

    assert status.state == "succeeded"
    assert status.video_url == "https://cdn.example/video.mp4"
    assert status.progress == 100.0
    assert status.provider == "orcarouter"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NOT_START", "pending"),
        ("SUBMITTED", "pending"),
        ("IN_PROGRESS", "running"),
        ("SUCCESS", "succeeded"),
        ("FAILURE", "failed"),
        ("UNKNOWN", "pending"),
        ("something-new", "pending"),
    ],
)
async def test_status_state_mapping(store, monkeypatch, raw, expected):
    payload = {"data": {"status": raw, "progress": "30%"}}
    if raw == "FAILURE":
        payload["data"]["fail_reason"] = "upstream rejected"
    provider = make_provider(store, lambda request: httpx.Response(200, json=payload), monkeypatch)
    status = await provider.get_status("t")
    assert status.state == expected
    if raw == "FAILURE":
        assert status.error == "upstream rejected"
    if raw != "SUCCESS":
        assert status.video_url is None        # never leak a URL from a non-success


# --------------------------------------------------------------------------- #
# Terminal 401, generation-safely
# --------------------------------------------------------------------------- #


async def test_401_marks_the_exact_generation_and_does_not_retry(store, monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

    provider = make_provider(store, handler, monkeypatch)
    with pytest.raises(ProviderReauthRequired):
        await provider.submit(VideoRequest(prompt="x", model="kling/kling-v3"))

    assert len(calls) == 1                      # no retry, no refresh attempt
    assert store.current().needs_reauth is True


async def test_stale_401_does_not_poison_a_replaced_credential(store, monkeypatch):
    """A late failure from an old request must not flag the new credential."""
    responses = [httpx.Response(401, json={"error": {"code": "invalid_api_key"}})]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    make_provider(store, handler, monkeypatch)      # wires the provider for this store
    stale_generation = store.current().generation

    # The user signs in again before the old request's 401 is processed.
    store.save("sk-orca-testonly-zzzzzzzzzzzzzzzz", method="pkce", scope="api")
    assert store.mark_needs_reauth(stale_generation) is False
    assert store.current().needs_reauth is False
    assert store.current().api_key.endswith("zzzz")


async def test_401_during_status_polling_also_marks_reauth(store, monkeypatch):
    provider = make_provider(
        store, lambda request: httpx.Response(401, json={"error": {"code": "revoked"}}), monkeypatch
    )
    with pytest.raises(ProviderReauthRequired):
        await provider.get_status("task_abc")
    assert store.current().needs_reauth is True


# --------------------------------------------------------------------------- #
# Errors must not leak the key
# --------------------------------------------------------------------------- #


async def test_api_error_text_carries_the_provider_message_not_the_key(store, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": {
                    "code": "model_access_denied",
                    "message": "This API key does not have access to model kling/kling-v3.",
                }
            },
        )

    provider = make_provider(store, handler, monkeypatch)
    with pytest.raises(APIError) as excinfo:
        await provider.submit(VideoRequest(prompt="x", model="kling/kling-v3"))

    message = str(excinfo.value)
    assert "model_access_denied" in message
    assert FAKE_KEY not in message
    assert excinfo.value.status_code == 403


async def test_non_json_error_body_is_handled(store, monkeypatch):
    provider = make_provider(
        store, lambda request: httpx.Response(502, text="<html>bad gateway</html>"), monkeypatch
    )
    with pytest.raises(APIError) as excinfo:
        await provider.submit(VideoRequest(prompt="x", model="kling/kling-v3"))
    assert "502" in str(excinfo.value)
    assert FAKE_KEY not in str(excinfo.value)


async def test_submit_without_a_task_id_is_an_error(store, monkeypatch):
    provider = make_provider(store, lambda request: httpx.Response(200, json={}), monkeypatch)
    with pytest.raises(APIError):
        await provider.submit(VideoRequest(prompt="x", model="kling/kling-v3"))


# --------------------------------------------------------------------------- #
# Model selection against the filtered catalog
# --------------------------------------------------------------------------- #


def test_default_model_comes_from_the_live_catalog(store, monkeypatch):
    provider = make_provider(store, lambda r: httpx.Response(200, json={}), monkeypatch)
    assert provider.resolve_model(None) == "kling/kling-v3"


def test_text_request_falls_back_to_the_default_when_the_catalog_is_empty(store, monkeypatch):
    empty = Catalog(models=(), source="seed", degraded=True)
    provider = make_provider(store, lambda r: httpx.Response(200, json={}), monkeypatch, catalog=empty)
    assert provider.resolve_model(None) == DEFAULT_VIDEO_MODEL


def test_image_request_may_not_use_a_model_without_image_modalities(store, monkeypatch):
    """Fail closed: kling/kling-v3 publishes no modalities, so it cannot do I2V."""
    provider = make_provider(store, lambda r: httpx.Response(200, json={}), monkeypatch)
    assert provider.resolve_model(None, requires_image=True) == "minimax/minimax-h3"
    with pytest.raises(APIError) as excinfo:
        provider.resolve_model("kling/kling-v3", requires_image=True)
    assert "image" in str(excinfo.value)


def test_unknown_model_id_is_rejected_with_the_valid_options(store, monkeypatch):
    provider = make_provider(store, lambda r: httpx.Response(200, json={}), monkeypatch)
    with pytest.raises(APIError) as excinfo:
        provider.resolve_model("not-a-real-model")
    assert "not a video model" in str(excinfo.value)
    assert "kling/kling-v3" in str(excinfo.value)


async def test_image_request_never_reaches_http_for_an_incompatible_model(store, monkeypatch):
    calls = []
    provider = make_provider(
        store, lambda r: calls.append(r) or httpx.Response(200, json={}), monkeypatch
    )
    with pytest.raises(APIError):
        await provider.submit(
            VideoRequest(prompt="x", model="kling/kling-v3", image_url="https://img/x.png")
        )
    assert calls == []


async def test_cancel_reports_honestly(store, monkeypatch):
    """There is no documented cancel endpoint; do not pretend otherwise."""
    provider = make_provider(store, lambda r: httpx.Response(200, json={}), monkeypatch)
    assert await provider.cancel("task_abc") is False


# --------------------------------------------------------------------------- #
# Request-body variants per model namespace
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model", "expected_namespace"),
    [("kling/kling-v3", "kling"), ("byteplus/dreamina-seedance-2-0-260128", "byteplus"),
     ("minimax/minimax-h3", "minimax"), ("orcarouter/auto", None)],
)
def test_namespace_detection(model, expected_namespace):
    assert namespace_of(model) == expected_namespace


def test_kling_body_shape():
    body = build_body(
        "kling/kling-v3",
        VideoRequest(prompt="p", duration=5, aspect_ratio="16:9", resolution="1080p"),
        requires_image=False,
    )
    assert body["model"] == "kling/kling-v3"
    assert body["metadata"]["mode"] == "pro"
    assert body["metadata"]["aspect_ratio"] == "16:9"
    assert body["metadata"]["duration"] == "5"
    assert "image" not in body


def test_kling_image_body_uses_the_image_field():
    body = build_body(
        "kling/kling-v3",
        VideoRequest(prompt="p", image_url="https://img/a.png"),
        requires_image=True,
    )
    assert body["image"] == "https://img/a.png"
    assert body["metadata"]["mode"] == "std"


def test_seedance_body_shape():
    body = build_body(
        "byteplus/dreamina-seedance-2-0-260128",
        VideoRequest(prompt="p", duration=8, aspect_ratio="9:16", resolution="720p"),
        requires_image=False,
    )
    assert body["metadata"]["ratio"] == "9:16"
    assert body["metadata"]["resolution"] == "720p"
    assert body["metadata"]["duration"] == "8"
    assert "mode" not in body           # not a Seedance field


def test_seedance_image_body_uses_the_content_role():
    body = build_body(
        "byteplus/dreamina-seedance-2-0-260128",
        VideoRequest(prompt="p", image_url="https://img/a.png"),
        requires_image=True,
    )
    content = body["metadata"]["content"]
    assert content[0]["role"] == "first_frame"
    assert content[0]["image_url"]["url"] == "https://img/a.png"


def test_minimax_body_shape():
    body = build_body(
        "minimax/minimax-h3",
        VideoRequest(prompt="p", duration=6, aspect_ratio="1:1", resolution="1080p"),
        requires_image=False,
    )
    assert body["duration"] == 6
    assert body["size"] == "1080x1080"
    assert body["metadata"]["ratio"] == "1:1"


def test_minimax_image_body_sets_both_image_fields():
    body = build_body(
        "minimax/minimax-h3",
        VideoRequest(prompt="p", image_url="https://img/a.png"),
        requires_image=True,
    )
    assert body["image"] == "https://img/a.png"
    assert body["metadata"]["first_frame_image"] == "https://img/a.png"


async def test_submit_merges_caller_extra_into_metadata(store, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"task_id": "t"})

    provider = make_provider(store, handler, monkeypatch)
    request = VideoRequest(
        prompt="p",
        model="kling/kling-v3",
        negative_prompt="blurry",
        extra={"metadata": {"sound": "on"}},
    )
    await provider.submit(request)
    assert seen["body"]["metadata"]["sound"] == "on"
    assert seen["body"]["metadata"]["negative_prompt"] == "blurry"
