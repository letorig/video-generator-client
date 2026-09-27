"""Live checks against the real OrcaRouter service.

Skipped unless ``ORCAROUTER_API_KEY`` is present. This is the only module that is
allowed to touch a real OrcaRouter origin, and it does so exclusively through the
provider code implemented in this change: the catalog comes from
``OrcaRouterProvider.catalog()`` and the authenticated reads go through
``OrcaRouterProvider.api_get()``, both of which resolve their own origins from
the credential seam this change adds. Nothing here builds a URL, picks a host or
attaches the credential by hand — a green run therefore is evidence about the
wiring, not about ``httpx`` or about a bare ``curl``.

Everything else in the suite runs on fake credentials.
"""

from __future__ import annotations

import os

import pytest

from video_gen.config import Config
from video_gen.exceptions import APIError, ProviderReauthRequired
from video_gen.models import VideoRequest
from video_gen.orcarouter.catalog import CAPABILITIES, SEED_MODELS, select
from video_gen.orcarouter.credentials import ENV_KEY, METHOD_ENV, CredentialStore
from video_gen.providers.orcarouter import OrcaRouterProvider

#: Captured at import time, before any other test can mutate the environment.
#: The web and credential tests legitimately clear the credential environment,
#: so the live key is pinned here and re-asserted per test rather than re-read.
LIVE_KEY = os.getenv("ORCAROUTER_API_KEY")

#: The verified outage-fallback ids. A live result must never contain one of
#: these unless the service itself is serving it.
SEED_IDS = frozenset(m.id for m in SEED_MODELS)

pytestmark = pytest.mark.skipif(
    not LIVE_KEY,
    reason="ORCAROUTER_API_KEY is not set; live checks are opt-in",
)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """The real provider, holding the real key, through the real credential store."""
    monkeypatch.setenv(ENV_KEY, LIVE_KEY)
    store = CredentialStore(tmp_path / ".env")
    store.save(LIVE_KEY, method=METHOD_ENV, scope="api")
    return OrcaRouterProvider(Config(), store=store)


def test_live_origins_are_the_documented_public_pair(provider):
    origins = provider.origins()
    assert origins.auth_base == "https://www.orcarouter.ai"
    assert origins.api_base == "https://api.orcarouter.ai/v1"
    # And the two are never each other with a path swapped.
    assert origins.models_url() == "https://api.orcarouter.ai/v1/models"
    assert origins.exchange_url() == "https://www.orcarouter.ai/api/v1/auth/keys"


def test_live_catalog_is_reachable_through_the_provider(provider):
    catalog = provider.catalog()
    assert catalog.degraded is False, f"catalog fell back: {catalog.detail}"
    assert catalog.source == "live"
    assert catalog.models, "the live catalog returned no models"

    for model in catalog.models:
        assert "/" in model.id, "vendor namespace must be preserved verbatim"
        assert model.id == model.id.strip()


def test_live_catalog_filters_produce_compatible_options(provider):
    catalog = provider.catalog()

    chat = select(catalog.models, "chat")
    assert chat, "the authenticated catalog exposed no chat models"
    for model in chat:
        # Never a model whose only routes are specialised.
        assert not set(model.endpoint_types) & {
            "image-generation", "openai-video", "jina-rerank", "embeddings"
        }
        assert set(model.endpoint_types) & {
            "openai", "anthropic", "gemini", "openai-response"
        }

    # Every capability must be answerable without raising, even when empty.
    for capability in CAPABILITIES:
        select(catalog.models, capability)

    # The video list must never contain a model the catalog did not mark as video.
    for model in select(catalog.models, "video"):
        assert "openai-video" in model.endpoint_types

    # And the image-conditioned video list must be a subset that declares image.
    for model in select(catalog.models, "video", ("image",)):
        assert "image" in model.input_modalities


def test_live_model_options_are_never_empty_strings(provider):
    for model in provider.catalog().models:
        assert model.id.strip()
        assert not model.id.startswith(" ")


async def test_live_submit_goes_through_the_provider(provider):
    """A real request on the implemented path.

    Two outcomes are acceptable and both prove the wiring:

    * the key is entitled to a video model and the submit is accepted; or
    * the key's scope does not cover video models, which the service reports as a
      typed ``403 model_access_denied``. A 404 here would mean the endpoint path
      is wrong, and a 401 would mean the credential path is wrong.
    """
    catalog = provider.catalog()
    options = [m.id for m in select(catalog.models, "video")] or ["kling/kling-v3"]
    model = options[0]

    request = VideoRequest(prompt="a calm ocean wave at sunrise, cinematic", model=model)
    try:
        ref = await provider.submit(request)
    except ProviderReauthRequired as exc:  # pragma: no cover - live environment
        pytest.fail(f"the stored live credential was rejected: {exc}")
    except APIError as exc:
        assert exc.status_code != 404, (
            "404 means the video endpoint path is wrong: " f"{exc}"
        )
        payload = exc.payload if isinstance(exc.payload, dict) else {}
        code = (payload.get("error") or {}).get("code")
        assert code == "model_access_denied", (
            f"unexpected live error for {model}: {exc.status_code} {exc}"
        )
        assert LIVE_KEY not in str(exc)
        return

    assert ref.task_id
    assert ref.provider == "orcarouter"

    # And the poll path, on the same task.
    status = await provider.get_status(ref.task_id)
    assert status.provider == "orcarouter"
    assert status.state in ("pending", "running", "succeeded", "failed", "cancelled")


async def test_live_chat_entitlement_proves_the_key_reaches_inference(provider):
    """Sanity anchor for the live checks: the key can reach the inference origin.

    The call is issued through ``provider.api_get``, so the origin, the path and
    the credential all come from the integration this change adds rather than
    from a hand-built request in the test.
    """
    catalog = provider.catalog()
    chat = select(catalog.models, "chat")
    if not chat:  # pragma: no cover - live environment
        pytest.skip("no chat models are entitled to this key")

    # A real, authenticated read against the inference origin on the provider's
    # own transport and origins.
    response = await provider.api_get("/models")
    assert response.status_code == 200, response.text[:300]
    assert LIVE_KEY not in response.text

    # The entitlement path is the catalog this provider already exposes, and it
    # must agree with the raw read above.
    ids = {m.id for m in catalog.models}
    assert chat[0].id in ids
    await provider.aclose()


async def test_live_text_selector_options_come_from_the_api_not_a_hardcoded_list(
    provider,
):
    """The dropdown is bound to the API's own list, not to a list in the source.

    ``video_gen.web._model_options`` builds the list the browser is offered, and
    this compares that exact list to a real authenticated ``GET /v1/models``: the
    two must agree model-for-model, which is only possible if the options really
    come from the service. A hardcoded example list, a stale seed baked into a
    successful run, or a list that leaks a non-video model all fail here.

    For this repository the text entry point is the prompt field of a video
    request, so the selector's options are the ``openai-video`` models rather
    than chat models. This key's workspace is not entitled to video generation,
    which is itself worth pinning: an unentitled entry point must report an
    honest empty list at ``source == "live"``, never a hardcoded example list and
    never a seed folded into a successful run. The same binding rule is then
    asserted on the chat filter, which this key *can* exercise.
    """
    from video_gen import web

    options, meta = web._model_options("orcarouter", False)
    ids = [option["id"] for option in options]

    # A real, authenticated read on the provider's own origins and transport.
    response = await provider.api_get("/models")
    assert response.status_code == 200, response.text[:300]
    payload = response.json()
    served = {record["id"] for record in payload.get("data", [])}
    assert served, "the authenticated catalog returned nothing to bind against"

    assert meta["source"] == "live", f"a live run must not report {meta['source']!r}"
    assert meta["degraded"] is False, meta
    assert meta["catalog_model_count"] == len(served), meta

    if ids:
        # Entitled: every offered option must be one the API actually serves.
        for model_id in ids:
            assert model_id in served, (
                f"{model_id!r} is offered by the selector but "
                f"{provider.origins().models_url()} does not serve it"
            )
            assert "/" in model_id, "vendor namespace must survive into the selector"
    else:
        # Not entitled to video: the empty list is an entitlement fact, not a
        # wiring failure, and it must not have been padded by a fallback.
        assert not any(
            "openai-video" in record.get("supported_endpoint_types", [])
            for record in payload.get("data", [])
        ), "the API serves video models but the selector offered none"
        for model_id in SEED_IDS:
            assert model_id not in ids, f"seed {model_id!r} leaked into a live result"

    # The chat filter — the same capability machinery, on a list this key is
    # entitled to — must be non-empty and equally bound to the API's own list.
    chat_ids = [m.id for m in select(provider.catalog().models, "chat")]
    assert chat_ids, "the authenticated catalog exposed no chat models"
    for model_id in chat_ids:
        assert model_id in served, f"{model_id!r} was not served by the API"
    await provider.aclose()
