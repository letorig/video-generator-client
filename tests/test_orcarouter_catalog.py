"""Model discovery, capability filtering, and selector-option binding.

The fixture covers every category the live catalog publishes — text-only chat,
image-input chat, embedding, image generation, video, and rerank — so each
filter is exercised against a record that should pass *and* records that must
fail closed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from video_gen.orcarouter.catalog import (
    CAPABILITIES,
    MAX_ITEMS,
    SEED_MODELS,
    Catalog,
    CatalogError,
    ModelRecord,
    _fallback,
    default_cache_path,
    discover,
    is_compatible,
    parse_models,
    select,
)

# --------------------------------------------------------------------------- #
# Fixture: one live-shaped catalog covering every capability
# --------------------------------------------------------------------------- #

LIVE_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "vendor/text-only-chat",
            "supported_endpoint_types": ["openai", "openai-response"],
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "context_length": 128000,
        },
        {
            "id": "vendor/vision-chat",
            "supported_endpoint_types": ["openai", "anthropic"],
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "context_length": 200000,
        },
        {
            "id": "vendor/video-input-chat",
            "supported_endpoint_types": ["gemini", "openai"],
            "architecture": {
                "input_modalities": ["text", "image", "audio", "video"],
                "output_modalities": ["text"],
            },
        },
        {
            # No architecture block at all: must fail closed for every modality.
            "id": "vendor/undeclared-chat",
            "supported_endpoint_types": ["openai"],
        },
        {
            "id": "vendor/embed",
            "supported_endpoint_types": ["embeddings"],
            "architecture": {"input_modalities": ["text"]},
        },
        {
            "id": "vendor/imagegen",
            "supported_endpoint_types": ["image-generation"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {
            "id": "vendor/video-text",
            "supported_endpoint_types": ["openai-video"],
            # Kling models really do publish no architecture block.
        },
        {
            "id": "vendor/video-multimodal",
            "supported_endpoint_types": ["openai-video"],
            "architecture": {
                "input_modalities": ["text", "image", "video", "audio"],
                "output_modalities": ["video"],
            },
        },
        {
            "id": "vendor/rerank",
            "supported_endpoint_types": ["jina-rerank"],
        },
        {
            # A model that advertises a specialised route must not be offered as
            # a general chat model, even though it also speaks `openai`.
            "id": "vendor/hybrid-image-model",
            "supported_endpoint_types": ["openai", "image-generation"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
    ],
}

ALL_IDS = {m["id"] for m in LIVE_PAYLOAD["data"]}


#: Captured before any fixture replaces it, so a mocked transport can still
#: construct a genuine client rather than recursing into the network guard.
_REAL_CLIENT = httpx.Client


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Any unmocked HTTP call in this module is a test bug, not a live probe.

    ``_patch_httpx`` installs a MockTransport on top of this for the tests that
    need a response; everything else fails loudly instead of reaching a real
    OrcaRouter origin.
    """

    def explode(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(httpx, "Client", explode)


@pytest.fixture
def models() -> tuple[ModelRecord, ...]:
    return tuple(parse_models(LIVE_PAYLOAD))


def ids(records) -> set[str]:
    return {m.id for m in records}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_models_reads_the_published_shape(models):
    by_id = {m.id: m for m in models}
    assert ids(models) == ALL_IDS
    vision = by_id["vendor/vision-chat"]
    assert vision.endpoint_types == ("openai", "anthropic")
    assert vision.input_modalities == ("text", "image")
    assert vision.context_length == 200000
    assert vision.source == "live"


@pytest.mark.parametrize(
    "payload",
    [
        [],                                     # not an object
        {},                                     # no data list
        {"data": "nope"},
        {"data": None, "models": 5},
    ],
)
def test_parse_models_rejects_malformed_payloads(payload):
    with pytest.raises(CatalogError):
        parse_models(payload)


def test_parse_models_drops_records_without_a_usable_id():
    parsed = parse_models({"data": [{"id": "ok"}, {"id": ""}, {"id": 5}, "nope", None]})
    assert [m.id for m in parsed] == ["ok"]


def test_parse_models_bounds_the_item_count():
    payload = {"data": [{"id": f"m/{i}"} for i in range(MAX_ITEMS + 500)]}
    assert len(parse_models(payload)) == MAX_ITEMS


def test_parse_models_tolerates_missing_optional_fields():
    parsed = parse_models({"data": [{"id": "bare/model"}]})
    assert parsed[0].input_modalities == ()
    assert parsed[0].context_length is None


# --------------------------------------------------------------------------- #
# Capability filters
# --------------------------------------------------------------------------- #


def test_chat_capability_selects_only_text_models(models):
    selected = ids(select(models, "chat"))
    assert selected == {
        "vendor/text-only-chat",
        "vendor/vision-chat",
        "vendor/video-input-chat",
        "vendor/undeclared-chat",
    }
    # No image-generation, video, embedding or rerank model leaks in.
    assert "vendor/imagegen" not in selected
    assert "vendor/video-text" not in selected
    assert "vendor/embed" not in selected
    assert "vendor/rerank" not in selected
    assert "vendor/hybrid-image-model" not in selected


def test_chat_capability_excludes_non_text_endpoint_types(models):
    """A model whose routes are only specialised is not a chat model."""
    selected = ids(select(models, "chat"))
    for model in models:
        if set(model.endpoint_types) & {"image-generation", "openai-video", "jina-rerank", "embeddings"}:
            assert model.id not in selected


def test_vision_capability_requires_a_declared_image_modality(models):
    selected = ids(select(models, "vision", ("image",)))
    assert selected == {"vendor/vision-chat", "vendor/video-input-chat"}
    # Text-only chat and the undeclared record must fail closed.
    assert "vendor/text-only-chat" not in selected
    assert "vendor/undeclared-chat" not in selected


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_multimodal_fails_closed_for_undeclared_modalities(models, modality):
    selected = ids(select(models, "chat", (modality,)))
    assert "vendor/undeclared-chat" not in selected
    for model in models:
        if model.id in selected:
            assert modality in model.input_modalities


def test_embedding_capability(models):
    assert ids(select(models, "embedding")) == {"vendor/embed"}


def test_image_capability(models):
    assert ids(select(models, "image")) == {"vendor/imagegen", "vendor/hybrid-image-model"}


def test_video_capability(models):
    assert ids(select(models, "video")) == {"vendor/video-text", "vendor/video-multimodal"}


def test_video_with_image_input_excludes_models_that_do_not_declare_it(models):
    """Kling-shaped records publish no modalities, so they fail closed here."""
    selected = ids(select(models, "video", ("image",)))
    assert selected == {"vendor/video-multimodal"}
    assert "vendor/video-text" not in selected


def test_rerank_capability(models):
    assert ids(select(models, "rerank")) == {"vendor/rerank"}


def test_capability_selection_is_deterministic(models):
    assert [m.id for m in select(models, "chat")] == sorted(ids(select(models, "chat")))


def test_unknown_capability_is_rejected(models):
    with pytest.raises(ValueError):
        select(models, "not-a-capability")


def test_every_advertised_capability_is_selectable(models):
    assert set(CAPABILITIES) == {"chat", "vision", "embedding", "image", "video", "rerank"}
    for capability in CAPABILITIES:
        select(models, capability)      # must not raise


# --------------------------------------------------------------------------- #
# Stale selection invalidation
# --------------------------------------------------------------------------- #


def test_is_compatible_invalidates_a_model_that_lost_image_support(models):
    assert is_compatible("vendor/video-multimodal", models, "video", ("image",))
    assert not is_compatible("vendor/video-text", models, "video", ("image",))
    assert not is_compatible("vendor/gone", models, "video", ("image",))


def test_filtered_options_shrink_when_an_image_is_attached(models):
    """The requirement: the selector's options must change with the input type."""
    text_options = ids(select(models, "video"))
    image_options = ids(select(models, "video", ("image",)))
    assert image_options < text_options          # a strict subset
    assert "vendor/video-text" in text_options
    assert "vendor/video-text" not in image_options


# --------------------------------------------------------------------------- #
# Catalog object behaviour
# --------------------------------------------------------------------------- #


def test_catalog_for_capability_delegates_to_the_same_filter(models):
    catalog = Catalog(models=models, source="live")
    assert ids(catalog.for_capability("video", ("image",))) == {"vendor/video-multimodal"}
    assert set(catalog.ids()) == ALL_IDS


# --------------------------------------------------------------------------- #
# Fallback: seed, cache, and honest degradation
# --------------------------------------------------------------------------- #


def test_seed_is_small_verified_and_carries_its_metadata():
    assert len(SEED_MODELS) <= 12
    by_id = {m.id: m for m in SEED_MODELS}
    assert {
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.8",
        "google/gemini-3.5-flash",
        "deepseek/deepseek-v4-pro",
        "orcarouter/auto",
    } <= set(by_id)
    # The verified reasoning ladder must survive the fallback path.
    assert by_id["openai/gpt-5.5"].reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert by_id["openai/gpt-5.5"].context_length == 400000
    assert "image" in by_id["openai/gpt-5.5"].input_modalities
    assert all(m.source == "seed" for m in SEED_MODELS)


def test_seed_offers_video_models_with_declared_modalities():
    text_video = select(SEED_MODELS, "video")
    image_video = select(SEED_MODELS, "video", ("image",))
    assert {m.id for m in text_video} >= {"kling/kling-v3", "minimax/minimax-h3"}
    assert {m.id for m in image_video} == {"minimax/minimax-h3"}


def test_fallback_uses_the_seed_when_there_is_no_cache(tmp_path):
    catalog = _fallback(tmp_path / "missing.json", detail="boom", use_cache=True)
    assert catalog.source == "seed"
    assert catalog.degraded is True
    assert catalog.detail == "boom"
    assert ids(catalog.models) == {m.id for m in SEED_MODELS}


# --------------------------------------------------------------------------- #
# Live discovery
# --------------------------------------------------------------------------- #


def _patch_httpx(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_discover_uses_the_api_origin_and_bearer_key(monkeypatch, tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=LIVE_PAYLOAD)

    _patch_httpx(monkeypatch, handler)
    catalog = discover("sk-orca-fake", cache_path=tmp_path / "c.json")

    assert catalog.source == "live"
    assert catalog.degraded is False
    assert seen["url"].startswith("https://api.orcarouter.ai/v1/models")
    assert "www.orcarouter.ai" not in seen["url"]
    assert seen["auth"] == "Bearer sk-orca-fake"


def test_discover_without_a_key_sends_no_authorization_header(monkeypatch, tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=LIVE_PAYLOAD)

    _patch_httpx(monkeypatch, handler)
    discover(None, cache_path=tmp_path / "c.json")
    assert seen["auth"] is None


def test_live_success_is_authoritative_and_never_mixes_in_the_seed(monkeypatch, tmp_path):
    def handler(request):
        return httpx.Response(200, json=LIVE_PAYLOAD)

    _patch_httpx(monkeypatch, handler)
    catalog = discover(None, cache_path=tmp_path / "c.json")

    assert ids(catalog.models) == ALL_IDS
    # Not one seed-only model may appear in a successful live result.
    for seed in SEED_MODELS:
        if seed.id not in ALL_IDS:
            assert seed.id not in catalog.ids()


def test_live_discovery_only_enriches_a_matching_record(monkeypatch, tmp_path):
    """A live record missing metadata may borrow the seed's, but no new model."""
    payload = {
        "data": [
            {"id": "openai/gpt-5.5", "supported_endpoint_types": ["openai"]},
            {"id": "unrelated/model", "supported_endpoint_types": ["openai"]},
        ]
    }
    _patch_httpx(monkeypatch, lambda request: httpx.Response(200, json=payload))
    catalog = discover(None, cache_path=tmp_path / "c.json")

    by_id = {m.id: m for m in catalog.models}
    assert set(by_id) == {"openai/gpt-5.5", "unrelated/model"}
    # Enriched from the verified seed, because the live record declared none.
    assert by_id["openai/gpt-5.5"].reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert by_id["unrelated/model"].input_modalities == ()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(404, text="nope"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"data": []}),
    ],
)
def test_discovery_failure_falls_back_and_is_marked_degraded(
    monkeypatch, tmp_path, response
):
    _patch_httpx(monkeypatch, lambda request: response)
    catalog = discover(None, cache_path=tmp_path / "c.json")

    assert catalog.degraded is True
    assert catalog.source == "seed"
    assert catalog.detail
    # A degraded catalog is still usable and still filtered.
    assert select(catalog.models, "video")


def test_transport_error_falls_back(monkeypatch, tmp_path):
    def handler(request):
        raise httpx.ConnectError("no route to host")

    _patch_httpx(monkeypatch, handler)
    catalog = discover(None, cache_path=tmp_path / "c.json")
    assert catalog.degraded is True
    assert catalog.source == "seed"


def test_oversized_response_is_refused(monkeypatch, tmp_path):
    from video_gen.orcarouter import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "MAX_RESPONSE_BYTES", 64)
    _patch_httpx(monkeypatch, lambda request: httpx.Response(200, json=LIVE_PAYLOAD))
    catalog = discover(None, cache_path=tmp_path / "c.json")
    assert catalog.degraded is True
    assert "size bound" in catalog.detail


def test_last_known_good_cache_is_used_and_marked(monkeypatch, tmp_path):
    cache = tmp_path / "c.json"
    _patch_httpx(monkeypatch, lambda request: httpx.Response(200, json=LIVE_PAYLOAD))
    assert discover(None, cache_path=cache).source == "live"
    assert cache.is_file()

    _patch_httpx(monkeypatch, lambda request: httpx.Response(503, text="down"))
    degraded = discover(None, cache_path=cache)
    assert degraded.source == "last-known-good"
    assert degraded.degraded is True
    assert ids(degraded.models) == ALL_IDS


def test_a_corrupt_cache_degrades_to_the_seed_instead_of_crashing(monkeypatch, tmp_path):
    cache = tmp_path / "c.json"
    cache.write_text("{not json", encoding="utf-8")
    _patch_httpx(monkeypatch, lambda request: httpx.Response(503, text="down"))
    catalog = discover(None, cache_path=cache)
    assert catalog.source == "seed"


def test_cache_round_trips_through_the_parser(tmp_path):
    from video_gen.orcarouter.catalog import _read_cache, _write_cache

    cache = tmp_path / "c.json"
    records = parse_models(LIVE_PAYLOAD)
    _write_cache(cache, records)
    reread = _read_cache(cache)
    assert ids(reread) == ALL_IDS
    assert {m.source for m in reread} == {"last-known-good"}


def test_default_cache_path_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv("ORCAROUTER_CATALOG_CACHE", raising=False)
    path = default_cache_path()
    assert path.name == "orcarouter-models.json"
    assert "video-gen" in str(path)
    monkeypatch.setenv("ORCAROUTER_CATALOG_CACHE", "/tmp/explicit.json")
    assert str(default_cache_path()) == "/tmp/explicit.json"


def test_seed_is_json_serialisable_and_contains_no_secret():
    blob = json.dumps([m.id for m in SEED_MODELS])
    assert "sk-orca" not in blob
    for seed in SEED_MODELS:
        assert "/" in seed.id            # vendor namespace preserved verbatim
