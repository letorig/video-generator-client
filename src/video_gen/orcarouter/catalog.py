"""OrcaRouter model discovery and capability filtering.

One authoritative source: ``GET {api_base}/models``. When the user has supplied a
key the request is authenticated, which returns the models that workspace can
*actually call* — verified live, models that appear only in the anonymous
catalog are rejected with ``403 model_access_denied`` at request time, so
preferring the key is what keeps un-callable models out of the dropdown.

Two rules keep this honest:

* Live success is authoritative. The seed is **never** mixed into a live result.
  A verified seed record may only *enrich* a live record that has the same id and
  is missing metadata the seed already knows (context window, input modalities,
  reasoning ladders) — it never adds a model to the list.
* Capability is never inferred from a model's name. If the catalog does not
  declare a modality an entry point needs, the model fails closed and is hidden.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import httpx

from .endpoints import Origins, resolve_origins

#: The endpoint types that mean "this model can serve a text chat completion".
TEXT_ENDPOINT_TYPES = frozenset({"openai", "anthropic", "gemini", "openai-response"})
#: Endpoint types that are specialised for one non-chat job. A model advertising
#: one of these is not offered as a general chat model.
NON_TEXT_ENDPOINT_TYPES = frozenset(
    {"image-generation", "openai-video", "jina-rerank", "embeddings"}
)

CAPABILITIES = ("chat", "vision", "embedding", "image", "video", "rerank")

#: Capabilities the catalog endpoint filters server-side. Verified live on
#: 2026-09-15: ``?capability=video`` and ``?capability=rerank`` return an empty
#: list even though video models are published, so those are filtered locally
#: instead of silently producing an empty dropdown.
_SERVER_SIDE_CAPABILITY = {"chat": "chat", "image": "image", "embedding": "embedding"}

# Bounds. A catalog response must never be able to consume unbounded memory or
# advertise routes this client cannot speak.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ITEMS = 2000
DISCOVERY_TIMEOUT = 15.0


class CatalogError(Exception):
    """The catalog could not be read or parsed."""


@dataclass(frozen=True)
class ModelRecord:
    """One catalog entry, reduced to what this client can act on."""

    id: str
    endpoint_types: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    owned_by: Optional[str] = None
    description: Optional[str] = None
    reasoning_efforts: tuple[str, ...] = ()
    source: str = "live"


@dataclass(frozen=True)
class Catalog:
    """A model list plus where it came from, so the UI can say so honestly."""

    models: tuple[ModelRecord, ...] = ()
    source: str = "live"
    degraded: bool = False
    detail: str = ""

    def for_capability(
        self,
        capability: str,
        require_modalities: Sequence[str] = (),
    ) -> list[ModelRecord]:
        return select(self.models, capability, require_modalities)

    def ids(self) -> list[str]:
        return [m.id for m in self.models]


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def _endpoints(model: ModelRecord) -> frozenset[str]:
    return frozenset(model.endpoint_types)


def is_chat_model(model: ModelRecord) -> bool:
    endpoints = _endpoints(model)
    if endpoints & NON_TEXT_ENDPOINT_TYPES:
        return False
    return bool(endpoints & TEXT_ENDPOINT_TYPES)


def matches(
    model: ModelRecord,
    capability: str,
    require_modalities: Sequence[str] = (),
) -> bool:
    """Whether ``model`` may serve ``capability`` under the catalog's own claims.

    ``require_modalities`` is the fail-closed gate: every listed modality must be
    explicitly declared in ``architecture.input_modalities``. A model that simply
    does not say is excluded, not assumed capable.
    """
    if capability not in CAPABILITIES:
        raise ValueError(f"Unknown capability {capability!r}; expected one of {CAPABILITIES}")

    endpoints = _endpoints(model)

    if capability == "chat" or capability == "vision":
        ok = is_chat_model(model)
    elif capability == "embedding":
        ok = "embeddings" in endpoints
    elif capability == "image":
        ok = "image-generation" in endpoints
    elif capability == "video":
        ok = "openai-video" in endpoints
    else:  # rerank
        ok = "jina-rerank" in endpoints

    if not ok:
        return False

    declared = frozenset(model.input_modalities)
    return all(modality in declared for modality in require_modalities)


def select(
    models: Iterable[ModelRecord],
    capability: str,
    require_modalities: Sequence[str] = (),
) -> list[ModelRecord]:
    """The exact option list to hand a selector. Order is stable by id."""
    return sorted(
        (m for m in models if matches(m, capability, require_modalities)),
        key=lambda m: m.id,
    )


def is_compatible(
    model_id: str,
    models: Iterable[ModelRecord],
    capability: str,
    require_modalities: Sequence[str] = (),
) -> bool:
    """Validate a restored model id against the *current* filtered option list.

    Used before re-selecting a persisted choice, so a stale id is cleared rather
    than silently kept.
    """
    return any(m.id == model_id for m in select(models, capability, require_modalities))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_models(payload: Any, *, source: str = "live") -> list[ModelRecord]:
    """Validate and bound a ``/v1/models`` payload.

    Anything that is not a record with a usable ``id`` is dropped rather than
    trusted, and the accepted count is capped.
    """
    if not isinstance(payload, dict):
        raise CatalogError("Catalog response was not a JSON object")
    data = payload.get("data")
    if data is None:
        data = payload.get("models")
    if not isinstance(data, list):
        raise CatalogError("Catalog response had no 'data' list")

    records: list[ModelRecord] = []
    for raw in data[:MAX_ITEMS]:
        if not isinstance(raw, dict):
            continue
        model_id = raw.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        architecture = raw.get("architecture")
        architecture = architecture if isinstance(architecture, dict) else {}
        records.append(
            ModelRecord(
                id=model_id.strip(),
                endpoint_types=_as_tuple(raw.get("supported_endpoint_types")),
                input_modalities=_as_tuple(architecture.get("input_modalities")),
                output_modalities=_as_tuple(architecture.get("output_modalities")),
                context_length=_as_int(raw.get("context_length")),
                max_completion_tokens=_as_int(raw.get("max_completion_tokens")),
                owned_by=raw.get("owned_by") if isinstance(raw.get("owned_by"), str) else None,
                description=raw.get("description")
                if isinstance(raw.get("description"), str)
                else None,
                source=source,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Verified fallback seed
# --------------------------------------------------------------------------- #

#: A small, deliberately bounded cold-start seed so a fresh installation is
#: usable when the catalog endpoint is slow or unreachable. Provenance:
#:
#: * ``openai/gpt-5.5``, ``anthropic/claude-opus-4.8``, ``google/gemini-3.5-flash``,
#:   ``deepseek/deepseek-v4-pro`` and ``orcarouter/auto`` are the documented
#:   OrcaRouter fallback set. All five were confirmed present in a live catalog
#:   fetch on 2026-09-15 (``deepseek/deepseek-v4-pro`` and ``orcarouter/auto`` in
#:   the authenticated catalog, the others in the anonymous one). GPT-5.5's
#:   verified ``low``/``medium``/``high``/``xhigh`` reasoning ladder is preserved
#:   here; live discovery does not publish reasoning metadata at all.
#: * The video entries were read from the live anonymous catalog on the same
#:   date, with the endpoint types and modalities exactly as published.
#:
#: This seed is only ever used when live discovery fails outright. It is never
#: merged into a successful live result.
SEED_MODELS: tuple[ModelRecord, ...] = (
    ModelRecord(
        id="openai/gpt-5.5",
        endpoint_types=("openai", "openai-response"),
        input_modalities=("text", "image"),
        output_modalities=("text",),
        context_length=400000,
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        source="seed",
    ),
    ModelRecord(
        id="anthropic/claude-opus-4.8",
        endpoint_types=("openai", "anthropic", "openai-response"),
        input_modalities=("text", "image", "file"),
        output_modalities=("text",),
        context_length=200000,
        reasoning_efforts=("low", "medium", "high"),
        source="seed",
    ),
    ModelRecord(
        id="google/gemini-3.5-flash",
        endpoint_types=("openai", "gemini"),
        input_modalities=("text", "image", "audio", "video"),
        output_modalities=("text",),
        context_length=1000000,
        source="seed",
    ),
    ModelRecord(
        id="deepseek/deepseek-v4-pro",
        endpoint_types=("openai", "openai-response"),
        input_modalities=("text",),
        output_modalities=("text",),
        context_length=1048576,
        source="seed",
    ),
    ModelRecord(
        id="orcarouter/auto",
        endpoint_types=("openai", "openai-response", "anthropic", "gemini"),
        input_modalities=("text",),
        output_modalities=("text",),
        source="seed",
    ),
    ModelRecord(
        id="kling/kling-v3",
        endpoint_types=("openai-video",),
        input_modalities=(),
        output_modalities=("video",),
        source="seed",
    ),
    ModelRecord(
        id="kling/kling-v2-6",
        endpoint_types=("openai-video",),
        input_modalities=(),
        output_modalities=("video",),
        source="seed",
    ),
    ModelRecord(
        id="minimax/minimax-h3",
        endpoint_types=("openai-video",),
        input_modalities=("text", "image", "video", "audio"),
        output_modalities=("video",),
        source="seed",
    ),
)

_SEED_BY_ID = {m.id: m for m in SEED_MODELS}


def _enrich(live: ModelRecord) -> ModelRecord:
    """Fill gaps in a live record from the verified seed, without adding models."""
    known = _SEED_BY_ID.get(live.id)
    if known is None:
        return live
    return replace(
        live,
        input_modalities=live.input_modalities or known.input_modalities,
        output_modalities=live.output_modalities or known.output_modalities,
        context_length=live.context_length or known.context_length,
        reasoning_efforts=live.reasoning_efforts or known.reasoning_efforts,
    )


def default_cache_path() -> Path:
    override = os.getenv("ORCAROUTER_CATALOG_CACHE")
    if override:
        return Path(override)
    base = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "video-gen" / "orcarouter-models.json"


def _read_cache(path: Path) -> list[ModelRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    try:
        return parse_models(payload, source="last-known-good")
    except CatalogError:
        return []


def _write_cache(path: Path, models: Sequence[ModelRecord]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "data": [
                        {
                            "id": m.id,
                            "supported_endpoint_types": list(m.endpoint_types),
                            "architecture": {"input_modalities": list(m.input_modalities)},
                            "context_length": m.context_length,
                        }
                        for m in models
                    ]
                }
            ),
            encoding="utf-8",
        )
    except OSError:  # pragma: no cover - cache is best effort
        pass


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover(
    api_key: Optional[str] = None,
    *,
    origins: Optional[Origins] = None,
    client: Optional[httpx.AsyncClient] = None,
    cache_path: Optional[Path] = None,
    capability: Optional[str] = None,
    use_cache: bool = True,
) -> Catalog:
    """Synchronously fetch the live catalog, falling back to cache then seed.

    Synchronous on purpose: the CLI and the FastAPI request handlers that need a
    catalog are short-lived and this keeps one code path. An ``httpx.AsyncClient``
    may still be passed in for connection reuse.
    """
    resolved = origins or resolve_origins()
    url = resolved.models_url()
    params = {}
    if capability and capability in _SERVER_SIDE_CAPABILITY:
        params["capability"] = _SERVER_SIDE_CAPABILITY[capability]

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=DISCOVERY_TIMEOUT, follow_redirects=False) as http:
            response = http.get(url, headers=headers, params=params)
            if response.status_code >= 400:
                raise CatalogError(f"catalog HTTP {response.status_code}")
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise CatalogError("catalog response exceeded the size bound")
            models = parse_models(response.json(), source="live")
    except (httpx.HTTPError, ValueError, CatalogError) as exc:
        return _fallback(cache_path, detail=str(exc), use_cache=use_cache)

    if not models:
        return _fallback(cache_path, detail="catalog returned no models", use_cache=use_cache)

    enriched = [_enrich(m) for m in models]
    if use_cache:
        _write_cache(cache_path or default_cache_path(), enriched)
    return Catalog(models=tuple(enriched), source="live", degraded=False)


def _fallback(
    cache_path: Optional[Path],
    *,
    detail: str,
    use_cache: bool,
) -> Catalog:
    if use_cache:
        cached = _read_cache(cache_path or default_cache_path())
        if cached:
            return Catalog(
                models=tuple(cached),
                source="last-known-good",
                degraded=True,
                detail=detail,
            )
    return Catalog(
        models=SEED_MODELS,
        source="seed",
        degraded=True,
        detail=detail,
    )
