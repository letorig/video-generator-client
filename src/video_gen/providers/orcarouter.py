"""OrcaRouter provider adapter.

OrcaRouter is an OpenAI-compatible AI gateway, but video generation on it is
*not* synchronous chat: it is an async submit-then-poll API on the same
inference origin.

* submit — ``POST {api_base}/video/generations``
* poll   — ``GET  {api_base}/video/generations/{task_id}``

The submit response is OpenAI-shaped and flat with a lowercase ``status``; the
poll response is a wrapped envelope with an uppercase ``status`` and the MP4 in
``data.result_url``. Both shapes are handled here and normalised onto this
project's :class:`~video_gen.models.TaskStatus`.

Model ids are used verbatim, ``vendor/model`` namespace preserved. The request
body shape is selected by the model's namespace prefix (``kling/``, ``byteplus/``,
``minimax/``), which is why the namespace must never be stripped or rewritten.

A ``401`` is terminal: there is no refresh grant for a durable OrcaRouter key, so
the exact credential generation that made the rejected request is flagged for
reauthentication and nothing is retried.
"""

from __future__ import annotations

import re
from typing import Optional

from ..exceptions import APIError, ProviderNotConfigured, ProviderReauthRequired
from ..models import TaskRef, TaskStatus, VideoRequest
from ..orcarouter.catalog import (
    Catalog,
    ModelRecord,
    discover,
    select,
)
from ..orcarouter.credentials import CredentialStore
from ..orcarouter.endpoints import resolve_origins
from .base import BaseProvider

#: Last-resort model for a text-to-video request when the catalog yields no
#: options at all. Its provenance is the live catalog: ``kling/kling-v3`` was
#: observed publishing ``supported_endpoint_types: ["openai-video"]`` on
#: 2026-09-15, and it is one of the verified fallback entries in
#: :mod:`video_gen.orcarouter.catalog`. Normal operation never reaches this — a
#: working catalog always supplies the model.
DEFAULT_VIDEO_MODEL = "kling/kling-v3"

#: Resolution -> Kling ``mode``. 4K is deliberately absent: the catalog does not
#: let this client prove which models accept it.
_MODE_BY_RESOLUTION = {"720p": "std", "1080p": "pro"}

_PERCENT = re.compile(r"(\d+(?:\.\d+)?)")

_STATUS_MAP = {
    "NOT_START": "pending",
    "SUBMITTED": "pending",
    "IN_PROGRESS": "running",
    "SUCCESS": "succeeded",
    "FAILURE": "failed",
    "UNKNOWN": "pending",
}

# Namespace -> the request-body variant OrcaRouter expects. Selecting the wrong
# one is a hard failure upstream, so the prefix is read, never assumed.
_KNOWN_NAMESPACES = ("kling/", "byteplus/", "minimax/")


def namespace_of(model_id: str) -> Optional[str]:
    for prefix in _KNOWN_NAMESPACES:
        if model_id.startswith(prefix):
            return prefix.rstrip("/")
    return None


class OrcaRouterProvider(BaseProvider):
    name = "orcarouter"

    def __init__(self, config, client=None, store: Optional[CredentialStore] = None):
        super().__init__(config, client=client)
        self._store = store or CredentialStore()

    # -- credential -------------------------------------------------------- #

    def credential(self):
        """The stored credential, whatever produced it."""
        return self._store.current()

    def is_configured(self) -> bool:
        credential = self.credential()
        return bool(credential and credential.usable)

    def _auth_headers(self) -> dict[str, str]:
        credential = self.credential()
        if credential is None:
            raise ProviderNotConfigured(
                "No OrcaRouter credential. Paste an API key from "
                "https://www.orcarouter.ai/console/token or run "
                "'video-gen orcarouter-login'."
            )
        if credential.needs_reauth:
            raise ProviderReauthRequired(
                "The stored OrcaRouter credential was rejected and needs to be "
                "replaced. Run 'video-gen orcarouter-login' or set a new "
                "ORCAROUTER_API_KEY."
            )
        return {
            "Authorization": f"Bearer {credential.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def api_get(self, path: str, *, params: Optional[dict] = None):
        """GET an inference-origin path with the stored credential.

        A small convenience for callers that need an authenticated read from the
        same origin the provider itself talks to (entitlement probes, health
        anchors). It exists so nothing outside this module has to build a
        request, pick an origin or attach the credential by hand: every
        authenticated call in this codebase goes through the credential seam
        above, which is also what keeps the key out of ad-hoc call sites.

        ``path`` is relative to the resolved API base and must start with ``/``.
        """
        if not path.startswith("/"):
            raise ValueError(f"api_get path must be absolute, got {path!r}")
        credential = self.credential()
        generation = credential.generation if credential else 0
        response = await self.client.get(
            f"{self.origins().api_base}{path}",
            params=params or {},
            headers=self._auth_headers(),
        )
        if response.status_code == 401:
            self._handle_401(generation, f"GET {path}")
        return response

    def _handle_401(self, generation: int, operation: str) -> None:
        """Mark the *exact* rejected generation for reauthentication.

        The generation is the one captured when this request was issued, so a
        late ``401`` from an old request cannot flag a credential that has since
        been replaced.
        """
        self._store.mark_needs_reauth(generation)
        raise ProviderReauthRequired(
            f"OrcaRouter rejected the stored credential during {operation}. "
            "Run 'video-gen orcarouter-login' to sign in again, or set a new "
            "ORCAROUTER_API_KEY."
        )

    # -- catalog ----------------------------------------------------------- #

    def catalog(self) -> Catalog:
        """The live catalog, authenticated when a key is available.

        An authenticated catalog is the entitlement list: models that appear only
        in the anonymous catalog are rejected at request time, so the key is
        preferred whenever one exists.
        """
        credential = self.credential()
        key = credential.api_key if credential and credential.usable else None
        return discover(api_key=key, origins=self.origins())

    def origins(self):
        return resolve_origins(
            getattr(self.config, "orcarouter_auth_base_url", None),
            getattr(self.config, "orcarouter_api_base_url", None),
            getattr(self.config, "orcarouter_base_url", None),
        )

    def video_models(self, requires_image: bool = False) -> list[ModelRecord]:
        """Video models, filtered by the modalities this entry actually sends.

        Fail closed: a model that declares no input modalities is offered for
        text-to-video, where nothing extra is uploaded, but never for
        image-to-video, where the catalog must explicitly say it accepts an image.
        """
        catalog = self.catalog()
        modalities = ("image",) if requires_image else ()
        return select(catalog.models, "video", modalities)

    def resolve_model(self, requested: Optional[str], *, requires_image: bool = False) -> str:
        """Pick the model id to send, validating any user-supplied value.

        A user-supplied id is checked against the catalog the same way a dropdown
        option would be, so a model that cannot accept the current input is
        rejected instead of being sent and failing upstream.
        """
        catalog = self.catalog()
        modalities = ("image",) if requires_image else ()
        options = select(catalog.models, "video", modalities)

        if requested:
            if not options:
                # No live video catalog to validate against; accept the caller's
                # id rather than blocking on a catalog outage.
                return requested
            if any(m.id == requested for m in options):
                return requested
            if any(m.id == requested for m in select(catalog.models, "video")):
                raise APIError(
                    f"'{requested}' does not declare the input modalities this "
                    f"request needs (image). Compatible models: "
                    f"{', '.join(m.id for m in options[:8])}"
                )
            raise APIError(
                f"'{requested}' is not a video model in the OrcaRouter catalog. "
                f"Choose one of: {', '.join(m.id for m in options[:8])}"
            )

        if options:
            return options[0].id
        return DEFAULT_VIDEO_MODEL

    # -- BaseProvider ------------------------------------------------------ #

    async def submit(self, request: VideoRequest) -> TaskRef:
        credential = self.credential()
        generation = credential.generation if credential else 0
        requires_image = bool(request.image_url)
        model = self.resolve_model(request.model, requires_image=requires_image)
        origins = self.origins()

        payload = build_body(model, request, requires_image=requires_image)
        extra = dict(request.extra)
        metadata = payload.setdefault("metadata", {})
        metadata.update(extra.pop("metadata", {}))
        payload.update(extra)

        url = origins.video_generations_url()
        response = await self.client.post(url, json=payload, headers=self._auth_headers())
        if response.status_code == 401:
            self._handle_401(generation, "submit")
        if response.status_code >= 400:
            raise APIError(
                f"OrcaRouter submit failed: {_error_text(response)}",
                response.status_code,
                _error_payload(response),
            )

        data = response.json()
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise APIError("OrcaRouter returned no task id", payload=data)
        return TaskRef(task_id=str(task_id), provider=self.name, raw=data)

    async def get_status(self, task_id: str) -> TaskStatus:
        credential = self.credential()
        generation = credential.generation if credential else 0
        url = f"{self.origins().video_generations_url()}/{task_id}"

        response = await self.client.get(url, headers=self._auth_headers())
        if response.status_code == 401:
            self._handle_401(generation, "status")
        if response.status_code >= 400:
            raise APIError(
                f"OrcaRouter status failed: {_error_text(response)}",
                response.status_code,
                _error_payload(response),
            )
        return _parse_status(task_id, response.json())

    async def cancel(self, task_id: str) -> bool:
        # The documented video surface is submit + poll only; there is no
        # cancel endpoint to call, so this reports honestly rather than
        # pretending to have cancelled anything.
        return False


_SIZE_BY_ASPECT = {
    ("16:9", "720p"): "1280x720",
    ("16:9", "1080p"): "1920x1080",
    ("9:16", "720p"): "720x1280",
    ("9:16", "1080p"): "1080x1920",
    ("1:1", "720p"): "720x720",
    ("1:1", "1080p"): "1080x1080",
}


def build_body(model: str, request: VideoRequest, *, requires_image: bool) -> dict:
    """Build the request body for the model's namespace variant.

    OrcaRouter selects the upstream schema from the ``model`` prefix, so each
    namespace gets the field placement its own documentation specifies rather
    than one lowest-common-denominator shape.
    """
    namespace = namespace_of(model)
    duration = str(request.duration)

    if namespace == "byteplus":
        # prompt + metadata.{content[], ratio, duration, resolution, seed, ...}
        metadata: dict = {
            "ratio": request.aspect_ratio,
            "duration": duration,
            "resolution": request.resolution,
        }
        if request.seed is not None:
            metadata["seed"] = request.seed
        body: dict = {"model": model, "prompt": request.prompt, "metadata": metadata}
        if requires_image:
            # The image goes in content[] with an explicit role for this variant.
            metadata["content"] = [{"type": "image_url", "image_url": {"url": request.image_url}, "role": "first_frame"}]
        return body

    if namespace == "minimax":
        # prompt + duration + size + image + metadata.{ratio, first_frame_image, ...}
        body = {
            "model": model,
            "prompt": request.prompt,
            "duration": request.duration,
            "size": _SIZE_BY_ASPECT.get(
                (request.aspect_ratio, request.resolution), "1280x720"
            ),
            "metadata": {"ratio": request.aspect_ratio},
        }
        if requires_image:
            body["image"] = request.image_url
            body["metadata"]["first_frame_image"] = request.image_url
        return body

    # Kling (the default documented variant): prompt + image + metadata.{...}
    metadata = {"duration": duration, "aspect_ratio": request.aspect_ratio}
    mode = _MODE_BY_RESOLUTION.get(request.resolution)
    if mode:
        metadata["mode"] = mode
    if request.negative_prompt:
        metadata["negative_prompt"] = request.negative_prompt
    if request.seed is not None:
        metadata["seed"] = request.seed
    body = {"model": model, "prompt": request.prompt, "metadata": metadata}
    if requires_image:
        body["image"] = request.image_url
    return body


def _error_payload(response) -> dict:
    try:
        return response.json()
    except ValueError:
        return {}


def _error_text(response) -> str:
    """A short, safe description. The API key is never part of a response body."""
    payload = _error_payload(response)
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type") or "error"
        message = error.get("message") or ""
        return f"HTTP {response.status_code} {code}: {message}"

    text = (response.text or "").strip()
    return f"HTTP {response.status_code} {text[:300]}"


def _parse_status(task_id: str, data: dict) -> TaskStatus:
    """Normalise the wrapped video-status envelope.

    ``data.status`` is uppercase (``NOT_START``/``SUBMITTED``/``IN_PROGRESS``/
    ``SUCCESS``/``FAILURE``/``UNKNOWN``) and ``progress`` is a percent *string*.
    """
    envelope = data.get("data") if isinstance(data.get("data"), dict) else data
    raw_status = str(envelope.get("status") or "").upper()

    state = _STATUS_MAP.get(raw_status)
    if state is None:
        # An unrecognised state must not be reported as success.
        state = "pending"

    progress = None
    raw_progress = envelope.get("progress")
    if isinstance(raw_progress, str):
        match = _PERCENT.search(raw_progress)
        if match:
            progress = float(match.group(1))
    elif isinstance(raw_progress, (int, float)) and not isinstance(raw_progress, bool):
        progress = float(raw_progress)

    return TaskStatus(
        task_id=task_id,
        provider="orcarouter",
        state=state,
        progress=progress,
        video_url=envelope.get("result_url") if state == "succeeded" else None,
        error=(envelope.get("fail_reason") or None) if state == "failed" else None,
        raw=data,
    )
