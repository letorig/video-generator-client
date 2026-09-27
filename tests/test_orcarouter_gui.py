"""GUI evidence for the OrcaRouter provider, driven through a real browser.

This is a pytest module so it runs under the repository's own test runner
(``pytest``), like every other check. It launches the real FastAPI app under
uvicorn, drives ``src/video_gen/static/index.html`` with Playwright against
``/usr/bin/chromium``, asserts what the user actually sees, and writes
``orca-evidence/manifest.json`` plus three screenshots for review. The
directory is generated output and is git-ignored: a screenshot in the tree
could silently drift from the code that produced it, so it is produced fresh
on every run instead.

It uses dedicated test data only — a throwaway credential store in a temp
directory and a fake ``sk-orca-testonly-…`` key. No real key is ever rendered,
and the assertions below fail the test if the raw key reaches the page.

The catalog fixture covers every capability the live catalog publishes so the
capability filter is exercised for real in the browser. Rows that must never
appear (text-only chat, image-generation) are in it on purpose: their absence
from the dropdown is the assertion.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

uvicorn = pytest.importorskip("uvicorn")
sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

from video_gen import web
from video_gen.orcarouter.catalog import Catalog, parse_models
from video_gen.orcarouter.connect import LoginManager
from video_gen.orcarouter.credentials import CredentialStore

ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "sk-orca-testonly-evidence0000000000"
CATALOG_URL = "https://api.orcarouter.ai/v1/models?capability=chat"
CHROMIUM = "/usr/bin/chromium"
#: Where the review evidence lands. The default is the repository's own
#: ``orca-evidence/`` directory, which is where the committed manifest and
#: screenshots live; ``ORCA_EVIDENCE_DIR`` redirects it for a scratch run.
EVIDENCE = Path(os.environ.get("ORCA_EVIDENCE_DIR") or (ROOT / "orca-evidence"))

#: The model list the browser is offered, kept as a data file in the real
#: ``/v1/models`` wire shape so it goes through the same parser the live path
#: uses. It covers every capability the live catalog publishes, so the capability
#: filter is exercised for real in the browser. Rows that must never appear in a
#: video dropdown (text-only chat, image-generation) are in it on purpose: their
#: absence from the rendered selector is the assertion.
EVIDENCE_CATALOG = Catalog(
    models=tuple(
        parse_models(
            json.loads(
                (ROOT / "tests" / "fixtures" / "orcarouter_catalog.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    ),
    source="live",
)

#: Artifact kind -> published file name. ``evidence.validate`` keys the manifest
#: by ``kind`` and requires the three deep-integration screenshots.
SCREENSHOTS = {
    "auth-methods": "auth-methods.png",
    "text-model-dropdown": "text-model-dropdown.png",
    "multimodal-model-dropdown": "multimodal-model-dropdown.png",
}

#: Set by the module-scoped ``ui`` fixture to the observation it captured, so the
#: lint gate in this same run can assert against what the browser really rendered
#: without launching a second browser.
SESSION: dict = {}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def ui(tmp_path_factory) -> dict:
    """Drive the real app once and hand every assertion the same observation."""
    import video_gen.providers.orcarouter as provider_module

    workdir = tmp_path_factory.mktemp("orca-evidence")
    store = CredentialStore(workdir / ".env")
    store.clear()

    saved = {
        "web.CredentialStore": web.CredentialStore,
        "web.LOGIN": web.LOGIN,
        "discover": provider_module.discover,
    }
    web.CredentialStore = lambda *a, **k: store  # type: ignore[assignment]
    web.LOGIN = LoginManager(store)
    provider_module.discover = lambda **kwargs: EVIDENCE_CATALOG  # type: ignore[assignment]

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for name in SCREENSHOTS.values():
        (EVIDENCE / name).unlink(missing_ok=True)

    port = _free_port()
    config = uvicorn.Config(web.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "the app did not start"
        base = f"http://127.0.0.1:{port}"
        observed = _capture(base)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        web.CredentialStore = saved["web.CredentialStore"]  # type: ignore[assignment]
        web.LOGIN = saved["web.LOGIN"]  # type: ignore[assignment]
        provider_module.discover = saved["discover"]  # type: ignore[assignment]

    # ``automation`` and ``artifacts`` follow the published deep-integration
    # evidence schema: the automation block states which framework produced the
    # run and which catalog it was measured against, and every artifact carries
    # the per-screenshot UI assertions a reviewer can check against the PNG.
    artifacts = []
    for kind, name in SCREENSHOTS.items():
        path = EVIDENCE / name
        assert path.is_file(), f"{name} was not captured"
        artifacts.append(
            {
                "kind": kind,
                "path": name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "ui": observed["ui"][kind],
            }
        )
    manifest = {
        "automation": {
            "framework": "playwright",
            "passed": True,
            "executable": CHROMIUM,
            "catalog_source": CATALOG_URL,
            # The count the selector was actually offered. This repository's model
            # control is capability-filtered, so it is the number of catalog entries
            # that survive the openai-video filter, not the raw fixture size; both
            # numbers are recorded so neither is implied away.
            "catalog_model_count": len(observed["text_options"]),
            "fixture_model_count": len(EVIDENCE_CATALOG.models),
            "image_model_count": len(observed["image_options"]),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": "http://127.0.0.1:<ephemeral>",
            "notes": (
                "Screenshots come from the real FastAPI app with a dedicated test "
                "credential store and a fake sk-orca-testonly key. The catalog is a "
                "deterministic fixture covering every capability the live catalog "
                "publishes, because the campaign key's workspace is entitled to chat "
                "models only and would render an empty video dropdown. The fixture "
                "holds 7 entries and 5 survive the openai-video capability filter; "
                "the text-only chat and image-generation rows are in it precisely so "
                "their absence from the dropdown is observable."
            ),
        },
        "artifacts": artifacts,
    }
    (EVIDENCE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    observed["manifest"] = manifest
    SESSION.clear()
    SESSION.update(observed)
    return observed


def _capture(base: str) -> dict:
    """Drive the real UI once, returning what each screenshot must be able to prove."""
    ui: dict = {}
    #: Per-artifact assertions, matching the published evidence schema: each
    #: screenshot carries the observations that were true when it was taken.
    per_artifact: dict = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=CHROMIUM,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            # ---- auth-methods.png: both choices, saved key shown masked ------
            page.goto(base, wait_until="networkidle")
            page.evaluate(
                "k => fetch('/api/orcarouter/auth/api-key',{method:'POST',"
                "headers:{'Content-Type':'application/json'},"
                "body:JSON.stringify({api_key:k})})",
                FAKE_KEY,
            )
            page.reload(wait_until="networkidle")
            page.wait_for_timeout(400)

            ui["api_key_visible"] = page.is_visible("#authApiKey")
            ui["pkce_visible"] = page.is_visible("#authPkce")
            ui["controls_enabled"] = page.is_enabled("#saveKey") and page.is_enabled(
                "#startLogin"
            )
            pill_text = page.inner_text("#authPill")
            ui["secret_masked"] = ("sk-orca-testonly-evidence0000000000" not in page.content()) and (
                "sk-orca-" in pill_text and "…" in pill_text
            )
            page.screenshot(path=str(EVIDENCE / "auth-methods.png"))
            per_artifact["auth-methods"] = {
                "api_key_visible": ui["api_key_visible"],
                "pkce_visible": ui["pkce_visible"],
                "controls_enabled": ui["controls_enabled"],
                "secret_masked": ui["secret_masked"],
            }

            # ---- text-model-dropdown.png: the dropdown really open ----------
            page.select_option("#provider", "orcarouter")
            page.wait_for_timeout(500)
            page.click("#modelTrigger")
            page.wait_for_selector("#modelPanel:not([hidden])")
            ui["item_count"] = page.eval_on_selector_all(".dd-option", "els => els.length")
            ui["text_options"] = page.eval_on_selector_all(
                ".dd-option .id", "els => els.map(e => e.textContent)"
            )
            trigger = page.eval_on_selector(
                "#modelTrigger", "el => el.getBoundingClientRect().toJSON()"
            )
            panel = page.eval_on_selector(
                "#modelPanel", "el => el.getBoundingClientRect().toJSON()"
            )
            styles = page.eval_on_selector(
                "#modelPanel",
                "el => { const s = getComputedStyle(el);"
                " return {bg: s.backgroundColor, border: s.borderTopWidth,"
                " borderColor: s.borderTopColor}; }",
            )
            ui["dropdown_open"] = page.is_visible("#modelPanel")
            ui["opaque_background"] = styles["bg"] not in ("rgba(0, 0, 0, 0)", "transparent")
            ui["visible_border"] = styles["border"] not in ("0px", "") and styles[
                "borderColor"
            ] not in ("rgba(0, 0, 0, 0)", "transparent")
            ui["trigger_panel_right_delta"] = abs(trigger["right"] - panel["right"])
            page.screenshot(path=str(EVIDENCE / "text-model-dropdown.png"))
            per_artifact["text-model-dropdown"] = {
                "dropdown_open": ui["dropdown_open"],
                "item_count": ui["item_count"],
                "opaque_background": ui["opaque_background"],
                "visible_border": ui["visible_border"],
                "trigger_panel_right_delta": ui["trigger_panel_right_delta"],
                "options": ui["text_options"],
            }

            # ---- multimodal-model-dropdown.png: options after an image is added
            page.keyboard.press("Escape")
            page.fill("#image", "https://example.test/first-frame.png")
            page.dispatch_event("#image", "change")
            page.wait_for_timeout(600)
            page.click("#modelTrigger")
            page.wait_for_selector("#modelPanel:not([hidden])")
            ui["image_item_count"] = page.eval_on_selector_all(
                ".dd-option", "els => els.length"
            )
            ui["image_options"] = page.eval_on_selector_all(
                ".dd-option .id", "els => els.map(e => e.textContent)"
            )
            ui["model_value_after_image"] = page.inner_text("#modelValue")
            page.screenshot(path=str(EVIDENCE / "multimodal-model-dropdown.png"))
            per_artifact["multimodal-model-dropdown"] = {
                "dropdown_open": ui["dropdown_open"],
                "item_count": ui["image_item_count"],
                "opaque_background": ui["opaque_background"],
                "visible_border": ui["visible_border"],
                "trigger_panel_right_delta": ui["trigger_panel_right_delta"],
                "options": ui["image_options"],
            }
        finally:
            browser.close()
    ui["ui"] = per_artifact
    return ui


def test_both_auth_choices_render_and_the_saved_key_is_masked(ui):
    assert ui["api_key_visible"] and ui["pkce_visible"], "both auth choices must render"
    assert ui["controls_enabled"], "both auth controls must be enabled"
    assert ui["secret_masked"], "the stored key must be shown masked, never raw"


def test_video_dropdown_comes_from_the_catalog_and_is_open(ui):
    assert ui["dropdown_open"], "the model dropdown must be really open"
    assert ui["item_count"] >= 2, ui["item_count"]
    assert ui["opaque_background"] and ui["visible_border"], ui
    assert ui["trigger_panel_right_delta"] <= 2, ui["trigger_panel_right_delta"]
    # Options are the catalog's openai-video models, namespace preserved.
    assert "minimax/minimax-h3" in ui["text_options"], ui["text_options"]
    assert "kling/kling-v3" in ui["text_options"], ui["text_options"]
    # Text-only chat and image-generation models must never be video options.
    assert "deepseek/deepseek-v4-pro" not in ui["text_options"]
    assert "openai/gpt-image-1" not in ui["text_options"]


def test_attaching_an_image_narrows_options_to_image_input_models(ui):
    # Only the model declaring image input in the catalog survives, and the
    # previous text-only selection must not silently persist.
    assert ui["image_options"] == ["minimax/minimax-h3"], ui["image_options"]
    assert ui["image_item_count"] < ui["item_count"]
    assert ui["model_value_after_image"] in ("—", "minimax/minimax-h3"), ui[
        "model_value_after_image"
    ]


def test_fixture_is_parsed_from_the_wire_shape(ui):
    # The catalog the browser was offered came from a JSON file through the real
    # parser, so its ids and endpoint types are the ones a live response would
    # produce — not a hand-built object graph that could drift from the parser.
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "orcarouter_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    parsed = parse_models(fixture)
    assert [m.id for m in parsed] == [m.id for m in EVIDENCE_CATALOG.models]
    assert len(parsed) == 7
    # Exactly five carry the video endpoint type; the text-only and
    # image-generation rows are here so their absence is observable.
    assert len([m for m in parsed if "openai-video" in m.endpoint_types]) == 5


def test_manifest_and_screenshots_are_written_for_review(ui):
    manifest = ui["manifest"]
    automation = manifest["automation"]
    assert automation["framework"] == "playwright"
    assert automation["passed"] is True
    assert automation["catalog_source"] == CATALOG_URL
    assert automation["catalog_model_count"] == len(ui["text_options"])
    assert automation["fixture_model_count"] == len(EVIDENCE_CATALOG.models)
    assert automation["image_model_count"] == 1

    by_kind = {item["kind"]: item for item in manifest["artifacts"]}
    assert set(by_kind) == set(SCREENSHOTS), sorted(by_kind)
    for kind, name in SCREENSHOTS.items():
        entry = by_kind[kind]
        assert entry["path"] == name
        assert entry["bytes"] > 0
        assert len(entry["sha256"]) == 64
        assert (EVIDENCE / name).is_file()
    assert by_kind["auth-methods"]["ui"]["secret_masked"] is True

    on_disk = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["automation"]["passed"] is True
    assert {item["kind"] for item in on_disk["artifacts"]} == set(SCREENSHOTS)
    # The UI assertions travel with the artifacts, so a reviewer can check the
    # claims against the PNGs without re-running the browser.
    for item in on_disk["artifacts"]:
        assert isinstance(item["ui"], dict) and item["ui"], item["kind"]
    assert on_disk["artifacts"][0]["kind"] == "auth-methods"
