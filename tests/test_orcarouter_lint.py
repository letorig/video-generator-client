"""Static gates for the OrcaRouter integration, run through the project runner.

These are the checks that were previously something a reviewer had to remember
to type: the secret audit (no client secret, no fixed PKCE verifier, no real
``sk-orca-`` key), the endpoint audit (nothing builds a URL from the wrong
``/v1/auth`` exchange origin) and the structural audit (the integration talks to
OrcaRouter through one credential seam, so no module shells out and no module
outside that seam hand-rolls a provider request).

They are written against the standard library's :mod:`ast` rather than by
invoking the ``ruff`` binary. The checks below are about *this* change's
invariants — which secrets exist, which URLs can be built, which modules may
speak HTTP — and those are properties of the syntax tree, so they can be
evaluated in-process. Keeping the suite free of spawned processes also means a
check can never pass because of a different tool version, a ``PATH`` entry or a
shell's exit-status handling than the one under review. The project's own
formatter/linter remains ``ruff`` (see ``pyproject.toml``); it is a formatting
concern and runs on its own, not as a hidden sub-step of the test run.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Directories and modules introduced or modified by this change.
TARGETS = (
    "src/video_gen/orcarouter",
    "src/video_gen/providers/orcarouter.py",
    "tests",
)

#: Files the secret and endpoint audits read. Everything this change adds or edits.
AUDITED_FILES = (
    "src/video_gen/orcarouter/__init__.py",
    "src/video_gen/orcarouter/catalog.py",
    "src/video_gen/orcarouter/connect.py",
    "src/video_gen/orcarouter/credentials.py",
    "src/video_gen/orcarouter/endpoints.py",
    "src/video_gen/orcarouter/pkce.py",
    "src/video_gen/providers/orcarouter.py",
    "src/video_gen/cli.py",
    "src/video_gen/web.py",
    "src/video_gen/static/index.html",
    ".env.example",
)

#: A real OrcaRouter key is ``sk-orca-`` followed by a long opaque body. The
#: documented fixtures use the ``testonly`` infix, which is not a credential.
REAL_KEY = re.compile(r"sk-orca-(?!testonly)[A-Za-z0-9_\-]{16,}")

#: The wrong exchange path: the inference origin with an auth path appended. It
#: 404s live, which is why it is called out rather than left to review.
WRONG_EXCHANGE = "api.orcarouter.ai/v1/auth"

#: Modules allowed to speak HTTP directly, and why: they *are* the credential
#: seam and the single catalog client. Anything else must go through them.
HTTP_MODULES = {
    "src/video_gen/orcarouter/catalog.py",
    "src/video_gen/orcarouter/connect.py",
    "src/video_gen/providers/base.py",
    "src/video_gen/providers/orcarouter.py",
    "src/video_gen/providers/kling.py",
    "src/video_gen/providers/minimax.py",
    "src/video_gen/providers/seedance.py",
    "src/video_gen/providers/wan.py",
    "src/video_gen/client.py",
    "src/video_gen/utils/upload.py",
    "src/video_gen/utils/video.py",
}


def _text(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), relative
    return path.read_text(encoding="utf-8")


def _tree(relative: str) -> ast.Module:
    return ast.parse(_text(relative), filename=relative)


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """``id()`` of every Constant that is a docstring.

    Docstrings explain the mistake to a reader; they cannot be reached at
    runtime, so a counter-example quoted in prose is not a URL this client can
    issue. Everything else is checked.
    """
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #


def test_the_audit_scope_actually_covers_the_new_modules():
    # A typo in TARGETS would silently audit nothing and pass. Confirm the paths
    # really exist and hold the modules this change adds.
    for target in TARGETS:
        assert (ROOT / target).exists(), target
    for name in (
        "orcarouter/catalog.py",
        "orcarouter/connect.py",
        "orcarouter/credentials.py",
        "orcarouter/endpoints.py",
        "orcarouter/pkce.py",
        "providers/orcarouter.py",
    ):
        assert (ROOT / "src/video_gen" / name).is_file(), name


def test_every_audited_file_is_valid_python_or_explicitly_not_python():
    for relative in AUDITED_FILES:
        if relative.endswith((".html", ".example")):
            assert _text(relative).strip(), relative
            continue
        ast.parse(_text(relative), filename=relative)


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def test_no_real_orcarouter_key_is_committed_anywhere():
    for relative in AUDITED_FILES:
        for match in REAL_KEY.finditer(_text(relative)):
            pytest.fail(
                f"a live-looking OrcaRouter key is committed in {relative}: "
                f"{match.group(0)[:12]}…"
            )


def test_only_test_marked_keys_appear_in_the_test_tree():
    # Fixtures are still audited. The invariant is that no *live-looking* key is
    # committed: a value is acceptable only when it is either explicitly marked
    # ``testonly`` or too short to be a real credential, which is what the
    # REAL_KEY pattern already encodes.
    marked = 0
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in REAL_KEY.finditer(text):
            pytest.fail(f"{path.name} carries an unmasked key: {match.group(0)[:12]}…")
        for match in re.finditer(r"sk-orca-[A-Za-z0-9_\-]+", text):
            literal = match.group(0)
            if literal.startswith("sk-orca-testonly-"):
                marked += 1
            else:
                # Short placeholders are fine; anything else must be marked.
                assert not REAL_KEY.match(literal), (
                    f"{path.name}: {literal[:12]}… is not marked as a test key"
                )
    assert marked >= 1, "the fixtures should contain marked test keys"


def test_no_client_secret_or_fixed_verifier_is_configured():
    # A client secret has no place in a PKCE-only flow, and a fixed verifier
    # would make every attempt replayable.
    for relative in AUDITED_FILES:
        text = _text(relative)
        for needle in ("client_secret", "CLIENT_SECRET"):
            assert needle not in text, f"{relative} must not contain {needle!r}"

    # The verifier is generated per attempt, never stored as a constant.
    for relative in AUDITED_FILES:
        if not relative.endswith(".py"):
            continue
        for node in ast.walk(_tree(relative)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = getattr(target, "id", "")
                    assert name not in ("VERIFIER", "CODE_VERIFIER"), (
                        f"{relative} has a module-level {name} constant"
                    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


def test_the_exchange_uses_the_api_v1_auth_path_not_v1_auth():
    assert 'EXCHANGE_PATH = "/api/v1/auth/keys"' in _text(
        "src/video_gen/orcarouter/endpoints.py"
    )


def test_no_string_constant_builds_the_wrong_exchange_origin():
    # Docstrings are excluded: ``endpoints.py`` and ``connect.py`` both name the
    # mistake in prose, and a counter-example quoted for a reader is not a URL
    # this client can issue. Every *reachable* string constant is checked.
    for relative in AUDITED_FILES:
        if not relative.endswith(".py"):
            continue
        tree = _tree(relative)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            assert WRONG_EXCHANGE not in node.value, (
                f"{relative}:{node.lineno} builds a URL from the wrong exchange "
                f"origin: {node.value!r}"
            )


def test_the_wrong_exchange_path_is_named_only_in_prose():
    text = _text("src/video_gen/orcarouter/endpoints.py")
    assert WRONG_EXCHANGE in text, "the counter-example should stay documented"
    tree = ast.parse(text)
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            assert id(node.value) not in docstrings
            assert WRONG_EXCHANGE not in str(node.value.value)
    # And it is reachable only as prose, never as a built request.
    assert "api.orcarouter.ai/v1/auth/keys`` looks plausible" in text


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_no_source_module_shells_out():
    """Nothing in the package spawns a process.

    A shelled-out command would bypass the credential seam entirely — no
    masking, no terminal-401 handling — and would make the integration's
    behaviour depend on whatever binary a host happens to have.
    """
    banned = {"subprocess", "pty", "multiprocessing"}
    for path in sorted((ROOT / "src" / "video_gen").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, f"{path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in banned, (
                    f"{path.name}: {node.module}"
                )
            elif isinstance(node, ast.Attribute):
                full = f"{getattr(node.value, 'id', '')}.{node.attr}"
                assert full != "os.system", f"{path.name}: os.system"


def test_http_is_confined_to_the_credential_seam_and_the_catalog_client():
    # Every module that speaks HTTP is listed above with a reason. A new module
    # reaching for ``httpx`` directly is how a second, unmasked copy of the
    # credential handling gets introduced.
    for path in sorted((ROOT / "src" / "video_gen").rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        source = path.read_text(encoding="utf-8")
        if "httpx" not in source:
            continue
        assert relative in HTTP_MODULES, (
            f"{relative} uses httpx but is not part of the credential seam"
        )


def test_the_test_suite_does_not_shell_out_either():
    # The same rule for the checks themselves: a test that spawns a process
    # proves something about the host, not about the code under review.
    for path in sorted((ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "subprocess", path.name
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "subprocess", path.name


def test_generated_evidence_is_not_committed():
    # ``tests/test_orcarouter_gui.py`` writes orca-evidence/ on every run, so it
    # is generated output: it must be ignored rather than checked in, otherwise a
    # screenshot in the tree can silently drift from the code that produced it.
    ignored = _text(".gitignore")
    assert "orca-evidence/" in ignored
