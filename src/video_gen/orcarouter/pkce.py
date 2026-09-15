"""PKCE primitives for the OrcaRouter connect flow.

Only the S256 challenge ever leaves this process before the code exchange. The
verifier is generated from a cryptographic RNG for every attempt and is never
written to a URL, a log, or an error message.

``hashlib`` and ``base64`` do SHA-256 and base64url in the standard library, so
this adds no dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

#: 32 bytes of entropy, base64url-encoded — well above RFC 7636's 43-char floor.
VERIFIER_BYTES = 32
#: Opaque CSRF token echoed back verbatim by the consent screen.
STATE_BYTES = 16


def _b64url(raw: bytes) -> str:
    """base64url without padding, as RFC 7636 requires."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_verifier() -> str:
    """A fresh high-entropy verifier.

    Must be called once per authorization attempt and must never be reused or
    derived from anything guessable (a timestamp, a username, a fixed salt).
    """
    return _b64url(secrets.token_bytes(VERIFIER_BYTES))


def generate_state() -> str:
    """A fresh opaque CSRF token for one authorization attempt."""
    return _b64url(secrets.token_bytes(STATE_BYTES))


def challenge_for(verifier: str) -> str:
    """``base64url(sha256(verifier))`` with no padding."""
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def state_matches(expected: str, received: str | None) -> bool:
    """Constant-time comparison of the state the consent screen echoed back.

    This is the only thing standing between the loopback listener and a code
    somebody else's page dropped on it.
    """
    if not received:
        return False
    return hmac.compare_digest(expected, received)
