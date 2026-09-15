"""OrcaRouter support: origins, credentials, PKCE connect flows, and catalog.

The credential seam lives here. Two adapters — a pasted API key
(:class:`~video_gen.orcarouter.credentials.ApiKeyAdapter`) and OAuth 2.0 + PKCE
(:mod:`video_gen.orcarouter.connect`) — both produce the same
:class:`~video_gen.orcarouter.credentials.Credential`, and nothing downstream
knows which was used.
"""

from .catalog import (
    CAPABILITIES,
    SEED_MODELS,
    Catalog,
    CatalogError,
    ModelRecord,
    discover,
    is_compatible,
    select,
)
from .connect import (
    AuthorizationDenied,
    ConnectError,
    ConnectTimeout,
    ExchangeError,
    LoginManager,
    ScopeInsufficient,
    StateMismatch,
    complete_oob,
    connect_loopback,
    exchange_code,
    start_oob,
)
from .credentials import (
    ENV_KEY,
    KEY_PREFIX,
    METHOD_API_KEY,
    METHOD_ENV,
    METHOD_PKCE,
    ApiKeyAdapter,
    Credential,
    CredentialError,
    CredentialStore,
    InvalidApiKey,
    NeedsReauth,
    mask_secret,
    redact,
)
from .endpoints import (
    DEFAULT_API_BASE_URL,
    DEFAULT_AUTH_BASE_URL,
    OriginError,
    Origins,
    resolve_origins,
)

__all__ = [
    "CAPABILITIES",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_AUTH_BASE_URL",
    "ENV_KEY",
    "KEY_PREFIX",
    "METHOD_API_KEY",
    "METHOD_ENV",
    "METHOD_PKCE",
    "SEED_MODELS",
    "ApiKeyAdapter",
    "AuthorizationDenied",
    "Catalog",
    "CatalogError",
    "ConnectError",
    "ConnectTimeout",
    "Credential",
    "CredentialError",
    "CredentialStore",
    "ExchangeError",
    "InvalidApiKey",
    "LoginManager",
    "ModelRecord",
    "NeedsReauth",
    "OriginError",
    "Origins",
    "ScopeInsufficient",
    "StateMismatch",
    "complete_oob",
    "connect_loopback",
    "discover",
    "exchange_code",
    "is_compatible",
    "mask_secret",
    "redact",
    "resolve_origins",
    "select",
    "start_oob",
]
