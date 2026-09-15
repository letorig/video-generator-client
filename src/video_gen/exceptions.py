"""Custom exceptions."""

class VideoGenError(Exception):
    """Base exception for this package."""


class ProviderNotConfigured(VideoGenError):
    """Provider credentials are missing."""


class ProviderNotSupported(VideoGenError):
    """The requested provider is not registered."""


class ProviderReauthRequired(VideoGenError):
    """The stored credential was rejected upstream and must be replaced.

    Terminal, not retryable: a durable OrcaRouter key has no refresh grant, so
    the only recovery is a new sign-in or a new API key.
    """


class APIError(VideoGenError):
    """A provider returned an HTTP/API error."""

    def __init__(self, message: str, status_code: int | None = None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class TaskFailed(VideoGenError):
    """A remote generation task failed or was cancelled."""


class TaskTimeout(VideoGenError):
    """A remote generation task exceeded the polling timeout."""
