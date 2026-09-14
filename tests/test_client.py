import pytest

from video_gen.exceptions import ProviderNotSupported
from video_gen.models import TaskStatus, VideoRequest
from video_gen.providers.registry import get_provider_class


def test_video_request_defaults():
    request = VideoRequest(prompt="hello")
    assert request.duration == 5
    assert request.aspect_ratio == "16:9"
    assert request.resolution == "720p"


def test_task_status_accepts_known_state():
    status = TaskStatus(task_id="abc", provider="seedance", state="running")
    assert status.state == "running"


def test_registry_returns_known_providers():
    for name in ("seedance", "kling", "minimax", "wan"):
        assert get_provider_class(name).name == name


def test_registry_rejects_unknown_provider():
    with pytest.raises(ProviderNotSupported):
        get_provider_class("not-a-provider")
