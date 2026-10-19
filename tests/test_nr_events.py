"""src/nr_events: the daemon's New Relic Event API reporter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import nr_events  # noqa: E402


class _Resp:
    def __init__(self, status_code: int):
        self.status_code = status_code


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Resp(200)

    monkeypatch.setattr(nr_events.requests, "post", fake_post)
    return calls


def test_no_key_sends_nothing(monkeypatch, posts):
    monkeypatch.delenv("NEW_RELIC_LICENSE_KEY", raising=False)
    assert nr_events.record_event("GtfsEtaDaemonError", {"level": "error"}, service="svc") is False
    assert posts == []


def test_event_shape(monkeypatch, posts):
    monkeypatch.setenv("NEW_RELIC_LICENSE_KEY", "eu01xx-test")
    monkeypatch.delenv("NEW_RELIC_ACCOUNT_ID", raising=False)
    assert nr_events.record_event("GtfsEtaDaemonError", {"level": "error", "message": "x" * 5000},
                                  service="gtfs-eta-daemon", commit="abc1234") is True
    (url, kw), = posts
    assert url == f"{nr_events.EVENT_API_HOST}/v1/accounts/{nr_events.ACCOUNT_ID}/events"
    assert kw["headers"]["Api-Key"] == "eu01xx-test"
    (event,) = kw["json"]
    assert event["eventType"] == "GtfsEtaDaemonError"
    assert event["service"] == "gtfs-eta-daemon" and event["commit"] == "abc1234"
    assert event["level"] == "error"
    assert len(event["message"]) == 4000        # NRDB rejects attributes over 4 KB


def test_network_failure_never_raises(monkeypatch):
    monkeypatch.setenv("NEW_RELIC_LICENSE_KEY", "eu01xx-test")

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(nr_events.requests, "post", boom)
    assert nr_events.record_event("GtfsEtaDaemonError", {}, service="svc") is False


def test_exception_attributes_keep_the_innermost_frames():
    def inner():
        raise ValueError("bad row")

    try:
        inner()
    except ValueError as exc:
        attrs = nr_events.exception_attributes(exc)
    assert attrs["errorClass"] == "ValueError" and attrs["message"] == "bad row"
    assert "inner" in attrs["traceback"] and attrs["traceback"].rstrip().endswith("bad row")
